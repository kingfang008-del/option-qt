import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import concurrent.futures
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# ==============================================================================
# 配置：定义我们要锁定的 6 个“锚点” (智能寻优版)
# ==============================================================================
ANCHOR_CONFIG = {
    # --- Front (近月) ---
    # 实盘逻辑: today + 7~13 days
    'FRONT_TARGET_DTE': 9,        # 理想锚点 (取中间值)
    'FRONT_MIN_DTE': 5,           # 最小允许
    'FRONT_MAX_DTE': 16,          # 最大允许
    
    # --- Next (次月) ---
    # 实盘逻辑: Front + 28 days -> 也就是 35~42 DTE 左右
    'NEXT_TARGET_DTE': 37,        # 理想锚点 (9 + 28)
    'NEXT_MIN_DTE': 25,           # 最小允许 (必须与 Front 拉开差距)
    'NEXT_MAX_DTE': 60,           # 最大允许
    
    # 选合约时的 Delta 参考标准
    'ATM_CENTER': 0.50,
    'OTM_CENTER': 0.25 
}

def safe_convert_to_ny_time(series):
    """时间清洗 (保持不变)"""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors='coerce', utc=True)
    if series.dt.tz is None:
        return series.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    else:
        return series.dt.tz_convert('America/New_York')
 
def get_daily_locked_contracts(df):
    """
    【核心逻辑：日内锁定选合约 - 实盘锚点对齐版】
    参考实盘逻辑: 
    1. Front = Today + ~1 week
    2. Next  = Front + ~4 weeks (28 days)
    确保 Front 和 Next 有足够的时间间隔，从而计算出正确的 Term Structure。
    """
    # 1. 辅助列
    df['date_str'] = df['timestamp'].dt.date.astype(str)
    df['abs_delta'] = df['delta'].abs()
    
    # 2. 预筛选 (宽泛范围)
    mask_dte = (df['dte'] >= 2) & (df['dte'] <= 90)
    candidates = df[mask_dte].copy()
    
    if candidates.empty: return None

    locked_map = [] 
    
    # 按天遍历
    for date_val, daily_group in candidates.groupby('date_str'):
        # 获取该天所有可选的 DTE
        available_dtes = daily_group['dte'].unique()
        if len(available_dtes) == 0: continue
        
        # --- A. 确定当天的 Front DTE 和 Next DTE (核心算法) ---
        
        # 1. 寻找 Front (目标: 9 DTE)
        # 逻辑: 找绝对距离最近的
        front_target = ANCHOR_CONFIG['FRONT_TARGET_DTE']
        front_options = [d for d in available_dtes if ANCHOR_CONFIG['FRONT_MIN_DTE'] <= d <= ANCHOR_CONFIG['FRONT_MAX_DTE']]
        
        if not front_options:
            # 兜底: 如果没有完美的 7-14 天，就找所有 DTE 中最小的那个（但不能是末日轮 < 3）
            valid_mins = [d for d in available_dtes if d >= 3]
            if not valid_mins: continue
            selected_front_dte = min(valid_mins)
        else:
            # 选离 Target 最近的
            selected_front_dte = min(front_options, key=lambda x: abs(x - front_target))
            
        # 2. 寻找 Next (目标: Front + 28)
        # 逻辑: 实盘是 Front Date + 28 days
        next_target = selected_front_dte + 28
        
        # 搜索范围: [Front+20, Front+45] 确保拉开差距
        min_next = selected_front_dte + 20
        max_next = selected_front_dte + 50
        
        next_options = [d for d in available_dtes if min_next <= d <= max_next]
        
        if not next_options:
            # 兜底: 只要比 Front 大 15 天以上就行
            fallbacks = [d for d in available_dtes if d > selected_front_dte + 15]
            if not fallbacks:
                # 实在没有次月合约，为了防止报错，Next = Front (Term Structure = 0)
                selected_next_dte = selected_front_dte
            else:
                # 选最小的那个（离 Front 最近的远期）
                selected_next_dte = min(fallbacks)
        else:
            # 选离 Target 最近的
            selected_next_dte = min(next_options, key=lambda x: abs(x - next_target))

        # ---------------------------------------------------------
        
        # 预计算 Volume Ranks
        volume_ranks = daily_group.groupby('contract_symbol')['volume'].sum()
        
        # 定义 6 个目标桶
        targets = [
            (0, True, False, 0.50), # Front Put ATM
            (1, True, False, 0.25), # Front Put OTM
            (2, True,  True, 0.50), # Front Call ATM
            (3, True,  True, 0.25), # Front Call OTM
            (4, False, False, 0.50), # Next Put ATM
            (5, False,  True, 0.50)  # Next Call ATM
        ]
        
        for b_id, is_front, is_call, target_delta in targets:
            # 1. 确定 DTE
            target_dte = selected_front_dte if is_front else selected_next_dte
            
            # 2. 筛选
            type_str = 'Call' if is_call else 'Put'
            
            # 组合掩码 (DTE + Type)
            # 注意: 这里使用严格的 DTE 匹配，因为我们已经在上面选好了具体的 DTE
            mask = (daily_group['dte'] == target_dte) & \
                   (daily_group['contract_type'].astype(str).str.upper().apply(lambda x: x.startswith(type_str[0].upper())))
            
            subset = daily_group[mask]
            if subset.empty: continue
            
            # 3. Delta 筛选 (寻找最接近 Target Delta 的)
            # 复用实盘逻辑: 找 Spot +/- 20% (对应 Delta 也就是找 Target +/- 0.15 左右)
            subset = subset.copy()
            subset['delta_dist'] = (subset['abs_delta'] - target_delta).abs()
            
            # 优先选 Delta 准的 (< 0.15 偏差)
            delta_candidates = subset[subset['delta_dist'] < 0.15]
            
            if delta_candidates.empty:
                # 如果没有满足条件的，直接选 Delta 偏差最小的那一个
                best_ticker = subset.sort_values('delta_dist').iloc[0]['contract_symbol']
            else:
                # 都有满足条件的，选最接近目标 Delta 的
                best_ticker = delta_candidates.sort_values('delta_dist').iloc[0]['contract_symbol']
            
            locked_map.append({
                'date_str': date_val,
                'contract_symbol': best_ticker,
                'bucket_id': b_id
            })
    
    return pd.DataFrame(locked_map)


def calculate_locked_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    【v9 Final - 锁定合约特征工程】
    
    修复列表:
    1. [Term Structure]: 修复负数问题 (前向填充 + 有效性检查).
    2. [Vanna/Charm]: 修复训练偏差 (离线强制对齐实盘的近似计算逻辑).
    3. [Weighting]: 修复零量黑洞 (当成交量为0时，自动退化为算术平均).
    """
    if df.empty: return pd.DataFrame()

    # --- 1. 清洗与基础计算 ---
    try:
        df['timestamp'] = safe_convert_to_ny_time(df['timestamp']).dt.round('1min')
        df['expiration'] = safe_convert_to_ny_time(df['expiration'])
    except: return pd.DataFrame()

    # DTE 计算
    df['dte'] = (df['expiration'].dt.normalize() - df['timestamp'].dt.normalize()).dt.days.fillna(-1).astype(int)    
    # 补全基础列
    cols = ['volume', 'iv', 'delta', 'gamma', 'vega', 'theta', 'stock_close', 'close']
    for c in cols: 
        if c not in df.columns: df[c] = 0.0
    
    # [核心修复 2] 离线强制计算 Vanna/Charm，对齐实盘逻辑
    # Realtime Engine 逻辑: vanna = vega / spot, charm = theta / spot
    # 防止除零
    epsilon = 1e-9
    df['vanna'] = df['vega'] / (df['stock_close'] + epsilon)
    df['charm'] = df['theta'] / (df['stock_close'] + epsilon)
    
    df['contract_type'] = df['contract_type'].astype(str).str.strip().str.title() 

    # --- 2. 获取锁定名单 ---
    lock_schedule = get_daily_locked_contracts(df)
    if lock_schedule is None or lock_schedule.empty: return pd.DataFrame()

    # --- 3. 关联数据 ---
    df['date_str'] = df['timestamp'].dt.date.astype(str)
    merged = pd.merge(df, lock_schedule, on=['date_str', 'contract_symbol'], how='inner')
    if merged.empty: return pd.DataFrame()

    # --- 4. 聚合 ---
    agg_dict = {
        'volume': 'sum', 
        'iv': 'mean', 'delta': 'mean', 'gamma': 'mean', 
        'vega': 'mean', 'theta': 'mean', 'vanna': 'mean', 'charm': 'mean',
        'stock_close': 'last'
    }
    grouped = merged.groupby(['timestamp', 'bucket_id']).agg(agg_dict).reset_index()

    # --- 5. 展平 (Pivot) ---
    pivot_cols = ['volume', 'iv', 'delta', 'gamma', 'vega', 'theta', 'vanna', 'charm']
    df_wide = grouped.pivot(index='timestamp', columns='bucket_id', values=pivot_cols)
    df_wide.columns = [f"{c[0]}_{c[1]}" for c in df_wide.columns]
    
    # 重新索引时间轴 & 智能填充
    full_idx = pd.date_range(start=df_wide.index.min(), end=df_wide.index.max(), freq='1min')
    df_wide = df_wide.reindex(full_idx)
    df_wide.index.name = 'timestamp'
    
    # Greeks/IV/Price: 前向填充 (FFill)
    greek_cols = [c for c in df_wide.columns if 'volume' not in c]
    # 修复后
    df_wide[greek_cols] = df_wide[greek_cols].replace(0.0, np.nan).ffill(limit=30).fillna(0.0)
    # Volume: 缺失去补 0
    vol_cols = [c for c in df_wide.columns if 'volume' in c]
    df_wide[vol_cols] = df_wide[vol_cols].fillna(0.0)
    
    # Stock Price: FFill
    stock_prices = merged.groupby('timestamp')['stock_close'].last()
    stock_prices = stock_prices.reindex(full_idx).ffill()
    df_wide['stock_close'] = stock_prices
    
    df_wide = df_wide.fillna(0.0)

    # --- 6. 特征计算 (对齐实盘) ---
    v = {i: df_wide.get(f'volume_{i}', 0) for i in range(6)}
    iv = {i: df_wide.get(f'iv_{i}', 0) for i in range(6)}
    
    # [核心修复 3] 零量加权保护
    # 计算 Front (Bucket 0-3) 的总成交量
    total_vol_front = v[0] + v[1] + v[2] + v[3]
    
    # 向量化判断: 哪一行的总成交量 < 1 (即没有成交)
    mask_no_vol = (total_vol_front < 1.0)
    
    def calc_vw(metric):
        m0 = df_wide.get(f'{metric}_0', 0)
        m1 = df_wide.get(f'{metric}_1', 0)
        m2 = df_wide.get(f'{metric}_2', 0)
        m3 = df_wide.get(f'{metric}_3', 0)
        
        # 正常情况: 加权平均
        weighted_sum = m0*v[0] + m1*v[1] + m2*v[2] + m3*v[3]
        res = weighted_sum / (total_vol_front + epsilon)
        
        # 异常情况 (无量): 算术平均
        # 仅对非零值取平均 (避免把 fillna(0) 的坑算进去)
        # 这里简化处理：直接取 4 个值的均值 (因为前面已经 ffill 了，大概率非零)
        simple_avg = (m0 + m1 + m2 + m3) / 4.0
        
        # 组合: 有量用加权，无量用算术
        return np.where(mask_no_vol, simple_avg, res)

    # 计算加权特征
    df_wide['options_vw_iv']    = calc_vw('iv')
    df_wide['options_vw_delta'] = calc_vw('delta')
    df_wide['options_vw_gamma'] = calc_vw('gamma')
    df_wide['options_vw_vega']  = calc_vw('vega')
    df_wide['options_vw_theta'] = calc_vw('theta')
    df_wide['options_vw_vanna'] = calc_vw('vanna')
    df_wide['options_vw_charm'] = calc_vw('charm')

    # B. 情绪特征
    # PCR: 也要防止分母为0
    denom = v[2] + v[3]
    df_wide['options_pcr_volume'] = np.where(denom > 0, (v[0] + v[1]) / denom, 0.7) # 0.7 是常见中值
    
    # Flow Skew
    avg_iv_put = (iv[0] + iv[1]) / 2.0
    avg_iv_call = (iv[2] + iv[3]) / 2.0
    mask_valid_skew = (avg_iv_call > 0.01)
    df_wide['options_flow_skew'] = np.where(mask_valid_skew, avg_iv_put / (avg_iv_call + epsilon), 1.0)

    # C. 结构特征 (Term Structure)
    front_atm = (iv[0] + iv[2]) / 2.0
    next_atm = (iv[4] + iv[5]) / 2.0
    
    df_wide['options_struc_atm_iv'] = front_atm
    
    # Struc Skew
    mask_valid_struc_skew = (iv[3] > 0.01)
    df_wide['options_struc_skew'] = np.where(mask_valid_struc_skew, iv[1] / (iv[3] + epsilon), 1.0)
    
    # Term Structure (双重有效性检查)
    mask_valid_term = (next_atm > 0.01) & (front_atm > 0.01)
    df_wide['options_struc_term'] = np.where(mask_valid_term, next_atm - front_atm, 0.0)

    # D. 时序特征
    df_wide['options_iv_momentum'] = df_wide['options_vw_iv'].pct_change(periods=5).fillna(0.0)
    df_wide['options_gamma_accel'] = df_wide['options_vw_gamma'].pct_change(periods=5).fillna(0.0)
    price_mom = df_wide['stock_close'].pct_change(periods=5).fillna(0.0)
    df_wide['options_iv_divergence'] = df_wide['options_iv_momentum'] - price_mom

    final_cols = [
        'stock_close',
        'options_vw_iv', 'options_vw_delta', 'options_vw_gamma', 'options_vw_vega', 'options_vw_theta',
        'options_vw_vanna', 'options_vw_charm',
        'options_pcr_volume', 'options_flow_skew',
        'options_iv_momentum', 'options_gamma_accel', 'options_iv_divergence',
        'options_struc_atm_iv', 'options_struc_skew', 'options_struc_term'
    ]
    
    return df_wide[final_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index()


# ==============================================================================
# Worker & Main (复用之前的架构)
# ==============================================================================
def process_single_file(args):
    file_path, output_dir, symbol = args
    try:
        raw_df = pd.read_parquet(file_path)
        # 映射列名
        rename_map = {'expiration_date': 'expiration', 'strike_price': 'strike', 'ticker': 'contract_symbol'}
        raw_df = raw_df.rename(columns={k:v for k,v in rename_map.items() if k in raw_df.columns})
        
        feat_df = calculate_locked_features(raw_df)
        
        if not feat_df.empty:
            out_file = output_dir / symbol / file_path.name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            feat_df.to_parquet(out_file, compression='zstd')
            return None
        return f"Empty result for {symbol}"
    except Exception as e:
        return f"Error {symbol}: {str(e)}\n{traceback.format_exc()}"

def main():
    RAW_DIR = Path.home() / "train_data/nq_options_monthly_iv/"
    OUTPUT_DIR = Path.home() / "train_data/nq_options_bucketed_v7/" # 覆盖之前的目录或新建
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from config import TARGET_SYMBOLS

    tasks = []
    print("Scanning files...")
    for sym in tqdm(TARGET_SYMBOLS):
        src_dir = RAW_DIR / sym / "standard"
        if not src_dir.exists(): continue
        for p in src_dir.glob("*.parquet"):
            tasks.append((p, OUTPUT_DIR, sym))

    print(f"Processing {len(tasks)} files with Locked Contract Logic...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_file, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            res = f.result()
            if res: logging.warning(res)

    print("✅ Done. Features now reflect 'Locked Contract' reality.")

if __name__ == '__main__':
    main()