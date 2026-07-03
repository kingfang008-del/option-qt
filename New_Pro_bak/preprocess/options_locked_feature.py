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

def safe_convert_to_ny_time(series):
    """时间清洗"""
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors='coerce', utc=True)
    if series.dt.tz is None:
        return series.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    else:
        return series.dt.tz_convert('America/New_York')

def calculate_locked_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    【V10 Sniper 架构专用版】
    前提：输入数据必须已经包含 bucket_id, spread_pct, volume_imbalance。
    由于合约已被精准锁定，直接进行 Pivot 展平和微观特征加权。
    """
    if df.empty or 'bucket_id' not in df.columns: 
        return pd.DataFrame()

    epsilon = 1e-9
    
    # --- 1. 时间清洗与 Vanna/Charm 计算 ---
    try:
        # 严格遵守时间物理法则：09:30:01 ~ 09:31:00 的 tick 统一打上 09:31:00 的标签
        df['timestamp'] = safe_convert_to_ny_time(df['timestamp']).dt.ceil('1min')
    except: 
        return pd.DataFrame()

    df['vanna'] = df['vega'] / (df['stock_close'] + epsilon)
    df['charm'] = df['theta'] / (df['stock_close'] + epsilon)

    # --- 2. 聚合 (去重) 与直接展平 (Pivot) ---
    # 如果同一分钟由于异常导致有重复数据，取最后一条
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp', 'bucket_id'], keep='last')

    # 将我们需要的所有微观和宏观列全部展平
    pivot_cols = [
        'volume', 'iv', 'delta', 'gamma', 'vega', 'theta', 'vanna', 'charm',
        'spread_pct', 'volume_imbalance'  # 👈 新引入的顶级微观特征
    ]
    
    # 核心降维打击：直接用自带的 bucket_id 展平！
    df_wide = df.pivot(index='timestamp', columns='bucket_id', values=pivot_cols)
    df_wide.columns = [f"{c[0]}_{int(c[1])}" for c in df_wide.columns]
    
    # --- 3. 重新索引时间轴 & 智能填充 ---
    full_idx = pd.date_range(start=df_wide.index.min(), end=df_wide.index.max(), freq='1min')
    df_wide = df_wide.reindex(full_idx)
    df_wide.index.name = 'timestamp'
    
    # 特征填充策略：
    # 盘口状态 (Greeks, Spread, Imbalance): 保持连续性，前向填充 30 分钟，超出填 0
    state_cols = [c for c in df_wide.columns if 'volume' not in c or 'volume_imbalance' in c]
    df_wide[state_cols] = df_wide[state_cols].replace(0.0, np.nan).ffill(limit=30).fillna(0.0)
    
    # 绝对数量 (Volume 即 Depth): 缺失代表当分钟无挂单，严格补 0
    vol_cols = [c for c in df_wide.columns if c.startswith('volume_') and 'imbalance' not in c]
    df_wide[vol_cols] = df_wide[vol_cols].fillna(0.0)
    
    # 还原 Stock Close (前向填充)
    stock_prices = df.groupby('timestamp')['stock_close'].last().reindex(full_idx).ffill()
    df_wide['stock_close'] = stock_prices
    
    df_wide = df_wide.fillna(0.0)

    # --- 4. 华尔街级微观结构特征计算 (严谨物理隔离版) ---
    # 获取底层数据字典，0: PUT_ATM, 1: PUT_OTM, 2: CALL_ATM, 3: CALL_OTM
    v = {i: df_wide.get(f'volume_{i}', 0) for i in range(6)}
    iv = {i: df_wide.get(f'iv_{i}', 0) for i in range(6)}
    vega = {i: df_wide.get(f'vega_{i}', 0) for i in range(6)}
    gamma = {i: df_wide.get(f'gamma_{i}', 0) for i in range(6)}
    delta = {i: df_wide.get(f'delta_{i}', 0) for i in range(6)}
    theta = {i: df_wide.get(f'theta_{i}', 0) for i in range(6)}  # 👈 [新增] 补上 Theta 字典
    
    total_vol_front = v[0] + v[1] + v[2] + v[3]
    mask_no_vol = (total_vol_front < 1.0)
    
    # ---------------------------------------------------------
    # A. 核心基准特征 (Baseline Metrics)
    # ---------------------------------------------------------
    # 1. 真实基准 IV (Baseline IV): 绝对不混合 Call 和 Put！
    # 业界标准：只取平值期权 (ATM) 的均值，因为 ATM 流动性最好，Skew 扭曲最小。
    df_wide['options_vw_iv'] = (iv[0] + iv[2]) / 2.0 
    
    # 2. 净 Delta 敞口 (Net Delta Exposure) - 按成交量加权是合理的！
    # 因为 Put 的 Delta 是负数，Call 是正数，按成交量加总正好代表了市场资金的“净做多/做空方向”
    net_delta_vol = delta[0]*v[0] + delta[1]*v[1] + delta[2]*v[2] + delta[3]*v[3]
    df_wide['options_vw_delta'] = np.where(mask_no_vol, 0.0, net_delta_vol / (total_vol_front + epsilon))

    # 3. 净 Gamma/Vega 敞口 (Market Maker Exposure)
    # 既然是测算做市商敞口，用简单的算术平均毫无意义。有量才有敞口。
    net_gamma = gamma[0]*v[0] + gamma[1]*v[1] + gamma[2]*v[2] + gamma[3]*v[3]
    net_vega = vega[0]*v[0] + vega[1]*v[1] + vega[2]*v[2] + vega[3]*v[3]
    net_theta = theta[0]*v[0] + theta[1]*v[1] + theta[2]*v[2] + theta[3]*v[3] # 👈 [新增] 净 Theta 敞口
    
    df_wide['options_vw_gamma'] = np.where(mask_no_vol, (gamma[0]+gamma[2])/2.0, net_gamma / (total_vol_front + epsilon))
    df_wide['options_vw_vega']  = np.where(mask_no_vol, (vega[0]+vega[2])/2.0, net_vega / (total_vol_front + epsilon))
    df_wide['options_vw_theta'] = np.where(mask_no_vol, (theta[0]+theta[2])/2.0, net_theta / (total_vol_front + epsilon)) # 👈 [新增] 净 Theta 敞口
    
    # Vanna 和 Charm 取 ATM 的平均 (因为 OTM 经常失真)
    df_wide['options_vw_vanna'] = (df_wide.get('vanna_0',0) + df_wide.get('vanna_2',0)) / 2.0
    df_wide['options_vw_charm'] = (df_wide.get('charm_0',0) + df_wide.get('charm_2',0)) / 2.0

    # ---------------------------------------------------------
    # B. 流动性与微观失衡特征 (Microstructure)
    # ---------------------------------------------------------
    def calc_vw_pure(metric):
        """纯量加权，断流时给 0，绝不用算术平均糊弄"""
        m0, m1, m2, m3 = [df_wide.get(f'{metric}_{i}', 0) for i in range(4)]
        weighted_sum = m0*v[0] + m1*v[1] + m2*v[2] + m3*v[3]
        return np.where(mask_no_vol, 0.0, weighted_sum / (total_vol_front + epsilon))

    # 做市商恐慌度: Spread 突然拉大是暴跌前兆
    df_wide['options_vw_spread'] = calc_vw_pure('spread_pct')
    
    # 订单簿失衡: 正数买盘多，负数卖盘多
    df_wide['options_vw_imbalance'] = calc_vw_pure('volume_imbalance')

    # ---------------------------------------------------------
    # C. 偏斜与期限结构 (Skew & Term Structure)
    # ---------------------------------------------------------
    # 真实资金倾向 (Put/Call Ratio)
    denom_call_vol = v[2] + v[3]
    df_wide['options_pcr_volume'] = np.where(denom_call_vol > 0, (v[0] + v[1]) / denom_call_vol, 1.0) 
    
    # 偏斜 (Volatility Skew): 严格遵守物理法则！
    # 1. Flow Skew: 虚值 Put 溢价 vs 虚值 Call 溢价 (衡量暴跌恐慌)
    df_wide['options_flow_skew'] = np.where(iv[3] > 0.01, iv[1] / (iv[3] + epsilon), 1.0)
    
    # 2. Struc Skew: 虚值 Put 溢价 vs 平值 Put 溢价 (衡量护盘成本)
    df_wide['options_struc_skew'] = np.where(iv[0] > 0.01, iv[1] / (iv[0] + epsilon), 1.0)

    # 期限结构: 次月 ATM vs 近月 ATM (Contango / Backwardation)
    front_atm = df_wide['options_vw_iv']
    next_atm = (iv[4] + iv[5]) / 2.0
    df_wide['options_struc_atm_iv'] = front_atm
    df_wide['options_struc_term'] = np.where((next_atm > 0.01) & (front_atm > 0.01), next_atm - front_atm, 0.0)

    # ---------------------------------------------------------
    # D. 时序导数特征
    # ---------------------------------------------------------
    df_wide['options_iv_momentum'] = df_wide['options_vw_iv'].pct_change(periods=5).fillna(0.0)
    df_wide['options_gamma_accel'] = df_wide['options_vw_gamma'].pct_change(periods=5).fillna(0.0)
    price_mom = df_wide['stock_close'].pct_change(periods=5).fillna(0.0)
    df_wide['options_iv_divergence'] = df_wide['options_iv_momentum'] - price_mom

    final_cols = [
        'stock_close',
        'options_vw_iv', 'options_vw_delta', 'options_vw_gamma', 'options_vw_vega', 'options_vw_theta',
        'options_vw_vanna', 'options_vw_charm',
        'options_vw_spread', 'options_vw_imbalance', 
        'options_pcr_volume', 'options_flow_skew',
        'options_iv_momentum', 'options_gamma_accel', 'options_iv_divergence',
        'options_struc_atm_iv', 'options_struc_skew', 'options_struc_term'
    ]
    
    return df_wide[final_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index()


def process_single_file(args):
    file_path, output_dir, symbol = args
    try:
        raw_df = pd.read_parquet(file_path)
        # 兼容旧表头习惯 (Polygon 的 ticker 带着 O: 前缀，这里统一去掉或映射)
        rename_map = {'ticker': 'contract_symbol'}
        raw_df = raw_df.rename(columns={k:v for k,v in rename_map.items() if k in raw_df.columns})
        
        # 确保去掉 O: 前缀，保持干净
        if 'contract_symbol' in raw_df.columns:
            raw_df['contract_symbol'] = raw_df['contract_symbol'].str.replace('O:', '', regex=False)
            
        feat_df = calculate_locked_features(raw_df)
        
        if not feat_df.empty:
            out_file = output_dir / symbol / file_path.name
            out_file.parent.mkdir(parents=True, exist_ok=True)
            feat_df.to_parquet(out_file, compression='zstd')
            return None
        return f"Empty result for {symbol} (No valid bucket data)"
    except Exception as e:
        return f"Error {symbol}: {str(e)}\n{traceback.format_exc()}"

def main():
    # 注意：RAW_DIR 现在指向你通过 option_cac_day_vectorized.py 算好希腊字母输出的文件夹！
    RAW_DIR = Path.home() / "train_data/quote_options_monthly_iv/"
    OUTPUT_DIR = Path.home() / "train_data/quote_options_bucketed_v7/" 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from config import TARGET_SYMBOLS

    tasks = []
    print("🚀 启动极速微观特征聚合管线...")
    for sym in tqdm(TARGET_SYMBOLS):
        src_dir = RAW_DIR / sym / "standard"
        if not src_dir.exists(): continue
        for p in src_dir.glob("*.parquet"):
            tasks.append((p, OUTPUT_DIR, sym))

    print(f"📦 共计 {len(tasks)} 个日切片，无需寻优，直接降维展平！")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_single_file, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            res = f.result()
            if res: logging.warning(res)

    print("✅ 终极特征生成完毕！微观 Alpha (Spread & Imbalance) 已注入模型血液！")

if __name__ == '__main__':
    main()