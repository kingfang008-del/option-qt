import os
import pandas as pd
import numpy as np
import concurrent.futures
import logging
import warnings
import gc
import glob
from tqdm import tqdm
from scipy.stats import norm

try:
    from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
except ImportError:
    raise ImportError("请先安装: pip install py_vollib_vectorized")

warnings.filterwarnings('ignore')

# 输入：你通过任何方式下载的原始 1s 期权和 1s 正股数据
RAW_OPT_DIR = "/mnt/s990/data/raw_1s/options"
RAW_STOCK_DIR = "/mnt/s990/data/raw_1s/stocks"

# 输出：带全套 Greeks 的高频压测数据
OUTPUT_DIR = "/mnt/s990/data/stress_test_1s_greeks"
RFR_CACHE_FILE = "/home/kingfang007/risk_free_rates.parquet"

MAX_WORKERS = 16

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= 全局变量与复权因子 =================
G_RFR_SERIES = None

def init_worker(rfr_series):
    global G_RFR_SERIES
    G_RFR_SERIES = rfr_series

KNOWN_SPLITS = {
    'NVDA': [('2021-07-20', 0.25), ('2024-06-10', 0.10)],
    'AMZN': [('2022-06-06', 0.05)],
    'GOOG': [('2022-07-18', 0.05)],
    'GOOGL':[('2022-07-18', 0.05)],
    'TSLA': [('2022-08-25', 0.33333333)],
    'AVGO': [('2024-07-15', 0.10)], 
    'SMCI': [('2024-10-01', 0.10)],
    'LRCX': [('2024-10-03', 0.10)],
    'CMG':  [('2024-06-26', 0.02)],
    'ANET': [('2024-11-18', 0.25)],
    'TTD':  [('2021-06-17', 0.10)],
    'PANW': [('2022-09-14', 0.33333333)],
    'FTNT': [('2022-06-23', 0.20)],
    'DXCM': [('2022-06-10', 0.25)],
    'ISRG': [('2021-10-05', 0.33333333)],
    'CSGP': [('2021-06-28', 0.25)],
    'MNST': [('2023-03-28', 0.50)],
    'CELH': [('2023-11-15', 0.33333333)],
    'ODFL': [('2024-03-28', 0.50)],
    'ROP':  [('2023-09-25', 0.50)],
    'SHW':  [('2021-04-01', 0.33333333)],
    'CTAS': [('2024-09-12', 0.25)],
}

# ================= 纯函数：高阶 Greeks 计算 =================
def get_d1_d2(S, K, t, r, sigma):
    valid_mask = (t > 1e-9) & (sigma > 1e-9)
    d1 = np.full_like(S, np.nan)
    d2 = np.full_like(S, np.nan)
    S_valid, K_valid, t_valid, r_valid, sigma_valid = S[valid_mask], K[valid_mask], t[valid_mask], r[valid_mask], sigma[valid_mask]
    sigma_sqrt_t = sigma_valid * np.sqrt(t_valid)
    d1_valid = (np.log(S_valid / K_valid) + (r_valid + 0.5 * sigma_valid**2) * t_valid) / sigma_sqrt_t
    d2_valid = d1_valid - sigma_sqrt_t
    d1[valid_mask] = d1_valid
    d2[valid_mask] = d2_valid
    return d1, d2

def vectorized_vanna(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    vanna = np.zeros_like(sigma)
    mask = sigma > 1e-9
    vanna[mask] = -norm.pdf(d1)[mask] * (d2_val[mask] / sigma[mask])
    return vanna

def vectorized_charm(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    sigma_sqrt_t = np.zeros_like(sigma)
    sigma_sqrt_t[sigma > 1e-9] = sigma[sigma > 1e-9] * np.sqrt(t[sigma > 1e-9])
    
    term1, term2 = np.zeros_like(sigma_sqrt_t), np.zeros_like(t)
    mask1 = sigma_sqrt_t > 1e-9
    term1[mask1] = r[mask1] / sigma_sqrt_t[mask1]
    mask2 = t > 1e-9
    term2[mask2] = d2_val[mask2] / (2 * t[mask2])
    
    charm = np.zeros_like(sigma)
    mask = sigma > 1e-9
    charm[mask] = -norm.pdf(d1)[mask] * (term1[mask] - term2[mask])
    return charm

def vectorized_rho(S, K, t, r, sigma, is_call):
    _, d2_val = get_d1_d2(S, K, t, r, sigma)
    rho = np.zeros_like(sigma)
    mask = sigma > 1e-9
    
    discount_factor = np.zeros_like(sigma)
    discount_factor[mask] = K[mask] * t[mask] * np.exp(-r[mask] * t[mask])
    
    call_mask = mask & is_call
    if call_mask.any(): rho[call_mask] = discount_factor[call_mask] * norm.cdf(d2_val[call_mask]) / 100.0
    put_mask = mask & (~is_call)
    if put_mask.any(): rho[put_mask] = -discount_factor[put_mask] * norm.cdf(-d2_val[put_mask]) / 100.0
    return rho

# ================= 核心处理逻辑 =================
def process_file(file_path):
    try:
        parts = file_path.split(os.sep)
        symbol, filename = parts[-2], parts[-1]
        date_str = filename.replace(f"{symbol}_", "").replace(".parquet", "")
        
        out_dir = os.path.join(OUTPUT_DIR, symbol)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        if os.path.exists(out_path): return f"⏩ {symbol} {date_str} exists"
            
        # 1. 加载期权 1s 数据，显式丢弃所有旧的 Greeks 字段，确保从底导 (P, S, K, T, r) 强制重算
        df_opt = pd.read_parquet(file_path)
        if df_opt.empty: return f"⚠️ {symbol} {date_str} opt is empty"
        
        cols_to_drop = ['iv', 'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'charm', 'vomma']
        existing_drops = [c for c in cols_to_drop if c in df_opt.columns]
        if existing_drops:
            df_opt = df_opt.drop(columns=existing_drops)
        
        time_col = 'timestamp' if 'timestamp' in df_opt.columns else 'ts'
        if pd.api.types.is_numeric_dtype(df_opt[time_col]):
            df_opt['timestamp'] = pd.to_datetime(df_opt[time_col], unit='s', utc=True).dt.tz_convert('America/New_York')
        else:
            df_opt['timestamp'] = pd.to_datetime(df_opt[time_col], utc=True).dt.tz_convert('America/New_York')
            
        # 过滤无效报价并计算中间价 (消灭 bid-ask bounce 噪音)
        df_opt = df_opt[(df_opt['bid'] > 0) & (df_opt['ask'] >= df_opt['bid'])].copy()
        if df_opt.empty: return f"⚠️ {symbol} {date_str} 无有效盘口"
        df_opt['mid_price'] = (df_opt['bid'] + df_opt['ask']) / 2.0
        
        # 提取基础信息
        df_opt['ticker'] = df_opt['ticker'].str.replace('O:', '', regex=False)
        extracted = df_opt['ticker'].str.extract(r'^[A-Z]+(\d{6})([CP])(\d{8})$')
        df_opt['expiration_date'] = pd.to_datetime('20' + extracted[0], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
        df_opt['contract_type'] = extracted[1].str.lower()
        df_opt['strike_price'] = extracted[2].astype(float) / 1000.0

        # 2. 加载正股 1s 数据 (假定已复权)
        stock_path = os.path.join(RAW_STOCK_DIR, symbol, f"{symbol}_{date_str}.parquet")
        if not os.path.exists(stock_path): return f"⚠️ {symbol} {date_str} 无正股数据"
        df_stk = pd.read_parquet(stock_path)
        
        stk_time_col = 'timestamp' if 'timestamp' in df_stk.columns else 'ts'
        if pd.api.types.is_numeric_dtype(df_stk[stk_time_col]):
            df_stk['timestamp'] = pd.to_datetime(df_stk[stk_time_col], unit='s', utc=True).dt.tz_convert('America/New_York')
        else:
            df_stk['timestamp'] = pd.to_datetime(df_stk[stk_time_col], utc=True).dt.tz_convert('America/New_York')
            
        target_col = 'close' if 'close' in df_stk.columns else 'price'
        df_stk = df_stk[['timestamp', target_col]].rename(columns={target_col: 'stock_close'})
        
        # 排序准备 asof 匹配
        df_opt = df_opt.sort_values('timestamp')
        df_stk = df_stk.sort_values('timestamp')

        # [核心对齐]：使用 merge_asof 允许最大 2 秒回溯，极速匹配最新正股切片
        df_merged = pd.merge_asof(
            df_opt, df_stk,
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('2s') # 秒级允许微小错位
        )
        df_merged.dropna(subset=['stock_close'], inplace=True)
        if df_merged.empty: return f"⚠️ {symbol} {date_str} Asof 匹配后无数据"

        # 3. 仅对期权执行复权 (同步正股水平)
        if symbol in KNOWN_SPLITS:
            restore_factor = np.ones(len(df_merged))
            splits = sorted(KNOWN_SPLITS[symbol], key=lambda x: x[0])
            for split_str, ratio in splits:
                split_date = pd.Timestamp(split_str).tz_localize('America/New_York')
                mask = df_merged['timestamp'] < split_date
                restore_factor[mask] /= ratio
            
            for col in ['strike_price', 'mid_price', 'bid', 'ask', 'price']:
                if col in df_merged.columns: df_merged[col] /= restore_factor
            for col in ['bid_size', 'ask_size', 'volume']:
                if col in df_merged.columns: df_merged[col] *= restore_factor

        # 4. BS 向量化求解 IV 与 Greeks
        S = df_merged['stock_close'].values
        K = df_merged['strike_price'].values
        P = df_merged['mid_price'].values
        
        current_ts = df_merged['timestamp']
        # 强制到期时间为 16:00:00 (对齐 1m)
        expiry_ts = pd.to_datetime(df_merged['expiration_date']).dt.tz_localize('America/New_York') + pd.Timedelta(hours=16)
        
        # [核心修复]：计算 T 时，将 current_ts 截断到秒，避免微秒级别的漂移导致 IV 不同
        time_diff = expiry_ts - current_ts
        T_years = time_diff.dt.total_seconds().values / 31557600.0
        T_years = np.maximum(T_years, 1e-6)

        # [核心修复]：动态加载 r，严禁硬编码 0.04
        r = np.full_like(S, np.nan)
        if G_RFR_SERIES is not None and len(G_RFR_SERIES) > 0:
            # 匹配当前交易日的利率
            search_keys = current_ts.dt.normalize().dt.tz_localize(None)
            r_idx = G_RFR_SERIES.index.searchsorted(search_keys)
            r_idx = np.clip(r_idx, 0, len(G_RFR_SERIES) - 1)
            r = G_RFR_SERIES.values[r_idx]

        # [新增验证逻辑]：如果 r 缺失或意外等于 0.04（老版本残留），直接熔断报错
        if np.any(np.isnan(r)) or np.any(np.isclose(r, 0.04, atol=1e-7)):
            bad_mask = np.isnan(r) | np.isclose(r, 0.04, atol=1e-7)
            bad_idx = np.where(bad_mask)[0][0]
            bad_ts = current_ts.iloc[bad_idx]
            raise RuntimeError(f"🚨 [对齐熔断] 检测到非法利率值(r={r[bad_idx]}) 在时间点 {bad_ts}。这会导致 1s 与 1m 信号漂移，请检查 RFR 原始数据。")

        is_call = (df_merged['contract_type'] == 'c').values
        is_put = (df_merged['contract_type'] == 'p').values

        # 算 IV
        iv = np.zeros_like(P, dtype=float)
        valid = (P > 0) & (S > 0) & (K > 0)
        
        # [DEBUG PIN] 捕捉 NVDA 在 10:00:00 的原始计算因子
        debug_mask = (df_merged['timestamp'] == pd.Timestamp('2026-01-02 10:00:00').tz_localize('America/New_York')) & (df_merged['ticker'].str.contains('NVDA'))
        if debug_mask.any():
            idx_dbg = np.where(debug_mask)[0][0]
            with open("debug_iv_factors.log", "a") as f:
                f.write(f"TS: {df_merged['timestamp'].iloc[idx_dbg]}, Ticker: {df_merged['ticker'].iloc[idx_dbg]}, P: {P[idx_dbg]}, S: {S[idx_dbg]}, K: {K[idx_dbg]}, T: {T_years[idx_dbg]}, r: {r[idx_dbg]}, IV_CALC: {iv[idx_dbg]}\n")
        
        if (is_call & valid).any():
            m = is_call & valid
            iv[m] = vectorized_implied_volatility(P[m], S[m], K[m], T_years[m], r[m], 'c', return_as='numpy', on_error='ignore')
        if (is_put & valid).any():
            m = is_put & valid
            iv[m] = vectorized_implied_volatility(P[m], S[m], K[m], T_years[m], r[m], 'p', return_as='numpy', on_error='ignore')
        
        iv = np.nan_to_num(iv, nan=0.0)
        df_merged['iv'] = iv

        # 算 Greeks
        delta, gamma, vega, theta = np.zeros_like(iv), np.zeros_like(iv), np.zeros_like(iv), np.zeros_like(iv)
        valid_iv = (iv > 0.01) & (iv < 5.0) & valid
        
        if is_call.any() and valid_iv[is_call].any():
            vc = is_call & valid_iv
            gd = get_all_greeks('c', S[vc], K[vc], T_years[vc], r[vc], iv[vc], return_as='dict')
            delta[vc], gamma[vc], vega[vc], theta[vc] = gd['delta'], gd['gamma'], gd['vega'], gd['theta']
            
        if is_put.any() and valid_iv[is_put].any():
            vp = is_put & valid_iv
            gd = get_all_greeks('p', S[vp], K[vp], T_years[vp], r[vp], iv[vp], return_as='dict')
            delta[vp], gamma[vp], vega[vp], theta[vp] = gd['delta'], gd['gamma'], gd['vega'], gd['theta']

        df_merged['delta'] = delta
        df_merged['gamma'] = gamma
        df_merged['vega'] = vega
        df_merged['theta'] = theta
        
        df_merged['vanna'] = vectorized_vanna(S, K, T_years, r, iv)
        df_merged['charm'] = vectorized_charm(S, K, T_years, r, iv)
        df_merged['rho'] = vectorized_rho(S, K, T_years, r, iv, is_call)

        # 5. 输出整理
        df_merged['volume'] = df_merged['bid_size'] + df_merged['ask_size']
        df_merged['spread_pct'] = (df_merged['ask'] - df_merged['bid']) / df_merged['mid_price']
        df_merged['volume_imbalance'] = (df_merged['bid_size'] - df_merged['ask_size']) / (df_merged['volume'] + 1e-6)

        final_cols = [
            'timestamp', 'ticker', 'bucket_id', 'contract_type', 'strike_price', 'expiration_date',
            'stock_close', 'mid_price', 'bid', 'ask', 'bid_size', 'ask_size', 'volume',
            'spread_pct', 'volume_imbalance', 'iv', 'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'charm'
        ]
        final_df = df_merged[[c for c in final_cols if c in df_merged.columns]]
        
        final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
        return f"✅ 压测数据生成 {symbol} {date_str}: {len(final_df)} bars"

    except Exception as e:
        return f"❌ 错误 {file_path}: {e}"

def main():
    if not os.path.exists(RAW_OPT_DIR):
        print(f"❌ 找不到期权目录: {RAW_OPT_DIR}")
        return
        
    all_files = []
    for root, dirs, files in os.walk(RAW_OPT_DIR):
        for f in files:
            if f.endswith(".parquet"): all_files.append(os.path.join(root, f))
                
    if not all_files:
        print("⚠️ 没有找到需要处理的文件")
        return

    rfr_series = None
    if os.path.exists(RFR_CACHE_FILE):
        try:
            rfr_df = pd.read_parquet(RFR_CACHE_FILE)
            rfr_df.index = pd.to_datetime(rfr_df.index).normalize()
            rfr_series = rfr_df['DGS3MO']
            print("✅ RFR 缓存加载成功")
        except: pass
        
    print(f"🚀 开始生成离线高频压测数据，并发 {MAX_WORKERS}...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(rfr_series,)) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating 1s Greeks"):
            res = future.result()
            if '✅' not in res and '⏩' not in res:
                tqdm.write(res)
                
    print("🎉 秒级压测数据生成完成！所有微观噪音和复权错位已抹平！")

if __name__ == "__main__":
    main()