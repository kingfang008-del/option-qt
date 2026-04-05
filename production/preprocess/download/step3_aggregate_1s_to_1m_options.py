import os
import pandas as pd
import numpy as np
import concurrent.futures
import logging
import warnings
from datetime import datetime
from tqdm import tqdm
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 秒级期权源数据目录
RAW_OPT_DIR = "/mnt/s990/data/raw_1s/stress_test_1s_greeks"

# 🚀 [核心修改] 直接读取 resample_1.py 产出的完美分钟级正股数据
CLEAN_STOCK_BASE = "/home/kingfang007/train_data/spnq_train_resampled"  

# 最终产出目录
OUTPUT_OPT_DIR = "/home/kingfang007/train_data/quote_options_day_iv" 
RFR_CACHE_FILE = "/home/kingfang007/risk_free_rates.parquet"

MAX_WORKERS = 16

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aggregate_Pipeline")

# ================= 全局变量与复权因子 =================
G_RFR_SERIES = None

def init_worker_rfr(rfr_series):
    global G_RFR_SERIES
    G_RFR_SERIES = rfr_series

# 因为正股已经是复权后的最新价格，所以历史期权数据必须进行同等除权，以匹配正股
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

# ================= 纯函数：向量化 Greeks 计算 =================
def get_d1_d2(S, K, t, r, sigma):
    valid_mask = (t > 1e-9) & (sigma > 1e-9)
    d1 = np.full_like(S, np.nan)
    d2 = np.full_like(S, np.nan)
    
    S_valid = S[valid_mask]
    K_valid = K[valid_mask]
    t_valid = t[valid_mask]
    r_valid = r[valid_mask]
    sigma_valid = sigma[valid_mask]
    
    sigma_sqrt_t = sigma_valid * np.sqrt(t_valid)
    d1_valid = (np.log(S_valid / K_valid) + (r_valid + 0.5 * sigma_valid**2) * t_valid) / sigma_sqrt_t
    d2_valid = d1_valid - sigma_sqrt_t
    
    d1[valid_mask] = d1_valid
    d2[valid_mask] = d2_valid
    return d1, d2

def vectorized_vanna(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    vanna = np.zeros_like(sigma)
    mask = sigma > 1e-9
    vanna[mask] = -phi_d1[mask] * (d2_val[mask] / sigma[mask])
    return vanna

def vectorized_charm(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    sigma_sqrt_t = np.zeros_like(sigma)
    sigma_sqrt_t[sigma > 1e-9] = sigma[sigma > 1e-9] * np.sqrt(t[sigma > 1e-9])
    
    term1 = np.zeros_like(sigma_sqrt_t)
    mask1 = sigma_sqrt_t > 1e-9
    term1[mask1] = r[mask1] / sigma_sqrt_t[mask1]
    
    term2 = np.zeros_like(t)
    mask2 = t > 1e-9
    term2[mask2] = d2_val[mask2] / (2 * t[mask2])
    
    charm = np.zeros_like(sigma)
    mask = sigma > 1e-9
    charm[mask] = -phi_d1[mask] * (term1[mask] - term2[mask])
    return charm

def vectorized_rho(S, K, t, r, sigma, is_call):
    _, d2_val = get_d1_d2(S, K, t, r, sigma)
    rho = np.zeros_like(sigma)
    mask = sigma > 1e-9
    
    discount_factor = np.zeros_like(sigma)
    discount_factor[mask] = K[mask] * t[mask] * np.exp(-r[mask] * t[mask])
    
    call_mask = mask & is_call
    if call_mask.any():
        rho[call_mask] = discount_factor[call_mask] * norm.cdf(d2_val[call_mask]) / 100.0
        
    put_mask = mask & (~is_call)
    if put_mask.any():
        rho[put_mask] = -discount_factor[put_mask] * norm.cdf(-d2_val[put_mask]) / 100.0
        
    return rho

# ================= 核心处理：处理期权 (1s -> 1m、期权复权并计算 Greeks) =================
def process_option_file(file_path):
    try:
        parts = file_path.split(os.sep)
        symbol = parts[-2]
        filename = parts[-1] # 形如 NVDA_2024-01-05.parquet
        date_str = filename.replace(symbol + "_", "").replace(".parquet", "")
        
        # 提取年月以匹配 resample_1.py 的输出路径
        year, month, day = date_str.split('-')
        
        out_dir = os.path.join(OUTPUT_OPT_DIR, symbol)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        if os.path.exists(out_path):
            return f"⏩ [Option] {symbol} {date_str} exists"
            
        df_1s = pd.read_parquet(file_path)
        if df_1s.empty: 
            return f"⚠️ [Option] {symbol} {date_str} is empty"
        
        # 🚀 1. 读取 resample_1.py 生成的绝对纯净正股数据
        stock_path = os.path.join(CLEAN_STOCK_BASE, symbol, "regular", "09:30-16:00", "1min", f"{year}-{month}.parquet")
        stock_map = {}
        if os.path.exists(stock_path):
            stk_1m = pd.read_parquet(stock_path)
            # 过滤出当天的正股数据，加速映射
            stk_1m = stk_1m[stk_1m['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
            stk_1m.set_index('timestamp', inplace=True)
            stock_map = stk_1m['close'].to_dict()
        else:
            return f"⚠️ [Stock Missing] {stock_path} not found. Ensure resample_1.py is run first."

        if not stock_map:
            return f"⚠️ [Stock Empty] No stock data found for {date_str} in monthly file."

        # 2. 统一期权时间戳，强制转换为美东时间
        if 'timestamp' in df_1s.columns:
            df_1s['timestamp'] = pd.to_datetime(df_1s['timestamp'], utc=True).dt.tz_convert('America/New_York')
        else:
            df_1s['timestamp'] = pd.to_datetime(df_1s['ts'], unit='s', utc=True).dt.tz_convert('America/New_York')
        
        # 🚀 3. 向量化重采样：强制使用 label='right', closed='left'，完美对齐正股基准
        df_1s.set_index('timestamp', inplace=True)
        agg_df = df_1s.groupby([
            'ticker', 
            pd.Grouper(freq='1min', label='right', closed='left')
        ]).agg({
            'price': ['first', 'max', 'min', 'last'],
            'bid': 'last', 'ask': 'last', 'bid_size': 'last', 'ask_size': 'last',
            'iv': 'last', 'delta': 'last', 'gamma': 'last', 'vega': 'last', 'theta': 'last',
            'bucket_id': 'last', 'strike': 'last', 'underlying': 'last'
        })
        
        agg_df.columns = [
            'open', 'high', 'low', 'close', 'bid', 'ask', 'bid_size', 'ask_size',
            'iv', 'delta', 'gamma', 'vega', 'theta', 'bucket_id', 'strike_price', 'underlying'
        ]
        
        agg_df.dropna(subset=['close'], inplace=True)
        if agg_df.empty: 
            return f"⚠️ [Option] {symbol} {date_str} no data after resample"
        agg_df.reset_index(inplace=True) # 这会将 ticker 和 timestamp 恢复为列
        
        # 4. 基础信息提取
        unique_tickers = pd.Series(agg_df['ticker'].unique())
        contract_types = unique_tickers.str.extract(r'[A-Z]+\d{6}([CP])\d{8}')[0].str.lower()
        exp_dates = pd.to_datetime('20' + unique_tickers.str.extract(r'[A-Z]+(\d{6})[CP]\d{8}')[0], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
        
        ticker_meta = pd.DataFrame({'ticker': unique_tickers, 'contract_type': contract_types, 'expiration_date': exp_dates})
        agg_df = agg_df.merge(ticker_meta, on='ticker', how='left')

        # 🚀 5. 精准正股对齐 (时间戳已绝对一致)
        agg_df['stock_close'] = agg_df['timestamp'].map(stock_map)
        agg_df['stock_close'] = agg_df['stock_close'].ffill() # 断流前填保护
        agg_df.dropna(subset=['stock_close'], inplace=True) # 丢弃盘前没有正股数据的游离期权单
        
        # ----------- 🚀 核心逻辑：仅对期权进行除权（缩小价格，放大数量） -----------
        # 因为获取到的 stock_close 已经是复权过的数据，这里处理让期权强制匹配现代的正股价格尺度
        if symbol in KNOWN_SPLITS:
            restore_factor = np.ones(len(agg_df))
            splits = sorted(KNOWN_SPLITS[symbol], key=lambda x: x[0])
            for split_str, ratio in splits:
                split_date = pd.Timestamp(split_str).tz_localize('America/New_York')
                mask = agg_df['timestamp'] < split_date
                restore_factor[mask] /= ratio
            
            # 期权价格类与行权价：缩水
            for col in ['strike_price', 'open', 'high', 'low', 'close', 'bid', 'ask']:
                if col in agg_df.columns:
                    agg_df[col] = agg_df[col] / restore_factor
                    
            # 期权数量类：放大
            for col in ['bid_size', 'ask_size']:
                if col in agg_df.columns:
                    agg_df[col] = agg_df[col] * restore_factor
                    
            # 原始历史 Greeks 的同步调整
            for col in ['vega', 'theta']:
                if col in agg_df.columns:
                    agg_df[col] = agg_df[col] / restore_factor
            if 'gamma' in agg_df.columns:
                agg_df['gamma'] = agg_df['gamma'] * restore_factor
        # ---------------------------------------------
        
        agg_df['volume'] = agg_df['bid_size'] + agg_df['ask_size'] 
        agg_df['spread_pct'] = (agg_df['ask'] - agg_df['bid']) / (agg_df['close'] + 1e-6)
        agg_df['volume_imbalance'] = (agg_df['bid_size'] - agg_df['ask_size']) / (agg_df['bid_size'] + agg_df['ask_size'] + 1e-6)
        
        # ================== 6. 真实高阶 Greeks 重算 ==================
        # 经过上面的除权，S (复权正股) 和 K (除权期权) 现在绝对处于同一维度！BSM 计算完全有效。
        S = agg_df['stock_close'].values
        K = agg_df['strike_price'].values
        iv = agg_df['iv'].values
        
        current_ts = agg_df['timestamp'] 
        expiry_ts = pd.to_datetime(agg_df['expiration_date'])
        if expiry_ts.dt.tz is None:
            expiry_ts = expiry_ts.dt.tz_localize('America/New_York')
            
        expiry_ts = expiry_ts + pd.Timedelta(hours=16) # 到期日美东下午4点
        T_years = (expiry_ts - current_ts).dt.total_seconds().values / 31557600.0
        T_years = np.maximum(T_years, 1e-6) 
        
        r = np.full_like(S, 0.04)
        if G_RFR_SERIES is not None and len(G_RFR_SERIES) > 0:
            search_keys = current_ts.dt.normalize().dt.tz_localize(None)
            r_idx = G_RFR_SERIES.index.searchsorted(search_keys)
            r_idx = np.clip(r_idx, 0, len(G_RFR_SERIES) - 1)
            r = np.nan_to_num(G_RFR_SERIES.values[r_idx], nan=0.04)
            
        is_call = (agg_df['contract_type'] == 'c').values

        agg_df['vanna'] = vectorized_vanna(S, K, T_years, r, iv)
        agg_df['charm'] = vectorized_charm(S, K, T_years, r, iv)
        agg_df['rho'] = vectorized_rho(S, K, T_years, r, iv, is_call)
            
        # 7. 整理最终列并输出
        final_cols = [
            'timestamp', 'ticker', 'bucket_id', 'expiration_date', 'contract_type', 'strike_price', 
            'open', 'high', 'low', 'close', 'volume', 
            'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance',
            'iv', 'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'charm', 'stock_close'
        ]
        
        for co in final_cols:
            if co not in agg_df.columns: agg_df[co] = 0.0
            
        final_df = agg_df[final_cols]
        final_df.to_parquet(out_path, index=False, compression='zstd')
        
        return f"✅ [Option] {symbol} {date_str} resampled & adjusted ({len(final_df)} bars)"

    except Exception as e:
        return f"❌ [Option Error] {file_path}: {e}"

# ================= 主控制流 =================
def run_pipeline(input_dir, process_func, desc, max_workers, initializer=None, initargs=()):
    if not os.path.exists(input_dir):
        print(f"❌ Directory not found: {input_dir}")
        return
        
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".parquet"):
                all_files.append(os.path.join(root, f))
                
    if not all_files:
        print(f"⚠️ No parquet files found in {input_dir}.")
        return
        
    print(f"🚀 {desc}: Processing {len(all_files)} files using {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=initializer, 
        initargs=initargs
    ) as executor:
        futures = {executor.submit(process_func, f): f for f in all_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            res = future.result()
            if '✅' not in res and '⏩' not in res:
                tqdm.write(res)

def main():
    rfr_series = None
    if os.path.exists(RFR_CACHE_FILE):
        try:
            rfr_df = pd.read_parquet(RFR_CACHE_FILE)
            rfr_df.index = pd.to_datetime(rfr_df.index).normalize()
            rfr_series = rfr_df['DGS3MO']
            print(f"✅ Loaded Risk Free Rates from cache (Data points: {len(rfr_series)})")
        except Exception as e:
            print(f"⚠️ Failed to parse RFR file: {e}. Will fallback to 4%.")
    else:
        print(f"⚠️ RFR file {RFR_CACHE_FILE} not found. Will fallback to 4%.")

    print("\n" + "="*50)
    print("🌟 Stage 1: Converting 1s Options to 1m Options & Matching with Resampled Stocks...")
    print("="*50)
    run_pipeline(
        RAW_OPT_DIR, 
        process_option_file, 
        "Aggregating Options", 
        MAX_WORKERS, 
        initializer=init_worker_rfr, 
        initargs=(rfr_series,)
    )
    
    print("\n🎉 All Option Data Processing Complete!")

if __name__ == "__main__":
    main()