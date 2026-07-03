import os
import datetime
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import logging
import re
from pytz import timezone
from polygon import RESTClient

# ================= 全局配置 =================
API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"  # 你的 Polygon API Key
TARGET_MAP_FILE = "/home/kingfang007/train_data/locked_targets_map.parquet"
OUTPUT_DIR = "/mnt/s990/data/raw_1s/options"
STOCK_OUTPUT_DIR = "/mnt/s990/data/raw_1s/stocks"
RFR_CACHE_FILE = "/home/kingfang007/risk_free_rates.parquet"

# Polygon 并发限制取决于您的套餐，商业版可以开很大
MAX_WORKERS = 90 
DOWNLOAD_OPTIONS = True  # [NEW] 是否从 Polygon 下载期权 Quotes
FORCE_GREEKS_RECOMPUTE = False # [NEW] 如果已存在，是否强制重新计算 Greeks (用于秒级正股对齐)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("Polygon_1s_Sniper")
eastern = timezone('America/New_York')
# ============================================

# ================= BSM 本地 Greeks 计算 =================
try:
    from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False
    logger.warning("⚠️ py_vollib_vectorized not found, Greeks will be zeros.")

def compute_greeks_for_df(df, stock_price_map, r_val=0.04):
    """
    本地 BSM 向量化计算 IV + Greeks
    """
    if not HAS_VOLLIB or df.empty:
        for col in ['iv', 'delta', 'gamma', 'vega', 'theta']:
            if col not in df.columns: df[col] = 0.0
        return df
    
    # 提取到期日
    clean_tickers = df['ticker'].str.replace('O:', '', regex=False)
    extracted = clean_tickers.str.extract(r'^[A-Z]+(\d{6})([CP])\d{8}$')
    df['expiry'] = pd.to_datetime('20' + extracted[0], format='%Y%m%d', errors='coerce')
    df['opt_type'] = extracted[1].map({'C': 'c', 'P': 'p'})
    
    # 映射股票价格 (修正：使用四舍五入后的秒级整数匹配，防止浮点精度问题)
    stk_ts = df['ts'].round(0).astype('int64')
    df['stock_close'] = stk_ts.map(stock_price_map)
    df['stock_close'] = df['stock_close'].ffill().bfill()
    
    # 到期时间
    current_ts = df['timestamp']
    expiry_ts = df['expiry'].dt.tz_localize('America/New_York') + pd.Timedelta(hours=16)
    T_years = (expiry_ts - current_ts).dt.total_seconds().values / 31557600.0
    T_years = np.maximum(T_years, 1e-6)
    
    P = df['price'].values.astype(float)
    S = df['stock_close'].values.astype(float)
    K = df['strike'].values.astype(float)
    r = np.full_like(P, r_val)
    
    is_call = (df['opt_type'] == 'c').values
    is_put = (df['opt_type'] == 'p').values
    
    iv = np.zeros_like(P, dtype=float)
    valid = (P > 0.0) & (S > 0.0) & (K > 0.0) & (T_years > 0.0)
    
    # Debug Logging
    # if valid.any():
    #     logger.info(f"  [Greeks Debug] P_avg={np.nanmean(P):.3f}, S_avg={np.nanmean(S):.3f}, K_avg={np.nanmean(K):.3f}, T_avg={np.nanmean(T_years):.4f}, r={r_val:.4f}")
    # else:
    #     logger.warning(f"  [Greeks Warning] No valid rows for Greeks! P>0:{ (P>0).sum() }, S>0:{ (S>0).sum() }, K>0:{ (K>0).sum() }")
    
    try:
        if (is_call & valid).any():
            m = is_call & valid
            iv[m] = vectorized_implied_volatility(P[m], S[m], K[m], T_years[m], r[m], 'c', return_as='numpy', on_error='ignore')
        if (is_put & valid).any():
            m = is_put & valid
            iv[m] = vectorized_implied_volatility(P[m], S[m], K[m], T_years[m], r[m], 'p', return_as='numpy', on_error='ignore')
        # if valid.any():
        #     logger.info(f"  [IV Sample] First 5 valid IV: {iv[valid][:5]}")
    except Exception as e:
        logger.debug(f"  [IV] vectorized_implied_volatility error: {e}")
    
    iv = np.nan_to_num(iv, nan=0.0)
    df['iv'] = iv
    
    delta, gamma, vega, theta = np.zeros_like(iv), np.zeros_like(iv), np.zeros_like(iv), np.zeros_like(iv)
    valid_iv_mask = (iv > 0.01) & (iv < 5.0) & valid
    
    try:
        if is_call.any() and valid_iv_mask[is_call].any():
            vc = is_call & valid_iv_mask
            g_df = get_all_greeks('c', S[vc], K[vc], T_years[vc], r[vc], iv[vc], return_as='dataframe')
            delta[vc] = g_df['delta'].values
            gamma[vc] = g_df['gamma'].values
            vega[vc] = g_df['vega'].values
            theta[vc] = g_df['theta'].values
            
        if is_put.any() and valid_iv_mask[is_put].any():
            vp = is_put & valid_iv_mask
            g_df = get_all_greeks('p', S[vp], K[vp], T_years[vp], r[vp], iv[vp], return_as='dataframe')
            delta[vp] = g_df['delta'].values
            gamma[vp] = g_df['gamma'].values
            vega[vp] = g_df['vega'].values
            theta[vp] = g_df['theta'].values
            
        # if valid_iv_mask.any():
        #     logger.info(f"  [Greeks Sample] First 5 valid Delta: {delta[valid_iv_mask][:5]}")
    except Exception as e:
        logger.info(f"  [Greeks ERROR] get_all_greeks failed: {e}")
    
    df['delta'], df['gamma'], df['vega'], df['theta'] = delta, gamma, vega, theta
    
    # if valid_iv_mask.any():
    #     logger.info(f"  [Greeks Done] IV_mean={np.nanmean(iv[valid_iv_mask]):.4f}, Delta_mean={np.nanmean(delta[valid_iv_mask]):.4f}")
    
    df.drop(columns=['expiry', 'opt_type', 'stock_close'], inplace=True, errors='ignore')
    return df

# ================= 核心 Worker =================
def process_single_day_polygon(args):
    symbol, date_str, group_df, r_val = args
    client = RESTClient(API_KEY)
    
    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path) and not FORCE_GREEKS_RECOMPUTE:
        return f"⏩ {symbol} {date_str} exists"

    all_option_1s = []

    # 1. 获取 1s 级别正股价格 (用于 Greeks 计算)
    # 逻辑：优先读本地缓存 -> 其次从 Polygon 下载 1s Aggs -> 最后回退到 1m 升采样
    stock_dir = os.path.join(STOCK_OUTPUT_DIR, symbol)
    os.makedirs(stock_dir, exist_ok=True)
    stock_path = os.path.join(stock_dir, f"{symbol}_{date_str}.parquet")
    
    stk_df = None
    if os.path.exists(stock_path):
        try:
            stk_df = pd.read_parquet(stock_path)
        except Exception: stk_df = None

    if stk_df is None:
        # [RESTORED] 从 Polygon 获取真实 1s Aggregates
        try:
            aggs = list(client.list_aggs(
                ticker=symbol, multiplier=1, timespan="second",
                from_=date_str, to=date_str, limit=50000
            ))
            if aggs:
                stk_df = pd.DataFrame([{
                    'ts': a.timestamp / 1000.0,
                    'open': a.open, 'high': a.high, 'low': a.low, 'close': a.close, 'volume': a.volume
                } for a in aggs])
                stk_df['timestamp'] = pd.to_datetime(stk_df['ts'], unit='s', utc=True).dt.tz_convert(eastern)
                stk_df.to_parquet(stock_path, index=False)
        except Exception as e:
            logger.warning(f"  [Stock] Polygon 1s download failed for {symbol} {date_str}: {e}")

    if stk_df is not None:
        # 使用整数秒作为 Key，提高 Map 匹配率
        stk_df['ts_int'] = stk_df['ts'].round(0).astype('int64')
        stock_price_map = dict(zip(stk_df['ts_int'], stk_df['close']))
    else:
        stock_price_map = {}
        month_str = date_str[:7]
        m1_path = f"/home/kingfang007/train_data/spnq_train_resampled/{symbol}/regular/09:30-16:00/1min/{month_str}.parquet"
        if os.path.exists(m1_path):
            try:
                m1_df = pd.read_parquet(m1_path)
                m1_df['date_only'] = m1_df['timestamp'].dt.date
                m1_df = m1_df[m1_df['date_only'] == pd.to_datetime(date_str).date()]
                if not m1_df.empty:
                    m1_df.set_index('timestamp', inplace=True)
                    start_ts = pd.to_datetime(f"{date_str} 09:30:00").tz_localize('America/New_York')
                    end_ts = pd.to_datetime(f"{date_str} 16:00:00").tz_localize('America/New_York')
                    grid = pd.date_range(start=start_ts, end=end_ts, freq='s', tz='America/New_York')
                    m1_1s = m1_df.reindex(grid).ffill().bfill()
                    m1_1s['ts'] = m1_1s.index.astype('int64') / 1e9
                    stock_price_map = dict(zip(m1_1s['ts'], m1_1s['close']))
            except Exception as e:
                logger.debug(f"  [Stock] 1m fallback failed for {symbol} {date_str}: {e}")

    # 2. 获取期权行情 (下载或从本地加载现有数据进行 Greeks 重算)
    if not DOWNLOAD_OPTIONS and os.path.exists(out_path):
        try:
            # 如果不下载且本地已有，加载现有结果重算 Greeks
            loaded_df = pd.read_parquet(out_path)
            # 如果已有 Greeks 列，删掉它们强制重算
            for col in ['iv', 'delta', 'gamma', 'vega', 'theta']:
                if col in loaded_df.columns: loaded_df.drop(columns=[col], inplace=True)
            all_option_1s.append(loaded_df)
        except Exception as e:
            logger.debug(f"  [Option] Failed to load existing option data for {symbol} {date_str}: {e}")

    if DOWNLOAD_OPTIONS:
        for _, row in group_df.iterrows():
            occ = row['contract_symbol']
            poly_ticker = occ if occ.startswith("O:") else f"O:{occ}"
            
            try:
                quotes = list(client.list_quotes(ticker=poly_ticker, timestamp_gte=date_str, timestamp_lte=date_str, limit=50000))
                if not quotes: continue
            
                df = pd.DataFrame([{
                    'timestamp': getattr(q, 'sip_timestamp', getattr(q, 'participant_timestamp', 0)),
                    'bid': getattr(q, 'bid_price', 0.0),
                    'ask': getattr(q, 'ask_price', 0.0),
                    'bid_size': getattr(q, 'bid_size', 0.0),
                    'ask_size': getattr(q, 'ask_size', 0.0)
                } for q in quotes])
                
                # 清洗 & 时间对焦
                df = df[(df['bid'] > 0) & (df['ask'] >= df['bid'])].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert(eastern)
                
                # RTH 过滤
                time_series = df['timestamp'].dt.time
                df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
                if df.empty: continue
                
                # 转为 1s 坐标点 (重采样保留最后一笔)
                df['ts_1s'] = df['timestamp'].dt.floor('1s')
                df = df.sort_values('timestamp').drop_duplicates(subset=['ts_1s'], keep='last').copy()
                
                df['timestamp'] = df['ts_1s']
                df['ts'] = df['timestamp'].astype('int64') / 1e9
                df['ticker'] = occ.replace('O:', '')
                df['bucket_id'] = row['bucket_id']
                df['tag'] = row.get('tag', '')
                df['underlying'] = symbol
                df['mid_price'] = (df['bid'] + df['ask']) / 2.0
                df['price'] = df['mid_price']
                
                # 提取行权价
                match = re.search(r'[CP](\d{8})$', df['ticker'].iloc[0])
                df['strike'] = float(match.group(1)) / 1000.0 if match else 0.0
                
                all_option_1s.append(df)
            except Exception as e:
                logger.error(f"  [Option] Contract {occ} process error: {e}", exc_info=True)
                continue

    if not all_option_1s:
        return f"⚠️ {symbol} {date_str}: No valid quotes."

    final_df = pd.concat(all_option_1s, ignore_index=True)
    
    # 3. 动态 BSM Greeks
    final_df = compute_greeks_for_df(final_df, stock_price_map, r_val=r_val)
    
    final_cols = [
        'ts', 'timestamp', 'ticker', 'tag', 'bucket_id', 'underlying',
        'bid', 'ask', 'bid_size', 'ask_size', 'price', 'strike',
        'iv', 'delta', 'gamma', 'vega', 'theta'
    ]
    final_df = final_df[[c for c in final_cols if c in final_df.columns]]
    final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    
    return f"🎯 {symbol} {date_str}: Success! {len(final_df)} rows of real 1s sniper data."

def load_rfr_series():
    try:
        import pandas_datareader.data as web
        import datetime
        fetch_start = pd.Timestamp(datetime.date(2020, 1, 1)).normalize()
        fetch_end = pd.Timestamp(datetime.date.today() + datetime.timedelta(days=14)).normalize()
        
        logger.info("Downloading fresh RFR data from FRED...")
        new_data = web.DataReader('DGS3MO', 'fred', fetch_start, fetch_end)
        new_data.index = pd.to_datetime(new_data.index).normalize()
        new_data = new_data / 100.0
        
        rfr_df = new_data.resample('D').ffill()
        if rfr_df.isnull().any().any():
            logger.warning("⚠️ RFR data has holes/nans AFTER ffill. Using 4% fallback for missing gaps.")
        rfr_df = rfr_df.fillna(0.04)
        
        try:
            rfr_df.to_parquet(RFR_CACHE_FILE)
            logger.info(f"RFR Cache saved to {RFR_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Could not save RFR cache: {e}")
            
        return rfr_df['DGS3MO']
    except Exception as e:
        logger.warning(f"Failed to download RFR: {e}. Falling back to cache.")
        try:
            if os.path.exists(RFR_CACHE_FILE):
                df = pd.read_parquet(RFR_CACHE_FILE)
                if isinstance(df, pd.DataFrame):
                    return df.iloc[:, 0]
                return df
        except Exception as cache_e:
            logger.error(f"Cache load also failed: {cache_e}")
    return None

def main():
    import time
    if not os.path.exists(TARGET_MAP_FILE):
        logger.error("❌ Target map not found.")
        return
        
    target_map = pd.read_parquet(TARGET_MAP_FILE)
    rfr_series = load_rfr_series()
    
    # 按照股票代码进行分组聚合
    symbols = target_map['symbol'].unique()
    logger.info(f"🚀 Starting Polygon 1s sniper for {len(symbols)} symbols...")

    for sym in symbols:
        sym_df = target_map[target_map['symbol'] == sym]
        
        date_tasks = []
        for d, g in sym_df.groupby('date_str'):
            # [Optimization] Skip if file already exists before sending to executor
            out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{d}.parquet")
            if os.path.exists(out_path) and not FORCE_GREEKS_RECOMPUTE:
                continue

            r_val = 0.045 # Fallback
            if rfr_series is not None:
                try:
                    t_date = pd.to_datetime(d).normalize()
                    # 采用与 option_cac_day_vectorized_day 一致的 searchsorted 逻辑
                    if t_date in rfr_series.index:
                        r_val = float(rfr_series.loc[t_date])
                    else:
                        idx = rfr_series.index.searchsorted(t_date.tz_localize(None))
                        idx = np.clip(idx, 0, len(rfr_series) - 1)
                        r_val = float(rfr_series.iloc[idx])
                except Exception as e: 
                    logger.debug(f"RFR lookup error for {d}: {e}")
            date_tasks.append((sym, d, g, r_val))
        
        if not date_tasks:
            logger.info(f"⏩ Symbol [{sym}] is already fully processed. Skipping.")
            continue
            
        logger.info(f"📥 Processing symbol [{sym}] for {len(date_tasks)} remaining days...")
        
        # 对于同一只股票的多个日期，我们可以有限并行的下载 (限制到 5 个 worker，稳一点)
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            list(tqdm(executor.map(process_single_day_polygon, date_tasks), 
                      total=len(date_tasks), desc=f"Symbol {sym}"))
        
        # [🔥 核心逻辑] 每下载完一只股票的所有数据，强制休息 20s，防止触发 Polygon 的 Pacing Limit
        logger.info(f"💤 Symbol {sym} done, cooling down for 20s...")
        #time.sleep(1)

    logger.info("🏁 All symbols processed.")

if __name__ == "__main__":
    main()
