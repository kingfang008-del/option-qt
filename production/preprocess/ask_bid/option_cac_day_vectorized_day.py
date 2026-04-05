import datetime
import pandas as pd
import numpy as np
import sqlite3
import os
import glob
import logging
from tqdm import tqdm
import pandas_datareader.data as web
from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
from scipy.stats import norm
import multiprocessing
import gc
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局过滤警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Found Below Intrinsic contracts")

# ==============================================================================
# 全局变量 (用于多进程共享 RFR 数据)
# ==============================================================================
G_RFR_SERIES = None

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

def init_worker_rfr(rfr_series):
    global G_RFR_SERIES
    G_RFR_SERIES = rfr_series

# ==============================================================================
# 纯函数：Greeks 计算
# ==============================================================================
def get_d1_d2(S, K, t, r, sigma):
    valid_mask = (t > 1e-9) & (sigma > 1e-9)
    d1 = np.full_like(S, np.nan)
    d2 = np.full_like(S, np.nan)
    
    # 修复点：确保所有参与计算的矩阵都经过 valid_mask 过滤！
    S_valid = S[valid_mask]
    K_valid = K[valid_mask]  # 👈 新增这一行
    t_valid = t[valid_mask]
    r_valid = r[valid_mask]
    sigma_valid = sigma[valid_mask]
    
    sigma_sqrt_t = sigma_valid * np.sqrt(t_valid)
    
    # 修复点：除以 K_valid 而不是 K
    d1_valid = (np.log(S_valid / K_valid) + (r_valid + 0.5 * sigma_valid**2) * t_valid) / sigma_sqrt_t
    d2_valid = d1_valid - sigma_sqrt_t
    
    d1[valid_mask] = d1_valid
    d2[valid_mask] = d2_valid
    
    return d1, d2
def vectorized_vanna(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    # [Fix] 使用 mask 避免 eager-division 触发 RuntimeWarning
    vanna = np.zeros_like(sigma)
    mask = sigma > 1e-9
    vanna[mask] = -phi_d1[mask] * (d2_val[mask] / sigma[mask])
    return vanna

def vectorized_charm(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    sigma_sqrt_t = sigma * np.sqrt(t)
    
    # [Fix] 分步计算并加 mask，消灭 RuntimeWarning: divide by zero
    term1 = np.zeros_like(sigma_sqrt_t)
    mask1 = sigma_sqrt_t > 1e-9
    term1[mask1] = r[mask1] / sigma_sqrt_t[mask1]
    
    term2 = np.zeros_like(t)
    mask2 = t > 1e-9
    term2[mask2] = d2_val[mask2] / (2 * t[mask2])
    
    charm = -phi_d1 * (term1 - term2)
    return charm

# ==============================================================================
# 核心计算逻辑 (按日处理)
# ==============================================================================
def compute_single_day_file(args):
    """
    【极简极速版】
    职责：加载6个锚点合约，计算Greeks，无损透传微观特征(bucket_id, spread_pct等)
    """
    parquet_path, symbol, underlying_df, iv_dir = args
    filename = os.path.basename(parquet_path)
    day_str = filename.replace('.parquet', '').split('_')[-1] 
    
    iv_filename = os.path.join(iv_dir, filename)
    
    if os.path.exists(iv_filename):
        return None 

    try:
        df_opt_raw = pd.read_parquet(parquet_path)
        if df_opt_raw.empty: return None
        
        df_opt = df_opt_raw.copy()
        
        # ================= 核心修复 1：时间戳格式化与强制美东时区对齐 =================
        if not pd.api.types.is_datetime64_any_dtype(df_opt['timestamp']):
            df_opt['timestamp'] = pd.to_datetime(df_opt['timestamp'])
            
        if df_opt['timestamp'].dt.tz is None:
            # 数据是不带时区的纯净美东时间，强制附加 NY 时区
            df_opt['timestamp'] = df_opt['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')
        else:
            df_opt['timestamp'] = df_opt['timestamp'].dt.tz_convert('America/New_York')
            
        df_opt = df_opt.sort_values('timestamp').set_index('timestamp')
        
        # ================= 合约提取 =================
        # O:NVDA230609P00390000 -> NVDA230609P00390000
        clean_tickers = df_opt['ticker'].str.replace('O:', '', regex=False)
        extracted = clean_tickers.str.extract(r'^[A-Z]+(\d{6})([CP])(\d{8})$')
        df_opt['expiration_date'] = pd.to_datetime('20' + extracted[0], format='%Y%m%d')
        df_opt['contract_type'] = extracted[1].str.lower()
        df_opt['strike_price'] = extracted[2].astype(float) / 1000.0

        df_opt.dropna(subset=['expiration_date', 'contract_type', 'strike_price'], inplace=True)
        if df_opt.empty: return None

        # ================= Time Series 合并底层股票数据 =================
        target_date = pd.to_datetime(day_str).tz_localize('America/New_York')
        day_start = target_date.replace(hour=0, minute=0, second=0)
        day_end = target_date.replace(hour=23, minute=59, second=59)
        
        u_df_day = underlying_df.loc[day_start:day_end]
        if u_df_day.empty: 
            return None

        # Asof Merge
        df_merged = pd.merge_asof(
            df_opt,
            u_df_day[['close']].rename(columns={'close': 'stock_close'}),
            left_index=True, right_index=True, direction='backward'
        )
        df_merged.dropna(subset=['stock_close'], inplace=True)
        if df_merged.empty: return None

        # ================= 期权除权(Split Adjust) =================
        if symbol in KNOWN_SPLITS:
            restore_factor = np.ones(len(df_merged))
            splits = sorted(KNOWN_SPLITS[symbol], key=lambda x: x[0])
            for split_str, ratio in splits:
                split_date = pd.Timestamp(split_str).tz_localize('America/New_York')
                mask = df_merged.index < split_date
                restore_factor[mask] /= ratio
            
            # 【核心修复 2：列名适配新数据格式 'open', 'high', 'low', 'close'】
            for col in ['strike_price', 'open', 'high', 'low', 'close']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col] / restore_factor
            
            if 'volume' in df_merged.columns:
                df_merged['volume'] = df_merged['volume'] * restore_factor

        # ================= 向量化运算 Greeks & IV =================
        P = df_merged['close'].values # 这里已修正为 close
        S = df_merged['stock_close'].values
        K = df_merged['strike_price'].values
        
        # current_ts_idx = df_merged.index
        # expiry_ts = pd.to_datetime(df_merged['expiration_date'])
        # if expiry_ts.dt.tz is None:
        #     expiry_ts = expiry_ts.dt.tz_localize('America/New_York')

        current_ts_idx = df_merged.index
        expiry_ts = pd.to_datetime(df_merged['expiration_date'])
        if expiry_ts.dt.tz is None:
            expiry_ts = expiry_ts.dt.tz_localize('America/New_York')
            
        # 🚀 致命修复：强制将到期时间推迟到下午 16:00，对齐真实世界的期权闭市时间！
        # 否则 0DTE 期权在早上 10 点算出来的 T 会是负数或极端异常值，导致 Greeks 爆炸！
        expiry_ts = expiry_ts + pd.Timedelta(hours=16)

        time_diff = expiry_ts - current_ts_idx
        T_years = time_diff.dt.total_seconds().values / 31557600.0
        T_years = np.maximum(T_years, 1e-6)
        
        search_keys = current_ts_idx.normalize().tz_localize(None)
        r_idx = G_RFR_SERIES.index.searchsorted(search_keys)
        r_idx = np.clip(r_idx, 0, len(G_RFR_SERIES) - 1)
        r = G_RFR_SERIES.values[r_idx]
        r = np.nan_to_num(r, nan=0.04)

        is_call = (df_merged['contract_type'] == 'c').values
        is_put = (df_merged['contract_type'] == 'p').values

        iv = np.zeros_like(P, dtype=float)
        
        if is_call.any():
            iv[is_call] = vectorized_implied_volatility(
                P[is_call], S[is_call], K[is_call], T_years[is_call], r[is_call], 'c', return_as='numpy', on_error='ignore'
            )
        if is_put.any():
            iv[is_put] = vectorized_implied_volatility(
                P[is_put], S[is_put], K[is_put], T_years[is_put], r[is_put], 'p', return_as='numpy', on_error='ignore'
            )

        # 【核心优化】：绝对不要删掉废行！将异常 IV 填充为 0。
        # 这是为了 100% 保持 6个 bucket 的结构，下游展平时会自动 ffill
        iv[np.isnan(iv)] = 0.0
        df_merged['iv'] = iv
        
        delta = np.zeros_like(iv)
        gamma = np.zeros_like(iv)
        vega = np.zeros_like(iv)
        theta = np.zeros_like(iv)
        rho = np.zeros_like(iv)

        # 仅对合理的 IV 计算 Greeks
        valid_iv_mask = (iv > 0) & (iv < 5.0)
        
        if is_call.any() and valid_iv_mask[is_call].any():
            vc = is_call & valid_iv_mask
            gd_c = get_all_greeks('c', S[vc], K[vc], T_years[vc], r[vc], iv[vc], return_as='dict')
            delta[vc] = gd_c['delta']
            gamma[vc] = gd_c['gamma']
            vega[vc] = gd_c['vega']
            theta[vc] = gd_c['theta']
            rho[vc] = gd_c['rho']

        if is_put.any() and valid_iv_mask[is_put].any():
            vp = is_put & valid_iv_mask
            gd_p = get_all_greeks('p', S[vp], K[vp], T_years[vp], r[vp], iv[vp], return_as='dict')
            delta[vp] = gd_p['delta']
            gamma[vp] = gd_p['gamma']
            vega[vp] = gd_p['vega']
            theta[vp] = gd_p['theta']
            rho[vp] = gd_p['rho']

        df_merged['delta'] = delta
        df_merged['gamma'] = gamma
        df_merged['vega'] = vega
        df_merged['theta'] = theta
        df_merged['rho'] = rho
        df_merged['vanna'] = vectorized_vanna(S, K, T_years, r, iv)
        df_merged['charm'] = vectorized_charm(S, K, T_years, r, iv)
        
        # ================= 核心修复 3：100% 透传最新微观特征 =================
        # 我们不再计算 high_features (直接删除了相关废代码)，直接无损抛给下游
        final_cols = [
            'ticker', 'bucket_id', 'expiration_date', 'contract_type', 'strike_price', 
            'open', 'high', 'low', 'close', 'volume', 
            'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance',
            'iv', 'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'charm', 'stock_close'
        ]
        
        df_res = df_merged[final_cols].copy()
        
        # 写入 Parquet
        df_res.reset_index().to_parquet(iv_filename, index=False)
        
        return day_str
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        return None

# ==============================================================================
# 主调度类
# ==============================================================================
class OptionIVCalculator:
    def __init__(self, data_root: str, db_path: str, option_root: str, iv_option_root: str):
        self.db_path = db_path
        self.data_root = data_root  
        self.option_root = option_root 
        self.iv_option_root = iv_option_root
        
        self.risk_free_cache = None
        self.risk_free_cache_file = '/home/kingfang007/risk_free_rates.parquet'
        os.makedirs(os.path.dirname(self.risk_free_cache_file), exist_ok=True)

    def _load_risk_free_rates(self, start_date: datetime.date, end_date: datetime.date):
        fetch_start = pd.Timestamp(start_date).normalize() - pd.Timedelta(days=14)
        fetch_end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=14)
        
        if self.risk_free_cache is not None:
            if self.risk_free_cache.index.min() <= fetch_start and self.risk_free_cache.index.max() >= fetch_end:
                return self.risk_free_cache

        rfr_df = pd.DataFrame()
        need_download = True

        if os.path.exists(self.risk_free_cache_file):
            try:
                df_cache = pd.read_parquet(self.risk_free_cache_file)
                df_cache.index = pd.to_datetime(df_cache.index).normalize()
                if not df_cache.empty:
                    rfr_df = df_cache
                    need_download = False
            except Exception: pass

        if need_download:
            try:
                logger.info("Downloading RFR...")
                new_data = web.DataReader('DGS3MO', 'fred', fetch_start, fetch_end)
                new_data.index = pd.to_datetime(new_data.index).normalize()
                new_data = new_data / 100.0
                if not rfr_df.empty:
                    rfr_df = new_data.combine_first(rfr_df)
                else:
                    rfr_df = new_data
                rfr_df = rfr_df.resample('D').ffill()
                if rfr_df.isnull().any().any():
                    logger.warning("⚠️ RFR data has holes/nans AFTER ffill. Using 4% fallback for missing gaps.")
                rfr_df = rfr_df.fillna(0.04)
                rfr_df.to_parquet(self.risk_free_cache_file)
            except Exception: pass
            
        if rfr_df.empty:
             idx = pd.date_range(fetch_start, fetch_end, freq='D').normalize()
             rfr_df = pd.DataFrame(index=idx, data={'DGS3MO': 0.04})

        self.risk_free_cache = rfr_df
        return self.risk_free_cache

    def get_target_symbols(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        from config import TARGET_SYMBOLS
         
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT symbol FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, TARGET_SYMBOLS)
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols

    def _get_underlying_df(self, symbol: str) -> pd.DataFrame | None:
        try:
            pattern = os.path.join(self.data_root, symbol, "**", "*.parquet")
            files = glob.glob(pattern, recursive=True)
            if not files: return None
            
            dfs = [pd.read_parquet(f) for f in files]
            if not dfs: return None
            
            df = pd.concat(dfs, ignore_index=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
                
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception:
            return None

    def process_symbol_task_entry(self, symbol):
        logger.info(f"🚀 [Process Start] {symbol}")
        
        underlying_df = self._get_underlying_df(symbol)
        if underlying_df is None: 
            logger.warning(f"No underlying data for {symbol}, Skipping.")
            return

        opt_dir = os.path.join(self.option_root, symbol)
        if not os.path.exists(opt_dir):
            return
            
        parquet_files = glob.glob(os.path.join(opt_dir, f"{symbol}_*.parquet"))
        if not parquet_files:
            return
            
        iv_dir = os.path.join(self.iv_option_root, symbol, "standard")
        os.makedirs(iv_dir, exist_ok=True)
        
        day_tasks = []
        for fp in parquet_files:
            day_tasks.append((fp, symbol, underlying_df, iv_dir))

        success_count = 0
        with ThreadPoolExecutor(max_workers=20) as executor:
            for day_str in executor.map(compute_single_day_file, day_tasks):
                if day_str: success_count += 1
                
        del underlying_df
        gc.collect()
        logger.info(f"✅ {symbol} Done. Processed {success_count} valid days.")

    def run(self, max_concurrent_stocks=5):
        symbols = self.get_target_symbols()
        if not symbols: return
    
        rfr_df = self._load_risk_free_rates(datetime.date(2020, 1, 1), datetime.date.today())
        rfr_series = rfr_df['DGS3MO']

        logger.info(f"Starting ProcessPool with {max_concurrent_stocks} workers for {len(symbols)} symbols...")
        
        with ProcessPoolExecutor(max_workers=max_concurrent_stocks, 
                                 initializer=init_worker_rfr, 
                                 initargs=(rfr_series,)) as executor:
            
            futures = {executor.submit(self.process_symbol_task_entry, sym): sym for sym in symbols}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
                sym = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Stock {sym} crashed: {e}", exc_info=True)
    
        logger.info("All processed.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass 

    calculator = OptionIVCalculator(
        db_path="/home/kingfang007/notebook/stocks.db",
        option_root="/mnt/s990/data/massive_options_1m_formatted", # 指向新下载的高精度文件
        data_root="/home/kingfang007/train_data/spnq_train_resampled",
        iv_option_root="/home/kingfang007/train_data/quote_options_day_iv" 
    )
    
    calculator.run(max_concurrent_stocks=12)