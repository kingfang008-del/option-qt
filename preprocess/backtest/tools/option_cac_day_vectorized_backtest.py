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
    S_valid = S[valid_mask]
    t_valid = t[valid_mask]
    r_valid = r[valid_mask]
    sigma_valid = sigma[valid_mask]
    sigma_sqrt_t = sigma_valid * np.sqrt(t_valid)
    d1_valid = (np.log(S_valid / K) + (r_valid + 0.5 * sigma_valid**2) * t_valid) / sigma_sqrt_t
    d2_valid = d1_valid - sigma_sqrt_t
    d1[valid_mask] = d1_valid
    d2[valid_mask] = d2_valid
    return d1, d2

def vectorized_vanna(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    vanna = np.where(sigma > 1e-9, -phi_d1 * (d2_val / sigma), 0)
    return vanna

def vectorized_charm(S, K, t, r, sigma):
    d1, d2_val = get_d1_d2(S, K, t, r, sigma)
    phi_d1 = norm.pdf(d1)
    sigma_sqrt_t = sigma * np.sqrt(t)
    term1 = np.where(sigma_sqrt_t > 1e-9, r / sigma_sqrt_t, 0)
    term2 = np.where(t > 1e-9, d2_val / (2 * t), 0)
    charm = -phi_d1 * (term1 - term2)
    return charm

def calculate_robust_atm_iv(df_slice: pd.DataFrame, stock_price: float) -> float:
    if df_slice.empty or pd.isna(stock_price) or stock_price <= 0: return np.nan
    valid_options = df_slice[df_slice['volume'] > 0].copy()
    if valid_options.empty: return np.nan
    
    valid_options['distance'] = abs(np.log(valid_options['strike_price'] / stock_price))
    top_n = valid_options.nsmallest(4, 'distance')
    if top_n.empty: return np.nan
    
    weights = 1.0 / (top_n['distance'] + 1e-6)
    return (top_n['iv'] * (weights / weights.sum())).sum()

def _calculate_features_on_slice(df_slice: pd.DataFrame) -> pd.Series:
    p = {"skew_moneyness_put": 0.95, "skew_moneyness_call": 1.05, "liquidity_min_volume": 10}
    result_cols = ['options_struc_atm_iv', 'options_struc_skew', 'options_pcr_volume', 'options_struc_term']
    
    liquid_df = df_slice[df_slice['volume'] >= p['liquidity_min_volume']].copy()
    if liquid_df.empty: return pd.Series(dtype=np.float64, index=result_cols)

    stock_price = liquid_df['stock_close'].iloc[0]
    
    # 获取当前时间
    current_timestamp = df_slice.name if hasattr(df_slice, 'name') else df_slice.index[0]
    if not isinstance(current_timestamp, pd.Timestamp): 
        current_timestamp = pd.to_datetime(current_timestamp)
    if current_timestamp.tz is None: 
        current_timestamp = current_timestamp.tz_localize('America/New_York')
    else: 
        current_timestamp = current_timestamp.tz_convert('America/New_York')

    if pd.isna(stock_price) or stock_price <= 0: 
        return pd.Series(dtype=np.float64, index=result_cols)

    atm_iv = calculate_robust_atm_iv(liquid_df, stock_price)
    
    # IV Skew
    put_target = stock_price * p['skew_moneyness_put']
    call_target = stock_price * p['skew_moneyness_call']
    puts = liquid_df[liquid_df['contract_type'] == 'p']
    calls = liquid_df[liquid_df['contract_type'] == 'c']
    
    closest_put_iv = np.nan
    closest_call_iv = np.nan
    
    if not puts.empty:
        idx_p = (puts['strike_price'] - put_target).abs().idxmin()
        closest_put_iv = puts.loc[idx_p, 'iv']
        if isinstance(closest_put_iv, pd.Series): closest_put_iv = closest_put_iv.iloc[0]
        
    if not calls.empty:
        idx_c = (calls['strike_price'] - call_target).abs().idxmin()
        closest_call_iv = calls.loc[idx_c, 'iv']
        if isinstance(closest_call_iv, pd.Series): closest_call_iv = closest_call_iv.iloc[0]

    iv_skew = np.nan
    if pd.notna(closest_put_iv) and pd.notna(closest_call_iv) and closest_call_iv > 0:
        iv_skew = closest_put_iv / closest_call_iv

    pcr = puts['volume'].sum() / (calls['volume'].sum() + 1e-6)

    # 距到期日天数
    # 由于我们把 expiration_date 当 tz-naive 处理了，在减法的时候需要同一时区
    expiry_series = pd.to_datetime(liquid_df['expiration_date'])
    if expiry_series.dt.tz is None:
        expiry_series = expiry_series.dt.tz_localize('America/New_York')
        
    liquid_df['expiry_days'] = (expiry_series - current_timestamp).dt.days
    
    near_expiry = liquid_df['expiry_days'].min()
    far_exp_df = liquid_df[liquid_df['expiry_days'] > near_expiry]
    far_expiry = far_exp_df['expiry_days'].min() if not far_exp_df.empty else np.nan
    
    near_iv = liquid_df[liquid_df['expiry_days'] == near_expiry]['iv'].mean() if pd.notna(near_expiry) else np.nan
    far_iv = liquid_df[liquid_df['expiry_days'] == far_expiry]['iv'].mean() if pd.notna(far_expiry) else near_iv
    
    term_structure = (near_iv - far_iv) * 100 if pd.notna(near_iv) and pd.notna(far_iv) else np.nan

    return pd.Series({
        'options_struc_atm_iv': atm_iv, 
        'options_struc_skew': iv_skew, 
        'options_pcr_volume': pcr, 
        'options_struc_term': term_structure
    })

# ==============================================================================
# 核心计算逻辑 (按日处理)
# ==============================================================================
def compute_single_day_file(args):
    """
    处理某个 Symbol 下单个日期的期权 Parquet 文件。
    输入文件包含该日该标的下的 *所有* 合约和所有分钟。
    """
    parquet_path, symbol, underlying_df, iv_dir = args
    filename = os.path.basename(parquet_path) # e.g., AAPL_2026-01-01.parquet
    day_str = filename.replace('.parquet', '').split('_')[-1] # "2026-01-01"
    
    iv_filename = os.path.join(iv_dir, filename)
    high_feat_filename = os.path.join(iv_dir, f"{symbol}_{day_str}_high_features.parquet")
    
    if os.path.exists(iv_filename) and os.path.exists(high_feat_filename):
        return None # 已经处理过

    try:
        # 1. 读期权当天数据
        df_opt_raw = pd.read_parquet(parquet_path)
        if df_opt_raw.empty: return None
        
        # S3 列名重命名统一处理 ('o', 'h', 'l', 'c', 'v') 已经在 S3 pipeline 完成
        # 这里重写 timestamp，并在原始文件中解析合约属性
        df_opt = df_opt_raw.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_opt['timestamp']):
            df_opt['timestamp'] = pd.to_datetime(df_opt['timestamp'])
            
        # 设置索引用于 merge_asof
        df_opt = df_opt.sort_values('timestamp').set_index('timestamp')
        
        # 2. 从 ticker 批量提取合约属性 (Vectorized Regex)
        # S3 ticker 格式: O:AAPL240119C00150000 -> 剔除 O: 后: AAPL240119C00150000
        # 提取 YYMMDD (日期), C/P (类型), 00150000 (行权价, 除以1000)
        clean_tickers = df_opt['ticker'].str.replace('O:', '', regex=False)
        extracted = clean_tickers.str.extract(r'^[A-Z]+(\d{6})([CP])(\d{8})$')
        df_opt['expiration_date'] = pd.to_datetime('20' + extracted[0], format='%Y%m%d')
        df_opt['contract_type'] = extracted[1].str.lower()
        df_opt['strike_price'] = extracted[2].astype(float) / 1000.0

        # 删除解析失败的行
        df_opt.dropna(subset=['expiration_date', 'contract_type', 'strike_price'], inplace=True)
        if df_opt.empty: return None

        # 3. Time Series 合并底层股票数据
        # 提前切片出那天的底层股票数据，提升 merge 速度
        target_date = pd.to_datetime(day_str).tz_localize('America/New_York')
        day_start = target_date.replace(hour=0, minute=0, second=0)
        day_end = target_date.replace(hour=23, minute=59, second=59)
        
        u_df_day = underlying_df.loc[day_start:day_end]
        if u_df_day.empty: 
            # 如果当天没有任何底层股票，无法计算IV
            return None

        # Asof Merge
        df_merged = pd.merge_asof(
            df_opt,
            u_df_day[['close']].rename(columns={'close': 'stock_close'}),
            left_index=True, right_index=True, direction='backward'
        )
        df_merged.dropna(subset=['stock_close'], inplace=True)
        if df_merged.empty: return None

        # 期权除权(Split Adjust Options)
        # 解释：因为传入的底层股票(stock_close)是已经经过前复权的(例如NVDA 2023年只有30刀)。
        # 而读取的期权 parquet 是历史真实数据(strike可能为340)。
        # 为保证计算匹配且输出到模型的标度一致，我们需要把期权的价格和行权价等比例除权，成交量等比例乘权。
        if symbol in KNOWN_SPLITS:
            restore_factor = np.ones(len(df_merged))
            splits = sorted(KNOWN_SPLITS[symbol], key=lambda x: x[0])
            for split_str, ratio in splits:
                split_date = pd.Timestamp(split_str).tz_localize('America/New_York')
                mask = df_merged.index < split_date
                restore_factor[mask] /= ratio
            
            # 对期权自身数据向下复权，与现行股票数据标度对齐
            # 兼容 S3 原始单字符列名 (o, h, l, c) 和 标准列名 (open, high, low, close)
            for col in ['strike_price', 'open', 'high', 'low', 'close', 'o', 'h', 'l', 'c']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col] / restore_factor
            
            # 股本膨胀后，历史真实成交量也同比例放大，以保证聚合计算（如PCR）的权重连贯性
            for col in ['volume', 'v']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col] * restore_factor

        # 4. 向量化运算 Greeks & IV (对全天数十万行矩阵一波带走)
        P = df_merged['c'].values
        S = df_merged['stock_close'].values
        K = df_merged['strike_price'].values
        
        if df_merged.index.tz is None:
            current_ts_idx = df_merged.index.tz_localize('America/New_York')
        else:
            current_ts_idx = df_merged.index.tz_convert('America/New_York')
            
        expiry_ts = pd.to_datetime(df_merged['expiration_date'])
        if expiry_ts.dt.tz is None:
            expiry_ts = expiry_ts.dt.tz_localize('America/New_York')

        time_diff = expiry_ts - current_ts_idx
        T_years = time_diff.dt.total_seconds().values / 31557600.0
        T_years = np.maximum(T_years, 1e-6)
        
        # 无风险利率对齐
        search_keys = current_ts_idx.normalize().tz_localize(None)
        r_idx = G_RFR_SERIES.index.searchsorted(search_keys)
        r_idx = np.clip(r_idx, 0, len(G_RFR_SERIES) - 1)
        r = G_RFR_SERIES.values[r_idx]
        r = np.nan_to_num(r, nan=0.04)

        # Py-vollib vectorized 目前只支持单一 string 类型传入计算, 我们做两次屏蔽处理
        is_call = (df_merged['contract_type'] == 'c').values
        is_put = (df_merged['contract_type'] == 'p').values

        # 调用向量化计算引擎 (将 calls 和 puts 分别计算后缝合)
        iv = np.zeros_like(P, dtype=float)
        
        if is_call.any():
            iv[is_call] = vectorized_implied_volatility(
                P[is_call], S[is_call], K[is_call], T_years[is_call], r[is_call], 'c', return_as='numpy', on_error='ignore'
            )
        if is_put.any():
            iv[is_put] = vectorized_implied_volatility(
                P[is_put], S[is_put], K[is_put], T_years[is_put], r[is_put], 'p', return_as='numpy', on_error='ignore'
            )

        # 筛除无效及异常值
        valid_mask = (iv > 0) & (iv < 5.0) & (~np.isnan(iv))
        if not np.any(valid_mask): return None
        
        df_valid = df_merged.iloc[valid_mask].copy()
        S, K, T, r, iv = S[valid_mask], K[valid_mask], T_years[valid_mask], r[valid_mask], iv[valid_mask]
        
        df_valid['iv'] = iv
        
        # 计算 Greeks
        is_call_valid = (df_valid['contract_type'] == 'c').values
        is_put_valid = (df_valid['contract_type'] == 'p').values

        delta = np.zeros_like(iv)
        gamma = np.zeros_like(iv)
        vega = np.zeros_like(iv)
        theta = np.zeros_like(iv)
        rho = np.zeros_like(iv)

        if is_call_valid.any():
            gd_c = get_all_greeks('c', S[is_call_valid], K[is_call_valid], T[is_call_valid], r[is_call_valid], iv[is_call_valid], return_as='dict')
            delta[is_call_valid] = gd_c['delta']
            gamma[is_call_valid] = gd_c['gamma']
            vega[is_call_valid] = gd_c['vega']
            theta[is_call_valid] = gd_c['theta']
            rho[is_call_valid] = gd_c['rho']

        if is_put_valid.any():
            gd_p = get_all_greeks('p', S[is_put_valid], K[is_put_valid], T[is_put_valid], r[is_put_valid], iv[is_put_valid], return_as='dict')
            delta[is_put_valid] = gd_p['delta']
            gamma[is_put_valid] = gd_p['gamma']
            vega[is_put_valid] = gd_p['vega']
            theta[is_put_valid] = gd_p['theta']
            rho[is_put_valid] = gd_p['rho']

        df_valid['delta'] = delta
        df_valid['gamma'] = gamma
        df_valid['vega'] = vega
        df_valid['theta'] = theta
        df_valid['rho'] = rho
        df_valid['vanna'] = vectorized_vanna(S, K, T, r, iv)
        df_valid['charm'] = vectorized_charm(S, K, T, r, iv)
        
        df_valid.rename(columns={'v':'volume', 'o':'open', 'h':'high', 'l':'low'}, inplace=True)
        final_cols = ['ticker', 'expiration_date', 'contract_type', 'strike_price', 
                      'open', 'high', 'low', 'c', 'volume', 'iv', 
                      'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'charm', 'stock_close']
        
        df_res = df_valid[final_cols].copy()
        
        # =================================================================
        # 5. 生成按日的聚合高阶特征 (High Features)
        # =================================================================
        high_features = df_res.groupby(level=0).apply(_calculate_features_on_slice)
        
        # 换名 'c' to 'close' 向下兼容
        df_res.rename(columns={'c':'close'}, inplace=True)
        
        # 6. 保存到磁盘
        df_res.reset_index().to_parquet(iv_filename, index=False)
        high_features.reset_index().to_parquet(high_feat_filename, index=False)
        
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
        self.data_root = data_root  # Underlying stock directory
        self.option_root = option_root # The raw S3 option directory (PROCESSED_DIR)
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
                rfr_df = rfr_df.resample('D').ffill().fillna(0.04)
                rfr_df.to_parquet(self.risk_free_cache_file)
            except Exception: pass
            
        if rfr_df.empty:
             idx = pd.date_range(fetch_start, fetch_end, freq='D').normalize()
             rfr_df = pd.DataFrame(index=idx, data={'DGS3MO': 0.04})

        self.risk_free_cache = rfr_df
        return self.risk_free_cache

    def get_target_symbols(self) -> list[str]:
        # 从本地股票池获取我们需要的标的
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
            
            # S3 底层数据如果有问题在这里修复
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
        
        # 1. 拿底层股票数据 (用来合并计算 IV 的基石)
        underlying_df = self._get_underlying_df(symbol)
        if underlying_df is None: 
            logger.warning(f"No underlying data for {symbol}, Skipping.")
            return

        # 2. 扫描该 Symbol 下的 S3 期权下载文件 e.g: AAPL_2026-01-01.parquet
        opt_dir = os.path.join(self.option_root, symbol)
        if not os.path.exists(opt_dir):
            return
            
        parquet_files = glob.glob(os.path.join(opt_dir, f"{symbol}_*.parquet"))
        if not parquet_files:
            return
            
        iv_dir = os.path.join(self.iv_option_root, symbol)
        os.makedirs(iv_dir, exist_ok=True)
        
        day_tasks = []
        for fp in parquet_files:
            day_tasks.append((fp, symbol, underlying_df, iv_dir))

        # 3. 按日级别并发计算
        success_count = 0
        with ThreadPoolExecutor(max_workers=20) as executor:
            for day_str in executor.map(compute_single_day_file, day_tasks):
                if day_str: success_count += 1
                
        # 清理内存
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

    # 修改此处路径配置，对齐你的 S3 下载数据
    calculator = OptionIVCalculator(
        db_path="/home/kingfang007/notebook/stocks.db",
        # option_root 是 S3 清洗出来的路径，包含按天存的 [AAPL_2026-01-01.parquet] 等文件
        option_root="/home/kingfang007/backtest/new_option_data_s3_202603", 
        data_root="/home/kingfang007/backtest/spnq_2603",
        # 结果输出路径
        iv_option_root="/home/kingfang007/backtest/nq_2603_options_day_iv"
    )
    
    # 既然已经按天打包极大优化了读写瓶颈，我们可以直接提高计算股票并发度
    calculator.run(max_concurrent_stocks=12)

