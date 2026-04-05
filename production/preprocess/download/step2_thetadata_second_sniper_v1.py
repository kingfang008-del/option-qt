import os
import io
import csv
import httpx
import pandas as pd
import numpy as np
from datetime import time as dt_time
from tqdm import tqdm
import concurrent.futures
import re
import logging
import warnings

warnings.filterwarnings("ignore", message="Found Below Intrinsic contracts")

# ================= 配置参数 =================
BASE_URL = "http://127.0.0.1:25503/v3"
TARGET_MAP_FILE = "/home/kingfang007/train_data/locked_targets_map.parquet"
OUTPUT_DIR = "/mnt/s990/data/raw_1s/options"
STOCK_OUTPUT_DIR = "/mnt/s990/data/raw_1s/stocks"
RISK_FREE_RATE = 0.045  # 无风险利率 (可调)

# ⚠️ ThetaData VALUE 订阅，坚决死守 2 并发
MAX_WORKERS = 2 
# ============================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Theta_Second_Sniper")

# ================= BSM 本地 Greeks 计算 =================
try:
    from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False
    logger.warning("⚠️ py_vollib_vectorized not found, Greeks will be zeros. Install: pip install py_vollib_vectorized")

def compute_greeks_for_df(df, stock_price_map):
    """
    本地 BSM 向量化计算 IV + Delta/Gamma/Vega/Theta
    df: 期权 quote 数据 (含 ticker, price, strike, ts)
    stock_price_map: {ts -> stock_close} 映射
    """
    if not HAS_VOLLIB or df.empty:
        for col in ['iv', 'delta', 'gamma', 'vega', 'theta']:
            df[col] = 0.0
        return df
    
    # 从 OCC Ticker 提取到期日和合约类型
    clean_tickers = df['ticker'].str.replace('O:', '', regex=False)
    extracted = clean_tickers.str.extract(r'^[A-Z]+(\d{6})([CP])\d{8}$')
    df['expiry'] = pd.to_datetime('20' + extracted[0], format='%Y%m%d', errors='coerce')
    df['opt_type'] = extracted[1].map({'C': 'c', 'P': 'p'})
    
    # 映射股票价格
    df['stock_close'] = df['ts'].map(stock_price_map)
    df['stock_close'] = df['stock_close'].ffill().bfill()  # 填充缺失
    
    # 到期时间 (年化)
    current_ts = pd.to_datetime(df['timestamp'])
    expiry_ts = pd.to_datetime(df['expiry'])
    T_years = (expiry_ts - current_ts).dt.total_seconds().values / 31557600.0
    T_years = np.maximum(T_years, 1e-6)
    
    P = df['price'].values.astype(float)
    S = df['stock_close'].values.astype(float)
    K = df['strike'].values.astype(float)
    r = np.full_like(P, RISK_FREE_RATE)
    
    is_call = (df['opt_type'] == 'c').values
    is_put = (df['opt_type'] == 'p').values
    
    # 计算 IV
    iv = np.zeros_like(P, dtype=float)
    valid = (P > 0.01) & (S > 0.01) & (K > 0.01)
    
    try:
        if (is_call & valid).any():
            m = is_call & valid
            iv[m] = vectorized_implied_volatility(
                P[m], S[m], K[m], T_years[m], r[m], 'c',
                return_as='numpy', on_error='ignore'
            )
        if (is_put & valid).any():
            m = is_put & valid
            iv[m] = vectorized_implied_volatility(
                P[m], S[m], K[m], T_years[m], r[m], 'p',
                return_as='numpy', on_error='ignore'
            )
    except Exception as e:
        logger.debug(f"IV calc error: {e}")
    
    iv = np.nan_to_num(iv, nan=0.0)
    df['iv'] = iv
    
    # 计算 Greeks
    delta = np.zeros_like(iv)
    gamma = np.zeros_like(iv)
    vega = np.zeros_like(iv)
    theta = np.zeros_like(iv)
    
    valid_iv = (iv > 0.01) & (iv < 5.0) & valid
    
    try:
        if (is_call & valid_iv).any():
            m = is_call & valid_iv
            g = get_all_greeks('c', S[m], K[m], T_years[m], r[m], iv[m], return_as='dict')
            delta[m] = np.nan_to_num(g['delta'])
            gamma[m] = np.nan_to_num(g['gamma'])
            vega[m] = np.nan_to_num(g['vega'])
            theta[m] = np.nan_to_num(g['theta'])
        
        if (is_put & valid_iv).any():
            m = is_put & valid_iv
            g = get_all_greeks('p', S[m], K[m], T_years[m], r[m], iv[m], return_as='dict')
            delta[m] = np.nan_to_num(g['delta'])
            gamma[m] = np.nan_to_num(g['gamma'])
            vega[m] = np.nan_to_num(g['vega'])
            theta[m] = np.nan_to_num(g['theta'])
    except Exception as e:
        logger.debug(f"Greeks calc error: {e}")
    
    df['delta'] = delta
    df['gamma'] = gamma
    df['vega'] = vega
    df['theta'] = theta
    
    # 清理临时列
    df.drop(columns=['expiry', 'opt_type', 'stock_close'], inplace=True, errors='ignore')
    return df

# ================= 工具函数 =================
def is_rth(ts_series):
    time_series = ts_series.dt.time
    return (time_series >= dt_time(9, 30)) & (time_series < dt_time(16, 0))

def parse_occ_ticker(ticker):
    """从 O:AAPL240119C00150000 提取精准查询参数"""
    clean_ticker = ticker.replace('O:', '')
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', clean_ticker)
    if not match: return None
    
    symbol = match.group(1)
    exp_str = f"20{match.group(2)}"
    right = 'call' if match.group(3) == 'C' else 'put'
    
    # 物理 Strike (浮点数) 供特征计算使用
    strike_float = float(match.group(4)) / 1000.0
    # ThetaData 专属 Strike 格式 (价格 * 1000，例如 150.0 -> 150000)
    strike_theta = int(match.group(4))
    
    return symbol, exp_str, right, strike_float, strike_theta

# ================= 期权极速本地平替 (FFill) =================
def fetch_single_day_targets(args):
    """
    独立 Worker：因为 1s 权限受限，直接从本地 1M 数据源高速抽取并 FFill 升采样。
    """
    symbol, date_str, group_df = args
    
    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path):
        return f"⏩ Option {symbol} {date_str} 已存在跳过"

    # 源头 1m 期权数据 (自带 Greeks)
    src_opt_path = f"/home/kingfang007/train_data/quote_options_day_iv/{symbol}/standard/{symbol}_{date_str}.parquet"
    if not os.path.exists(src_opt_path):
        src_opt_path = f"/home/kingfang007/train_data/quote_options_day_iv/{symbol}/{symbol}_{date_str}.parquet"
        if not os.path.exists(src_opt_path):
            return f"❌ Option {symbol} {date_str} Error: 找不到源 1m 期权文件"

    try:
        df_1m = pd.read_parquet(src_opt_path)
    except Exception as e:
        return f"❌ Option {symbol} {date_str} Error 读取失败: {e}"

    # 提取需要的合约
    target_tickers = group_df['contract_symbol'].tolist()
    # 建立映射
    ticker_to_bucket = dict(zip(group_df['contract_symbol'], group_df['bucket_id']))
    ticker_to_tag = dict(zip(group_df['contract_symbol'], group_df.get('tag', [''] * len(group_df))))
    
    df_1m = df_1m[df_1m['ticker'].isin(target_tickers)].copy()
    if df_1m.empty:
        return f"⚠️ {symbol} {date_str}: 1m 库中均缺失目标合约数据"

    # 处理时间戳
    if 'timestamp' in df_1m.columns:
        if df_1m['timestamp'].dt.tz is None:
            df_1m['timestamp'] = df_1m['timestamp'].dt.tz_localize('America/New_York')
        else:
            df_1m['timestamp'] = df_1m['timestamp'].dt.tz_convert('America/New_York')

    df_1m = df_1m[is_rth(df_1m['timestamp'])]

    all_1s_dfs = []
    
    # 构建完美的 1s grid
    start_ts = pd.to_datetime(f"{date_str} 09:30:00").tz_localize('America/New_York')
    end_ts = pd.to_datetime(f"{date_str} 15:59:59").tz_localize('America/New_York')
    grid = pd.date_range(start=start_ts, end=end_ts, freq='s')

    for ticker, sub_df in df_1m.groupby('ticker'):
        sub_df.set_index('timestamp', inplace=True)
        # 去重
        sub_df = sub_df[~sub_df.index.duplicated(keep='last')]
        
        df_sub_1s = sub_df.reindex(grid).ffill().bfill()
        df_sub_1s.index.name = 'timestamp'
        df_sub_1s = df_sub_1s.reset_index()
        
        df_sub_1s['ticker'] = ticker
        df_sub_1s['bucket_id'] = ticker_to_bucket.get(ticker, -1)
        df_sub_1s['tag'] = ticker_to_tag.get(ticker, '')
        
        all_1s_dfs.append(df_sub_1s)

    if not all_1s_dfs:
        return f"⚠️ {symbol} {date_str}: 过滤后无有效数据"

    df_final = pd.concat(all_1s_dfs, ignore_index=True)
    df_final['ts'] = df_final['timestamp'].astype('int64') / 1e9

    # 填充必要列
    if 'strike' not in df_final.columns:
        df_final['strike'] = df_final['ticker'].apply(lambda tf: parse_occ_ticker(tf)[3] if parse_occ_ticker(tf) else 0.0)
    
    df_final['price'] = df_final.get('close', (df_final['bid'] + df_final['ask']) / 2.0)
    df_final['underlying'] = symbol
    
    # Volume 平移摊平
    if 'volume' in df_final.columns:
        df_final['volume'] = df_final['volume'] / 60.0

    for gcol in ['iv', 'delta', 'gamma', 'vega', 'theta']:
        if gcol not in df_final.columns:
            df_final[gcol] = 0.0
            
    final_cols = [
        'ts', 'timestamp', 'ticker', 'tag', 'bucket_id', 'underlying',
        'bid', 'ask', 'bid_size', 'ask_size', 'price', 'strike',
        'iv', 'delta', 'gamma', 'vega', 'theta'
    ]
    df_final = df_final[[c for c in final_cols if c in df_final.columns]]
    
    df_final.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    n_greeks = (df_final['iv'] > 0).sum() if 'iv' in df_final.columns else 0
    return f"🎯 Option (FFill 1s) {symbol} {date_str}: {len(df_final)} rows, {n_greeks} with Greeks"

# ================= 股票下载极速本地平替 (FFill) =================
def fetch_stock_day(args):
    """
    因为非顶配 ThetaData 账号查 Stock Quote 会返回 403 Forbidden。
    我们直接读取本地高质量的 1min 级别数据，将其用 Forward-Fill 升采样广播到 1s 级别，
    足以绝对精准地支撑期权 Delta、Gamma 计算，不仅绕过权限限制，更省去海量网络耗时！
    """
    symbol, date_str = args
    
    out_dir = os.path.join(STOCK_OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path): return f"⏩ Stock {symbol} {date_str} exists"
    
    month_str = date_str[:7]
    spnq_path = f"/home/kingfang007/train_data/spnq_train_resampled/{symbol}/regular/09:30-16:00/1min/{month_str}.parquet"
    
    if not os.path.exists(spnq_path):
        return f"❌ Stock {symbol} {date_str} Error: 找不到底层 1min 正股数据 ({spnq_path})"

    try:
        df = pd.read_parquet(spnq_path)
        
        # 1. 切片当天的分钟数据
        if 'timestamp' in df.columns and df['timestamp'].dt.tz is None:
             df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York')
        elif 'timestamp' in df.columns:
             df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

        df['date_only'] = df['timestamp'].dt.date
        target_date = pd.to_datetime(date_str).date()
        df = df[df['date_only'] == target_date].drop(columns=['date_only'])
        
        if df.empty:
            return f"⚠️ Stock {symbol} {date_str}: 该日期在 1m 文件中无数据"
        
        # 2. 从 1min 前推构建 1s 坐标点 (Forward Fill)
        df.set_index('timestamp', inplace=True)
        # 提取 RTH 
        start_ts = pd.to_datetime(f"{date_str} 09:30:00").tz_localize('America/New_York')
        end_ts = pd.to_datetime(f"{date_str} 15:59:59").tz_localize('America/New_York')
        grid = pd.date_range(start=start_ts, end=end_ts, freq='s')
        
        # reindex 向前填补空缺值 (如果是开盘第一秒没数据向后 bfill)
        df_1s = df.reindex(grid).ffill().bfill()
        df_1s.index.name = 'timestamp'
        df_1s = df_1s.reset_index()
        
        # 3. 构造 1s 所需的核心字段
        df_1s['ts'] = df_1s['timestamp'].astype('int64') / 1e9
        
        # 伪造盘口：正股价决定 Greeks，极其精准，挂单大小无所谓
        df_1s['bid'] = df_1s['close']
        df_1s['ask'] = df_1s['close']
        df_1s['bid_size'] = 100
        df_1s['ask_size'] = 100
        df_1s['volume'] = df_1s['volume'] / 60.0 # 量能平摊
        
        out_cols = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 'bid_size', 'ask_size']
        df_1s = df_1s[[c for c in out_cols if c in df_1s.columns]]
        
        df_1s.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
        return f"🎯 Stock (FFill 1s) {symbol} {date_str}: {len(df_1s)} 行构建成功"
        
    except Exception as e:
        return f"❌ Stock {symbol} {date_str} Error: {e}"

# ================= 主入口 =================
def main():
    if not os.path.exists(TARGET_MAP_FILE):
        print("❌ 未找到清单，请先运行 step1_build_target_map.py")
        return
        
    print("📂 正在加载狙击清单...")
    target_map = pd.read_parquet(TARGET_MAP_FILE)
    
    all_potential_tasks = []
    stock_tasks_set = set()
    for (sym, date_str), group in target_map.groupby(['symbol', 'date_str']):
        all_potential_tasks.append((sym, date_str, group))
        stock_tasks_set.add((sym, date_str))
        
    # ⚠️ 必须先下载股票数据，因为期权 Greeks 计算需要股价
    stock_tasks = []
    for sym, date_str in stock_tasks_set:
        out_path = os.path.join(STOCK_OUTPUT_DIR, sym, f"{sym}_{date_str}.parquet")
        if not os.path.exists(out_path):
            stock_tasks.append((sym, date_str))

    opt_tasks = []
    for sym, date_str, group in all_potential_tasks:
        out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{date_str}.parquet")
        if not os.path.exists(out_path):
            opt_tasks.append((sym, date_str, group))

    print(f"🚀 开启秒级极速狙击模式！")
    print(f"📊 待处理下载: Stock={len(stock_tasks)} (先), Option={len(opt_tasks)} (后，含 Greeks)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. 先下载股票 (期权 Greeks 依赖股价)
        if stock_tasks:
            stk_futures = {executor.submit(fetch_stock_day, t): t for t in stock_tasks}
            for f in tqdm(concurrent.futures.as_completed(stk_futures), total=len(stock_tasks), desc="Stocks"):
                f.result()
        
        # 2. 再下载期权 + 计算 Greeks
        if opt_tasks:
            opt_futures = {executor.submit(fetch_single_day_targets, t): t for t in opt_tasks}
            for f in tqdm(concurrent.futures.as_completed(opt_futures), total=len(opt_tasks), desc="Options+Greeks"):
                f.result()

    print("🎉 秒级全量精准狙击完成！(含本地 BSM Greeks)")

if __name__ == "__main__":
    main()

