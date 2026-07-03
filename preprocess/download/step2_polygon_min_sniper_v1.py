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
OUTPUT_DIR = "/mnt/s990/data/massive_options_1m"

# Polygon 并发限制
MAX_WORKERS = 100 
FORCE_RECOMPUTE = False 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger("Polygon_1m_Sniper")
eastern = timezone('America/New_York')
# ============================================


# ================= 核心 Worker =================
# ================= 核心 Worker =================
def process_single_day_polygon_min(args):
    symbol, date_str, group_df = args
    client = RESTClient(API_KEY)
    
    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path) and not FORCE_RECOMPUTE:
        return f"⏩ {symbol} {date_str} exists"

    all_option_1m = []

    # 获取期权行情 (使用 list_quotes 方法并重采样至 1m)
    for _, row in group_df.iterrows():
        occ = row['contract_symbol']
        poly_ticker = occ if occ.startswith("O:") else f"O:{occ}"
        
        try:
            # 下载原始报价数据并重采样
            quotes = list(client.list_quotes(ticker=poly_ticker, timestamp_gte=date_str, timestamp_lte=date_str, limit=50000))
            if not quotes: continue
        
            df = pd.DataFrame([{
                'timestamp': getattr(q, 'sip_timestamp', getattr(q, 'participant_timestamp', 0)),
                'bid': getattr(q, 'bid_price', 0.0),
                'ask': getattr(q, 'ask_price', 0.0),
                'bid_size': getattr(q, 'bid_size', 0.0),
                'ask_size': getattr(q, 'ask_size', 0.0)
            } for q in quotes])
            
            # 过滤：至少有一边有报价，且卖价不低于买价（除非买价为0）
            df = df[(df['bid'] > 0) | (df['ask'] > 0)].copy()
            # 如果买价卖价都存在，取中价；如果有一边为0，取另一边非零值，防止价格被减半
            df['mid_price'] = np.where(
                (df['bid'] > 0) & (df['ask'] > 0), 
                (df['bid'] + df['ask']) / 2.0, 
                df[['bid', 'ask']].max(axis=1)
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert(eastern)
            
            # RTH 过滤
            time_series = df['timestamp'].dt.time
            df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
            if df.empty: continue
            
            # 转为 1min 坐标点 (保留每分钟最后一笔报价作为快照)
            df['ts_1min'] = df['timestamp'].dt.floor('1min')
            df = df.sort_values('timestamp').drop_duplicates(subset=['ts_1min'], keep='last').copy()
            
            df['timestamp'] = df['ts_1min']
            df['ts'] = df['timestamp'].astype('int64') / 1e9
            df['ticker'] = occ.replace('O:', '')
            df['bucket_id'] = row['bucket_id']
            df['tag'] = row.get('tag', '')
            df['underlying'] = symbol
            
            # 提取行权价
            match = re.search(r'[CP](\d{8})$', df['ticker'].iloc[0])
            df['strike'] = float(match.group(1)) / 1000.0 if match else 0.0
            
            all_option_1m.append(df)
        except Exception as e:
            logger.error(f"  [Option] Contract {occ} process error: {e}")
            continue

    if not all_option_1m:
        return f"⚠️ {symbol} {date_str}: No valid option data."

    final_df = pd.concat(all_option_1m, ignore_index=True)
    
    final_cols = [
        'ts', 'timestamp', 'ticker', 'tag', 'bucket_id', 'underlying',
        'bid', 'ask', 'bid_size', 'ask_size', 'mid_price', 'strike'
    ]
    final_df = final_df[[c for c in final_cols if c in final_df.columns]]
    final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    
    return f"🎯 {symbol} {date_str}: Success! {len(final_df)} rows of 1m sniper data."

def main():
    if not os.path.exists(TARGET_MAP_FILE):
        logger.error("❌ Target map not found.")
        return
        
    target_map = pd.read_parquet(TARGET_MAP_FILE)
    
    symbols = target_map['symbol'].unique()
    logger.info(f"🚀 Starting Polygon 1m sniper for {len(symbols)} symbols...")

    for sym in symbols:
        sym_df = target_map[target_map['symbol'] == sym]
        
        date_tasks = []
        for d, g in sym_df.groupby('date_str'):
            out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{d}.parquet")
            if os.path.exists(out_path) and not FORCE_RECOMPUTE:
                continue
            date_tasks.append((sym, d, g))
        
        if not date_tasks:
            logger.info(f"⏩ Symbol [{sym}] is already fully processed.")
            continue
            
        logger.info(f"📥 Processing symbol [{sym}] for {len(date_tasks)} remaining days...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(tqdm(executor.map(process_single_day_polygon_min, date_tasks), 
                          total=len(date_tasks), desc=f"Symbol {sym}"))
            for res in results:
                if "🎯" not in res and "⏩" not in res:
                    logger.warning(res)
        
        logger.info(f"💤 Symbol {sym} done, cooling down...")

    logger.info("🏁 All symbols processed.")

if __name__ == "__main__":
    main()
