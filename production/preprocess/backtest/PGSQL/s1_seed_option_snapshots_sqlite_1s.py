#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import warnings
import gc
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pytz
import argparse

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= 配置区域 =================
PROJECT_ROOT = Path.home() / "quant_project"
DB_DIR       = PROJECT_ROOT / "data" / "history_sqlite_1s"  # 建立专属的 1s 数据库目录防止冲突

# 1s 数据源路径
OPT_RAW_DIR  = Path("/mnt/s990/data/stress_test_1s_greeks")
STOCK_RAW_DIR= Path("/mnt/s990/data/raw_1s/stocks")

NY_TZ = pytz.timezone('America/New_York')

try:
    from config import TARGET_SYMBOLS 
except ImportError:
    # 如果找不到 config，提供一个默认列表供测试
    TARGET_SYMBOLS = ['NVDA', 'AAPL', 'TSLA', 'SPY', 'QQQ']

# ================= 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Option Snapshot Seeder (1s High-Frequency Edition)")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="只处理该日期及之后的数据")
    parser.add_argument("--end_date", type=str, default="2099-12-31", help="只处理该日期及之前的数据")
    return parser.parse_args()

args = parse_args()

START_DATE = pd.to_datetime(args.start_date).date()
END_DATE   = pd.to_datetime(args.end_date).date()

# ================= 数据库与检查逻辑 =================
def init_full_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    # 设置更大的缓存应对高频数据
    conn.execute("PRAGMA cache_size = -64000") 
    c = conn.cursor()
    
    # 秒级正股表
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_bars_1s (
            symbol TEXT, ts INTEGER, datetime_ny TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_mb_1s_ts ON market_bars_1s (ts)")
    
    # 周期级正股表 (1m/5m)
    for res in ['1m', '5m']:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS market_bars_{res} (
                symbol TEXT, ts INTEGER, datetime_ny TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, ts)
            )
        """)
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_mb_{res}_ts ON market_bars_{res} (ts)")

    # 期权表 (1s/1m/5m)
    for res in ['1s', '1m', '5m']:
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS option_snapshots_{res} (
                symbol TEXT, ts INTEGER, datetime_ny TEXT, buckets_json TEXT,
                PRIMARY KEY (symbol, ts)
            )
        """)
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_os_{res}_ts ON option_snapshots_{res} (ts)")

    conn.commit()
    conn.close()

def is_data_seeded(db_path, table_name, symbol):
    if not db_path.exists():
        return False
    try:
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not c.fetchone(): return False
            c.execute(f"SELECT 1 FROM {table_name} WHERE symbol = ? LIMIT 1", (symbol,))
            return c.fetchone() is not None
    except sqlite3.Error:
        return False

def write_data_to_sqlite(db_path, table_name, data):
    init_full_db_schema(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    if 'option' in table_name:
        ph = "?,?,?,?" 
    else:
        ph = "?,?,?,?,?,?,?,?"
    c.executemany(f"REPLACE INTO {table_name} VALUES ({ph})", data)
    conn.commit()
    conn.close()

# ================= 智能数据处理流程 =================
def process_stock_data_1s(sym):
    table_name = 'market_bars_1s'
    src_dir = STOCK_RAW_DIR / sym 
    
    if not src_dir.exists(): return
    target_files = sorted(list(src_dir.glob("*.parquet")))
    if not target_files: return

    for p_file in tqdm(target_files, desc=f"Stock {sym} [1s]", leave=False):
        try:
            date_str_raw = p_file.stem.split('_')[-1]
            try:
                file_date = pd.to_datetime(date_str_raw).date()
            except: continue
                
            if file_date < START_DATE or file_date > END_DATE: continue
            
            date_str_from_file = date_str_raw.replace('-', '')
            db_path = DB_DIR / f"market_{date_str_from_file}.db"
            
            if is_data_seeded(db_path, table_name, sym): continue

            df_stock = pd.read_parquet(p_file)
            if df_stock.empty: continue
            
            # 时间对齐
            time_col = 'timestamp' if 'timestamp' in df_stock.columns else 'ts'
            if pd.api.types.is_numeric_dtype(df_stock[time_col]):
                df_stock['timestamp'] = pd.to_datetime(df_stock[time_col], unit='s', utc=True).dt.tz_convert(NY_TZ).dt.round('1s')
            else:
                df_stock['timestamp'] = pd.to_datetime(df_stock[time_col], utc=True).dt.tz_convert(NY_TZ).dt.round('1s')
            
            # 过滤 RTH
            df_stock = df_stock.set_index('timestamp').between_time('09:30', '16:05').reset_index()
            if df_stock.empty: continue
            
            # 兼容字段：有的只有 close/price，没有 OHLC
            target_col = 'close' if 'close' in df_stock.columns else 'price'
            if 'open' not in df_stock.columns:
                for c in ['open', 'high', 'low']: df_stock[c] = df_stock[target_col]
            if 'volume' not in df_stock.columns:
                df_stock['volume'] = 0.0

            records = []
            # [性能优化] 使用 itertuples 替代 iterrows，速度提升 50 倍
            for row in df_stock.itertuples(index=False):
                records.append((
                    sym, int(row.timestamp.timestamp()), 
                    row.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(row.open), float(row.high), 
                    float(row.low), float(getattr(row, target_col)), 
                    float(row.volume)
                ))
            
            if records:
                logging.info(f"🔍 [Sample Stock] First 3 records for {sym}:")
                for r in records[:3]: logging.info(f"   {r}")
                write_data_to_sqlite(db_path, 'market_bars_1s', records)
            
            # [新增] 自动降采样同步 1m/5m 表
            for res_min in [1, 5]:
                res_suffix = f'{res_min}m'
                table_res = f'market_bars_{res_suffix}'
                if is_data_seeded(db_path, table_res, sym): continue
                
                df_res = df_stock.set_index('timestamp').resample(f'{res_min}T' ).agg({
                    'open': 'first', 'high': 'max', 'low': 'min', target_col: 'last', 'volume': 'sum'
                }).dropna().reset_index()
                
                res_records = []
                for row in df_res.itertuples(index=False):
                    res_records.append((
                        sym, int(row.timestamp.timestamp()), 
                        row.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        float(row.open), float(row.high), 
                        float(row.low), float(getattr(row, target_col)), 
                        float(row.volume)
                    ))
                if res_records:
                    write_data_to_sqlite(db_path, table_res, res_records)

            del df_stock; gc.collect()
        except Exception as e:
            logging.error(f"Error stock {p_file}: {e}")

def process_option_data_1s(sym):
    table_name = 'option_snapshots_1s'
    src_dir = OPT_RAW_DIR / sym  
    if not src_dir.exists(): return

    parquet_files = sorted(list(src_dir.glob("*.parquet")))
    if not parquet_files: return

    for p_file in tqdm(parquet_files, desc=f"Option {sym} [1s]", leave=False):
        try:
            date_str_raw = p_file.stem.split('_')[-1] 
            try:
                file_date = pd.to_datetime(date_str_raw).date()
            except: continue
                
            if file_date < START_DATE or file_date > END_DATE: continue
            
            date_str_from_file = date_str_raw.replace('-', '')
            db_path = DB_DIR / f"market_{date_str_from_file}.db"
            
            if is_data_seeded(db_path, table_name, sym): continue
            
            df_day = pd.read_parquet(p_file)
            if df_day.empty or 'bucket_id' not in df_day.columns: continue
                
            # 时间对齐
            if pd.api.types.is_numeric_dtype(df_day['timestamp']):
                df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], unit='s', utc=True).dt.tz_convert(NY_TZ).dt.round('1s')
            else:
                df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], utc=True).dt.tz_convert(NY_TZ).dt.round('1s')
            
            df_focus = df_day.set_index('timestamp').between_time('09:30', '16:05').reset_index()
            if df_focus.empty: continue

            # 秒级数据我们直接按 `timestamp` 分组（假设前面生成时已经对齐好了）
            records = []
            
            # 分钟级数据量少可逐行，但秒级一天 23400 秒，我们必须提速
            for ts_idx, sec_df in df_focus.groupby('timestamp'):
                ts_unix = int(ts_idx.timestamp())
                buckets = np.zeros((6, 12), dtype=float)
                contracts = [""] * 6
                
                # 遍历这 1 秒内的多个合约
                for row in sec_df.itertuples(index=False):
                    b_id = int(row.bucket_id)
                    if b_id < 0 or b_id > 5: continue
                    
                    # 适配 calc_offline_1s_greeks 生成的列名
                    price = float(getattr(row, 'mid_price', getattr(row, 'close', 0.0)))
                    contracts[b_id] = str(row.ticker)
                    
                    vals = [
                        price, 
                        float(getattr(row, 'delta', 0.0)),
                        float(getattr(row, 'gamma', 0.0)), 
                        float(getattr(row, 'vega', 0.0)),
                        float(getattr(row, 'theta', 0.0)), 
                        float(getattr(row, 'strike_price', getattr(row, 'strike', 0.0))),
                        float(getattr(row, 'volume', 0.0)), 
                        float(getattr(row, 'iv', 0.0)),
                        float(getattr(row, 'bid', price)), 
                        float(getattr(row, 'ask', price)),
                        float(getattr(row, 'bid_size', 100.0)), 
                        float(getattr(row, 'ask_size', 100.0))
                    ]
                    buckets[b_id] = [v if pd.notnull(v) else 0.0 for v in vals]
                
                records.append((
                    sym, ts_unix, 
                    ts_idx.strftime('%Y-%m-%d %H:%M:%S'),
                    json.dumps({'buckets': buckets.tolist(), 'contracts': contracts})
                ))
            
            if records:
                logging.info(f"🔍 [Sample Option] First 3 records for {sym}:")
                for r in records[:3]: logging.info(f"   {sym} | {r[1]} | {datetime.fromtimestamp(r[1], NY_TZ)}")
                write_data_to_sqlite(db_path, 'option_snapshots_1s', records)

            # [新增] 自动降采样同步 1m/5m 期权快照
            for res_min in [1, 5]:
                res_suffix = f'{res_min}m'
                table_res = f'option_snapshots_{res_suffix}'
                if is_data_seeded(db_path, table_res, sym): continue
                
                # 期权快照通常取周期内的最后一次
                df_res = df_focus.set_index('timestamp').resample(f'{res_min}T' ).last().dropna().reset_index()
                res_records = []
                for ts_idx, sec_df in df_res.groupby('timestamp'):
                    # 这里我们需要从原始 df_focus 中再次聚合该分钟的快照 (通常取最后一次完整的快照)
                    # 简化处理：直接取该分钟最后一秒的记录
                    ts_unix = int(ts_idx.timestamp())
                    # 查找对应时间戳的原始记录
                    last_sec = df_focus[df_focus['timestamp'] == ts_idx]
                    if last_sec.empty: continue
                    
                    buckets = np.zeros((6, 12), dtype=float)
                    contracts = [""] * 6
                    for row in last_sec.itertuples(index=False):
                        b_id = int(row.bucket_id)
                        if b_id < 0 or b_id > 5: continue
                        price = float(getattr(row, 'mid_price', getattr(row, 'close', 0.0)))
                        contracts[b_id] = str(row.ticker)
                        vals = [
                            price, float(getattr(row, 'delta', 0.0)),
                            float(getattr(row, 'gamma', 0.0)), float(getattr(row, 'vega', 0.0)),
                            float(getattr(row, 'theta', 0.0)), float(getattr(row, 'strike_price', getattr(row, 'strike', 0.0))),
                            float(getattr(row, 'volume', 0.0)), float(getattr(row, 'iv', 0.0)),
                            float(getattr(row, 'bid', price)), float(getattr(row, 'ask', price)),
                            float(getattr(row, 'bid_size', 100.0)), float(getattr(row, 'ask_size', 100.0))
                        ]
                        buckets[b_id] = [v if pd.notnull(v) else 0.0 for v in vals]
                    res_records.append((
                        sym, ts_unix, 
                        ts_idx.strftime('%Y-%m-%d %H:%M:%S'),
                        json.dumps({'buckets': buckets.tolist(), 'contracts': contracts})
                    ))

                if res_records:
                    write_data_to_sqlite(db_path, table_res, res_records)

            del df_day, df_focus; gc.collect()
        except Exception as e:
            logging.error(f"Error processing options file {p_file}: {e}")

if __name__ == "__main__":
    if not DB_DIR.exists(): DB_DIR.mkdir(parents=True)
    
    logging.info(f"🚀 Starting 1-Second High-Frequency SQLite Seeder. Filtering data >= {START_DATE} ...")
    #'VIXY', 'SPY', 'QQQ'
    SKIP_OPTIONS_SYMBOLS = {} # 视你需要跳过ETF期权而定
    
    for symbol in TARGET_SYMBOLS:
        logging.info(f"========== Processing 1s Data for {symbol} ==========")
        # 1. 刷入正股 1s
        process_stock_data_1s(symbol)
        
        # 2. 刷入期权 1s
        if symbol not in SKIP_OPTIONS_SYMBOLS:
            process_option_data_1s(symbol)
        else:
            logging.info(f"⏭️ Skipping Option processing for Index/ETF: {symbol}")
            
    logging.info("✅ All 1s High-Frequency Data Seeded to SQLite Successfully!")