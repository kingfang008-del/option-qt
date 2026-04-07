#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import warnings
import gc
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import pytz
import argparse
from datetime import timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= 配置区域 =================
PROJECT_ROOT = Path.home() / "quant_project"
DB_DIR       = PROJECT_ROOT / "data" / "history_sqlite_1m" 

# ================= 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Option Snapshot Seeder (V10)")
    parser.add_argument("--start_date", type=str, default="2026-01-02", help="只处理该日期及之后的数据 (格式: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2099-12-31", help="只处理该日期及之前的数据 (格式: YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="强制重新播种，即使数据库中已有数据")
    return parser.parse_args()

args = parse_args()

START_DATE = pd.to_datetime(args.start_date).date()
END_DATE   = pd.to_datetime(args.end_date).date()

OPT_RAW_DIR  = Path.home() / "train_data/quote_options_day_iv" 
STOCK_RAW_DIR= Path.home() / "train_data/spnq_train_resampled" 

NY_TZ = pytz.timezone('America/New_York')

# 🚀 [Fix] 添加 baseline 路径，解决 ModuleNotFoundError
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "baseline"))

from config import TARGET_SYMBOLS 

# ================= 数据库与检查逻辑 =================
def init_full_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_bars_1m (
            symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_mb_1m_ts ON market_bars_1m (ts)")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS option_snapshots_1m (
            symbol TEXT, ts INTEGER, buckets_json TEXT,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_os_1m_ts ON option_snapshots_1m (ts)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS market_bars_5m (
            symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_mb_5m_ts ON market_bars_5m (ts)")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS option_snapshots_5m (
            symbol TEXT, ts INTEGER, buckets_json TEXT,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_os_5m_ts ON option_snapshots_5m (ts)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            ts REAL, datetime_ny TEXT, symbol TEXT, action TEXT, qty REAL, price REAL, details_json TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_trd_sym_ts ON trade_logs (symbol, ts)")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS alpha_logs (
            ts REAL, datetime_ny TEXT, symbol TEXT, alpha REAL, iv REAL, price REAL, vol_z REAL, event_prob REAL,
            PRIMARY KEY (symbol, ts)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_alpha_ts ON alpha_logs (ts)")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS feature_logs (
            symbol TEXT, ts INTEGER, fast_norm_blob BLOB, slow_norm_blob BLOB,
            PRIMARY KEY (symbol, ts)
        )
    """)
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
    if not data: return
    ph = ",".join(["?"] * len(data[0]))
    c.executemany(f"REPLACE INTO {table_name} VALUES ({ph})", data)
    conn.commit()
    conn.close()

# ================= 智能数据处理流程 =================
def process_stock_data(sym, res_type='1min'):
    res_suffix = '1m' if res_type == '1min' else '5m'
    table_name = f'market_bars_{res_suffix}'

    src_dir = STOCK_RAW_DIR / sym
    # 🚀 [Filter] 只要 regular (盘中) 数据
    target_files = sorted(list(src_dir.glob(f"**/regular/**/{res_type}/*.parquet")))
    
    if not target_files:
        logging.warning(f"⚠️ [Stock] No 'regular' session files found for {sym} in {src_dir}")
        return

    #logging.info(f"🔎 [Stock] Processing {len(target_files)} 'regular' session files for {sym}")
    # 🚀 [Fix] resample_new.py 会按 session (pre_market, regular, after_hours) 生成多个月度文件
    # 我们需要按文件名（如 2026-01.parquet）归类，合并所有 session 的数据一起处理。
    from collections import defaultdict
    monthly_files = defaultdict(list)
    for p_file in target_files:
        monthly_files[p_file.name].append(p_file)
    
    for fname, files in tqdm(monthly_files.items(), desc=f"Stock {sym} [Monthly]", leave=False):
        try:
            # 1. 合并当前月份的所有 session 文件
            dfs = []
            for f in files:
                try: dfs.append(pd.read_parquet(f))
                except: continue
            if not dfs: continue
            df_stock = pd.concat(dfs, ignore_index=True)
            #logging.info(f"📦 [Stock] Read {len(df_stock)} rows from {len(files)} session files for {fname}")

            ts_series = pd.to_datetime(df_stock['timestamp'])
            if ts_series.dt.tz is None:
                ts_series = ts_series.dt.tz_localize(NY_TZ, ambiguous='infer')
            else:
                ts_series = ts_series.dt.tz_convert(NY_TZ)
            
            df_stock['timestamp'] = ts_series
            df_stock['date'] = ts_series.dt.date
            
            unique_dates = df_stock['date'].unique()
            dates_to_process = []
            for d in unique_dates:
                if d < START_DATE or d > END_DATE: continue
                date_str = d.strftime('%Y%m%d')
                db_path = DB_DIR / f"market_{date_str}.db"
                if args.force or not is_data_seeded(db_path, table_name, sym):
                    dates_to_process.append(d)
            
            if not dates_to_process:
                continue

            df_stock = df_stock[df_stock['date'].isin(dates_to_process)]
            #logging.info(f"📦 [Stock] Read {len(df_stock)} rows from {p_file.name}")
            
            df_stock['timestamp'] = pd.to_datetime(df_stock['timestamp'])
            if df_stock['timestamp'].dt.tz is None:
                # 🚀 [Fix] resample_new.py 产出的 Parquet 是不带时区的本地美东时间 (Local NY Time)
                df_stock['timestamp'] = df_stock['timestamp'].dt.tz_localize(NY_TZ, ambiguous='infer')
            else:
                df_stock['timestamp'] = df_stock['timestamp'].dt.tz_convert(NY_TZ)
            
            df_stock['date'] = df_stock['timestamp'].dt.date
            df_stock = df_stock[df_stock['date'].isin(dates_to_process)]
            #logging.info(f"📂 [Stock] After Date Filter: {len(df_stock)} rows remaining")
            
            df_stock = df_stock.set_index('timestamp').between_time('09:30', '16:05').reset_index()
            #logging.info(f"⏰ [Stock] After Between_Time Filter: {len(df_stock)} rows remaining")
            
            if len(df_stock) == 0:
                logging.warning(f"⚠️ [Stock] {sym} is empty after time filtering. Check your Parquet timestamps.")
                continue
            
            for current_date, df_day in df_stock.groupby('date'):
                records = []
                for _, row in df_day.iterrows():
                    records.append((
                        sym, int(row['timestamp'].timestamp()), 
                        float(row['open']), float(row['high']), 
                        float(row['low']), float(row['close']), 
                        float(row['volume'])
                    ))
                
                if records:
                    # 🚀 [Diagnostic Check] 只要数据异常少，就发出警告并直接跳过，不要整个 return
                    if len(records) < 10:
                         logging.error(f"❌ [Stock Anomaly] {sym} {current_date}: too few records ({len(records)}). SKIPPING date.")
                         continue

                    date_str = current_date.strftime('%Y%m%d')
                    db_path = DB_DIR / f"market_{date_str}.db"
                    write_data_to_sqlite(db_path, table_name, records)
                    #logging.info(f"✅ [Stock] {sym} {date_str} seeded {len(records)} bars into {table_name}")
                    
            del df_stock, ts_series; gc.collect()
        except Exception as e:
            logging.error(f"Error stock {p_file}: {e}")

def process_option_data(sym, res_type='1min'):
    res_suffix = '1m' if res_type == '1min' else '5m'
    table_name = f'option_snapshots_{res_suffix}'
    
    if res_type == '1min':
        # 🚀 [Fix] 优先探测 standard 子目录 (V10 结构)
        src_dir = OPT_RAW_DIR / sym / "standard"
        if not src_dir.exists():
            src_dir = OPT_RAW_DIR / sym  
    else:
        src_dir = OPT_RAW_DIR / sym / "standard_5m"
        if not src_dir.exists():
            src_dir = OPT_RAW_DIR / sym / "standard"
    
    logging.info(f"🔎 [Option] Checking {sym} {res_type} in: {src_dir} (exists: {src_dir.exists()})")
    if not src_dir.exists(): 
        logging.warning(f"⚠️ [Option] Directory NOT FOUND: {src_dir}")
        return

    # 🚀 [Optimization] 只构造日期范围内的 Glob Pattern，避免在 2400 个文件中循环
    parquet_files = []
    curr_d = START_DATE
    while curr_d <= END_DATE:
        d_str = curr_d.strftime('%Y-%m-%d')
        # 兼容 SYMBOL_YYYY-MM-DD.parquet 或 YYYY-MM-DD.parquet
        matched = list(src_dir.glob(f"*{d_str}.parquet"))
        parquet_files.extend(matched)
        curr_d += timedelta(days=1)
    
    parquet_files = sorted(list(set(parquet_files))) # 去重并排序
    logging.info(f"📂 [Option] Found {len(parquet_files)} targeted parquet files for {sym} {res_type} in range {START_DATE}~{END_DATE}")

    for p_file in tqdm(parquet_files, desc=f"Option {sym} [{res_type}]", leave=False):
        try:
            date_str_raw = p_file.stem.split('_')[-1] 
            try:
                file_date = pd.to_datetime(date_str_raw).date()
            except:
                logging.error(f"❌ [Option] Cannot parse date from filename: {p_file.name}")
                continue
                
            date_str_raw = p_file.stem.split('_')[-1] 
            date_str_from_file = date_str_raw.replace('-', '')
            db_path = DB_DIR / f"market_{date_str_from_file}.db"
            
            if not args.force and is_data_seeded(db_path, table_name, sym):
                logging.info(f"⏭️ [Option] {sym} on {date_str_from_file} already exists in {db_path.name}. Skipping (use --force to overwrite).")
                continue
            
            logging.info(f"⚡ [Option] Processing: {p_file.name} -> {db_path.name}")
            df_day = pd.read_parquet(p_file)
            if df_day.empty: 
                logging.warning(f"⚠️ [Option] {p_file.name} is EMPTY")
                continue
            
            if 'bucket_id' not in df_day.columns: 
                logging.error(f"❌ [Option] 'bucket_id' MISSING in {p_file.name}. Columns: {df_day.columns.tolist()}")
                continue
                
            df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
            if df_day['timestamp'].dt.tz is None:
                # 🚀 [Fix] 同上，期权同步对齐美东本地时间
                df_day['timestamp'] = df_day['timestamp'].dt.tz_localize(NY_TZ, ambiguous='infer')
            else:
                df_day['timestamp'] = df_day['timestamp'].dt.tz_convert(NY_TZ)
            
            t_min, t_max = df_day['timestamp'].min(), df_day['timestamp'].max()
            logging.info(f"🕒 [Option] {p_file.name} Time Range: {t_min.strftime('%H:%M:%S')} to {t_max.strftime('%H:%M:%S')} (NY Time)")

            df_focus = df_day.set_index('timestamp').between_time('09:30', '16:05')
            if df_focus.empty: 
                logging.warning(f"⚠️ [Option] {p_file.name} has NO data between 09:30-16:05 (Actual range: {t_min.strftime('%H:%M:%S')}-{t_max.strftime('%H:%M:%S')})")
                continue
            
            resample_freq = '1Min' if res_type == '1min' else '5Min'
            df_focus = df_focus.reset_index().set_index('timestamp')
            
             
            df_list = []
            for b_id, group in df_focus.groupby('bucket_id'):
                resampled = group.resample(resample_freq).last()

                resampled['bucket_id'] = b_id
                resampled = resampled.ffill()
                df_list.append(resampled)

            
            if not df_list: 
                logging.warning(f"⚠️ [Option] {p_file.name} resample resulted in empty list")
                continue
            df_focus = pd.concat(df_list).reset_index()

            records = []
            for ts_idx, minute_df in df_focus.groupby('timestamp'):
                ts_unix = int(ts_idx.timestamp())
                buckets = np.zeros((6, 12), dtype=float)
                contracts = [""] * 6
                
                for _, row_data in minute_df.iterrows():
                    b_id = int(row_data['bucket_id'])
                    if b_id < 0 or b_id > 5: continue
                    
                    price = float(row_data.get('close', 0.0))
                    contracts[b_id] = str(row_data.get('ticker', ''))
                    
                    vals = [
                        price, float(row_data.get('delta', 0.0)),
                        float(row_data.get('gamma', 0.0)), float(row_data.get('vega', 0.0)),
                        float(row_data.get('theta', 0.0)), float(row_data.get('strike_price', 0.0)),
                        float(row_data.get('volume', 0.0)), float(row_data.get('iv', 0.0)),
                        float(row_data.get('bid', price)), float(row_data.get('ask', price)),
                        float(row_data.get('bid_size', 100.0)), float(row_data.get('ask_size', 100.0))
                    ]
                    buckets[b_id] = [v if pd.notnull(v) else 0.0 for v in vals]
                
                records.append((sym, ts_unix, json.dumps({'buckets': buckets.tolist(), 'contracts': contracts})))
            
            if records:
                logging.info(f"💾 [Option] Writing {len(records)} snapshots into {db_path.name} -> {table_name}")
                write_data_to_sqlite(db_path, table_name, records)
                logging.info(f"✅ [Option] {sym} {date_str_from_file} seeded {len(records)} snapshots into {table_name}")
            
            del df_day, df_focus; gc.collect()
        except Exception as e:
            logging.error(f"Error processing options file {p_file}: {e}")

if __name__ == "__main__":
    if not DB_DIR.exists(): DB_DIR.mkdir(parents=True)
    
    logging.info(f"🚀 Starting Dual-Resolution SQLite Seeder. Filtering data >= {START_DATE} ...")
    #'VIXY','SPY','QQQ'
    SKIP_OPTIONS_SYMBOLS = {'VIXY'}
    
    for symbol in TARGET_SYMBOLS:
        logging.info(f"========== Processing {symbol} ==========")
        process_stock_data(symbol, res_type='1min')
        process_stock_data(symbol, res_type='5min')
        
        if symbol not in SKIP_OPTIONS_SYMBOLS:
            process_option_data(symbol, res_type='1min')
            process_option_data(symbol, res_type='5min')
        else:
            logging.info(f"⏭️ Skipping Option processing for Index/ETF: {symbol}")
            
    logging.info("✅ All Dual-Resolution Data Seeded to SQLite Successfully!")