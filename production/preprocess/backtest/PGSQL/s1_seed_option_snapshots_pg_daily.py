#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import psycopg2
import json
import logging
import warnings
import gc
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import pytz
import argparse

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Option Snapshot Seeder (V10 PG Direct Inject)")
    # 仅保留起始日期控制，摒弃旧的 date 路径拼接
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="只处理该日期及之后的数据 (格式: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2099-12-31", help="只处理该日期及之前的数据 (格式: YYYY-MM-DD)")
    return parser.parse_args()

args = parse_args()

# 将输入的字符串转换为 date 对象，方便后续作比较
START_DATE = pd.to_datetime(args.start_date).date()
END_DATE = pd.to_datetime(args.end_date).date()

# 纯净固定的 V10 高精度数据路径
OPT_RAW_DIR  = Path.home() / "train_data/quote_options_day_iv" 
STOCK_RAW_DIR= Path.home() / "train_data/spnq_train_resampled" 

NY_TZ = pytz.timezone('America/New_York')

from config import PG_DB_URL
from config import TARGET_SYMBOLS 

# ================= PostgreSQL 建表与检查 =================
def _get_pg_conn():
    return psycopg2.connect(PG_DB_URL)

def ensure_pg_tables(date_str, res='1m'):
    from datetime import datetime as _dt, timedelta
    day = _dt.strptime(date_str, '%Y%m%d')
    start_dt = NY_TZ.localize(day)
    end_dt = start_dt + timedelta(days=1)
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    
    conn = _get_pg_conn()
    conn.autocommit = True
    c = conn.cursor()
    
    # 动态表名
    mb_table = f"market_bars_{res}"
    os_table = f"option_snapshots_{res}"

    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {mb_table} (
            symbol TEXT, ts DOUBLE PRECISION,
            open DOUBLE PRECISION, high DOUBLE PRECISION,
            low DOUBLE PRECISION, close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        ) PARTITION BY RANGE (ts);
    """)
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {os_table} (
            symbol TEXT, ts DOUBLE PRECISION, buckets_json JSONB,
            PRIMARY KEY (symbol, ts)
        ) PARTITION BY RANGE (ts);
    """)
    
    for table in [mb_table, os_table]:
        part_name = f"{table}_{date_str}"
        try:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {part_name} PARTITION OF {table}
                FOR VALUES FROM ({start_ts}) TO ({end_ts});
            """)
        except Exception:
            pass
    c.close(); conn.close()

def is_data_seeded(date_str, table_base, res, symbol):
    table_name = f"{table_base}_{res}"
    try:
        conn = _get_pg_conn()
        c = conn.cursor()
        c.execute("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=%s)", 
                  (f"{table_name}_{date_str}",))
        if not c.fetchone()[0]:
            c.close(); conn.close()
            return False
        c.execute(f"SELECT 1 FROM {table_name}_{date_str} WHERE symbol = %s LIMIT 1", (symbol,))
        result = c.fetchone() is not None
        c.close(); conn.close()
        return result
    except Exception:
        return False

def write_bars_to_pg(date_str, data, res='1m'):
    ensure_pg_tables(date_str, res)
    table_name = f"market_bars_{res}"
    conn = _get_pg_conn()
    c = conn.cursor()
    c.executemany(f"""
        INSERT INTO {table_name} (symbol, ts, open, high, low, close, volume) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, ts) DO UPDATE SET 
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, 
            close=EXCLUDED.close, volume=EXCLUDED.volume
    """, data)
    conn.commit()
    c.close(); conn.close()

def write_options_to_pg(date_str, data, res='1m'):
    ensure_pg_tables(date_str, res)
    table_name = f"option_snapshots_{res}"
    conn = _get_pg_conn()
    c = conn.cursor()
    c.executemany(f"""
        INSERT INTO {table_name} (symbol, ts, buckets_json)
        VALUES (%s, %s, %s)
        ON CONFLICT (symbol, ts) DO UPDATE SET buckets_json=EXCLUDED.buckets_json
    """, data)
    conn.commit()
    c.close(); conn.close()

# ================= 智能数据处理流程 =================

def process_stock_data(sym, res_type='1min'):
    # 支持 1min 和 5min 路径探测
    res_suffix = '1m' if res_type == '1min' else '5m'
    
    src_dir = STOCK_RAW_DIR / sym 
    # 深度搜索 1min/5min 文件夹
    target_files = sorted(list(src_dir.glob(f"**/{res_type}/*.parquet")))
    
    if not target_files:
        # 兜底：如果直接在根目录
        target_files = sorted(list(src_dir.glob("*.parquet")))
        if not target_files: return

    for p_file in tqdm(target_files, desc=f"Stock {sym} [{res_type}]", leave=False):
        try:
            df_ts = pd.read_parquet(p_file, columns=['timestamp'])
            ts_series = pd.to_datetime(df_ts['timestamp'])
            if ts_series.dt.tz is None:
                ts_series = ts_series.dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            else:
                ts_series = ts_series.dt.tz_convert(NY_TZ)
            
            # 🚨 剔除错误的时间偏移
            # ts_series = ts_series + pd.Timedelta(minutes=shift_mins)
                
            unique_dates = ts_series.dt.date.unique()
            dates_to_process = []
            for d in unique_dates:
                if d < START_DATE or d > END_DATE: continue
                date_str = d.strftime('%Y%m%d')
                if not is_data_seeded(date_str, 'market_bars', res_suffix, sym):
                    dates_to_process.append(d)
            
            if not dates_to_process: continue
            
            df_stock = pd.read_parquet(p_file)
            df_stock['timestamp'] = pd.to_datetime(df_stock['timestamp'])
            if df_stock['timestamp'].dt.tz is None:
                df_stock['timestamp'] = df_stock['timestamp'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            else:
                df_stock['timestamp'] = df_stock['timestamp'].dt.tz_convert(NY_TZ)
            
            # 🚨 剔除错误的时间偏移
            # df_stock['timestamp'] = df_stock['timestamp'] + pd.Timedelta(minutes=shift_mins)
            df_stock['date'] = df_stock['timestamp'].dt.date
            
            df_stock = df_stock[df_stock['date'].isin(dates_to_process)]
            # 1min 正常截断，5min 也要确保在交易时间内
            df_stock = df_stock.set_index('timestamp').between_time('09:30', '16:05').reset_index()
            
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
                    date_str = current_date.strftime('%Y%m%d')
                    write_bars_to_pg(date_str, records, res_suffix)
                    
            del df_stock, df_ts; gc.collect()
        except Exception as e:
            logging.error(f"Error stock {p_file}: {e}")

def process_option_data(sym, res_type='1min'):
    res_suffix = '1m' if res_type == '1min' else '5m'
    
    # 假设 5min 期权数据在 standard_5m 类似结构，或者通过替换路径实现
    if res_type == '1min':
        src_dir = OPT_RAW_DIR / sym / "standard"
    else:
        # 如果不存在 standard_5m，尝试在 standard 下找，或者其他可能路径
        src_dir = OPT_RAW_DIR / sym / "standard_5m"
        if not src_dir.exists():
            # 尝试在原路径下寻找带 5m/5min 标识的文件
            src_dir = OPT_RAW_DIR / sym / "standard"
    
    if not src_dir.exists(): return

    pattern = "*.parquet"
    parquet_files = sorted(list(src_dir.glob(pattern)))
    
    for p_file in tqdm(parquet_files, desc=f"Option {sym} [{res_type}]", leave=False):
        # 如果是 5min 请求但文件不含 5m/5min 关键字且 1min 请求存在，则跳过
        if res_type == '5min' and "5m" not in str(p_file).lower() and (OPT_RAW_DIR / sym / "standard").exists() and res_type != '1min':
             pass

        try:
            date_str_raw = p_file.stem.split('_')[-1] 
            try:
                file_date = pd.to_datetime(date_str_raw).date()
            except:
                continue
                
            if file_date < START_DATE or file_date > END_DATE: continue
            
            date_str_from_file = date_str_raw.replace('-', '')
            
            df_day = pd.read_parquet(p_file)
            if df_day.empty: continue
            
            if 'bucket_id' not in df_day.columns: continue
                
            df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
            if df_day['timestamp'].dt.tz is None:
                df_day['timestamp'] = df_day['timestamp'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
            else:
                df_day['timestamp'] = df_day['timestamp'].dt.tz_convert(NY_TZ)
            
            # 🚨 剔除错误的时间偏移
            # df_day['timestamp'] = df_day['timestamp'] + pd.Timedelta(minutes=shift_mins)
            
            df_focus = df_day.set_index('timestamp').between_time('09:30', '16:05')
            if df_focus.empty: continue
            
            # Resample 逻辑适配 (改进版：确保 bucket_id 不丢失)
            resample_freq = '1Min' if res_type == '1min' else '5Min'
            
            # 1. 设置索引并进行 GroupBy
            df_focus = df_focus.reset_index().set_index('timestamp')
            
            # [逻辑修正] 使用 closed='right', label='right' 确保 10:00:01 ~ 10:01:00 的数据归入 10:01:00
            # 这与正股及大部分数据源的 End-of-Bar 标注逻辑一致
            df_list = []
            for b_id, group in df_focus.groupby('bucket_id'):
                # 对单个 bucket 进行重采样
                resampled = group.resample(resample_freq, closed='right', label='right').last()
                # 重新填充丢失的 bucket_id
                resampled['bucket_id'] = b_id
                # 🚨 [关键修复] 不要执行 ffill()！
                resampled = resampled.dropna(subset=['close']) if 'close' in resampled.columns else resampled.dropna(how='all')
                df_list.append(resampled)

            
            if not df_list: continue
            df_focus = pd.concat(df_list).reset_index()

            # 🚀 [🔥 核心修复] 强制再次对齐时间戳，确保 5min 模式下不会因为原始数据有 1min 偏移而产生多余的行
            if res_type != '1min':
                df_focus['timestamp'] = df_focus['timestamp'].dt.floor('5Min')
                df_focus = df_focus.groupby(['timestamp', 'bucket_id']).last().reset_index()

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
                write_options_to_pg(date_str_from_file, records, res_suffix)
            
            del df_day, df_focus; gc.collect()
        except Exception as e:
            logging.error(f"Error processing options file {p_file}: {e}")

if __name__ == "__main__":
    logging.info(f"🚀 Starting Dual-Resolution PG Snapshot Seeder. Filtering [{START_DATE}] to [{END_DATE}] ...")
    
    SKIP_OPTIONS_SYMBOLS = {'VIXY','SPY','QQQ'}
    
    for symbol in TARGET_SYMBOLS:
        logging.info(f"========== Processing {symbol} ==========")
        # 1. 处理股票数据 (1min & 5min)
        process_stock_data(symbol, res_type='1min')
        process_stock_data(symbol, res_type='5min')
        
        # 2. 处理期权数据 (1min & 5min)
        if symbol not in SKIP_OPTIONS_SYMBOLS:
            process_option_data(symbol, res_type='1min')
            process_option_data(symbol, res_type='5min')
        else:
            logging.info(f"⏭️ Skipping Option processing for Index/ETF: {symbol}")
            
    logging.info("✅ All Dual-Resolution Data Seeded Successfully!")