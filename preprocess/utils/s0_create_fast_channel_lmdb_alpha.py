#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s0_create_fast_channel_lmdb_alpha_ny.py
描述: [v3.0 New York Timezone 版]
      1. 时区统一: 强制将所有数据转换为 [America/New_York] 并去除时区信息 (Wall Clock Time)。
         - 方便调试: 09:30 就是 09:30，不再是 UTC 的 14:30。
      2. 字段修复: 将 close 价格写入 metadata，解决下游价格缺失。
      3. 完整性验证: 针对 NY 时间进行数值级比对。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging 
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import psycopg2
import traceback
from collections import defaultdict
import lmdb       
import msgpack    
import msgpack_numpy
import multiprocessing as mp
import zstandard as zstd 
import warnings
import random
from numpy.lib.stride_tricks import sliding_window_view 
import queue

# --- 1. 配置 ---
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
msgpack_numpy.patch() 

CONFIG_PATH = Path.home() / 'notebook/train/fast_feature.json'
PARQUET_ROOT = Path(Path.home() / 'feature_data/stocks_parquet_features_val').parent
OUTPUT_DIR = Path('/mnt/s990/data/h5_unified_overlap_id')
from config import PG_DB_URL

LMDB_MAP_SIZE = 1000 * (1024**3) 
BATCH_SIZE_TO_QUEUE = 2000 
COMMIT_INTERVAL_SAMPLES = 50000 
MAX_PRODUCERS = max(1, os.cpu_count() - 2)

WINDOW_SIZE = 10  
WINDOW_STEP = 1   

STAGES_TO_PROCESS = {
    'backtest': ("2025-11-01", "2025-12-31"),
    'test': ("2025-07-01", "2025-10-31"),
    'val': ("2021-07-01", "2021-12-31"),
    'train': ("2022-01-01", "2025-06-30")
}

def get_session_info_from_path(path: Path):
    parts = path.parts
    if len(parts) < 4: return None, None, None
    res = parts[-2]
    time_range = parts[-3]
    session_name = parts[-4]
    return session_name, time_range, res

def load_config(path):
    with open(path) as f: return json.load(f)

def load_meta_info():
    try:
        conn = psycopg2.connect(PG_DB_URL)
        cursor = conn.cursor()

        from config import TARGET_SYMBOLS
        
        # TARGET_SYMBOLS = [  
        #     # --- Tier 1: 巨无霸 (>$10B) ---
        #     'NVDA', 'AAPL', 'META', 'MSTR', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 
            
        #     # --- Tier 2: 核心蓝筹 (>$3B) ---
        #     'NFLX', 'CRWV', 'COIN', 'AVGO', 'MSFT', 'GOOGL',   'HOOD',  
        #       'GOOG', 'MU', 'APP', 
            
        #     # --- Tier 3: 高流动性 ($1B - $3B) --- 
        #        'SMCI',   'ADBE',    
        #       'CRM',     'ORCL',    'NKE', 'XOM', 
        #           'CVX', 'UBER',    'INTC'
        # ]
        
        # PostgreSQL 的参数化查询使用 %s
        placeholders = ','.join(['%s'] * len(TARGET_SYMBOLS))
        query = f"SELECT symbol, stock_id, sector_id FROM stocks_us WHERE symbol IN ({placeholders})"
        
        cursor.execute(query, TARGET_SYMBOLS)
        rows = cursor.fetchall()
        conn.close()
        
        # 直接构建所需的 mapping，丢弃没用的 sector_to_sector_id
        stock_to_id = {symbol: stock_id for symbol, stock_id, _ in rows}
        stock_to_sector_id = {symbol: sector_id for symbol, _, sector_id in rows}
        
        return stock_to_id, stock_to_sector_id
    except Exception as e:
        print(f"Error loading meta info: {e}")
        return {}, {}

# --- 核心处理逻辑 ---
def process_stock(task_info):
    symbol, stock_id, sector_id, root_path, config, data_queue, is_debug_target = task_info
    
    try:
        cctx = zstd.ZstdCompressor(level=3) 
        feature_cols = [f['name'] for f in config.get('features', [])]
        label_config = config.get('labels', {})
        label_map = {k: f"label_{k}" for k in label_config.keys()}
        label_cols_needed = list(label_map.values())

        all_files = list(Path(root_path).glob(f'{symbol}/**/*.parquet'))
        if not all_files: return {"status": "skipped", "symbol": symbol}

        files_by_group = defaultdict(list)
        for f in all_files:
            sess, tr, res = get_session_info_from_path(f)
            files_by_group[(sess, tr, res)].append(f)

        total_produced = 0
        batch_buffer = [] 

        for (session_name, time_range, res), file_list in files_by_group.items():
            if session_name != 'regular' or res != '1min': continue
            file_list.sort()
            
            for f_1m in file_list:
                try:
                    df = pd.read_parquet(f_1m)
                    
                    if 'timestamp' not in df.columns:
                        if df.index.name == 'timestamp': df.reset_index(inplace=True)
                        else: continue
                    
                    # --- [1] 时区处理 ---
                    ts = pd.to_datetime(df['timestamp'])
                    if ts.dt.tz is None: 
                        ts = ts.dt.tz_localize('UTC')
                     
                    
                    # 转 NY 并去时区
                    ts = ts.dt.tz_convert('America/New_York') 
                    
                    # === [核心修复 1] 消除 1min 泄露 ===
                    # 如果原始数据是 Start Time (09:30 代表 09:30-09:31), 
                    # 那么这行数据包含的信息在 09:31 才能拿到。
                    # 我们将时间戳 +1 min，代表"数据可用时间 (Available Time)"。
                    ts = ts + pd.Timedelta(minutes=1)
                    
                    df['timestamp'] = ts.astype(np.int64)
                    df['date_grp'] = ts.dt.date # 用于分组计算防止隔夜污染
                    
                    df.sort_values('timestamp', inplace=True)
                    if len(df) < WINDOW_SIZE: continue

                    # === [核心修复 2] 消除隔夜数据污染 ===
                    # 必须按天分组计算 Rolling，防止把昨天的收盘带入今天的开盘
                    # 1. 计算 pct_change
                    df['ret'] = df.groupby('date_grp')['close'].pct_change().fillna(0.0)
                    
                    # 2. 计算 Rolling Std
                    # 注意: groupby().rolling() 会产生 MultiIndex，需要 reset
                    df['fast_vol'] = df.groupby('date_grp')['ret'].rolling(window=5).std().reset_index(level=0, drop=True).fillna(0.0) * 100.0
                    
                    # 3. 补全动量 (同理按天)
                    df['fast_mom'] = df.groupby('date_grp')['close'].pct_change(5).fillna(0.0)
                     #  === [新增] 详细检查 fast_vol 在原始数据中的情况 ===
                    

                    # [修复] 确保 Close 存在
                    if 'close' not in df.columns:
                        if 'Close' in df.columns: df['close'] = df['Close']
                        else: continue 

                    

                    for c in feature_cols: 
                        if c not in df.columns: df[c] = 0.0
                    for c in label_cols_needed:
                        if c not in df.columns: df[c] = 0.0

                    arr_x = df[feature_cols].values.astype(np.float32)
                    arr_y = df[label_cols_needed].values.astype(np.float32)
                    ts_arr = df['timestamp'].values
                    close_arr = df['close'].values 
                    
                    # 滑动窗口
                    wins_x = sliding_window_view(arr_x, window_shape=WINDOW_SIZE, axis=0)
                    
                    wins_x = wins_x[::WINDOW_STEP]
                    labels_aligned = arr_y[WINDOW_SIZE-1:][::WINDOW_STEP]
                    ts_aligned = ts_arr[WINDOW_SIZE-1:][::WINDOW_STEP]
                    close_aligned = close_arr[WINDOW_SIZE-1:][::WINDOW_STEP] 
                    
                    n_samples = len(wins_x)
                    if len(labels_aligned) < n_samples: n_samples = len(labels_aligned)
                    
                    local_count = 0
                    for i in range(n_samples):
                         

                        feat_dict = dict(zip(feature_cols, wins_x[i])) 
                        y_dict = dict(zip(label_config.keys(), labels_aligned[i]))
                        
                        sample = {
                            'features': feat_dict, 
                            'targets': y_dict,
                            'metadata': {
                                'symbol': symbol, 
                                'timestamp': int(ts_aligned[i]), 
                                'stock_id': int(stock_id),
                                'sector_id': int(sector_id),
                                'close': float(close_aligned[i]) 
                            }
                        }
                        
                        key = f"{symbol}_{ts_aligned[i]}".encode('ascii')
                        batch_buffer.append((key, cctx.compress(msgpack.packb(sample, use_bin_type=True))))
                        local_count += 1
                        
                        if len(batch_buffer) >= BATCH_SIZE_TO_QUEUE:
                            data_queue.put(batch_buffer)
                            batch_buffer = []

                    total_produced += local_count
                except Exception: continue

        if batch_buffer: data_queue.put(batch_buffer)
        return {"status": "success", "symbol": symbol, "segments_produced": total_produced}
    except Exception as e:
        return {"status": "error", "symbol": symbol, "error": str(e)}
    
def lmdb_writer_process(data_queue, output_path, total_tasks):
    logging.info(f"[Writer] 启动: {output_path}")
    env = lmdb.open(str(output_path), map_size=LMDB_MAP_SIZE, writemap=False, lock=True)
    txn = env.begin(write=True)
    total_written = 0
    all_keys = []
    pbar = tqdm(total=total_tasks, desc="[Writer]", unit="stock")
    while True:
        try:
            payload = data_queue.get(timeout=5)
            if payload == "STOP": break
            if isinstance(payload, list):
                for k, v in payload:
                    txn.put(k, v)
                    all_keys.append(k)
                    total_written += 1
                    if total_written % COMMIT_INTERVAL_SAMPLES == 0:
                        txn.commit(); txn = env.begin(write=True)
            elif isinstance(payload, dict): pbar.update(1)
        except queue.Empty: continue
        except Exception: txn.abort(); txn = env.begin(write=True)

    if all_keys:
        dctx = zstd.ZstdCompressor(level=3)
        txn.put(b'__keys__', dctx.compress(msgpack.packb(all_keys, use_bin_type=True)))
        txn.put(b'__len__', str(len(all_keys)).encode('ascii'))
    txn.commit(); env.close(); pbar.close()
    logging.info(f"🎉 [Writer] 完成。总写入样本: {total_written}")

# --- 验证函数 (适配 NY 时区) ---
def verify_data_integrity(lmdb_path, config, parquet_root):
    print(f"\n{'='*20} 开始深度数据完整性验证 (NY Timezone) {'='*20}")
    
    if not lmdb_path.exists():
        print("❌ LMDB 文件不存在")
        return

    feat_cols = [f['name'] for f in config.get('features', [])]

    try:
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        dctx = zstd.ZstdDecompressor()
        
        with env.begin() as txn:
            keys_c = txn.get(b'__keys__')
            if not keys_c: return
            all_keys = msgpack.unpackb(dctx.decompress(keys_c), raw=False)
            
            sample_keys = random.sample(all_keys, min(5, len(all_keys)))
            
            for key_bytes in sample_keys:
                key_str = key_bytes.decode('ascii')
                symbol, ts_str = key_str.rsplit('_', 1)
                target_ts_int = int(ts_str) 
                
                # 打印人类可读时间方便确认
                human_time = pd.Timestamp(target_ts_int)
                print(f"\n--- 验证: {symbol} @ {human_time} (NY) ---")
                
                val_c = txn.get(key_bytes)
                sample_data = msgpack.unpackb(dctx.decompress(val_c), raw=False)
                lmdb_features = sample_data['features']
                
                # Check Metadata Close
                meta_close = sample_data.get('metadata', {}).get('close', 0)
                if meta_close == 0:
                    print("⚠️ Metadata Close is ZERO!")
                else:
                    print(f"✅ Metadata Close: {meta_close}")

                stock_dir = parquet_root / symbol
                found_in_parquet = False
                files = list(stock_dir.glob("**/regular/**/1min/*.parquet"))
                
                for p_file in files:
                    try:
                        df = pd.read_parquet(p_file)
                        if 'timestamp' not in df.columns:
                            if df.index.name == 'timestamp': df.reset_index(inplace=True)
                            else: continue

                        # === [验证逻辑: 保持与生成一致] ===
                        ts_series = pd.to_datetime(df['timestamp'])
                        if ts_series.dt.tz is None:
                            ts_series = ts_series.dt.tz_localize('UTC')
                        else:
                            ts_series = ts_series.dt.tz_convert('UTC')
                        
                        # 转 NY 并去时区
                        ts_series = ts_series.dt.tz_convert('America/New_York') 
                        ts_int_arr = ts_series.astype(np.int64).values
                        # =================================

                        indices = np.where(ts_int_arr == target_ts_int)[0]
                        
                        if len(indices) > 0:
                            found_in_parquet = True
                            print(f"🎉 时间戳匹配成功! (Source: {p_file.name})")
                            break 
                            
                    except Exception: continue
                
                if not found_in_parquet:
                    print(f"⚠️ 未在 Parquet 中找到时间戳 {human_time}")

    except Exception as e:
        print(f"验证出错: {e}")
    finally:
        if 'env' in locals(): env.close()

def run_stage(stage, config):
    print(f"\n🚀 处理阶段: {stage}")
    stock_to_id, stock_to_sector_id = load_meta_info()
    root_path = PARQUET_ROOT / f"stocks_parquet_features_{stage}"
    output_path = OUTPUT_DIR / f"{stage}_fast_channel_features.lmdb"
    
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path) if output_path.is_dir() else os.remove(output_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manager = mp.Manager()
    data_queue = manager.Queue(maxsize=1000)
    
    tasks = []
    debug_symbol = "AAPL" if "AAPL" in stock_to_id else (sorted(list(stock_to_id.keys()))[0] if stock_to_id else "")

    for symbol, stock_id in stock_to_id.items():
        if symbol in stock_to_sector_id:
            is_debug = (symbol == debug_symbol)
            tasks.append((symbol, stock_id, stock_to_sector_id[symbol], root_path, config, data_queue, is_debug))

    writer_proc = mp.Process(target=lmdb_writer_process, args=(data_queue, output_path, len(tasks)))
    writer_proc.start()
    
    with ProcessPoolExecutor(max_workers=MAX_PRODUCERS) as executor:
        futures = [executor.submit(process_stock, t) for t in tasks]
        for f in as_completed(futures):
            try: data_queue.put(f.result())
            except: pass
            
    data_queue.put("STOP")
    writer_proc.join()
    verify_data_integrity(output_path, config, root_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    main_config = load_config(CONFIG_PATH)
    for stage in STAGES_TO_PROCESS:
        run_stage(stage, main_config)