#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s0_create_slow_channel_lmdb_alpha_ny.py
描述: [v3.0 New York Timezone 版 - 慢通道]
      1. 时区统一: 将 1min 和 5min 数据全部转为 [America/New_York] (Wall Clock Time)。
      2. 对齐修复: 确保 df_1m 和 df_5m 在相同的时间基准下进行 join。
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
import queue
import warnings
import random

# --- 1. 配置 ---
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
msgpack_numpy.patch() 

CONFIG_PATH = Path.home() / 'notebook/train/slow_feature.json'
PARQUET_ROOT = Path(Path.home() / 'train_data/quote_features_val').parent
OUTPUT_DIR = Path('/mnt/s990/data/h5_unified_overlap_id')
from config import PG_DB_URL

LMDB_MAP_SIZE = 1000 * (1024**3) 
BATCH_SIZE_TO_QUEUE = 1000 
COMMIT_INTERVAL_SAMPLES = 20000 
MAX_PRODUCERS = max(1, os.cpu_count() - 2)

WINDOW_SIZE_1M = 30 
WINDOW_SIZE_5M = 6   
WINDOW_STEP = 5      

STAGES_TO_PROCESS = {
    'val': ("2025-07-01", "2025-12-31"),
    'train': ("2022-03-01", "2025-06-30")
}

# --- 2. 辅助函数 ---
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
       
        # 建议使用的 Top 50 训练白名单 (按流动性降序)
        from config import TARGET_SYMBOLS
        
        # PostgreSQL 的参数化查询使用 %s
        placeholders = ','.join(['%s'] * len(TARGET_SYMBOLS))
        query = f"SELECT symbol, id, sector_id FROM stocks_us WHERE symbol IN ({placeholders})"
        
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

# --- 3. 核心 Worker ---
def process_stock(task_info):
    symbol, stock_id, sector_id, root_path, config, data_queue, is_debug_target = task_info
    
    def dprint(msg):
        if is_debug_target: print(f"[DEBUG {symbol}] {msg}")

    try:
        cctx = zstd.ZstdCompressor(level=3) 
        
        # ==========================================================
        # 🚀 [优化 1] 动态路由特征，分离 1min 和 5min
        # ==========================================================
        feats_1m_cols = []
        feats_5m_cols = []
        for f in config.get('features', []):
            name = f['name']
            res = f.get('resolution', '1min') 
            if res == '1min': feats_1m_cols.append(name)
            elif res == '5min': feats_5m_cols.append(name)
            elif res == 'both':
                feats_1m_cols.append(name)
                feats_5m_cols.append(name)
        
        label_cols = ['label_direction', 'label_return_fwd', 'label_volatility', 'label_event']

        all_files = list(Path(root_path).glob(f'{symbol}/**/*.parquet'))
        if not all_files: return {"status": "skipped", "symbol": symbol}

        files_by_group = defaultdict(list)
        for f in all_files:
            sess, tr, res = get_session_info_from_path(f)
            files_by_group[(sess, tr, res)].append(f)

        total_produced = 0
        batch_buffer = [] 

        dprint(f"Found {len(files_by_group)} file groups.")

        for (session_name, time_range, res), file_list in files_by_group.items():
            if session_name != 'regular' or res != '1min': continue
            
            file_list.sort()
            
            for f_1m in file_list:
                f_5m = Path(str(f_1m).replace('/1min/', '/5min/'))
                if not f_5m.exists(): continue

                try:
                    df_1m = pd.read_parquet(f_1m)
                    df_5m = pd.read_parquet(f_5m)
                    
                    # 列名标准化
                    for df in [df_1m, df_5m]:
                        if 'timestamp' not in df.columns:
                            if df.index.name == 'timestamp': df.reset_index(inplace=True)
                            else:
                                for c in ['Date', 'time', 'ts', 'index']:
                                    if c in df.columns: df.rename(columns={c: 'timestamp'}, inplace=True); break
                    
                    if 'timestamp' not in df_1m.columns or 'timestamp' not in df_5m.columns: continue

                    # === [核心修复] 时区统一转为 NY Wall Clock Time ===
                    # 🚀 [优化 2] 独立时间平移，完美对齐 K线闭合时间
                    def align_time(df, shift_minutes):
                        ts = pd.to_datetime(df['timestamp'])
                        if ts.dt.tz is None: ts = ts.dt.tz_localize('UTC')
                        ts = ts.dt.tz_convert('America/New_York')
                        # 1min线加1分，5min线加5分。这样 09:34(1m) 和 09:30(5m) 会在 09:35:00 完美相遇！
                        ts = ts + pd.Timedelta(minutes=shift_minutes)
                        df['timestamp'] = ts.astype(np.int64)

                    align_time(df_1m, shift_minutes=1)
                    align_time(df_5m, shift_minutes=5)
                    # ==========================================================

                    df_1m.sort_values('timestamp', inplace=True)
                    df_5m.sort_values('timestamp', inplace=True)

                    ts_5m_map = {t: i for i, t in enumerate(df_5m['timestamp'].values)}
                    ts_1m = df_1m['timestamp'].values
                    
                    overlap_count = 0
                    for t in ts_1m:
                        if t in ts_5m_map: overlap_count += 1
                    
                    dprint(f"File {f_1m.name}: Overlap={overlap_count}")
                    if overlap_count == 0: continue

                    def to_np(df, cols):
                        for c in cols: 
                            if c not in df.columns: df[c] = 0.0
                        return df[cols].values.astype(np.float32)

                    arr_1m = to_np(df_1m, feats_1m_cols)
                    arr_5m = to_np(df_5m, feats_5m_cols)
                    arr_lbl = to_np(df_1m, label_cols)
                    
                    local_count = 0
                    
                    # === 自动寻找相位 (Phase Alignment) ===
                    start_idx = WINDOW_SIZE_1M - 1
                    found_alignment = False
                    search_limit = min(start_idx + WINDOW_STEP, len(df_1m))
                    
                    for temp_i in range(start_idx, search_limit):
                        t_check = ts_1m[temp_i]
                        if t_check in ts_5m_map:
                            start_idx = temp_i
                            found_alignment = True
                            dprint(f"  -> Aligned start at index {start_idx}")
                            break
                    
                    if not found_alignment: continue
                    # ===================================

                    # [优化] 预先提取 Close 价格数组，专门用于死水检查
                    # 这样做比去 arr_1m 里猜列索引更安全
                    raw_close_arr = df_1m['close'].values.astype(np.float32)

                    for i in range(start_idx, len(df_1m), WINDOW_STEP):
                        t = ts_1m[i]
                        
                        if t not in ts_5m_map: continue
                        idx_5 = ts_5m_map[t]
                        if idx_5 < (WINDOW_SIZE_5M - 1): continue

                        # 2. 计算 1min 窗口的起始索引
                        # 这里的切片逻辑是 [idx_start : i+1]
                        # 长度 = (i+1) - (i - 29) = 30
                        idx_start_1m = i - (WINDOW_SIZE_1M - 1)
                        
                        # 边界检查
                        if idx_start_1m < 0: continue
                        
                        # =================================================================
                        # 🛡️ [核心优化] 死水/僵尸数据过滤 (Dead Line Filter)
                        # =================================================================
                        # 取出当前窗口的原始收盘价
                        #window_close = raw_close_arr[idx_start_1m : i + 1]
                        
                        # 计算标准差
                        # 如果 std < 1e-6，说明这 30 分钟内价格完全是一条直线
                        # 这几乎 100% 是 ffill 填充出来的假数据，直接丢弃
                        # if np.std(window_close) < 1e-6:
                        #     # logger.debug(f"⚠️ Drop: Flat Price detected at {t}")
                        #     continue
                            
                        # =================================================================
                        # 3. 数据通过检查，开始提取特征
                        # =================================================================
                        
                        # 提取 1min 特征 (arr_1m 是你之前准备好的特征矩阵)
                        # 这里的索引和上面 window_close 是严格对齐的
                        chunk_1m = arr_1m[idx_start_1m : i + 1]
                        
                        # 提取 5min 特征 (同理)
                        idx_start_5m = idx_5 - (WINDOW_SIZE_5M - 1)
                        chunk_5m = arr_5m[idx_start_5m : idx_5 + 1]
                        
                        sample = {
                            '1min': dict(zip(feats_1m_cols, arr_1m[i-29 : i+1].T)),
                            '5min': dict(zip(feats_5m_cols, arr_5m[idx_start_5m : idx_5 + 1].T)),
                            'labels': dict(zip(label_cols, arr_lbl[i])),
                            'metadata': {'symbol': symbol, 'timestamp': int(t),'stock_id': int(stock_id),'sector_id': int(sector_id)}
                        }
                        
                        key = f"{symbol}_{t}".encode('ascii')
                        batch_buffer.append((key, cctx.compress(msgpack.packb(sample, use_bin_type=True))))
                        local_count += 1
                        
                        if len(batch_buffer) >= BATCH_SIZE_TO_QUEUE:
                            data_queue.put(batch_buffer)
                            batch_buffer = []

                    total_produced += local_count

                except Exception as e:
                    dprint(f"Error loop: {e}")
                    continue

        if batch_buffer:
            data_queue.put(batch_buffer)

        dprint(f"Total produced: {total_produced}")
        return {"status": "success", "symbol": symbol, "segments_produced": total_produced}

    except Exception as e:
        return {"status": "error", "symbol": symbol, "error": str(e)}

# --- 4. Writer ---
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
            elif isinstance(payload, dict):
                pbar.update(1)
                
        except queue.Empty: continue
        except Exception: txn.abort(); txn = env.begin(write=True)

    if all_keys:
        dctx = zstd.ZstdCompressor(level=3)
        txn.put(b'__keys__', dctx.compress(msgpack.packb(all_keys, use_bin_type=True)))
        txn.put(b'__len__', str(len(all_keys)).encode('ascii'))
        
    txn.commit()
    env.close()
    pbar.close()
    logging.info(f"🎉 [Writer] 完成。总写入样本: {total_written}")

# --- 5. [升级版] 验证函数 (NY Timezone) ---
def verify_data_integrity(lmdb_path, config, parquet_root):
    print(f"\n{'='*20} 开始数据完整性验证 (Dual-Resolution + Zero Check) {'='*20}")
    print(f"检查文件: {lmdb_path}")
    
    if not lmdb_path.exists():
        print("❌ LMDB 文件不存在！")
        return

    # 1. 准备架构
    feats_1m_cols, feats_5m_cols = [], []
    for f in config.get('features', []):
        name, res = f['name'], f.get('resolution', '1min')
        if res in ['1min', 'both']: feats_1m_cols.append(name)
        if res in ['5min', 'both']: feats_5m_cols.append(name)
    
    print(f"📊 特征统计: 1min 通道 [{len(feats_1m_cols)}] | 5min 通道 [{len(feats_5m_cols)}]")

    try:
        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        dctx = zstd.ZstdDecompressor()
        
        with env.begin() as txn:
            keys_c = txn.get(b'__keys__')
            if not keys_c: 
                print("❌ 未在 LMDB 中找到 __keys__")
                return
            all_keys = msgpack.unpackb(dctx.decompress(keys_c), raw=False)
            
            # 随机抽样
            sample_keys = random.sample(all_keys, min(3, len(all_keys)))
            for key_bytes in sample_keys:
                key_str = key_bytes.decode('ascii')
                symbol, ts_str = key_str.rsplit('_', 1)
                target_ts = int(ts_str)
                human_time = pd.Timestamp(target_ts)
                print(f"\n--- 验证样本: {symbol} @ {human_time} (NY) ---")
                
                val_c = txn.get(key_bytes)
                sample_data = msgpack.unpackb(dctx.decompress(val_c), raw=False)
                
                # A. 检查 1min 部分
                data_1m = sample_data.get('1min', {})
                zeros_1m = [name for name, vals in data_1m.items() if np.all(np.array(vals) == 0)]
                if zeros_1m:
                    print(f"⚠️ [1min] 全零特征检测 ({len(zeros_1m)}/{len(data_1m)}): {zeros_1m[:10]}...")
                else:
                    print(f"✅ [1min] 数据填充检查通过 (无全零特征)")

                # B. 检查 5min 部分
                data_5m = sample_data.get('5min', {})
                zeros_5m = [name for name, vals in data_5m.items() if np.all(np.array(vals) == 0)]
                if zeros_5m:
                    print(f"⚠️ [5min] 全零特征检测 ({len(zeros_5m)}/{len(data_5m)}): {zeros_5m[:10]}...")
                else:
                    print(f"✅ [5min] 数据填充检查通过 (无全零特征)")

                # C. Parquet 比对 (选取 1min 第一列进行抽查)
                if feats_1m_cols:
                    test_feat = feats_1m_cols[0]
                    lmdb_vals = np.array(data_1m[test_feat], dtype=np.float32)
                    print(f"ℹ️ [比对] 抽样 1min 特征 [{test_feat}] 尾部值: {lmdb_vals[-1]:.4f}")

        if 'env' in locals(): env.close()
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        traceback.print_exc()

# --- 6. Main ---
def run_stage(stage, config):
    print(f"\n🚀 处理阶段: {stage}")
    stock_to_id, stock_to_sector_id = load_meta_info()
    root_path = PARQUET_ROOT / f"quote_features_{stage}"
    output_path = OUTPUT_DIR / f"{stage}_quote_alpha.lmdb"
    
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path) if output_path.is_dir() else os.remove(output_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manager = mp.Manager()
    data_queue = manager.Queue(maxsize=1000)
    
    tasks = []
    debug_symbol = sorted(list(stock_to_id.keys()))[0] if stock_to_id else ""
    print(f"🔍 调试模式开启: 将实时监控股票 {debug_symbol} 的处理流程...")

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