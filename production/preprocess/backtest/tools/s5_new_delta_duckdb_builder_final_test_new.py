#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s5_parallel_pipeline_direct_source.py
描述: [New Delta] 高性能并行推理流水线 (直连源头版)
修改点:
    1. Step 4 弃用 H5，改用 DuckDB 直接查询原始 Parquet 源文件。
    2. 自动适配源文件字段 (expiration_date -> expiration 等)。
    3. 集成 "无限兜底" 分桶逻辑，解决覆盖率为 0 的问题。
"""

import os
import sys
import json
import logging
import duckdb
import pandas as pd
import numpy as np
import lmdb
import msgpack
import msgpack_numpy
import zstandard as zstd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import warnings
 
import concurrent.futures
import sqlite3
import shutil
from torch.utils.data import Dataset, DataLoader

msgpack_numpy.patch()
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
BASE_PATH = Path.home() / 'notebook/train'
DATA_ROOT = Path(Path.home() / "data")
LMDB_ROOT = Path('/mnt/s990/data/h5_unified_overlap_id')
DB_PATH = Path.home() / "notebook/stocks.db"

stage = "val"

LMDB_FAST_PATH = LMDB_ROOT / f'{stage}_fast_channel_features.lmdb' 
LMDB_SLOW_PATH = LMDB_ROOT / f'{stage}_quote_alpha.lmdb'

# [修改点] 源数据目录 (Parquet)
OPTION_SOURCE_DIR = DATA_ROOT / "spnq_options_monthly_iv"

CONFIG_FAST = BASE_PATH / 'fast_feature.json'
CONFIG_SLOW = BASE_PATH / 'slow_feature.json'
CKPT_SLOW = BASE_PATH / "quant_project/checkpoints_advanced_alpha/advanced_alpha_best.pth"
CKPT_FAST = BASE_PATH / "quant_project/checkpoints_fast_final/fast_final_best.pth"

# 输出目录
TEMP_DIR = LMDB_ROOT / f'temp_inference_stage_{stage}'
TEMP_FAST_DIR = TEMP_DIR / 'fast'
TEMP_SLOW_DIR = TEMP_DIR / 'slow'
TEMP_MERGED_DIR = TEMP_DIR / f'merged_signals_no_opt_{stage}'  
FINAL_OUTPUT_DIR = LMDB_ROOT / f'rl_feed_parquet_new_delta_{stage}'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_BATCH_CHUNK = 8192 

# --- 并行度设置 ---
NUM_LOADER_WORKERS = 12 
NUM_WRITER_THREADS = 8 
MAX_MERGE_WORKERS = 16

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParallelPipe")


# 导入模型 (运行时可选)
sys.path.append(os.getcwd())
try:
    from trading_tft_stock_embed import AdvancedAlphaNet, UnifiedLMDBDataset, collate_fn as train_collate_fn
    from train_fast_channel_microstructure  import FastMicrostructureModel, UnifiedFastDataset
except ImportError: pass

# ================= 通用辅助 =================
def load_db_info(db_path):
    # (保持不变)
    try:
        if not os.path.exists(db_path): return {'stock': 9000, 'sector': 200}, {}
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT max(id) FROM stocks_us")
        max_sid = c.fetchone()[0]
        c.execute("SELECT count(distinct sector) FROM stocks_us")
        max_sec = c.fetchone()[0]
        c.execute("SELECT symbol, sector FROM stocks_us WHERE sector IS NOT NULL")
        rows = c.fetchall()
        unique_sectors = sorted(list(set([r[1] for r in rows])))
        sec_to_id = {s: i for i, s in enumerate(unique_sectors)}
        stock_sector_map = {sym: sec_to_id.get(sec, 0) for sym, sec in rows}
        conn.close()
        return {'stock': (max_sid or 5000) + 100, 'sector': (max_sec or 100) + 10}, stock_sector_map
    except: return {'stock': 9000, 'sector': 200}, {}

def scan_lmdb_keys(lmdb_path):
    # (保持不变)
    if not lmdb_path.exists(): return []
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    symbols = set()
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, _ in cursor:
            try:
                k_str = k.decode('ascii')
                if not k_str[0].isalnum() or k_str.startswith("__"): continue
                sym = k_str.split('_')[0]
                symbols.add(sym)
            except: pass
    env.close()
    return sorted(list(symbols))

# ================= Dataset 定义 =================
class SymbolLMDBDataset(Dataset):
    def __init__(self, lmdb_path, symbols, feature_cfg, mode='fast', db_info=None):
        self.lmdb_path = str(lmdb_path)
        self.symbols = symbols
        self.cfg = feature_cfg
        self.mode = mode
        self.db_info = db_info
        
        if mode == 'fast':
            self.stock_cols = [f['name'] for f in self.cfg['features'] if f.get('calc') != 'options']
            self.opt_cols = [f['name'] for f in self.cfg['features'] if f.get('calc') == 'options']
        else:
            # === [修改点 1] Slow 模式下的特征预处理 (分流) ===
            self.caps, self.stock_sector_map = db_info
            self.slow_stock_map = []
            self.slow_option_map = []
            
            # 预先计算好每个特征在最终矩阵中的位置
            idx_stock = 0
            idx_option = 0
            
            for f in self.cfg['features']:
                name = f['name']
                # 跳过静态特征
                if name in ['stock_id', 'sector_id', 'day_of_week']: continue
                
                res = f.get('resolution', '1min')
                is_option = name.startswith('options_')
                
                # 记录: (特征名, 源数据resolution, 目标矩阵列索引)
                if is_option:
                    self.slow_option_map.append({'name': name, 'res': res, 'idx': idx_option})
                    idx_option += 1
                else:
                    self.slow_stock_map.append({'name': name, 'res': res, 'idx': idx_stock})
                    idx_stock += 1
            
            self.n_stock = idx_stock
            self.n_option = idx_option

    def __len__(self): return len(self.symbols)

    def __getitem__(self, idx):
        symbol = self.symbols[idx]
        # 打开 LMDB 环境 (Read-Only, No Lock)
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        dctx = zstd.ZstdDecompressor()
        data_packet = {}
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                # 定位到该股票的第一个 Key
                if cursor.set_range(f"{symbol}_".encode('ascii')):
                    
                    # ==========================================
                    # Mode: FAST (保持原有逻辑不变)
                    # ==========================================
                    if self.mode == 'fast':
                        ts_list, xs_list, xo_list, close_list = [], [], [], []
                        for key, val in cursor:
                            if not key.decode('ascii').startswith(f"{symbol}_"): break
                            try:
                                item = msgpack.unpackb(dctx.decompress(val), raw=False)
                                feats = item.get('features', {})
                                meta = item.get('metadata', {})
                                ts = int(key.decode('ascii').split('_')[-1])
                                
                                xs = np.stack([feats.get(n, np.zeros(10)) for n in self.stock_cols], axis=1)
                                xo = np.stack([feats.get(n, np.zeros(10)) for n in self.opt_cols], axis=1)
                                
                                ts_list.append(ts)
                                xs_list.append(xs)
                                xo_list.append(xo)
                                close_val = meta.get('close', feats.get('close', [0])[-1] if isinstance(feats.get('close'), list) else 0.0)
                                close_list.append(float(close_val))
                            except: continue
                            
                        if ts_list:
                            data_packet = {
                                'symbol': symbol, 
                                'ts': np.array(ts_list, dtype=np.int64), 
                                'xs': np.stack(xs_list).astype(np.float32), 
                                'xo': np.stack(xo_list).astype(np.float32), 
                                'close': np.array(close_list, dtype=np.float32)
                            }

                    # ==========================================
                    # Mode: SLOW (双流拆分 + 全零检测)
                    # ==========================================
                    else: 
                        ts_list = []
                        x_stock_list = []
                        x_option_list = []
                        s_list, sec_list, dow_list = [], [], []
                        
                        fallback_sector = self.stock_sector_map.get(symbol, 0)
                        
                        # 遍历该股票的所有时间步
                        for key, val in cursor:
                            if not key.decode('ascii').startswith(f"{symbol}_"): break
                            try:
                                item = msgpack.unpackb(dctx.decompress(val), raw=False)
                                ts = int(key.decode('ascii').split('_')[-1])
                                
                                # 获取源数据字典
                                src_1m = item.get('1min', {})
                                src_5m = item.get('5min', {}) 
                                
                                # --- 1. 构建 Stock 矩阵 (30, F_stock) ---
                                arr_stock = np.zeros((30, self.n_stock), dtype=np.float32)
                                for m in self.slow_stock_map:
                                    # 根据 resolution 选择源
                                    src = src_5m if m['res'] == '5min' else src_1m
                                    raw_data = src.get(m['name'])
                                    if raw_data is not None:
                                        v = np.array(raw_data, dtype=np.float32)
                                        l = min(len(v), 30)
                                        if l > 0: arr_stock[-l:, m['idx']] = v[-l:]
                                
                                # --- 2. 构建 Option 矩阵 (30, F_option) ---
                                arr_option = np.zeros((30, self.n_option), dtype=np.float32)
                                for m in self.slow_option_map:
                                    src = src_5m if m['res'] == '5min' else src_1m
                                    raw_data = src.get(m['name'])
                                    if raw_data is not None:
                                        v = np.array(raw_data, dtype=np.float32)
                                        l = min(len(v), 30)
                                        if l > 0: arr_option[-l:, m['idx']] = v[-l:]
                                
                                x_stock_list.append(arr_stock)
                                x_option_list.append(arr_option)
                                ts_list.append(ts)
                                
                                # Metadata
                                meta = item.get('metadata', {})
                                s_list.append(int(meta.get('stock_id', 0)))
                                sec_list.append(int(meta.get('sector_id', fallback_sector)))
                                try:
                                    dow = pd.Timestamp(ts, unit='ns').dayofweek
                                except: dow = 0
                                dow_list.append(dow)
                                
                            except Exception as e: 
                                continue
                                
                        if ts_list:
                            # 转换为 Numpy 数组
                            final_x_option = np.stack(x_option_list).astype(np.float32)
                            final_x_stock = np.stack(x_stock_list).astype(np.float32)
                            
                            # === [关键检测] 检查 x_option 是否全为 0 ===
                            # 如果期权矩阵全为 0，且该股票本应有期权特征 (n_option > 0)
                            if self.n_option > 0 and np.count_nonzero(final_x_option) == 0:
                                # 仅打印一次警告，防止刷屏
                                print(f"⚠️ [WARNING] {symbol} Option Matrix is ALL ZEROS! (Check s0 generation)")
                            
                            data_packet = {
                                'symbol': symbol, 
                                'ts': np.array(ts_list, dtype=np.int64), 
                                'x_stock': final_x_stock, 
                                'x_option': final_x_option,
                                's': np.array(s_list, dtype=np.int64), 
                                'sec': np.array(sec_list, dtype=np.int64), 
                                'dow': np.array(dow_list, dtype=np.int64)
                            }

        except Exception as e: pass
        finally: env.close()
        return data_packet
def custom_collate(batch): return batch[0]

# ================= 写入线程 (保持不变) =================
def write_fast_result(args):
    symbol, ts, vol, evt, close, spy, qqq, out_dir = args
    df = pd.DataFrame({'timestamp': ts, 'fast_vol': vol, 'fast_evt': evt, 'close': close, 'spy_roc_5min': spy, 'qqq_roc_5min': qqq})
    df.to_parquet(out_dir / f"{symbol}.parquet", compression='ZSTD')
    return symbol

def write_slow_result(args):
    """
    [修改版] 增加 EMA 平滑，提升 IC 和 稳定性
    """
    symbol, ts, score, out_dir = args
    
    # 1. 构造 DataFrame
    df = pd.DataFrame({'timestamp': ts, 'raw_score': score})
    
    df['alpha_score'] = df['raw_score'] # 直接使用原始分
    
    # 3. [可选] 鲁棒性处理 (Winsorize / Clip)
    # 防止某些极端值 (如 > 5.0) 干扰 RL 训练
    # df['alpha_score'] = df['alpha_score'].clip(-3.0, 3.0)
    
    # 4. 落盘 (只保留处理后的 alpha_score)
    final_df = df[['timestamp', 'alpha_score']]
    final_df.to_parquet(out_dir / f"{symbol}.parquet", compression='ZSTD')
    return symbol

 


# ==============================================================================
# [新增] 专用于 Fast Channel 推理的 Dataset (继承复用)
# ==============================================================================
class InferenceFastDataset(UnifiedFastDataset):
    """
    继承自 UnifiedFastDataset。
    目标: 
    1. 复用训练时的特征映射 (feat_map_stock/opt) 和预处理逻辑。
    2. 强制对 keys 排序，以便流式推理。
    3. 额外返回 symbol, ts, close 用于后续处理。
    """
    def __init__(self, db_path, config):
        # 调用父类初始化 (建立 feat_map, dctx 等)
        # 注意: train 脚本中 seq_len 默认为 10，这里显式传入以防万一
        super().__init__(db_path, config, stage='inference', seq_len=10)
        
        # [关键] 强制排序，保证数据按 Symbol 聚类
        logger.info("⏳ Sorting Fast Dataset keys for sequential inference...")
        self.keys = sorted(self.keys)
        logger.info(f"✅ Fast Keys sorted. Total: {len(self.keys)}")

    def __getitem__(self, idx):
        # 这里的逻辑主要参考 UnifiedFastDataset.__getitem__，
        # 但我们需要在一个 pass 里拿到 metadata (close)，所以这里选择 Copy-Modify 逻辑
        # 而不是调用 super().__getitem__ 再解压一次 (性能考虑)。

        self._init_env()
        key_bytes = self.keys[idx]
        val_bytes = self.txn.get(key_bytes)
        if val_bytes is None: return None
        
        try:
            data = msgpack.unpackb(self.dctx.decompress(val_bytes), raw=False)
        except: return None
        
        features_dict = data.get('features', {})
        if not features_dict: return None
        
        # --- 1. 复用父类的 feat_map 进行特征提取 ---
        # Stock Features
        x_stock = np.zeros((self.seq_len, len(self.feat_map_stock)), dtype=np.float32)
        for i, name in enumerate(self.feat_map_stock):
            arr = features_dict.get(name)
            if arr is not None:
                L = len(arr)
                if L >= self.seq_len: x_stock[:, i] = arr[-self.seq_len:]
                else: x_stock[-L:, i] = arr
        
        # Option Features
        x_opt = np.zeros((self.seq_len, len(self.feat_map_opt)), dtype=np.float32)
        for i, name in enumerate(self.feat_map_opt):
            arr = features_dict.get(name)
            if arr is not None:
                L = len(arr)
                if L >= self.seq_len: x_opt[:, i] = arr[-self.seq_len:]
                else: x_opt[-L:, i] = arr

        # NaN 处理 (保持与训练一致)
        x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=0.0, neginf=0.0)
        x_opt = np.nan_to_num(x_opt, nan=0.0, posinf=0.0, neginf=0.0)
        # ----------------------------------------

        # --- 2. 提取推理所需的额外信息 ---
        # Key 格式: b'SYMBOL_TIMESTAMP'
        key_str = key_bytes.decode('ascii')
        symbol, ts_str = key_str.split('_')
        ts = int(ts_str)
        
        # 提取 Close 用于计算 Volatility
        meta = data.get('metadata', {})
        close = meta.get('close', 0.0)
        if close == 0.0:
            # 兜底：尝试从特征中获取
            c_feat = features_dict.get('close') or features_dict.get('Close')
            if c_feat: close = c_feat[-1]

        return {
            'x_stock': torch.from_numpy(x_stock),
            'x_opt': torch.from_numpy(x_opt),
            'ts': ts,
            'symbol': symbol,
            'close': float(close)
        }

def inference_fast_collate_fn(batch):
    batch = [b for b in batch if b]
    if not batch: return None
    
    x_stk = torch.stack([b['x_stock'] for b in batch])
    x_opt = torch.stack([b['x_opt'] for b in batch])
    ts = [b['ts'] for b in batch]
    sym = [b['symbol'] for b in batch]
    close = [b['close'] for b in batch] # Keep as list for now
    
    return x_stk, x_opt, ts, sym, close

# ==============================================================================
# [重写] Step 1: Fast Channel (Batch Infer + Streaming Stats)
# ==============================================================================
def run_step_1_fast():
    logger.info("="*60)
    logger.info("🚀 Step 1: Fast Channel (Unified Dataset + Batch Infer)")
    logger.info("="*60)
    
    if TEMP_FAST_DIR.exists(): shutil.rmtree(TEMP_FAST_DIR)
    TEMP_FAST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Config & Model
    with open(CONFIG_FAST) as f: cfg = json.load(f)
    
    # 从 Dataset 初始化一次以获取 feature map 长度，用于初始化模型
    # 或者直接用 cfg 计算。为了稳妥，我们实例化 Dataset。
    dataset = InferenceFastDataset(str(LMDB_FAST_PATH), cfg)
    
    n_stock = len(dataset.feat_map_stock)
    n_opt = len(dataset.feat_map_opt)
    logger.info(f"Model Input Dims: Stock={n_stock}, Option={n_opt}")
    
    model = FastMicrostructureModel(n_stock, n_opt).to(DEVICE)
    if CKPT_FAST.exists():
        logger.info(f"🔄 Loading Fast Checkpoint: {CKPT_FAST}")
        st = torch.load(CKPT_FAST, map_location=DEVICE, weights_only=False) # 注意 weights_only=False
        model.load_state_dict(st.get('state_dict', st), strict=False)
    model.eval()

    # [新增] 获取 fast_vol 在特征维度中的索引
    feat_cols = cfg.get('features', [])
    print("Feature columns:", [f['name'] for f in feat_cols])
    col_names = [f['name'] for f in feat_cols]
    if 'fast_vol' not in col_names:
        raise ValueError("fast_vol column not found in fast_feature.json")
    vol_idx = col_names.index('fast_vol')
    spys_idx = col_names.index('spy_roc_5min')
    qqq_idx = col_names.index('qqq_roc_5min')

    
    # 2. DataLoader
    BATCH_SIZE = 4096
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 必须 False，保证有序
        num_workers=NUM_LOADER_WORKERS,
        collate_fn=inference_fast_collate_fn,
        pin_memory=True
    )
    
    # 3. 推理与统计循环
    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITER_THREADS)
    
    current_symbol = None
    buffer_ts = []
    buffer_close = []
    buffer_evt = []
    buffer_vol = [] # [新增]
    buffer_spy = []  # [新增]
    buffer_qqq = []  # [新增]
    
    def flush_buffer(sym, tss, closes, evts, vols, spys, qqqs): # [修改] 增加 vols 参数
        if not sym or not tss: return
        
        # [删除] 删掉原有的 s_close = pd.Series(closes) ... vol_arr = ... 这一整段
            
        # [修改] 写入 Parquet (直接传入 vols)
        write_executor.submit(
            write_fast_result,
            (sym, np.array(tss, dtype=np.int64), np.array(vols, dtype=np.float32), np.array(evts, dtype=np.float32), np.array(closes, dtype=np.float32), np.array(spys, dtype=np.float32), np.array(qqqs, dtype=np.float32), TEMP_FAST_DIR)
        )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Fast Batch Infer"):
            if not batch: continue
            
            x_stk, x_opt, ts_list, sym_list, close_list = batch
            x_stk, x_opt = x_stk.to(DEVICE), x_opt.to(DEVICE)
            
            # A. 模型推理
            preds, _ = model(x_stk, x_opt)
            evt_scores = torch.sigmoid(preds['event']).cpu().numpy().flatten()

            # B. [新增] 从 x_stk 中提取 fast_vol
            # x_stk 形状通常是 [Batch, SeqLen, FeatDim]，取最后一个时点 [-1]
            stk_vols = x_stk[:, -1, vol_idx].cpu().numpy().flatten()
            stk_spys = x_stk[:, -1, spys_idx].cpu().numpy().flatten()
            stk_qqqs = x_stk[:, -1, qqq_idx].cpu().numpy().flatten()

            # --- Buffer & Group by Symbol ---
            for i, sym in enumerate(sym_list):
                if sym != current_symbol:
                    # [修改] 增加 buffer_vol 参数
                    flush_buffer(current_symbol, buffer_ts, buffer_close, buffer_evt, buffer_vol, buffer_spy, buffer_qqq)
                    current_symbol = sym
                    buffer_ts, buffer_close, buffer_evt, buffer_vol, buffer_spy, buffer_qqq = [], [], [], [], [], [] # [重置]
                
                buffer_ts.append(ts_list[i])
                buffer_close.append(close_list[i])
                buffer_evt.append(evt_scores[i])
                buffer_vol.append(stk_vols[i]) # [新增] 直接存入提取的值
                buffer_spy.append(stk_spys[i])  # [新增] 存入提取的 spy 值
                buffer_qqq.append(stk_qqqs[i])  # [新增] 存入提取的 qqq 值
                
    # [修改] 最后一次 flush 也要带上 buffer_vol, buffer_spy, buffer_qqq
    flush_buffer(current_symbol, buffer_ts, buffer_close, buffer_evt, buffer_vol, buffer_spy, buffer_qqq)
    
    write_executor.shutdown(wait=True)
    logger.info("✅ Step 1 Done. Processed all keys using UnifiedFastDataset.")


# ================= Step 1.5: 快通道标准化 (Z-Score) =================
# ================= Step 1.5: Normalize Fast (Vol & Evt) =================
def run_step_1_5_normalize_fast():
    logger.info("="*60)
    logger.info("⚡ Step 1.5: Normalize Fast Channel (Z-Score for BOTH Vol & Evt)")
    logger.info("="*60)
    
    files = list(TEMP_FAST_DIR.glob("*.parquet"))
    if not files: return

    # --- 1. 全局统计 ---
    all_vols = []
    all_evts = [] # [新增] 收集模型分
    
    for p in tqdm(files, desc="Scanning Fast Stats"):
        try:
            df = pd.read_parquet(p, columns=['fast_vol', 'fast_evt'])
            
            # --- 修复 fast_vol 的统计污染 ---
            v_vals = df['fast_vol'].values
            # [核心修复] 过滤掉精确为 0.0 的无效填充值
            v_vals = v_vals[(np.isfinite(v_vals)) & (v_vals != 0.0)]
            if len(v_vals) > 0:
                all_vols.append(v_vals)
            
            # --- 修复 fast_evt 的统计污染 ---
            e_vals = df['fast_evt'].values
            e_vals = e_vals[(np.isfinite(e_vals)) & (e_vals != 0.0)]
            if len(e_vals) > 0:
                all_evts.append(e_vals)
        except: continue
            
    if not all_vols: return

    # 统计 Fast Vol
    full_vol = np.concatenate(all_vols)
    vol_mean = np.mean(full_vol)
    vol_std = np.std(full_vol)
    
    # [新增] 统计 Fast Evt
    full_evt = np.concatenate(all_evts)
    evt_mean = np.mean(full_evt)
    evt_std = np.std(full_evt)
    
    logger.info(f"📊 Global Statistics:")
    logger.info(f"   [Rule] fast_vol: Mean={vol_mean:.6f}, Std={vol_std:.6f}")
    logger.info(f"   [Model] fast_evt: Mean={evt_mean:.6f}, Std={evt_std:.6f}")

    # --- 2. 原地更新 ---
    def process_file(p):
        try:
            df = pd.read_parquet(p)
            # 归一化 Vol
            if vol_std > 1e-6:
                df['fast_vol'] = np.clip((df['fast_vol'] - vol_mean) / vol_std, -5.0, 5.0).astype(np.float32)
            
            # [新增] 归一化 Evt
            if evt_std > 1e-6:
                df['fast_evt'] = np.clip((df['fast_evt'] - evt_mean) / evt_std, -5.0, 5.0).astype(np.float32)
            
            df.to_parquet(p, compression='ZSTD')
            return "OK"
        except Exception as e: return str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITER_THREADS) as executor:
        futures = {executor.submit(process_file, p): p for p in files}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Normalizing"):
            pass
            
    logger.info(f"✅ Fast Channel Normalized (Vol & Evt).")

# ==============================================================================
# [新增] 专用于 Slow Channel 推理的 Dataset (复用训练逻辑)
# ==============================================================================
class InferenceDataset(UnifiedLMDBDataset):
    """
    继承自 UnifiedLMDBDataset，完全复用其数据处理逻辑 (Upsampling, Features Map 等)。
    唯一区别：__getitem__ 额外返回 symbol 字符串，用于生成文件名。
    """
    def __init__(self, db_path, config):
        # 初始化父类
        super().__init__(db_path, config, stage='inference')
        
        # [关键优化] 强制对 keys 进行排序！
        # LMDB keys 格式为 b'SYMBOL_TIMESTAMP'。
        # 排序后，相同 Symbol 的数据会连续排列 (e.g., AAPL_100, AAPL_101, ..., MSFT_100...)
        # 这允许我们在推理循环中进行 "Streaming Write"，无需一次性将所有结果读入内存。
        logger.info("⏳ Sorting dataset keys for sequential inference...")
        self.keys = sorted(self.keys)
        logger.info(f"✅ Keys sorted. Total samples: {len(self.keys)}")

    def __getitem__(self, idx):
        # 1. 调用父类获取处理好的张量
        # 父类返回: (x_stock, x_option, static, tgt, ts)
        data = super().__getitem__(idx)
        if data is None: return None
        
        # 2. 从 Key 中解析 Symbol
        # self.keys[idx] 是 bytes, e.g., b'AAPL_1623456789000000000'
        raw_key = self.keys[idx]
        symbol_str = raw_key.decode('ascii').split('_')[0]
        
        # 3. 返回扩展后的 Tuple
        # (x_stock, x_option, static, ts, symbol)
        # 注意：我们丢弃了 tgt (target)，因为推理时不需要 label
        return data[0], data[1], data[2], data[4], symbol_str

def inference_collate_fn(batch):
    """
    适配 InferenceDataset 的 Collate 函数
    Batch 结构: [(x_stk, x_opt, static, ts, symbol), ...]
    """
    batch = [b for b in batch if b]
    if not batch: return None
    
    # 堆叠 Tensor
    x_stk = torch.stack([torch.from_numpy(b[0]) for b in batch])
    x_opt = torch.stack([torch.from_numpy(b[1]) for b in batch])
    
    # 堆叠 Static (Dict of Tensors)
    st_k = batch[0][2].keys()
    s = {k: torch.tensor([b[2][k] for b in batch]) for k in st_k}
    
    # 收集 Timestamp 和 Symbol
    ts = [b[3] for b in batch]
    symbols = [b[4] for b in batch]
    
    return x_stk, x_opt, s, ts, symbols

# ================= Step 2: Slow Channel =================
# ==============================================================================
# [重写] Step 2: Slow Channel (使用 Unified Dataset + 大 Batch)
# ==============================================================================
def run_step_2_slow():
    logger.info("="*60)
    logger.info("🚀 Step 2: Slow Channel (Unified Dataset + Batch Infer)")
    logger.info("="*60)
    
    if TEMP_SLOW_DIR.exists(): shutil.rmtree(TEMP_SLOW_DIR)
    TEMP_SLOW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Config & Model
    db_info = load_db_info(DB_PATH)
    caps, _ = db_info
    caps['dow'] = 7
    with open(CONFIG_SLOW) as f: cfg = json.load(f)
    
    model = AdvancedAlphaNet(cfg, caps).to(DEVICE)
    if CKPT_SLOW.exists():
        logger.info(f"🔄 Loading Checkpoint: {CKPT_SLOW}")
        st = torch.load(CKPT_SLOW, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st.get('state_dict', st), strict=False)
    model.eval()
    
    # 2. Dataset (使用我们定义的 InferenceDataset)
    # 注意：这里直接传入 LMDB 路径和配置
    dataset = InferenceDataset(str(LMDB_SLOW_PATH), cfg)
    
    # 3. DataLoader (大 Batch Size!)
    # shuffle=False 非常关键，配合 dataset.keys.sort() 保证数据按 Symbol 顺序涌入
    BATCH_SIZE = 4096  # 建议设大，利用 GPU 并行能力，且稳定 BN
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_LOADER_WORKERS, 
        collate_fn=inference_collate_fn,
        pin_memory=True
    )
    
    # 4. 推理循环 (Streaming Buffer)
    current_symbol = None
    buffer_ts = []
    buffer_scores = []
    
    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITER_THREADS)
    futures = []
    
    def flush_buffer(sym, tss, scs):
        """将缓存写入 Parquet"""
        if not sym or not tss: return
        # 提交给线程池写入
        futures.append(write_executor.submit(
            write_slow_result,
            (sym, np.array(tss, dtype=np.int64), np.array(scs, dtype=np.float32), TEMP_SLOW_DIR)
        ))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Slow Batch Infer"):
            if not batch: continue
            
            x_stk, x_opt, s, ts_list, sym_list = batch
            x_stk, x_opt = x_stk.to(DEVICE), x_opt.to(DEVICE)
            s = {k: v.to(DEVICE) for k, v in s.items()}
            
            # 模型推理
            out = model(x_stk, x_opt, s)
            scores = out['rank_score'].cpu().numpy().flatten()
            
            # 处理结果 (按 Symbol 分组)
            # 由于 Loader 是有序的，batch 内部 symbol 可能会变，但只会变有限次
            for i, sym in enumerate(sym_list):
                if sym != current_symbol:
                    # 符号切换：Flush 上一个符号的 buffer
                    flush_buffer(current_symbol, buffer_ts, buffer_scores)
                    # 重置 buffer
                    current_symbol = sym
                    buffer_ts = []
                    buffer_scores = []
                
                buffer_ts.append(ts_list[i])
                buffer_scores.append(scores[i])
    
    # 循环结束，Flush 最后一个 Symbol
    flush_buffer(current_symbol, buffer_ts, buffer_scores)
    
    write_executor.shutdown(wait=True)
    logger.info(f"✅ Step 2 Done. Processed all keys in Unified Dataset.")


# ================= Step 2.5: 中间层标准化 (Z-Score) =================
def run_step_2_5_normalize_slow():
    logger.info("="*60)
    logger.info("⚖️  Step 2.5: Normalize Slow Channel (Z-Score)")
    logger.info("="*60)
    
    files = list(TEMP_SLOW_DIR.glob("*.parquet"))
    if not files:
        logger.warning("⚠️  No slow channel files found to normalize.")
        return

    # --- 第一步：全局统计 (Global Stats Calculation) ---
    logger.info(f"1. Scanning {len(files)} files to compute global stats...")
    
    all_scores = []
    # 为了防止内存爆炸，我们使用蓄水池抽样或分批读取
    # 鉴于 Slow Channel 数据量较小（每只股票几万行），抽样 20% 或读取全部通常是可以的
    # 这里我们读取全部，因为准确的 Mean/Std 对 RL 很重要
    
    for p in tqdm(files, desc="Scanning Stats"):
        try:
            # 只读 alpha_score 列加速
            df = pd.read_parquet(p, columns=['alpha_score'])
            vals = df['alpha_score'].values
            # 移除 NaN 和 Inf
            vals = vals[np.isfinite(vals)]
            all_scores.append(vals)
        except Exception:
            continue
            
    if not all_scores:
        logger.error("❌ No valid data found for normalization.")
        return

    # 合并大数组计算
    full_data = np.concatenate(all_scores)
    
    global_mean = np.mean(full_data)
    global_std = np.std(full_data)
    
    logger.info(f"📊 Global Statistics for 'alpha_score':")
    logger.info(f"   Mean: {global_mean:.6f}")
    logger.info(f"   Std : {global_std:.6f}")
    
    if global_std < 1e-6:
        logger.warning("⚠️  Std is close to zero! Skipping normalization to avoid NaN.")
        return

    # --- 第二步：原地更新 (In-Place Update) ---
    logger.info(f"2. Applying Z-Score Normalization to {len(files)} files...")
    
    def process_file(p):
        try:
            df = pd.read_parquet(p)
            if 'alpha_score' not in df.columns: return "SKIP"
            
            # Z-Score 公式
            # x_new = (x - mean) / std
            original_vals = df['alpha_score'].values
            norm_vals = (original_vals - global_mean) / global_std
            
            # [关键] 钳位 (Clipping)
            # 防止极少数异常值 (Outliers) 在 RL 中产生过大的梯度
            # 这里的 [-5, 5] 对应 5 个标准差，保留了绝大部分信息
            norm_vals = np.clip(norm_vals, -5.0, 5.0)
            
            df['alpha_score'] = norm_vals.astype(np.float32)
            
            # 覆盖写入
            df.to_parquet(p, compression='ZSTD')
            return "OK"
        except Exception as e:
            return f"ERR: {e}"

    # 并行处理写入
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITER_THREADS) as executor:
        futures = {executor.submit(process_file, p): p for p in files}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Overwriting"):
            if fut.result() == "OK": success += 1
            
    logger.info(f"✅ Normalization Complete. Updated {success}/{len(files)} files.")


# ================= Step 3: 合并 Fast & Slow (保持不变) =================
def worker_step_3_merge_signals(symbol):
    try:
        f_path = TEMP_FAST_DIR / f"{symbol}.parquet"
        s_path = TEMP_SLOW_DIR / f"{symbol}.parquet"
        
        if not f_path.exists(): return f"SKIP: {symbol} (No Fast Data)"
        
        df_fast = pd.read_parquet(f_path)
        df_fast['timestamp'] = pd.to_datetime(df_fast['timestamp'])
        df_fast.sort_values('timestamp', inplace=True)
        
        if s_path.exists():
            df_slow = pd.read_parquet(s_path)
            df_slow['timestamp'] = pd.to_datetime(df_slow['timestamp'])
            df_slow.sort_values('timestamp', inplace=True)
            merged = pd.merge_asof(
                df_fast, df_slow, 
                on='timestamp', direction='backward', 
                tolerance=pd.Timedelta(minutes=30)
            )
        else:
            merged = df_fast
            merged['alpha_score'] = 0.0
            
        out_file = TEMP_MERGED_DIR / f"{symbol}.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(merged), out_file, compression='ZSTD')
        return "OK"
    except Exception as e:
        return f"ERR: {str(e)}"

def run_step_3_merge_signals():
    logger.info("="*60)
    logger.info("🚀 Step 3: Merge Fast & Slow Signals")
    logger.info("="*60)
    if TEMP_MERGED_DIR.exists(): shutil.rmtree(TEMP_MERGED_DIR)
    TEMP_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    fast_files = list(TEMP_FAST_DIR.glob("*.parquet"))
    symbols = [f.stem for f in fast_files]
    success = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_MERGE_WORKERS) as executor:
        futures = {executor.submit(worker_step_3_merge_signals, sym): sym for sym in symbols}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Signal Merge"):
            if fut.result() == "OK": success += 1
    logger.info(f"✅ Step 3 Done. Signals Merged: {success}/{len(symbols)}")


# ================= Step 4: [重写版] 直连 Parquet 源 =================
# ================= Step 4: 修复版 (类型对齐 + 向量化分桶) =================
def get_bucket_vector_strict(full_chain, opt_type, min_d, max_d):
    """
    [严苛版分桶] 
    1. 仅保留黄金区间 (min_d ~ max_d) 内的数据。
    2. 去除所有兜底逻辑。如果区间内无数据，直接返回空。
    """
    if full_chain.empty: 
        return pd.DataFrame()

    # 1. 类型筛选 (向量化)
    type_s = full_chain['option_type'].astype(str).str.upper().str.strip().str[0]
    target_char = 'C' if opt_type.upper().startswith('C') else 'P'
    
    # 2. 严格 Delta 区间筛选
    # 必须同时满足: 类型匹配 AND Delta在区间内
    mask = (type_s == target_char) & \
           (full_chain['abs_delta'] >= min_d) & \
           (full_chain['abs_delta'] < max_d)
           
    subset = full_chain[mask].copy()
    
    if subset.empty: 
        return pd.DataFrame()

    # 3. 择优 (Volume 最大)
    if 'volume' in subset.columns:
        # 按 时间(升序), 成交量(降序) 排序
        subset = subset.sort_values(['timestamp', 'volume'], ascending=[True, False])
    else:
        subset = subset.sort_values(['timestamp'])
        
    # 4. 去重 (每分钟保留一个最佳)
    best_df = subset.drop_duplicates(subset=['timestamp'], keep='first')
    
    # 返回 timestamp 索引
    return best_df.set_index('timestamp')

def get_bucket_vector_robust(full_chain, opt_type, min_d, max_d):
    """
    [修复版 V2] 纯向量化实现。
    1. 彻底解决 'Series' object has no attribute 'columns' 报错。
    2. 彻底解决 精度不匹配问题 (逻辑上不依赖精度，但在 worker 里会强转)。
    """
    if full_chain.empty: 
        return pd.DataFrame()

    # 1. 类型筛选 (Vectorized String Op)
    # 提取首字母并转大写: 'Call' -> 'C'
    type_s = full_chain['option_type'].astype(str).str.upper().str.strip().str[0]
    target_char = 'C' if opt_type.upper().startswith('C') else 'P'
    
    subset = full_chain[type_s == target_char].copy()
    if subset.empty: 
        return pd.DataFrame()

    # 2. 计算指标
    target_delta = (min_d + max_d) / 2
    subset['delta_dist'] = (subset['abs_delta'] - target_delta).abs()
    subset['in_range'] = (subset['abs_delta'] >= min_d) & (subset['abs_delta'] < max_d)

    # 3. 策略A: 黄金区间内 (Volume 最大优先)
    df_in = subset[subset['in_range']]
    best_in = pd.DataFrame()
    if not df_in.empty:
        if 'volume' in df_in.columns:
            best_in = df_in.sort_values(['timestamp', 'volume'], ascending=[True, False])
        else:
            best_in = df_in.sort_values(['timestamp'])
        best_in = best_in.drop_duplicates(subset=['timestamp'], keep='first')

    # 4. 策略B: 全局兜底 (Delta 距离最近优先)
    best_fallback = subset.sort_values(['timestamp', 'delta_dist'], ascending=[True, True])
    best_fallback = best_fallback.drop_duplicates(subset=['timestamp'], keep='first')

    # 5. 合并 (Combine First)
    best_in = best_in.set_index('timestamp')
    best_fallback = best_fallback.set_index('timestamp')
    final_best = best_in.combine_first(best_fallback)
    
    return final_best


def get_daily_locked_tickers(full_chain, opt_type, min_d, max_d):
    """
    [核心修复] 日内锁定选择器
    逻辑:
    1. 按【天】分组，而不是按【分钟】。
    2. 每天筛选出符合 Delta 范围的候选合约。
    3. 在候选者中，选出【全天总成交量最大】的那个合约 (Liquidity King)。
    4. 锁定该合约 ID，将该天内该 ID 的所有分钟数据取回。
    5. 结果: 一天之内，Ticker 绝对不会变。
    """
    if full_chain.empty:
        return pd.DataFrame()

    # 1. 类型筛选 ('Call' -> 'C')
    # 向量化处理首字母
    type_s = full_chain['option_type'].astype(str).str.upper().str.strip().str[0]
    target_char = 'C' if opt_type.upper().startswith('C') else 'P'
    subset = full_chain[type_s == target_char].copy()
    
    if subset.empty: return pd.DataFrame()

    # 2. 提取日期字符串 (用于按天分组)
    subset['date_str'] = subset['timestamp'].dt.date.astype(str)
    
    # 3. 筛选候选合约 (Candidates)
    # 只要合约在当天【任意时刻】进入过 Delta 区间，就有资格参选
    # (更严格的做法是只看 9:30-10:00，但按成交量选通常能自动选到最合适的 ATM)
    mask = (subset['abs_delta'] >= min_d) & (subset['abs_delta'] < max_d)
    candidates = subset[mask]
    
    if candidates.empty: return pd.DataFrame()

    # 4. 【选妃环节】每天选出一个成交量最大的 Ticker
    # Group By Date + Ticker -> Sum Volume
    daily_stats = candidates.groupby(['date_str', 'ticker'])['volume'].sum().reset_index()
    
    # 对每一天，按 Volume 降序排列，取第一名
    # 结果: [date_str, ticker] (每天唯一的 King)
    daily_best = daily_stats.sort_values(['date_str', 'volume'], ascending=[True, False])
    daily_best = daily_best.drop_duplicates(subset=['date_str'], keep='first')
    
    # 5. 【回表环节】将选中的 Ticker 广播回该天的所有分钟
    # 使用 Inner Join: 原始数据 <-> 每日最佳名单
    # 这样就过滤掉了所有非当选合约
    locked_df = pd.merge(
        subset, 
        daily_best[['date_str', 'ticker']], 
        on=['date_str', 'ticker'], 
        how='inner'
    )
    
    # 6. 清理与排序
    if 'date_str' in locked_df.columns:
        del locked_df['date_str']
        
    # 按时间排序，确保每分钟只有一条数据 (防止数据源本身有重复)
    locked_df = locked_df.sort_values('timestamp')
    locked_df = locked_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    return locked_df.set_index('timestamp')


# ==============================================================================
# [核心逻辑] 开盘锁定选择器 (Opening Range Lock)
# ==============================================================================
def get_bucket_vector_opening_lock_BACK(full_chain, opt_type, min_d, max_d):
    """
    [实盘仿真] 开盘锁定选择器 (Opening Range Lock)
    逻辑:
    1. 模拟实盘: 每天只在开盘窗口 (09:30-09:45) 做一次决策。
    2. 筛选: 找出窗口内 Delta 在 [min_d, max_d] 且成交量最大的合约。
    3. 锁定: 全天 (09:30-16:00) 强制持有该合约。
    4. 兜底: 若开盘无数据，降级为全天成交量最大。
    """
    if full_chain.empty: return pd.DataFrame()

    # 1. 类型筛选 ('Call' -> 'C')
    type_s = full_chain['option_type'].astype(str).str.upper().str.strip().str[0]
    target_char = 'C' if opt_type.upper().startswith('C') else 'P'
    subset = full_chain[type_s == target_char].copy()
    if subset.empty: return pd.DataFrame()

    # 2. 辅助列
    subset['date_str'] = subset['timestamp'].dt.date.astype(str)
    
    # 3. 定义开盘窗口 (09:30 - 09:45)
    is_opening = (subset['timestamp'].dt.hour == 9) & (subset['timestamp'].dt.minute <= 45)
    
    # 4. 计算评分 (Score)
    # 只有 Delta 达标的 Volume 才算数
    mask_valid_delta = (subset['abs_delta'] >= min_d) & (subset['abs_delta'] < max_d)
    
    # 技巧: 用 numpy where 避免 SettingWithCopyWarning
    subset['valid_vol'] = np.where(mask_valid_delta, subset['volume'], 0)
    subset['window_vol'] = np.where(is_opening, subset['valid_vol'], 0)
    
    # 5. 选王 (Group by Date)
    # 找出每一天: 开盘量最大(优先) 或 全天量最大(兜底) 的合约代码
    daily_stats = subset.groupby(['date_str', 'contract_symbol']).agg({
        'window_vol': 'sum', 
        'valid_vol': 'sum'
    }).reset_index()
    
    # 排序: 日期 -> 开盘量(降序) -> 全天量(降序)
    daily_best = daily_stats.sort_values(
        ['date_str', 'window_vol', 'valid_vol'], 
        ascending=[True, False, False]
    )
    
    # 每天只取第一名 (The Chosen One)
    daily_best = daily_best.drop_duplicates(subset=['date_str'], keep='first')
    
    # 6. 锁定广播 (Lock & Broadcast)
    # 将选中的合约 Inner Join 回原始数据
    # 这样一天之中，无论该合约后续 Delta 变成多少，我们都强制持有它
    locked_df = pd.merge(
        subset, 
        daily_best[['date_str', 'contract_symbol']], 
        on=['date_str', 'contract_symbol'], 
        how='inner'
    )
    
    # 7. 清理
    cols_to_drop = ['date_str', 'window_vol', 'valid_vol', 'score_vol']
    locked_df.drop(columns=[c for c in cols_to_drop if c in locked_df.columns], inplace=True)
    
    # 排序并去重 (保留最后一条以防微秒级重复)
    locked_df = locked_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    return locked_df.set_index('timestamp')


# ==============================================================================
# [核心逻辑] 开盘锁定选择器 (Opening Range Lock)
# ==============================================================================
def get_bucket_vector_opening_lock(full_chain, opt_type, min_d, max_d):
    """
    [实盘仿真] 开盘锁定选择器 (Opening Range Lock)
    逻辑:
    1. 模拟实盘: 每天只在开盘窗口 (09:30-09:45) 做一次决策。
    2. 筛选: 找出窗口内 Delta 在 [min_d, max_d] 且成交量最大的合约。
    3. 锁定: 全天 (09:30-16:00) 强制持有该合约。
    4. 兜底: 若开盘无数据，降级为全天成交量最大。
    """
    if full_chain.empty: return pd.DataFrame()

    # 1. 类型筛选 ('Call' -> 'C')
    type_s = full_chain['option_type'].astype(str).str.upper().str.strip().str[0]
    target_char = 'C' if opt_type.upper().startswith('C') else 'P'
    subset = full_chain[type_s == target_char].copy()
    if subset.empty: return pd.DataFrame()

    # 2. 辅助列
    subset['date_str'] = subset['timestamp'].dt.date.astype(str)
    
    # 3. 定义开盘窗口 (09:30 - 09:45)
    is_opening = (subset['timestamp'].dt.hour == 9) & (subset['timestamp'].dt.minute <= 45)
    
    # 4. 计算评分 (Score)
    # 只有 Delta 达标的 Volume 才算数
    mask_valid_delta = (subset['abs_delta'] >= min_d) & (subset['abs_delta'] < max_d)
    
    # 技巧: 用 numpy where 避免 SettingWithCopyWarning
    subset['valid_vol'] = np.where(mask_valid_delta, subset['volume'], 0)
    subset['window_vol'] = np.where(is_opening, subset['valid_vol'], 0)
    
    # 5. 选王 (Group by Date)
    # 找出每一天: 开盘量最大(优先) 或 全天量最大(兜底) 的合约代码
    daily_stats = subset.groupby(['date_str', 'contract_symbol']).agg({
        'window_vol': 'sum', 
        'valid_vol': 'sum'
    }).reset_index()
    
    # 排序: 日期 -> 开盘量(降序) -> 全天量(降序)
    daily_best = daily_stats.sort_values(
        ['date_str', 'window_vol', 'valid_vol'], 
        ascending=[True, False, False]
    )
    
    # 每天只取第一名 (The Chosen One)
    daily_best = daily_best.drop_duplicates(subset=['date_str'], keep='first')
    
    # 6. 锁定广播 (Lock & Broadcast)
    # 将选中的合约 Inner Join 回原始数据
    # 这样一天之中，无论该合约后续 Delta 变成多少，我们都强制持有它
    locked_df = pd.merge(
        subset, 
        daily_best[['date_str', 'contract_symbol']], 
        on=['date_str', 'contract_symbol'], 
        how='inner'
    )
    
    # 7. 清理
    cols_to_drop = ['date_str', 'window_vol', 'valid_vol', 'score_vol']
    locked_df.drop(columns=[c for c in cols_to_drop if c in locked_df.columns], inplace=True)
    
    # 排序并去重 (保留最后一条以防微秒级重复)
    locked_df = locked_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    return locked_df.set_index('timestamp')

def worker_step_4_attach_options_strict_map(symbol):
    try:
        in_path = TEMP_MERGED_DIR / f"{symbol}.parquet"
        if not in_path.exists(): return f"SKIP: {symbol} (No Merged Signal)"
        
        # 1. 读取主信号
        main_df = pd.read_parquet(in_path)
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
        
        # 确保 Main DF 是 Naive NY Time
        if main_df['timestamp'].dt.tz is not None:
            main_df['timestamp'] = main_df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
        # ==============================================================================
        # [核心修复 1] 正确带着整表排序，并强制移除源信号中可能携带的重复时间点
        # ==============================================================================
        main_df['timestamp'] = main_df['timestamp'].astype('datetime64[ns]')
        main_df = main_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
        
        # [新增] 构造日期辅助列，用于后续 ID 强制对齐
        main_df['date_str'] = main_df['timestamp'].dt.date.astype(str)
        
        if main_df.empty: return f"SKIP: {symbol} (Empty Signal)"
        
        # 确定范围
        min_ts = main_df['timestamp'].min() - pd.Timedelta(days=1)
        max_ts = main_df['timestamp'].max() + pd.Timedelta(days=1)
        
        symbol_opt_dir = OPTION_SOURCE_DIR / symbol / "standard"
        if not symbol_opt_dir.exists():
            return f"SKIP_WARN: NO_OPT_DIR: {symbol}"
            
        parquet_glob = str(symbol_opt_dir / "*.parquet")
        
        # ==============================================================================
        # [优化 1] SQL 查询: 增加 7-14 DTE 过滤
        # ==============================================================================
        query = f"""
        SELECT 
            timestamp,
            expiration_date as expiration,
            contract_type as option_type,
            strike_price as strike,
            delta, gamma, vega, theta, iv, close, volume,
            ticker as contract_symbol ,
            stock_close
        FROM '{parquet_glob}'
        WHERE timestamp >= '{min_ts}' AND timestamp <= '{max_ts}'
          AND (CAST(expiration_date AS DATE) - CAST(timestamp AS DATE)) BETWEEN 7 AND 14
        """
        
        try:
            full_chain = duckdb.query(query).to_df()
        except Exception as e:
            return f"ERR_DB: {str(e)}"
            
        final_opt_cols = {}
        final_opt_ids = {} 
        valid_match_count = 0

        if not full_chain.empty:
            # ------------------------------------------------------------------
            # 数据预处理
            # ------------------------------------------------------------------
            full_chain['timestamp'] = pd.to_datetime(full_chain['timestamp'])
            
            # 时区对齐 (UTC -> NY Naive)
            if full_chain['timestamp'].dt.tz is not None:
                full_chain['timestamp'] = full_chain['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            full_chain['timestamp'] = full_chain['timestamp'].astype('datetime64[ns]').dt.round('1min')
            
            # 修复 expiration 时区问题
            full_chain['expiration'] = pd.to_datetime(full_chain['expiration'])
            if full_chain['expiration'].dt.tz is not None:
                 full_chain['expiration'] = full_chain['expiration'].dt.tz_convert('America/New_York').dt.tz_localize(None)

            # [优化 2] Pandas 二次确认 DTE
            full_chain['dte_days'] = (full_chain['expiration'] - full_chain['timestamp']).dt.days
            full_chain = full_chain[(full_chain['dte_days'] >= 7) & (full_chain['dte_days'] <= 14)]
            
            # 清洗
            full_chain = full_chain[(full_chain['close'] > 0.01) & (full_chain['close'] < 5000.0)]
            if 'iv' in full_chain.columns:
                full_chain = full_chain[(full_chain['iv'] > 0.001) & (full_chain['iv'] < 5.0)]
            
            full_chain['price'] = full_chain['close']
            
            # Delta 归一化
            if 'delta' in full_chain.columns:
                if full_chain['delta'].abs().mean() > 1.0:
                    full_chain['delta'] = full_chain['delta'] / 100.0
                full_chain['abs_delta'] = full_chain['delta'].abs()
            
            # ------------------------------------------------------------------
            # 分桶匹配 (保留 ATM + OTM)
            # ------------------------------------------------------------------
            buckets = [
                ('P', 0.50, 0.35, 0.65, 0),  # ATM Put
                ('C', 0.50, 0.35, 0.65, 8),  # ATM Call
                ('P', 0.25, 0.15, 0.35, 16), # OTM Put (严格 OTM < 0.35)
                ('C', 0.25, 0.15, 0.35, 24)  # OTM Call
            ]
            
            left_axis = main_df[['timestamp']].copy()
            
            for otype, target_d, min_d, max_d, offset_idx in buckets:
                # 1. 选出最佳数据 (包含 timestamp, contract_symbol)
                best_df = get_bucket_vector_opening_lock(full_chain, otype, min_d, max_d)
                
                if best_df.empty: continue
                
                # ==========================================================
                # [核心修复 2] ID 强制按天广播 (Daily Broadcast) 使用 map 映射
                # 不依赖 merge_asof 和 ffill，确保 ID 全天绝对唯一且连续，绝不膨胀
                # ==========================================================
                best_df_reset = best_df.reset_index()
                best_df_reset['date_str'] = best_df_reset['timestamp'].dt.date.astype(str)
                
                # 提取 (Date -> ID) 映射表
                daily_id_map = best_df_reset[['date_str', 'contract_symbol']].drop_duplicates(subset=['date_str'], keep='last')
                
                # 使用 pandas map 完美映射，行数绝对不会变
                id_series = daily_id_map.set_index('date_str')['contract_symbol']
                final_opt_ids[f'opt_{offset_idx}_id'] = main_df['date_str'].map(id_series).fillna("").astype(str).values
                
                # ==========================================================
                # 数值数据依然使用 asof merge (允许空缺，后续补0或ffill)
                # ==========================================================
                right_df = best_df.reset_index().sort_values('timestamp')
                right_df = right_df.drop_duplicates(subset=['timestamp'], keep='last')

                merged_bucket = pd.merge_asof(
                    left_axis, right_df, 
                    on='timestamp', direction='backward', 
                    tolerance=pd.Timedelta(minutes=30)
                )
                
                # 提取特征 (价格, Greeks)
                feat_map = [
                    (merged_bucket['price'], 0), (merged_bucket['delta'], 1), (merged_bucket['gamma'], 2), 
                    (merged_bucket['vega'], 3), (merged_bucket['theta'], 4), (merged_bucket['strike'], 5), 
                    (pd.Series(0, index=merged_bucket.index), 6), (merged_bucket['iv'], 7)
                ]
                
                has_data = False
                for series, i in feat_map:
                    vals = series.values
                    final_opt_cols[f'opt_{offset_idx + i}'] = vals
                    if np.count_nonzero(~np.isnan(vals)) > 0: has_data = True
                
                if has_data: valid_match_count += 1

        if valid_match_count == 0:
            return f"SKIP_WARN: ZERO_MATCH: {symbol}"
            
        # ------------------------------------------------------------------
        # 组装结果
        # ------------------------------------------------------------------
        
        # 1. 组装数值矩阵 & FFILL
        opt_matrix = pd.DataFrame(0.0, index=main_df.index, columns=[f'opt_{i}' for i in range(32)])
        for col, values in final_opt_cols.items():
            opt_matrix[col] = values
            
        # 价格数据前向填充 (fillna 0)
        opt_matrix = opt_matrix.replace(0.0, np.nan).ffill(limit=5).fillna(0.0)
        
        # 2. 组装 ID 列
        id_matrix = pd.DataFrame('', index=main_df.index, columns=[f'opt_{i}_id' for i in [0, 8, 16, 24]])
        for col, values in final_opt_ids.items():
            if col in id_matrix.columns:
                id_matrix[col] = values
        
        # [优化] ID 列不再需要 FFill，因为它是通过 Date Join 生成的，天生连续
        # 只需要 fillna('') 即可
        id_matrix = id_matrix.fillna('')
            
        # 3. 合并所有 (移除临时的 date_str)
        if 'date_str' in main_df.columns:
            main_df.drop(columns=['date_str'], inplace=True)
            
        final_df = pd.concat([main_df, opt_matrix, id_matrix], axis=1)
        
        # ==============================================================================
        # [核心修复 3] 最后的终极防线：在存盘前强制剥离任何意外产生的重复行
        # ==============================================================================
        final_df = final_df.drop_duplicates(subset=['timestamp'], keep='last')
        
        out_file = FINAL_OUTPUT_DIR / f"{symbol}.parquet"
        pq.write_table(pa.Table.from_pandas(final_df), out_file, compression='ZSTD')
        
        return f"OK"
        
    except Exception as e:
        import traceback
        return f"ERR: {str(e)} | {traceback.format_exc()}"
    
# ================= Step 4: 挂载期权 (严格模式) =================
def worker_step_4_attach_options_strict(symbol):
    try:
        in_path = TEMP_MERGED_DIR / f"{symbol}.parquet"
        if not in_path.exists(): return f"SKIP: {symbol} (No Merged Signal)"
        
        # 1. 读取主信号
        main_df = pd.read_parquet(in_path)
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
        
        # 确保 Main DF 是 Naive NY Time
        if main_df['timestamp'].dt.tz is not None:
            main_df['timestamp'] = main_df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
        # ==============================================================================
        # [核心修复 1] 正确带着整表排序，并强制移除源信号中可能携带的重复时间点
        # ==============================================================================
        main_df['timestamp'] = main_df['timestamp'].astype('datetime64[ns]')
        main_df = main_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
        
        # [新增] 构造日期辅助列，用于后续 ID 强制对齐
        main_df['date_str'] = main_df['timestamp'].dt.date.astype(str)
        
        if main_df.empty: return f"SKIP: {symbol} (Empty Signal)"
        
        # 确定范围
        min_ts = main_df['timestamp'].min() - pd.Timedelta(days=1)
        max_ts = main_df['timestamp'].max() + pd.Timedelta(days=1)
        
        symbol_opt_dir = OPTION_SOURCE_DIR / symbol / "standard"
        if not symbol_opt_dir.exists():
            return f"SKIP_WARN: NO_OPT_DIR: {symbol}"
            
        parquet_glob = str(symbol_opt_dir / "*.parquet")
        
        # ==============================================================================
        # [优化 1] SQL 查询: 增加 7-14 DTE 过滤
        # ==============================================================================
        query = f"""
        SELECT 
            timestamp,
            expiration_date as expiration,
            contract_type as option_type,
            strike_price as strike,
            delta, gamma, vega, theta, iv, close, volume,
            ticker as contract_symbol ,
            stock_close
        FROM '{parquet_glob}'
        WHERE timestamp >= '{min_ts}' AND timestamp <= '{max_ts}'
          AND (CAST(expiration_date AS DATE) - CAST(timestamp AS DATE)) BETWEEN 7 AND 14
        """
        
        try:
            full_chain = duckdb.query(query).to_df()
        except Exception as e:
            return f"ERR_DB: {str(e)}"
            
        final_opt_cols = {}
        final_opt_ids = {} 
        valid_match_count = 0

        if not full_chain.empty:
            # ------------------------------------------------------------------
            # 数据预处理
            # ------------------------------------------------------------------
            full_chain['timestamp'] = pd.to_datetime(full_chain['timestamp'])
            
            # 时区对齐 (UTC -> NY Naive)
            if full_chain['timestamp'].dt.tz is not None:
                full_chain['timestamp'] = full_chain['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            full_chain['timestamp'] = full_chain['timestamp'].astype('datetime64[ns]').dt.round('1min')
            
            # 修复 expiration 时区问题
            full_chain['expiration'] = pd.to_datetime(full_chain['expiration'])
            if full_chain['expiration'].dt.tz is not None:
                 full_chain['expiration'] = full_chain['expiration'].dt.tz_convert('America/New_York').dt.tz_localize(None)

            # [优化 2] Pandas 二次确认 DTE
            full_chain['dte_days'] = (full_chain['expiration'] - full_chain['timestamp']).dt.days
            full_chain = full_chain[(full_chain['dte_days'] >= 7) & (full_chain['dte_days'] <= 14)]
            
            # 清洗
            full_chain = full_chain[(full_chain['close'] > 0.01) & (full_chain['close'] < 5000.0)]
            if 'iv' in full_chain.columns:
                full_chain = full_chain[(full_chain['iv'] > 0.001) & (full_chain['iv'] < 5.0)]
            
            full_chain['price'] = full_chain['close']
            
            # Delta 归一化
            if 'delta' in full_chain.columns:
                if full_chain['delta'].abs().mean() > 1.0:
                    full_chain['delta'] = full_chain['delta'] / 100.0
                full_chain['abs_delta'] = full_chain['delta'].abs()
            
            # ------------------------------------------------------------------
            # 分桶匹配 (保留 ATM + OTM)
            # ------------------------------------------------------------------
            buckets = [
                ('P', 0.50, 0.35, 0.65, 0),  # ATM Put
                ('C', 0.50, 0.35, 0.65, 8),  # ATM Call
                ('P', 0.25, 0.15, 0.35, 16), # OTM Put (严格 OTM < 0.35)
                ('C', 0.25, 0.15, 0.35, 24)  # OTM Call
            ]
            
            left_axis = main_df[['timestamp']].copy()
            
            for otype, target_d, min_d, max_d, offset_idx in buckets:
                # 1. 选出最佳数据 (包含 timestamp, contract_symbol)
                best_df = get_bucket_vector_opening_lock(full_chain, otype, min_d, max_d)
                
                if best_df.empty: continue
                
                # ==========================================================
                # [核心修复 2] ID 强制按天广播 (Daily Broadcast) 使用 map 映射
                # 不依赖 merge_asof 和 ffill，确保 ID 全天绝对唯一且连续，绝不膨胀
                # ==========================================================
                best_df_reset = best_df.reset_index()
                best_df_reset['date_str'] = best_df_reset['timestamp'].dt.date.astype(str)
                
                # 提取 (Date -> ID) 映射表
                daily_id_map = best_df_reset[['date_str', 'contract_symbol']].drop_duplicates(subset=['date_str'], keep='last')
                
                # 使用 pandas map 完美映射，行数绝对不会变
                id_series = daily_id_map.set_index('date_str')['contract_symbol']
                final_opt_ids[f'opt_{offset_idx}_id'] = main_df['date_str'].map(id_series).fillna("").astype(str).values
                
                # ==========================================================
                # 数值数据依然使用 asof merge (允许空缺，后续补0或ffill)
                # ==========================================================
                right_df = best_df.reset_index().sort_values('timestamp')
                right_df = right_df.drop_duplicates(subset=['timestamp'], keep='last')

                merged_bucket = pd.merge_asof(
                    left_axis, right_df, 
                    on='timestamp', direction='backward', 
                    tolerance=pd.Timedelta(minutes=30)
                )
                
                # 提取特征 (价格, Greeks)
                feat_map = [
                    (merged_bucket['price'], 0), (merged_bucket['delta'], 1), (merged_bucket['gamma'], 2), 
                    (merged_bucket['vega'], 3), (merged_bucket['theta'], 4), (merged_bucket['strike'], 5), 
                    (pd.Series(0, index=merged_bucket.index), 6), (merged_bucket['iv'], 7)
                ]
                
                has_data = False
                for series, i in feat_map:
                    vals = series.values
                    final_opt_cols[f'opt_{offset_idx + i}'] = vals
                    if np.count_nonzero(~np.isnan(vals)) > 0: has_data = True
                
                if has_data: valid_match_count += 1

        if valid_match_count == 0:
            return f"SKIP_WARN: ZERO_MATCH: {symbol}"
            
        # ------------------------------------------------------------------
        # 组装结果
        # ------------------------------------------------------------------
        
        # 1. 组装数值矩阵 & FFILL
        opt_matrix = pd.DataFrame(0.0, index=main_df.index, columns=[f'opt_{i}' for i in range(32)])
        for col, values in final_opt_cols.items():
            opt_matrix[col] = values
            
        # 价格数据前向填充 (fillna 0)
        opt_matrix = opt_matrix.replace(0.0, np.nan).ffill(limit=5).fillna(0.0)
        
        # 2. 组装 ID 列
        id_matrix = pd.DataFrame('', index=main_df.index, columns=[f'opt_{i}_id' for i in [0, 8, 16, 24]])
        for col, values in final_opt_ids.items():
            if col in id_matrix.columns:
                id_matrix[col] = values
        
        # [优化] ID 列不再需要 FFill，因为它是通过 Date Join 生成的，天生连续
        # 只需要 fillna('') 即可
        id_matrix = id_matrix.fillna('')
            
        # 3. 合并所有 (移除临时的 date_str)
        if 'date_str' in main_df.columns:
            main_df.drop(columns=['date_str'], inplace=True)
            
        final_df = pd.concat([main_df, opt_matrix, id_matrix], axis=1)
        
        # ==============================================================================
        # [核心修复 3] 最后的终极防线：在存盘前强制剥离任何意外产生的重复行
        # ==============================================================================
        final_df = final_df.drop_duplicates(subset=['timestamp'], keep='last')
        
        out_file = FINAL_OUTPUT_DIR / f"{symbol}.parquet"
        pq.write_table(pa.Table.from_pandas(final_df), out_file, compression='ZSTD')
        
        return f"OK"
        
    except Exception as e:
        import traceback
        return f"ERR: {str(e)} | {traceback.format_exc()}"

def worker_step_4_attach_options_direct(symbol):
    """
    Step 4 Worker (Direct Source):
    [修正时区逻辑] 
    针对源文件是 NY 时区的情况，必须先 tz_convert('America/New_York') 
    确保时间回退到 09:30，然后再去时区。
    """
    try:
        in_path = TEMP_MERGED_DIR / f"{symbol}.parquet"
        if not in_path.exists(): return f"SKIP: {symbol} (No Merged Signal)"
        
        # 1. 读取主信号 (Fast/Slow)
        # 这些数据通常已经是 Naive NY Time (09:30)
        main_df = pd.read_parquet(in_path)
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
        
        # 防御性处理：如果主信号意外带有时区，先转 NY 再去时区
        if main_df['timestamp'].dt.tz is not None:
            main_df['timestamp'] = main_df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
        # 强制精度对齐
        main_df['timestamp'] = main_df['timestamp'].astype('datetime64[ns]')
        main_df = main_df.sort_values('timestamp')
        
        if main_df.empty: return f"SKIP: {symbol} (Empty Signal)"
        
        # 确定范围
        min_ts = main_df['timestamp'].min() - pd.Timedelta(days=5)
        max_ts = main_df['timestamp'].max() + pd.Timedelta(days=5)
        
        symbol_opt_dir = OPTION_SOURCE_DIR / symbol / "standard"
        if not symbol_opt_dir.exists():
            return f"SKIP_WARN: NO_OPT_DIR: {symbol}"
            
        parquet_glob = str(symbol_opt_dir / "*.parquet")
        
        # DuckDB 查询
        query = f"""
        SELECT 
            timestamp,
            expiration_date as expiration,
            contract_type as option_type,
            strike_price as strike,
            delta, gamma, vega, theta, iv, close, volume,
            ticker as contract_symbol
        FROM '{parquet_glob}'
        WHERE timestamp >= '{min_ts}' AND timestamp <= '{max_ts}'
        """
        
        try:
            full_chain = duckdb.query(query).to_df()
        except Exception as e:
            return f"ERR_DB: {str(e)}"
            
        final_opt_cols = {}
        valid_match_count = 0

        if not full_chain.empty:

             
            # ------------------------------------------------------------------
            # [新增] 数据清洗与异常过滤
            # ------------------------------------------------------------------
            
            # 1. 基础价格清洗
            # 过滤掉价格 <= 0.01 的末日废纸或错误数据（除非你的策略专门做末日轮，否则建议过滤）
            # 过滤掉价格 > 5000 的异常高价 (根据具体标的调整，SPX可能需要更高，但个股5000通常是错的)
            full_chain = full_chain[
                (full_chain['close'] > 0.01) & 
                (full_chain['close'] < 5000.0)
            ]
            
            # 2. IV 清洗 (非常重要)
            # IV = 0 通常是无效数据
            # IV > 5.0 (500%) 通常是极其深度的价外期权或者是数据错误，会导致 Greeks 计算极其不稳定
            if 'iv' in full_chain.columns:
                full_chain = full_chain[
                    (full_chain['iv'] > 0.001) & 
                    (full_chain['iv'] < 5.0)
                ]

            # 3. Delta 清洗
            # Delta 绝对值不应超过 1.0 (如果源数据是百分比，这里需要注意适配)
            # 你的代码后面有除以100的逻辑，建议先清洗极端值
            if 'delta' in full_chain.columns:
                # 假设源数据可能偶尔出现 > 100 或 < -100 的错误
                full_chain = full_chain[
                    (full_chain['delta'].abs() > 0.0001) &  # 过滤完全无敏感度的
                    (full_chain['delta'].abs() < 200.0)     # 过滤明显错误 (预留一些 buffer 给后面除以100)
                ]

            # 4. (可选) 过滤流动性极差的数据
            # 如果 volume 和 open_interest 都是 0，这个价格可能是几天前的收盘价
            # 这种“僵尸价格”会严重误导 RL
            if 'volume' in full_chain.columns:
                 # 这是一个权衡：太过严格会导致数据缺失。
                 # 建议：如果是分钟级数据，Volume=0 很正常，不要过滤。
                 # 但如果是日线级别，Volume=0 且 OpenInterest=0 应该过滤。
                 # 既然是分钟级，这里暂时保留 Volume=0 的数据，依靠上面的 IV/Price 过滤。
                 pass

            if full_chain.empty:
                 return f"SKIP_WARN: EMPTY_AFTER_CLEAN: {symbol}"

            # ------------------------------------------------------------------
            # [原有逻辑继续]
            # ------------------------------------------------------------------

            # [关键修复] 时区对齐逻辑
            full_chain['timestamp'] = pd.to_datetime(full_chain['timestamp'])
            
            # 1. 如果有时区信息 (DuckDB 读入通常是 UTC 或 原样)
            if full_chain['timestamp'].dt.tz is not None:
                # 必须显式转为纽约时间，把 13:30(UTC) 变回 09:30(NY)
                full_chain['timestamp'] = full_chain['timestamp'].dt.tz_convert('America/New_York')
                # 然后去掉时区标签，变成 Naive 的 09:30
                full_chain['timestamp'] = full_chain['timestamp'].dt.tz_localize(None)
            else:
                # 如果 DuckDB 读进来已经是 Naive 的，我们假设它已经是 Wall Time
                # 但为了保险，如果它是 UTC Naive (13:30)，这步无法识别。
                # 鉴于你的源文件带时区，DuckDB output 99% 是带时区的，所以上面 if 会命中。
                pass

            # 2. 强制转为纳秒精度 (解决 incompatible merge keys)
            full_chain['timestamp'] = full_chain['timestamp'].astype('datetime64[ns]')
            
            # 3. Round 对齐
            full_chain['timestamp'] = full_chain['timestamp'].dt.round('1min')
            
            # 4. Delta 归一化
            if full_chain['delta'].abs().mean() > 1.0:
                full_chain['delta'] = full_chain['delta'] / 100.0
            full_chain['abs_delta'] = full_chain['delta'].abs()
            
            full_chain['price'] = full_chain['close']
            
            # 分桶匹配
            buckets = [
                ('P', 0.50, 0.35, 0.65, 0), ('C', 0.50, 0.35, 0.65, 8),
                ('P', 0.25, 0.15, 0.40, 16), ('C', 0.25, 0.15, 0.40, 24)
            ]
            
            left_axis = main_df[['timestamp']].copy()
            
            for otype, target_d, min_d, max_d, offset_idx in buckets:
                best_df = get_bucket_vector_robust(full_chain, otype, min_d, max_d)
                
                if best_df.empty: continue
                
                right_df = best_df.reset_index().sort_values('timestamp')
                right_df = right_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # ASOF Merge
                merged_bucket = pd.merge_asof(
                    left_axis, right_df, 
                    on='timestamp', direction='backward', 
                    tolerance=pd.Timedelta(minutes=30)
                )
                
                feat_map = [
                    (merged_bucket['price'], 0), (merged_bucket['delta'], 1), (merged_bucket['gamma'], 2), 
                    (merged_bucket['vega'], 3), (merged_bucket['theta'], 4), (merged_bucket['strike'], 5), 
                    (pd.Series(0, index=merged_bucket.index), 6), (merged_bucket['iv'], 7)
                ]
                
                has_data = False
                for series, i in feat_map:
                    vals = series.values
                    final_opt_cols[f'opt_{offset_idx + i}'] = vals
                    if np.count_nonzero(~np.isnan(vals)) > 0:
                        has_data = True
                
                if has_data: valid_match_count += 1

        if valid_match_count == 0:
            return f"SKIP_WARN: ZERO_MATCH: {symbol}"
            
        opt_matrix = pd.DataFrame(0.0, index=main_df.index, columns=[f'opt_{i}' for i in range(32)])
        for col, values in final_opt_cols.items():
            opt_matrix[col] = values
            
        opt_matrix = opt_matrix.replace(0.0, np.nan).ffill(limit=5).fillna(0.0)
        final_df = pd.concat([main_df, opt_matrix], axis=1)
        
        out_file = FINAL_OUTPUT_DIR / f"{symbol}.parquet"
        pq.write_table(pa.Table.from_pandas(final_df), out_file, compression='ZSTD')
        
        return f"OK: {symbol} (Buckets: {valid_match_count}/4)"
        
    except Exception as e:
        return f"ERR: {str(e)}"
    

def run_step_4_attach_options():
    logger.info("="*60)
    logger.info("🚀 Step 4: Attach Options (Direct Parquet Source + Robust Fallback)")
    logger.info("="*60)
    
    if FINAL_OUTPUT_DIR.exists(): shutil.rmtree(FINAL_OUTPUT_DIR)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    merged_files = list(TEMP_MERGED_DIR.glob("*.parquet"))
    symbols = [f.stem for f in merged_files]
    
    if not symbols:
        logger.error("❌ Step 3 output is empty.")
        return

    success = 0
    warnings_count = 0
    
    # 建议根据磁盘 IO 调整 max_workers (DuckDB 并发读文件 IO 开销较大)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, MAX_MERGE_WORKERS)) as executor:
        futures = {executor.submit(worker_step_4_attach_options_strict, sym): sym for sym in symbols}
        
        try:
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="Attach Options"):
                res = fut.result()
                
                if res.startswith("SKIP_WARN"):
                    logger.warning(f"⚠️ {res}")
                    warnings_count += 1
                elif res.startswith("OK"):
                    logger.info(f"✅ {res}")
                    success += 1
                elif res.startswith("ERR"):
                    logger.error(f"❌ {res}")
                else:
                    logger.debug(f"ℹ️ {res}")
                    
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            sys.exit(1)
            
    logger.info("-" * 40)
    logger.info(f"✅ Step 4 Complete.")
    logger.info(f"   Success: {success}")
    logger.info(f"   Skipped: {warnings_count}")

def run_step_4_verify():
    # 简单的验证逻辑
    files = list(FINAL_OUTPUT_DIR.glob("*.parquet"))
    if not files: return
    logger.info(f"🔍 Verifying {len(files)} output files...")
    
    cnt_ok = 0
    for p in files[:10]: # 抽查
        try:
            df = pd.read_parquet(p)
            # 检查是否有期权数据 (opt_0 到 opt_31)
            opt_sum = df.loc[:, 'opt_0':'opt_31'].sum().sum()
            if opt_sum > 0: cnt_ok += 1
        except: pass
    logger.info(f"   Sample Check: {cnt_ok}/10 have option data.")

# ================= 验证与统计 (Data Quality Check) =================
def run_full_validation():
    logger.info("="*60)
    logger.info("📊 Final Validation: Option Coverage Statistics")
    logger.info("="*60)
    
    files = list(FINAL_OUTPUT_DIR.glob("*.parquet"))
    if not files:
        logger.error("❌ No output files found!")
        return

    logger.info(f"🔍 Analyzing {len(files)} files...")
    
    stats = []
    
    # 进度条遍历
    for p in tqdm(files, desc="Validating"):
        try:
            # 只读取几列关键数据以加速
            # opt_0 (Put Price), opt_8 (Call Price)
            df = pd.read_parquet(p, columns=['timestamp', 'opt_0', 'opt_8'])
            
            total_rows = len(df)
            if total_rows == 0: continue
            
            # 定义“有效”: 任意一个桶有数据 (非0 且 非NaN)
            # 这里检查 ATM Call (opt_8) 和 ATM Put (opt_0)
            # 宽松标准: 只要 opt_8 > 0 就认为有数据
            valid_mask = (df['opt_8'] > 1e-6) | (df['opt_0'] > 1e-6)
            valid_rows = valid_mask.sum()
            
            coverage = valid_rows / total_rows
            stats.append({'symbol': p.stem, 'coverage': coverage, 'rows': total_rows})
            
        except Exception as e:
            logger.error(f"Read error {p.stem}: {e}")

    # 生成统计报告
    df_stats = pd.DataFrame(stats)
    if df_stats.empty:
        logger.error("❌ No valid stats generated.")
        return

    avg_cov = df_stats['coverage'].mean()
    
    # 分桶统计
    bins = [0, 0.01, 0.1, 0.5, 0.9, 1.0]
    labels = ['Dead (0%)', 'Poor (<10%)', 'Fair (10-50%)', 'Good (50-90%)', 'Perfect (>90%)']
    # 为了防止边界问题，稍微调整 bin
    bins[0] = -0.01
    bins[-1] = 1.01
    
    df_stats['grade'] = pd.cut(df_stats['coverage'], bins=bins, labels=labels)
    distribution = df_stats['grade'].value_counts().sort_index()
    
    logger.info("\n" + "-"*40)
    logger.info(f"📈 Coverage Report (Total Stocks: {len(df_stats)})")
    logger.info(f"   Average Option Coverage: {avg_cov:.1%}")
    logger.info("-" * 40)
    logger.info("Distribution:")
    for label, count in distribution.items():
        pct = count / len(df_stats)
        logger.info(f"   {label:<15}: {count:4d} ({pct:.1%})")
        
    # 打印覆盖率为 0 的股票 (前20个)
    dead_stocks = df_stats[df_stats['coverage'] < 0.01]['symbol'].tolist()
    if dead_stocks:
        logger.info("-" * 40)
        logger.info(f"⚠️  Dead Stocks (Coverage ~ 0%): {len(dead_stocks)}")
        logger.info(f"    Examples: {dead_stocks[:20]} ...")
        
    logger.info("-" * 40)
    logger.info("✅ Validation Complete.")

# ================= Alpha 有效性速检 (Signal Quality Check) =================
def calculate_ic(series_a, series_b):
    """计算斯皮尔曼秩相关系数 (Rank IC)"""
    if len(series_a) < 100 or len(series_b) < 100: return 0.0
    # 移除 NaN
    valid_mask = np.isfinite(series_a) & np.isfinite(series_b)
    if valid_mask.sum() < 100: return 0.0
    
    # 使用 Pandas 的 corr 计算 Rank Correlation
    return pd.Series(series_a[valid_mask]).corr(pd.Series(series_b[valid_mask]), method='spearman')

def run_signal_quality_check(sample_size=200):
    logger.info("="*60)
    logger.info("🧪 Final Step: Signal Quality Check (Rule vs Model)")
    logger.info("="*60)
    
    files = list(FINAL_OUTPUT_DIR.glob("*.parquet"))
    if not files: return

    import random
    selected_files = random.sample(files, min(len(files), sample_size))
    
    ic_stats = []
    
    for p in tqdm(selected_files, desc="Checking Signals"):
        try:
            # [修改] 读取 fast_evt
            df = pd.read_parquet(p, columns=['timestamp', 'close', 'alpha_score', 'fast_vol', 'fast_evt'])
            df = df.sort_values('timestamp')
            
            # Future Realized Volatility (5 step)
            df['future_vol'] = df['close'].pct_change().rolling(5).std().shift(-5)
            
            # 1. Rule IC
            vol_ic = calculate_ic(df['fast_vol'], df['future_vol'])
            
            # 2. [新增] Model IC
            evt_ic = calculate_ic(df['fast_evt'], df['future_vol'])
            
            ic_stats.append({'vol_ic': vol_ic, 'evt_ic': evt_ic})
            
        except: continue

    if not ic_stats: return

    df_stats = pd.DataFrame(ic_stats)
    mean_vol_ic = df_stats['vol_ic'].mean()
    mean_evt_ic = df_stats['evt_ic'].mean()
    
    logger.info("\n" + "-"*40)
    logger.info(f"🏆 Volatility Signal Battle (Target: Future 5min Realized Vol)")
    logger.info("-" * 40)
    
    logger.info(f"1. [Rule]  Fast Vol IC : {mean_vol_ic:.4f}")
    logger.info(f"2. [Model] Fast Evt IC : {mean_evt_ic:.4f}")
    
    if mean_evt_ic > mean_vol_ic:
        logger.info(f"🎉 Winner: AI Model! (Improvement: {(mean_evt_ic - mean_vol_ic)*100:.2f}%)")
    else:
        logger.info(f"🛡️ Winner: Simple Rule. (Model might need retraining)")
    
    logger.info("-" * 40)
    
# ================= [新增] Z-Score vs Raw 对比检测 =================
def run_zscore_comparison_check(sample_size=100):
    logger.info("="*60)
    logger.info("⚖️  Comparative Analysis: RAW vs Z-SCORE Scaled Signals")
    logger.info("="*60)
    
    files = list(FINAL_OUTPUT_DIR.glob("*.parquet"))
    if not files: return

    import random
    from scipy.stats import spearmanr
    
    # 随机抽样
    selected_files = random.sample(files, min(len(files), sample_size))
    
    # 容器
    all_raw_alpha = []
    all_raw_vol = []
    all_future_ret = [] # 用于计算 Alpha IC
    all_future_vol = [] # 用于计算 Vol IC
    
    logger.info(f"📥 Loading sample data from {len(selected_files)} files...")
    
    for p in tqdm(selected_files, desc="Sampling Data"):
        try:
            df = pd.read_parquet(p, columns=['timestamp', 'close', 'alpha_score', 'fast_vol'])
            df = df.sort_values('timestamp')
            
            # 1. 准备 Target (30min Return)
            df['ret_30m'] = df['close'].shift(-30) / df['close'] - 1.0
            
            # 2. 准备 Vol Target (5min Realized Vol)
            df['real_vol_5m'] = df['close'].pct_change().rolling(5).std().shift(-5)
            
            # 移除 NaN
            df = df.dropna()
            
            if len(df) > 0:
                # 为了内存考虑，每只股票只取最近 1000 条
                df_sample = df.iloc[-1000:]
                all_raw_alpha.extend(df_sample['alpha_score'].values)
                all_raw_vol.extend(df_sample['fast_vol'].values)
                all_future_ret.extend(df_sample['ret_30m'].values)
                all_future_vol.extend(df_sample['real_vol_5m'].values)
                
        except Exception as e:
            continue
            
    # 转为 Numpy
    raw_alpha = np.array(all_raw_alpha, dtype=np.float32)
    raw_vol = np.array(all_raw_vol, dtype=np.float32)
    f_ret = np.array(all_future_ret, dtype=np.float32)
    f_vol = np.array(all_future_vol, dtype=np.float32)
    
    # === 计算 Z-Score (模拟全局归一化) ===
    # 注意：RL 实际运行时通常是 Rolling Z-Score，但全局分布能很好地反映数量级问题
    
    # 1. Alpha Z-Score
    alpha_mean = np.mean(raw_alpha)
    alpha_std = np.std(raw_alpha) + 1e-6
    z_alpha = (raw_alpha - alpha_mean) / alpha_mean
    # 修正：通常 Z-Score 是 (x-u)/std
    z_alpha_real = (raw_alpha - alpha_mean) / alpha_std
    
    # 2. Vol Z-Score
    vol_mean = np.mean(raw_vol)
    vol_std = np.std(raw_vol) + 1e-6
    z_vol_real = (raw_vol - vol_mean) / vol_std
    
    # === 计算统计指标 ===
    def get_stats(data, target, name):
        # 基础统计
        mean_val = np.mean(data)
        std_val = np.std(data)
        q05 = np.percentile(data, 5)
        q95 = np.percentile(data, 95)
        
        # IC (Rank Correlation) - 理论上 Raw 和 Z-Score 的 Rank IC 是一样的
        # 但我们需要确认是否有数值精度问题
        ic, _ = spearmanr(data, target)
        
        return {
            'Name': name,
            'Mean': f"{mean_val:.4f}",
            'Std': f"{std_val:.4f}",
            'Range(5%-95%)': f"[{q05:.3f}, {q95:.3f}]",
            'IC': f"{ic:.4f}"
        }

    logger.info("\n" + "-"*80)
    logger.info(f"{'Metric':<15} | {'Mean':<10} | {'Std (Energy)':<12} | {'Range (5%-95%)':<20} | {'IC (Pred Power)':<10}")
    logger.info("-" * 80)
    
    # 1. Alpha 对比
    s1 = get_stats(raw_alpha, f_ret, "Raw Alpha")
    s2 = get_stats(z_alpha_real, f_ret, "Z-Score Alpha")
    
    logger.info(f"{s1['Name']:<15} | {s1['Mean']:<10} | {s1['Std']:<12} | {s1['Range(5%-95%)']:<20} | {s1['IC']:<10}")
    logger.info(f"{s2['Name']:<15} | {s2['Mean']:<10} | {s2['Std']:<12} | {s2['Range(5%-95%)']:<20} | {s2['IC']:<10}")
    logger.info("-" * 80)
    
    # 2. Vol 对比
    v1 = get_stats(raw_vol, f_vol, "Raw Vol")
    v2 = get_stats(z_vol_real, f_vol, "Z-Score Vol")
    
    logger.info(f"{v1['Name']:<15} | {v1['Mean']:<10} | {v1['Std']:<12} | {v1['Range(5%-95%)']:<20} | {v1['IC']:<10}")
    logger.info(f"{v2['Name']:<15} | {v2['Mean']:<10} | {v2['Std']:<12} | {v2['Range(5%-95%)']:<20} | {v2['IC']:<10}")
    logger.info("-" * 80)
    
    # === 智能建议 ===
    logger.info("🤖 AI Recommendation:")
    
    # 判据 1: 信号消失检测 (Vanishing Signal)
    if float(s1['Std']) < 0.1:
        logger.warning("⚠️  Raw Alpha Std is too low (< 0.1). Neural Networks will struggle to see this.")
        logger.warning("   👉 MUST USE Z-SCORE or Scaling (multiply by 10 or 50).")
    elif float(s1['Std']) > 10.0:
        logger.warning("⚠️  Raw Alpha Std is too high (> 10). Gradients might explode.")
        logger.warning("   👉 MUST USE Z-SCORE to normalize.")
    else:
        logger.info("✅ Raw Alpha magnitude looks acceptable, but Z-Score is usually safer.")

    # 判据 2: 偏度/分布检测
    q05_z = float(s2['Range(5%-95%)'].split(',')[0][1:])
    q95_z = float(s2['Range(5%-95%)'].split(',')[1][:-1])
    
    if q95_z - q05_z > 6.0:
        logger.warning("⚠️  Z-Score distribution is extremely wide (Fat Tails). Consider RankGauss or Clipping.")
    else:
        logger.info("✅ Z-Score distribution looks healthy (Gaussian-like).")

    logger.info("="*60)


# ================= Step 5: 生成全市场矩阵 (Matrix Builder) =================
def run_step_5_create_market_matrix():
    """
    [新增] 将所有个股数据聚合为以 timestamp 为索引的大矩阵。
    目的: 为 V8 回测引擎提供“上帝视角”，支持逐分钟全市场扫描。
    """
    logger.info("="*60)
    logger.info("📦 Step 5: Generating Unified Market Matrix (Time-Indexed)")
    logger.info("="*60)
    
    # 输出路径
    matrix_output_path = LMDB_ROOT / f'unified_market_matrix_{stage}.parquet'
    
    # 输入: Step 4 生成的所有最终 Parquet
    all_files_glob = str(FINAL_OUTPUT_DIR / "*.parquet")
    
    # 使用 DuckDB 极速聚合
    # 只选取回测所需的核心列，减小体积
    # 注意: 增加了 stock_close 以确保信号源的准确性
    query = f"""
    COPY (
        SELECT 
            timestamp, 
            '{stage}' as stage, 
            replace(filename, '.parquet', '') as symbol,
            alpha_score, 
            fast_vol, 
            close, 
            close AS stock_close, 
            opt_0, opt_8,    -- ATM Put, Call Price
            opt_0_id, opt_8_id -- ATM Put, Call ID
        FROM read_parquet('{all_files_glob}', filename=true)
        ORDER BY timestamp ASC, alpha_score DESC
    ) TO '{matrix_output_path}' (FORMAT 'parquet', COMPRESSION 'ZSTD');
    """
    
    try:
        # 执行查询
        duckdb.execute(query)
        logger.info(f"✅ Market Matrix created successfully at:")
        logger.info(f"   📂 {matrix_output_path}")
        
        # 简单验证
        res = duckdb.query(f"SELECT count(*) FROM '{matrix_output_path}'").fetchone()
        logger.info(f"   📊 Total Rows: {res[0]:,}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create matrix: {str(e)}")

def check_id_stability():
    files = list(FINAL_OUTPUT_DIR.glob("*.parquet"))
    print(f"Checking {len(files)} files for ID stability...")
    
    drift_count = 0
    
    for p in tqdm(files[:100]): # 抽查 100 个文件
        try:
            df = pd.read_parquet(p, columns=['timestamp', 'opt_0_id', 'opt_8_id'])
            df['date'] = df['timestamp'].dt.date
            
            # 检查每一天是否只有一个唯一 ID
            for col in ['opt_0_id', 'opt_8_id']:
                # 过滤掉空值
                valid = df[df[col] != ""]
                if valid.empty: continue
                
                # 按天分组，计算每天的唯一 ID 数量
                daily_counts = valid.groupby('date')[col].nunique()
                
                # 如果某天有 > 1 个 ID，说明锁定失败
                drifts = daily_counts[daily_counts > 1]
                
                if not drifts.empty:
                    print(f"❌ FAIL: {p.stem} in {col} on dates: {drifts.index.tolist()}")
                    drift_count += 1
                    break # 发现一个就报错
        except Exception as e:
            pass
            
    if drift_count == 0:
        print("✅ PASS: 所有抽查文件的 ID 在单日内均保持唯一 (锁定成功)。")
    else:
        print(f"❌ FAIL: 发现 {drift_count} 个文件存在单日 ID 漂移 (锁定失败)。请重跑 s5。")

if __name__ == "__main__":
    import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    
    # run_step_1_fast()
    # run_step_1_5_normalize_fast()
    run_step_2_slow()
    run_step_2_5_normalize_slow()
    
    # run_step_3_merge_signals()
    # run_step_4_attach_options()
     
    # check_id_stability()
    # run_full_validation()
    # run_signal_quality_check(sample_size=500)

    # # 7. [新增] Raw vs Z-Score 深度对比
    # # 这将帮助你决定是否需要在 RL Env 中开启 feature scaling
    # run_zscore_comparison_check(sample_size=200)
    