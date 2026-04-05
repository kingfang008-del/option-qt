#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: apply_rolling_norm_nested_final.py
描述: 
    [最终优化版] 嵌套目录滚动归一化 + 自动化质检
    
    功能:
    1. 支持 STAGES 配置 (["train", "val"])，自动切换目录。
    2. 并行处理股票(目录) + 串行处理时间(文件)，维护 Rolling Buffer。
    3. 增加 verify_data_quality 函数，随机抽样检查数据分布，防止"全0"或"全NaN"灾难。
    
    路径假设: 
    ~/data/stocks_parquet_features_train/...
    ~/data/stocks_parquet_features_val/...
"""

import pandas as pd
import numpy as np
import json
import logging
import warnings
import os
import random
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 基础数据根目录 (包含 train/val 文件夹的父目录)
BASE_ROOT = Path.home() / "train_data" 

# 需要处理的阶段
STAGES = ["train", "val", "test"] 

# 配置文件
CONFIG_PATH = Path.home() / "notebook/train/feature_all.json"

# 窗口配置
ROLLING_WINDOW = 2000  
MIN_PERIODS = 100      
USE_TANH = True        # True: Tanh(-1, 1), False: Clip(-5, 5)
ZSCORE_CLIP = 10      # 如果 USE_TANH=False，则使用此截断值

# 并行核数
MAX_WORKERS = max(1, os.cpu_count() - 2) 
# ===========================================

def load_target_features(config_path):
    """加载特征列表"""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    bounded_features = {
        'session', 'day_of_week', 'hour', 'is_holiday', 'rsi_divergence',
        'rsi', 'k', 'd', 'adx', 'rvi', 'vw_delta', 'vp_corr_15',
        'minute', 'is_expiry', 'is_fed_meeting', 'stock_id', 'timestamp', 'date',
        'symbol', 'open', 'high', 'low', 'close', 'volume', 'fast_vol', 'spy_roc_5min', 'qqq_roc_5min'
    }

    target_features = set()
    for f in config.get('features', []):
        name = f['name']
        is_real = f.get('type') == 'real'
        is_not_raw = f.get('calc') != 'raw'
        if is_real and is_not_raw and name not in bounded_features and not name.startswith('label_'):
            target_features.add(name)
    
    return list(target_features)

def process_single_directory(args):
    """
    处理单个目录（即单个股票的所有月份文件）
    """
    dir_path, norm_cols = args
    
    files = sorted(list(dir_path.glob("*.parquet")))
    if not files:
        return None
    
    buffer_df = None
    processed_count = 0

    # [新增常量] 标记需要防污染的极度肥尾特征
    FAT_TAIL_FEATURES = {'options_iv_momentum', 'options_gamma_accel', 'options_iv_divergence'}
    
    try:
        # 按月份顺序串行处理
        for file_path in files:
            df = pd.read_parquet(file_path)
            if df.empty: continue
            
            # 检查特征列
            cols_to_process = [c for c in norm_cols if c in df.columns]
            if not cols_to_process: continue
            
            # 强制 float32
            df[cols_to_process] = df[cols_to_process].astype(np.float32)

            # ======================================================================
            # 🚀 [核心新增]：对极值期权特征进行 Signed Log1p 预处理，彻底消灭窗口污染！
            # ======================================================================
            fat_tail_cols = [c for c in cols_to_process if c in FAT_TAIL_FEATURES]
            if fat_tail_cols:
                # 公式: sign(x) * log(1 + abs(x))
                # 这种变换能无缝处理 0 和 负数，且大幅压制 1000x 级别的肥尾暴击
                df[fat_tail_cols] = np.sign(df[fat_tail_cols]) * np.log1p(np.abs(df[fat_tail_cols]))
            # ======================================================================
            
            # 拼接 Buffer (处理跨月边界)
            if buffer_df is not None:
                # [安全对齐] 先对齐列再拼接
                buffer_aligned = buffer_df.reindex(columns=cols_to_process)
                combined_block = pd.concat([buffer_aligned, df[cols_to_process]], axis=0)
            else:
                combined_block = df[cols_to_process]
            
            # 滚动计算
            roller = combined_block.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS)
            roll_mean = roller.mean()
            roll_std = roller.std()
            
            # Z-Score
            z_block = (combined_block - roll_mean) / (roll_std + 1e-6)
            
            # 切片当前月
            current_z = z_block.iloc[-len(df):]
            
            # 异常值处理
            if USE_TANH:
                current_z = np.tanh(current_z / 3.0)
            else:
                current_z = current_z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)
            
            # 填充 NaN
            current_z = current_z.fillna(0.0)
            
            # 更新 Buffer (保存原始数据)
            if len(combined_block) > ROLLING_WINDOW:
                new_buffer = combined_block.iloc[-ROLLING_WINDOW:]
            else:
                new_buffer = combined_block
            buffer_df = new_buffer
            
            # [安全写回] 确保右值和左值的列严格一致
            df[cols_to_process] = current_z[cols_to_process].values
            
            df.to_parquet(file_path, index=False)
            processed_count += 1
            
        return f"OK: {dir_path.name}"

    except Exception as e:
        import traceback
        return f"ERROR in {dir_path}: {str(e)}"

def find_leaf_directories(root_path):
    """递归查找股票目录"""
    leaf_dirs = set()
    for p in root_path.rglob("*.parquet"):
        leaf_dirs.add(p.parent)
    return sorted(list(leaf_dirs))

def verify_data_quality(root_path, norm_cols, sample_count=20):
    """
    [新增] 自动化数据验证函数
    随机抽取 sample_count 个文件，检查归一化结果是否合理
    """
    print(f"\n🔍 [质检] 正在对 {root_path.name} 进行抽样检查 (Samples={sample_count})...")
    
    all_files = list(root_path.rglob("*.parquet"))
    if not all_files:
        print("❌ 目录为空，无法检查")
        return
    
    samples = random.sample(all_files, min(len(all_files), sample_count))
    
    issues = 0
    passed = 0
    
    for f in samples:
        try:
            df = pd.read_parquet(f)
            check_cols = [c for c in norm_cols if c in df.columns]
            
            if not check_cols: continue
            
            data = df[check_cols].values
            
            # 检查 1: 是否全为 0 (说明窗口不够或填充错误)
            zero_ratio = (data == 0).mean()
            
            # 检查 2: 是否有 NaN
            nan_count = np.isnan(data).sum()
            
            # 检查 3: 范围是否合理 (Tanh 应该在 -1 到 1 之间)
            max_val = np.max(data)
            min_val = np.min(data)
            
            file_status = "✅"
            msg = ""
            
            # 判定逻辑
            if nan_count > 0:
                file_status = "❌ NaN Detected"
                issues += 1
            elif zero_ratio > 0.8: # 允许部分0，但不能全是0
                file_status = "⚠️ Mostly Zeros"
                msg = f"(Zero Ratio: {zero_ratio:.1%})"
                issues += 1
            elif USE_TANH and (max_val > 1.0001 or min_val < -1.0001):
                file_status = "❌ Tanh Range Error"
                msg = f"(Range: {min_val:.2f} ~ {max_val:.2f})"
                issues += 1
            else:
                passed += 1
                
            # 只打印有问题的，或少量正常的
            if file_status != "✅" or passed <= 3:
                print(f"  {file_status} {f.name} {msg}")
                
        except Exception as e:
            print(f"  ❌ Read Error {f.name}: {e}")
            issues += 1
            
    print("-" * 40)
    if issues == 0:
        print(f"✨ 质检通过! 所有抽样文件数据正常。")
    else:
        print(f"⚠️ 质检发现 {issues} 个潜在问题文件，请检查上方日志。")
    print("-" * 40)

def main():
    print("="*60)
    print("🚀 嵌套目录滚动归一化 (多 Stage 版)")
    print(f"📄 配置: {CONFIG_PATH.name}")
    print(f"🪟 窗口: {ROLLING_WINDOW} | Tanh: {USE_TANH}")
    print(f"⚙️  Stages: {STAGES}")
    print("="*60)

    try:
        norm_cols = load_target_features(CONFIG_PATH)
        print(f"📝 目标特征数: {len(norm_cols)}")
    except Exception as e:
        logging.error(f"Config Error: {e}")
        return

    # --- 循环处理每个 Stage ---
    for stage in STAGES:
        stage_dir_name = f"quote_features_{stage}"
        stage_root = BASE_ROOT / stage_dir_name
        
        print(f"\nProcessing Stage: [{stage}] -> {stage_root}")
        
        if not stage_root.exists():
            logging.error(f"❌ 目录不存在: {stage_root}，跳过。")
            continue
            
        # 1. 扫描任务
        target_dirs = find_leaf_directories(stage_root)
        if not target_dirs:
            logging.warning(f"⚠️  [{stage}] 目录下没有找到 Parquet 文件。")
            continue
            
        print(f"✅ Found {len(target_dirs)} stock directories.")

        # 2. 并行处理
        tasks = [(d, norm_cols) for d in target_dirs]
        error_cnt = 0
        
        # 使用 max_workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(process_single_directory, tasks), total=len(tasks), desc=f"Norm {stage}"))
            
            for res in results:
                if res and res.startswith("ERROR"):
                    error_cnt += 1
                    logging.error(res)
        
        if error_cnt > 0:
            print(f"⚠️  [{stage}] 完成，但有 {error_cnt} 个股票处理出错。")
        else:
            print(f"🎉 [{stage}] 处理完成，无错误。")
            
        # 3. 执行质检
        verify_data_quality(stage_root, norm_cols)

    print("\n✅ 所有 Stage 处理完毕。")

if __name__ == "__main__":
    main()