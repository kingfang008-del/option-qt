#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: convert_db_to_parquet.py
描述: 全量 SQLite 提纯与 Parquet 融合工具 (Flat File Structure)
功能:
    1. 自动扫描 history_sqlite 目录下的所有 market_*.db
    2. 解决 ts 类型不匹配问题 (强制 float64)
    3. 采用 Map-Reduce 架构，将多日数据按标的无缝纵向拼接
    4. 输出格式与 s5 离线管道完全一致: root/AAPL.parquet
    5. [新增] 提取期权真实成交量 (Volume)，用于回测的流动性拦截。
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
import shutil
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
PROJECT_ROOT = Path.home() / "quant_project"
DB_DIR = PROJECT_ROOT / "data" / "history_sqlite_1s"

# 临时碎片目录 (按天切片)
TEMP_DIR = PROJECT_ROOT / "data" / "temp_db_convert"
# 最终的扁平化输出目录 (与 S5 回测格式一致)
PARQUET_OUT_BASE = PROJECT_ROOT / "data" / "rl_feed_parquet_batch"

# ============================================

def parse_option_buckets(json_str):
    """提取期权快照的核心字段 (含 Volume)"""
    try:
        data = json.loads(json_str)
        buckets = data.get('buckets', [])
        contracts = data.get('contracts', [])
        
        # Put ATM (Bucket 0)
        p_price = float(buckets[0][0]) if len(buckets) > 0 and len(buckets[0]) > 0 else 0.0
        p_k     = float(buckets[0][5]) if len(buckets) > 0 and len(buckets[0]) > 5 else 0.0
        p_vol   = float(buckets[0][6]) if len(buckets) > 0 and len(buckets[0]) > 6 else 0.0 
        p_iv    = float(buckets[0][7]) if len(buckets) > 0 and len(buckets[0]) > 7 else 0.0
        p_bid   = float(buckets[0][8]) if len(buckets) > 0 and len(buckets[0]) > 8 else p_price
        p_ask   = float(buckets[0][9]) if len(buckets) > 0 and len(buckets[0]) > 9 else p_price
        p_bid_s = float(buckets[0][10]) if len(buckets) > 0 and len(buckets[0]) > 10 else 100.0
        p_ask_s = float(buckets[0][11]) if len(buckets) > 0 and len(buckets[0]) > 11 else 100.0
        p_id    = str(contracts[0])    if len(contracts) > 0 else ""

        # Call ATM (Bucket 2)
        c_price = float(buckets[2][0]) if len(buckets) > 2 and len(buckets[2]) > 0 else 0.0
        c_k     = float(buckets[2][5]) if len(buckets) > 2 and len(buckets[2]) > 5 else 0.0
        c_vol   = float(buckets[2][6]) if len(buckets) > 2 and len(buckets[2]) > 6 else 0.0
        c_iv    = float(buckets[2][7]) if len(buckets) > 2 and len(buckets[2]) > 7 else 0.0
        c_bid   = float(buckets[2][8]) if len(buckets) > 2 and len(buckets[2]) > 8 else c_price
        c_ask   = float(buckets[2][9]) if len(buckets) > 2 and len(buckets[2]) > 9 else c_price
        c_bid_s = float(buckets[2][10]) if len(buckets) > 2 and len(buckets[2]) > 10 else 100.0
        c_ask_s = float(buckets[2][11]) if len(buckets) > 2 and len(buckets[2]) > 11 else 100.0
        c_id    = str(contracts[2])    if len(contracts) > 2 else ""

        return pd.Series([
            p_price, p_k, p_vol, p_iv, p_bid, p_ask, p_bid_s, p_ask_s, p_id,
            c_price, c_k, c_vol, c_iv, c_bid, c_ask, c_bid_s, c_ask_s, c_id
        ])
    except Exception:
        return pd.Series([0.0]*8 + [""] + [0.0]*8 + [""])

def process_single_db(date_str: str, db_path: Path):
    """[Map 阶段] 处理单日 SQLite，生成各标的当天碎片 Parquet"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
        if not cursor.fetchone():
            conn.close()
            return

        # 1. 提取并强制转换 float，彻底解决 merge_asof 崩溃问题
        df_alpha = pd.read_sql("SELECT symbol, ts, alpha as alpha_score FROM alpha_logs", conn)
        if df_alpha.empty:
            conn.close()
            return

        df_alpha['ts'] = df_alpha['ts'].astype(float)
        # 🚀 [新增核心逻辑]：预计算截面 Alpha Z-Score (完美模拟实盘的截面打分)
        # 按 ts 分组，计算当前这一分钟全市场的 mean 和 std，映射到 [-5.0, 5.0]
        df_alpha['alpha_score'] = df_alpha.groupby('ts')['alpha_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        ).clip(-5.0, 5.0)
        
        df_bars = pd.read_sql("SELECT symbol, ts, close, volume FROM market_bars_1s", conn)
        df_bars['ts'] = df_bars['ts'].astype(float)
        
        df_opts = pd.read_sql("SELECT symbol, ts, buckets_json FROM option_snapshots_1s", conn)
        df_opts['ts'] = df_opts['ts'].astype(float)
        
        conn.close()

        # 🚀 [频率无关重构] 使用时间偏移窗口计算大盘动量 (确保 1s/5s 逻辑一致)
        df_bars['dt'] = pd.to_datetime(df_bars['ts'], unit='s')
        df_spy = df_bars[df_bars['symbol'] == 'SPY'].sort_values('dt').copy()
        
        # 使用 5分钟 时间窗口计算 ROC
        df_spy = df_spy.set_index('dt')
        df_spy['spy_roc_5min'] = df_spy['close'].pct_change(freq='300s').fillna(0)
        spy_map = dict(zip(df_spy['ts'], df_spy['spy_roc_5min']))
        
        df_qqq = df_bars[df_bars['symbol'] == 'QQQ'].sort_values('ts').copy()
        df_qqq['dt'] = pd.to_datetime(df_qqq['ts'], unit='s')
        df_qqq = df_qqq.set_index('dt')
        df_qqq['qqq_roc_5min'] = df_qqq['close'].pct_change(freq='300s').fillna(0)
        qqq_map = dict(zip(df_qqq['ts'], df_qqq['qqq_roc_5min']))

        symbols = df_alpha['symbol'].unique()
        for sym in symbols:
            s_alpha = df_alpha[df_alpha['symbol'] == sym].sort_values('ts').copy()
            s_bar = df_bars[df_bars['symbol'] == sym].sort_values('ts').copy()
            s_opt = df_opts[df_opts['symbol'] == sym].sort_values('ts').copy()

            if s_bar.empty: continue

            # 🚀 [频率无关重构] 使用 1分钟 时间窗口计算波动率
            s_bar['dt'] = pd.to_datetime(s_bar['ts'], unit='s')
            s_bar = s_bar.set_index('dt')
            
            # 1. 计算 1分钟 收益率 (对齐 1min 回放语义)
            s_bar['log_ret_1m'] = np.log(s_bar['close'] / s_bar['close'].shift(freq='60s').replace(0, np.nan)).fillna(0.0)
            
            # 2. 计算 5分钟 (300s) 滚动标准差作为原始波动率
            raw_vol_1m = s_bar['log_ret_1m'].rolling('300s').std().fillna(0.0)
            
            # 使用自适应窗口进行 Z-Score 标准化
            v_mean = raw_vol_1m.expanding(min_periods=60).mean()
            v_std = raw_vol_1m.expanding(min_periods=60).std()
            fast_vol_all = ((raw_vol_1m - v_mean) / (v_std + 1e-6)).clip(-5.0, 5.0).fillna(0.0)

            # 🚀 [Parity Fix] 确保波动率在分钟内稳定，对齐 1m 历史基准
            # 仅保留分钟整点 (ts % 60 == 0) 的值，其余秒帧通过 ffill 保持恒定
            s_bar['is_min_boundary'] = (s_bar['ts'] % 60 == 0)
            s_bar['fast_vol'] = np.where(s_bar['is_min_boundary'], fast_vol_all, np.nan)
            s_bar['fast_vol'] = pd.Series(s_bar['fast_vol']).ffill().fillna(0.0).values
            
            s_bar = s_bar.reset_index(drop=True)

            # 🚀 [终极对齐：后移 60s]
            # 确保信号产生（基于前一分钟事实）后，至少延迟 1 分钟才撮合价格（T+1 执行）
            s_alpha_adjusted = s_alpha[['ts', 'alpha_score']].copy()
            s_alpha_adjusted['ts'] = s_alpha_adjusted['ts'] + 60.0
            
            # 🚀 [1s 核心修正] 必须以 s_bar 为基准，才能保留秒级价格序列！
            df_merged = pd.merge_asof(s_bar, s_alpha_adjusted, on='ts', direction='backward')
            # 移除 tolerance=10.0，让数据能跨越更大的空位进行 ffill
            df_merged = pd.merge_asof(df_merged, s_opt[['ts', 'buckets_json']], on='ts', direction='backward')

            df_merged['buckets_json'] = df_merged['buckets_json'].fillna("{}")
            df_merged['spy_roc_5min'] = df_merged['ts'].map(spy_map).fillna(0)
            df_merged['qqq_roc_5min'] = df_merged['ts'].map(qqq_map).fillna(0)

            df_merged[[
                'opt_0', 'opt_5', 'feed_put_vol', 'opt_7', 'feed_put_bid', 'feed_put_ask', 'feed_put_bid_size', 'feed_put_ask_size', 'opt_0_id',
                'opt_8', 'opt_13', 'feed_call_vol', 'opt_15', 'feed_call_bid', 'feed_call_ask', 'feed_call_bid_size', 'feed_call_ask_size', 'opt_8_id'
            ]] = df_merged['buckets_json'].apply(parse_option_buckets)

            # [修改] 构建最终 DataFrame
            df_final = df_merged[[
                'ts', 'close', 'alpha_score', 'fast_vol', 'spy_roc_5min', 'qqq_roc_5min',
                'opt_0', 'opt_8', 'opt_0_id', 'opt_8_id', 'opt_5', 'opt_7', 'opt_13', 'opt_15',
                'feed_put_vol', 'feed_call_vol',
                'feed_put_bid', 'feed_put_ask', 'feed_put_bid_size', 'feed_put_ask_size',
                'feed_call_bid', 'feed_call_ask', 'feed_call_bid_size', 'feed_call_ask_size'
            ]].copy()
            df_final['is_new_minute'] = (df_final['ts'] % 60 == 0).astype(int)

            # [核心修复] 执行 Forward-Fill 填充期权缺失秒帧，防止回测成交价变为 0 导致爆仓
            opt_cols = [
                'opt_0', 'opt_8', 'opt_5', 'opt_7', 'opt_13', 'opt_15',
                'feed_put_vol', 'feed_call_vol',
                'feed_put_bid', 'feed_put_ask', 'feed_put_bid_size', 'feed_put_ask_size',
                'feed_call_bid', 'feed_call_ask', 'feed_call_bid_size', 'feed_call_ask_size'
            ]
            for col in opt_cols:
                if col in df_final.columns:
                    df_final[col] = df_final[col].replace(0.0, np.nan).ffill().fillna(0.0)
            
            # ID 列单独处理
            for col in ['opt_0_id', 'opt_8_id']:
                if col in df_final.columns:
                    df_final[col] = df_final[col].replace("", None).ffill().fillna("")
            
            # [终极修复] 保留绝对 Unix 时间戳 ts 供底层引擎使用，timestamp 仅作参考
            df_final['timestamp'] = pd.to_datetime(df_final['ts'], unit='s', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            # 绝对不要 drop('ts')！把它留在 Parquet 里！
            # df_final.drop(columns=['ts'], inplace=True)  <-- 删掉或注释掉这行
            
            # 落入临时碎片目录
            sym_temp_dir = TEMP_DIR / sym
            sym_temp_dir.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(sym_temp_dir / f"{date_str}.parquet", compression='ZSTD')
            
    except Exception as e:
        print(f"❌ 处理 {date_str} 失败: {e}")

def merge_symbol_parquets(sym: str):
    """[Reduce 阶段] 将一个标的的所有日期碎片合并为一个文件"""
    try:
        sym_dir = TEMP_DIR / sym
        files = list(sym_dir.glob("*.parquet"))
        if not files: return "SKIP"
        
        dfs = [pd.read_parquet(f) for f in files]
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 按时间排序并去重 (防止两天预热交界处的数据重叠)
        merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
        
        # 写入最终的平铺目录
        out_path = PARQUET_OUT_BASE / f"{sym}.parquet"
        merged_df.to_parquet(out_path, compression='ZSTD')
        return "OK"
    except Exception as e:
        return f"ERR: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default="20260101", help="Start date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="20991231", help="End date (YYYYMMDD)")
    args = parser.parse_args()

    if not DB_DIR.exists():
        print(f"❌ Database directory not found: {DB_DIR}")
        return

    all_dbs = sorted([f for f in DB_DIR.glob("market_*.db") if f.stem.startswith("market_") and len(f.stem) == 15])
    target_dbs = []
    for db in all_dbs:
        date_str = db.stem.split('_')[1]
        if args.start_date <= date_str <= args.end_date:
            target_dbs.append((date_str, db))
            
    if not target_dbs:
        print(f"❌ 未在区间内找到任何 DB。")
        return

    # 初始化目录
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"🔍 阶段 1: [Map] 提取 {len(target_dbs)} 天数据并按天打碎...")
    for date_str, db_path in tqdm(target_dbs, desc="📆 逐日提取"):
        process_single_db(date_str, db_path)

    # 获取所有生成的标的
    symbols = [d.name for d in TEMP_DIR.iterdir() if d.is_dir()]
    
    if not symbols:
        print(f"⚠️ [Reduce Skip] No symbols found in {TEMP_DIR}. Check if [Map] phase produced any data.")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    print(f"\n🔗 阶段 2: [Reduce] 合并 {len(symbols)} 只标的的时间序列...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(16, len(symbols)))) as executor:
        futures = {executor.submit(merge_symbol_parquets, sym): sym for sym in symbols}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(symbols), desc="📦 合并压缩"):
            if fut.result() == "OK": success += 1

    # 清理临时文件
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print("\n" + "🎉"*20)
    print(f"✅ 全量数据合并完成！共生成 {success} 个平铺的特征文件。")
    print(f"📂 数据集路径: {PARQUET_OUT_BASE}")

if __name__ == "__main__":
    main()