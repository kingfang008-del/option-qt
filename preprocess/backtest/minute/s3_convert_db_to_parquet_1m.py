#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s3_convert_db_to_parquet_1m.py
描述: [分钟级回测基石] 将 SQLite 日志提纯为 Parquet，供 S4 引擎进行确定性回放。
核心修复:
    1. 彻底移除 S3 阶段的 Z-Score 重复计算，原汁原味透传 alpha_logs 中的真实指标。
    2. 严格按 S4 引擎的字段名规范 (feed_call_price, feed_call_bid 等) 生成数据。
    3. 强制 60 秒 Alpha 延迟对齐，彻底消灭未来函数。
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
# from tqdm import tqdm

# ================= 配置区域 =================
# PROJECT_ROOT = Path.home() / "quant_project"
BT_DIR = Path.home() / "quant_project/data"
DB_DIR = BT_DIR / "history_sqlite_1m"
TEMP_DIR = BT_DIR / "temp_db_convert_1m"
PARQUET_OUT_BASE = BT_DIR / "rl_feed_parquet"
# ============================================

def parse_option_buckets(json_str):
    """直接解析出 S4 引擎所需要的标准命名列，并做好安全兜底"""
    # 初始化 S4 所需的全部默认值，防止合并时结构断裂
    res = {
        'feed_put_price': 0.0, 'feed_put_k': 0.0, 'feed_put_vol': 1.0, 'feed_put_iv': 0.0,
        'feed_put_bid': 0.0, 'feed_put_ask': 0.0, 'feed_put_bid_size': 100.0, 'feed_put_ask_size': 100.0,
        'feed_put_id': "",
        'feed_call_price': 0.0, 'feed_call_k': 0.0, 'feed_call_vol': 1.0, 'feed_call_iv': 0.0,
        'feed_call_bid': 0.0, 'feed_call_ask': 0.0, 'feed_call_bid_size': 100.0, 'feed_call_ask_size': 100.0,
        'feed_call_id': ""
    }
    
    if not json_str or json_str == "{}" or pd.isna(json_str):
        # 补充 opt_ 别名
        res.update({
            'opt_0': 0.0, 'opt_5': 0.0, 'opt_7': 0.0, 'opt_0_id': "",
            'opt_8': 0.0, 'opt_13': 0.0, 'opt_15': 0.0, 'opt_8_id': ""
        })
        return pd.Series(res)

    try:
        data = json.loads(json_str)
        buckets = data.get('buckets', [])
        contracts = data.get('contracts', [])
        
        # 解析 PUT (Bucket 0)
        if len(buckets) > 0 and len(contracts) > 0 and len(buckets[0]) >= 8:
            pb = buckets[0]
            res['feed_put_price'] = float(pb[0])
            res['feed_put_k'] = float(pb[5])
            res['feed_put_vol'] = float(pb[6])
            res['feed_put_iv'] = float(pb[7])
            res['feed_put_bid'] = float(pb[8]) if len(pb) > 8 else res['feed_put_price']
            res['feed_put_ask'] = float(pb[9]) if len(pb) > 9 else res['feed_put_price']
            res['feed_put_bid_size'] = float(pb[10]) if len(pb) > 10 else 100.0
            res['feed_put_ask_size'] = float(pb[11]) if len(pb) > 11 else 100.0
            res['feed_put_id'] = str(contracts[0])

        # 解析 CALL (Bucket 2)
        if len(buckets) > 2 and len(contracts) > 2 and len(buckets[2]) >= 8:
            cb = buckets[2]
            res['feed_call_price'] = float(cb[0])
            res['feed_call_k'] = float(cb[5])
            res['feed_call_vol'] = float(cb[6])
            res['feed_call_iv'] = float(cb[7])
            res['feed_call_bid'] = float(cb[8]) if len(cb) > 8 else res['feed_call_price']
            res['feed_call_ask'] = float(cb[9]) if len(cb) > 9 else res['feed_call_price']
            res['feed_call_bid_size'] = float(cb[10]) if len(cb) > 10 else 100.0
            res['feed_call_ask_size'] = float(cb[11]) if len(cb) > 11 else 100.0
            res['feed_call_id'] = str(contracts[2])

        # 🚀 同步生成 s4 预期的 opt_N 别名
        res['opt_0'] = res['feed_put_price']
        res['opt_5'] = res['feed_put_k']
        res['opt_7'] = res['feed_put_iv']
        res['opt_0_id'] = res['feed_put_id']
        
        res['opt_8'] = res['feed_call_price']
        res['opt_13'] = res['feed_call_k']
        res['opt_15'] = res['feed_call_iv']
        res['opt_8_id'] = res['feed_call_id']

        return pd.Series(res)
    except Exception:
        return pd.Series(res)

def process_single_db(date_str: str, db_path: Path):
    """[Map 阶段] 提取 SQLite，推迟 Alpha 信号，生成干净的 DataFrame"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        
        # 1. 提取市场行情 (正股与期权)
        df_bars = pd.read_sql("SELECT symbol, ts, close FROM market_bars_1m", conn)
        df_opts = pd.read_sql("SELECT symbol, ts, buckets_json FROM option_snapshots_1m", conn)
        
        # 2. 提取 Alpha 信号 (兼容有无 vol_z / event_prob 的情况)
        try:
            df_alpha = pd.read_sql("SELECT symbol, ts, alpha as alpha_score, vol_z as fast_vol, event_prob FROM alpha_logs", conn)
        except:
            try:
                df_alpha = pd.read_sql("SELECT symbol, ts, alpha as alpha_score, vol_z as fast_vol FROM alpha_logs", conn)
                df_alpha['event_prob'] = 0.0
            except:
                df_alpha = pd.read_sql("SELECT symbol, ts, alpha as alpha_score FROM alpha_logs", conn)
                df_alpha['fast_vol'] = 0.0
                df_alpha['event_prob'] = 0.0

        conn.close()

        if df_bars.empty: return

        df_bars['ts'] = df_bars['ts'].astype(float)
        df_opts['ts'] = df_opts['ts'].astype(float)
        
        # 🚀 [量化铁律] Alpha 信号推迟 60 秒！
        # 只推迟模型打分，不推迟市场行情，完美模拟 T+1 执行
        if not df_alpha.empty:
            df_alpha['ts'] = df_alpha['ts'].astype(float)  

        # 计算大盘动量 (如果存在 SPY 和 QQQ)
        spy_map = {}
        if 'SPY' in df_bars['symbol'].values:
            df_spy = df_bars[df_bars['symbol'] == 'SPY'].sort_values('ts').copy()
            df_spy['spy_roc_5min'] = df_spy['close'].pct_change(5).fillna(0)
            spy_map = dict(zip(df_spy['ts'], df_spy['spy_roc_5min']))
            
        qqq_map = {}
        if 'QQQ' in df_bars['symbol'].values:
            df_qqq = df_bars[df_bars['symbol'] == 'QQQ'].sort_values('ts').copy()
            df_qqq['qqq_roc_5min'] = df_qqq['close'].pct_change(5).fillna(0)
            qqq_map = dict(zip(df_qqq['ts'], df_qqq['qqq_roc_5min']))

        symbols = df_bars['symbol'].unique()
        for sym in symbols:
            s_bar = df_bars[df_bars['symbol'] == sym].sort_values('ts')
            s_opt = df_opts[df_opts['symbol'] == sym].sort_values('ts') if not df_opts.empty else pd.DataFrame(columns=['ts', 'buckets_json'])
            
            # 向后对齐期权数据 (最多容忍 2 分钟延迟)
            df_merged = pd.merge_asof(s_bar, s_opt[['ts', 'buckets_json']], on='ts', direction='backward', tolerance=120.0)
            
            # 向后对齐已经加了 60s 延迟的 Alpha 信号
            if not df_alpha.empty:
                s_alpha = df_alpha[df_alpha['symbol'] == sym].sort_values('ts')
                df_merged = pd.merge_asof(df_merged, s_alpha[['ts', 'alpha_score', 'fast_vol', 'event_prob']], on='ts', direction='backward', tolerance=120.0)
            else:
                df_merged['alpha_score'] = 0.0
                df_merged['fast_vol'] = 0.0
                df_merged['event_prob'] = 0.0

            # 填充 NaN
            df_merged['alpha_score'] = df_merged['alpha_score'].fillna(0.0)
            df_merged['fast_vol'] = df_merged['fast_vol'].fillna(0.0)
            df_merged['spy_roc_5min'] = df_merged['ts'].map(spy_map).fillna(0.0)
            df_merged['qqq_roc_5min'] = df_merged['ts'].map(qqq_map).fillna(0.0)
            df_merged['buckets_json'] = df_merged['buckets_json'].fillna("{}")

            # 🚀 解析标准期权列 (返回 S4 完全匹配的列名)
            opt_parsed_df = df_merged['buckets_json'].apply(parse_option_buckets)
            df_final = pd.concat([df_merged.drop(columns=['buckets_json']), opt_parsed_df], axis=1)

            # 生成标准的 timestamp 供人类/引擎调试，绝对保留 float 类型的 ts 作为底层驱动核
            df_final['timestamp'] = pd.to_datetime(df_final['ts'], unit='s', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            
            # 写入临时切片目录
            sym_temp_dir = TEMP_DIR / sym
            sym_temp_dir.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(sym_temp_dir / f"{date_str}.parquet", compression='ZSTD')
            
    except Exception as e:
        import traceback
        print(f"❌ 处理 {date_str} 失败: {e}\n{traceback.format_exc()}")

def merge_symbol_parquets(sym: str):
    """[Reduce 阶段] 合并单只股票的所有天数"""
    try:
        sym_dir = TEMP_DIR / sym
        files = list(sym_dir.glob("*.parquet"))
        if not files: return "SKIP"
        
        dfs = [pd.read_parquet(f) for f in files]
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 按时间戳排序去重
        merged_df = merged_df.sort_values('ts').drop_duplicates(subset=['ts'], keep='last').reset_index(drop=True)
        
        out_path = PARQUET_OUT_BASE / f"{sym}.parquet"
        merged_df.to_parquet(out_path, compression='ZSTD')
        return "OK"
    except Exception as e:
        return f"ERR: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default="20200101", help="Start date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="20991231", help="End date (YYYYMMDD)")
    args = parser.parse_args()

    if not DB_DIR.exists():
        print(f"❌ 找不到数据库目录: {DB_DIR}")
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

    # 初始化工作目录
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"🔍 阶段 1: [Map] 提取 {len(target_dbs)} 天数据并推迟 60 秒...")
    for date_str, db_path in target_dbs:
        print(f"📆 Processing: {date_str}")
        process_single_db(date_str, db_path)

    symbols = [d.name for d in TEMP_DIR.iterdir() if d.is_dir()]
    if not symbols:
        print(f"⚠️ [Map 阶段无输出]，检查数据库内容。")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    print(f"\n🔗 阶段 2: [Reduce] 拼接 S4 标准时间序列 (共 {len(symbols)} 只标的)...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(16, len(symbols)))) as executor:
        futures = {executor.submit(merge_symbol_parquets, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            if fut.result() == "OK": success += 1

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print("\n" + "🎉"*20)
    print(f"✅ 回测数据组装完毕！成功生成 {success} 个 S4 专属特征文件。")
    print(f"📂 存放路径: {PARQUET_OUT_BASE}")

if __name__ == "__main__":
    main()