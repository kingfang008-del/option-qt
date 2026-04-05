#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: convert_pgsql_to_parquet.py
描述: 全量 PostgreSQL 提纯与 Parquet 融合工具 (Flat File Structure)
功能:
    1. 自动连接 PostgreSQL 数据库加载数据
    2. 按天提取数据，确保在大数据量下内存安全
    3. 解决 ts 类型不匹配问题 (强制 float64)
    4. 采用 Map-Reduce 架构，将多日数据按标的无缝纵向拼接
    5. 输出格式与 s5 离线管道完全一致: root/AAPL.parquet
    6. 提取期权真实成交量 (Volume)，用于回测的流动性拦截。
"""

import os
import psycopg2
import pandas as pd
import numpy as np
import json
import argparse
import shutil
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import pytz

# 引入项目根目录以加载 config
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config import PG_DB_URL
except ImportError:
    print("❌ 无法导入 config 模块。请确保在正确的目录下执行。")
    sys.exit(1)

# ================= 配置区域 =================
PROJECT_ROOT = Path.home() / "quant_project" / "data"
TEMP_DIR = PROJECT_ROOT / "temp_pg_convert"
PARQUET_OUT_BASE = PROJECT_ROOT / "rl_feed_parquet_batch"

 

def parse_option_buckets(json_str):
    """提取期权快照的核心字段 (含 Volume)"""
    try:
        # 如果从 postgres 返回已经是 dict (因为 jsonb 类型)，或者是 str
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
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

def get_sqlalchemy_uri(postgres_connection_string):
    """
    将 psycopg2 格式的 DSN 连接串 (dbname=x user=y host=z ...) 
    转换为 SQLAlchemy 支持的标准 URI (postgresql://user:pass@host:port/dbname)
    从而消除 pandas read_sql 的警告。
    """
    parts = dict(p.split('=') for p in postgres_connection_string.split())
    # 提取字段并构建 URI
    uri = f"postgresql://{parts.get('user', '')}:{parts.get('password', '')}@{parts.get('host', 'localhost')}:{parts.get('port', '5432')}/{parts.get('dbname', '')}"
    return uri

def process_single_day(date_str: str, start_ts: float, end_ts: float):
    """[Map 阶段] 处理单天 PGSQL，生成各标的当天碎片 Parquet"""
    try:
        # => 预生成给 pandas read_sql 使用的通用 URI 连接串
        db_uri = get_sqlalchemy_uri(PG_DB_URL)
        
        # 1. 提取并强制转换 float，彻底解决 merge_asof 崩溃问题
        query_alpha = f"SELECT symbol, ts, alpha as alpha_score FROM alpha_logs WHERE ts >= {start_ts} AND ts < {end_ts}"        
        df_alpha = pd.read_sql(query_alpha, db_uri)
        if df_alpha.empty:
            return

        df_alpha['ts'] = df_alpha['ts'].astype(float)
        # 🚀 [新增核心逻辑]：预计算截面 Alpha Z-Score (完美模拟实盘的截面打分)
        # 按 ts 分组，计算当前这一分钟全市场的 mean 和 std，映射到 [-5.0, 5.0]
        df_alpha['alpha_score'] = df_alpha.groupby('ts')['alpha_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        ).clip(-5.0, 5.0)
        
        query_bars = f"SELECT symbol, ts, close FROM market_bars_1m WHERE ts >= {start_ts} AND ts < {end_ts}"
        df_bars = pd.read_sql(query_bars, db_uri)
        df_bars['ts'] = df_bars['ts'].astype(float)
        
        query_opts = f"SELECT symbol, ts, buckets_json FROM option_snapshots_1m WHERE ts >= {start_ts} AND ts < {end_ts}"
        df_opts = pd.read_sql(query_opts, db_uri)
        df_opts['ts'] = df_opts['ts'].astype(float)

        # 计算大盘动量
        df_spy = df_bars[df_bars['symbol'] == 'SPY'].sort_values('ts').copy()
        df_spy['spy_roc_5min'] = df_spy['close'].pct_change(5).fillna(0)
        spy_map = dict(zip(df_spy['ts'], df_spy['spy_roc_5min']))
        
        df_qqq = df_bars[df_bars['symbol'] == 'QQQ'].sort_values('ts').copy()
        df_qqq['qqq_roc_5min'] = df_qqq['close'].pct_change(5).fillna(0)
        qqq_map = dict(zip(df_qqq['ts'], df_qqq['qqq_roc_5min']))

        symbols = df_alpha['symbol'].unique()
        for sym in symbols:
            s_alpha = df_alpha[df_alpha['symbol'] == sym].sort_values('ts').copy()
            s_bar = df_bars[df_bars['symbol'] == sym].sort_values('ts').copy()
            s_opt = df_opts[df_opts['symbol'] == sym].sort_values('ts').copy()

            if s_bar.empty: continue

            # 🚀 [适配终极 Z-Score] 动态计算 fast_vol，并将其直接转化为 Z-Score 落盘
            s_bar['log_ret'] = np.log(s_bar['close'] / s_bar['close'].shift(1).replace(0, np.nan)).fillna(0.0)
            raw_vol = s_bar['log_ret'].ewm(span=20, min_periods=2).std().fillna(0.0)
            
            # 使用较长的平滑窗口来模拟 Orchestrator 的慢速基准追踪
            vol_mean = raw_vol.ewm(span=1000, min_periods=10).mean()
            vol_std = raw_vol.ewm(span=1000, min_periods=10).std()
            
            # 直接写入标准化后的 Vol Z-Score！
            s_bar['fast_vol'] = ((raw_vol - vol_mean) / (vol_std + 1e-6)).clip(-5.0, 5.0)
            # 时间向后对齐 (注意这里把计算好的 fast_vol 一起 merge 进主表)
            df_merged = pd.merge_asof(s_alpha, s_bar[['ts', 'close', 'fast_vol']], on='ts', direction='backward')
            df_merged = pd.merge_asof(df_merged, s_opt[['ts', 'buckets_json']], on='ts', direction='backward', tolerance=300.0)

            df_merged['buckets_json'] = df_merged['buckets_json'].fillna("{}")

            #df_merged['fast_vol'] = df_merged['vol_z'] * FAST_VOL_STD + FAST_VOL_MEAN
            df_merged['spy_roc_5min'] = df_merged['ts'].map(spy_map).fillna(0)
            df_merged['qqq_roc_5min'] = df_merged['ts'].map(qqq_map).fillna(0)

            df_merged[[
                'opt_0', 'opt_5', 'feed_put_vol', 'opt_7', 'feed_put_bid', 'feed_put_ask', 'feed_put_bid_size', 'feed_put_ask_size', 'opt_0_id',
                'opt_8', 'opt_13', 'feed_call_vol', 'opt_15', 'feed_call_bid', 'feed_call_ask', 'feed_call_bid_size', 'feed_call_ask_size', 'opt_8_id'
            ]] = df_merged['buckets_json'].apply(parse_option_buckets)

            df_final = df_merged[[
                'ts', 'close', 'alpha_score', 'fast_vol', 'spy_roc_5min', 'qqq_roc_5min',
                'opt_0', 'opt_8', 'opt_0_id', 'opt_8_id', 'opt_5', 'opt_7', 'opt_13', 'opt_15',
                'feed_put_vol', 'feed_call_vol',
                'feed_put_bid', 'feed_put_ask', 'feed_put_bid_size', 'feed_put_ask_size',
                'feed_call_bid', 'feed_call_ask', 'feed_call_bid_size', 'feed_call_ask_size'
            ]].copy()

            
            # [终极修复] 保留绝对 Unix 时间戳 ts 供底层引擎使用，timestamp 仅作参考
            df_final['timestamp'] = pd.to_datetime(df_final['ts'], unit='s', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            
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

def get_db_date_range():
    """获取数据库中最小和最大的日期"""
    try:
        conn = psycopg2.connect(PG_DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(ts), MAX(ts) FROM alpha_logs")
        min_ts, max_ts = cursor.fetchone()
        conn.close()
        return min_ts, max_ts
    except Exception as e:
        print(f"❌ 获取数据库日期范围失败: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default="20260101", help="Start date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="", help="End date (YYYYMMDD)")
    args = parser.parse_args()

    min_ts, max_ts = get_db_date_range()
    if min_ts is None or max_ts is None:
        print("❌ 数据库中没有找到 alpha_logs 数据。")
        return

    ny_tz = pytz.timezone('America/New_York')
    
    db_start_date = datetime.fromtimestamp(min_ts, tz=pytz.utc).astimezone(ny_tz).date()
    db_end_date = datetime.fromtimestamp(max_ts, tz=pytz.utc).astimezone(ny_tz).date()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y%m%d").date()
    else:
        start_date = db_start_date
        
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y%m%d").date()
    else:
        end_date = db_end_date

    # 生成要处理的日期列表
    target_dates = []
    curr_date = start_date
    while curr_date <= end_date:
        # 转为美东时间的当天 00:00:00 到次日 00:00:00 的 ts
        dt_start = ny_tz.localize(datetime.combine(curr_date, datetime.min.time()))
        dt_end = dt_start + timedelta(days=1)
        target_dates.append((curr_date.strftime("%Y%m%d"), dt_start.timestamp(), dt_end.timestamp()))
        curr_date += timedelta(days=1)
            
    if not target_dates:
        print(f"❌ 未在区间内找到任何需要处理的日期。")
        return

    # 初始化目录
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"🔍 阶段 1: [Map] 提取 {len(target_dates)} 天数据并按天打碎...")
    for date_str, start_ts, end_ts in tqdm(target_dates, desc="📆 逐日提取"):
        process_single_day(date_str, start_ts, end_ts)

    # 获取所有生成的标的
    symbols = [d.name for d in TEMP_DIR.iterdir() if d.is_dir()]
    
    if not symbols:
        print("❌ 没有生成任何标的数据，可能是由于日期区间内无数据。")
        return
        
    print(f"\n🔗 阶段 2: [Reduce] 合并 {len(symbols)} 只标的的时间序列...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(symbols))) as executor:
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
