#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 S4 确定性总线批量回测工具
用法: python3.10 run_batch_s4.py [--limit N]
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
import shutil
import time

# 设置环境
os.environ['PARITY_MODE'] = 'PLAN_A'
os.environ['RUN_MODE'] = 'BACKTEST'

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
HIST_DIR = BASE_DIR / "history_sqlite_1m"
S4_SCRIPT = BASE_DIR / "s4_run_historical_replay.py"
LOG_DIR = Path.home() / "quant_project/logs"
RAW_TRADES_FILE = LOG_DIR / "replay_trades_v8.csv"
BATCH_OUT_DIR = BASE_DIR / "batch_results"

def get_all_dates():
    """从文件名中提取所有日期"""
    dbs = sorted(HIST_DIR.glob("market_*.db"))
    dates = []
    for db in dbs:
        # market_20260312.db -> 20260312
        date_str = db.name.split('_')[1].split('.')[0]
        dates.append(date_str)
    return dates

def run_single_date(date_str):
    """运行单日回测"""
    print(f"\n" + "="*60)
    print(f"📅 Running Backtest for {date_str} ...")
    print("="*60)
    
    cmd = [
        "python3.10", str(S4_SCRIPT),
        "--date", date_str
    ]
    
    start_time = time.time()
    try:
        # 使用 subprocess.run 确保串行执行
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"✅ {date_str} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {date_str} FAILED with exit code {e.returncode}")
        return False

def aggregate_results(processed_dates):
    """合并所有交易记录"""
    print("\n" + "="*60)
    print("📊 Aggregating Results ...")
    print("="*60)
    
    all_dfs = []
    for date in processed_dates:
        csv_path = BATCH_OUT_DIR / f"trades_{date}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
    
    if not all_dfs:
        print("⚠️ No trades found in any processed dates.")
        return None
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    # 按平仓时间排序
    if 'exit_ts' in master_df.columns:
        master_df = master_df.sort_values('exit_ts')
    
    master_path = LOG_DIR / "master_trades_s4_batch.csv"
    master_df.to_csv(master_path, index=False)
    print(f"💾 Master trades saved to {master_path}")
    return master_df

def print_summary(df):
    """打印总报表"""
    if df is None or df.empty: return
    
    total_trades = len(df)
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).mean()
    
    # 按天统计
    df['date_obj'] = pd.to_datetime(df['date'])
    daily_pnl = df.groupby('date_obj')['pnl'].sum()
    
    # 模拟资金曲线 (初始 50,000)
    capital = 50000.0
    daily_equity = capital + daily_pnl.cumsum()
    total_ret = (daily_equity.iloc[-1] - capital) / capital
    
    print("\n" + "="*60)
    print("🧠 S4 BATCH BACKTEST FINAL REPORT")
    print("="*60)
    print(f"Processed Days: {len(daily_pnl)}")
    print(f"Total Trades:   {total_trades}")
    print(f"Total PnL:      ${total_pnl:,.2f}")
    print(f"Total Return:   {total_ret:.2%}")
    print(f"Win Rate:       {win_rate:.2%}")
    print("-" * 60)
    
    # 打印每日明细
    print(f"{'Date':<12} | {'Trades':<6} | {'PnL ($)':<10} | {'ROI':<8}")
    print("-" * 60)
    for date, pnl in daily_pnl.items():
        d_str = date.strftime('%Y-%m-%d')
        count = len(df[df['date_obj'] == date])
        roi = pnl / capital
        print(f"{d_str:<12} | {count:<6} | {pnl:10.1f} | {roi:8.2%}")
    print("="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of days to run")
    args = parser.parse_args()

    # 初始化输出目录
    if BATCH_OUT_DIR.exists():
        shutil.rmtree(BATCH_OUT_DIR)
    BATCH_OUT_DIR.mkdir(parents=True)

    all_dates = get_all_dates()
    if args.limit:
        all_dates = all_dates[:args.limit]
    
    print(f"🚀 Found {len(all_dates)} dates to process.")
    
    processed_dates = []
    for date_str in all_dates:
        # 清理旧的记录，防止干扰
        if RAW_TRADES_FILE.exists():
            RAW_TRADES_FILE.unlink()
            
        success = run_single_date(date_str)
        if success and RAW_TRADES_FILE.exists():
            # 将当日结果备份
            shutil.copy(RAW_TRADES_FILE, BATCH_OUT_DIR / f"trades_{date_str}.csv")
            processed_dates.append(date_str)
        
    master_df = aggregate_results(processed_dates)
    print_summary(master_df)

if __name__ == "__main__":
    main()
