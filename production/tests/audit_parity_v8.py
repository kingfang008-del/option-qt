#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
audit_parity_v8.py
用于对比 1s 和 1m 模式生成的 Alpha 日志差异。
主要检查：
1. 时间戳对齐情况 (ts)
2. Alpha 原始分数差异 (alpha)
3. 价格一致性 (price)
4. 波动率一致性 (vol_z)
"""

import sqlite3
import pandas as pd
from pathlib import Path

# --- 配置区域 ---
DB_1M = Path("/home/kingfang007/quant_project/data/history_sqlite_1m/market_20260102.db")
DB_1S = Path("/home/kingfang007/quant_project/data/history_sqlite_1s/market_20260102.db")
REPORT_PATH = Path("/home/kingfang007/option-qt/divergence_report_v8.md")

def load_alpha_logs(db_path):
    print(f"📡 Loading from {db_path.name}...")
    conn = sqlite3.connect(db_path)
    # 提取 alpha_logs
    df = pd.read_sql("SELECT ts, datetime_ny, symbol, alpha, iv, price, vol_z FROM alpha_logs", conn)
    conn.close()
    return df

def run_audit():
    if not DB_1M.exists() or not DB_1S.exists():
        print(f"❌ Error: Database files NOT found!")
        if not DB_1M.exists(): print(f"   Missing: {DB_1M}")
        if not DB_1S.exists(): print(f"   Missing: {DB_1S}")
        return

    # 1. 加载数据
    df_1m = load_alpha_logs(DB_1M)
    df_1s = load_alpha_logs(DB_1S)

    print(f"✅ Loaded 1M: {len(df_1m)} rows, 1S: {len(df_1s)} rows")

    # 2. 对齐
    # 我们只关心整分钟点（1m 数据的时刻）在 1s 数据中是否存在且一致
    # 注意：1s 数据的 alpha_logs 可能由于 is_new_minute 逻辑也是每分钟一条
    # Join on ts, symbol
    merged = pd.merge(
        df_1m, df_1s, 
        on=['ts', 'symbol'], 
        suffixes=('_1m', '_1s'),
        how='inner'
    )

    print(f"✅ Merged aligned rows: {len(merged)}")

    if merged.empty:
        print("❌ No overlapping (ts, symbol) pairs found! Check if timestamps are shifted.")
        # 探测是否有 60s 偏移
        df_1s_shifted = df_1s.copy()
        df_1s_shifted['ts'] += 60.0
        merged_shift = pd.merge(df_1m, df_1s_shifted, on=['ts', 'symbol'], how='inner')
        if not merged_shift.empty:
            print(f"⚠️ Found {len(merged_shift)} matches AFTER 60s shift. The 60s discrepancy persists in these DBs!")
        return

    # 3. 计算差异
    merged['diff_alpha'] = merged['alpha_1s'] - merged['alpha_1m']
    merged['diff_price'] = merged['price_1s'] - merged['price_1m']
    merged['diff_vol_z'] = merged['vol_z_1s'] - merged['vol_z_1m']

    # 4. 生成报告
    stats = {
        'total_aligned': len(merged),
        'alpha_mae': merged['diff_alpha'].abs().mean(),
        'alpha_max': merged['diff_alpha'].abs().max(),
        'price_mae': merged['diff_price'].abs().mean(),
        'vol_z_mae': merged['diff_vol_z'].abs().mean(),
    }

    report = []
    report.append("# 1s vs 1m Alpha Parity Audit Report")
    report.append(f"Date: 2026-01-02")
    report.append(f"")
    report.append(f"## Summary Metrics")
    report.append(f"| Metric | Value |")
    report.append(f"|---|---|")
    report.append(f"| Aligned Rows | {stats['total_aligned']} |")
    report.append(f"| Alpha MAE | {stats['alpha_mae']:.6f} |")
    report.append(f"| Alpha Max Error | {stats['alpha_max']:.6f} |")
    report.append(f"| Price MAE | {stats['price_mae']:.6f} |")
    report.append(f"| Vol_Z MAE | {stats['vol_z_mae']:.6f} |")
    report.append(f"")

    # 列出差异最大的 Top 10
    top_diffs = merged.sort_values('diff_alpha', key=abs, ascending=False).head(10)
    report.append(f"## Top 10 Alpha Divergences")
    report.append(f"| Time (NY) | Symbol | Alpha_1m | Alpha_1s | Diff |")
    report.append(f"|---|---|---|---|---|")
    for _, row in top_diffs.iterrows():
        report.append(f"| {row['datetime_ny_1m']} | {row['symbol']} | {row['alpha_1m']:.4f} | {row['alpha_1s']:.4f} | {row['diff_alpha']:.4f} |")

    # 检查价格差异（这通常意味着数据源不对齐）
    price_diffs = merged[merged['diff_price'].abs() > 0.01].sort_values('diff_price', key=abs, ascending=False).head(10)
    if not price_diffs.empty:
        report.append(f"")
        report.append(f"## ⚠️ Price Discrepancies (> 0.01)")
        report.append(f"| Time (NY) | Symbol | Price_1m | Price_1s | Diff |")
        report.append(f"|---|---|---|---|---|")
        for _, row in price_diffs.iterrows():
            report.append(f"| {row['datetime_ny_1m']} | {row['symbol']} | {row['price_1m']:.2f} | {row['price_1s']:.2f} | {row['diff_price']:.2f} |")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"🎉 Audit complete. Report saved to: {REPORT_PATH}")
    print(f"Summary: MAE Alpha = {stats['alpha_mae']:.6f}")

if __name__ == "__main__":
    run_audit()
