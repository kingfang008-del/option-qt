import sqlite3
import json
import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
from datetime import datetime
import pytz

# ================= 路径配置 =================
PROJECT_ROOT = Path("/Users/fangshuai/Documents/GitHub/option-qt")
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production/baseline"))
sys.path.append(str(PROJECT_ROOT / "production/baseline/DAO"))

DB_PATH = PROJECT_ROOT / "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
TARGET_TS = 1767366000 # 2026-01-02 10:00:00 NY
SYMBOL = "NVDA"

try:
    from utils.greeks_math import calculate_bucket_greeks
except ImportError:
    # 兼容路径
    sys.path.append(str(PROJECT_ROOT / "production"))
    from utils.greeks_math import calculate_bucket_greeks

def run_standalone_audit():
    print(f"📊 [IV 深度审计] 目标时间: 2026-01-02 10:00:00 | TS: {TARGET_TS}")
    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. 提取价格
    res_price = pd.read_sql(f"SELECT close FROM market_bars_1s WHERE ts={TARGET_TS} AND symbol='{SYMBOL}'", conn)
    if res_price.empty:
        print("❌ 未找到价格数据。")
        return
    price = res_price.iloc[0]['close']
    
    # 2. 提取期权快照
    res_snap = pd.read_sql(f"SELECT buckets_json FROM option_snapshots_1s WHERE ts={TARGET_TS} AND symbol='{SYMBOL}'", conn)
    if res_snap.empty:
        print("❌ 未找到期权快照。")
        return
    snap_json = res_snap.iloc[0]['buckets_json']
    blob = json.loads(snap_json)
    buckets_orig = np.array(blob['buckets'])
    contracts = blob['contracts']
    conn.close()

    print(f"✅ 数据加载成功: Stock Price = {price:.2f} | Buckets: {buckets_orig.shape}")

    # 3. 实验不同的参数组合
    # 注意：S1 基准 IV 是 0.350150
    # S2 当前 IV 是 0.347564
    scenarios = [
        {"name": "基准 (R=0.045, Days=365, T=7d)", "r": 0.045, "days": 365, "T_plus": 0},
        {"name": "低利率 (R=0.040, Days=365)", "r": 0.40, "days": 365, "T_plus": 0},
        {"name": "高利率 (R=0.050, Days=365)", "r": 0.50, "days": 365, "T_plus": 0},
        {"name": "精确年 (R=0.045, Days=365.25)", "r": 0.045, "days": 365.25, "T_plus": 0},
    ]

    print("-" * 85)
    print(f"{'实验场景':<40} | {'IV (ATM Call)':<15}")
    print("-" * 85)

    for sc in scenarios:
        # 复制一份 buckets 避免污染
        temp_buckets = buckets_orig.copy()
        T_base = 7 / sc['days']
        T_final = float(T_base + sc['T_plus'])
        
        # 尝试重算
        calculate_bucket_greeks(temp_buckets, price, T=T_final, r=sc['r'], contracts=contracts, current_ts=TARGET_TS)
        
        # 提取第 3 行第 7 列 (ATM Call IV) - Row 2 in 0-indexed
        iv = temp_buckets[2][7] if len(temp_buckets) > 2 else 0.0
        print(f"{sc['name']:<40} | {iv:<15.6f}")

if __name__ == "__main__":
    run_standalone_audit()
