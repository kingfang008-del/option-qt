import sqlite3
import json
import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
from datetime import datetime
import pytz

# ================= 路径配置 (请确保与你的本地环境一致) =================
PROJECT_ROOT = Path("/Users/fangshuai/Documents/GitHub/option-qt")
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production/baseline"))
sys.path.append(str(PROJECT_ROOT / "production/baseline/DAO"))

# 数据库路径
DB_PATH = PROJECT_ROOT / "production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
SYMBOL = "NVDA"

# 尝试导入，如果失败则 Mock
try:
    from realtime_feature_engine import RealTimeFeatureEngine
    HAS_ENGINE = True
except Exception as e:
    print(f"⚠️ 无法加载 RealTimeFeatureEngine: {e}. 脚本将降级为手动 Greeks 计算审计。")
    HAS_ENGINE = False

# 导入底层希腊值计算函数
try:
    from utils.greeks_math import calculate_bucket_greeks
except ImportError:
    # 兼容逻辑：如果 production/utils 没在 path 里
    sys.path.append(str(PROJECT_ROOT / "production"))
    from utils.greeks_math import calculate_bucket_greeks

def run_nvda_audit():
    print(f"🔍 [AUDIT] 正在扫描 NVDA 数据: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"❌ 数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    
    # 选取 10:00:00 前后的 10 秒数据进行深度肉眼审计 (TS: 1767366000)
    start_ts = 1767366000 - 10  # 约 09:59:50
    end_ts = 1767366000 + 65    # 约 10:01:05
    
    df_bars = pd.read_sql(f"SELECT ts, open, high, low, close, volume FROM market_bars_1s WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    df_opts = pd.read_sql(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    conn.close()

    if df_bars.empty:
        print("❌ 未找到 NVDA 原始秒级数据。")
        return

    print(f"📊 成功加载 {len(df_bars)} 行行情数据和 {len(df_opts)} 行期权快照。")
    print("-" * 85)
    print(f"{'Time (NY)':<12} | {'TS':<10} | {'Stock':<7} | {'IV (ATM)':<8} | {'Delta':<7} | {'NewMin'}")
    print("-" * 85)

    ny_tz = pytz.timezone('America/New_York')

    for _, row in df_bars.iterrows():
        ts = int(row['ts'])
        price = row['close']
        dt_ny = datetime.fromtimestamp(ts, ny_tz)
        
        # 匹配期权快照
        opt_match = df_opts[df_opts['ts'] == ts]
        iv_val = 0.0
        delta_val = 0.0
        
        if not opt_match.empty:
            buckets_json = opt_match.iloc[0]['buckets_json']
            if isinstance(buckets_json, str):
                blob = json.loads(buckets_json)
            else:
                blob = {}
            
            buckets = np.array(blob.get('buckets', []))
            contracts = blob.get('contracts', [])
            
            # 使用基准 RFR 或尝试读取
            rfr = 0.045 
            
            # 这里的 t (Time to Expiration) 也是一个关键变量。计算 1/2 的 IV 基准
            # 假设到期日是 7 天后
            calculate_bucket_greeks(buckets, price, T=7/365.0, r=rfr, contracts=contracts, current_ts=ts)
            
            # 提取 ATM Call 的 IV (Row 2, Index 7)
            iv_val = buckets[2][7] if len(buckets) > 2 else 0.0
            delta_val = buckets[2][1] if len(buckets) > 2 else 0.0

        # 判断是否是分钟边界
        is_new_minute = (ts % 60 == 0)
        
        # 标记出 user 提到的 jitter 点
        jitter_mark = " <-- JITTER" if ts == 1767366001 else ""
        
        print(f"{dt_ny.strftime('%H:%M:%S'):<12} | {ts:<10} | {price:<7.2f} | {iv_val:<8.4f} | {delta_val:<7.4f} | {is_new_minute}{jitter_mark}")

if __name__ == "__main__":
    run_nvda_audit()
