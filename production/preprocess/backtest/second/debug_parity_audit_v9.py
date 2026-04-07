import sqlite3
import json
import pandas as pd
import numpy as np
import sys, os
from pathlib import Path
from datetime import datetime
import pytz

# ================= 统一路径对齐 =================
# 根据 s2_run_realtime_replay_sqlite_1s.py 对齐路径
PROJECT_ROOT = Path("/Users/fangshuai/Documents/GitHub/option-qt")
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production/baseline"))
sys.path.append(str(PROJECT_ROOT / "production/baseline/DAO"))

from feature_compute_service_v8 import FeatureComputeService
from config import TARGET_SYMBOLS, REDIS_CFG

# 🚀 强制设置环境变量为回放模式
os.environ['RUN_MODE'] = 'LIVEREPLAY'

# 数据库路径 (使用 find 找到的真实路径)
DB_PATH = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1s/market_20260102.db"
DATE_STR = "20260102"
SYMBOL = "NVDA"

# 配置路径
base_dir = PROJECT_ROOT
config_paths = {
    'fast': str(base_dir / "daily_backtest/fast_feature.json"), 
    'slow': str(base_dir / "daily_backtest/slow_feature.json")
}

def audit_nvda_parity():
    print(f"🔍 [AUDIT] Connecting to DB: {DB_PATH}")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    
    # 1. 抓取时间窗口 (10:00:00 附近)
    NY_TZ = pytz.timezone('America/New_York')
    start_dt = NY_TZ.localize(datetime.strptime(DATE_STR + " 09:59:00", "%Y%m%d %H:%M:%S"))
    end_dt = NY_TZ.localize(datetime.strptime(DATE_STR + " 10:02:00", "%Y%m%d %H:%M:%S"))
    start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
    
    print(f"📅 [AUDIT] Window: 09:59:00 ({start_ts}) to 10:02:00 ({end_ts})")

    # 2. 读取数据
    df_bars = pd.read_sql(f"SELECT * FROM market_bars_1s WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    df_opts = pd.read_sql(f"SELECT * FROM option_snapshots_1s WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    df_bars_5m = pd.read_sql(f"SELECT * FROM market_bars_5m WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    df_opts_5m = pd.read_sql(f"SELECT * FROM option_snapshots_5m WHERE ts >= {start_ts} AND ts <= {end_ts} AND symbol='{SYMBOL}'", conn)
    conn.close()

    if df_bars.empty:
        print("❌ [AUDIT] No data found for NVDA in this window.")
        return

    # 3. 初始化 Feature Engine
    print("🛠️ [AUDIT] Initializing FeatureComputeService...")
    # 注意: FeatureComputeService 需要 redis 实例，这里我们传入真实配置，但内部调用 run_compute_cycle(return_payload=True)
    svc = FeatureComputeService(REDIS_CFG, TARGET_SYMBOLS, config_paths)
    
    # 模拟 09:30 的清洗 (跳过，因为我们就在 10:00 附近)
    svc._premarket_flushed_date = datetime.strptime(DATE_STR, "%Y%m%d").date()

    # 数据索引化
    full_ts = sorted(list(set(df_bars['ts']) | set(df_opts['ts'])))
    
    print(f"\n{'Timestamp':<20} | {'Stock':<8} | {'IV':<8} | {'Alpha':<10} | {'IsNewMin'}")
    print("-" * 70)

    last_5m_stock = {}
    last_5m_opts = {}

    for ts in full_ts:
        # 组装 Batch Payload (与 s2_run_realtime_replay_sqlite_1s.py 逻辑一致)
        bar_row = df_bars[df_bars['ts'] == ts]
        opt_row = df_opts[df_opts['ts'] == ts]
        bar5_row = df_bars_5m[df_bars_5m['ts'] == ts]
        opt5_row = df_opts_5m[df_opts_5m['ts'] == ts]
        
        if bar5_row.empty == False: 
            last_5m_stock = bar5_row.iloc[0].to_dict()
        if opt5_row.empty == False:
            blob = json.loads(opt5_row.iloc[0]['buckets_json']) if isinstance(opt5_row.iloc[0]['buckets_json'], str) else {}
            last_5m_opts = {'option_buckets_5m': blob.get('buckets', []), 'option_contracts_5m': blob.get('contracts', [])}

        tick_payload = {
            'ts': ts,
            'symbol': SYMBOL,
            'stock': bar_row.iloc[0].to_dict() if not bar_row.empty else {'open':0,'high':0,'low':0,'close':0,'volume':0},
        }
        if not opt_row.empty:
            blob = json.loads(opt_row.iloc[0]['buckets_json']) if isinstance(opt_row.iloc[0]['buckets_json'], str) else {}
            tick_payload['option_buckets'] = blob.get('buckets', [])
            tick_payload['option_contracts'] = blob.get('contracts', [])
        
        if last_5m_stock: tick_payload.update({'stock_5m': last_5m_stock})
        if last_5m_opts: tick_payload.update(last_5m_opts)

        # 4. 执行计算
        # 模拟 Redis Stream 发送行为 (batch 包装)
        import asyncio
        loop = asyncio.get_event_loop()
        
        # 处理数据
        loop.run_until_complete(svc.process_market_data([tick_payload]))
        
        # 执行推理
        # 🚨 我们需要设置 replay:current_ts 模拟逻辑时钟，否则 svc 内部 time.time() 会乱
        import redis
        r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        r.set("replay:current_ts", str(ts))

        res = loop.run_until_complete(svc.run_compute_cycle(ts_from_payload=ts, return_payload=True))
        
        if res:
            p_ts = res['ts']
            dt_str = datetime.fromtimestamp(p_ts, NY_TZ).strftime('%H:%M:%S')
            is_new = res.get('is_new_minute', False)
            
            # 提取 IV (NVDA 通常在 symbols 列表中的某个位置)
            try:
                sym_idx = res['symbols'].index(SYMBOL)
                stock_p = res['stock_price'][sym_idx]
                alpha = res['alpha_score'][sym_idx] if 'alpha_score' in res else 0.0
                
                # 从 buckets 获取 IV (根据 TAG_TO_INDEX, CALL_ATM 通常是 index 2)
                # buckets 结构: [6, 12]
                live_opt = res['live_options'].get(SYMBOL, {})
                buckets = live_opt.get('buckets', [])
                iv = buckets[2][7] if len(buckets) > 2 and len(buckets[2]) > 7 else 0.0
                
                if ts % 60 == 0 or is_new or ts % 10 == 0:
                    print(f"{dt_str} ({p_ts}) | {stock_p:<8.2f} | {iv:<8.4f} | {alpha:<10.4f} | {is_new}")
            except Exception as e:
                pass

if __name__ == "__main__":
    audit_nvda_parity()
