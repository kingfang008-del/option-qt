
import redis
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os

 
# 🚀 [核心修复] 强制同步仿真测试的运行模式
os.environ['RUN_MODE'] = 'LIVEREPLAY'

try:
    from config import get_redis_db, REDIS_CFG, TARGET_SYMBOLS
except ImportError:
    REDIS_CFG = {'host': 'localhost', 'port': 6379}
    def get_redis_db(): return 1 # 仿真默认 DB 1
    TARGET_SYMBOLS = ['NVDA', 'AAPL', 'SPY']

NY_TZ = pytz.timezone('America/New_York')

def get_ny_ts(hour, minute, second=0):
    """构造当日 (2026-01-02) 的纽约时间戳"""
    dt = datetime(2026, 1, 2, hour, minute, second)
    return int(NY_TZ.localize(dt).timestamp())

def compare_buckets(r_raw, s_raw, spot_price):
    """按固定 bucket 槽位对比 PUT/CALL ATM（与 config.BUCKET_SPECS 保持一致）"""
    try:
        def extract_tagged_info(data_raw):
            buckets = data_raw.get('buckets', []) if isinstance(data_raw, dict) else data_raw
            if not buckets or not isinstance(buckets, list):
                return None, None

            def parse_row(idx):
                if idx >= len(buckets):
                    return None
                row = buckets[idx]
                if not isinstance(row, list) or len(row) < 10:
                    return None
                return {"iv": row[7], "bid": row[8], "ask": row[9]}

            # config.py 定义:
            # 0=PUT_ATM, 2=CALL_ATM
            return parse_row(2), parse_row(0)

        r_c, r_p = extract_tagged_info(r_raw)
        s_c, s_p = extract_tagged_info(s_raw)
        
        # 结果对位输出:
        # (r_c_iv, s_c_iv, r_p_iv, s_p_iv,
        #  r_c_bid, s_c_bid, r_c_ask, s_c_ask,
        #  r_p_bid, s_p_bid, r_p_ask, s_p_ask)
        def v(obj, k): return obj[k] if obj else 0.0
        return (
            v(r_c, 'iv'), v(s_c, 'iv'), v(r_p, 'iv'), v(s_p, 'iv'),
            v(r_c, 'bid'), v(s_c, 'bid'), v(r_c, 'ask'), v(s_c, 'ask'),
            v(r_p, 'bid'), v(s_p, 'bid'), v(r_p, 'ask'), v(s_p, 'ask')
        )
    except Exception as e:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def run_parity_audit(symbol='NVDA', start_h=9, start_m=55, end_h=10, end_m=20):
    # 路径自动适配 (优先使用服务器路径)
    db_path = '/home/kingfang007/quant_project/data/history_sqlite_1m/market_20260102.db'
    if not os.path.exists(db_path):
        db_path = '/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1m/market_20260102.db'

    r_db = get_redis_db()
    r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=r_db)
    
    print(f"📡 [Redis Connectivity] {REDIS_CFG['host']}:{REDIS_CFG['port']} | DB: {r_db}")
    
    # 🚀 [新功能] Redis 存量探测
    all_keys = r.hkeys(f"BAR:1M:{symbol}")
    if all_keys:
        t_list = sorted([int(k.decode()) for k in all_keys])
        print(f"📊 [Redis Sniffing] Found {len(all_keys)} keys for {symbol}.")
        print(f"📊 [Sample Keys] Earliest: {t_list[0]} | Latest: {t_list[-1]}")
    else:
        print(f"❌ [Redis Sniffing] NO KEYS FOUND in BAR:1M:{symbol} on DB {r_db}!")
    
    print(f"🚀 [Engine Parity Audit] Symbol: {symbol} | Redis DB: {r_db}")
    print(f"📅 Range: {start_h:02d}:{start_m:02d} -> {end_h:02d}:{end_m:02d} (NY Time)")
    print("-" * 110)
    header = f"{'Time':<10} | {'Type':<6} | {'Field':<10} | {'Redis':<12} | {'SQLite':<12} | {'Diff':<12} | {'Status'}"
    print(header)
    print("-" * 110)

    # 构造时间序列
    curr_minute = get_ny_ts(start_h, start_m)
    end_ts = get_ny_ts(end_h, end_m)
    
    conn = sqlite3.connect(db_path)
    
    while curr_minute <= end_ts:
        time_str = datetime.fromtimestamp(curr_minute, NY_TZ).strftime('%H:%M:%S')
        key_str = str(int(curr_minute))
        
        # 1. Check Stock Bar
        r_bar_raw = r.hget(f"BAR:1M:{symbol}", key_str)
        sql_bar = pd.read_sql(f"SELECT * FROM market_bars_1m WHERE symbol='{symbol}' AND ts={curr_minute}", conn)
        
        current_spot = 0
        if not sql_bar.empty: 
            current_spot = sql_bar.iloc[0]['close']

        if r_bar_raw and not sql_bar.empty:
            rb = json.loads(r_bar_raw.decode('utf-8'))
            sb = sql_bar.iloc[0]
            r_c, s_c = rb['close'], sb['close']
            r_v, s_v = rb['volume'], sb['volume']
            c_diff = abs(r_c - s_c)
            v_diff = abs(r_v - s_v)
            c_st = "✅" if c_diff < 0.001 else "❌"
            v_tag = "FLAT" if r_v == 0 else "LIVE"
            v_st = "✅" if v_diff < 0.1 else "❌"
            print(f"{time_str:<10} | STOCK  | Close ({v_tag}) | {r_c:<12.4f} | {s_c:<12.4f} | {c_diff:<12.6f} | {c_st}")
            if v_diff > 0.1:
                print(f"{'':<10} | STOCK  | Volume     | {r_v:<12.1f} | {s_v:<12.1f} | {v_diff:<12.1f} | {v_st}")
             
        # 2. Check Option Snapshot
        r_opt_raw = r.hget(f"BAR_OPT:1M:{symbol}", key_str)
        sql_opt = pd.read_sql(f"SELECT * FROM option_snapshots_1m WHERE symbol='{symbol}' AND ts={curr_minute}", conn)
        
        if r_opt_raw and not sql_opt.empty:
            ro = json.loads(r_opt_raw.decode('utf-8'))
            so_json = json.loads(sql_opt.iloc[0]['buckets_json'])
            
            # 🚀 [终极对位] 注入当前时刻的股价作为 ATM 探测锚点
            # 🚀 [终极对位] 注入完整字典，激活语义识别逻辑
            (
                r_c_iv, s_c_iv, r_p_iv, s_p_iv,
                r_c_bid, s_c_bid, r_c_ask, s_c_ask,
                r_p_bid, s_p_bid, r_p_ask, s_p_ask
            ) = compare_buckets(ro, so_json, current_spot)
            
            c_iv_diff = abs(r_c_iv - s_c_iv)
            p_iv_diff = abs(r_p_iv - s_p_iv)
            c_bid_diff = abs(r_c_bid - s_c_bid)
            c_ask_diff = abs(r_c_ask - s_c_ask)
            p_bid_diff = abs(r_p_bid - s_p_bid)
            p_ask_diff = abs(r_p_ask - s_p_ask)
            
            c_iv_st = "✅" if c_iv_diff < 0.0001 else "❌"
            p_iv_st = "✅" if p_iv_diff < 0.0001 else "❌"
            c_bid_st = "✅" if c_bid_diff < 0.001 else "❌"
            c_ask_st = "✅" if c_ask_diff < 0.001 else "❌"
            p_bid_st = "✅" if p_bid_diff < 0.001 else "❌"
            p_ask_st = "✅" if p_ask_diff < 0.001 else "❌"
            
            print(f"{'':<10} | OPTION | ATM_C_IV   | {r_c_iv:<12.6f} | {s_c_iv:<12.6f} | {c_iv_diff:<12.8f} | {c_iv_st}")
            print(f"{'':<10} | OPTION | ATM_P_IV   | {r_p_iv:<12.6f} | {s_p_iv:<12.6f} | {p_iv_diff:<12.8f} | {p_iv_st}")
            print(f"{'':<10} | OPTION | CALL_BID   | {r_c_bid:<12.4f} | {s_c_bid:<12.4f} | {c_bid_diff:<12.6f} | {c_bid_st}")
            print(f"{'':<10} | OPTION | CALL_ASK   | {r_c_ask:<12.4f} | {s_c_ask:<12.4f} | {c_ask_diff:<12.6f} | {c_ask_st}")
            print(f"{'':<10} | OPTION | PUT_BID    | {r_p_bid:<12.4f} | {s_p_bid:<12.4f} | {p_bid_diff:<12.6f} | {p_bid_st}")
            print(f"{'':<10} | OPTION | PUT_ASK    | {r_p_ask:<12.4f} | {s_p_ask:<12.4f} | {p_ask_diff:<12.6f} | {p_ask_st}")
        elif not r_opt_raw and not sql_opt.empty:
             print(f"{time_str:<10} | OPTION | MISSING    | {'EMPTY':<12} | {'DATA_EXIST':<12} | {'N/A':<12} | ⚠️ Redis Missing")

        print("-" * 110)
        curr_minute += 60

    conn.close()
    print("\n💡 Tip: If Close checks PASS but IV checks FAIL, investigate Greeks re-calculation logic in RealTimeFeatureEngine.")
    print("💡 Tip: If Redis is MISSING data, ensure s2_run_realtime_replay_sqlite_1s.py --turbo has finished processing that time period.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='NVDA')
    args = parser.parse_args()
    
    run_parity_audit(symbol=args.symbol)
