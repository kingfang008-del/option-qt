
import redis
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os
from pathlib import Path

 
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
                if not isinstance(row, list) or len(row) < 12:
                    return None
                return {
                    "last": row[0],
                    "delta": row[1],
                    "gamma": row[2],
                    "vega": row[3],
                    "theta": row[4],
                    "strike": row[5],
                    "volume": row[6],
                    "iv": row[7],
                    "bid": row[8],
                    "ask": row[9],
                    "bid_size": row[10],
                    "ask_size": row[11],
                }

            # config.py 定义:
            # 0=PUT_ATM, 2=CALL_ATM
            return parse_row(2), parse_row(0)

        r_c, r_p = extract_tagged_info(r_raw)
        s_c, s_p = extract_tagged_info(s_raw)

        def v(obj, k):
            return float(obj[k]) if obj and k in obj else 0.0

        return {
            'call': {k: (v(r_c, k), v(s_c, k)) for k in ['last', 'delta', 'gamma', 'vega', 'theta', 'strike', 'volume', 'iv', 'bid', 'ask', 'bid_size', 'ask_size']},
            'put': {k: (v(r_p, k), v(s_p, k)) for k in ['last', 'delta', 'gamma', 'vega', 'theta', 'strike', 'volume', 'iv', 'bid', 'ask', 'bid_size', 'ask_size']},
        }
    except Exception:
        zero_pair = (0.0, 0.0)
        return {
            'call': {k: zero_pair for k in ['last', 'delta', 'gamma', 'vega', 'theta', 'strike', 'volume', 'iv', 'bid', 'ask', 'bid_size', 'ask_size']},
            'put': {k: zero_pair for k in ['last', 'delta', 'gamma', 'vega', 'theta', 'strike', 'volume', 'iv', 'bid', 'ask', 'bid_size', 'ask_size']},
        }


def _print_row(time_str, typ, field, redis_val, sql_val, tol):
    diff = abs(float(redis_val) - float(sql_val))
    status = "✅" if diff <= tol else "❌"
    print(f"{time_str:<10} | {typ:<6} | {field:<12} | {redis_val:<12.6f} | {sql_val:<12.6f} | {diff:<12.8f} | {status}")

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
            _print_row(time_str, "STOCK", "Open", rb.get('open', 0.0), sb['open'], 0.001)
            _print_row("", "STOCK", "High", rb.get('high', 0.0), sb['high'], 0.001)
            _print_row("", "STOCK", "Low", rb.get('low', 0.0), sb['low'], 0.001)
            _print_row("", "STOCK", "Close", rb.get('close', 0.0), sb['close'], 0.001)
            _print_row("", "STOCK", "Volume", rb.get('volume', 0.0), sb['volume'], 0.1)
            if 'vwap' in rb:
                _print_row("", "STOCK", "VWAP", rb.get('vwap', 0.0), rb.get('vwap', 0.0), 0.001)
             
        # 2. Check Option Snapshot
        r_opt_raw = r.hget(f"BAR_OPT:1M:{symbol}", key_str)
        sql_opt = pd.read_sql(f"SELECT * FROM option_snapshots_1m WHERE symbol='{symbol}' AND ts={curr_minute}", conn)
        
        if r_opt_raw and not sql_opt.empty:
            ro = json.loads(r_opt_raw.decode('utf-8'))
            so_json = json.loads(sql_opt.iloc[0]['buckets_json'])

            bucket_cmp = compare_buckets(ro, so_json, current_spot)
            field_tols = {
                'last': 0.001, 'delta': 0.0001, 'gamma': 0.0001, 'vega': 0.0001, 'theta': 0.0001,
                'strike': 0.001, 'volume': 0.1, 'iv': 0.0001, 'bid': 0.001, 'ask': 0.001,
                'bid_size': 0.1, 'ask_size': 0.1
            }
            field_labels = {
                'last': 'LAST', 'delta': 'DELTA', 'gamma': 'GAMMA', 'vega': 'VEGA', 'theta': 'THETA',
                'strike': 'STRIKE', 'volume': 'VOLUME', 'iv': 'IV', 'bid': 'BID', 'ask': 'ASK',
                'bid_size': 'BID_SIZE', 'ask_size': 'ASK_SIZE'
            }

            first = True
            for side_key, side_prefix in [('call', 'CALL_ATM'), ('put', 'PUT_ATM')]:
                for field_name, (r_val, s_val) in bucket_cmp[side_key].items():
                    row_time = "" if not first else ""
                    _print_row(row_time, "OPTION", f"{side_prefix}_{field_labels[field_name]}", r_val, s_val, field_tols[field_name])
                    first = False
        elif not r_opt_raw and not sql_opt.empty:
             print(f"{time_str:<10} | OPTION | MISSING    | {'EMPTY':<12} | {'DATA_EXIST':<12} | {'N/A':<12} | ⚠️ Redis Missing")

        print("-" * 110)
        curr_minute += 60

    conn.close()
    print("\n💡 Tip: If Close checks PASS but IV checks FAIL, investigate Greeks re-calculation logic in RealTimeFeatureEngine.")
    print("💡 Tip: If Redis is MISSING data, ensure s2_run_realtime_replay_sqlite_1s.py --turbo has finished processing that time period.")


def _fmt_float(val):
    try:
        return f"{float(val):.8f}"
    except Exception:
        return str(val)


def _fmt_ts_array(arr):
    vals = np.asarray(arr).astype(np.int64).tolist()
    out = []
    for ts in vals:
        try:
            dt_str = datetime.fromtimestamp(int(ts), NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
            out.append(f"{ts}({dt_str})")
        except Exception:
            out.append(str(ts))
    return out


def run_feature_parity_audit(left_npz: str, right_npz: str, top_n: int = 20):
    left_path = Path(left_npz).expanduser().resolve()
    right_path = Path(right_npz).expanduser().resolve()

    if not left_path.exists():
        raise FileNotFoundError(f"Left feature snapshot not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right feature snapshot not found: {right_path}")

    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)

    left_keys = set(left.files)
    right_keys = set(right.files)
    common = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)

    print(f"🧪 [Feature Parity Audit] Left:  {left_path}")
    print(f"🧪 [Feature Parity Audit] Right: {right_path}")
    print("-" * 140)

    if left_only:
        print(f"⚠️ Left-only keys: {', '.join(left_only)}")
    if right_only:
        print(f"⚠️ Right-only keys: {', '.join(right_only)}")
    if left_only or right_only:
        print("-" * 140)

    diffs = []
    exact_match_count = 0
    left_hist_ts = np.asarray(left['history_tail_ts']) if 'history_tail_ts' in left.files else None
    right_hist_ts = np.asarray(right['history_tail_ts']) if 'history_tail_ts' in right.files else None

    def infer_ts_for_idx(key, idx_tuple):
        if not idx_tuple or left_hist_ts is None or right_hist_ts is None:
            return ""
        if not key.startswith("hist_"):
            return ""
        if len(idx_tuple) != 1:
            return ""
        pos = int(idx_tuple[0])
        if pos < 0:
            return ""
        # hist_*_30 corresponds to the most recent 30 rows of history_tail_ts
        left_tail30 = left_hist_ts[-30:] if left_hist_ts.shape[0] >= 30 else left_hist_ts
        right_tail30 = right_hist_ts[-30:] if right_hist_ts.shape[0] >= 30 else right_hist_ts
        if pos >= len(left_tail30) or pos >= len(right_tail30):
            return ""
        l_ts = int(left_tail30[pos])
        r_ts = int(right_tail30[pos])
        l_dt = datetime.fromtimestamp(l_ts, NY_TZ).strftime('%H:%M:%S')
        r_dt = datetime.fromtimestamp(r_ts, NY_TZ).strftime('%H:%M:%S')
        return f"{l_dt}/{r_dt}"

    for key in common:
        a = np.asarray(left[key])
        b = np.asarray(right[key])

        if a.shape != b.shape:
            diffs.append({
                'key': key,
                'shape_mismatch': True,
                'left_shape': a.shape,
                'right_shape': b.shape
            })
            continue

        delta = np.abs(a - b)
        max_diff = float(np.max(delta)) if delta.size else 0.0
        mean_diff = float(np.mean(delta)) if delta.size else 0.0
        sum_diff = float(np.sum(delta)) if delta.size else 0.0

        if max_diff <= 1e-12:
            exact_match_count += 1
            continue

        flat_idx = int(np.argmax(delta)) if delta.size else 0
        unravel_idx = np.unravel_index(flat_idx, delta.shape) if delta.size else ()
        diffs.append({
            'key': key,
            'shape_mismatch': False,
            'shape': a.shape,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'sum_diff': sum_diff,
            'argmax': unravel_idx,
            'left_val': float(a[unravel_idx]) if delta.size else 0.0,
            'right_val': float(b[unravel_idx]) if delta.size else 0.0,
            'ts_hint': infer_ts_for_idx(key, unravel_idx),
        })

    print(f"✅ Exact matches: {exact_match_count}")
    print(f"❌ Different keys : {len(diffs)}")
    print("-" * 140)

    diffs.sort(
        key=lambda x: (
            1 if x.get('shape_mismatch') else 0,
            x.get('max_diff', 0.0),
            x.get('mean_diff', 0.0),
            x['key']
        ),
        reverse=True
    )

    if not diffs:
        print("All common keys match exactly.")
        return

    header = f"{'Key':<40} | {'Shape':<18} | {'MaxDiff':<14} | {'MeanDiff':<14} | {'WorstIdx':<16} | {'TimeHint':<17} | {'Left':<14} | {'Right':<14}"
    print(header)
    print("-" * 140)

    for item in diffs[:top_n]:
        if item.get('shape_mismatch'):
            print(
                f"{item['key']:<40} | "
                f"{str(item['left_shape']):<18} | "
                f"{'SHAPE':<14} | "
                f"{'MISMATCH':<14} | "
                f"{'-':<16} | "
                f"{'-':<17} | "
                f"{str(item['left_shape']):<14} | "
                f"{str(item['right_shape']):<14}"
            )
            continue

        print(
            f"{item['key']:<40} | "
            f"{str(item['shape']):<18} | "
            f"{item['max_diff']:<14.8f} | "
            f"{item['mean_diff']:<14.8f} | "
            f"{str(item['argmax']):<16} | "
            f"{item.get('ts_hint', ''):<17} | "
            f"{_fmt_float(item['left_val']):<14} | "
            f"{_fmt_float(item['right_val']):<14}"
        )

    print("-" * 140)
    if 'history_tail_ts' in common:
        left_hist = np.asarray(left['history_tail_ts'])
        right_hist = np.asarray(right['history_tail_ts'])
        same_shape = left_hist.shape == right_hist.shape
        same_values = same_shape and np.array_equal(left_hist, right_hist)
        if (not same_shape) or (not same_values):
            print("🕒 history_tail_ts mismatch detail:")
            print(f"LEFT  ({left_hist.shape}):")
            for item in _fmt_ts_array(left_hist):
                print(f"  {item}")
            print(f"RIGHT ({right_hist.shape}):")
            for item in _fmt_ts_array(right_hist):
                print(f"  {item}")
            print("-" * 140)
    print("💡 Tip: If only a few option features diverge, inspect FCS frozen option snapshot / gate timing.")
    print("💡 Tip: If many stock features diverge together, inspect normalizer state and minute history length.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='NVDA')
    parser.add_argument('--feature-left', type=str, default=None)
    parser.add_argument('--feature-right', type=str, default=None)
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()

    if args.feature_left or args.feature_right:
        if not (args.feature_left and args.feature_right):
            raise SystemExit("Both --feature-left and --feature-right are required for feature parity audit.")
        run_feature_parity_audit(args.feature_left, args.feature_right, top_n=args.top_n)
    else:
        run_parity_audit(symbol=args.symbol)
