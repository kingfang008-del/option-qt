import sqlite3
import json
import os

def compare_options():
    db_path = 'production/preprocess/backtest/history_sqlite_1s/market_20260102.db'
    if not os.path.exists(db_path):
        # Alternative path if current is different
        db_path = 'market_20260102.db'
        
    conn = sqlite3.connect(db_path)
    
    # 🧪 测试点：10:01:59 (秒) vs 10:01:00 (分标)
    ts_1s = 1767366119
    ts_1m = 1767366060
    
    def get_json(ts, table):
        cursor = conn.execute(f"SELECT buckets_json FROM {table} WHERE symbol='NVDA' AND ts={ts}")
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    opt_1s = get_json(ts_1s, 'option_snapshots_1s')
    opt_1m = get_json(ts_1m, 'option_snapshots_1m')
    
    print(f"🚀 [Option Audit] Comparing 1s@10:01:59 vs 1m@10:01:00")
    if not opt_1s or not opt_1m:
        print(f"❌ Data missing for NVDA at these timestamps.")
        return

    b_1s = opt_1s['buckets']
    b_1m = opt_1m['buckets']
    
    match_count = 0
    total = len(b_1m)
    
    # 检查核心槽位 (Bid/Ask/IV)
    # bucket format: [last, iv, delta, gamma, vega, k, vol, oi, bid, ask, b_sz, a_sz]
    for i in range(min(len(b_1s), len(b_1m))):
        s = b_1s[i]
        m = b_1m[i]
        
        # 精确对比
        if s == m:
            match_count += 1
        else:
            print(f"⚠️ Mismatch @ Bucket {i} (Strike={m[5]}):")
            print(f"   - 1s (:59): IV={s[1]:.4f}, Bid={s[8]}, Ask={s[9]}")
            print(f"   - 1m (:00): IV={m[1]:.4f}, Bid={m[8]}, Ask={m[9]}")

    print(f"\n📊 [Parity Summary] Matches: {match_count}/{total}")
    if match_count == total:
        print("✅ SUCCESS: Option snapshots are perfectly consistent.")
    else:
        print("❌ FAILURE: Discrepancy detected in option snapshots.")

if __name__ == "__main__":
    compare_options()
