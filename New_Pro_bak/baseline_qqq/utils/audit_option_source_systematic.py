import sqlite3
import json

def systematic_audit():
    db_path = 'production/preprocess/backtest/history_sqlite_1s/market_20260102.db'
    conn = sqlite3.connect(db_path)
    
    test_minutes = [
        1767364500, # 09:35:00
        1767365400, # 09:50:00
        1767366600, # 10:10:00
        1767367800  # 10:30:00
    ]
    
    print(f"{'Timestamp':<12} | {'Time':<10} | {'0s Match':<10} | {'59s Match':<10}")
    print("-" * 50)
    
    for ts in test_minutes:
        # Get 1m target
        res_1m = conn.execute(f"SELECT buckets_json FROM option_snapshots_1m WHERE symbol='NVDA' AND ts={ts}").fetchone()
        if not res_1m: continue
        m_buckets = json.loads(res_1m[0])['buckets']
        
        # Check 0s in 1s table
        res_0s = conn.execute(f"SELECT buckets_json FROM option_snapshots_1s WHERE symbol='NVDA' AND ts={ts}").fetchone()
        s0_buckets = json.loads(res_0s[0])['buckets'] if res_0s else None
        match_0s = "✅ YES" if m_buckets == s0_buckets else "❌ NO"
        
        # Check 59s in 1s table
        res_59s = conn.execute(f"SELECT buckets_json FROM option_snapshots_1s WHERE symbol='NVDA' AND ts={ts+59}").fetchone()
        s59_buckets = json.loads(res_59s[0])['buckets'] if res_59s else None
        match_59s = "✅ YES" if m_buckets == s59_buckets else "❌ NO"
        
        from datetime import datetime
        import pytz
        time_str = datetime.fromtimestamp(ts, pytz.timezone('America/New_York')).strftime('%H:%M:%S')
        print(f"{ts:<12} | {time_str:<10} | {match_0s:<10} | {match_59s:<10}")

if __name__ == "__main__":
    systematic_audit()
