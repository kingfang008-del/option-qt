import sqlite3
import json

def analyze_composition():
    db_path = 'production/preprocess/backtest/history_sqlite_1s/market_20260102.db'
    conn = sqlite3.connect(db_path)
    
    # Target: 1m bar at 10:01:00
    ts_target = 1767366060
    
    m_row = conn.execute(f"SELECT buckets_json FROM option_snapshots_1m WHERE symbol='NVDA' AND ts={ts_target}").fetchone()
    if not m_row:
        print("❌ 1m data missing.")
        return
        
    m_obj = json.loads(m_row[0])
    m_buckets = m_obj['buckets']
    m_contracts = m_obj['contracts']
    
    print(f"--- Composition Analysis of 1m@10:01:00 ---")
    
    # Load all 1s snapshots for that minute
    s_rows = conn.execute(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE symbol='NVDA' AND ts >= {ts_target} AND ts < {ts_target + 60}").fetchall()
    s_stream = { r[0]: json.loads(r[1])['buckets'] for r in s_rows }
    
    for i in range(len(m_buckets)):
        target_bucket = m_buckets[i]
        contract = m_contracts[i]
        strike = target_bucket[5]
        
        # Check matching seconds
        matches = []
        for s_ts, s_buckets in s_stream.items():
            if i < len(s_buckets) and s_buckets[i] == target_bucket:
                matches.append(s_ts % 60)
        
        if matches:
            print(f"✅ Bucket {i} ({contract:<20} | K={strike}): Matches 1s stream at seconds: {matches}")
        else:
            # Maybe it's from the PREVIOUS minute?
            prev_rows = conn.execute(f"SELECT ts, buckets_json FROM option_snapshots_1s WHERE symbol='NVDA' AND ts >= {ts_target-60} AND ts < {ts_target}").fetchall()
            found_prev = []
            for ps_ts, ps_buckets in prev_rows:
                ps_b = json.loads(ps_buckets)['buckets']
                if i < len(ps_b) and ps_b[i] == target_bucket:
                    found_prev.append(ps_ts % 60)
            
            if found_prev:
                print(f"⚠️ Bucket {i} ({contract:<20} | K={strike}): Matches PREVIOUS minute at seconds: {found_prev}")
            else:
                print(f"❌ Bucket {i} ({contract:<20} | K={strike}): NO MATCH in 120s window!")

if __name__ == "__main__":
    analyze_composition()
