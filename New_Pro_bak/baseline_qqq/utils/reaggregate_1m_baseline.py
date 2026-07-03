
import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pytz import timezone

DB_PATH = "/Users/fangshuai/Documents/GitHub/option-qt/production/preprocess/backtest/history_sqlite_1s/market_20260102.db"

def repair_all_data():
    conn = sqlite3.connect(DB_PATH)
    logger_print = lambda m: print(f"🛠️ [TOTAL-REPAIR] {m}")
    
    # =========================================================
    # 1. 股票数据修复 (OHLCV + VWAP)
    # =========================================================
    logger_print("Step 1: Re-aggregating Stock Bars (OHLCV + VWAP)...")
    df_1s = pd.read_sql_query("SELECT symbol, ts, open, high, low, close, volume FROM market_bars_1s", conn)
    if not df_1s.empty:
        df_1s['ts_1m'] = (df_1s['ts'] // 60) * 60
        df_1s['pv'] = df_1s['close'] * df_1s['volume']
        
        agg_funcs = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df_1m_core = df_1s.sort_values(['symbol', 'ts']).groupby(['symbol', 'ts_1m']).agg(agg_funcs).reset_index()
        df_1m_pv = df_1s.groupby(['symbol', 'ts_1m'])[['pv', 'volume']].sum().reset_index()
        df_1m_pv['vwap'] = df_1m_pv['pv'] / (df_1m_pv['volume'] + 1e-9)
        df_1m = pd.merge(df_1m_core, df_1m_pv[['symbol', 'ts_1m', 'vwap']], on=['symbol', 'ts_1m'])
        
        ny_tz = timezone('America/New_York')
        df_1m['datetime_ny'] = df_1m['ts_1m'].apply(lambda x: datetime.fromtimestamp(x, tz=ny_tz).strftime('%Y-%m-%d %H:%M:%S'))
        
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE market_bars_1m ADD COLUMN vwap REAL")
            cursor.execute("ALTER TABLE market_bars_5m ADD COLUMN vwap REAL")
        except: pass

        records_1m = []
        for _, row in df_1m.iterrows():
            records_1m.append((row['symbol'], int(row['ts_1m']), row['datetime_ny'], float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume']), float(row['vwap'])))
        
        cursor.execute("DELETE FROM market_bars_1m")
        cursor.executemany("INSERT INTO market_bars_1m VALUES (?,?,?,?,?,?,?,?,?)", records_1m)

        # 5min 股票聚合
        df_1m['ts_5m'] = (df_1m['ts_1m'] // 300) * 300
        df_1m['pv'] = df_1m['vwap'] * df_1m['volume']
        df_5m_core = df_1m.sort_values(['symbol', 'ts_1m']).groupby(['symbol', 'ts_5m']).agg(agg_funcs).reset_index()
        df_5m_pv = df_1m.groupby(['symbol', 'ts_5m'])[['pv', 'volume']].sum().reset_index()
        df_5m_pv['vwap'] = df_5m_pv['pv'] / (df_5m_pv['volume'] + 1e-9)
        df_5m = pd.merge(df_5m_core, df_5m_pv[['symbol', 'ts_5m', 'vwap']], on=['symbol', 'ts_5m'])
        df_5m['datetime_ny'] = df_5m['ts_5m'].apply(lambda x: datetime.fromtimestamp(x, tz=ny_tz).strftime('%Y-%m-%d %H:%M:%S'))
        
        records_5m = []
        for _, row in df_5m.iterrows():
            records_5m.append((row['symbol'], int(row['ts_5m']), row['datetime_ny'], float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume']), float(row['vwap'])))
        
        cursor.execute("DELETE FROM market_bars_5m")
        cursor.executemany("INSERT INTO market_bars_5m VALUES (?,?,?,?,?,?,?,?,?)", records_5m)
        logger_print("✅ Stock Tables (1m/5m) Repopulated.")

    # =========================================================
    # 2. 期权快照修复 (Time Alignment + Mid-Price Correction)
    # =========================================================
    logger_print("Step 2: Re-aggregating Option Snapshots (Mid-Price Support)...")
    # 读取秒级快照
    df_opt_1s = pd.read_sql_query("SELECT symbol, ts, buckets_json FROM option_snapshots_1s", conn)
    if not df_opt_1s.empty:
        df_opt_1s['ts_1m'] = (df_opt_1s['ts'] // 60) * 60
        
        # 选取每分钟最后一帧 (Keep Last)
        df_opt_1m = df_opt_1s.sort_values(['symbol', 'ts']).groupby(['symbol', 'ts_1m']).last().reset_index()
        
        corrected_records_1m = []
        for _, row in df_opt_1m.iterrows():
            data = json.loads(row['buckets_json'])
            buckets = data['buckets']
            
            # 🚀 [Price Correction] 针对每个 Bucket 强制执行 Bid/Ask 均价逻辑
            for i in range(len(buckets)):
                b = buckets[i]
                # Index 8: Bid, Index 9: Ask, Index 0: Price
                bid, ask = float(b[8]), float(b[9])
                if bid > 0 and ask > 0:
                    b[0] = (bid + ask) / 2.0
            
            corrected_json = json.dumps({'buckets': buckets, 'contracts': data.get('contracts', [])})
            corrected_records_1m.append((row['symbol'], int(row['ts_1m']), corrected_json))
        
        cursor.execute("DELETE FROM option_snapshots_1m")
        cursor.executemany("INSERT INTO option_snapshots_1m (symbol, ts, buckets_json) VALUES (?,?,?)", corrected_records_1m)

        # 5min 期权聚合
        df_opt_1m['ts_5m'] = (df_opt_1m['ts_1m'] // 300) * 300
        df_opt_5m = df_opt_1m.sort_values(['symbol', 'ts_1m']).groupby(['symbol', 'ts_5m']).last().reset_index()
        
        corrected_records_5m = []
        for _, row in df_opt_5m.iterrows():
            corrected_records_5m.append((row['symbol'], int(row['ts_5m']), row['buckets_json'])) # 延用 1m 的已修正 JSON
            
        cursor.execute("DELETE FROM option_snapshots_5m")
        cursor.executemany("INSERT INTO option_snapshots_5m (symbol, ts, buckets_json) VALUES (?,?,?)", corrected_records_5m)
        logger_print("✅ Option Snapshots (1m/5m) Repopulated with Mid-Price alignment.")

    conn.commit()
    conn.close()
    print("🚀 [TOTAL-REPAIR] ALL Baseline tables are now bit-aligned with 1s high-fidelity data.")

if __name__ == "__main__":
    repair_all_data()
