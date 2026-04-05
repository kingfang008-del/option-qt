import redis
import pickle
import pandas as pd
import numpy as np
import sqlite3
import logging
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from utils import serialization_utils as ser

# ================= Configuration =================
REDIS_CFG = {'host': 'localhost', 'port': 6379, 'db': 1}
STREAM_KEY = 'unified_inference_stream'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SQLite_Driver] - %(message)s')
logger = logging.getLogger("SQLite_Driver")

class SQLiteDriver:
    def __init__(self, db_path, symbols=None):
        self.r = redis.Redis(**REDIS_CFG)
        self.db_path = Path(db_path)
        self.symbols = symbols
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"❌ Database not found: {self.db_path}")

    def _parse_option_buckets(self, json_str):
        """将 SQLite 中的 buckets_json 解析为 S4 引擎需要的 6x12 Matrix"""
        buckets = np.zeros((6, 12), dtype=float).tolist()
        if not json_str: return buckets
        try:
            data = json.loads(json_str)
            raw_buckets = data.get('buckets', [])
            for i, b in enumerate(raw_buckets):
                if i >= 6: break
                # S4 期望 12 个指标: [price, delta, gamma, vega, theta, strike, volume, iv, bid, ask, bid_size, ask_size]
                # SQLite 中的格式通常是 [price, strike, type, iv, bid, ask, bid_size, ask_size, ...]
                # 这里需要根据实际 DB 结构映射。如果 DB 只有基本项，其余补默认值。
                if len(b) >= 8:
                    buckets[i] = [
                        float(b[0]),  # price
                        0.0,          # delta (SQLite 通常没存)
                        0.0,          # gamma
                        0.0,          # vega
                        0.0,          # theta
                        float(b[1]),  # strike
                        1.0,          # volume (default)
                        float(b[3]),  # iv
                        float(b[4]),  # bid
                        float(b[5]),  # ask
                        float(b[6]),  # bid_size
                        float(b[7])   # ask_size
                    ]
        except: pass
        return buckets

    def load_data(self):
        logger.info(f"📂 Loading data directly from SQLite: {self.db_path}...")
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        
        # 1. 加载主表
        df_a = pd.read_sql("SELECT ts, symbol, alpha as alpha_score, vol_z as fast_vol, event_prob FROM alpha_logs", conn)
        df_s = pd.read_sql("SELECT ts, symbol, close, open, high, low, volume FROM market_bars_1m", conn)
        df_o = pd.read_sql("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
        conn.close()

        # 2. 符号过滤
        if self.symbols:
            logger.info(f"🎯 Filtering for {len(self.symbols)} target symbols...")
            df_a = df_a[df_a['symbol'].isin(self.symbols)]
            df_s = df_s[df_s['symbol'].isin(self.symbols)]
            df_o = df_o[df_o['symbol'].isin(self.symbols)]

        # 3. 合并数据 (模拟 plan_a 的 inner join 逻辑，保证数据不出异常/无未来函数)
        logger.info("🔗 Merging tables (Inner Join on TS/Symbol)...")
        df = pd.merge(df_a, df_s, on=['ts', 'symbol'], how='inner')
        df = pd.merge(df, df_o, on=['ts', 'symbol'], how='inner')
        
        if df.empty:
            logger.error("❌ Merged dataset is EMPTY! Check DB integrity or symbol mapping.")
            return None
            
        return df.sort_values('ts')

    def run(self):
        df = self.load_data()
        if df is None or df.empty: return

        logger.info(f"🚀 Starting Stream: {len(df)} rows across {df['ts'].nunique()} intervals...")
        
        # 重置 Redis Stream
        self.r.delete(STREAM_KEY)
        try:
            self.r.xgroup_destroy(STREAM_KEY, 'backtest_group')
        except: pass
        
        try:
            self.r.xgroup_create(STREAM_KEY, 'backtest_group', mkstream=True, id='$')
        except: pass

        # 按时间戳分组发送
        grouped = df.groupby('ts')
        last_minute = -1
        
        for ts_val, group in tqdm(grouped, desc="Streaming SQLite"):
            ts_float = float(ts_val)
            current_minute = int(ts_float // 60)
            is_new_minute = (last_minute != current_minute)
            last_minute = current_minute
            
            # 更新 Redis 里的当前时间指针
            self.r.set("replay:current_ts", str(ts_float))
            
            # 构建 S4 协议包
            symbols = group['symbol'].tolist()
            
            # 基础指标数组 (Numpy Arrays)
            prices = group['close'].values.astype(np.float32)
            alphas = group['alpha_score'].values.astype(np.float32)
            vols = group['fast_vol'].values.astype(np.float32)
            
            # 解析期权矩阵 (List of Lists)
            # 因为 S4 协议期望每个 Symbol 对应一个矩阵，所以这里需要循环
            call_prices, put_prices = [], []
            call_k, put_k = [], []
            call_iv, put_iv = [], []
            call_bid, call_ask = [], []
            put_bid, put_ask = [], []
            
            for _, row in group.iterrows():
                bks = self._parse_option_buckets(row['buckets_json'])
                # S4 简单协议提取
                put_prices.append(bks[0][0])
                put_k.append(bks[0][5])
                put_iv.append(bks[0][7])
                put_bid.append(bks[0][8])
                put_ask.append(bks[0][9])
                
                call_prices.append(bks[2][0])
                call_k.append(bks[2][5])
                call_iv.append(bks[2][7])
                call_bid.append(bks[2][8])
                call_ask.append(bks[2][9])

            packet = {
                'symbols': symbols,
                'ts': ts_float,
                'stock_price': prices,
                'fast_vol': vols,
                'precalc_alpha': alphas,
                'is_new_minute': is_new_minute,
                
                'feed_put_price': np.array(put_prices, dtype=np.float32),
                'feed_call_price': np.array(call_prices, dtype=np.float32),
                'feed_put_id': ["MOCK_PUT"] * len(symbols),
                'feed_call_id': ["MOCK_CALL"] * len(symbols),
                'feed_put_k': np.array(put_k, dtype=np.float32),
                'feed_put_iv': np.array(put_iv, dtype=np.float32),
                'feed_call_k': np.array(call_k, dtype=np.float32),
                'feed_call_iv': np.array(call_iv, dtype=np.float32),
                
                'feed_put_vol': np.ones(len(symbols), dtype=np.float32),
                'feed_call_vol': np.ones(len(symbols), dtype=np.float32),
                
                'feed_put_bid': np.array(put_bid, dtype=np.float32),
                'feed_put_ask': np.array(put_ask, dtype=np.float32),
                'feed_call_bid': np.array(call_bid, dtype=np.float32),
                'feed_call_ask': np.array(call_ask, dtype=np.float32),
                
                'feed_call_bid_size': np.full(len(symbols), 100.0, dtype=np.float32),
                'feed_call_ask_size': np.full(len(symbols), 100.0, dtype=np.float32),
                'feed_put_bid_size': np.full(len(symbols), 100.0, dtype=np.float32),
                'feed_put_ask_size': np.full(len(symbols), 100.0, dtype=np.float32),
                
                'slow_1m': np.zeros((len(symbols), 30, 1), dtype=np.float32), 
            }
            
            self.r.xadd(STREAM_KEY, {'data': ser.pack(packet)})
            self._wait_for_sync(ts_float)

        self.r.set("replay:status", "DONE")
        logger.info("🎉 SQLite Replay Feed Finished.")

    def _wait_for_sync(self, ts):
        import time
        while True:
            done_ts = self.r.get("sync:orch_done")
            if done_ts and float(done_ts.decode()) >= ts:
                break
            time.sleep(0.005)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help="Path to market_*.db")
    parser.add_argument("--symbols", type=str, default="DELL", help="Target symbols")
    args = parser.parse_args()
    
    syms = [s.strip() for s in args.symbols.split(',')]
    driver = SQLiteDriver(db_path=args.db, symbols=syms)
    driver.run()
