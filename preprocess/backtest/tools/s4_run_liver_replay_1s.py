import os
# 🚨 绝对第一优先级！在所有模块加载前，将当前进程及其衍生的配置定死为流式回放模式！
os.environ['RUN_MODE'] = 'LIVEREPLAY'

import redis
import pickle
import pandas as pd
import numpy as np
import time
import logging
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import serialization_utils as ser
# 🚀 修复: 直接引入 config 中的流名称，杜绝硬编码错位
from config import TARGET_SYMBOLS, HASH_OPTION_SNAPSHOT, STREAM_FUSED_MARKET

# ================= Configuration =================
STOCK_1S_DIR = Path('/mnt/s990/data/raw_1s/stocks')
OPTION_1S_DIR = Path('/mnt/s990/data/stress_test_1s_greeks')

REDIS_CFG = {'host': 'localhost', 'port': 6379, 'db': 1}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Raw_1s_Driver] - %(message)s')
logger = logging.getLogger("Raw_1s_Driver")

class Raw1sDriver:
    def __init__(self, symbols, start_date=None, end_date=None):
        self.r = redis.Redis(**REDIS_CFG)
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date).tz_localize('America/New_York') if start_date else None
        self.end_date = pd.to_datetime(end_date).tz_localize('America/New_York') if end_date else None
        
        self.rfr_cache = None
        self.rfr_file = '/home/kingfang007/risk_free_rates.parquet'

    def _load_risk_free_rates(self):
        """Loads RFR from local cache"""
        if os.path.exists(self.rfr_file):
            try:
                df = pd.read_parquet(self.rfr_file)
                df.index = pd.to_datetime(df.index).normalize()
                self.rfr_cache = df
                return
            except Exception: pass
        self.rfr_cache = pd.DataFrame(columns=['DGS3MO'])

    def _get_available_dates(self, symbol):
        opt_dir = OPTION_1S_DIR / symbol
        if not opt_dir.exists(): return []
        files = sorted(list(opt_dir.glob(f"{symbol}_*.parquet")))
        dates = []
        for f in files:
            try:
                date_str = f.stem.split('_')[-1].replace('.parquet', '')
                dt = pd.to_datetime(date_str).tz_localize('America/New_York')
                dates.append(dt)
            except Exception: pass
        return sorted(dates)
    
    def _load_day_data(self, date_tz, symbols):
        """Loads and pre-filters data for each symbol into memory"""
        date_str = date_tz.strftime('%Y-%m-%d')
        merged_dfs = {}

        for sym in symbols:
            spnq_path = STOCK_1S_DIR / sym / f"{sym}_{date_str}.parquet"
            opt_path = OPTION_1S_DIR / sym / f"{sym}_{date_str}.parquet"

            if not spnq_path.exists() or not opt_path.exists():
                continue

            try:
                engine = 'fastparquet' if 'fastparquet' in os.environ.get('PANDAS_ENGINES', '') else 'auto'
                
                df_spnq = pd.read_parquet(spnq_path, engine=engine)
                df_spnq['timestamp'] = pd.to_datetime(df_spnq['timestamp'])
                df_spnq = df_spnq.set_index('timestamp').sort_index()

                df_opt = pd.read_parquet(opt_path, engine=engine)
                df_opt['timestamp'] = pd.to_datetime(df_opt['timestamp'])
                df_opt = df_opt.set_index('timestamp').sort_index()

                target_date = date_tz.date()
                df_spnq = df_spnq[df_spnq.index.normalize().date == target_date]
                df_opt = df_opt[df_opt.index.normalize().date == target_date]

                if df_spnq.empty or df_opt.empty: continue
                
                logger.info(f"✅ Loaded {sym} {date_str}: {len(df_spnq)}s ticks | Range: {df_spnq.index[0]} to {df_spnq.index[-1]}")
                merged_dfs[sym] = (df_spnq, df_opt)

            except Exception as e:
                logger.warning(f"Error loading {sym} for {date_str}: {e}")

        return merged_dfs

    def run(self, speed_factor=float('inf'), sync_mode=True):
        if not self.symbols: return

        all_dates = self._get_available_dates(self.symbols[0])
        if self.start_date: all_dates = [d for d in all_dates if d >= self.start_date]
        if self.end_date: all_dates = [d for d in all_dates if d <= self.end_date]

        if not all_dates:
            logger.error("❌ No dates found.")
            return
            
        logger.info(f"🚀 Replay Starting (1-Minute Aggregation Mode)")
        
        logger.info("🧨 Clearing specific Stream keys to ensure a clean replay environment...")
        keys_to_clear = [
            STREAM_FUSED_MARKET, 'trade_log_stream', 'unified_inference_stream', 
            HASH_OPTION_SNAPSHOT, 'sync:feature_calc_done', 'sync:orch_done', 'replay:status'
        ]
        for k in keys_to_clear:
            self.r.delete(k)

        # 👇 [🚀 致命修复：计算 start_ts，防止 PG 清理崩溃]
        start_ts = all_dates[0].replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        logger.info("🧨 Clearing PostgreSQL state for a clean replay...")
        try:
            import psycopg2
            from config import PG_DB_URL
            
            conn = psycopg2.connect(PG_DB_URL)
            c = conn.cursor()
            
            c.execute("DELETE FROM market_bars_1m WHERE ts >= %s", (start_ts,))
            c.execute("DELETE FROM market_bars_5m WHERE ts >= %s", (start_ts,))
            c.execute("DELETE FROM option_snapshots_1m WHERE ts >= %s", (start_ts,))
            c.execute("DELETE FROM alpha_logs WHERE ts >= %s", (start_ts,))
            c.execute("DELETE FROM trade_logs_backtest;")
            c.execute("DELETE FROM symbol_state;")
            conn.commit()
            conn.close()
            logger.info("✅ PostgreSQL 状态与交易流水清理完成！")
        except Exception as e:
            logger.error(f"⚠️ 无法清理 PostgreSQL (可能未配置或数据库未启动): {e}")
        
        if sync_mode:
            self.r.set("sync:feature_calc_done", "0")
            self.r.set("sync:orch_done", "0")
            self.r.set("replay:status", "STARTING")
        
        self._load_risk_free_rates()
        REPLAY_LOCK = "/tmp/replay_active.lock"

        # 辅助函数：安全提取字典值
        def safe_get(opt_row, col_name, prev_val):
            val = opt_row.get(col_name, 0.0)
            try:
                f_val = float(val)
                return prev_val if np.isnan(f_val) or f_val == 0.0 else f_val
            except:
                return prev_val

        try:
            with open(REPLAY_LOCK, "w") as f:
                f.write(str(time.time()))
            logger.info(f"🔒 Mode Lock Created: {REPLAY_LOCK} (System wide LIVEREPLAY active)")

            for date_tz in all_dates:
                if sync_mode:
                    self.r.set("sync:feature_calc_done", "0")
                    self.r.set("sync:orch_done", "0")
                    self.r.set("replay:status", f"REPLAY_{date_tz.strftime('%Y%m%d')}")

                date_tz = date_tz.normalize()
                date_str = date_tz.strftime('%Y-%m-%d')
                day_data_map = self._load_day_data(date_tz, self.symbols)
                if not day_data_map: continue
                
                # Master 1-Min timeline: 09:30 - 16:00
                timeline = pd.date_range(
                    start=date_tz.replace(hour=9, minute=30, second=0),
                    end=date_tz.replace(hour=16, minute=0, second=0),
                    freq='1min'
                )
                
                last_seen_state = {sym: {
                    'stock': {'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0},
                    'opt_buckets': np.zeros((6, 12), dtype=float).tolist(),
                    'opt_contracts': [""] * 6,
                    'stock_5m_buffer': [] 
                } for sym in self.symbols}

                for ts in tqdm(timeline, desc=f"Minute Replay ({date_str})"):
                    ts_59 = ts + pd.Timedelta(seconds=59)
                    ts_float = float(ts_59.timestamp())
                    ts_end = ts_59 
                    is_5m_boundary = (ts.minute % 5 == 0)
                    
                    if sync_mode:
                        self.r.set("replay:current_ts", str(ts_float))
                    
                    batch_payloads = []
                    
                    for sym in self.symbols:
                        state = last_seen_state[sym]
                        
                        if sym in day_data_map:
                            df_s, df_o = day_data_map[sym]
                            
                            # 🚀 优化 1：使用 .loc 极速切片
                            m_s = df_s.loc[ts:ts_end]
                            if not m_s.empty:
                                bar_1m = {
                                    'open': float(m_s['open'].iloc[0]),
                                    'high': float(m_s['high'].max()),
                                    'low': float(m_s['low'].min()),
                                    'close': float(m_s['close'].iloc[-1]),
                                    'volume': float(m_s['volume'].sum())
                                }
                                state['stock'] = bar_1m
                            else:
                                state['stock']['volume'] = 0.0
                                bar_1m = state['stock'].copy()

                            state['stock_5m_buffer'].append(bar_1m)
                            if len(state['stock_5m_buffer']) > 5: state['stock_5m_buffer'].pop(0)

                            # 🚀 优化 2：使用 .loc 极速切片
                            m_o = df_o.loc[ts:ts_end]
                            if not m_o.empty:
                                if 'bucket_id' in m_o.columns:
                                    last_opts = m_o.groupby('bucket_id').last().reset_index()
                                else:
                                    last_opts = m_o.groupby('tag').last().reset_index() if 'tag' in m_o.columns else m_o
                                
                                buckets = state['opt_buckets']
                                contracts = state['opt_contracts']
                                
                                for _, opt in last_opts.iterrows():
                                    # 🚀 修复 3：防止 NaN 导致 int() 崩溃
                                    b_id_raw = opt.get('bucket_id', -1)
                                    b_id = int(b_id_raw) if pd.notna(b_id_raw) else -1
                                    
                                    if b_id == -1 and 'tag' in opt:
                                        TAG_VALS = {'PUT_ATM':0, 'PUT_OTM':1, 'CALL_ATM':2, 'CALL_OTM':3, 'NEXT_PUT_ATM':4, 'NEXT_CALL_ATM':5}
                                        b_id = TAG_VALS.get(opt['tag'], -1)

                                    if 0 <= b_id < 6:
                                        vol_val = float(opt.get('volume', 1.0))
                                        if np.isnan(vol_val) or vol_val <= 0: vol_val = 1.0 
                                        
                                        prev_bucket = buckets[b_id]
                                        
                                        _price = safe_get(opt, 'price', 0)
                                        if _price == 0.0: _price = safe_get(opt, 'close', prev_bucket[0])
                                        
                                        buckets[b_id] = [
                                            _price,
                                            safe_get(opt, 'delta', prev_bucket[1]),
                                            safe_get(opt, 'gamma', prev_bucket[2]),
                                            safe_get(opt, 'vega', prev_bucket[3]),
                                            safe_get(opt, 'theta', prev_bucket[4]),
                                            safe_get(opt, 'strike_price', prev_bucket[5]),
                                            vol_val,
                                            safe_get(opt, 'iv', prev_bucket[7]),
                                            safe_get(opt, 'bid', prev_bucket[8]),
                                            safe_get(opt, 'ask', prev_bucket[9]),
                                            safe_get(opt, 'bid_size', prev_bucket[10]),
                                            safe_get(opt, 'ask_size', prev_bucket[11])
                                        ]
                                        
                                        buckets[b_id] = [0.0 if np.isnan(x) else x for x in buckets[b_id]]
                                        contracts[b_id] = str(opt.get('ticker', f"{sym}_REPLAY"))
                                
                                state['opt_buckets'] = buckets
                                state['opt_contracts'] = contracts

                        payload = {
                            'symbol': sym, 'ts': ts_float,
                            'stock': state['stock'],
                            'option_buckets': state['opt_buckets'],
                            'option_contracts': state['opt_contracts']
                        }

                        if is_5m_boundary and state['stock_5m_buffer']:
                            buf = state['stock_5m_buffer']
                            payload['stock_5m'] = {
                                'open': buf[0]['open'],
                                'high': max(b['high'] for b in buf),
                                'low': min(b['low'] for b in buf),
                                'close': buf[-1]['close'],
                                'volume': sum(b['volume'] for b in buf)
                            }
                            payload['option_buckets_5m'] = state['opt_buckets']
                            payload['option_contracts_5m'] = state['opt_contracts']

                        batch_payloads.append(payload)
                        
                        snap_data = {'buckets': state['opt_buckets'], 'contracts': state['opt_contracts']}
                        self.r.hset(HASH_OPTION_SNAPSHOT, sym, ser.pack(snap_data))
                    
                    if batch_payloads:
                        # 🚀 修复: 使用 config 导入的标准流名称 STREAM_FUSED_MARKET
                        self.r.xadd(STREAM_FUSED_MARKET, {'batch': ser.pack(batch_payloads)}, maxlen=10000)
                    
                    if sync_mode:
                        timeout = 0
                        while True:
                            f_done = self.r.get("sync:feature_calc_done")
                            f_done = f_done.decode() if f_done else "0"
                            o_done = self.r.get("sync:orch_done")
                            o_done = o_done.decode() if o_done else "0"
                            
                            # 🚀 致命修复：解开 Orchestrator 同步锁，防止幽灵死锁和积压！
                            if float(f_done) >= ts_float and float(o_done) >= ts_float:
                                break
                            
                            time.sleep(0.01) # 略微缩短轮询间隔，提升回放速度
                            timeout += 1
                            if timeout > 1500: # 15 seconds (1500 * 0.01)
                                logger.warning(f"⏰ Sync Timeout at {ts} (F:{f_done}, O:{o_done}) - Skipping wait")
                                break

                    if 0 < speed_factor < float('inf'):
                        time.sleep(1.0 / speed_factor)

        finally:
            if os.path.exists(REPLAY_LOCK):
                os.remove(REPLAY_LOCK)
                logger.info(f"🔓 Mode Lock Removed: {REPLAY_LOCK}")

        self.r.set("replay:status", "DONE")
        logger.info("🎉 Minute-based Replay Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(TARGET_SYMBOLS))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--speed", type=float, default=float('inf'))
    parser.add_argument("--no-sync", action="store_true")
    
    args = parser.parse_args()
    sym_list = [s.strip() for s in args.symbols.split(',')]
    driver = Raw1sDriver(symbols=sym_list, start_date=args.start_date, end_date=args.end_date)
    driver.run(speed_factor=args.speed, sync_mode=not args.no_sync)