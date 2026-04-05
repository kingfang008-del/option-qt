import redis
import pickle
import pandas as pd
import numpy as np
import time
import logging
import os
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks

# ================= Configuration =================
SPNQ_ROOT = Path('/home/kingfang007/train_data/spnq_train_resampled')
OPTIONS_ROOT = Path('/home/kingfang007/train_data/quote_options_day_iv')

REDIS_CFG = {'host': 'localhost', 'port': 6379, 'db': 1}
STREAM_KEY_FUSED = 'fused_market_stream'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Raw_1m_Driver] - %(message)s')
logger = logging.getLogger("Raw_1m_Driver")

class Raw1mDriver:
    def __init__(self, symbols, start_date=None, end_date=None):
        self.r = redis.Redis(**REDIS_CFG)
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date).tz_localize('America/New_York') if start_date else None
        self.end_date = pd.to_datetime(end_date).tz_localize('America/New_York') if end_date else None
        
        self.rfr_cache = None
        self.rfr_file = '/home/kingfang007/risk_free_rates.parquet'

    def _load_risk_free_rates(self):
        """Loads RFR from local cache or FRED (aligned with option_cac_day_vectorized_day.py)"""
        if os.path.exists(self.rfr_file):
            try:
                df = pd.read_parquet(self.rfr_file)
                df.index = pd.to_datetime(df.index).normalize()
                self.rfr_cache = df
                return
            except Exception: pass
        
        # Fallback default if file missing
        logger.warning(f"⚠️ RFR cache not found at {self.rfr_file}, using default 0.04")
        self.rfr_cache = pd.DataFrame(columns=['DGS3MO'])

    def _get_available_dates(self, symbol):
        opt_dir = OPTIONS_ROOT / symbol / 'standard'
        if not opt_dir.exists():
            return []
        files = list(opt_dir.glob(f"{symbol}_*.parquet"))
        dates = []
        for f in files:
            try:
                # Filename format: DELL_YYYY-MM-DD.parquet
                date_str = f.stem.split('_')[-1]
                dates.append(pd.to_datetime(date_str).tz_localize('America/New_York'))
            except Exception:
                pass
        return sorted(dates)
    
    def _load_day_data(self, date_tz, symbols):
        """Loads and merges SPNQ and Options data for a given day across all target symbols"""
        date_str = date_tz.strftime('%Y-%m-%d')
        month_str = date_tz.strftime('%Y-%m')
        merged_dfs = []

        for sym in symbols:
            spnq_path = SPNQ_ROOT / sym / 'regular' / '09:30-16:00' / '1min' / f"{month_str}.parquet"
            opt_path = OPTIONS_ROOT / sym / 'standard' / f"{sym}_{date_str}.parquet"

            if not spnq_path.exists() or not opt_path.exists():
                logger.debug(f"Missing data for {sym} on {date_str}. Spnq:{spnq_path.exists()} Opt:{opt_path.exists()}")
                continue

            try:
                # Load SPNQ
                engine = 'fastparquet' if 'fastparquet' in os.environ.get('PANDAS_ENGINES', '') else 'auto'
                df_spnq = pd.read_parquet(spnq_path, engine=engine)
                
                # Filter SPNQ to just the current day to prevent duplicate timestamps across loops
                df_spnq['date_only'] = df_spnq['timestamp'].dt.date
                df_spnq = df_spnq[df_spnq['date_only'] == date_tz.date()].drop(columns=['date_only'])
                
                # Load Options
                df_opt = pd.read_parquet(opt_path, engine=engine)

                if df_spnq.empty or df_opt.empty:
                    continue
                
                # Align timestamps
                # Both should have 'timestamp' column as datetime with tz
                df_spnq = df_spnq.set_index('timestamp')
                df_opt = df_opt.set_index('timestamp')

                df_spnq['symbol'] = sym
                
                merged_dfs.append((df_spnq, df_opt))

            except Exception as e:
                logger.warning(f"Error loading {sym} for {date_str}: {e}")

        return merged_dfs

    def run(self, speed_factor=float('inf'), sync_mode=True):
        if not self.symbols:
            logger.error("No symbols provided.")
            return

        # Gather trading days based on the first symbol
        all_dates = self._get_available_dates(self.symbols[0])
        
        # Filter by start/end date
        if self.start_date:
            all_dates = [d for d in all_dates if d >= self.start_date]
        if self.end_date:
            all_dates = [d for d in all_dates if d <= self.end_date]

        if not all_dates:
            logger.error(f"❌ No valid dates found for replay.")
            return
            
        logger.info(f"🚀 Starting Replay for {len(all_dates)} days ({all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')})")
        
        # Reset sync signals
        if sync_mode:
            self.r.delete(STREAM_KEY_FUSED)
            self.r.delete("sync:feature_calc_done")
            self.r.delete("sync:orch_done")
            self.r.delete("replay:status")
        
        self._load_risk_free_rates()
        
        for date_tz in all_dates:
            date_str = date_tz.strftime('%Y-%m-%d')
            # 获取当前日期的 RFR
            rfr_val = 0.04
            if self.rfr_cache is not None and not self.rfr_cache.empty:
                d_norm = date_tz.normalize().replace(tzinfo=None)
                if d_norm in self.rfr_cache.index:
                    rfr_val = float(self.rfr_cache.loc[d_norm, 'DGS3MO'])
                    
            logger.info(f"📂 Loading data for {date_str} (RFR: {rfr_val:.4f})...")
            
            day_data = self._load_day_data(date_tz, self.symbols)
            if not day_data:
                logger.warning(f"⚠️ No data to replay for {date_str}")
                continue
            
            # Combine all unique timestamps for this day
            all_ts = set()
            for df_spnq, _ in day_data:
                all_ts.update(df_spnq.index)
            unique_ts = sorted(list(all_ts))
            
            for ts in tqdm(unique_ts, desc=f"Streaming 1m Snaps ({date_str})"):
                ts_float = float(ts.timestamp())
                
                # 1. Update Simulation Clock
                if sync_mode:
                    self.r.set("replay:current_ts", str(ts_float))
                
                batch_payloads = []
                
                for df_spnq, df_opt in day_data:
                    sym = df_spnq['symbol'].iloc[0]
                    
                    if ts not in df_spnq.index:
                        continue
                        
                    stock_row = df_spnq.loc[ts]
                    if isinstance(stock_row, pd.DataFrame):
                        stock_row = stock_row.iloc[0] # handle duplicates
                    
                    # Get options for this timestamp
                    if ts in df_opt.index:
                        opt_rows = df_opt.loc[[ts]]
                    else:
                        opt_rows = pd.DataFrame()
                        
                    buckets = np.zeros((6, 12), dtype=float).tolist()
                    contracts = [""] * 6
                    
                    if not opt_rows.empty:
                        for _, opt in opt_rows.iterrows():
                            b_id = int(opt['bucket_id'])
                            if 0 <= b_id < 6:
                                opt_price = float(opt.get('close', 0))
                                S_price = float(stock_row.get('close', 0))
                                K_price = float(opt.get('strike_price', 0))
                                
                                iv = float(opt.get('iv', 0))
                                delta = float(opt.get('delta', 0))
                                gamma = float(opt.get('gamma', 0))
                                vega = float(opt.get('vega', 0))
                                theta = float(opt.get('theta', 0))
                                
                                ticker_str = str(opt.get('ticker', f"{sym}_MOCK"))
                                
                                # 🚀 [新增/修复]: 动态 Greeks 兜底计算 (当源头仅有价格没有希腊值时启用)
                                if (iv == 0 or delta == 0) and K_price > 0 and opt_price > 0:
                                    match = re.search(r'([A-Z]+)(\d{6})([CPcp])(\d{8})', ticker_str)
                                    if match:
                                        opt_type = match.group(3).lower()
                                        expiry_date = pd.to_datetime('20' + match.group(2), format='%Y%m%d')
                                        if expiry_date.tz is None:
                                            expiry_date = expiry_date.tz_localize('America/New_York')
                                        t_years = max(1e-6, (expiry_date - ts).total_seconds() / 31557600.0)
                                        
                                        try:
                                            calc_iv = vectorized_implied_volatility(
                                                np.array([opt_price]), np.array([S_price]), np.array([K_price]),
                                                np.array([t_years]), np.array([rfr_val]), opt_type, return_as='numpy', on_error='ignore'
                                            )[0]
                                            if np.isnan(calc_iv) or calc_iv <= 0: calc_iv = 0.5
                                            iv = float(calc_iv)
                                            
                                            greeks = get_all_greeks(
                                                opt_type, np.array([S_price]), np.array([K_price]),
                                                np.array([t_years]), np.array([rfr_val]), np.array([iv]), return_as='dict'
                                            )
                                            delta = float(greeks['delta'][0])
                                            gamma = float(greeks['gamma'][0])
                                            vega = float(greeks['vega'][0])
                                            theta = float(greeks['theta'][0])
                                        except Exception:
                                            pass

                                buckets[b_id] = [
                                    opt_price,
                                    delta,
                                    gamma,
                                    vega,
                                    theta,
                                    K_price,
                                    float(opt.get('volume', 0)),
                                    iv,
                                    float(opt.get('bid', 0)),
                                    float(opt.get('ask', 0)),
                                    float(opt.get('bid_size', 100)),
                                    float(opt.get('ask_size', 100))
                                ]
                                contracts[b_id] = ticker_str
                    
                    payload = {
                        'symbol': sym,
                        'ts': ts_float,
                        'stock': {
                            'close': float(stock_row.get('close', 0)),
                            'open': float(stock_row.get('open', 0)),
                            'high': float(stock_row.get('high', 0)),
                            'low': float(stock_row.get('low', 0)),
                            'volume': float(stock_row.get('volume', 0))
                        },
                        'option_buckets': buckets,
                        'option_contracts': contracts
                    }
                    batch_payloads.append(payload)
                
                if batch_payloads:
                    self.r.xadd(STREAM_KEY_FUSED, {'batch': pickle.dumps(batch_payloads)}, maxlen=5000)
                
                # 2. Master-Clock Handshake (Barrier)
                if sync_mode:
                    wait_start = time.time()
                    while True:
                        f_done = self.r.get("sync:feature_calc_done")
                        o_done = self.r.get("sync:orch_done")
                        
                        f_ok = f_done and float(f_done) >= ts_float
                        o_ok = o_done and float(o_done) >= ts_float
                        
                        if f_ok and o_ok: break
                        if time.time() - wait_start > 10.0:
                            logger.warning(f"⏩ Handshake Timeout @ {ts_float}")
                            break
                        time.sleep(0.01)

                # 3. Speed Control
                if 0 < speed_factor < float('inf'):
                    time.sleep(60.0 / speed_factor)

        self.r.set("replay:status", "DONE")
        logger.info("🎉 1m Replay Feed Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw 1m Driver Replay")
    parser.add_argument("--symbols", type=str, default="DELL", help="Comma-separated symbols to track")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--speed", type=float, default=float('inf'), help="Speed factor")
    parser.add_argument("--no-sync", action="store_true", help="Disable sync mode")
    
    args = parser.parse_args()
    sym_list = [s.strip() for s in args.symbols.split(',')]
    
    driver = Raw1mDriver(symbols=sym_list, start_date=args.start_date, end_date=args.end_date)
    driver.run(speed_factor=args.speed, sync_mode=not args.no_sync)
