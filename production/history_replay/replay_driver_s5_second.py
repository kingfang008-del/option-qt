import redis
import pickle
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from tqdm import tqdm

# [配置] 数据目录
SLOW_DIR = Path('/mnt/s990/data/h5_unified_overlap_id/rl_feed_parquet_new_delta_backtest')
FAST_DIR = Path('/mnt/s990/data/h5_unified_overlap_id/rl_feed_parquet_1s_quotes') # 假设的秒级目录
REDIS_CFG = {'host': 'localhost', 'port': 6379, 'db': 1}
STREAM_KEY = 'unified_inference_stream'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("S5_Second_Driver")

class S5SecondDriver:
    def __init__(self, slow_dir=SLOW_DIR, fast_dir=FAST_DIR):
        self.r = redis.Redis(**REDIS_CFG)
        self.slow_dir = Path(slow_dir)
        self.fast_dir = Path(fast_dir)

    def load_slow_data(self):
        """加载 1min 维度的特征与信号数据"""
        logger.info(f"📂 Loading SLOW (1min) data from {self.slow_dir}...")
        files = list(self.slow_dir.glob("*.parquet"))
        dfs = []
        for f in tqdm(files, desc="Reading Slow"):
            df = pd.read_parquet(f)
            df['symbol'] = f.stem
            df['replay_type'] = 'slow'
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def load_fast_data(self):
        """加载 1sec 维度的盘口报价数据"""
        if not self.fast_dir.exists():
            logger.warning(f"⚠️ FAST (1sec) directory {self.fast_dir} not found. Simulation mode only.")
            return pd.DataFrame()
            
        logger.info(f"📂 Loading FAST (1sec) data from {self.fast_dir}...")
        files = list(self.fast_dir.glob("*.parquet"))
        dfs = []
        for f in tqdm(files, desc="Reading Fast"):
            df = pd.read_parquet(f)
            df['symbol'] = f.stem
            df['replay_type'] = 'fast'
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def run(self):
        df_slow = self.load_slow_data()
        df_fast = self.load_fast_data()
        
        if df_slow.empty:
            logger.error("❌ No slow data loaded.")
            return

        # 合并并按时间戳严格排序
        full_df = pd.concat([df_slow, df_fast], ignore_index=True)
        full_df.sort_values(['ts', 'replay_type'], ascending=[True, False], inplace=True) # Slow (1min) first if same TS
        
        logger.info(f"🚀 Starting High-Fidelity Stream: {len(full_df)} rows...")
        self.r.delete(STREAM_KEY)
        
        grouped = full_df.groupby('ts')
        for ts_val, group in tqdm(grouped, desc="Replaying"):
            ts_val = float(ts_val)
            
            # 区分开 Slow 和 Fast 的行
            slow_rows = group[group['replay_type'] == 'slow']
            fast_rows = group[group['replay_type'] == 'fast']
            
            # 1. 如果有 Slow 数据，按分钟发送特征包
            if not slow_rows.empty:
                packet = self._build_packet(slow_rows, ts_val, 'slow')
                self.r.xadd(STREAM_KEY, {'data': pickle.dumps(packet)})
            
            # 2. 如果有 Fast 数据，按秒发送 Tick 包
            if not fast_rows.empty:
                packet = self._build_packet(fast_rows, ts_val, 'fast')
                self.r.xadd(STREAM_KEY, {'data': pickle.dumps(packet)})

    def _build_packet(self, group, ts_val, r_type):
        """复用 V8 格式并打上 replay_type 标签"""
        packet = {
            'replay_type': r_type,
            'ts': ts_val,
            'symbols': group['symbol'].tolist(),
            'stock_price': group['close'].values.astype(np.float32),
            'fast_vol': group['fast_vol'].values.astype(np.float32) if 'fast_vol' in group else np.zeros(len(group), dtype=np.float32),
            'precalc_alpha': group['alpha_score'].values.astype(np.float32) if 'alpha_score' in group else np.zeros(len(group), dtype=np.float32),
            
            # Put Side
            'feed_put_price': group['opt_0'].values.astype(np.float32) if 'opt_0' in group else np.zeros(len(group), dtype=np.float32),
            'feed_put_bid': group['feed_put_bid'].values.astype(np.float32) if 'feed_put_bid' in group else np.zeros(len(group), dtype=np.float32),
            'feed_put_ask': group['feed_put_ask'].values.astype(np.float32) if 'feed_put_ask' in group else np.zeros(len(group), dtype=np.float32),
            
            # Call Side
            'feed_call_price': group['opt_8'].values.astype(np.float32) if 'opt_8' in group else np.zeros(len(group), dtype=np.float32),
            'feed_call_bid': group['feed_call_bid'].values.astype(np.float32) if 'feed_call_bid' in group else np.zeros(len(group), dtype=np.float32),
            'feed_call_ask': group['feed_call_ask'].values.astype(np.float32) if 'feed_call_ask' in group else np.zeros(len(group), dtype=np.float32),
            
            # Macro
            'spy_roc_5min': group['spy_roc_5min'].values.astype(np.float32) if 'spy_roc_5min' in group else np.zeros(len(group), dtype=np.float32),
            'qqq_roc_5min': group['qqq_roc_5min'].values.astype(np.float32) if 'qqq_roc_5min' in group else np.zeros(len(group), dtype=np.float32),
        }
        
        # 补全其他元数据 (IV, Strike 等)
        for col, target in [('opt_7', 'feed_put_iv'), ('opt_15', 'feed_call_iv'), ('opt_5', 'feed_put_k'), ('opt_13', 'feed_call_k')]:
            if col in group:
                packet[target] = group[col].values.astype(np.float32)
                
        return packet

if __name__ == "__main__":
    driver = S5SecondDriver()
    driver.run()
