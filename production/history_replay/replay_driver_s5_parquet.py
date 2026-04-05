import redis
import pickle
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from tqdm import tqdm
from utils import serialization_utils as ser

# [配置] S5 生成的 Parquet 文件夹路径
PARQUET_DIR = Path('/mnt/s990/data/h5_unified_overlap_id/rl_feed_parquet_new_delta_backtest')
#redis 1 是回测专用，0是实盘专用
REDIS_CFG = {'host': 'localhost', 'port': 6379, 'db': 1}
STREAM_KEY = 'unified_inference_stream'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("S5_Driver")

class S5ParquetDriver:
    def __init__(self, parquet_dir=None, stream_key=None, run_id=None):
        self.r = redis.Redis(**REDIS_CFG)
        self.stream_key = stream_key if stream_key else STREAM_KEY
        self.run_id = run_id
        
        if parquet_dir:
            self.parquet_dir = Path(parquet_dir)
        else:
            # Default fallback (Legacy)
            self.parquet_dir = Path('rl_feed_parquet_new_delta_backtest')

        if not self.parquet_dir.exists():
            logger.warning(f"⚠️ Parquet dir not found: {self.parquet_dir}")

    def load_and_sort_data(self):
        logger.info(f"📂 Loading S5 Parquet files from {self.parquet_dir}...")
        files = list(self.parquet_dir.glob("*.parquet"))
        dfs = []
        for f in tqdm(files, desc="Reading"):
            try:
                # [核心新增] 读取所需的列，包含新增的成交量字段
                cols = ['ts',
                    'timestamp', 'close', 'alpha_score', 'fast_vol', 'spy_roc_5min', 'qqq_roc_5min',
                    'opt_0', 'opt_8', 'opt_0_id', 'opt_8_id',
                    'opt_5', 'opt_7', 'opt_13', 'opt_15',
                    'feed_put_vol', 'feed_call_vol',
                    'feed_put_bid', 'feed_put_ask', 'feed_call_bid', 'feed_call_ask',
                    'feed_call_bid_size', 'feed_call_ask_size', 'feed_put_bid_size', 'feed_put_ask_size'
                ]
                
                # [优化] 先读取列名获取 Schema，避免 ValueError 导致重复读取负载 (针对旧版本数据)
                df_sample = pd.read_parquet(f, engine='pyarrow')
                df_cols = df_sample.columns.tolist()
                
                available_cols = [c for c in cols if c in df_cols]
                df = df_sample[available_cols].copy()
                
                # 补充缺失列的默认值
                for c in ['feed_put_vol', 'feed_call_vol']:
                    if c not in df.columns: df[c] = 1.0
                for c in ['feed_call_bid_size', 'feed_call_ask_size', 'feed_put_bid_size', 'feed_put_ask_size']:
                    if c not in df.columns: df[c] = 100.0
                    
                df['symbol'] = f.stem 
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {f.name}: {e}")
                pass
        
        if not dfs: return pd.DataFrame()
        
        full_df = pd.concat(dfs, ignore_index=True)
        # 确保按时间戳排序，这是回放的关键
        full_df.sort_values('timestamp', inplace=True)
        return full_df
    
    def run(self):
        df = self.load_and_sort_data()
        if df.empty:
            logger.error("❌ No data loaded.")
            return

        logger.info(f"🚀 Starting Stream: {len(df)} rows...")
        
        # 重置 Redis Stream
        self.r.delete(self.stream_key)
        try:
            # 🚀 [CRITICAL] 彻底清理旧的消费者组，防止 Offset 残留
            self.r.xgroup_destroy(self.stream_key, 'backtest_group')
        except: pass

        try:
            # 创建消费者组
            # [Revert] 回滚 id='0' 修改，恢复为默认的 '$'
            self.r.xgroup_create(self.stream_key, 'backtest_group', mkstream=True, id='$')
            logger.info(f"✅ Created backtest consumer group with ID '$'")
        except redis.exceptions.ResponseError: 
            pass

        # 按时间戳分组发送
        # [DEBUG] 检查各标的数据覆盖范围
        stats = df.groupby('symbol')['ts'].agg(['min', 'max', 'count'])
        logger.info(f"🔍 [DEBUG] Symbol Stats:\n{stats}")

        # [🔥 核心增强] 自动寻找所有标的的交集时间段，避免尾部标的孤掌难鸣
        start_ts = stats['min'].max()  # 取所有标的首个点的最大值（最晚开始的那个）
        end_ts = stats['max'].min()    # 取所有标的末个点的最小值（最早结束的那个）
        
        if end_ts > start_ts:
            logger.info(f"⚡ [Intersection] Cropping replay to common range: {start_ts} -> {end_ts}")
            df = df[(df['ts'] >= start_ts) & (df['ts'] <= end_ts)].copy()
        else:
            logger.warning("⚠️ [Intersection] No common time range found across all symbols! Replaying all data.")
        
        # [🔥 终极修正: 数据空洞填齐]
        # 对于回测数据，必须先进行前向填充，防止 NaN 被 fillna(0) 导致价格暴跌触发爆仓 (Blowout)
        ffill_cols = [
            'opt_0', 'opt_8', 'opt_13', 'opt_5', 'opt_7', 'opt_15', 
            'opt_0_id', 'opt_8_id', 'feed_put_bid', 'feed_put_ask', 
            'feed_call_bid', 'feed_call_ask'
        ]
        logger.info("🛠️ Applying Forward-Fill to Option columns to prevent blowout...")
        # 必须按 Symbol 分组填充，防止跨品种污染
        for col in ffill_cols:
            if col in df.columns:
                df[col] = df.groupby('symbol')[col].ffill().bfill()

        # [🔥 修正: 按秒级粒度精确回放] 
        # 不再按分钟字符串分钟分组，改为按原始浮点时间戳分组，确保 1s 数据的位完美。
        # 同时确保数据按时间排序。
        unique_groups = df['ts'].nunique()
        logger.info(f"🔍 [DEBUG] Unique logical timestamps found: {unique_groups}")
        
        grouped = df.sort_values('ts').groupby('ts')
        
        count = 0
        last_minute = -1
        for ts_val, group in tqdm(grouped, desc="Replaying", total=unique_groups):
            # [🔥 每秒检测新分钟]
            current_minute = int(ts_val // 60)
            is_new_minute = (last_minute != current_minute)
            last_minute = current_minute
            
            # 基础数据
            symbols = group['symbol'].tolist()
            prices = group['close'].values.astype(np.float32)
            alphas = group['alpha_score'].values.astype(np.float32)
            vols = group['fast_vol'].values.astype(np.float32)

            # [新增] 提取大盘 ROC 数组
            spy_rocs = group['spy_roc_5min'].fillna(0.0).values.astype(np.float32)
            qqq_rocs = group['qqq_roc_5min'].fillna(0.0).values.astype(np.float32)

            # [核心修复] 字段重命名：cheat -> feed
            # Put Side
            feed_put_price = group['opt_0'].fillna(0.0).values.astype(np.float32)
            feed_put_id = group['opt_0_id'].fillna("").astype(str).tolist()
            feed_put_k = group['opt_5'].fillna(0.0).values.astype(np.float32)
            feed_put_iv = group['opt_7'].fillna(0.0).values.astype(np.float32)
            feed_put_vol = group['feed_put_vol'].fillna(0.0).values.astype(np.float32) # <--- [New] 提取 Put 成交量

            # Call Side
            feed_call_price = group['opt_8'].fillna(0.0).values.astype(np.float32)
            feed_call_id = group['opt_8_id'].fillna("").astype(str).tolist()
            feed_call_k = group['opt_13'].fillna(0.0).values.astype(np.float32)
            feed_call_iv = group['opt_15'].fillna(0.0).values.astype(np.float32)
            feed_call_vol = group['feed_call_vol'].fillna(0.0).values.astype(np.float32)
            
            # [🔥 Market Quotes & Sizes 传递]
            # [Fix] fillna(ndarray) is not supported. Use Series-to-Series filling for index alignment.
            feed_put_bid = group['feed_put_bid'].fillna(group['opt_0']).fillna(0.0).values.astype(np.float32)
            feed_put_ask = group['feed_put_ask'].fillna(group['opt_0']).fillna(0.0).values.astype(np.float32)
            feed_call_bid = group['feed_call_bid'].fillna(group['opt_8']).fillna(0.0).values.astype(np.float32)
            feed_call_ask = group['feed_call_ask'].fillna(group['opt_8']).fillna(0.0).values.astype(np.float32)

            feed_call_bid_size = group['feed_call_bid_size'].fillna(100.0).values.astype(np.float32)
            feed_call_ask_size = group['feed_call_ask_size'].fillna(100.0).values.astype(np.float32)
            feed_put_bid_size = group['feed_put_bid_size'].fillna(100.0).values.astype(np.float32)
            feed_put_ask_size = group['feed_put_ask_size'].fillna(100.0).values.astype(np.float32)
            
            # 更新 Redis 里的当前时间指针 (可选，供 dashboard 使用)
            self.r.set("replay:current_ts", str(ts_val))
            
            packet = {
                'symbols': symbols,
                'ts': ts_val,
                'stock_price': prices,
                'fast_vol': vols,         # 对应 V8 中的 raw_vols
                'precalc_alpha': alphas,  # 对应 V8 中的 use_precalc_feed
                
                # [新增] 大盘属性注入
                'spy_roc_5min': spy_rocs,
                'qqq_roc_5min': qqq_rocs,
                'is_new_minute': is_new_minute,
                
                # [Fix] 使用标准化的 feed_ 前缀
                'feed_put_price': feed_put_price,
                'feed_call_price': feed_call_price,
                'feed_put_id': feed_put_id,
                'feed_call_id': feed_call_id,
                'feed_put_k': feed_put_k,
                'feed_put_iv': feed_put_iv,
                'feed_call_k': feed_call_k,
                'feed_call_iv': feed_call_iv,
                
                # [🔥 核心新增] 传递给 Orchestrator 用于拦截断流/无流动性假信号
                'feed_put_vol': feed_put_vol,
                'feed_call_vol': feed_call_vol,
                
                # [🔥 Market Quotes]
                'feed_put_bid': feed_put_bid,
                'feed_put_ask': feed_put_ask,
                'feed_call_bid': feed_call_bid,
                'feed_call_ask': feed_call_ask,

                'feed_call_bid_size': feed_call_bid_size,
                'feed_call_ask_size': feed_call_ask_size,
                'feed_put_bid_size': feed_put_bid_size,
                'feed_put_ask_size': feed_put_ask_size,
                
                # 占位符: 实盘模式需要的 slow_1m 特征，在回放模式下设为 0 即可
                'slow_1m': np.zeros((len(symbols), 30, 1), dtype=np.float32), 
            }
            
       
            self.r.xadd(self.stream_key, {'data': ser.pack(packet)})
            count += 1

            # [🔥 Step-Sync] 等待 Orchestrator 处理完当前帧
            self._wait_for_sync(ts_val)

        # 设置完成标志 (使用 Run ID 隔离)
        status_key = f"replay:status:{self.run_id}" if self.run_id else "replay:status"
        self.r.set(status_key, "DONE")
        logger.info(f"🎉 Replay Finished. Pushed {count} batches. Status Key: {status_key}")

    def _wait_for_sync(self, ts):
        """等待 Orchestrator (SE + OMS) 处理完当前时间戳"""
        import time
        while True:
            done_ts = self.r.get("sync:orch_done")
            if done_ts and float(done_ts.decode()) >= ts:
                break
            time.sleep(0.005)

if __name__ == "__main__":
    driver = S5ParquetDriver()
    driver.run()