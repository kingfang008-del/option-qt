#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import logging
import asyncio
import numpy as np
import pandas as pd
import redis
import psycopg2
from datetime import datetime
from pathlib import Path
from pytz import timezone

# =========================================================
# 🚀 [环境配置] 强制注入，确保行为与实盘执行引擎对齐
# =========================================================
 
os.environ['RUN_MODE']              = 'LIVEREPLAY'    # 🚨 激活执行引擎的 Redis 流式消费与 Mock 撮合
os.environ['SYNC_EXECUTION']        = '1'             # 强制发球机与执行引擎同步步调，实现原子成交
os.environ['PURE_ALPHA_REPLAY']     = '1'             # 纯净信号回放
os.environ['DISABLE_ICEBERG']       = '1'             # 禁用冰山拆单，防止秒级价格漂移
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'           # 禁用高频扫荡对齐分钟级

# 导入项目内部序列化工具
# 假设项目根目录在 sys.path 中
try:
    from utils import serialization_utils as ser
    from config import PG_DB_URL, STREAM_ORCH_SIGNAL, REDIS_CFG, NY_TZ
except ImportError:
    # 兼容性处理：如果路径不对，请根据实际情况调整
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import serialization_utils as ser
    from config import PG_DB_URL, STREAM_ORCH_SIGNAL, REDIS_CFG, NY_TZ

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Fused_Pitcher_V8")

# =========================================================
# 核心逻辑：Payload 组装 (原 oms_replay_pitcher_core 逻辑)
# =========================================================

def build_fused_inference_payload(ts_val: float, frame_id: int, fused_rows: list[dict]) -> dict:
    """
    直接接收 DB 融合视图的一组记录 (同一 ts 下的多个 symbol)，
    组装成 execution_engine_v8 期望的格式。
    """
    symbols = []
    stock_price = []
    stock_volume = []
    precalc_alpha = []
    fast_vol = []
    
    # 可以在此处扩展其他特征，如 SPY/QQQ 的 ROC
    spy_rocs = []
    
    for row in fused_rows:
        symbols.append(row['symbol'])
        stock_price.append(float(row['close']))
        stock_volume.append(float(row['volume']))
        
        # 核心：直接读取融合好的信号字段
        # 注意：这里的 key 'alpha' 和 'vol_z' 需与数据库表/视图字段名对应
        precalc_alpha.append(float(row.get('alpha', row.get('precalc_alpha', 0.0))))
        fast_vol.append(float(row.get('vol_z', row.get('fast_vol', 0.0))))
        spy_rocs.append(float(row.get('spy_roc_5min', 0.0)))

    if not symbols: 
        return None

    # Payload 结构必须与 signal_engine_v8 实盘发出的格式完全一致
    return {
        "ts": float(ts_val),
        "frame_id": frame_id,
        "symbols": symbols,
        "stock_price": np.asarray(stock_price, dtype=np.float32),
        "stock_volume": np.asarray(stock_volume, dtype=np.float32),
        "precalc_alpha": np.asarray(precalc_alpha, dtype=np.float32),
        "fast_vol": np.asarray(fast_vol, dtype=np.float32),
        "spy_roc_5min": np.asarray(spy_rocs, dtype=np.float32),
        "alpha_label_ts": np.full(len(symbols), float(ts_val), dtype=np.float64),
        "source": "fused_replay_v8"
    }

# =========================================================
# 主类：Standalone OMS Replay Pitcher
# =========================================================

class FusedOMSReplayPitcher:
    def __init__(self, date_str: str, flush_redis: bool = True):
        self.date_str = date_str
        
        # 🚀 [终极修复] 改用“白名单”机制，仅提取原生的 Redis 连接参数
        # 彻底无视 config.py 中混入的各种 stream 和 group 字段
        valid_redis_args = ['host', 'port', 'db', 'password', 'username', 'socket_timeout', 'decode_responses']
        conn_cfg = {k: v for k, v in REDIS_CFG.items() if k in valid_redis_args}
        
        # 现在的 conn_cfg 只会有 {'host': 'localhost', 'port': 6379, 'db': 1}
        self.r = redis.Redis(**conn_cfg)
        
        if flush_redis:
            logger.info("🧹 Flushing Redis DB...")
            self.r.flushdb()
            
    def _get_ts_range(self):
        """根据 YYYYMMDD 计算美东开盘闭盘的 Unix 时间戳"""
        ny_tz = timezone('America/New_York')
        dt = datetime.strptime(self.date_str, '%Y%m%d')
        start_dt = ny_tz.localize(dt.replace(hour=9, minute=30, second=0))
        end_dt = ny_tz.localize(dt.replace(hour=16, minute=0, second=0))
        return start_dt.timestamp(), end_dt.timestamp()

    def fetch_fused_data(self):
        """从数据库读取行情与信号的关联宽表"""
        start_ts, end_ts = self._get_ts_range()
        logger.info(f"📥 Fetching fused data from PG for {self.date_str}...")
        
        conn = psycopg2.connect(PG_DB_URL)
        # 这里使用了 LEFT JOIN 逻辑，保证即使某分钟没信号，行情也能发出（Alpha=0）
        # 如果你已经创建了视图，直接 SELECT * FROM fused_view 即可
        query = f"""
            SELECT 
                m.symbol, m.ts, m.close, m.volume,
                COALESCE(a.alpha, 0.0) as alpha,
                COALESCE(a.vol_z, 0.0) as vol_z
            FROM market_bars_1m m
            LEFT JOIN alpha_logs a ON m.symbol = a.symbol AND m.ts = a.ts
            WHERE m.ts >= {start_ts} AND m.ts <= {end_ts}
            ORDER BY m.ts ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def _wait_for_oms_sync(self, ts_val: float, timeout: float = 10.0):
        """[核心同步锁] 等待执行引擎处理完当前帧"""
        sync_key = f"sync:orch_done:{int(ts_val)}"
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if self.r.exists(sync_key):
                self.r.delete(sync_key) # 清理锁
                return True
            time.sleep(0.01)
        logger.warning(f"⏳ Sync Timeout for TS: {ts_val}")
        return False

    def run(self):
        df_fused = self.fetch_fused_data()
        if df_fused.empty:
            logger.error("❌ No data found!")
            return

        grouped = df_fused.groupby('ts')
        total_frames = len(grouped)
        logger.info(f"🚀 Starting Fused Replay: {total_frames} frames to process.")

        frame_id = 0
        for ts_val, group_df in grouped:
            frame_id += 1
            rows = group_df.to_dict('records')
            
            # 1. 组装 Fused Payload
            payload = build_fused_inference_payload(ts_val, frame_id, rows)
            if not payload: continue
            
            # 2. 序列化并推送到 Redis Stream
            packed = ser.pack(payload)
            self.r.xadd(STREAM_ORCH_SIGNAL, {b'pickle': packed})
            
            # 3. 如果是同步模式，等待 OMS 确认执行完毕
            if os.environ.get('SYNC_EXECUTION') == '1':
                self._wait_for_oms_sync(ts_val)
            
            if frame_id % 60 == 0:
                logger.info(f"✅ Processed {frame_id}/{total_frames} minutes...")

        logger.info("🏁 Fused Replay Finished.")

# =========================================================
# 入口函数
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused Market+Alpha Replay Pitcher")
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--no-flush", action="store_true", help="Don't flush Redis")
    args = parser.parse_args()

    pitcher = FusedOMSReplayPitcher(args.date, flush_redis=not args.no_flush)
    
    start_time = time.time()
    pitcher.run()
    end_time = time.time()
    
    logger.info(f"⏱️ Replay Duration: {end_time - start_time:.2f} seconds.")