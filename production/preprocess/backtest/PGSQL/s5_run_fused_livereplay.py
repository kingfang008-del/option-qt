#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: s5_run_fused_livereplay.py
描述: [下一代回测基准] 基于 Fused 视图 + Redis Stream 的一体化 LiveReplay
特点: 
  1. 零模型推理 (预读 fused_market_alpha_1m)
  2. 执行引擎使用实盘完全一致的 Redis 消费链路
  3. 通过 sync_key 保证协程级别的帧同步，绝不错位
"""

import os
import sys
import time
import logging
import asyncio
import numpy as np
import pandas as pd
import redis
import psycopg2
from datetime import datetime
from pytz import timezone

# =========================================================
# 🚀 1. [S5 对齐配置] 强制环境注入，确保高保真执行
# =========================================================
os.environ['RUN_MODE']              = 'LIVEREPLAY'  # 激活 EE 的 Redis 消费与 MockIBKR 撮合
os.environ['SYNC_EXECUTION']        = '1'           # 强制严格帧同步
os.environ['PURE_ALPHA_REPLAY']     = '1'           # 纯净信号回放
os.environ['DISABLE_ICEBERG']       = '1'           # 禁用冰山拆单，对齐基础收益
os.environ['DUAL_CONVERGE_TO_SINGLE'] = '1'         # 对齐单向收敛

# =========================================================
# 导入业务模块 (确保在设置环境变量之后导入)
# =========================================================
from config import PG_DB_URL, STREAM_ORCH_SIGNAL, REDIS_CFG, TARGET_SYMBOLS
from utils import serialization_utils as ser

# 导入执行引擎与 Mock 撮合器
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical_1s import MockIBKRHistorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FusedReplay")

# =========================================================
# 🚀 2. 异步发球机 (Async Pitcher)
# =========================================================
class AsyncFusedPitcher:
    def __init__(self, date_str: str, mock_ibkr: MockIBKRHistorical):
        self.date_str = date_str
        self.mock_ibkr = mock_ibkr # 将 mock 传入，为了提前灌入价格基准
        
        # 白名单过滤 Redis 配置
        valid_redis_args = ['host', 'port', 'db', 'password', 'socket_timeout', 'decode_responses']
        conn_cfg = {k: v for k, v in REDIS_CFG.items() if k in valid_redis_args}
        self.r = redis.Redis(**conn_cfg)
        
        logger.info("🧹 Flushing Redis DB for clean run...")
        self.r.flushdb()
        
    def _get_ts_range(self):
        ny_tz = timezone('America/New_York')
        dt = datetime.strptime(self.date_str, '%Y%m%d')
        start_dt = ny_tz.localize(dt.replace(hour=9, minute=30, second=0))
        end_dt = ny_tz.localize(dt.replace(hour=16, minute=0, second=0))
        return start_dt.timestamp(), end_dt.timestamp()

    def fetch_fused_data(self):
        start_ts, end_ts = self._get_ts_range()
        logger.info(f"📥 Loading fused view from PG for {self.date_str}...")
        conn = psycopg2.connect(PG_DB_URL)
        # 直接读取我们在数据库做好的 1m 行情+信号大宽表
        query = f"""
            SELECT * FROM fused_market_alpha_1m 
            WHERE ts >= {start_ts} AND ts <= {end_ts}
            ORDER BY ts ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def build_payload(self, ts_val, frame_id, rows):
        symbols, stock_price, stock_volume, precalc_alpha, fast_vol = [], [], [], [], []
        spy_rocs, qqq_rocs = [], []
        
        for row in rows:
            symbols.append(row['symbol'])
            stock_price.append(float(row['close']))
            stock_volume.append(float(row['volume']))
            precalc_alpha.append(float(row.get('precalc_alpha', 0.0)))
            fast_vol.append(float(row.get('fast_vol', 0.0)))
            spy_rocs.append(float(row.get('spy_roc_5min', 0.0)))
            qqq_rocs.append(float(row.get('qqq_roc_5min', 0.0)))

        if not symbols: return None

        return {
            "ts": float(ts_val),
            "frame_id": frame_id,
            "symbols": symbols,
            "stock_price": np.asarray(stock_price, dtype=np.float32),
            "stock_volume": np.asarray(stock_volume, dtype=np.float32),
            "precalc_alpha": np.asarray(precalc_alpha, dtype=np.float32),
            "fast_vol": np.asarray(fast_vol, dtype=np.float32),
            "spy_roc_5min": np.asarray(spy_rocs, dtype=np.float32),
            "qqq_roc_5min": np.asarray(qqq_rocs, dtype=np.float32),
            "alpha_label_ts": np.full(len(symbols), float(ts_val), dtype=np.float64),
            "source": "fused_replay_v8"
        }

    async def _wait_for_oms_sync(self, ts_val: float):
        """异步等待执行引擎处理完毕，不阻塞主事件循环"""
        sync_key = f"sync:orch_done:{int(ts_val)}"
        while True:
            if self.r.exists(sync_key):
                self.r.delete(sync_key)
                return
            await asyncio.sleep(0.005) # 让出 CPU 控制权给 Execution Engine

    async def run_pitching(self):
        df_fused = self.fetch_fused_data()
        if df_fused.empty:
            logger.error("❌ No data found!")
            return

        grouped = df_fused.groupby('ts')
        total_frames = len(grouped)
        logger.info(f"🚀 Started Pitching {total_frames} frames to Stream: {STREAM_ORCH_SIGNAL}")

        frame_id = 0
        for ts_val, group_df in grouped:
            frame_id += 1
            rows = group_df.to_dict('records')
            
            payload = self.build_payload(ts_val, frame_id, rows)
            if not payload: continue
            
            # 提前向 MockIBKR 灌入当刻的价格，确保执行成交有基准价
            # 注: 如果你的 EE 自己会更新价格，这步可以保留作为双保险
            if hasattr(self.mock_ibkr, 'record_market_data'):
                self.mock_ibkr.record_market_data(payload, alphas=payload['precalc_alpha'])
            
            # 打包发送到执行引擎的信号队列
            packed = ser.pack(payload)
            self.r.xadd(STREAM_ORCH_SIGNAL, {b'pickle': packed})
            
            # 严格帧同步等待
            await self._wait_for_oms_sync(ts_val)
            
            if frame_id % 60 == 0:
                print(f"✅ Processed {frame_id}/{total_frames} minutes...", end='\r')

        print(f"\n🏁 All {total_frames} frames pitched successfully.")


# =========================================================
# 🚀 3. 主控制流 (Main Loop)
# =========================================================
async def run_unified_replay(date_str: str):
    start_time = time.time()
    
    # 初始化底座
    mock_ibkr = MockIBKRHistorical()
    mock_ibkr.initial_capital = 50000.0  # 手动设置初始资金
    # 🚀 [Fix] 传入 TARGET_SYMBOLS 必填参数，并注入外部 mock_ibkr
    exec_engine = ExecutionEngineV8(TARGET_SYMBOLS, ibkr=mock_ibkr)
    pitcher = AsyncFusedPitcher(date_str, mock_ibkr=mock_ibkr)
    
    # 将 ExecutionEngine 的主监听器放入后台任务 (守护协程)
    logger.info("启动执行引擎 Redis 监听器...")
    ee_task = asyncio.create_task(exec_engine.run())
    
    # 稍微等一秒，确保 EE 的 Consumer Group 创建好
    await asyncio.sleep(1)
    
    # 启动发球机 (阻塞主线程，直到所有数据发完)
    await pitcher.run_pitching()
    
    # 发球结束，清盘
    logger.info("发起全剧终清仓指令...")
    await exec_engine.force_close_all()
    await asyncio.sleep(0.5) # 给清盘任务一点时间执行
    
    # 停止执行引擎监听器
    ee_task.cancel()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Fused LiveReplay Finished in {elapsed:.1f}s.")
    
    # 保存并打印账单
    mock_ibkr.save_trades(filename=f"replay_trades_s5_fused_{date_str}.csv")
    
    print("\n" + "=" * 50)
    print(f"📊 FINAL BACKTEST PERFORMANCE (S5 FUSED - {date_str})")
    print("=" * 50)
    print(f"💰 Final Cash Balance: ${exec_engine.mock_cash:,.2f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python s5_run_fused_livereplay.py YYYYMMDD")
        sys.exit(1)
        
    target_date = sys.argv[1]
    
    try:
        asyncio.run(run_unified_replay(target_date))
    except KeyboardInterrupt:
        print("\n⏹️ Replay Terminated by User.")