#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: verify_strategy_playback_sqlite.py
描述: 策略回放验证工具（V2 - 生产级对齐版）。
      直接从 history_sqlite 开发生成的离线数据库中读取 Alpha、K线和期权快照数据，
      并将其“回放”到 V8 Orchestrator 的 Redis 入口流中。
      这用于验证在输入 Alpha 完全一致的情况下，实盘引擎的交易逻辑是否与离线回测一致。
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import redis
import pickle
import json
import time
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# 强制设置环境为回放模式
os.environ['RUN_MODE'] = 'LIVEREPLAY'
from config import REDIS_CFG, DB_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [STRAT_PLAYBACK] - %(message)s')
logger = logging.getLogger("StrategyPlayback")

class StrategyPlayback:
    def __init__(self, db_path, stream_key="backtest_stream"):
        self.r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=REDIS_CFG['db'])
        self.db_path = Path(db_path)
        self.stream_key = stream_key
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"❌ Database not found: {self.db_path}")

    def load_full_market_data(self):
        logger.info(f"📂 Loading data from {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        
        # 1. 加载 Alpha 日志
        logger.info("  -> Reading alpha_logs...")
        df_alpha = pd.read_sql_query("SELECT ts, symbol, alpha, price, vol_z FROM alpha_logs", conn)
        
        # 2. 加载 K 线数据
        logger.info("  -> Reading market_bars_1m...")
        df_bars = pd.read_sql_query("SELECT ts, symbol, close, volume FROM market_bars_1m", conn)
        
        # 3. 加载期权快照
        logger.info("  -> Reading option_snapshots_1m...")
        df_opts = pd.read_sql_query("SELECT ts, symbol, buckets_json FROM option_snapshots_1m", conn)
        
        conn.close()
        
        logger.info("🧩 Merging tables...")
        # 以 Alpha 为基准进行合并
        df = pd.merge(df_alpha, df_bars, on=['ts', 'symbol'], how='left')
        df = pd.merge(df, df_opts, on=['ts', 'symbol'], how='left')
        
        df.sort_values(['ts', 'symbol'], inplace=True)
        return df

    def run(self, symbols_filter=None):
        df = self.load_full_market_data()
        if df.empty:
            logger.error("❌ No data found.")
            return

        if symbols_filter:
            df = df[df['symbol'].isin(symbols_filter)]

        # 彻底清理 Redis 遗留状态
        self.r.delete(self.stream_key)
        self.r.delete("trade_log_stream") # 清理旧的交易日志
        try:
            self.r.xgroup_destroy(self.stream_key, "v8_orch_group")
        except: pass

        logger.info(f"🚀 Starting Playback: {len(df)} rows across {df['ts'].nunique()} intervals...")
        
        grouped = df.groupby('ts')
        count = 0
        
        # 为了生成 PnL 曲线，我们需要记录初始资金
        from config import INITIAL_ACCOUNT
        equity = INITIAL_ACCOUNT
        equity_curve = []

        for ts_val, group in tqdm(grouped, desc="Replaying to Redis"):
            symbols = group['symbol'].tolist()
            prices = group['close'].fillna(group['price']).values.astype(np.float32)
            alphas = group['alpha'].values.astype(np.float32)
            # note: vol_z 在这里对应模型直接输出的 z-score
            vol_z = group['vol_z'].fillna(0.0).values.astype(np.float32)
            
            live_options = {}
            for _, row in group.iterrows():
                try:
                    if pd.notnull(row['buckets_json']):
                        # 直接透传字典结构，对齐生产环境 `_get_opt_data_realtime` 需要的格式
                        live_options[row['symbol']] = json.loads(row['buckets_json'])
                except Exception as e:
                    logger.debug(f"JSON decode error for {row['symbol']} @ {ts_val}: {e}")

            packet = {
                'symbols': symbols,
                'ts': float(ts_val),
                'stock_price': prices,
                'precalc_alpha': alphas,
                'fast_vol': vol_z, # 在离线 feed 中，vol_z 已经代表了预处理后的量能特征
                'live_options': live_options,
                'is_new_minute': True, # 回放离线数据，每一条都是新的逻辑分钟
                
                # 兼容旧代码需要的占位符
                'spy_roc_5min': np.zeros(len(symbols), dtype=np.float32),
                'qqq_roc_5min': np.zeros(len(symbols), dtype=np.float32),
                'slow_1m': np.zeros((len(symbols), 30, 1), dtype=np.float32),
            }
            
            # 推送到 Redis
            self.r.xadd(self.stream_key, {'data': pickle.dumps(packet)})
            self.r.set("replay:current_ts", str(ts_val))
            count += 1
            
        # 设置完成标志
        run_id = f"verification_{int(time.time())}"
        self.r.set(f"replay:status:{run_id}", "DONE")
        logger.info(f"✅ Playback Finished. Sent {count} batches. RunID: {run_id}")

        # --- 新增：结果统计与报表生成 ---
        self.generate_reports(run_id)

    def generate_reports(self, run_id):
        logger.info("📊 Collecting results from trade_log_stream...")
        time.sleep(2) # 等待 Orchestrator 处理完最后的平仓
        
        raw_logs = self.r.xrange("trade_log_stream")
        trades = []
        for _, entry in raw_logs:
            if b'pickle' in entry:
                payload = pickle.loads(entry[b'pickle'])
                # 我们只关心平仓记录来计算 PnL
                if payload.get('action') == 'CLOSE':
                    trades.append(payload)
        
        if not trades:
            logger.warning("⚠️ No trades recorded in this run. Check your Alpha thresholds.")
            return

        df_trades = pd.DataFrame(trades)
        
        # 1. 保存详细成交记录
        os.makedirs("logs", exist_ok=True)
        csv_trades = f"logs/playback_trades_{run_id}.csv"
        df_trades.to_csv(csv_trades, index=False)
        logger.info(f"💾 Detailed trades saved to: {csv_trades}")

        # 2. 生成 PnL 曲线
        df_trades['cum_pnl'] = df_trades['pnl'].cumsum()
        from config import INITIAL_ACCOUNT
        df_trades['equity'] = INITIAL_ACCOUNT + df_trades['cum_pnl']
        
        csv_equity = f"logs/playback_equity_{run_id}.csv"
        df_trades[['ts', 'symbol', 'pnl', 'equity']].to_csv(csv_equity, index=False)
        logger.info(f"📈 Equity curve saved to: {csv_equity}")

        # 3. 打印摘要对比
        total_pnl = df_trades['pnl'].sum()
        win_rate = (df_trades['pnl'] > 0).mean()
        logger.info(f"🏁 Final Summary: Net PnL: ${total_pnl:,.2f} | Win Rate: {win_rate:.1%} | Count: {len(df_trades)}")

        print("\n" + "="*50)
        print("🚀 验证结论:")
        print(f"- 累计盈亏: ${total_pnl:,.2f}")
        print(f"- 交易笔数: {len(df_trades)}")
        print(f"- 盈亏文件: {csv_equity}")
        print("💡 请将此文件与离线回测产生的 csv 进行对比，观察曲线是否重合。")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database (e.g. data/history_sqlite/market_20260302.db)")
    args = parser.parse_args()
    
    # 默认使用通用回放流名
    playback = StrategyPlayback(args.db, stream_key="unified_inference_stream")
    playback.run()
