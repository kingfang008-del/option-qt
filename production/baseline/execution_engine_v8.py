#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: system_orchestrator_v8.py
描述: [V8 最终生产版 - 时间基准修复]
修复日志:
    1. [CRITICAL] 修复 "300分钟幽灵持仓" 问题。
       - 原因: 开仓时间(Wall Time)与当前时间(Real NY Time)存在5小时时区差。
       - 解决: _execute_entry 和 _execute_exit 统一使用 process_batch 算出的 curr_ts。
    2. [Log] 修复日志时间显示问题。
       - 原因: datetime.fromtimestamp 默认使用系统本地时区。
       - 解决: 强制转换为 'America/New_York' 进行日志显示。
"""

import asyncio
import math
import copy

import redis

import logging
import json
import numpy as np
import time
import re
import pandas as pd
import os
import psycopg2
from config import PG_DB_URL
import threading
import os
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from collections import deque
from pytz import timezone
from scipy.stats import norm 
import traceback

# 引入纯策略核心


# [Refactor] 引入模块化执行组件
from orchestrator_state_manager import OrchestratorStateManager
from orchestrator_accounting import OrchestratorAccounting
from orchestrator_execution import OrchestratorExecution
from orchestrator_reconciler import OrchestratorReconciler
from utils import serialization_utils as ser

 
    # 🚀 [Fix] 切换为秒级专用模拟器 (1s Precision)
from mock_ibkr_historical_1s import MockIBKRHistorical
     
 
 
from ibkr_connector_v8 import IBKRConnectorFinal
 
from config import get_redis_db, RUN_MODE, REDIS_CFG

# Log Stream Key
from config import (
    STREAM_TRADE_LOG,           # [Fix] Use shared config
    SYNC_EXECUTION,
    STREAM_FUSED_MARKET,        # [New] Fast Tick Stream for exits
    STREAM_ORCH_SIGNAL,        # [New] SE → OMS
    GROUP_OMS,                  # [New] OMS Consumer Group
    TRADING_ENABLED,            # 全局交易开关 (True=实盘下单, False=只读模式)
    MAX_POSITIONS,              # 最大同时持仓数
    POSITION_RATIO,             # 单标的最大仓位比例
    MAX_TRADE_CAP,              # 单笔交易最大金额
    GLOBAL_EXPOSURE_LIMIT,      # 全局风险敞口上限
    COMMISSION_PER_CONTRACT,    # 期权手续费 ($/手)
    SLIPPAGE_ENTRY_PCT,         # [Fix] Added to imports
    SLIPPAGE_EXIT_PCT,          # [Fix] Added to imports
    OMS_SIGNAL_DELAY_BARS,
    OMS_SIGNAL_DELAY_ACTIONS,
    IS_LIVEREPLAY,
    IS_BACKTEST,
    IS_SIMULATED,
    IS_REALTIME_DRY
)


#from train_fast_channel_microstructure import FastMicrostructureModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s')
logger = logging.getLogger("V8_ExecutionEngine")

# [Fix] 显式添加 FileHandler 确保写入文件
from config import LOG_DIR
file_handler = logging.FileHandler(LOG_DIR / "ExecutionEngine.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

 
# Log Stream Key
from config import (
    REDIS_CFG,                  # [🔥 新增] 统一导入 Redis 配置
    STREAM_TRADE_LOG,           
    STREAM_FUSED_MARKET,        
    TRADING_ENABLED)


# [联动] 实盘平仓防滑点开关: TRADING_ENABLED=True → LMT (防做市商收割)
#                           TRADING_ENABLED=False → MKT (极速回测)
# 注意：EXIT_ORDER_TYPE 现在可以从 config.py 或 strategy_config 获取
EXIT_ORDER_TYPE = 'LMT' if TRADING_ENABLED else 'MKT'

 

# COMMISSION_PER_CONTRACT 已从 config.py 导入


# [BSM 定价函数已废弃]
# def black_scholes_price(S, K, T, r, sigma, option_type='call'):
#     ...

class SymbolState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.prices = deque(maxlen=60)
        
        # Position State
        self.position = 0      # 0, 1(Call), -1(Put)
        self.qty = 0
        self.entry_price = 0.0 
        self.entry_stock = 0.0 
        self.entry_ts = 0.0
        self.entry_spy_roc = 0.0 

        self.entry_index_trend = 0  
        self.entry_alpha_z = 0.0 
        self.entry_iv = 0.0      
        self.last_alpha_z = 0.0  
        self.prev_alpha_z = 0.0  
        self.max_roi = -1.0
        self.cooldown_until = 0.0
        self.contract_id = None
        self.latest_call_id = "" 
        self.latest_put_id = ""  
        self.is_pending = False
        self.pending_side = None
        self.pending_action = None # [幽灵 C] 记录当前锁定的动作
        self.pending_ts = 0.0      # [幽灵 C] 记录加锁时间戳

        self.prev_macd_hist = 0.0
        
        self.strike_price = 0.0
        self.expiry_date = None 
        self.last_valid_iv = 0.5
        self.opt_type = 'call'

        self.warmup_complete = False
        
        self.last_spread_pct = 0.0
        self.last_snap_roc = 0.0
        self.last_vol_z = 0.0
        self.last_min_ts = 0
        # [State Safety] 初始化持久化字段，防止 to_dict/from_dict 期间 AttributeError
        self.correction_mode = 'NORMAL'
        self.alpha_history = deque(maxlen=120)
        self.pct_history = deque(maxlen=120)
        self.ema_fast_val = None
        self.ema_slow_val = None
        self.dea_val = None

    def get_reversal_count(self, window_mins=30, threshold=0.001):
        """[Market Regime Guard] 计算过去 N 分钟内价格反转(洗盘)的频率"""
        if len(self.prices) < 2: return 0
        prices_list = list(self.prices)[-window_mins:]
        if len(prices_list) < 2: return 0
        reversals = 0
        last_dir = 0
        for i in range(1, len(prices_list)):
            diff_pct = (prices_list[i] - prices_list[i-1]) / prices_list[i-1]
            if abs(diff_pct) >= threshold:
                curr_dir = 1 if diff_pct > 0 else -1
                if last_dir != 0 and curr_dir != last_dir:
                    reversals += 1
                last_dir = curr_dir
        return reversals

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'position': self.position,
            'qty': self.qty,
            'entry_price': self.entry_price,
            'entry_stock': self.entry_stock,
            'entry_ts': self.entry_ts,
            'entry_spy_roc': self.entry_spy_roc,
            'entry_index_trend': self.entry_index_trend,  # <--- [新增] 存入数据库
            'entry_alpha_z': self.entry_alpha_z,   # 🚨 [修复]
            'entry_iv': self.entry_iv,             # 🚨 [修复]
            'max_roi': self.max_roi,
            'cooldown_until': self.cooldown_until,
            'contract_id': self.contract_id,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'last_valid_iv': self.last_valid_iv,
            'opt_type': self.opt_type,
            'warmup_complete': self.warmup_complete,
            'correction_mode': self.correction_mode, # 🚨 [修复]
            'prev_macd_hist': self.prev_macd_hist,   # 🚨 [修复]
            'last_spread_pct': self.last_spread_pct, # 🚨 [修复]

            # [新增] 历史数据 Buffer 持久化
            'prices': list(self.prices),
            'alpha_history': list(self.alpha_history),
            'pct_history': list(self.pct_history),
            'ema_fast_val': self.ema_fast_val,
            'ema_slow_val': self.ema_slow_val,
            'dea_val': self.dea_val
        }

    def from_dict(self, data):
        """从字典恢复状态 (含 Buffer)"""
        self.position = data.get('position', 0)
        self.qty = data.get('qty', 0)
        self.entry_price = data.get('entry_price', 0.0)
        self.entry_stock = data.get('entry_stock', 0.0)
        self.entry_ts = data.get('entry_ts', 0.0)
        self.entry_spy_roc = data.get('entry_spy_roc', 0.0)
        self.entry_index_trend = data.get('entry_index_trend', 0)  # <--- [新增] 安全恢复，老数据默认给 0
        self.entry_alpha_z = data.get('entry_alpha_z', 0.0)       # 🚨 [修复]
        self.entry_iv = data.get('entry_iv', 0.0)                 # 🚨 [修复]
        self.max_roi = data.get('max_roi', -1.0)
        self.cooldown_until = data.get('cooldown_until', 0.0)
        self.contract_id = data.get('contract_id')
        self.strike_price = data.get('strike_price', 0.0)
        
        ex_str = data.get('expiry_date')
        if ex_str:
            try: self.expiry_date = datetime.fromisoformat(ex_str)
            except: self.expiry_date = None
            
        self.last_valid_iv = data.get('last_valid_iv', 0.5)
        self.opt_type = data.get('opt_type', 'call')
        self.warmup_complete = data.get('warmup_complete', False)

        self.correction_mode = data.get('correction_mode', 'NORMAL') # 🚨 [修复]
        self.prev_macd_hist = data.get('prev_macd_hist', 0.0)        # 🚨 [修复]
        self.last_spread_pct = data.get('last_spread_pct', 0.0)      # 🚨 [修复]
        
        # [新增] 恢复 Buffer
        if 'prices' in data: self.prices = deque(data['prices'], maxlen=self.prices.maxlen)
        if 'alpha_history' in data: self.alpha_history = deque(data['alpha_history'], maxlen=self.alpha_history.maxlen)
        if 'pct_history' in data: self.pct_history = deque(data['pct_history'], maxlen=self.pct_history.maxlen)
        
        self.ema_fast_val = data.get('ema_fast_val')
        self.ema_slow_val = data.get('ema_slow_val')
        self.dea_val = data.get('dea_val')

class ExecutionEngineV8:
    def __init__(self, symbols, mode='realtime', config_paths=None, model_paths=None, shared_states=None, signal_queue=None):
        print(f"DEBUG: V8Orchestrator Initializing... Mode={mode}")
        self.mode = mode
        self.symbols = symbols
        
        # 🚀 [架构分流] 如果注入了 shared_states，直接使用该物理内存！
        if shared_states is not None:
            self.states = shared_states
        else:
            self.states = {s: SymbolState(s) for s in symbols}
            
        self.symbol_states = self.states # Alias
        self.signal_queue = signal_queue
        self.use_shared_mem = (signal_queue is not None and shared_states is not None)
        # Strategy runs on Signal Engine

        self.disable_db_save = True
        
        # [Refactor] 模块化组件初始化
        self.state_manager = OrchestratorStateManager(self)
        self.accounting = OrchestratorAccounting(self)
        self.execution = OrchestratorExecution(self)
        self.reconciler = OrchestratorReconciler(self)
        
        # Redis Init
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print("DEBUG: Redis Initialized.")
        
        # IBKR backend selection:
        # - REALTIME / REALTIME_DRY: real connector
        # - BACKTEST / LIVEREPLAY: mock connector
        try:
            if self.mode == 'realtime' and not IS_SIMULATED:
                self.ibkr = IBKRConnectorFinal(client_id=999)
                logger.info(
                    "🔌 [OMS] IBKR backend=REAL | RUN_MODE=%s | IS_REALTIME_DRY=%s | TRADING_ENABLED=%s",
                    RUN_MODE, IS_REALTIME_DRY, TRADING_ENABLED
                )
            else:
                self.ibkr = MockIBKRHistorical()
                logger.info("🧪 [OMS] IBKR backend=MOCK | RUN_MODE=%s", RUN_MODE)
        except Exception as e:
            logger.error(f"❌ [OMS] IBKR backend init failed: {e}")
            self.ibkr = None

        # =========================================================
        # 🚀 [新增] 动态 Alpha 归一化追踪器 (Dynamic Alpha Tracker)
        # =========================================================
        self.dynamic_alpha_mean = 0.0
        self.dynamic_alpha_std = 1.0
        self.alpha_count = 0

        # Create lightweight config proxy for execution/accounting dependencies
        from strategy_selector import StrategyConfig
        class DummyStrategy:
            def __init__(self):
                self.cfg = StrategyConfig()
        self.strategy = DummyStrategy()

        # Global State Defaults
        # [已统一] 所有的策略、风控参数通过 self.strategy.cfg 访问
        self.cfg = self.strategy.cfg
        
        self.last_date = None
        self.mock_cash = self.cfg.INITIAL_ACCOUNT
        self.index_opening_prices = {}
        self.consecutive_stop_losses = 0
        self.global_cooldown_until = 0
        
        # [新增] 延时信号队列
        self.delayed_signal_queue = []
        
        # 兼容性重定向 (指向统一的 config 对象)
        self.CIRCUIT_BREAKER_THRESHOLD = self.cfg.CIRCUIT_BREAKER_THRESHOLD
        self.CIRCUIT_BREAKER_MINUTES = self.cfg.CIRCUIT_BREAKER_MINUTES
        self.MIN_OPTION_PRICE = self.cfg.MIN_OPTION_PRICE
        
        self.last_save_time = 0
        self.last_pnl_report_ts = 0.0

        # [新增] 逆势交易统计 (Counter-trend tracking)
        self.stats_counter_trend_long_count = 0
        self.stats_counter_trend_long_win_count = 0
        self.stats_counter_trend_long_pnl = 0.0
        self.stats_counter_trend_short_count = 0
        self.stats_counter_trend_short_win_count = 0
        self.stats_counter_trend_short_pnl = 0.0
        self.stats_liquidity_drought_liquidations = 0

        # [🔥 核心 PnL 监控 - Ground Truth]
        self.realized_pnl = 0.0          # 累计已实现盈亏 (扣除手续费)
        self.last_broadcast_ts = 0.0
        self.total_commission = 0.0      # 累计手续费
        self.trade_count = 0             # 总交易笔数
        self.win_count = 0               # 盈利笔数
        self.loss_count = 0              # 亏损笔数

        # [NEW] 每日交易数据池 (用于盘后分析)
        self.daily_trades = []
        self.last_index_trend = 0 
        self.spy_ema_roc = 0.0     # [NEW] 5min EMA ROC (势能平滑)
        self.qqq_ema_roc = 0.0
        
        # 赋予一个先验初始值 (防止冷启动期间的极端缩放)
        self.dynamic_vol_mean = 0.0739 
        self.dynamic_vol_std = 0.1106
        
    def _deep_decode_bytes(self, data):
        """[Diagnostics] 递归将字典/列表中的所有 bytes 键值对转换为 str，根治 JSON 报错"""
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='replace')
        if isinstance(data, list):
            return [self._deep_decode_bytes(i) for i in data]
        if isinstance(data, dict):
            return {
                (k.decode('utf-8', errors='replace') if isinstance(k, bytes) else k): self._deep_decode_bytes(v)
                for k, v in data.items()
            }
        return data

    def _reconstruct_market_packet(self, payload_list):
        """[Parity] 将 Pitcher 发来的分品种 Payload 列表重组为 S4 模拟器所需的 Batch 格式"""
        if not payload_list or not isinstance(payload_list, list):
            return None
        
        # 提取公共时间戳
        ts = payload_list[0].get('ts', 0.0)
        
        reconstructed = {
            'ts': float(ts),
            'symbols': [],
            'stock_price': [],
            'feed_call_price': [],
            'feed_put_price': [],
            'feed_call_bid': [],
            'feed_call_ask': [],
            'feed_put_bid': [],
            'feed_put_ask': []
        }
        
        for p in payload_list:
            reconstructed['symbols'].append(p.get('symbol'))
            reconstructed['stock_price'].append(p.get('stock', {}).get('close', 0.0))
            
            buckets = p.get('option_buckets', [])
            # 索引对齐 S4 (s4_run_historical_replay_pg_1s.py): Bucket 0=Put, 2=Call
            put_bk = buckets[0] if len(buckets) > 0 else []
            call_bk = buckets[2] if len(buckets) > 2 else []
            
            # 数值对齐 S4: 0=mid/price, 8=bid, 9=ask
            reconstructed['feed_put_price'].append(float(put_bk[0]) if len(put_bk) > 0 else 0.0)
            reconstructed['feed_put_bid'].append(float(put_bk[8]) if len(put_bk) > 8 else 0.0)
            reconstructed['feed_put_ask'].append(float(put_bk[9]) if len(put_bk) > 9 else 0.0)
            
            reconstructed['feed_call_price'].append(float(call_bk[0]) if len(call_bk) > 0 else 0.0)
            reconstructed['feed_call_bid'].append(float(call_bk[8]) if len(call_bk) > 8 else 0.0)
            reconstructed['feed_call_ask'].append(float(call_bk[9]) if len(call_bk) > 9 else 0.0)
            
        return reconstructed


       
        
        # ML Models run on Signal Engine

        

    def _mark_exec_done(self, curr_ts: float, frame_id: str = None):
        """[Parity] 向 Redis 报告执行引擎已完成当前帧的处理，解除发球机阻塞"""
        if not hasattr(self, 'r'): return
        # 🚀 [S5 对齐] 发球机在等待这个特定格式的 Key
        sync_key = f"sync:orch_done:{int(curr_ts)}"
        self.r.set(sync_key, "1")
        self.r.expire(sync_key, 60)
        
        # 兼容旧版汇报
        self.r.set("sync:exec_done", str(curr_ts))
        self.r.expire("sync:exec_done", 120)
        if frame_id:
            self.r.set("sync:exec_done_frame_id", str(frame_id))
            self.r.expire("sync:exec_done_frame_id", 120)
        try:
            self.r.hincrby("diag:ee:counters", "exec_done", 1)
        except Exception:
            pass

    def _get_fair_market_price(self, base_price: float, bid: float, ask: float, prev_price: float = 0.0) -> float:
        """
        [NEW]统一计算期权公允市价
        如果是真空切片 (Bid/Ask 均为 0)，且 Last Price 与前一秒偏差超过 10%，则沿用上一秒价格。
        """
        # 计算市场价 (Mid)
        if bid > 0.01 and ask > 0.01:
            market_price = (bid + ask) / 2.0
        elif bid > 0.01:
            market_price = bid
        elif ask > 0.01:
            market_price = ask
        else:
            market_price = base_price
            # [🔥 高频防插针核心逻辑]
            if prev_price > 0.01 and market_price > 0.01:
                if abs(market_price - prev_price) / prev_price > 0.10:
                    market_price = prev_price

        return market_price

    def _should_delay_signal(self, action: str) -> bool:
        return OMS_SIGNAL_DELAY_BARS > 0 and str(action).upper() in OMS_SIGNAL_DELAY_ACTIONS

    def _eligible_trade_ts(self, signal_ts: float) -> float:
        return float(signal_ts) + float(OMS_SIGNAL_DELAY_BARS * 60)

    async def _queue_delayed_signal(self, payload: dict):
        sym = payload.get('symbol')
        action = str(payload.get('action', '')).upper()
        curr_ts = float(payload.get('ts', 0.0) or 0.0)
        eligible_ts = self._eligible_trade_ts(curr_ts)
        queued_payload = copy.deepcopy(payload)
        queued_payload['_delay_eligible_ts'] = eligible_ts
        self.delayed_signal_queue.append(queued_payload)
        self.delayed_signal_queue.sort(key=lambda x: float(x.get('_delay_eligible_ts', 0.0)))

        if sym in self.states:
            st = self.states[sym]
            st.is_pending = True
            st.pending_side = action

        logger.info(
            f"⏳ [OMS Delay] Queued {action} for {sym} | "
            f"signal_ts={curr_ts:.0f} | eligible_ts={eligible_ts:.0f} | delay_bars={OMS_SIGNAL_DELAY_BARS}"
        )
        await self._broadcast_state_to_redis()

    async def _flush_delayed_signals(self, curr_ts: float):
        if not self.delayed_signal_queue:
            return

        ready = []
        pending = []
        for item in self.delayed_signal_queue:
            if float(item.get('_delay_eligible_ts', 0.0)) <= float(curr_ts):
                ready.append(item)
            else:
                pending.append(item)
        self.delayed_signal_queue = pending

        for item in ready:
            sym = item.get('symbol')
            action = str(item.get('action', '')).upper()
            if sym in self.states:
                st = self.states[sym]
                st.is_pending = False
                st.pending_side = None

            released = copy.deepcopy(item)
            released['_delay_released'] = True
            logger.info(
                f"▶️ [OMS Delay] Releasing {action} for {sym} at ts={curr_ts:.0f} "
                f"(eligible_ts={float(item.get('_delay_eligible_ts', 0.0)):.0f})"
            )
            await self._handle_trade_signal(released, allow_delay_queue=False)

    def _prepare_ny_time(self, batch: dict):
        """[Refactor] 统一解析美东时间及处理保存逻辑"""
        from datetime import timezone as dt_timezone
        from pytz import timezone
        ny_tz = timezone('America/New_York')
        ts_raw = batch.get('ts')
        
        if ts_raw is not None:
            ts_float = float(ts_raw)
            if ts_float > 1e12: ts_float /= 1000.0
            dt_utc = datetime.fromtimestamp(ts_float, dt_timezone.utc)
            ny_now = dt_utc.astimezone(ny_tz)
        else:
            ny_now = datetime.now(ny_tz)

        curr_ts = ny_now.timestamp()
        # 👇 [🔥 修复: 全局保存逻辑时间，供高频tick备用，绝不让物理时间渗透！]
        self.last_curr_ts = curr_ts
        # 👆
        
        # 周期性保存状态 (每60秒)
        if time.time() - self.last_save_time > 60:
            self.save_state()
            self.last_save_time = time.time()
            self._publish_warmup_status()
        
        # 🚀 [🔥 核心新增] 周期性 PnL 汇报 (逻辑时间基准: 每 15 逻辑分钟)
        # 即使在极速回放中，也会按照仿真时间的流逝进行稳定汇报
        if curr_ts - self.last_pnl_report_ts >= 900:  # 15分钟逻辑时间
            # 调用原本的 monitor 逻辑，但传入当前逻辑时间戳
            asyncio.create_task(self._report_pnl_status_logic(curr_ts, "LOGICAL"))
            self.last_pnl_report_ts = curr_ts
        
        # 交易时间过滤
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = ny_now.time()
        
        if current_time < market_open or current_time > market_close:
            has_position = any(st.position != 0 for st in self.states.values())
            if not has_position:
                return None, None
            
        return ny_now, curr_ts

    # === Execution Delegations ===
    async def _execute_entry(self, sym, sig, stock_price, curr_ts, batch_idx):
        return await self.execution._execute_entry(sym, sig, stock_price, curr_ts, batch_idx)

    async def _execute_exit(self, sym, sig, stock_price, curr_ts, batch_idx):
        return await self.execution._execute_exit(sym, sig, stock_price, curr_ts, batch_idx)

    async def _iceberg_open_order(self, sym, sig, stock_price, curr_ts, target_total_qty, chunks, fill_price, total_est_cost, total_est_comm):
        return await self.execution._iceberg_open_order(sym, sig, stock_price, curr_ts, target_total_qty, chunks, fill_price, total_est_cost, total_est_comm)

    async def _monitor_realtime_order(self, sym, trade, cost, commission, expected_qty, start_time, limit_price, stock_price, sig, st):
        return await self.execution._monitor_realtime_order(sym, trade, cost, commission, expected_qty, start_time, limit_price, stock_price, sig, st)

    async def _force_clear_all(self, batch, reason, custom_ts, custom_dt):
        return await self.execution._force_clear_all(batch, reason, custom_ts, custom_dt)

    async def force_close_all(self):
        return await self.execution.force_close_all()

    async def _smart_exit_order(self, sym, real_contract, total_qty, base_price, stock_price, curr_ts=None, is_force=False, bid=0.0, ask=0.0, reason=""):
        return await self.execution._smart_exit_order(sym, real_contract, total_qty, base_price, stock_price, curr_ts, is_force, bid, ask, reason)

    async def _emergency_cancel_all(self, sym, st):
        return await self.execution._emergency_cancel_all(sym, st)

    # === Accounting Delegations ===
    def _process_exit_accounting(self, sym, st, filled_qty, fill_price, stock_price, curr_ts, reason, duration, ratio):
        return self.accounting._process_exit_accounting(sym, st, filled_qty, fill_price, stock_price, curr_ts, reason, duration, ratio)

    def _generate_daily_analysis_report(self, report_date_str: str = None):
        return self.accounting._generate_daily_analysis_report(report_date_str)

    def _emit_trade_log(self, payload):
        return self.accounting._emit_trade_log(payload)

    def print_counter_trend_summary(self):
        return self.accounting.print_counter_trend_summary()

    async def _pnl_monitor_loop(self):
        return await self.accounting._pnl_monitor_loop()

    async def _report_pnl_status_logic(self, timestamp, label="Summary"):
        return await self.accounting._report_pnl_status_logic(timestamp, label)

    # === State Manager Delegations ===
    def _recover_warmup_from_pg(self):
        return self.state_manager._recover_warmup_from_pg()

    def _recover_warmup_from_sqlite(self):
        return self.state_manager._recover_warmup_from_sqlite()

    def _ensure_consumer_group(self):
        """确保所需的 Redis 消费者组存在"""
        from config import STREAM_FUSED_MARKET, IS_SIMULATED
        
        # 🚀 [核心修复] OMS 必须优先寻找自己的专有组和专有流，防止被通用的 GROUP_FEATURE 带偏
        from config import GROUP_OMS, STREAM_ORCH_SIGNAL
        target_group = REDIS_CFG.get('oms_group') or REDIS_CFG.get('orch_group') or GROUP_OMS
        self._oms_group = target_group
        target_stream = REDIS_CFG.get('signal_stream') or STREAM_ORCH_SIGNAL
        
        streams_to_init = [target_stream]
        if self.mode != 'backtest' or IS_SIMULATED:
            streams_to_init.append(STREAM_FUSED_MARKET)

        for s in streams_to_init:
            try:
                # [🛡️ Idempotent Group Check] 检查组是否已存在，避免频繁 Destroy 重建导致流位点复位
                group_exists = False
                try:
                    infos = self.r.xinfo_groups(s)
                    if any(g[b'name'].decode('utf-8') == target_group for g in infos):
                        group_exists = True
                except: pass

                if not group_exists:
                    group_id = '0' if IS_SIMULATED else '$'
                    self.r.xgroup_create(s, target_group, mkstream=True, id=group_id)
                    db_idx = self.r.connection_pool.connection_kwargs.get('db')
                    logger.info(f"✅ [OMS] Created group {target_group} on DB {db_idx} stream {s} with ID {group_id}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e): 
                    logger.error(f"Group Create Error on {s}: {e}")

    # 在 SystemOrchestratorV8 类中添加以下方法

    async def _evaluate_shadow_signals(self, payload: dict):
        """
        🚀 [Shadow Logic] 基于融合包内的 Alpha 直接生成交易指令
        仅在 PURE_ALPHA_REPLAY 模式下激活。
        """
        ts = payload.get('ts')
        symbols = payload.get('symbols', [])
        alphas = payload.get('precalc_alpha', [])
        stock_prices = payload.get('stock_price', [])
        
        for i, sym in enumerate(symbols):
            alpha = float(alphas[i])
            s_price = float(stock_prices[i])
            st = self.states.get(sym)
            if not st: continue
            
            # 1. 出场评估 (如有持仓)
            if st.position != 0:
                # 逻辑出场：如果 Alpha 强度反转或彻底消失 (与 S4 保持一致)
                should_exit = False
                if st.position == 1 and alpha < 0.2: # CALL 出场
                    should_exit = True
                elif st.position == -1 and alpha > -0.2: # PUT 出场
                    should_exit = True
                
                # 特殊出场：Alpha 方向性完全调转 (FLIP)
                if (st.position == 1 and alpha < -0.6) or (st.position == -1 and alpha > 0.6):
                    should_exit = True
                
                if should_exit:
                    exit_payload = {
                        'symbol': sym,
                        'action': 'SELL',
                        'sig': {'reason': 'SHADOW_EXIT', 'alpha': alpha},
                        'stock_price': s_price,
                        'ts': ts
                    }
                    await self._handle_trade_signal(exit_payload, allow_delay_queue=False)
            
            # 2. 进场评估 (如为空仓且 Alpha 足够强)
            else:
                if abs(alpha) > 0.8:
                    action = 'BUY'
                    side = 'CALL' if alpha > 0 else 'PUT'
                    entry_payload = {
                        'symbol': sym,
                        'action': 'BUY',
                        'sig': {
                            'reason': f'SHADOW_ENTRY_{side}',
                            'alpha': alpha,
                            'target_side': 1 if side == 'CALL' else -1
                        },
                        'stock_price': s_price,
                        'ts': ts
                    }
                    await self._handle_trade_signal(entry_payload, allow_delay_queue=False)

    async def _broadcast_state_to_redis(self):
        """将当前 OMS 的真实持仓快照发布到 Redis，供 Signal Engine 对齐"""
        if getattr(self, 'use_shared_mem', False):
            return # 🚀 共享内存模式下，对象修改瞬间双端可见，禁止网络广播！
        import json
        active_states = {}
        
        for sym, st in self.states.items():
            # 🚀 [核心修复] 如果 sym 是 bytes (来自 Redis)，必须转换为 str 否则 json.dumps 会触发 TypeError
            if isinstance(sym, bytes):
                sym = sym.decode('utf-8')
                
            # 只要 OMS 确认持仓，或者正在处理订单，就广播出去
            if st.position != 0 or st.is_pending:
                active_states[sym] = json.dumps({
                    'pos': st.position,
                    'qty': getattr(st, 'qty', 0),
                    'price': st.entry_price,
                    'stock': st.entry_stock,
                    'ts': st.entry_ts,
                    'max_roi': getattr(st, 'max_roi', 0.0),
                    'is_pending': st.is_pending,
                    # 👇 [核心修复] 必须补齐 SE 退出评估依赖的策略元数据
                    'entry_spy_roc': getattr(st, 'entry_spy_roc', 0.0),
                    'entry_index_trend': getattr(st, 'entry_index_trend', 0),
                    'entry_alpha_z': getattr(st, 'entry_alpha_z', 0.0),
                    'entry_iv': getattr(st, 'entry_iv', getattr(st, 'last_valid_iv', 0.0))
                })
        
        # 👑 同时同步资金池状态
        active_states['____SYSTEM_CASH____'] = json.dumps({
            'cash': self.mock_cash,
            'ts': time.time()
        })
                
        pipe = self.r.pipeline()
        pipe.delete("oms:live_positions")  # 先清空全量
        if active_states:
            pipe.hset("oms:live_positions", mapping=active_states)
        pipe.execute()

    async def _handle_trade_signal(self, payload: dict, allow_delay_queue: bool = True):
        source = payload.get('source')
        curr_ts = payload.get('ts')
        curr_ts_float = float(curr_ts) if curr_ts is not None else None
        
        # 🚀 [Fused Replay Protocol] 下一代高速对齐协议支持
        if source == 'fused_replay_v8':
            if curr_ts_float is not None:
                # 1. 价格对齐：从 Fused 包同步当前瞬间的 Mid 价格
                symbols = payload.get('symbols', [])
                for i, sym in enumerate(symbols):
                    if sym in self.states:
                        # 注入正股价格与 Alpha 到状态机
                        self.states[sym].last_stock_price = payload['stock_price'][i]
                        # 更新期权基准价 (s5 发送的是 Mid 基准)
                        # 注意：s5 并不直接发送期权价格，而是通过 MockIBKR 记录
                        # 我们这里手动更新 st.last_opt_price 以供止损评估使用
                        # 止损评估需要的是期权当前 Mid
                        pass
                
                # 2. 影子信号评估 (Shadow Logic)
                # 如果开启了 PURE_ALPHA_REPLAY，我们直接在此处通过 Alpha 生成买卖决策
                pure_alpha_mode = os.environ.get('PURE_ALPHA_REPLAY') == '1'
                if pure_alpha_mode:
                    await self._evaluate_shadow_signals(payload)
                
                # 3. 🚨 [CRITICAL] 释放屏障：无论是否有交易，必须通知发球机本帧已处理完毕
                self.r.set(f"sync:orch_done:{int(curr_ts_float)}", "1")
                self.r.expire(f"sync:orch_done:{int(curr_ts_float)}", 60)
            return

        action = payload.get('action')

        if curr_ts_float is not None:
            self.last_curr_ts = curr_ts_float
            await self._flush_delayed_signals(curr_ts_float)

        if allow_delay_queue and action in ('BUY', 'SELL') and self._should_delay_signal(action):
            await self._queue_delayed_signal(payload)
            return
        
        # 👇 拦截 SE 发来的同步锁信号
        if action == 'SYNC':
            if curr_ts:
                # 🛑 [核心修复 3: 同步最新价格，防止 OMS 强平时用 0 元结算导致暴亏]
                latest_prices = payload.get('prices', {})
                for sym, price in latest_prices.items():
                    if sym in self.states:
                        self.states[sym].last_opt_price = price
                
                # ✅ SYNC 由 OMS 侧确认完成后再释放 orch 屏障，避免 SE 提前 ACK 造成时序漂移
                try:
                    self.r.set("sync:orch_done", str(curr_ts))
                    self.r.expire("sync:orch_done", 120)
                    frame_id = payload.get('frame_id')
                    if frame_id:
                        self.r.set("sync:orch_done_frame_id", str(frame_id))
                        self.r.expire("sync:orch_done_frame_id", 120)
                    self.r.hincrby("diag:ee:counters", "sync_ack", 1)
                except Exception as e:
                    logger.warning(f"⚠️ [OMS-SYNC] failed to set orch_done: {e}")
            return
            
        sym = payload.get('symbol')
        if not sym:
            return

        sig = payload.get('sig')
        stock_price = payload.get('stock_price', 0.0)

        batch_idx = payload.get('batch_idx', -1)

        if not sym or sym not in self.states: return
        st = self.states[sym]
        
        logger.info(f"📥 [OMS] Received {action} signal for {sym}: {sig.get('reason', '')}")
        
        if action in ('BUY', 'SELL'):
            # 🚀 [核心修复: 全生命周期锁保护]
            # 我们将检查、加锁、执行和解锁全部纳入 try...finally 闭环
            try:
                # 1. 拦截正在处理中的订单 [Ghost C]
                if st.is_pending and not getattr(self, 'use_shared_mem', False):
                    # 如果当前已经锁死超过 60 秒，强制解锁 (自救机制)
                    pending_duration = time.time() - getattr(st, 'pending_ts', 0)
                    if pending_duration > 60:
                        logger.warning(f"⚠️ [{sym}] 检出长延时 Pending ({pending_duration:.1f}s)，强制解锁！")
                        st.is_pending = False
                    else:
                        logger.warning(f"🛡️ [{sym}] 信号重叠拦截: {action} (当前 Pending: {st.pending_action or st.pending_side})")
                        return

                # 2. 同步加锁
                st.is_pending = True
                st.pending_action = action
                st.pending_ts = time.time()
                st.pending_side = action

                if action == 'BUY':
                    if st.position != 0:
                        logger.warning(f"🚫 [OMS] {sym} already has position! Ignoring BUY signal.")
                        return
                    # 3. 阻塞执行
                    await self._execute_entry(sym, sig, stock_price, curr_ts, batch_idx)
                else: # SELL
                    # 🚀 [修复 1] 共享内存下，SE 为了光速复用资金已提前将 position 清 0，OMS 必须无条件放行平仓单！
                    if st.position == 0 and not getattr(self, 'use_shared_mem', False):
                        logger.warning(f"🚫 [OMS] {sym} has no position! Ignoring SELL signal.")
                        return
                    # 3. 阻塞执行
                    await self._execute_exit(sym, sig, stock_price, curr_ts, batch_idx)
            except Exception as e:
                logger.error(f"❌ [OMS] Error executing {action} for {sym}: {e}")
            finally:
                # 🛡️ 最终防线：无论成交与否，只要完成了本轮逻辑，必须释放标志位
                st.is_pending = False
                st.pending_action = None
                st.pending_side = None
        
        # 👑 处理完毕后 (仅限交易信号)，全网广播最新的真实持仓状态！
        await self._broadcast_state_to_redis()

    async def process_trade_signal(self, payload: dict):
        await self._handle_trade_signal(payload, allow_delay_queue=True)

    async def run(self):
        """
        [V8 终极执行引擎主循环]
        采用非阻塞异步 Redis 流消费，完美兼容 LiveReplay 发球机架构与实盘。
        """
        from config import REDIS_CFG, STREAM_ORCH_SIGNAL, STREAM_FUSED_MARKET, IS_LIVEREPLAY, SYNC_EXECUTION, TRADING_ENABLED
        import os
        import traceback
        import asyncio
        import redis
        from utils import serialization_utils as ser
        
        logger.info(f"🔥 Execution Engine (OMS) Started (DB: {self.r.connection_pool.connection_kwargs.get('db')})")
        
        # 1. 确保消费组存在
        self._ensure_consumer_group()
        
        # 2. 启动后台监控任务 (非回放模式或启用同步时)
        if not IS_LIVEREPLAY and not SYNC_EXECUTION:
            asyncio.create_task(self._pnl_monitor_loop())
            if TRADING_ENABLED:
                asyncio.create_task(self.reconciler.run_reconciliation_loop())
        
        target_group = getattr(self, '_oms_group', REDIS_CFG.get('oms_group') or REDIS_CFG.get('orch_group') or GROUP_OMS)
        consumer_name = f"oms_consumer_{os.getpid()}"
        
        # 我们需要同时监听信号流(指令)和行情流(兜底/状态同步)
        streams = {STREAM_ORCH_SIGNAL: '>', STREAM_FUSED_MARKET: '>'}
        last_stats_log = time.time()
        stats = {'msgs': 0, 'signal': 0, 'fused': 0, 'acks': 0}
        
        while True:
            try:
                # 3. 优先消费未确认的历史消息 (Pending)
                pending = self.r.xreadgroup(
                    groupname=target_group,
                    consumername=consumer_name,
                    streams={STREAM_ORCH_SIGNAL: '0', STREAM_FUSED_MARKET: '0'},
                    count=100
                )
                
                if pending:
                    messages = pending
                    logger.debug(f"🔍 [OMS] Found {len(pending)} pending messages in streams.")
                else:
                    messages = self.r.xreadgroup(
                        groupname=target_group,
                        consumername=consumer_name,
                        streams=streams,
                        count=10,
                        block=1 
                    )

                if not messages:
                    await asyncio.sleep(0.001)
                    continue

                # 5. 处理消息
                for stream, msgs in messages:
                    stats['msgs'] += len(msgs)
                    stream_str = stream.decode('utf-8') if isinstance(stream, bytes) else stream
                    for msg_id, data in msgs:
                        # 🚀 [Debug] 捕获所有流量
                        logger.info(f"📥 [OMS] Received msg {msg_id} from {stream_str}")
                        try:
                            raw_data = data.get(b'data') or data.get(b'pickle') or data.get(b'batch') or b''
                            if raw_data == b'DONE':
                                logger.info(f"🏁 [OMS] Received DONE on {stream_str}")
                                self.r.xack(stream_str, target_group, msg_id)
                                continue
                                
                            payload = ser.unpack(raw_data)
                            if payload:
                                payload = self._deep_decode_bytes(payload)
                                if stream_str == STREAM_ORCH_SIGNAL:
                                    # 处理交易信号 (包含 fused_replay 协议)
                                    logger.info(f"⚡ [OMS] Processing trade/fused signal (ts={payload.get('ts')})")
                                    await self.process_trade_signal(payload)
                                    stats['signal'] += 1
                                elif stream_str == STREAM_FUSED_MARKET:
                                    if self.ibkr and hasattr(self.ibkr, 'record_market_data'):
                                        market_packet = self._reconstruct_market_packet(payload)
                                        if market_packet:
                                            self.ibkr.record_market_data(market_packet)
                                    stats['fused'] += 1
                            
                            try:
                                self.r.xack(stream_str, target_group, msg_id)
                                stats['acks'] += 1
                                
                                # [Parity Barrier] 如果开启了回放模式，处理完后报备锁
                                ts_val = payload.get('ts') if isinstance(payload, dict) else None
                                if IS_LIVEREPLAY and ts_val:
                                    logger.info(f"🔓 [OMS] Releasing barrier for ts={ts_val}")
                                    self._mark_exec_done(ts_val)
                                elif IS_LIVEREPLAY:
                                    logger.warning(f"⚠️ [OMS] IS_LIVEREPLAY active but ts missing in payload from {stream_str}")
                            except Exception as ack_err:
                                logger.error(f"ACK/Sync Error: {ack_err}")
                                
                        except Exception as msg_err:
                            logger.error(f"OMS Error processing msg_id {msg_id}: {msg_err}")
                            self.r.xack(stream_str, target_group, msg_id) # 即使失败也确认，防止死转
                            stats['acks'] += 1
                            
                await asyncio.sleep(0.001)
                if time.time() - last_stats_log >= 60:
                    logger.info(
                        f"📊 [OMS-Stats] 60s msgs={stats['msgs']} signal={stats['signal']} fused={stats['fused']} acked={stats['acks']}"
                    )
                    stats = {'msgs': 0, 'signal': 0, 'fused': 0, 'acks': 0}
                    last_stats_log = time.time()
                
            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    self._ensure_consumer_group()
                else:
                    logger.error(f"Redis Stream Error: {e}")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"OMS Fatal Loop Error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(1)


if __name__ == "__main__":
    pass
