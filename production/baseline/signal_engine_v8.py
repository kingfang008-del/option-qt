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

import redis
import torch

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

# [🆕 新增] 动态模型目录注入 (对齐 S4 环境)
import sys, os
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
if MODEL_DIR not in sys.path: sys.path.insert(0, MODEL_DIR)

# 引入纯策略核心
from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION, StrategyCore, StrategyConfig

# [Refactor] 引入模块化执行组件
from orchestrator_state_manager import OrchestratorStateManager
from orchestrator_accounting import OrchestratorAccounting
from orchestrator_execution import OrchestratorExecution
from orchestrator_reconciler import OrchestratorReconciler

from utils import serialization_utils as ser

# 动态引入组件
try:
    from mock_ibkr_historical import MockIBKRHistorical
except ImportError:
    MockIBKRHistorical = None

try:
    from ibkr_connector_v8 import IBKRConnectorFinal
except ImportError:
    IBKRConnectorFinal = None

# Log Stream Key
from config import (
    STREAM_TRADE_LOG,           # [Fix] Use shared config
    SYNC_EXECUTION,
    STREAM_FUSED_MARKET,        # [New] Fast Tick Stream for exits
    TRADING_ENABLED,            # 全局交易开关 (True=实盘下单, False=只读模式)
    MAX_POSITIONS,              # 最大同时持仓数
    POSITION_RATIO,             # 单标的最大仓位比例
    MAX_TRADE_CAP,              # 单笔交易最大金额
    GLOBAL_EXPOSURE_LIMIT,      # 全局风险敞口上限
    COMMISSION_PER_CONTRACT,    # 期权手续费 ($/手)
    USE_BID_ASK_PRICING,        # [New] 价格模式开关
    NON_TRADABLE_SYMBOLS,
    ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS,
    INDEX_TREND_SYMBOLS,
    IS_LIVEREPLAY,
    IS_BACKTEST,
    IS_SIMULATED
)

from trading_tft_stock_embed import AdvancedAlphaNet
#from train_fast_channel_microstructure import FastMicrostructureModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s')
logger = logging.getLogger("V8_SignalEngine")
PURE_ALPHA_REPLAY = os.environ.get('PURE_ALPHA_REPLAY') == '1'

# [Fix] 显式添加 FileHandler 确保写入文件
from config import LOG_DIR
file_handler = logging.FileHandler(LOG_DIR / "SignalEngine.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

 
# Log Stream Key
from config import (
    REDIS_CFG,                  # [🔥 新增] 统一导入 Redis 配置
    STREAM_TRADE_LOG,           
    STREAM_FUSED_MARKET,        
    TRADING_ENABLED)


RISK_FREE_RATE = 0.045

# [联动] 实盘平仓防滑点开关: TRADING_ENABLED=True → LMT (防做市商收割)
#                           TRADING_ENABLED=False → MKT (极速回测)
EXIT_ORDER_TYPE = 'LMT' if TRADING_ENABLED else 'MKT'

class SymbolState:
    def __init__(self, symbol, config: StrategyConfig = None):
        self.symbol = symbol
        self.cfg = config if config else StrategyConfig()
        self.prices = deque(maxlen=60)
        self.last_tick_price = None  # 🚀 [NEW] 锁存上一秒股价 (用于分钟对位)
        self.last_tick_opt_data = None # 🚀 [NEW] 锁存上一秒期权快照 (用于分钟对位)
        
        # 止损/PnL 辅助
        self.alpha_history = deque(maxlen=self.cfg.ROLLING_WINDOW_MINS + 10)
        self.pct_history = deque(maxlen=self.cfg.ROLLING_WINDOW_MINS + 10)
        self.correction_mode = "NORMAL"
        self.locked_cash = 0.0
        # MACD State
        self.ema_fast_val = None; self.ema_slow_val = None; self.dea_val = None
        self.k_fast = 2 / (8 + 1); self.k_slow = 2 / (21 + 1); self.k_sig  = 2 / (5 + 1)
        
        # Position State
        self.position = 0      # 0, 1(Call), -1(Put)
        self.qty = 0
        self.entry_price = 0.0 
        self.entry_stock = 0.0 
        self.entry_ts = 0.0
        self.entry_spy_roc = 0.0 

        self.entry_index_trend = 0  # <--- [新增] 初始化大盘趋势

        self.entry_alpha_z = 0.0 # [NEW]
        self.entry_iv = 0.0      # [NEW]
        self.last_alpha_z = 0.0  # [NEW]
        self.prev_alpha_z = 0.0  # [NEW]
        self.max_roi = -1.0
        self.cooldown_until = 0.0
        self.contract_id = None
        self.latest_call_id = "" # [NEW] 缓存今日最近一次出现过的真实 Call 合约 ID
        self.latest_put_id = ""  # [NEW] 缓存今日最近一次出现过的真实 Put 合约 ID
        self.is_pending = False

        # [🔥 新增] 记录上一分钟明 MACD 柱子，用于计算斜率(导数)
        self.prev_macd_hist = 0.0
        
        # 期权元数据
        self.strike_price = 0.0
        self.expiry_date = None 
        self.last_valid_iv = 0.5
        self.opt_type = 'call'

        # [新增] 预热完成标志
        self.warmup_complete = False
        
        # [🔥 核心新增] 流动性趋势追踪
        self.last_spread_pct = 0.0
        self.last_snap_roc = 0.0

        # [核心新增] 高频基础价格
        self.last_price = 0.0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.cached_min_roc = 0.0
        self.cached_macd_hist = 0.0
        self.cached_macd_hist_slope = 0.0
        self.last_vol_z = 0.0
        self.last_min_ts = 0  # [Fix A] 记录上一次分钟指标更新的分钟时间戳
        self.last_seen_ts = 0.0
        self.pending_exit_reason = ""
        self.pending_exit_count = 0
        self.pending_exit_first_ts = 0.0

    def get_reversal_count(self, window_mins=30, threshold=0.001):
        """[Market Regime Guard] 计算过去 N 分钟内价格反转(洗盘)的频率"""
        if len(self.prices) < 2: return 0
        
        prices_list = list(self.prices)[-window_mins:]
        if len(prices_list) < 2: return 0
        
        reversals = 0
        last_dir = 0 # 1: up, -1: down
        
        for i in range(1, len(prices_list)):
            diff_pct = (prices_list[i] - prices_list[i-1]) / prices_list[i-1]
            if abs(diff_pct) >= threshold:
                curr_dir = 1 if diff_pct > 0 else -1
                if last_dir != 0 and curr_dir != last_dir:
                    reversals += 1
                last_dir = curr_dir
        return reversals

    def update_tick_state(self, price, bid=None, ask=None):
        """[高频轨道] 每秒更新一次，仅用于止损判定与 PnL 计算"""
        self.last_price = float(price)
        if bid is not None: self.bid_price = float(bid)
        if ask is not None: self.ask_price = float(ask)
        # 🚨 [CRITICAL BUG FIX] 不要在这里更新 self.prices[-1]！
        # 否则分钟级的 update_indicators 会因为 prices[-1] 已经变成现价而导致 ROC 永远为 0。

    def update_indicators(self, price, raw_alpha_val, ts=None, use_precalc_feed=False):
        """更新分钟级指标 (MACD, Alpha History, ROC)"""
        # [Fix A] 强制分钟级幂等性，防止 1s 驱动重复追加
        if ts is not None:
            curr_min = int(ts / 60)
            if curr_min <= self.last_min_ts:
                return
            self.last_min_ts = curr_min
            
        price = float(price)
        raw_alpha_val = float(raw_alpha_val)
        if price <= 0: return

        # 1. 价格 Buffer
        self.prices.append(price)
        
        # 2. Alpha/Pct History (用于计算相关性/归一化)
        # 🚨 [关键修复] 为了对齐 corrcoef，两个 Buffer 必须保持同步追加！
        if len(self.prices) >= 2:
            self.alpha_history.append(raw_alpha_val)
            self.pct_history.append((price - self.prices[-2]) / self.prices[-2])
            self.cached_min_roc = self.pct_history[-1]
        else:
            self.cached_min_roc = 0.0
        
        # 3. MACD 计算 (轻量级 EWMA)
        if self.ema_fast_val is None:
            self.ema_fast_val = float(price)
            self.ema_slow_val = float(price)
            self.dea_val = 0.0
        else:
            self.ema_fast_val = float(price * self.k_fast + self.ema_fast_val * (1 - self.k_fast))
            self.ema_slow_val = float(price * self.k_slow + self.ema_slow_val * (1 - self.k_slow))
            dif = self.ema_fast_val - self.ema_slow_val
            self.dea_val = float(dif * self.k_sig + self.dea_val * (1 - self.k_sig))
            
        macd_hist = float((self.ema_fast_val - self.ema_slow_val) - self.dea_val)
        macd_hist_slope = macd_hist - self.prev_macd_hist
        self.prev_macd_hist = macd_hist
        
        # 缓存结果供全分钟共用
        self.cached_macd_hist = macd_hist
        self.cached_macd_hist_slope = macd_hist_slope
        
        # 4. Alpha 修正模式
        # [Fix] 如果是预计算模式 (use_precalc_feed)，不需要 30 分钟预热
        if len(self.alpha_history) >= self.cfg.ROLLING_WINDOW_MINS or use_precalc_feed:
            self.warmup_complete = True
            if len(self.alpha_history) >= self.cfg.ROLLING_WINDOW_MINS:
                alphas = list(self.alpha_history)[-self.cfg.ROLLING_WINDOW_MINS:]
                pcts = list(self.pct_history)[-self.cfg.ROLLING_WINDOW_MINS:]
                if np.std(alphas) > 1e-6 and np.std(pcts) > 1e-6:
                    corr = np.corrcoef(alphas, pcts)[0, 1]
                    if self.correction_mode == "NORMAL" and corr < (self.cfg.CORR_THRESHOLD - 0.05):
                        self.correction_mode = "INVERT"
                    elif self.correction_mode == "INVERT" and corr > (self.cfg.CORR_THRESHOLD + 0.05):
                        self.correction_mode = "NORMAL"
        else:
            self.warmup_complete = False

    def get_strategy_metrics(self):
        """[策略接口] 获取稳定的分钟级指标，用于计算买入信号"""
        macd_hist = float(getattr(self, 'cached_macd_hist', 0.0))
        macd_hist_slope = float(getattr(self, 'cached_macd_hist_slope', 0.0))
        snap_roc = float(getattr(self, 'cached_min_roc', 0.0))
        
        roc_5m = 0.0
        if len(self.prices) >= 6:
            prev_5m = self.prices[-6]
            if prev_5m > 0: 
                roc_5m = (self.prices[-1] - prev_5m) / prev_5m

        return roc_5m, macd_hist, macd_hist_slope, snap_roc

        
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
            ,
            'last_seen_ts': self.last_seen_ts,
            'pending_exit_reason': self.pending_exit_reason,
            'pending_exit_count': self.pending_exit_count,
            'pending_exit_first_ts': self.pending_exit_first_ts,
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
        self.last_seen_ts = data.get('last_seen_ts', 0.0)
        self.pending_exit_reason = data.get('pending_exit_reason', "")
        self.pending_exit_count = data.get('pending_exit_count', 0)
        self.pending_exit_first_ts = data.get('pending_exit_first_ts', 0.0)

class SignalEngineV8:
    def __init__(self, symbols, mode='realtime', config_paths=None, model_paths=None):
        print(f"DEBUG: V8Orchestrator Initializing... Mode={mode}")
        self.mode = mode
        self.symbols = symbols
        self.strategy = StrategyCore(StrategyConfig())
        self.cfg = self.strategy.cfg
        logger.info(f"🧭 Active strategy core: {ACTIVE_STRATEGY_CORE_VERSION}")
        self.states = {s: SymbolState(s, config=self.cfg) for s in symbols}
        self.symbol_states = self.states # Alias
        
        # [Refactor] 模块化组件初始化
        self.state_manager = OrchestratorStateManager(self)
        
        # [NEW] 允许 SE 复用 Accounting 逻辑进行 Alpha Logging (防止重复定义)
        from orchestrator_accounting import OrchestratorAccounting
        self.accounting = OrchestratorAccounting(self)
        
        # [OMS handles Execution]
        
        # Redis Init
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print("DEBUG: Redis Initialized.")

        # Global State Defaults
        # [已统一] 所有的策略、风控参数通过 self.cfg 访问
        self.last_date = None
        self.mock_cash = self.cfg.INITIAL_ACCOUNT
        self.index_opening_prices = {}
        self.consecutive_stop_losses = 0
        self.global_cooldown_until = 0
        
        # 兼容性重定向 (指向统一的 config 对象)
        self.CIRCUIT_BREAKER_THRESHOLD = self.cfg.CIRCUIT_BREAKER_THRESHOLD
        self.CIRCUIT_BREAKER_MINUTES = self.cfg.CIRCUIT_BREAKER_MINUTES
        self.MIN_OPTION_PRICE = self.cfg.MIN_OPTION_PRICE
        
        self.last_save_time = 0

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
        self.total_commission = 0.0      # 累计手续费
        self.trade_count = 0             # 总交易笔数
        self.win_count = 0               # 盈利笔数
        self.loss_count = 0              # 亏损笔数

        # [NEW] 每日交易数据池 (用于盘后分析)
        self.daily_trades = []
        self.last_index_trend = 0 
        self.spy_ema_roc = 0.0     # [NEW] 5min EMA ROC (势能平滑)
        self.qqq_ema_roc = 0.0
        self.last_spy_roc_val = 0.0
        self.last_qqq_roc_val = 0.0

        # =========================================================
        # 🚀 [新增] 动态 Alpha 归一化追踪器 (Dynamic Alpha Tracker)
        # =========================================================
        self.dynamic_alpha_mean = 0.0
        self.dynamic_alpha_std = 1.0
        self.alpha_count = 0
        # 赋予一个先验初始值 (防止冷启动期间的极端缩放)
        self.dynamic_vol_mean = 0.0739 
        self.dynamic_vol_std = 0.1106

        # 👇 零干扰优化：初始化缓存与高频识别
        self.is_high_freq = (self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY')
        self.cached_alphas = {}
        self.cached_event_probs = {} # [NEW] 存储事件爆发概率
        self.cached_vol_z = {}
        self.sym_vol_mean = {}
        self.sym_vol_var = {}
        self.sym_last_vol_price = {}
        self.processed_frame_ids = deque(maxlen=50000)
        self.processed_frame_set = set()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Device: {self.device}")

        # [Fix] 提前将模型属性初始化为 None
        # 若 _load_models 因路径缺失或 import 失败未执行，process_batch 可以给出明确报错而非 AttributeError
        self.slow_model = None
        self.slow_cfg = None
        self.fast_cfg = None
        self.slow_stock_indices = []
        self.slow_option_indices = []

        # 1. State Management (Load ONCE)
        self._load_state() 
        print(f"DEBUG: State Loaded. Symbols: {list(self.states.keys())}")
        
        # [🔥] 盈亏报告时间追踪
        self.last_pnl_report_ts = 0.0
        
        # 2. Models (Load ONCE if paths provided)
        if config_paths and model_paths:
            try:
                self._load_models(config_paths, model_paths)
                print("DEBUG: Models Loaded.")
            except Exception as e:
                logger.error(f"❌ Model Load Failed: {e}")
        
        # 3. Mode-Specific Init
        logger.info("🚀 V8 Signal Engine Init (No Execution)...")
        self.ibkr = None
        
        # [Shadow Validation] Alpha Only Mode
        from config import ONLY_LOG_ALPHA
        self.only_log_alpha = ONLY_LOG_ALPHA
        if self.only_log_alpha:
            logger.info("🕵️ Shadow System: ONLY_LOG_ALPHA mode enabled. Trading is strictly disabled.")

        # [🆕 新增] 信号审计日志 (Alpha Audit Log) - 仅在回测模式开启
        self.audit_log_path = LOG_DIR / "alpha_audit.csv"
        if IS_BACKTEST:
            with open(self.audit_log_path, 'w', encoding='utf-8') as f:
                f.write("timestamp,symbol,price,alpha,cs_alpha_z,event_prob,vol_z,roc_5m\n")
            logger.info(f"📊 [Audit] Alpha Audit logging enabled: {self.audit_log_path}")

        # [NEW] 模式启动声明
        mode_str = "✅ [Bid/Ask + BSM 校准模式]" if USE_BID_ASK_PRICING else "⚠️ [成交价模式 (Transaction Price Mode)]"
        logger.info(f"📊 Orchestrator 启动! 当前价格计算模式: {mode_str}")

    def _is_high_freq_tick(self, st, curr_ts):
        prev_ts = float(getattr(st, 'last_seen_ts', 0.0) or 0.0)
        st.last_seen_ts = float(curr_ts)
        if prev_ts <= 0:
            return False
        dt_sec = float(curr_ts) - prev_ts
        return 0.0 < dt_sec <= 2.0

    def _should_confirm_exit(self, reason):
        prefixes = getattr(self.cfg, 'EXIT_CONFIRM_REASON_PREFIXES', ())
        return any(str(reason).startswith(prefix) for prefix in prefixes)

    def _confirm_high_freq_exit(self, st, exit_sig, curr_ts):
        reason = str(exit_sig.get('reason', ''))
        if not self._should_confirm_exit(reason):
            st.pending_exit_reason = ""
            st.pending_exit_count = 0
            st.pending_exit_first_ts = 0.0
            return exit_sig

        required = max(1, int(getattr(self.cfg, 'EXIT_CONFIRM_SECONDS_1S', 1)))
        if st.pending_exit_reason == reason:
            st.pending_exit_count += 1
        else:
            st.pending_exit_reason = reason
            st.pending_exit_count = 1
            st.pending_exit_first_ts = float(curr_ts)

        if st.pending_exit_count < required:
            logger.debug(f"⏳ [Exit Confirm] {st.symbol} waiting {st.pending_exit_count}/{required} for {reason}")
            return None

        exit_sig['reason'] = f"{reason}|CONFIRM_{st.pending_exit_count}s"
        st.pending_exit_reason = ""
        st.pending_exit_count = 0
        st.pending_exit_first_ts = 0.0
        return exit_sig

    def _reset_exit_confirmation(self, st):
        st.pending_exit_reason = ""
        st.pending_exit_count = 0
        st.pending_exit_first_ts = 0.0

    def _load_state(self):
        return self.state_manager._load_state()

    def _publish_warmup_status(self):
        return self.state_manager._publish_warmup_status()

    def _load_state_from_db(self):
        return self.state_manager._load_state_from_db()

    def save_state(self):
        return self.state_manager.save_state()

    def _get_pg_conn(self):
        return self.state_manager._get_pg_conn()

    def _init_state_db(self):
        return self.state_manager._init_state_db()

    def _save_state_to_db(self, state_data):
        return self.state_manager._save_state_to_db(state_data)

    def _ensure_consumer_group(self):
        streams_to_init = [REDIS_CFG['input_stream']]
        if self.mode != 'backtest' or IS_SIMULATED:
            streams_to_init.append(STREAM_FUSED_MARKET)
            
        for s in streams_to_init:
            try:
                # [Fix] 回放模式必须从 '0' 开始，否则会错过发球机先发出的消息
                group_id = '0' if IS_SIMULATED else '$'
                group_name = REDIS_CFG.get('orch_group') # 优先使用驱动注入的 group
                self.r.xgroup_create(s, group_name, mkstream=True, id=group_id)
                logger.info(f"✅ Created consumer group {group_name} for stream {s} with ID {group_id}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e): 
                    logger.error(f"Group Create Error on {s}: {e}")

    def load_db_info(self):
        try:
            conn = psycopg2.connect(PG_DB_URL)
            c = conn.cursor()
            c.execute("SELECT id, sector_id FROM stocks_us")
            rows = c.fetchall()
            conn.close()
            max_sid = max([r[0] for r in rows]) if rows else 15000
            sec_set = set([r[1] for r in rows if r[1]])
            return {'max_stock_id': max_sid + 100, 'max_sector_id': len(sec_set) + 10}
    
        except Exception as e:
            logger.error(f"Error loading meta info from PG: {e}")
            return {'max_stock_id': 18000, 'max_sector_id': 200}


    def _load_models(self, config_paths, model_paths):
        logger.info("🧠 Loading Models...")
        if not AdvancedAlphaNet: return

        # load_db_info 现在返回 dict {'max_stock_id':..., 'max_sector_id':...}
        db_info = self.load_db_info()
        caps = {
            'stock': db_info['max_stock_id'],
            'sector': db_info['max_sector_id'],
            'dow': 7,
        }

        if 'slow' not in model_paths:
            logger.warning("   ⚠️ No slow model path provided, skipping.")
            return
        slow_path = Path(model_paths['slow'])
        if not slow_path.exists():
            raise FileNotFoundError(f"❌ Slow model checkpoint not found: {slow_path}")

        # 先读 checkpoint，从 embedding shape 更正 caps（权重是唯一权威来源）
        st = torch.load(slow_path, map_location=self.device, weights_only=False)
        ckpt_state = st.get('state_dict', st)
        if 'static_stock_embed.weight' in ckpt_state:
            # AdvancedAlphaNet 里是 Embedding(caps['stock']+1)，所以传 shape-1
            caps['stock'] = ckpt_state['static_stock_embed.weight'].shape[0] - 1
        if 'static_sector_embed.weight' in ckpt_state:
            caps['sector'] = ckpt_state['static_sector_embed.weight'].shape[0] - 1
        if 'fast' in config_paths:
            with open(config_paths['fast']) as f: self.fast_cfg = json.load(f)

        with open(config_paths['slow']) as f: self.slow_cfg = json.load(f)
        self.slow_model = AdvancedAlphaNet(self.slow_cfg, caps).to(self.device).eval()

        # 手动过滤：只加载形状完全匹配的参数，跳过特征数不同的层，防止 RuntimeError
        model_state = self.slow_model.state_dict()
        compatible, skipped = {}, []
        for k, v in ckpt_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(f"{k}: ckpt={tuple(v.shape)} != model={tuple(model_state[k].shape) if k in model_state else 'missing'}")
        self.slow_model.load_state_dict(compatible, strict=False)
        if skipped:
            logger.warning(f"   ⚠️ {len(skipped)} layers skipped (shape mismatch, check slow_feature.json):")
            for s in skipped[:5]: logger.warning(f"      {s}")
        logger.info(f"   ✅ Slow Model Loaded ({len(compatible)}/{len(ckpt_state)} layers matched)")

        # [🔥 核心关键修复] 针对旧版 Payload 的 Index-Based 兼容逻辑
        # 只要 Payload 还没切换到 features_dict，Orchestrator 就依赖以下索引进行切片
        EXCLUDE_FEATS = {'open', 'high', 'low', 'close', 'volume', 'stock_id', 'sector_id', 'day_of_week'}
        self.slow_stock_indices = []
        self.slow_option_indices = []
        self.slow_stk_cat_indices = []
        self.slow_opt_cat_indices = []
        
        if self.slow_cfg and 'features' in self.slow_cfg:
            idx_stk, idx_opt = 0, 0
            for absolute_idx, f in enumerate(self.slow_cfg['features']):
                name = f['name']
                if name in EXCLUDE_FEATS: continue
                
                if name.startswith('options_'):
                    self.slow_option_indices.append(absolute_idx)
                    if f.get('type') == 'categorical': self.slow_opt_cat_indices.append(idx_opt)
                    idx_opt += 1
                else:
                    self.slow_stock_indices.append(absolute_idx)
                    if f.get('type') == 'categorical': self.slow_stk_cat_indices.append(idx_stk)
                    idx_stk += 1
                    
        logger.info(f"   📐 Feature Map: Stock={len(self.slow_stock_indices)}, Option={len(self.slow_option_indices)}, CatStk={len(self.slow_stk_cat_indices)}, CatOpt={len(self.slow_opt_cat_indices)}")

    def _calc_trading_minutes(self, start_ts, end_ts):
        """计算两个时间点之间的有效交易分钟数 (排除闭市时间)"""
        if end_ts < start_ts: return 0.0
        
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone('America/New_York'))
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone('America/New_York'))
        
        minutes = 0.0
        curr = start_dt
        ny_tz = timezone('America/New_York')
        while curr.date() <= end_dt.date():
            # 获取当日开收盘时间 (使用 localize 避免 DST 陷阱)
            market_open = ny_tz.localize(datetime.combine(curr.date(), dt_time(9, 30)))
            market_close = ny_tz.localize(datetime.combine(curr.date(), dt_time(16, 0)))
            
            # 排除周末
            if curr.weekday() >= 5: 
                curr += timedelta(days=1)
                curr = curr.replace(hour=9, minute=30)
                continue
                
            # 计算当日有效区间
            daily_start = max(curr, market_open)
            daily_end = min(end_dt, market_close)
            
            if daily_start < daily_end:
                diff = (daily_end - daily_start).total_seconds() / 60.0
                minutes += diff
            
            curr += timedelta(days=1)
            curr = curr.replace(hour=9, minute=30)
            
        return max(0.0, minutes)



    @staticmethod
    def _get_fair_market_price(base_price: float, bid: float, ask: float, prev_price: float = 0.0) -> float:
        """
        [NEW] 统一计算期权公允市价
        如果是真空切片 (Bid/Ask 均为 0)，且 Last Price 与前一秒偏差超过 10%，则沿用上一秒价格。
        """
        if not USE_BID_ASK_PRICING:
            return base_price

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
            # 只有在进入真空（无买卖单）时，才触发 10% 偏离保护
            if prev_price > 0.01 and market_price > 0.01:
                if abs(market_price - prev_price) / prev_price > 0.10:
                    market_price = prev_price
        
        return market_price

    async def _process_fast_fused_tick(self, payload: dict):
        """[高频通道] 处理来自 fused_market_stream 的混合 Tick，执行毫秒级平仓保护"""
        from config import HASH_OPTION_SNAPSHOT, TARGET_SYMBOLS
        is_live_replay = IS_LIVEREPLAY
        
        sym = payload.get('symbol')
        if not sym: return
        
        # [🔥 核心增强] 在 LIVEREPLAY 模式下，将回放的期权数据“镜像”到 Redis Hash 中
        # 彻底解决 process_batch 对 live_option_snapshot 的依赖（使其不再处于断流状态）
        if IS_LIVEREPLAY and sym in TARGET_SYMBOLS:
            opt_buckets = payload.get('option_buckets')
            opt_contracts = payload.get('opt_contracts')
            if opt_buckets and opt_contracts:
                try:
                    snap_payload = {
                        'real_history_len': elapsed_minutes,
                        'symbol': sym,
                        'ts': payload.get('ts'),
                        'buckets': opt_buckets,
                        'contracts': opt_contracts
                    }
                    self.r.hset(HASH_OPTION_SNAPSHOT, sym, ser.pack(snap_payload))
                except Exception as e:
                    logger.error(f"❌ Failed to mirror option snapshot for {sym}: {e}")

        if sym not in self.states: return
        
        st = self.states[sym]
        if st.position == 0 or st.is_pending: return 

       # 👇 [🔥 终极修复 1：拔除 time.time() 毒药，强制使用逻辑时钟]
        curr_ts = payload.get('ts')
        if not curr_ts:
            if os.environ.get('RUN_MODE') == 'LIVEREPLAY' and hasattr(self, 'last_curr_ts'):
                curr_ts = self.last_curr_ts
            else:
                curr_ts = time.time()
        # 👆
        
        # 👇 [🔥 终极修复 2：防止 60.0 秒整点浮点数死锁，改为 59.0]
        if curr_ts - st.entry_ts < 59.0: 
            return
        # 👆
            
        # =================================================================
        # ⏱️ [终极修复 2] 实盘风控节流阀 (Throttle Lock)
        # =================================================================
        last_fast_check = getattr(st, 'last_fast_check', 0)
        if curr_ts - last_fast_check < 15.0: # 高频扫描间隔设为 15 秒
            return  
        
        
            
        st.last_fast_check = curr_ts 
        
        # =================================================================
        # 🔍 [终极修复 3] 取消原先错误的注释，恢复真正的数据解析逻辑
        # =================================================================
        stock_data = payload.get('stock', {})
        stock_price = float(stock_data.get('close', 0.0))
        if stock_price <= 0: return
        
        opt_buckets = payload.get('option_buckets', [])
        if not opt_buckets or len(opt_buckets) < 2: return
        
        from config import TAG_TO_INDEX
        idx_c = TAG_TO_INDEX.get('CALL_ATM', 2)
        idx_p = TAG_TO_INDEX.get('PUT_ATM', 0)
        
        # 根据当前持仓方向选择对应的期权 Bucket
        idx = idx_c if st.position == 1 else idx_p
        
        try:
            opt_data = opt_buckets[idx]
            base_price = float(opt_data[0])
            bid = float(opt_data[8]) if len(opt_data) > 8 else 0.0
            ask = float(opt_data[9]) if len(opt_data) > 9 else 0.0
            
            # 使用系统统一的公允价计算逻辑
            market_opt_price = self._get_fair_market_price(base_price, bid, ask)
                
        except Exception as e:
            # 捕获异常而不是静默吞掉，方便未来排错
            logger.debug(f"⚠️ Fast tick parse error for {sym}: {e}")
            return
            
        # 极度干涸或错误盘口，交给 1min 主循环的断流估值法去处理，高频这里不接锅
        if market_opt_price <= 0.01:
            logger.debug(f"⚠️ [Fast Tick 忽略] {sym} 高频市价过低 ({market_opt_price})，跳过本次高频检查。")
            return 

        # [🎯 靶向日志 A1] 确认高频流已捕捉到持仓，并准备送审
        st_held_mins = (curr_ts - st.entry_ts) / 60.0
        st_current_roi = (market_opt_price - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
        logger.debug(f"⚡ [Fast Tick 嗅探] {sym} | 现价: {market_opt_price:.2f} | 成本: {st.entry_price:.2f} | 当前 ROI: {st_current_roi*100:.2f}% | 已持仓: {st_held_mins:.1f}分钟")

        from datetime import datetime
        from pytz import timezone
        ny_now = datetime.fromtimestamp(curr_ts, tz=timezone('America/New_York'))
        
        # 组装 check_exit 所需的上下文 (补齐了 V15 策略需要的全部字段)
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': stock_price,
            'alpha_z': getattr(st, 'last_alpha_z', 0.0),
            'stock_roc': (stock_price - st.entry_stock) / st.entry_stock if st.entry_stock > 0 else 0.0,
            'macd_hist': getattr(st, 'last_macd_hist', 0.0),
            'macd_hist_slope': getattr(st, 'last_macd_hist_slope', 0.0),
            'spy_roc': getattr(st, 'last_spy_roc', 0.0), 
            'qqq_roc': getattr(st, 'last_qqq_roc', 0.0),
            'position': st.position, 
            'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete, 
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts),
            'stock_iv': st.last_valid_iv,
            'holding': {
                'entry_price': st.entry_price, 'entry_stock': st.entry_stock, 
                'entry_ts': st.entry_ts, 'dir': st.position, 
                'max_roi': st.max_roi, 'entry_spy_roc': getattr(st, 'entry_spy_roc', 0.0),
                'entry_index_trend': getattr(st, 'entry_index_trend', 0)
            },
            'curr_price': market_opt_price, 'curr_stock': stock_price,
            'bid': bid,
            'ask': ask,
            'snap_roc': getattr(st, 'last_snap_roc', 0.0),
            'global_regime_reversal_cnt': getattr(self, 'last_global_regime_reversal_cnt', 0),
            'regime_reversal_count': getattr(self, 'last_global_regime_reversal_cnt', 0),
            'is_volatile_regime': getattr(self, 'last_global_is_volatile_regime', False),
        }
        
        # 高频更新 max_roi (让 Protect 和 Trailing Stop 的追踪更加灵敏)
        if st.entry_price > 0:
            current_roi = (market_opt_price - st.entry_price) / st.entry_price
            if current_roi > st.max_roi:
                st.max_roi = current_roi
            ctx['holding']['max_roi'] = st.max_roi
            
        # 调用核心策略进行裁决
        exit_sig = self.strategy.check_exit(ctx)
        
        if exit_sig:
            exit_sig['price'] = market_opt_price
            exit_sig['market_price'] = market_opt_price
            exit_sig['bid'] = bid
            exit_sig['ask'] = ask
            
            logger.info(f"⚡ [Fast Tick 触发] {sym} 捕捉到高频平仓信号: {exit_sig['reason']}")
            
            # 执行离场
            await self._execute_exit(sym, exit_sig, stock_price, curr_ts, -1)

    def _get_opt_data_realtime(self, sym, st, ny_now, price, batch=None):
        """[Refactor] 解析实时期权快照 (Redis)"""
        from config import HASH_OPTION_SNAPSHOT, TAG_TO_INDEX
        
        # 初始化默认值
        res = {
            'has_feed': False, 'call_price': 0.0, 'put_price': 0.0,
            'call_id': "", 'put_id': "", 'call_k': 0.0, 'put_k': 0.0,
            'call_iv': 0.0, 'put_iv': 0.0, 'call_bid': 0.0, 'call_ask': 0.0,
            'put_bid': 0.0, 'put_ask': 0.0, 'call_vol': 1.0, 'put_vol': 1.0,
            'call_bid_size': 0.0, 'call_ask_size': 0.0, 'put_bid_size': 0.0, 'put_ask_size': 0.0
        }

        # =========================================================
        # 🚀 [终极时空对齐] 优先使用推流时合并好的绝对对齐数据！
        # =========================================================
        buckets, contracts = [], []
        if batch and 'live_options' in batch and sym in batch['live_options']:
            snap_data = batch['live_options'][sym]
            buckets = snap_data.get('buckets', [])
            contracts = snap_data.get('contracts', [])
            logger.debug(f"🎯 [Pre-Join] Using aligned option data for {sym}")
        else:
            # frame 驱动的推理主链：禁止回退读 HASH，避免事实源分叉
            if batch and ('frame_id' in batch):
                return res
            # [兜底降级] 如果 batch 里没有，才去 Redis 查（实盘单股快照可能走这里）
            raw_snap = self.r.hget(HASH_OPTION_SNAPSHOT, sym)
            if not raw_snap: return res
            try:
                snap_data = ser.unpack(raw_snap)
                buckets = snap_data.get('buckets', [])
                contracts = snap_data.get('contracts', [])
                logger.debug(f"📡 [Fallback] Using Redis option data for {sym}")
            except Exception as e:
                logger.warning(f"Failed to parse redis option snapshot for {sym}: {e}")
                return res

        try:
            # 此处不再需要重复 ser.unpack(raw_snap)，因为上面已经处理过了
            # buckets 和 contracts 已经就绪
            if buckets is None or len(buckets) == 0: return res

            
            idx_c = TAG_TO_INDEX.get('CALL_ATM')
            idx_p = TAG_TO_INDEX.get('PUT_ATM')
            
            if idx_c is None or idx_p is None:
                logger.error(f"标签映射失败! sym={sym}")
                return res

            if len(buckets) > max(idx_c, idx_p) and len(contracts) > max(idx_c, idx_p):
                # CALL
                _c_last = float(buckets[idx_c][0])
                _c_bid  = float(buckets[idx_c][8]) if len(buckets[idx_c]) > 8 else 0.0
                _c_ask  = float(buckets[idx_c][9]) if len(buckets[idx_c]) > 9 else 0.0
                res['call_price'] = self._get_fair_market_price(_c_last, _c_bid, _c_ask)
                res['call_k']     = float(buckets[idx_c][5])
                res['call_vol']   = float(buckets[idx_c][6])
                res['call_iv']    = float(buckets[idx_c][7])
                res['call_id']    = str(contracts[idx_c])
                res['call_bid']   = _c_bid
                res['call_ask']   = _c_ask
                res['call_bid_size'] = float(buckets[idx_c][10]) if len(buckets[idx_c]) > 10 else 0.0
                res['call_ask_size'] = float(buckets[idx_c][11]) if len(buckets[idx_c]) > 11 else 0.0

                # PUT
                _p_last = float(buckets[idx_p][0])
                _p_bid  = float(buckets[idx_p][8]) if len(buckets[idx_p]) > 8 else 0.0
                _p_ask  = float(buckets[idx_p][9]) if len(buckets[idx_p]) > 9 else 0.0
                res['put_price']  = self._get_fair_market_price(_p_last, _p_bid, _p_ask)
                res['put_k']      = float(buckets[idx_p][5])
                res['put_vol']    = float(buckets[idx_p][6])
                res['put_iv']     = float(buckets[idx_p][7])
                res['put_id']     = str(contracts[idx_p])
                res['put_bid']    = _p_bid
                res['put_ask']    = _p_ask
                res['put_bid_size'] = float(buckets[idx_p][10]) if len(buckets[idx_p]) > 10 else 0.0
                res['put_ask_size'] = float(buckets[idx_p][11]) if len(buckets[idx_p]) > 11 else 0.0
                
                res['has_feed'] = True
        except Exception as e:
            logger.warning(f"Failed to parse live option snapshot for {sym}: {e}")
        
        return res

    def _mark_orch_done(self, curr_ts: float, frame_id: str = None):
        if not hasattr(self, 'r'):
            return
        self.r.set("sync:orch_done", str(curr_ts))
        if frame_id:
            self.r.set("sync:orch_done_frame_id", str(frame_id))

    def _is_duplicate_frame(self, frame_id: str) -> bool:
        if not frame_id:
            return False
        return frame_id in self.processed_frame_set

    def _remember_frame(self, frame_id: str):
        if not frame_id:
            return
        if frame_id in self.processed_frame_set:
            return
        self.processed_frame_ids.append(frame_id)
        self.processed_frame_set.add(frame_id)
        while len(self.processed_frame_set) > self.processed_frame_ids.maxlen:
            old = self.processed_frame_ids.popleft()
            self.processed_frame_set.discard(old)

    def _get_opt_data_backtest(self, batch, i, sym, st):
        """[Refactor] 解析离线期权数据 (Batch Payload) + NaN 过滤"""
        def _safe_f(val, fallback=0.0):
            try:
                v = float(val)
                return fallback if np.isnan(v) else v
            except:
                return fallback

        res = {
            'has_feed': False, 'call_price': 0.0, 'put_price': 0.0,
            'call_id': "", 'put_id': "", 'call_k': 0.0, 'put_k': 0.0,
            'call_iv': 0.0, 'put_iv': 0.0, 'call_bid': 0.0, 'call_ask': 0.0,
            'put_bid': 0.0, 'put_ask': 0.0, 'call_vol': 1.0, 'put_vol': 1.0,
            'call_bid_size': 0.0, 'call_ask_size': 0.0, 'put_bid_size': 0.0, 'put_ask_size': 0.0
        }
        
        if 'feed_call_price' in batch or 'cheat_call' in batch:
            res['has_feed'] = True
            res['call_price'] = _safe_f(batch.get('feed_call_price', batch.get('cheat_call', [0]*len(batch['symbols'])))[i])
            res['put_price']  = _safe_f(batch.get('feed_put_price', batch.get('cheat_put', [0]*len(batch['symbols'])))[i])
            
            # [Fix] 这里的 bid/ask/size 不能为 0，否则触发 Orchestrator 的 Strict Guard 从而拒绝开仓
            res['call_bid'] = float(batch.get('feed_call_bid', [res['call_price']]*len(batch['symbols']))[i])
            res['call_ask'] = float(batch.get('feed_call_ask', [res['call_price']]*len(batch['symbols']))[i])
            res['put_bid']  = float(batch.get('feed_put_bid', [res['put_price']]*len(batch['symbols']))[i])
            res['put_ask']  = float(batch.get('feed_put_ask', [res['put_price']]*len(batch['symbols']))[i])
            
            res['call_bid_size'] = _safe_f(batch.get('feed_call_bid_size', [100.0]*len(batch['symbols']))[i], 100.0)
            res['call_ask_size'] = _safe_f(batch.get('feed_call_ask_size', [100.0]*len(batch['symbols']))[i], 100.0)
            res['put_bid_size']  = _safe_f(batch.get('feed_put_bid_size', [100.0]*len(batch['symbols']))[i], 100.0)
            res['put_ask_size']  = _safe_f(batch.get('feed_put_ask_size', [100.0]*len(batch['symbols']))[i], 100.0)
            
            res['call_vol'] = _safe_f(batch.get('feed_call_vol', batch.get('cheat_call_vol', [1.0]*len(batch['symbols'])))[i], 1.0)
            res['put_vol']  = _safe_f(batch.get('feed_put_vol', batch.get('cheat_put_vol', [1.0]*len(batch['symbols'])))[i], 1.0)
            
            c_ids = batch.get('feed_call_id', batch.get('cheat_call_id', [""]*len(batch['symbols'])))
            p_ids = batch.get('feed_put_id', batch.get('cheat_put_id', [""]*len(batch['symbols'])))
            c_id_raw = str(c_ids[i]) if i < len(c_ids) else ""
            p_id_raw = str(p_ids[i]) if i < len(p_ids) else ""
            if c_id_raw and c_id_raw.lower() not in ["nan", "none"]: st.latest_call_id = c_id_raw
            if p_id_raw and p_id_raw.lower() not in ["nan", "none"]: st.latest_put_id = p_id_raw
            
            res['call_id'] = st.latest_call_id if st.latest_call_id else f"{sym}_C_MOCK"
            res['put_id']  = st.latest_put_id  if st.latest_put_id  else f"{sym}_P_MOCK"
            
            res['call_k']  = _safe_f(batch.get('feed_call_k', batch.get('cheat_call_k', [0]*len(batch['symbols'])))[i])
            res['put_k']   = _safe_f(batch.get('feed_put_k', batch.get('cheat_put_k', [0]*len(batch['symbols'])))[i])
            res['call_iv'] = _safe_f(batch.get('feed_call_iv', batch.get('cheat_call_iv', [0]*len(batch['symbols'])))[i])
            res['put_iv']  = _safe_f(batch.get('feed_put_iv', batch.get('cheat_put_iv', [0]*len(batch['symbols'])))[i])

        else:
            # 👇 [🔥 致命漏洞修复防弹装甲] 
            # 如果 batch 里根本没有期权数组，说明外壳脚本传错了模式 (把 LIVEREPLAY 传成了 backtest)
            # 我们直接在此处动态拦截，强行转去读取 Redis 的实时快照！
            return self._get_opt_data_realtime(sym, st, None, None)
            # 👆 修复结束
            #  
        return res

    def _prep_symbol_metrics(self, i, sym, stock_prices, raw_alphas, opt_data, metrics_batch, use_precalc_feed=False, metrics_batch_manual=None):
        """[核心] 为单个标的准备分钟级指标，支持分钟边界的价格与期权采样纠偏"""
        converge_to_single = os.environ.get('DUAL_CONVERGE_TO_SINGLE') == '1'
        if sym not in self.states:
            if self.mode == 'backtest': self.states[sym] = SymbolState(sym)
            else: return None

        st = self.states[sym]
        
        # 🚀 [Parity Fix] 统一对位点：始终相信 Payload 提供的分钟级事实
        # 不再根据模式进行采样对位修正，彻底解决秒级回放与分钟级基准的 Alpha 漂移
        price = float(stock_prices[i]) if i < len(stock_prices) else 0.0
        calc_price = price
        calc_opt_data = opt_data
        
        raw_alpha_val = float(raw_alphas[i])
        raw_vols = metrics_batch.get('fast_vol', [0.0] * (i + 1))
        alpha_label_ts_arr = metrics_batch.get('alpha_label_ts', [0.0] * (i + 1))
        alpha_available_ts_arr = metrics_batch.get('alpha_available_ts', [0.0] * (i + 1))
        
        # 1. 更新指标状态 (MACD, ROC)
        st.update_indicators(calc_price, raw_alpha_val, ts=metrics_batch.get('curr_ts'), use_precalc_feed=use_precalc_feed)
        
        # 🧪 [策略接口] 获取稳定的分钟级指标
        roc_5m, macd, macd_slope, snap_roc = st.get_strategy_metrics()
        
        # [🔥 保存] 暂存指标供 evaluate 使用
        st.last_snap_roc = snap_roc
        st.last_macd_hist = macd
        st.last_macd_hist_slope = macd_slope

        # Alpha & Vol 缩放
        vol_z_dict = metrics_batch.get('vol_z_dict', {})
        if sym in NON_TRADABLE_SYMBOLS:
            if sym in vol_z_dict:
                try:
                    st.last_vol_z = float(vol_z_dict[sym])
                except Exception:
                    pass
            return None
        
        if use_precalc_feed:
            # 🚀 [回测/预计算模式] 直接透传预处理好的 Z-Score
            alpha_z = float(raw_alpha_val)
            vol_z = float(raw_vols[i])
            cross_z = alpha_z # 🚀 [修复] 定义缺失的 cs_alpha_z 变量，防止回测崩溃
            if sym == 'DELL' and abs(alpha_z) > 1.4:
                logger.info(f"🧪 [Alpha-Debug] DELL raw:{raw_alpha_val:.4f} | alpha_z:{alpha_z:.4f}")
        else:
            # 📡 [实时/对冲模式] 双重 Alpha Z-Score 校验
            
            # 1. 截面 Z-Score (Cross-Sectional): 当前这 1 分钟，它在全市场里的相对排名
            batch_alpha_mean = metrics_batch.get('alpha_mean', self.dynamic_alpha_mean)
            batch_alpha_std  = metrics_batch.get('alpha_std', self.dynamic_alpha_std)
            cross_z = float((raw_alpha_val - batch_alpha_mean) / (batch_alpha_std + 1e-6))
            
            # 🚀 [Parity Fix] 统一使用 100% 截面归一化，确保实盘/秒级回访与分钟级基准完全对齐
            alpha_z = max(-5.0, min(5.0, cross_z))
            
            # Vol 提取非对称 EWMA 结果
            if sym in vol_z_dict:
                vol_z = float(vol_z_dict[sym])
            else:
                batch_vol_mean   = getattr(self, 'dynamic_vol_mean', 0.07)
                batch_vol_std    = getattr(self, 'dynamic_vol_std', 0.1)
                vol_z = float((float(raw_vols[i]) - batch_vol_mean) / (batch_vol_std + 1e-6))
                vol_z = max(-5.0, min(5.0, vol_z))
        
        st.last_vol_z = vol_z
        
        # [Fix] 统一采用标准归一化，支持自适应反转修正
        final_alpha = float(-alpha_z if st.correction_mode == "INVERT" else alpha_z)
            
        st.prev_alpha_z = float(st.last_alpha_z)
        st.last_alpha_z = final_alpha
        
        # [Diagnostic]
        if (IS_BACKTEST or getattr(self, '_alpha_log_count', 0) < 20000) and sym in self.symbols:
            #logger.info(f"🔍 [Alpha Trace] {sym} | Z: {alpha_z:.4f} | ROC: {snap_roc:.4f} | MACD: {macd:.4f} | Dir: {st.correction_mode}")
            self._alpha_log_count = getattr(self, '_alpha_log_count', 0) + 1
        
        metrics = {
            'price': price,
            'final_alpha': final_alpha,
            'cs_alpha_z': cross_z, # [🆕 新增] 纯截面 R-Alpha (只看相对大盘的强弱)
            'vol_z': vol_z,
            'roc_5m': roc_5m,
            'macd': macd,
            'macd_slope': macd_slope,
            'snap_roc': snap_roc,
            'event_prob': self.cached_event_probs.get(sym, 0.0), # [🆕 新增]
            'alpha_label_ts': float(alpha_label_ts_arr[i]) if i < len(alpha_label_ts_arr) else 0.0,
            'alpha_available_ts': float(alpha_available_ts_arr[i]) if i < len(alpha_available_ts_arr) else 0.0,
            'st': st
        }
        return metrics
        
    # 🚀 [零干扰修复] 签名增加 should_update_full，用于决定是否写 Alpha Log
    async def _evaluate_symbol_signals(self, i, sym, metrics, opt_data, ny_now, curr_ts, spy_roc, qqq_roc, is_zombie_market, index_trend=0, global_regime_reversal_cnt=0, global_is_volatile_regime=None, should_update_full=True):
        """[Refactor] 核心策略评价 logic (平仓与开仓信号收集) - 修复空仓不交易BUG"""
        from config import USE_BID_ASK_PRICING
        st = metrics['st']
        if global_is_volatile_regime is None:
            global_is_volatile_regime = getattr(self, 'last_global_is_volatile_regime', False)
        
        # 🚀 [终极期权对位] 
        # 如果当前是分钟边界 (:00 秒触发的决策)，为了对齐 1m 基准表，必须使用上一秒缓存的期权快照 (:59)
        real_opt_data = opt_data
        if int(curr_ts) % 60 == 0 and st.last_tick_opt_data is not None:
            real_opt_data = st.last_tick_opt_data
        
        price = metrics['price']
        final_alpha = metrics['final_alpha']
        alpha_log_iv = st.last_valid_iv

        # Alpha 日志固定记录“当前 batch 自带的分钟快照 IV”，
        # 不跟随 real_opt_data 的 :59 回退逻辑，避免表里看起来慢一拍。
        if opt_data['has_feed']:
            if st.position == 1:
                log_iv_candidate = opt_data.get('call_iv', 0.0)
            elif st.position == -1:
                log_iv_candidate = opt_data.get('put_iv', 0.0)
            else:
                log_call_iv = opt_data.get('call_iv', 0.0)
                log_put_iv = opt_data.get('put_iv', 0.0)
                if log_call_iv > 0.01 and log_put_iv > 0.01:
                    log_iv_candidate = (log_call_iv + log_put_iv) / 2.0
                elif log_call_iv > 0.01:
                    log_iv_candidate = log_call_iv
                else:
                    log_iv_candidate = log_put_iv

            if log_iv_candidate > 0.01:
                alpha_log_iv = log_iv_candidate

         # 1. 更新 IV 状态：消除均值污染，采用流动性择优法则
        if real_opt_data['has_feed']:
            if st.position == 1: 
                curr_iv = real_opt_data['call_iv']
            elif st.position == -1: 
                curr_iv = real_opt_data['put_iv']
            else: 
                # 🚀 [对齐修复] 强制使用均值 IV，确保与 1m 基准 100% 对位
                if real_opt_data.get('call_iv', 0) > 0 and real_opt_data.get('put_iv', 0) > 0:
                    curr_iv = (real_opt_data['call_iv'] + real_opt_data['put_iv']) / 2.0
                elif real_opt_data.get('call_iv', 0) > 0:
                    curr_iv = real_opt_data['call_iv']
                else:
                    curr_iv = real_opt_data['put_iv']
                    
            if curr_iv > 0.01:
                st.last_valid_iv = curr_iv

        # 🚀 [终极修复]：动态方向推断 (Dynamic Direction Inference)
        # 如果空仓，我们利用 Alpha 的方向预判策略想看哪个盘口，避免传入 0.0 导致策略的风险校验拒单！
        eval_dir = st.position if st.position != 0 else (1 if final_alpha > 0 else -1)
        
        ctx_bid = real_opt_data['call_bid'] if eval_dir == 1 else real_opt_data['put_bid']
        ctx_ask = real_opt_data['call_ask'] if eval_dir == 1 else real_opt_data['put_ask']
        market_opt_price = real_opt_data['call_price'] if eval_dir == 1 else real_opt_data['put_price']
        
        # 计算 Context 中的公允价
        ctx_curr_price = 0.0
        if real_opt_data['has_feed']:
            # 🚀 [诊断日志] 打印 Signal Engine 接收到的期权数据细节
            if sym == 'NVDA' and getattr(self, '_iv_se_loud_count', 0) < 3:
                logger.info(f"📢 [SE_LOUD_TRACE] {sym} | IV_from_OptData: {opt_data.get('call_iv', -1):.4f} | Has_Feed: {opt_data['has_feed']}")
                self._iv_se_loud_count = getattr(self, '_iv_se_loud_count', 0) + 1
            
            ctx_curr_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask, getattr(st, 'last_opt_price', 0.0))
        elif st.position != 0:
            ctx_curr_price = max(st.entry_price, 0.01)

        # 2. 构建 Context
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': price,
            'alpha': final_alpha, # 🚀 [Critical Fix] 补充策略接口必需的 alpha 键
            'alpha_z': final_alpha, 'cs_alpha_z': metrics.get('cs_alpha_z', final_alpha), # [🆕 新增]
            'vol_z': metrics['vol_z'], 'stock_roc': metrics['roc_5m'],
            'event_prob': self.cached_event_probs.get(sym, 0.0),  
            'macd_hist': metrics['macd'], 'macd_hist_slope': metrics['macd_slope'],
            'spy_roc': spy_roc, 'qqq_roc': qqq_roc,
            'index_trend': index_trend, 
            'position': st.position, 'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete,
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts) if st.position != 0 else 0.0,
            'stock_iv': st.last_valid_iv,
            'holding': {'entry_price': st.entry_price, 'entry_stock': st.entry_stock, 'entry_ts': st.entry_ts, 'dir': st.position, 'max_roi': st.max_roi, 'entry_spy_roc': st.entry_spy_roc, 'entry_index_trend': getattr(st, 'entry_index_trend', 0)} if st.position != 0 else None,
            'curr_price': ctx_curr_price, 'curr_stock': price,
            'bid': ctx_bid,
            'ask': ctx_ask,
            'spread_divergence': 0.0, 
            'snap_roc': metrics['snap_roc'],
            'global_regime_reversal_cnt': global_regime_reversal_cnt, # 🚀 [NEW]
            'regime_reversal_count': global_regime_reversal_cnt,     # For V0 compatibility
            'is_volatile_regime': bool(global_is_volatile_regime),
            'state': st
        }
        
        # 🚨 [IMPORTANT] 在整个信号处理流的末尾，缓存当前价格与期权
        # 给下一秒 (:00) 的分钟指标计算锁定物理截面
        st.last_tick_price = price
        st.last_tick_opt_data = opt_data
 
        
        # 补齐 Spread Divergence 与 ROI 更新 (仅持仓时进行计算，防止污染空仓环境)
        if st.position != 0:
            # 🛡️ [价格防投毒] 只有当期权价格合理时才更新缓存
            # 若 ctx_curr_price 超过股票价格的 50%，说明是数据污染，拒绝覆盖
            _stock_ref = float(ctx.get('curr_stock', price))
            _is_valid_opt_price = (ctx_curr_price > 0.01 and 
                                   (_stock_ref <= 0 or ctx_curr_price < _stock_ref * 0.5))
            if _is_valid_opt_price:
                st.last_opt_price = ctx_curr_price
            if ctx_bid > 0 and ctx_ask > 0 and ctx_curr_price > 0.01:
                curr_s = (ctx_ask - ctx_bid) / ctx_curr_price
                if st.last_spread_pct > 0:
                    ctx['spread_divergence'] = curr_s - st.last_spread_pct
                st.last_spread_pct = curr_s

            if st.entry_price > 0:
                current_roi = (ctx_curr_price - st.entry_price) / st.entry_price
                st.max_roi = max(st.max_roi, current_roi)
                ctx['holding']['max_roi'] = st.max_roi

        # Alpha Log (零干扰：仅在 full update 窗口写入)
        if should_update_full:
            self._emit_trade_log({
                'action': 'ALPHA', 
                'ts': getattr(self, 'current_log_ts', curr_ts), # Use logical log_ts for DB storage
                'symbol': sym,
                'alpha': final_alpha, 'iv': alpha_log_iv, 'price': price, 'vol_z': metrics['vol_z'],
                'event_prob': self.cached_event_probs.get(sym, 0.0), 
                'index_trend': index_trend 
            })

        # 3. 执行平仓
        if st.position != 0:
            # if curr_ts - st.entry_ts < 59.0:
            #     return None

            current_roi = (ctx_curr_price - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
          
            exit_sig = self.strategy.check_exit(ctx)
            if exit_sig:
                exit_sig['price'] = ctx_curr_price
                exit_sig['market_price'] = market_opt_price
                if opt_data['has_feed']:
                    exit_sig['bid'] = ctx_bid
                    exit_sig['ask'] = ctx_ask
                    exit_sig['bid_size'] = opt_data['call_bid_size'] if st.position == 1 else opt_data['put_bid_size']
                    exit_sig['ask_size'] = opt_data['call_ask_size'] if st.position == 1 else opt_data['put_ask_size']
                else:
                    exit_sig['bid'] = ctx_curr_price
                    exit_sig['ask'] = ctx_curr_price
                    exit_sig['bid_size'] = 999.0
                    exit_sig['ask_size'] = 999.0
                    
                if not getattr(self, 'only_log_alpha', False):
                    await self._execute_exit(sym, exit_sig, price, curr_ts, i)
            return None

        # 4. 开仓决策
        if not should_update_full: return None  # 🚀 [核心门控] 高频增量帧仅允许平仓，坚决禁止开仓检查

        if not ctx['is_ready']:
            if getattr(self, '_warmup_log_count', 0) < 5:
                logger.info(f"⏳ [SE-Gate] {sym} not ready (Warmup: {len(self.states[sym].alpha_history)})")
                self._warmup_log_count = getattr(self, '_warmup_log_count', 0) + 1
            return None
        
        if curr_ts < self.global_cooldown_until:
            logger.info(f"🛡️ [SE-Gate] Global Cooldown active until {self.global_cooldown_until}")
            return None
            
        if is_zombie_market:
            # logger.info(f"🧟 [SE-Gate] Zombie Market detected")
            return None
        
        no_entry_h = self.strategy.cfg.NO_ENTRY_HOUR
        no_entry_m = self.strategy.cfg.NO_ENTRY_MINUTE
        if ny_now.time() >= dt_time(no_entry_h, no_entry_m): return None
        
        entry_sig = self.strategy.decide_entry(ctx)
        if not entry_sig:
            # logger.info(f"⚪ [SE-Gate] {sym} strategy rejected entry")
            return None

        # [严格守卫]：策略决定好方向后，精准提取对应方向的真实参数提交订单！
        if opt_data['has_feed']:
            is_call = (entry_sig['dir'] == 1)
            t_price  = opt_data['call_price'] if is_call else opt_data['put_price']
            t_id     = opt_data['call_id']    if is_call else opt_data['put_id']
            t_k      = opt_data['call_k']     if is_call else opt_data['put_k']
            t_iv     = opt_data['call_iv']    if is_call else opt_data['put_iv']
            t_vol    = opt_data['call_vol']   if is_call else opt_data['put_vol']
            t_bid    = opt_data['call_bid']   if is_call else opt_data['put_bid']
            t_ask    = opt_data['call_ask']   if is_call else opt_data['put_ask']
            t_bs     = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
            t_as     = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']

            if USE_BID_ASK_PRICING:
                if t_bid <= 0 or t_ask <= 0:
                    t_bid = t_price
                    t_ask = t_price
            else:
                if t_price <= 0:
                    return None

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask)
            if fair_p < 0.05:
                logger.info(f"🚫 [SE-Gate] {sym} Fair Price {fair_p:.4f} too low")
                return None
            if not t_id:
                logger.info(f"🚫 [SE-Gate] {sym} missing Option ID")
                return None
            
            strike_valid = (t_k > 1.0 and abs(t_k - price) / max(price, 1.0) < 0.80)
            if strike_valid:
                intrinsic = max(0.0, price - t_k) if is_call else max(0.0, t_k - price)
                if fair_p < intrinsic * 0.9: return None
            
            entry_sig.update({
                'price': fair_p, 'contract_id': t_id,
                'meta': {
                    'strike': t_k, 'iv': t_iv, 'contract_id': t_id, 'volume': t_vol, 
                    'bid': t_bid, 'ask': t_ask, 'bid_size': t_bs, 'ask_size': t_as, 
                    'spy_roc': spy_roc, 'alpha_z': final_alpha, 
                    'index_trend': index_trend,
                    'alpha_label_ts': metrics.get('alpha_label_ts', 0.0),
                    'alpha_available_ts': metrics.get('alpha_available_ts', curr_ts),
                }
            })
            return entry_sig
        return None

    async def _evaluate_symbol_signals_back(self, i, sym, metrics, opt_data, ny_now, curr_ts, spy_roc, qqq_roc, is_zombie_market, index_trend=0):
        """[Refactor] 核心策略评价 logic (平仓与开仓信号收集)"""
        from config import USE_BID_ASK_PRICING
        st = metrics['st']
        price = metrics['price']
        final_alpha = metrics['final_alpha']
        alpha_log_iv = st.last_valid_iv

        if opt_data['has_feed']:
            if st.position == 1:
                log_iv_candidate = opt_data.get('call_iv', 0.0)
            elif st.position == -1:
                log_iv_candidate = opt_data.get('put_iv', 0.0)
            else:
                log_call_iv = opt_data.get('call_iv', 0.0)
                log_put_iv = opt_data.get('put_iv', 0.0)
                if log_call_iv > 0.01 and log_put_iv > 0.01:
                    log_iv_candidate = (log_call_iv + log_put_iv) / 2.0
                elif log_call_iv > 0.01:
                    log_iv_candidate = log_call_iv
                else:
                    log_iv_candidate = log_put_iv

            if log_iv_candidate > 0.01:
                alpha_log_iv = log_iv_candidate
        
        # 1. 更新 IV 状态
        if opt_data['has_feed']:
            if st.position == 1: curr_iv = opt_data['call_iv']
            elif st.position == -1: curr_iv = opt_data['put_iv']
            else:
                cv = opt_data['call_iv']
                pv = opt_data['put_iv']
                if cv > 0.01 and pv > 0.01: curr_iv = (cv + pv) / 2.0
                elif cv > 0.01: curr_iv = cv
                elif pv > 0.01: curr_iv = pv
                else: curr_iv = 0.0
            
            # 🚀 [Debug] 最终阶段 TRACE
            if sym == 'NVDA' and getattr(self, '_iv_se_count', 0) < 5:
                # 检查 opt_data 原始值
                c_data = opt_data.get('call_iv', -1)
                p_data = opt_data.get('put_iv', -1)
                logger.info(f"🧪 [IV_TRACE_3] {sym} | SE Raw OptData | Call_IV_Data: {c_data:.4f} | Put_IV_Data: {p_data:.4f} | Final_Curr_IV: {curr_iv:.4f}")
                self._iv_se_count = getattr(self, '_iv_se_count', 0) + 1

            if curr_iv > 0.01:
                st.last_valid_iv = curr_iv

        

        # 2. 构建 Context
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': price,
            'alpha_z': final_alpha, 'vol_z': metrics['vol_z'], 'stock_roc': metrics['roc_5m'],
            'macd_hist': metrics['macd'], 'macd_hist_slope': metrics['macd_slope'],
            'spy_roc': spy_roc, 'qqq_roc': qqq_roc,
            'index_trend': index_trend, # [NEW] 传导当日大盘趋势
            'position': st.position, 'cooldown_until': st.cooldown_until,
            'is_ready': st.warmup_complete,
            'is_banned': curr_ts < self.global_cooldown_until,
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts) if st.position != 0 else 0.0,
            'stock_iv': st.last_valid_iv,
            'holding': {'entry_price': st.entry_price, 'entry_stock': st.entry_stock, 'entry_ts': st.entry_ts, 'dir': st.position, 'max_roi': st.max_roi, 'entry_spy_roc': st.entry_spy_roc, 'entry_index_trend': getattr(st, 'entry_index_trend', 0)} if st.position != 0 else None,
            'curr_price': 0.0, 'curr_stock': price,
            # [🔥 核心新增] 传递 Bid/Ask 供策略进行流动性（Spread）校验
            'bid': opt_data.get('call_bid' if st.position >= 0 else 'put_bid', 0.0) if opt_data['has_feed'] else 0.0,
            'ask': opt_data.get('call_ask' if st.position >= 0 else 'put_ask', 0.0) if opt_data['has_feed'] else 0.0,
            'spread_divergence': 0.0, # 默认为 0
            'snap_roc': metrics['snap_roc'] # [🔥 新增] 传导最后一分钟/Snap的价格Delta
        }
        
        # 计算价差变化率 (Spread Divergence)
        if ctx['bid'] > 0 and ctx['ask'] > 0 and ctx['curr_price'] > 0.01:
             curr_s = (ctx['ask'] - ctx['bid']) / ctx['curr_price']
             if st.last_spread_pct > 0:
                 ctx['spread_divergence'] = curr_s - st.last_spread_pct
             st.last_spread_pct = curr_s

        # Alpha Log
        self._emit_trade_log({
            'action': 'ALPHA', 
            'ts': getattr(self, 'current_log_ts', curr_ts), # Use logical log_ts for DB storage
            'symbol': sym,
            'alpha': final_alpha, 'iv': alpha_log_iv, 'price': price, 'vol_z': metrics['vol_z'],
            'index_trend': index_trend # [NEW] 记录入场时的趋势背景
        })

        # 3. 计算期权公允价
        market_opt_price = 0.0
        if opt_data['has_feed']:
            market_opt_price = opt_data['call_price'] if st.position >= 0 else opt_data['put_price'] # st.position=0 时默认 call
            if st.position != 0:
                bid = opt_data['call_bid'] if st.position == 1 else opt_data['put_bid']
                ask = opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']
                
                # 在回测模式或盘口正常时，使用公允价 (Mid-price from Feature Service)
                # 🚀 [终极修正] 彻底删除 Delta 0.5 投影估值逻辑，始终信任数据源提供的公允价。
                ctx['curr_price'] = self._get_fair_market_price(market_opt_price, bid, ask)
                st.last_opt_price = ctx['curr_price'] # 👈 [新增] 缓存正常公允价
                
                if st.entry_price > 0:
                    current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price
                    st.max_roi = max(st.max_roi, current_roi)
                    ctx['holding']['max_roi'] = st.max_roi
                     
            else:
                ctx['curr_price'] = market_opt_price
                st.last_opt_price = ctx['curr_price'] # 👈 [新增]
        else:
            if st.position != 0:
                # 🛡️ [防断流核心 3] 彻底缺失数据 (has_feed=False)，直接用开仓价兜底
                effective_price = st.entry_price
                if effective_price <= 0.01: effective_price = 0.01
                
                # 👇 [🔥 把 Debug 改为 Error，强制暴露问题]
                ctx['curr_price'] = effective_price
                st.last_opt_price = ctx['curr_price'] # 👈 [新增] 缓存兜底价
                logger.error(f"🚨 [致命盲区] {sym} 期权行情彻底丢失(has_feed=False)！系统被蒙住双眼，当前 ROI 强制归 0.0！")
            else:
                ctx['curr_price'] = 0.0

        # 4. 执行平仓 (必须在 is_zombie_market 之前，确保 EOD 和止损能随时触发)
        if st.position != 0:
            high_freq_tick = self._is_high_freq_tick(st, curr_ts)
            # 🛡️ [终极修复] 建仓绝对保护期 (Entry Breathing Room)
            # 绝对禁止在建仓后的最初 60 秒内通过常规逻辑平仓
            if curr_ts - st.entry_ts < 59.0:
                logger.debug(f"🔒 [平仓屏蔽] {sym} 处于 59 秒建仓保护期内，暂不进行平仓评估。")
                return None

            # [🎯 靶向日志 B2] 送入 check_exit 前的最终状态切片
            current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
            logger.info(f"🔍 [主循环平仓送审] {sym} | 模式: {self.mode} | 当前价: {ctx['curr_price']:.2f} | 成本: {st.entry_price:.2f} | ROI: {current_roi*100:.2f}% | Max ROI: {st.max_roi*100:.2f}%")
          
            
            exit_sig = self.strategy.check_exit(ctx)
            if exit_sig:
                if high_freq_tick:
                    exit_sig = self._confirm_high_freq_exit(st, exit_sig, curr_ts)
                    if not exit_sig:
                        return None
                exit_sig['price'] = ctx['curr_price']
                exit_sig['market_price'] = market_opt_price
                if opt_data['has_feed']:
                    is_call = (st.position == 1)
                    exit_sig['bid'] = opt_data['call_bid'] if is_call else opt_data['put_bid']
                    exit_sig['ask'] = opt_data['call_ask'] if is_call else opt_data['put_ask']
                    # 🚀 [新增] 穿透传输深度数据，支持秒级分段批量成交
                    exit_sig['bid_size'] = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
                    exit_sig['ask_size'] = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']
                else:
                    exit_sig['bid'] = ctx['curr_price']
                    exit_sig['ask'] = ctx['curr_price']
                    exit_sig['bid_size'] = 999.0
                    exit_sig['ask_size'] = 999.0
                if not self.only_log_alpha:
                    await self._execute_exit(sym, exit_sig, price, curr_ts, i)
                    return None # 非 Parity 模式，等待 OMS 异步清算回调后再充当空仓
                else:
                    logger.info(f"📝 [Shadow] {sym} Exit signal detected, but skipping execution (Alpha-Only).")
                    return None
            else:
                if high_freq_tick:
                    self._reset_exit_confirmation(st)
                # [🎯 靶向日志 B3] 明确是被策略内部拒绝
                logger.debug(f"🛡️ [策略拒平] {sym} 的 check_exit 返回 None，当前无满足条件的平仓信号。")
                return None

        # 5. 开仓决策 (受僵尸市场和冷却时间保护)
        if curr_ts < self.global_cooldown_until or is_zombie_market: return None
        
        # [🔥 修复] 移除硬编码的 15:30，动态读取策略配置的停止开仓时间
        no_entry_h = self.strategy.cfg.NO_ENTRY_HOUR
        no_entry_m = self.strategy.cfg.NO_ENTRY_MINUTE
        if ny_now.time() >= dt_time(no_entry_h, no_entry_m): return None
        
        entry_sig = self.strategy.decide_entry(ctx)
        if not entry_sig:
            # logger.info(f"⚪ [SE-Gate] {sym} strategy rejected entry")
            return None

        # [严格守卫]
        if opt_data['has_feed']:
            is_call = (entry_sig['dir'] == 1)
            t_price  = opt_data['call_price'] if is_call else opt_data['put_price']
            t_id     = opt_data['call_id']    if is_call else opt_data['put_id']
            t_k      = opt_data['call_k']     if is_call else opt_data['put_k']
            t_iv     = opt_data['call_iv']    if is_call else opt_data['put_iv']
            t_vol    = opt_data['call_vol']   if is_call else opt_data['put_vol']
            t_bid    = opt_data['call_bid']   if is_call else opt_data['put_bid']
            t_ask    = opt_data['call_ask']   if is_call else opt_data['put_ask']
            t_bs     = opt_data['call_bid_size'] if is_call else opt_data['put_bid_size']
            t_as     = opt_data['call_ask_size'] if is_call else opt_data['put_ask_size']

            if USE_BID_ASK_PRICING:
                # 🚀 [放宽实盘与回测限制] 不再要求 Size > 0。如果 Bid/Ask 为 0，统一使用 Last Price 兜底放行
                if t_bid <= 0 or t_ask <= 0:
                    t_bid = t_price
                    t_ask = t_price
            else:
                if t_price <= 0:
                    logger.warning(f"🚫 [严格防线] {sym} 拦截! 成交价为零")
                    return None

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask)
            if fair_p < 0.05:
                logger.info(f"🚫 [SE-Gate] {sym} Fair Price {fair_p:.4f} too low")
                return None
            if not t_id:
                logger.info(f"🚫 [SE-Gate] {sym} missing Option ID")
                return None
            
            # 内在价值校验
            strike_valid = (t_k > 1.0 and abs(t_k - price) / max(price, 1.0) < 0.80)
            if strike_valid:
                intrinsic = max(0.0, price - t_k) if is_call else max(0.0, t_k - price)
                if fair_p < intrinsic * 0.9: return None
            
            # 记录空跑截影
            spread_pct = (t_ask - t_bid) / fair_p if fair_p > 0 else 0
            logger.info(f"📸 [实盘空跑截影] Alpha: {final_alpha:.2f} | {sym} | Bid:{t_bid:.2f} Ask:{t_ask:.2f} | Fair:{fair_p:.2f}")

            entry_sig.update({
                'price': fair_p, 'contract_id': t_id,
                'meta': {
                    'strike': t_k, 'iv': t_iv, 'contract_id': t_id, 'volume': t_vol, 
                    'bid': t_bid, 'ask': t_ask, 'bid_size': t_bs, 'ask_size': t_as, 
                    'spy_roc': spy_roc, 'alpha_z': final_alpha, 'ask_size': t_as, # [Fix] 显式传导 ask_size 供流动性评估
                    'index_trend': index_trend,
                    'alpha_label_ts': metrics.get('alpha_label_ts', 0.0),
                    'alpha_available_ts': metrics.get('alpha_available_ts', curr_ts),
                }
            })
            return entry_sig
        return None
    def _calculate_index_trend(self, symbols, stock_prices, spy_rocs, qqq_rocs):
        """[Helper] 计算 SPY/QQQ 复合趋势"""
        index_trend = 0
        spy_day_roc = 0.0
        qqq_day_roc = 0.0
        
        for i, sym in enumerate(symbols):
            if sym in INDEX_TREND_SYMBOLS:
                p = stock_prices[i]
                if sym not in self.index_opening_prices and p > 1.0:
                    self.index_opening_prices[sym] = p
                if sym in self.index_opening_prices:
                    open_p = self.index_opening_prices[sym]
                    day_roc = (p - open_p) / open_p
                    if sym == 'SPY': spy_day_roc = day_roc
                    else: qqq_day_roc = day_roc
        
        s_5m = np.mean([r for r in spy_rocs if abs(r) > 1e-9]) if any(abs(r) > 1e-9 for r in spy_rocs) else 0.0
        q_5m = np.mean([r for r in qqq_rocs if abs(r) > 1e-9]) if any(abs(r) > 1e-9 for r in qqq_rocs) else 0.0
        
        alpha_ema = 0.4
        self.spy_ema_roc = alpha_ema * s_5m + (1 - alpha_ema) * self.spy_ema_roc
        self.qqq_ema_roc = alpha_ema * q_5m + (1 - alpha_ema) * self.qqq_ema_roc
        
        is_bull_day = (spy_day_roc > 0.0002 or qqq_day_roc > 0.0002)
        is_bear_day = (spy_day_roc < -0.0002 or qqq_day_roc < -0.0002)
        
        if is_bull_day and (self.spy_ema_roc > -0.0001 or self.qqq_ema_roc > -0.0001):
            index_trend = 1
        elif is_bear_day and (self.spy_ema_roc < 0.0001 or self.qqq_ema_roc < 0.0001):
            index_trend = -1
            
        return index_trend, spy_day_roc, qqq_day_roc

    async def _run_model_inference(self, batch, symbols, prices, ny_now):
        """[Helper] 执行模型推斷与缓存"""
        # 🛡️ [核心修复] 预热期硬核熔断：严禁在数据不满 30 条时记录任何 Alpha 指标
        # real_len = batch.get('real_history_len', 30)
        # if real_len < 30:
        #     # 只有在整分时刻提示，防止刷屏日志
        #     if ny_now.second == 0:
        #         logger.warning(f"⚠️ [SE-Guard] Warmup Phase: real_len={real_len}/30. Forcing Alpha=0.0")
        #     for s in symbols: 
        #         self.cached_alphas[s] = 0.0
        #         self.cached_event_probs[s] = 0.0
        #     return None

        use_precalc = 'alpha_score' in batch or 'precalc_alpha' in batch
        if use_precalc:
            alphas = batch.get('precalc_alpha', batch.get('alpha_score'))
            # [🆕 新增] 提取 Event Prob 并缓存
            if 'event_prob' in batch:
                eprobs = batch['event_prob']
                # logger.info(f"📡 [SE-Model] Extracted {len(eprobs)} event_probs from batch")
                for j, s in enumerate(symbols):
                    self.cached_event_probs[s] = float(eprobs[j])
            return alphas
        
       
            
        features_dict = batch.get('features_dict')
        if features_dict:
            x_stk_list = []; x_opt_list = []
            for f in self.slow_cfg['features']:
                name = f['name']
                if name in {'open', 'high', 'low', 'close', 'volume', 'stock_id', 'sector_id', 'day_of_week'}: continue
                feat_data = features_dict.get(name)
                if feat_data is None: feat_data = np.zeros((len(symbols), 30), dtype=np.float32)
                
                t_feat = torch.from_numpy(feat_data.copy()).float().to(self.device)
                if f.get('type') == 'categorical': t_feat = torch.clamp(t_feat, 0, 50)
                
                if name.startswith('options_'): x_opt_list.append(t_feat)
                else: x_stk_list.append(t_feat)
            x_stk = torch.stack(x_stk_list, dim=-1)
            x_opt = torch.stack(x_opt_list, dim=-1)
        else:
            # Fallback for legacy format
            if 'x_stock' in batch and 'x_option' in batch:
                x_stk = torch.from_numpy(batch['x_stock']).float().to(self.device)
                x_opt = torch.from_numpy(batch['x_option']).float().to(self.device)
            else:
                x_slow = torch.from_numpy(batch['slow_1m']).float().to(self.device)
                x_stk = x_slow[..., self.slow_stock_indices]
                x_opt = x_slow[..., self.slow_option_indices]

        s_mock = {
            'stock_id': torch.from_numpy(batch['stock_id'].copy()).long().to(self.device) if 'stock_id' in batch else torch.zeros(len(symbols), dtype=torch.long).to(self.device),
            'sector_id': torch.zeros(len(symbols), dtype=torch.long).to(self.device),
            'day_of_week': torch.full((len(symbols),), ny_now.weekday(), dtype=torch.long).to(self.device),
            'hour': torch.full((len(symbols),), ny_now.hour, dtype=torch.long).to(self.device),
            'minute': torch.full((len(symbols),), ny_now.minute, dtype=torch.long).to(self.device)
        }
        
        with torch.no_grad():
            # 🚀 [Fingerprint Audit] Trace NVDA at 10:00:00
            if "NVDA" in symbols and ny_now.hour == 10 and ny_now.minute == 0 and ny_now.second == 0:
                idx_nvda = symbols.index("NVDA")
                stk_sample = x_stk[idx_nvda:idx_nvda+1]
                logger.info(f"📊 [TRACE-NVDA] Tensor Fingerprint | Mean: {stk_sample.mean():.6f} | Std: {stk_sample.std():.6f} | Max: {stk_sample.max():.6f}")
                # 🧪 [NEW] 保存全量指纹矩阵，用于像素级对碰
                np.save("nvda_replay_1000.npy", stk_sample.cpu().numpy())
                logger.info("💾 [TRACE-NVDA] Full Matrix saved to nvda_replay_1000.npy")
            out = self.slow_model(x_stk, x_opt, s_mock)
            
            # 1. 提取 Rank Score (Alpha)
            preds = out['rank_score'].cpu().numpy().flatten()
            for i_p, s_p in enumerate(symbols): self.cached_alphas[s_p] = preds[i_p]
            
            # 2. [NEW] 提取 Event Probability (爆发概率)
            if 'logits_event' in out:
                event_probs = torch.softmax(out['logits_event'], dim=-1)[:, 1].cpu().numpy()
                for i_p, s_p in enumerate(symbols): self.cached_event_probs[s_p] = float(event_probs[i_p])
            else:
                for i_p, s_p in enumerate(symbols): self.cached_event_probs[s_p] = 0.0
                
        return preds

    def _update_volatility_metrics(self, batch, symbols):
        """[Helper] 更新波动率 Z-Score 与 市场状态判定 (非对称 EWMA)"""
        vol_z_dict = {}
        import math
        raw_vols = batch.get('fast_vol', np.zeros(len(symbols)))
        stock_prices = batch.get('stock_price', np.zeros(len(symbols)))
        use_price_proxy_env = os.environ.get('VOL_Z_USE_PRICE_PROXY', '1').strip().lower()
        use_price_proxy = use_price_proxy_env not in {'0', 'false', 'no', 'off'}
        for idx_v, s_v in enumerate(symbols):
            r_v = float(raw_vols[idx_v]) if idx_v < len(raw_vols) else 0.0

            # 统一口径：优先使用价格变化代理，避免秒级/分钟级 fast_vol 口径差导致的漂移
            if use_price_proxy:
                px = float(stock_prices[idx_v]) if idx_v < len(stock_prices) else 0.0
                prev_px = float(self.sym_last_vol_price.get(s_v, 0.0))
                if px > 0:
                    if prev_px > 0:
                        # 将分钟收益映射到与历史风控相近量级，便于平滑迁移
                        r_v = abs((px - prev_px) / prev_px) * 30.0
                    else:
                        r_v = self.sym_vol_mean.get(s_v, 0.01)
                    self.sym_last_vol_price[s_v] = px
            
            # [Fix B] 波动率防爆保护：防断流
            if abs(r_v) < 1e-9:
                st = self.states.get(s_v)
                if st and hasattr(st, 'cached_min_roc'):
                    r_v = abs(st.cached_min_roc) * 10.0 # 粗略放大系数
                else:
                    r_v = self.sym_vol_mean.get(s_v, 0.01)

            if s_v not in self.sym_vol_mean:
                self.sym_vol_mean[s_v] = r_v
                self.sym_vol_var[s_v] = float(getattr(self, 'dynamic_vol_std', 0.1) ** 2)
            
            diff = r_v - self.sym_vol_mean[s_v]
            
            # 🚀 [核心适配 1：非对称 EWMA] 
            # 适应 5 分钟 Vol 尖峰：飙升时反应极快(15分钟窗口)，冷却时反应慢(60分钟窗口)
            if diff > 0:
                vol_ewma = 2.0 / (15 + 1)
            else:
                vol_ewma = 2.0 / (60 + 1)
                
            self.sym_vol_mean[s_v] += vol_ewma * diff
            self.sym_vol_var[s_v] = (1 - vol_ewma) * self.sym_vol_var[s_v] + vol_ewma * (diff ** 2)
            
            v_std = math.sqrt(self.sym_vol_var[s_v]) if self.sym_vol_var[s_v] > 1e-9 else 1e-4
            vz = (r_v - self.sym_vol_mean[s_v]) / (v_std + 1e-6)
            
            if abs(vz) > 10.0:
                vz = 0.0
                
            self.cached_vol_z[s_v] = max(-5.0, min(5.0, vz))
            vol_z_dict[s_v] = self.cached_vol_z[s_v]
            
        zero_vol_pct = sum(1 for s_v in symbols if abs(float(self.sym_vol_mean.get(s_v, 0.0))) < 1e-5) / len(symbols) if symbols else 0
        self.last_is_zombie = (zero_vol_pct > 0.50)
        return vol_z_dict

    def _update_vixy_regime_state(self, symbols, stock_prices, curr_ts):
        """只更新 VIXY 的分钟状态，用于 regime / hard stop 切档，不触发交易。"""
        for idx, sym in enumerate(symbols):
            if sym != 'VIXY':
                continue
            st = self.states.get(sym)
            if st is None:
                st = self.states[sym] = SymbolState(sym, config=self.cfg)
            try:
                price = float(stock_prices[idx]) if idx < len(stock_prices) else 0.0
            except Exception:
                price = 0.0
            if price > 0:
                st.update_indicators(price, 0.0, ts=curr_ts, use_precalc_feed=True)


    async def process_batch(self, batch: dict):
        # 🚀 [零干扰引擎] 自动识别数据模式
        # 如果是实盘/回放模式，或者环境变量强制开启了高频模式，则进入“升频执行，降频推理”
        parity_mode_1s = os.environ.get('REPLAY_1S_PARITY_MODE') == '1'
        is_high_freq = (
            not parity_mode_1s and
            (self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY' or os.environ.get('FORCE_HIGH_FREQ') == '1')
        )
        is_new_minute = batch.get('is_new_minute', True)
        symbols = batch.get('symbols', [])
        stock_prices = batch.get('stock_price', [])
        frame_id = str(batch.get('frame_id')) if batch.get('frame_id') is not None else None
        
        # 🚀 [Refine] use_precalc_feed should depend on whether alpha is actually provided in the batch
        use_precalc_feed = 'alpha_score' in batch or 'precalc_alpha' in batch
        
        # 🧪 [Parity Fix] 如果 batch 里带了 event_prob，直接同步到缓存供 metrics 使用
        if 'event_prob' in batch:
            for idx_p, s_p in enumerate(symbols):
                 self.cached_event_probs[s_p] = float(batch['event_prob'][idx_p])
        raw_vols = batch.get('fast_vol', np.zeros(len(symbols)))
        
        # 👑 每次处理 K 线前，先同步真实账本！
        self._sync_state_from_oms()

        # 1. 统一时间基准与 EOD 处理
        ny_now, curr_ts = self._prepare_ny_time(batch)
        if ny_now is None:
             return

        if frame_id and self._is_duplicate_frame(frame_id):
            logger.warning(f"♻️ [Frame-Dedupe] Skip duplicated frame_id={frame_id}")
            self._mark_orch_done(curr_ts, frame_id=frame_id)
            return

        # 🚀 [频率无关重构] 即使驱动没传 is_new_minute，SE 应该自己通过 ts 跨越来判定
        last_t = getattr(self, 'last_process_ts_for_gating', 0.0)
        is_new_min_crossing = (int(curr_ts / 60) > int(last_t / 60)) or (last_t == 0)
        self.last_process_ts_for_gating = curr_ts

        # 分钟级回测模式：每一帧都是 should_update_full，确保计算不被缓存拦截
        # 秒级模式 (LIVEREPLAY)：只有在 is_new_minute 为 True 时才是 should_update_full (由特征服务结算驱动)
        converge_to_single = os.environ.get('DUAL_CONVERGE_TO_SINGLE') == '1'
        if is_high_freq:
            # 🚨 [关键修复]：高频回放模式下，强制信任来自特征服务的结算信号！
            # 防止因为 ts 抖动 (10:00:00 -> 10:01:01) 导致的二次误触发。
            should_update_full = is_new_minute or (last_t == 0)
        else:
            # 分钟级回测/普通模式：沿用 K 线跨越检查
            should_update_full = is_new_min_crossing

        if should_update_full and is_high_freq:
            logger.info(f"🆕 [Minute Boundary] {ny_now} | ts: {curr_ts} | Symbols: {len(symbols)}")

        if should_update_full:
            self._update_vixy_regime_state(symbols, stock_prices, curr_ts)
        
        # ✅ [核心重构] 第一阶段：秒级更新（始终执行）
        # 1. 更新所有标的的 Tick 视图（价格、买卖盘）
        for idx_t, s_t in enumerate(symbols):
            if s_t in self.states:
                # 尝试获取买卖盘 (Simulated 下可能没有)
                bid = batch.get('bid', [None]*len(symbols))[idx_t]
                ask = batch.get('ask', [None]*len(symbols))[idx_t]
                self.states[s_t].update_tick_state(stock_prices[idx_t], bid, ask)

        # 2. 秒级风控扫荡（持仓止损、止盈判定）
        if not (converge_to_single and self.mode == 'backtest'):
            await self._process_exits(batch)

        # ✅ [核心重构] 第二阶段：分钟级策略（仅在分钟跨越或非高频模式下执行）
        if not should_update_full:
            # 高频增量帧：仅同步 PnL 后退出
            if IS_LIVEREPLAY:
                 await self._report_pnl_status_logic(curr_ts, "LIVEREPLAY_SYNC")
            await self._oms_sync(curr_ts, frame_id=frame_id)
            return

        # 🚀 以下逻辑仅在分钟整点运行 🚀
        if not use_precalc_feed and not bool(batch.get('is_warmed_up', False)):
            if getattr(self, '_warmup_gate_log_count', 0) < 10:
                logger.info(
                    f"⏳ [SE-Warmup-Gate] Skip inference at {ny_now.strftime('%H:%M:%S')} "
                    f"| real_history_len={batch.get('real_history_len', 0)}/{batch.get('warmup_required_len', 31)} "
                    f"| total_history_len={batch.get('total_history_len', 0)} "
                    f"| real_norm_history_len={batch.get('real_norm_history_len', 0)} "
                    f"| cross_day={batch.get('has_cross_day_warmup', False)}"
                )
                self._warmup_gate_log_count = getattr(self, '_warmup_gate_log_count', 0) + 1
            await self._oms_sync(curr_ts, frame_id=frame_id)
            return
        
        # 3. 记录行情数据 (为 Alpha 日志准备时间戳)
        # 🚀 [Surgery 24] 逻辑时间戳对齐：如果是整分结算，强制用 ts (逻辑上代表的那一分钟)，否则用 log_ts (物理时间)
        self.current_log_ts = batch.get('ts' if is_new_minute else 'log_ts', curr_ts)
        current_time = ny_now.time()
        current_date = ny_now.date()

        # 4. 主动强平与跨日清理
        close_h = self.strategy.cfg.CLOSE_HOUR
        close_m = self.strategy.cfg.CLOSE_MINUTE
        is_eod_time = (ny_now.hour == close_h and ny_now.minute >= close_m) or (ny_now.hour > close_h)
        
        if is_eod_time:
            last_clear_day = getattr(self, 'last_eod_clear_day', None)
            if last_clear_day != current_date:
                logger.warning(f"⏰ [时间风控] 到达策略指定平仓时间 {close_h}:{close_m}，启动全局主动强制清仓！")
                await self._force_clear_all(batch, "EOD_HARD_CLEAR", curr_ts, ny_now)
                self.last_eod_clear_day = current_date

        if self.last_date is not None and current_date > self.last_date:
            yesterday_close_ny = ny_now.replace(year=self.last_date.year, month=self.last_date.month, day=self.last_date.day, hour=close_h, minute=close_m, second=0, microsecond=0)
            await self._force_clear_all(batch, "EOD_FORCE", yesterday_close_ny.timestamp(), yesterday_close_ny)
            self.consecutive_stop_losses = 0; self.global_cooldown_until = 0; self.index_opening_prices = {}
            self._generate_daily_analysis_report(report_date_str=self.last_date.strftime('%Y%m%d'))
            self.daily_trades = []
        
        self.last_date = current_date

        # 5. 计算大盘趋势与指标
        spy_rocs = np.array(batch.get('spy_roc_5min', [0.0] * len(symbols)))
        qqq_rocs = np.array(batch.get('qqq_roc_5min', [0.0] * len(symbols)))
        self.last_spy_roc_val = float(np.mean(spy_rocs)) if len(spy_rocs) > 0 else 0.0
        self.last_qqq_roc_val = float(np.mean(qqq_rocs)) if len(qqq_rocs) > 0 else 0.0
        index_trend, spy_day_roc, qqq_day_roc = self._calculate_index_trend(symbols, stock_prices, spy_rocs, qqq_rocs)
        
        if is_new_min_crossing:
            logger.info(f"📈 [Index Audit] Trend: {index_trend} | SPY EMA ROC: {self.spy_ema_roc:.6f} | QQQ EMA ROC: {self.qqq_ema_roc:.6f}")
        
        batch['index_trend'] = index_trend
        self.last_index_trend = index_trend

        # 6. 模型推理 (Alpha Score)
        raw_alphas = await self._run_model_inference(batch, symbols, stock_prices, ny_now)
        
        # # 🚀 [Bug Fix] 如果推理引擎返回 None (热身期熔断)，则立刻中断当前批次的处理，防止后续迭代崩溃
        # if raw_alphas is None:
        #     return

        # 7. 自适应归一化 (仅在分钟边界更新均值/标差)
        # 🚀 [Parity Fix] 排除指数标的 (SPY, QQQ, VIXY) 对截面均值的污染，确保与 1m 离线基准对齐
        exclude_indices = {i for i, s in enumerate(symbols) if s in ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS}
        valid_alphas = [a for i, a in enumerate(raw_alphas) if i not in exclude_indices]
        
        if valid_alphas:
            mean_a = np.mean(valid_alphas)
            std_a = np.std(valid_alphas)
        else:
            mean_a = np.mean(raw_alphas); std_a = np.std(raw_alphas)

        # 🚀 [Parity Fix] 将瞬时截面指标注入 Batch，确保 _prep_symbol_metrics 
        # 使用当前时刻的“事实均值”进行归一化，彻底消除 EWMA 带来的状态漂移。
        batch['alpha_mean'] = float(mean_a)
        batch['alpha_std']  = float(std_a)

        if self.alpha_count == 0:
            self.dynamic_alpha_mean = float(mean_a); self.dynamic_alpha_std = float(std_a if std_a > 1e-5 else 1.0)
        else:
            alpha_ewma = 0.05
            self.dynamic_alpha_mean = float((1 - alpha_ewma) * self.dynamic_alpha_mean + alpha_ewma * mean_a)
            self.dynamic_alpha_std = float((1 - alpha_ewma) * self.dynamic_alpha_std + alpha_ewma * (std_a if std_a > 1e-5 else 1.0))
        self.alpha_count += 1

        # 8. 波动率 Z-Score 与 市场状态判定
        # 优先消费上游 FCS 已计算好的分钟事实，避免下游二次递推产生路径漂移。
        vol_z_dict = batch.get('vol_z_dict')
        if isinstance(vol_z_dict, dict) and vol_z_dict:
            for s_v, v_v in vol_z_dict.items():
                try:
                    self.cached_vol_z[s_v] = float(v_v)
                    if s_v in self.states:
                        self.states[s_v].last_vol_z = float(v_v)
                except Exception:
                    pass
        else:
            vol_z_dict = self._update_volatility_metrics(batch, symbols)

        # 🚀 [NEW] 计算全局市场状态 (Market Regime Guard)
        vixy_st = self.states.get('VIXY')
        global_regime_reversal_cnt = 0
        global_is_volatile_regime = False
        vixy_roc_5m = 0.0
        raw_vixy_volatile_regime = False
        if vixy_st:
            # [Fix] 增加 getattr 鲁棒性，应对 Config 尚未刷新但 Engine 已升级的情况
            regime_window = getattr(self.cfg, 'REGIME_WINDOW_MINS', 30)
            regime_thresh = getattr(self.cfg, 'REGIME_REVERSAL_PERCENT', 0.0015)
            global_regime_reversal_cnt = vixy_st.get_reversal_count(
                window_mins=regime_window,
                threshold=regime_thresh
            )
            regime_limit = getattr(self.cfg, 'REGIME_REVERSAL_THRESHOLD', 6)
            vixy_roc_5m, _, _, _ = vixy_st.get_strategy_metrics()
            vixy_roc_threshold = getattr(self.cfg, 'REGIME_VIXY_ROC_THRESHOLD', 0.003)
            raw_vixy_volatile_regime = (
                global_regime_reversal_cnt > regime_limit
                or vixy_roc_5m >= vixy_roc_threshold
            )
            require_neutral_index = getattr(self.cfg, 'REGIME_REQUIRE_NEUTRAL_INDEX_FOR_TIGHT_STOP', True)
            global_is_volatile_regime = (
                raw_vixy_volatile_regime
                and (not require_neutral_index or index_trend == 0)
            )
        self.last_global_regime_reversal_cnt = int(global_regime_reversal_cnt)
        prev_regime_flag = getattr(self, 'last_global_is_volatile_regime', None)
        self.last_global_is_volatile_regime = bool(global_is_volatile_regime)
        if prev_regime_flag is None or prev_regime_flag != self.last_global_is_volatile_regime:
            logger.info(
                f"🧭 [Regime Audit] VIXY_ROC_5M={vixy_roc_5m:.4%} | "
                f"VIXY_Reversals={int(global_regime_reversal_cnt)} | "
                f"IndexTrend={index_trend} | RawVol={raw_vixy_volatile_regime} | "
                f"Volatile={self.last_global_is_volatile_regime}"
            )
        
        # 9. 符号处理循环 (更新分钟指标 & 评估入场信号)
        entry_candidates = []
        metrics_batch = {
            'alpha_mean': mean_a,
            'alpha_std': std_a,
            'vol_z_dict': vol_z_dict,
            'curr_ts': curr_ts,
            'fast_vol': raw_vols
        }
        final_alphas_for_ic = np.zeros(len(symbols))
        
        for i, sym in enumerate(symbols):
            # 🚀 [Parity Audit] 仅针对 NVDA 打印深度对碰字段
            if sym == 'NVDA':
                raw_alpha_val = float(raw_alphas[i])
                logger.info(f"🔍 [Alpha-Audit] {sym} | Raw:{raw_alpha_val:.6f} | CS_Count:{len(valid_alphas)} | CS_Mean:{mean_a:.6f} | CS_Std:{std_a:.6f} | Mode:{self.states.get(sym).correction_mode if self.states.get(sym) else 'N/A'}")

            # 1. 优先获取期权合约数据 (用于指标计算时的对位缓存)
            st = self.states.get(sym)
            if not st and self.mode == 'backtest':
                st = self.states[sym] = SymbolState(sym)
            
            if st:
                if self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                    opt_data = self._get_opt_data_realtime(sym, st, ny_now, stock_prices[i], batch)
                else:
                    opt_data = self._get_opt_data_backtest(batch, i, sym, st)
            else:
                opt_data = {'has_feed': False}

            # 2. 准备分钟级指标 (此时已具备 opt_data)
            metrics = self._prep_symbol_metrics(i, sym, stock_prices, raw_alphas, opt_data, metrics_batch, use_precalc_feed)
            if not metrics: continue

            final_alphas_for_ic[i] = metrics['final_alpha']
            st = metrics['st']
            
            # [🆕 新增] 信号审计落盘
            if IS_BACKTEST:
                with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{curr_ts},{sym},{metrics['price']:.4f},{metrics['final_alpha']:.4f},{metrics['cs_alpha_z']:.4f},{self.cached_event_probs.get(sym, 0.0):.4f},{metrics['vol_z']:.4f},{metrics['roc_5m']:.6f}\n")

            entry_sig = await self._evaluate_symbol_signals(
                i, sym, metrics, opt_data, ny_now, curr_ts, spy_rocs[i], qqq_rocs[i], 
                getattr(self, 'last_is_zombie', False), index_trend, 
                global_regime_reversal_cnt=global_regime_reversal_cnt, # 🚀 [NEW]
                global_is_volatile_regime=global_is_volatile_regime,
                should_update_full=True
            )
            if entry_sig:
                logger.info(f"🎯 [Signal Found] {sym} | Reason: {entry_sig.get('reason')} | Alpha: {metrics['final_alpha']:.2f}")
                # S4 回测纯执行模式下，排序只由 alpha 和价格动量驱动，不再让 IV 参与二次筛选。
                if PURE_ALPHA_REPLAY:
                    raw_rank = abs(metrics['final_alpha'])
                else:
                    raw_rank = abs(metrics['final_alpha']) / max(0.1, st.last_valid_iv)
                mom_multiplier = 1.0 + abs(metrics['roc_5m']) * 100.0
                entry_candidates.append({
                    'sym': sym, 'sig': entry_sig, 'price': metrics['price'],
                    'curr_ts': curr_ts, 'batch_idx': i,
                    'alpha_strength': raw_rank * mom_multiplier
                })

        # 10. 截面排序与开仓执行
        min_symbols = 10
        max_entries = 3
        
        if entry_candidates and len(symbols) >= min_symbols:
            # 🚀 [PARITY] 统一开启 Alpha 排序，取强度最高的前 3 个入场
            entry_candidates.sort(key=lambda x: x['alpha_strength'], reverse=True)
            
            for cand in entry_candidates[:max_entries]:
                # 🚀 [PARITY] 彻底实现严格计数：入场前实时重计当前持仓+待开仓，超过 MAX 立即熔断
                current_active = sum(1 for s_s in self.states.values() 
                                    if s_s.position != 0 or (getattr(s_s, 'is_pending', False) and getattr(s_s, 'pending_action', '') == 'BUY'))
                
                if current_active >= self.cfg.MAX_POSITIONS: 
                    logger.info(f"🚫 [Batch-Limit] {cand['sym']} blocked | active_count: {current_active} >= {self.cfg.MAX_POSITIONS}")
                    break

                if not getattr(self, 'only_log_alpha', False):
                    await self._execute_entry(cand['sym'], cand['sig'], cand['price'], cand['curr_ts'], cand['batch_idx'])

        # 11. 同步与结算
        if IS_LIVEREPLAY:
             await self._report_pnl_status_logic(curr_ts, "LIVEREPLAY_SYNC")

        # 🚀 [Surgery 16] 回放同步锁解除逻辑
        # 发送心跳以防在没有信号的分钟内导致 Orchestrator 停滞
        from config import IS_SIMULATED
        if IS_SIMULATED:
            heartbeat_payload = {'ts': curr_ts, 'action': 'HEARTBEAT'}
            self.r.xadd('orch_trade_signals', {'data': ser.pack(heartbeat_payload)}, maxlen=5000)

        self._remember_frame(frame_id)
        await self._oms_sync(curr_ts, frame_id=frame_id)

    async def _process_exits(self, batch: dict):
        """[高频风控] 每一秒执行一次，检查持仓标的是否达到策略止损/止盈阈值"""
        symbols = batch['symbols']
        prices = batch['stock_price']
        curr_ts = getattr(self, 'last_curr_ts', time.time())
        from pytz import timezone
        ny_now = datetime.fromtimestamp(curr_ts, timezone('America/New_York'))
        
        # 1. 预判断是否有持仓，减少无谓计算
        has_any_pos = any(st.position != 0 for st in self.states.values())
        if not has_any_pos: return

        # 2. 遍历检查所有持仓标的
        for i, sym in enumerate(symbols):
            st = self.states.get(sym)
            if not st or st.position == 0: continue
            
            # 3. 获取期权合约数据 (适配实时或回测模式)
            if self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                opt_data = self._get_opt_data_realtime(sym, st, ny_now, prices[i], batch)
            else:
                opt_data = self._get_opt_data_backtest(batch, i, sym, st)

            # 🚀 [核心对齐] 出场检查使用的是：
            # 1. 当前秒级的最新价格 (prices[i])
            # 2. 上一分钟结算好的稳定指标快照 (get_strategy_metrics)
            roc_5m, macd, macd_slope, snap_roc = st.get_strategy_metrics()
            
            market_opt_price = opt_data['call_price'] if st.position == 1 else opt_data['put_price']
            ctx_bid = opt_data['call_bid'] if st.position == 1 else opt_data['put_bid']
            ctx_ask = opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']
            
            raw_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask, getattr(st, 'last_opt_price', 0.0))
            st.last_opt_price = raw_price
            
            metrics = {
                'price': float(prices[i]),
                'roc_5m': roc_5m, 'macd': macd, 'macd_slope': macd_slope,
                'snap_roc': snap_roc, 'st': st, 
                'final_alpha': st.last_alpha_z, 'vol_z': st.last_vol_z
            }
                
            # 4. 执行信号检查 (should_update_full=False 会跳过入场逻辑)
            await self._evaluate_symbol_signals(
                i, sym, metrics, opt_data, ny_now, curr_ts, 
                self.last_spy_roc_val, self.last_qqq_roc_val, 
                getattr(self, 'last_is_zombie', False), self.last_index_trend,
                global_regime_reversal_cnt=getattr(self, 'last_global_regime_reversal_cnt', 0),
                should_update_full=False
            )

    async def _oms_sync(self, curr_ts: float, frame_id: str = None):
        """[Consolidated] 统一处理 OMS 状态同步与卡位控制"""
        if not curr_ts: return

        latest_prices = {sym: getattr(st, 'last_opt_price', 0.0) 
                         for sym, st in self.states.items() 
                         if st.position != 0 and hasattr(st, 'last_opt_price')}
        
        # 1. 共享内存模式 (s4 Deterministic Bus)
        if getattr(self, 'use_shared_mem', False):
            sync_payload = {'action': 'SYNC', 'ts': curr_ts, 'prices': latest_prices}
            await self.signal_queue.put(sync_payload)
            # await self.signal_queue.join() # ❌ [BUG FIX] S4 driver consumes queue sequentially *after* process_tick returns. Calling join() here causes an infinite deadlock.
            if hasattr(self, 'r'):
                self._mark_orch_done(curr_ts, frame_id=frame_id)
            return

        # 2. Redis 模式 (s2 / Realtime)
        if hasattr(self, 'r'):
            from utils import serialization_utils as ser
            payload = {'action': 'SYNC', 'ts': curr_ts, 'prices': latest_prices}
            self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)
            
            # 🚀 [核心修复] 如果是纯推理模式 (only_log_alpha=True)，SE 就是终点站，直接更新打卡
            if getattr(self, 'only_log_alpha', False):
                self._mark_orch_done(curr_ts, frame_id=frame_id)
            else:
                # 否则需要等待执行引擎 (OMS) 真正撮合完毕
                wait_start = time.time()
                while True:
                    oms_done = self.r.get("sync:orch_done")
                    if oms_done and float(oms_done) >= curr_ts: break
                    await asyncio.sleep(0.005) 
                    if time.time() - wait_start > 30: 
                        logger.error(f"🚨 [SE Deadlock] OMS Sync Timeout at TS: {curr_ts}")
                        # 自救措施：强制更新同步位，防止 Driver 彻底卡死
                        self._mark_orch_done(curr_ts, frame_id=frame_id)
                        break

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

    # === Signal Emission ===
    
    # ================= 替换掉旧的 _emit_trade_signal =================
    async def _emit_trade_signal(self, action, sym, sig, stock_price, curr_ts, batch_idx):
        st = self.states[sym]
        original_position = st.position
        
        # 🚨 [真正的无状态设计]
        # SE 只负责发出意图，绝对不允许越权修改本地的 st.position 等真实资金状态。
        # 所有的状态变更必须由 OMS 撮合后通过 Redis 反向同步过来。
        
        if action == 'BUY':
            st.is_pending = True
            st.pending_action = 'BUY'  # 👈 [新增] 标记为正在等待买入确认
            
        elif action == 'SELL':
            st.is_pending = True
            st.pending_action = 'SELL'
            sig['original_position'] = original_position
            
            # 🚀 [对齐单引擎] 立即释放方向锁 (position)
            # 单引擎中 _execute_exit 同步执行，在符号循环中间就清零了 position。
            # 这里模拟相同行为：释放槽位让同一帧的后续符号可以开仓。
            st.position = 0
            
            # 🛑 绝对禁止在这里写 st.entry_price = 0 或 st.qty = 0！
            # 必须把这些带着成本记忆的原始数据，原封不动地留在共享内存里。
            # 留给 OMS (_execute_exit) 算完精准的利润后，由 OMS 亲自去清空！
            
            # 🛑 绝对禁止在这里写 st.cooldown_until = ...！
            # 止盈单不需要冷却！把冷却的判断权交还给 OMS 的止损记账模块！


        payload = {
            'ts': curr_ts,
            'symbol': sym,
            'action': action,
            'sig': sig,
            'stock_price': stock_price,
            'batch_idx': batch_idx,
            'logical_roi': getattr(st, 'max_roi', 0.0) if action == 'SELL' else 0.0,
            'prices': {sym: getattr(st, 'last_opt_price', 0.0)} 
        }
        
        if getattr(self, 'use_shared_mem', False):
            # 🚀 压入内存队列 (耗时 1 微秒)
            await self.signal_queue.put(payload)
        else:
            # 原有的 Redis 打包发送逻辑
            self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)
            logger.info(f"🚀 [SIGNAL_ENGINE] Published {action} signal for {sym} to Redis.")


    # ================= 新增：从 OMS 同步真实账本的方法 =================
    def _sync_state_from_oms(self):
        """[核心解耦] 每次评估前，从 OMS 获取最真实的仓位与成本"""
        if getattr(self, 'use_shared_mem', False):
            # 🚀 共享内存下，OMS 如果接受了订单，早已把 is_pending 设为 False。
            # 任何存留到现在的 is_pending，绝对是资金不足或风控触发的拒单。瞬间解锁！
            for sym, st in self.states.items():
                if getattr(st, 'is_pending', False):
                    st.is_pending = False
                    st.pending_action = ''
            return
            
        raw_states = self.r.hgetall("oms:live_positions")
        active_syms = set()
        
        # 1. 恢复 OMS 确认的持仓
        for sym_b, data_b in raw_states.items():
            sym = sym_b.decode('utf-8')
            
            # 特殊项：系统资金
            if sym == "____SYSTEM_CASH____":
                try:
                    cash_data = json.loads(data_b.decode('utf-8'))
                    self.mock_cash = cash_data['cash']
                except: pass
                continue

            if sym in self.states:
                import json
                data = json.loads(data_b.decode('utf-8'))
                st = self.states[sym]
                
                # 👇 [核心修复 1: 保护最大收益率不被 OMS 的陈旧数据覆盖]
                old_pos = st.position
                new_pos = data['pos']
                
                st.position = new_pos
                st.qty = data.get('qty', 0)
                st.entry_price = data.get('price', 0.0)
                st.entry_stock = data.get('stock', 0.0)
                st.entry_ts = data.get('ts', 0.0)
                
                # 🛑 记忆护城河：如果是新开仓，初始化 max_roi；如果已经在持仓，坚决保留 SE 本地追踪的 max_roi！
                if old_pos == 0 and new_pos != 0:
                    st.max_roi = 0.0
                elif new_pos == 0:
                    st.max_roi = -1.0
                    
                # 恢复其他的元数据 (这些在入场时固定，可以安全覆盖)
                st.entry_spy_roc = data.get('entry_spy_roc', 0.0)
                st.entry_index_trend = data.get('entry_index_trend', 0)
                st.entry_alpha_z = data.get('entry_alpha_z', 0.0)
                st.entry_iv = data.get('entry_iv', getattr(st, 'last_valid_iv', 0.0))
                
                # [🎯 核心修复] 释放锁
                st.is_pending = False
                st._pending_frames = 0
                active_syms.add(sym)
                
        # 2. 清理幻觉：如果 SE 以为有仓位，但 OMS 账本里没有，强制清零！
        for sym, st in self.states.items():
            if sym not in active_syms:
                # 如果是刚刚发出的 BUY 指令，OMS 还没来得及确认，我们容忍几帧
                if getattr(st, 'is_pending', False) and getattr(st, 'pending_action', '') == 'BUY':
                    from config import IS_SIMULATED
                    threshold = 0 if IS_SIMULATED else 3
                    st._pending_frames = getattr(st, '_pending_frames', 0) + 1
                    if st._pending_frames > threshold:
                        logger.debug(f"🚨 [SE_SYNC] Rejected by OMS for {sym}! Releasing lock.")
                        st.is_pending = False
                        st._pending_frames = 0
                        st.pending_action = ''
                else:
                    # 🚀 [核心修复：光速垃圾回收]
                    # 如果是 SELL 之后的确认，或者压根没发单 OMS 也不承认，立刻果断清空！
                    # 绝不允许幽灵状态占用 active_count！
                    st.position = 0
                    st.is_pending = False
                    st.entry_price = 0.0
                    st._pending_frames = 0
                    st.pending_action = ''


    async def _execute_entry(self, sym, sig, stock_price, curr_ts, batch_idx):
        return await self._emit_trade_signal('BUY', sym, sig, stock_price, curr_ts, batch_idx)

    async def _execute_exit(self, sym, sig, stock_price, curr_ts, batch_idx):
        return await self._emit_trade_signal('SELL', sym, sig, stock_price, curr_ts, batch_idx)

    async def _force_clear_all(self, batch, reason, custom_ts, custom_dt):
        logger.warning(f"🧹 [SIGNAL_ENGINE] {reason}: Requesting OMS to clear all positions.")
        for sym, st in self.states.items():
            if st.position != 0:
                sig = {'action': 'SELL', 'reason': reason, 'dir': -st.position, 'price': 0.1}
                await self._emit_trade_signal('SELL', sym, sig, 0.1, custom_ts, -1)

    # === Accounting Logic ===
    def _emit_trade_log(self, payload):
        """复用 Accounting 模块的日志发送逻辑，确保 Cash 和 Mode 标记一致"""
        return self.accounting._emit_trade_log(payload)
    async def _pnl_monitor_loop(self): pass
    async def _report_pnl_status_logic(self, timestamp, label="Summary"): pass
    def _generate_daily_analysis_report(self, report_date_str=None): pass
    def _process_exit_accounting(self, *args, **kwargs): pass

    # === State Manager Delegations ===
    def _recover_warmup_from_pg(self):
        return self.state_manager._recover_warmup_from_pg()

    def _recover_warmup_from_sqlite(self):
        return self.state_manager._recover_warmup_from_sqlite()

    # 在 SystemOrchestratorV8 类中添加以下方法


    # 修改 run 方法，加入心跳监测
    async def run(self):
        # [🔥 动态 DB 切换] 启动时根据 get_redis_db 重新刷新 Redis 连接，防止静态配置失效
        from config import get_redis_db
        target_db = get_redis_db()
        if self.r.connection_pool.connection_kwargs.get('db') != target_db:
            logger.info(f"🔄 Re-connecting Redis to DB {target_db} (Dynamic Mode Detection)")
            self.r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=target_db)
            # [🔥 关键修复] 在新 DB 中建立 Orchestrator 专属消费者组
            try:
                self.r.xgroup_create(REDIS_CFG['input_stream'], REDIS_CFG.get('orch_group'), mkstream=True, id='0')
            except Exception: pass
            
        logger.info(f"🔥 V8 Engine Started (Mode: {self.mode}, DB: {target_db})")
        from config import RUN_MODE, IS_SIMULATED, IS_LIVEREPLAY, IS_BACKTEST
        logger.info(f"🔍 DEBUG: RUN_MODE={RUN_MODE}, IS_SIMULATED={IS_SIMULATED}, IS_LIVEREPLAY={IS_LIVEREPLAY}, IS_BACKTEST={IS_BACKTEST}")
        
        if self.mode == 'backtest':
            # Replay 模式下优先从本地 SQLite 恢复 (针对 s2)
            self._recover_warmup_from_sqlite()
        elif self.mode == 'realtime' and not IS_SIMULATED:
            # 实盘模式下从 PG 恢复
            self._recover_warmup_from_pg()
        

        elif IS_LIVEREPLAY:
             logger.info(f"🎞️ Live Replay Mode: Skipping IBKR connection.")
        self._ensure_consumer_group()
        
        # [新增] 心跳计时器
        last_heartbeat = time.time()
        # 定义心跳超时阈值 (秒)
        HEARTBEAT_TIMEOUT = 120 
        
        # 🚀 [🔥 核心新增] 启动内存盈亏监视器，作为数据库之外的第二真相源
        # 在 LIVEREPLAY 模式下由 process_batch 同步驱动，不在后台扫描以保证确定性

            
        # 🛡️ [🔥 硬核新增] 启动防掉单真实对账器

        
        while True:
            try:
                 # 检查心跳 - 只在实盘且真实连接时才监控
                time_since_last = time.time() - last_heartbeat
                is_true_realtime = False # Signal engine operates without direct IBKR link
                
                if is_true_realtime and time_since_last > HEARTBEAT_TIMEOUT:
                    # [Fix] 心跳超时只在以下情况才触发紧急撤单:
                    # 1. 当前是纽约时间 RTH 交易时段 (9:30 - 16:00)
                    # 2. 有持仓需要保护
                    # 非交易时段 (开盘前/收盘后) 的数据断流是正常现象，不应误触发撤单
                    ny_now_hb = datetime.fromtimestamp(time.time(), tz=timezone('America/New_York'))
                    ny_time_hb = ny_now_hb.time()
                    is_rth = dt_time(9, 30) <= ny_time_hb <= dt_time(16, 0)
                    has_open_positions = any(st.position != 0 for st in self.states.values())
                    
                    if is_rth and has_open_positions:
                        logger.warning(f"⚠️ Heartbeat Lost for {time_since_last:.1f}s! (RTH + 持仓 → 触发紧急撤单)")
                        await self._emergency_cancel_all(reason="Data Stream Timeout")
                    elif is_rth:
                        logger.warning(f"⚠️ Heartbeat Lost for {time_since_last:.1f}s! (RTH 但无持仓 → 仅记录，不撤单)")
                    else:
                        # 非交易时段的正常断流，静默重置计时器
                        pass
                    # 无论如何重置计时器防止无限触发
                    last_heartbeat = time.time()

                streams_to_read = {REDIS_CFG['input_stream']: '>'}
                if self.mode != 'backtest' or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                    streams_to_read[STREAM_FUSED_MARKET] = '>'
                    
                resp = self.r.xreadgroup(REDIS_CFG.get('orch_group'), 'worker_v8', streams_to_read, count=50, block=100)
                last_heartbeat = time.time()
                
                if not resp: 
                    # logger.debug("Waiting for data...")
                    await asyncio.sleep(0.01)
                    continue
                
                #logger.info(f"📩 Orchestrator received {sum(len(msgs) for _, msgs in resp)} messages from {len(resp)} streams")
                current_ts_to_sync = None  # 记录本轮最后处理的时间戳

                for stream_name, msgs in resp:
                    stream_str = stream_name.decode('utf-8') if isinstance(stream_name, bytes) else stream_name
                    for msg_id, data in msgs:
                        payloads = []
                        # 🚀 统一装载逻辑，消灭格式差异带来的遗漏
                        if b'batch' in data:
                            payloads = ser.unpack(data[b'batch'])
                        elif b'data' in data:
                            payloads = [ser.unpack(data[b'data'])]
                        elif b'pickle' in data:
                            payloads = [ser.unpack(data[b'pickle'])]

                        for payload in payloads:
                            if stream_str == STREAM_FUSED_MARKET:
                                await self._process_fast_fused_tick(payload)
                            else:
                                await self.process_batch(payload)
                            # 提取逻辑时间戳
                            current_ts_to_sync = payload.get('ts', current_ts_to_sync)
                            
                        self.r.xack(stream_name, REDIS_CFG.get('orch_group'), msg_id)


            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    self._ensure_consumer_group()
                else:
                    logger.error(f"Redis Error: {e}")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Orchestrator Run Error: {e}", exc_info=True)
                await asyncio.sleep(1)
 
if __name__ == "__main__":
    pass
