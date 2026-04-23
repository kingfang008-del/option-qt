#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: signal_engine_v8.py
描述: [V8 AlphaEngine - 纯 Alpha 推理]
职责:
    - 特征抽取 + 模型推理 + 截面归一化 + Alpha 日志落库 (STREAM_TRADE_LOG action=ALPHA)
    - 向 OMS 发 ALPHA_FRAME (分钟帧) + SYNC (秒级同步点)
    - 不做任何交易决策/仓位/熔断 (全部由 OMS/StrategyCore 拥有)

备注:
    - 时间基准: 统一使用 process_batch 算出的 curr_ts (America/New_York).
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
import config
from config import PG_DB_URL
import threading
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from collections import deque
from pytz import timezone
from scipy.stats import norm

# [🆕 新增] 动态模型目录注入 (对齐 S4 环境)
import sys, os
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
if MODEL_DIR not in sys.path: sys.path.insert(0, MODEL_DIR)
DAO_DIR = os.path.join(os.path.dirname(__file__), 'DAO')
if DAO_DIR not in sys.path: sys.path.insert(0, DAO_DIR)

# 引入纯策略核心
from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION, StrategyConfig

# [Refactor] 引入模块化执行组件
# AlphaEngine 只保留 state 持久化 + ALPHA 日志落库, 策略/交易已下沉到 OMS (execution_engine_v8).
from orchestrator_state_manager import OrchestratorStateManager
from orchestrator_accounting import OrchestratorAccounting
from fcs_market_profile import build_market_profile

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
    NON_TRADABLE_SYMBOLS,
    ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS,
    INDEX_TREND_SYMBOLS,
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
        self.pending_action = ''      # [⏱️ 显式声明] 配合 last_buy_emit_ts 做 emit cooldown
        self.last_buy_emit_ts = 0.0    # [⏱️ 显式声明] 最近一次 emit BUY 的 curr_ts, 用于冷却判断

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
        # [🛡️ Defensive Coerce] 历史上曾出现 entry_spy_roc / entry_alpha_z 被写成 dict
        # 的污染数据, 一旦恢复到持仓字段, 后续 strategy.check_exit 会抛 TypeError 把
        # SE 整批 tick 的 alpha 陪葬 (表现: alpha_logs 连续 4-8 分钟整块断档).
        # 这里做 load 级兜底: 非数字值一律归零, 宁可失去一次保护逻辑, 也不阻断主循环.
        def _coerce_float(v, default=0.0):
            if isinstance(v, dict): return default
            try: return float(v)
            except (TypeError, ValueError): return default
        def _coerce_int(v, default=0):
            if isinstance(v, dict): return default
            try: return int(v)
            except (TypeError, ValueError): return default
        self.position = _coerce_int(data.get('position', 0))
        self.qty = _coerce_int(data.get('qty', 0))
        self.entry_price = _coerce_float(data.get('entry_price', 0.0))
        self.entry_stock = _coerce_float(data.get('entry_stock', 0.0))
        self.entry_ts = _coerce_float(data.get('entry_ts', 0.0))
        self.entry_spy_roc = _coerce_float(data.get('entry_spy_roc', 0.0))
        self.entry_index_trend = _coerce_int(data.get('entry_index_trend', 0))
        self.entry_alpha_z = _coerce_float(data.get('entry_alpha_z', 0.0))
        self.entry_iv = _coerce_float(data.get('entry_iv', 0.0))
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
    def __init__(self, symbols, mode='realtime', config_paths=None, model_paths=None, ibkr=None, shared_signal_queue=None):
        print(f"DEBUG: V8Orchestrator Initializing... Mode={mode}")
        self.mode = mode
        self.symbols = symbols

        # [🔥 Dependency Injection] 允许外部注入底座实例 (如 MockIBKR)
        self.ibkr = ibkr
        self.signal_queue = shared_signal_queue
        self.use_shared_mem = (shared_signal_queue is not None)
        # AlphaEngine only owns feature/alpha state. StrategyCore/Accounting/Trading
        # truth-of-source all live on OMS (execution_engine_v8). SE keeps self.cfg
        # locally only to drive normalization / regime / market-profile helpers.
        self.cfg = StrategyConfig()
        # [Back-compat shim] legacy backtest scripts (s4_run_historical_replay_s2.py /
        # _pg_1s.py 等) 仍然 `signal_engine.strategy.cfg` 读 cfg 后同步给 OMS.
        # 新代码不要再依赖 SE 上的 .strategy —— OMS 才是真策略执行方.
        from types import SimpleNamespace as _SNS
        self.strategy = _SNS(cfg=self.cfg)
        logger.info(f"🧭 Active strategy core moved to OMS: {ACTIVE_STRATEGY_CORE_VERSION}")
        self.states = {s: SymbolState(s, config=self.cfg) for s in symbols}
        self.symbol_states = self.states # Alias

        # [Refactor] 模块化组件初始化
        # state_manager: 负责 SE 自己的 SymbolState 持久化 / warmup recovery
        # accounting:    仅用 _emit_trade_log 把 ALPHA 事件写 STREAM_TRADE_LOG,
        #                其它交易/PnL 逻辑都在 OMS 上 (本文件不调用).
        self.state_manager = OrchestratorStateManager(self)
        self.accounting = OrchestratorAccounting(self)

        # Redis Init
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print("DEBUG: Redis Initialized.")

        # Global State Defaults
        # [已统一] 所有的策略、风控参数通过 self.cfg 访问
        self.last_date = None
        # mock_cash: 仅为 accounting._emit_trade_log 的 ALPHA payload 填 account_cash 字段;
        # 外部 backtest 脚本 (s4_run_historical_replay_s2_1m / s6_fast_memory_backtest) 会
        # 从 exec_engine.mock_cash 同步过来, 保证 ALPHA 日志上的现金跟 OMS 一致.
        if self.ibkr:
            if hasattr(self.ibkr, 'cash'):
                self.mock_cash = float(self.ibkr.cash)
            elif hasattr(self.ibkr, 'initial_capital'):
                self.mock_cash = float(self.ibkr.initial_capital)
        else:
            self.mock_cash = self.cfg.INITIAL_ACCOUNT
        self.index_opening_prices = {}
        self.last_save_time = 0

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
        self.is_high_freq = (self.mode == 'realtime')
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
        profile_name = os.environ.get("MARKET_PROFILE", "equity_us")
        warmup_len = int(os.environ.get("FCS_WARMUP_REQUIRED_LEN", "31"))
        self.market_profile = build_market_profile(
            profile_name,
            ny_tz=timezone('America/New_York'),
            warmup_required_len=warmup_len,
            non_tradable_symbols=NON_TRADABLE_SYMBOLS,
        )
        self.non_tradable_symbols = self.market_profile.get_non_tradable_set()
        self.alpha_norm_exclude_symbols = set(ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS) | self.non_tradable_symbols
        logger.info(f"🧭 Signal Market Profile: {self.market_profile.name}")

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
        mode_str = "✅ [Fair Price (Mid) + BSM 校准模式]"
        logger.info(f"📊 Orchestrator 启动! 当前价格计算模式: {mode_str}")

    # 退出确认 (_is_high_freq_tick / _should_confirm_exit / _confirm_high_freq_exit /
    # _reset_exit_confirmation) 已全部下沉到 OMS StrategyCore - SE 不再做 exit 判定.

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
        return float(self.market_profile.calc_effective_minutes(start_dt, end_dt))



    @staticmethod
    def _get_fair_market_price(base_price: float, bid: float, ask: float, prev_price: float = 0.0) -> float:
        """
        [NEW] 统一计算期权公允市价
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
            # 只有在进入真空（无买卖单）时，才触发 10% 偏离保护
            if prev_price > 0.01 and market_price > 0.01:
                if abs(market_price - prev_price) / prev_price > 0.10:
                    market_price = prev_price

        return market_price

    async def _process_fast_fused_tick(self, payload: dict):
        """AlphaEngine boundary: fused ticks are not allowed to trigger strategy exits."""
        return

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

                # 🚀 [Parity & Guard Logic]
                # 0: 原始模式 (不干预) | 1: 严格过滤 (总深度>100) | 2: 基准对齐 (强制100)
                from config import STRICT_LIQUIDITY_MODE
                if STRICT_LIQUIDITY_MODE == 2:
                    res['call_bid_size'] = 100.0
                    res['call_ask_size'] = 100.0
                    res['put_bid_size']  = 100.0
                    res['put_ask_size']  = 100.0

                res['has_feed'] = True
        except Exception as e:
            logger.warning(f"Failed to parse live option snapshot for {sym}: {e}")

        return res

    def _mark_orch_done(self, curr_ts: float, frame_id: str = None):
        if not hasattr(self, 'r'):
            return
        self.r.set("sync:orch_done", str(curr_ts))
        self.r.expire("sync:orch_done", 120)
        if frame_id:
            self.r.set("sync:orch_done_frame_id", str(frame_id))
            self.r.expire("sync:orch_done_frame_id", 120)
        try:
            self.r.hincrby("diag:se:counters", "orch_done", 1)
        except Exception:
            pass

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
            # 如果 batch 里根本没有期权数组，说明外壳脚本传错了模式。
            # 这里直接转去读取 Redis 的实时快照作为兜底。
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
        if sym in self.non_tradable_symbols:
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
        # 如果是实盘模式，或者环境变量强制开启了高频模式，则进入“升频执行，降频推理”
        parity_mode_1s = os.environ.get('REPLAY_1S_PARITY_MODE') == '1'
        is_high_freq = (
            not parity_mode_1s and
            (self.mode == 'realtime' or os.environ.get('FORCE_HIGH_FREQ') == '1')
        )
        # is_new_minute = batch.get('is_new_minute', True) # [REMOVED - Re-defined below with safe fallback]
        symbols = batch.get('symbols', [])
        stock_prices = batch.get('stock_price', [])
        frame_id = str(batch.get('frame_id')) if batch.get('frame_id') is not None else None

        # 🚀 [Refine] use_precalc_feed should depend on whether alpha is actually provided in the batch
        use_precalc_feed = 'alpha_score' in batch or 'precalc_alpha' in batch

        # 🧪 [Parity Fix] 如果 batch 里带了 event_prob，直接同步到缓存供 metrics 使用
        if 'event_prob' in batch:
            for idx_p, s_p in enumerate(symbols):
                 self.cached_event_probs[s_p] = float(batch['event_prob'][idx_p])
        raw_vols = batch.get('fast_vol')
        if raw_vols is None:
            raw_vols = np.zeros(len(symbols))

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

        # 分钟级回测模式：每一帧都是 should_update_full，确保计算不被缓存拦截。
        # 高频实盘模式：只有在 is_new_minute 为 True 时才是 should_update_full。
        # 🚀 [Parity Fix] 如果 batch 里没有传 is_new_minute (Standalone Pitcher)，则安全回退到 ts 跨越逻辑
        is_new_minute = batch.get('is_new_minute', is_new_min_crossing)

        if is_high_freq:
            # 🚨 [关键修复]：高频回放模式下，强制信任来自特征服务的结算信号！
            should_update_full = is_new_minute or (last_t == 0)

            # 🚀 [Parity Fix] 如果开启了 1s 物理对齐模式，则每一秒都强制进入全量评价流程 (对齐 S4)
            if parity_mode_1s:
                should_update_full = True
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

        # 2. AlphaEngine no longer owns exit decisions. Seconds-level ticks may still
        # refresh local feature caches, but StrategyCore/check_exit only runs on OMS.

        # ✅ [核心重构] 第二阶段：分钟级策略（仅在分钟跨越或非高频模式下执行）
        if not should_update_full:
            # 高频增量帧：SE 只驱动 SYNC 屏障, PnL 报告由 OMS 自己做.
            await self._oms_sync(curr_ts, frame_id=frame_id)
            return

        # 🚀 以下逻辑仅在分钟整点运行 🚀
        allow_dry_alpha_during_warmup = (os.environ.get('RUN_MODE', '').upper() == 'REALTIME_DRY')
        if not use_precalc_feed and not bool(batch.get('is_warmed_up', False)) and not allow_dry_alpha_during_warmup:
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
        elif allow_dry_alpha_during_warmup and not bool(batch.get('is_warmed_up', False)):
            if getattr(self, '_dry_warmup_bypass_log_count', 0) < 10:
                logger.info(
                    f"🧪 [SE-Dry-Warmup-Bypass] Continue alpha emission in REALTIME_DRY "
                    f"| real_history_len={batch.get('real_history_len', 0)}/{batch.get('warmup_required_len', 31)}"
                )
                self._dry_warmup_bypass_log_count = getattr(self, '_dry_warmup_bypass_log_count', 0) + 1

        # 3. 记录行情数据 (为 Alpha 日志准备时间戳)
        # 🚀 [Surgery 24] 逻辑时间戳对齐：如果是整分结算，强制用 ts (逻辑上代表的那一分钟)，否则用 log_ts (物理时间)
        self.current_log_ts = batch.get('ts' if is_new_minute else 'log_ts', curr_ts)
        current_date = ny_now.date()

        # 4. 主动强平与跨日清理
        close_h = self.cfg.CLOSE_HOUR
        close_m = self.cfg.CLOSE_MINUTE
        is_eod_time = (ny_now.hour == close_h and ny_now.minute >= close_m) or (ny_now.hour > close_h)

        if is_eod_time:
            last_clear_day = getattr(self, 'last_eod_clear_day', None)
            if last_clear_day != current_date:
                logger.warning(
                    f"⏰ [AlphaEngine] 到达策略指定平仓时间 {close_h}:{close_m}; "
                    "EOD 平仓由 OMS/StrategyCore 在 ALPHA_FRAME 中统一裁决。"
                )
                self.last_eod_clear_day = current_date

        if self.last_date is not None and current_date > self.last_date:
            logger.warning(
                "📅 [AlphaEngine] 检测到跨日；持仓清理/熔断重置/日终报告 统一交给 OMS/StrategyCore。"
            )
            # SE 仅重置 alpha 推理相关的当日基准; 交易/熔断/gate-trace 等 Redis 键由 OMS 日终处理.
            self.index_opening_prices = {}

        self.last_date = current_date

        # Global gate snapshots are now owned by OMS/StrategyCore. SE only publishes alpha facts.

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
        exclude_indices = {i for i, s in enumerate(symbols) if s in self.alpha_norm_exclude_symbols}
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
        use_upstream_volz = isinstance(vol_z_dict, dict) and bool(vol_z_dict)
        if use_upstream_volz:
            valid_vals = []
            for s_v, v_v in vol_z_dict.items():
                try:
                    fv = float(v_v)
                    if np.isfinite(fv):
                        valid_vals.append(abs(fv))
                    self.cached_vol_z[s_v] = fv
                    if s_v in self.states:
                        self.states[s_v].last_vol_z = fv
                except Exception:
                    pass
            # 质量兜底：若上游 vol_z_dict 退化为全 0（或全部无效），则本地重算，避免 alpha_logs 全量写 0。
            if (not valid_vals) or (max(valid_vals) < 1e-9):
                if getattr(self, '_volz_fallback_log_count', 0) < 10:
                    logger.warning("⚠️ [VolZ-Fallback] Upstream vol_z_dict is degenerate(all-zero/invalid), recomputing locally.")
                    self._volz_fallback_log_count = getattr(self, '_volz_fallback_log_count', 0) + 1
                vol_z_dict = self._update_volatility_metrics(batch, symbols)
        else:
            vol_z_dict = self._update_volatility_metrics(batch, symbols)

        # 🚀 [NEW] 计算全局市场状态 (Market Regime Guard)
        vixy_st = self.states.get('VIXY')
        global_regime_reversal_cnt = 0
        global_is_volatile_regime = False
        global_regime_score = 0.0
        global_regime_band = getattr(self, 'last_global_regime_band', 'calm')
        vixy_roc_5m = 0.0
        raw_vixy_volatile_regime = False
        raw_regime_band = 'calm'
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
            reversal_ratio = float(global_regime_reversal_cnt) / max(1.0, float(regime_limit))
            roc_ratio = float(max(vixy_roc_5m, 0.0)) / max(float(vixy_roc_threshold), 1e-9)
            global_regime_score = max(reversal_ratio, roc_ratio)
            mixed_score_threshold = getattr(self.cfg, 'REGIME_MIXED_SCORE_THRESHOLD', 0.60)
            volatile_score_threshold = getattr(self.cfg, 'REGIME_VOLATILE_SCORE_THRESHOLD', 1.00)
            if global_regime_score >= volatile_score_threshold:
                raw_regime_band = 'volatile'
            elif global_regime_score >= mixed_score_threshold:
                raw_regime_band = 'mixed'
            else:
                raw_regime_band = 'calm'
            raw_vixy_volatile_regime = (
                global_regime_reversal_cnt > regime_limit
                or vixy_roc_5m >= vixy_roc_threshold
            )
            require_neutral_index = getattr(
                self.cfg,
                'REGIME_REQUIRE_NEUTRAL_INDEX_FOR_ENTRY_GUARD',
                True,
            )
            global_is_volatile_regime = (
                raw_vixy_volatile_regime
                and (not require_neutral_index or index_trend == 0)
            )
        prev_raw_band = getattr(self, '_regime_raw_band_prev', None)
        prev_raw_count = getattr(self, '_regime_raw_band_count', 0)
        if raw_regime_band == prev_raw_band:
            raw_band_count = prev_raw_count + 1
        else:
            raw_band_count = 1
        self._regime_raw_band_prev = raw_regime_band
        self._regime_raw_band_count = raw_band_count

        confirmed_regime_band = getattr(self, 'last_global_regime_band', 'calm')
        enter_confirm_bars = max(1, int(getattr(self.cfg, 'REGIME_BAND_ENTER_CONFIRM_BARS', 2)))
        exit_confirm_bars = max(1, int(getattr(self.cfg, 'REGIME_BAND_EXIT_CONFIRM_BARS', 4)))
        if raw_regime_band != confirmed_regime_band:
            if raw_regime_band == 'calm':
                needed = exit_confirm_bars
            elif confirmed_regime_band == 'volatile' and raw_regime_band == 'mixed':
                needed = exit_confirm_bars
            else:
                needed = enter_confirm_bars
            if raw_band_count >= needed:
                confirmed_regime_band = raw_regime_band

        self.last_global_regime_reversal_cnt = int(global_regime_reversal_cnt)
        prev_regime_flag = getattr(self, 'last_global_is_volatile_regime', None)
        self.last_global_is_volatile_regime = bool(global_is_volatile_regime)
        prev_regime_band = getattr(self, 'last_global_regime_band', None)
        self.last_global_regime_band = str(confirmed_regime_band)
        self.last_global_regime_score = float(global_regime_score)
        if prev_regime_flag is None or prev_regime_flag != self.last_global_is_volatile_regime:
            logger.info(
                f"🧭 [Regime Audit] VIXY_ROC_5M={vixy_roc_5m:.4%} | "
                f"VIXY_Reversals={int(global_regime_reversal_cnt)} | "
                f"IndexTrend={index_trend} | RawVol={raw_vixy_volatile_regime} | "
                f"Volatile={self.last_global_is_volatile_regime}"
            )
        if prev_regime_band is None or prev_regime_band != self.last_global_regime_band:
            logger.info(
                f"🧭 [Regime Band] Score={global_regime_score:.2f} | "
                f"Raw={raw_regime_band}({raw_band_count}) | "
                f"Confirmed={self.last_global_regime_band}"
            )

        # 9. 符号处理循环 (更新分钟指标 & 评估入场信号)
        # 9. 符号处理循环 (更新分钟指标 & 评估入场信号)
        alpha_items = []
        metrics_batch = {
            'alpha_mean': mean_a,
            'alpha_std': std_a,
            'vol_z_dict': vol_z_dict,
            'curr_ts': curr_ts,
            'fast_vol': raw_vols,
            # 🚀 [日志修复] 补充提取时间戳，供 _prep_symbol_metrics 读取
            'alpha_label_ts': batch.get('alpha_label_ts', [0.0] * len(symbols)),
            'alpha_available_ts': batch.get('alpha_available_ts', [0.0] * len(symbols))
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
                if self.mode == 'realtime':
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

            # AlphaEngine keeps ALPHA logs, but never runs StrategyCore or emits
            # BUY/SELL. OMS consumes this canonical frame and decides with its
            # local position/cash truth.
            # Keep the legacy S4 decision alignment: at an exact minute boundary,
            # strategy evaluation used the previously latched option snapshot
            # while alpha logging still used the current batch snapshot.
            decision_opt_data = opt_data
            if int(curr_ts) % 60 == 0 and getattr(st, 'last_tick_opt_data', None) is not None:
                decision_opt_data = st.last_tick_opt_data

            alpha_log_iv = st.last_valid_iv
            if opt_data.get('has_feed'):
                c_iv = float(opt_data.get('call_iv', 0.0) or 0.0)
                p_iv = float(opt_data.get('put_iv', 0.0) or 0.0)
                alpha_log_iv = (c_iv + p_iv) / 2.0 if c_iv > 0.01 and p_iv > 0.01 else max(c_iv, p_iv, alpha_log_iv)
            self._emit_trade_log({
                'action': 'ALPHA',
                'ts': getattr(self, 'current_log_ts', curr_ts),
                'symbol': sym,
                'alpha': metrics['final_alpha'],
                'iv': alpha_log_iv,
                'price': metrics['price'],
                'vol_z': metrics['vol_z'],
                'event_prob': self.cached_event_probs.get(sym, 0.0),
                'index_trend': index_trend,
            })
            alpha_items.append({
                'symbol': sym,
                'batch_idx': i,
                'stock_price': metrics['price'],
                'alpha': metrics['final_alpha'],
                'cs_alpha_z': metrics['cs_alpha_z'],
                'vol_z': metrics['vol_z'],
                'roc_5m': metrics['roc_5m'],
                'macd': metrics['macd'],
                'macd_slope': metrics['macd_slope'],
                'snap_roc': metrics['snap_roc'],
                'event_prob': self.cached_event_probs.get(sym, 0.0),
                'is_ready': bool(getattr(st, 'warmup_complete', False)),
                'last_valid_iv': float(getattr(st, 'last_valid_iv', 0.5) or 0.5),
                'correction_mode': getattr(st, 'correction_mode', 'NORMAL'),
                'alpha_label_ts': metrics.get('alpha_label_ts', 0.0),
                'alpha_available_ts': metrics.get('alpha_available_ts', curr_ts),
                'opt_data': decision_opt_data,
            })
            st.last_tick_price = metrics['price']
            st.last_tick_opt_data = opt_data

        if alpha_items and not getattr(self, 'only_log_alpha', False):
            await self._publish_alpha_frame(
                curr_ts=curr_ts,
                frame_id=frame_id,
                symbols=symbols,
                alpha_items=alpha_items,
                index_trend=index_trend,
                spy_rocs=spy_rocs,
                qqq_rocs=qqq_rocs,
                is_zombie_market=getattr(self, 'last_is_zombie', False),
                global_regime_reversal_cnt=global_regime_reversal_cnt,
                global_is_volatile_regime=global_is_volatile_regime,
                global_regime_band=self.last_global_regime_band,
                global_regime_score=global_regime_score,
            )

        # 11. 同步与结算 — PnL 报告由 OMS 负责; SE 只发 heartbeat + SYNC 屏障.
        if IS_SIMULATED:
            heartbeat_payload = {'ts': curr_ts, 'action': 'HEARTBEAT'}
            self.r.xadd('orch_trade_signals', {'data': ser.pack(heartbeat_payload)}, maxlen=5000)

        self._remember_frame(frame_id)
        await self._oms_sync(curr_ts, frame_id=frame_id)

    async def _oms_sync(self, curr_ts: float, frame_id: str = None):
        """Send an execution barrier only; do not synchronize trading state.

        Redis-mode SE is Alpha Engine only. It must not infer active OMS
        positions or push position-derived prices back to OMS. Shared-memory
        replay may still pass latest prices because both sides share one state
        object and there is no Redis state authority involved.
        """
        if not curr_ts: return

        # 1. 共享内存模式 (s4 Deterministic Bus)
        if getattr(self, 'use_shared_mem', False):
            latest_prices = {sym: getattr(st, 'last_opt_price', 0.0)
                             for sym, st in self.states.items()
                             if st.position != 0 and hasattr(st, 'last_opt_price')}
            sync_payload = {'action': 'SYNC', 'ts': curr_ts, 'prices': latest_prices}
            await self.signal_queue.put(sync_payload)
            # await self.signal_queue.join() # ❌ [BUG FIX] S4 driver consumes queue sequentially *after* process_tick returns. Calling join() here causes an infinite deadlock.
            # only_log_alpha 下若未启 OMS，保留直写兜底，防止发球机等待超时
            if hasattr(self, 'r') and getattr(self, 'only_log_alpha', False):
                self._mark_orch_done(curr_ts, frame_id=frame_id)
            return

        # 2. Redis 模式 (s2 / Realtime)
        if hasattr(self, 'r'):
            from utils import serialization_utils as ser
            payload = {'action': 'SYNC', 'ts': curr_ts, 'prices': {}, 'frame_id': frame_id}
            self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)
            # 屏障由 OMS 在处理 SYNC 后回写，SE 不再提前 ACK。
            # 仅在 only_log_alpha 场景下保留 SE 兜底 ACK。
            if getattr(self, 'only_log_alpha', False):
                self._mark_orch_done(curr_ts, frame_id=frame_id)
            return
    def _prepare_ny_time(self, batch):
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

        # 周期性保存状态 (每60秒) — SE 只存自己的 alpha/feature state
        if time.time() - self.last_save_time > 60:
            self.save_state()
            self.last_save_time = time.time()
            self._publish_warmup_status()

        # 交易时间过滤
        if not self.market_profile.is_rth_minute(ny_now):
            has_position = any(st.position != 0 for st in self.states.values())
            if not has_position:
                return None, None

        return ny_now, curr_ts

    # === Signal Emission ===

    async def _publish_alpha_frame(
        self,
        curr_ts,
        frame_id,
        symbols,
        alpha_items,
        index_trend,
        spy_rocs,
        qqq_rocs,
        is_zombie_market,
        global_regime_reversal_cnt,
        global_is_volatile_regime,
        global_regime_band,
        global_regime_score,
    ):
        """Publish one canonical minute alpha frame for OMS-side strategy.

        This is the new architecture boundary: SignalEngineV8 is an AlphaEngine,
        so it emits facts only. OMS owns StrategyCore and all BUY/SELL decisions.
        """
        payload = {
            'source': 'alpha_engine_v8',
            'action': 'ALPHA_FRAME',
            'ts': curr_ts,
            'frame_id': frame_id,
            'symbols': list(symbols or []),
            'items': alpha_items,
            'index_trend': int(index_trend or 0),
            'spy_roc_5min': [float(x) for x in (list(spy_rocs) if spy_rocs is not None else [])],
            'qqq_roc_5min': [float(x) for x in (list(qqq_rocs) if qqq_rocs is not None else [])],
            'is_zombie_market': bool(is_zombie_market),
            'global_regime_reversal_cnt': int(global_regime_reversal_cnt or 0),
            'global_is_volatile_regime': bool(global_is_volatile_regime),
            'global_regime_band': str(global_regime_band or 'calm'),
            'global_regime_score': float(global_regime_score or 0.0),
        }
        if getattr(self, 'use_shared_mem', False):
            await self.signal_queue.put(payload)
        else:
            self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)
        logger.info(
            f"📡 [ALPHA_ENGINE] Published ALPHA_FRAME ts={curr_ts:.0f} "
            f"frame_id={frame_id} symbols={len(alpha_items)}"
        )

    def _emit_trade_log(self, payload):
        """复用 Accounting 模块的日志发送逻辑, 仅用于 ALPHA 日志 (SE 不产生 BUY/SELL)."""
        return self.accounting._emit_trade_log(payload)

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
        from config import RUN_MODE, IS_SIMULATED, IS_BACKTEST
        logger.info(f"🔍 DEBUG: RUN_MODE={RUN_MODE}, IS_SIMULATED={IS_SIMULATED}, IS_BACKTEST={IS_BACKTEST}")

        # [Config Snapshot] strategy_config0 快照由 OMS (ExecutionEngineV8.run) 负责广播,
        # 它才持有真正生效的 self.strategy.cfg 实例; SE 不再参与.

        if self.mode == 'backtest':
            # Replay 模式下优先从本地 SQLite 恢复 (针对 s2)
            self._recover_warmup_from_sqlite()
        elif self.mode == 'realtime' and not IS_SIMULATED:
            # 实盘模式下从 PG 恢复
            self._recover_warmup_from_pg()

        self._ensure_consumer_group()

        # [新增] 心跳计时器
        last_heartbeat = time.time()
        last_stats_log = time.time()
        stats = {'msgs': 0, 'payloads': 0, 'sync_calls': 0}
        # 定义心跳超时阈值 (秒)
        HEARTBEAT_TIMEOUT = 120

        # 🚀 [🔥 核心新增] 启动内存盈亏监视器，作为数据库之外的第二真相源


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
                    is_rth = bool(self.market_profile.is_rth_minute(ny_now_hb))
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
                if self.mode != 'backtest':
                    streams_to_read[STREAM_FUSED_MARKET] = '>'

                resp = self.r.xreadgroup(REDIS_CFG.get('orch_group'), 'worker_v8', streams_to_read, count=50, block=100)
                last_heartbeat = time.time()

                if not resp:
                    # logger.debug("Waiting for data...")
                    await asyncio.sleep(0.01)
                    continue

                #logger.info(f"📩 Orchestrator received {sum(len(msgs) for _, msgs in resp)} messages from {len(resp)} streams")
                current_ts_to_sync = None  # 记录本轮最后处理的时间戳
                current_frame_id = None    # 记录本轮的帧 ID
                stats['msgs'] += sum(len(msgs) for _, msgs in resp)

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
                            stats['payloads'] += 1
                            if stream_str == STREAM_FUSED_MARKET:
                                await self._process_fast_fused_tick(payload)
                            else:
                                await self.process_batch(payload)
                            # 提取逻辑时间戳与帧ID
                            current_ts_to_sync = payload.get('ts', current_ts_to_sync)
                            current_frame_id = payload.get('frame_id', current_frame_id)

                        self.r.xack(stream_name, REDIS_CFG.get('orch_group'), msg_id)

                # 🚀 [核心修复] 在处理完本轮所有消息后，统一发起一次同步锁
                # 只有发起 SYNC，OMS 才会更新 sync:orch_done，发球机才能继续推进
                if current_ts_to_sync:
                    await self._oms_sync(current_ts_to_sync, frame_id=current_frame_id)
                    stats['sync_calls'] += 1

                if time.time() - last_stats_log >= 60:
                    logger.info(
                        f"📊 [SE-Stats] 60s msgs={stats['msgs']} payloads={stats['payloads']} sync_calls={stats['sync_calls']}"
                    )
                    stats = {'msgs': 0, 'payloads': 0, 'sync_calls': 0}
                    last_stats_log = time.time()



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
