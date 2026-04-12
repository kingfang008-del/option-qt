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
import pickle
import logging
import json
import numpy as np
import time
import re
import pandas as pd
import os
import sys
# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import serialization_utils as ser
import psycopg2
from config import PG_DB_URL
import threading
import os
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from collections import deque
from pytz import timezone
from scipy.stats import norm 

# 引入纯策略核心
from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION, StrategyCore, StrategyConfig

# [Refactor] 引入模块化执行组件
from orchestrator_state_manager import OrchestratorStateManager
from orchestrator_accounting import OrchestratorAccounting
from orchestrator_execution import OrchestratorExecution
from orchestrator_reconciler import OrchestratorReconciler

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
    IS_LIVEREPLAY,
    IS_BACKTEST,
    IS_SIMULATED
)

from trading_tft_stock_embed import AdvancedAlphaNet
#from train_fast_channel_microstructure import FastMicrostructureModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s')
logger = logging.getLogger("V8_Orchestrator")

# [Fix] 显式添加 FileHandler 确保写入文件
from config import LOG_DIR
file_handler = logging.FileHandler(LOG_DIR / "Orchestrator.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - [V8_Orch] - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

 
# Log Stream Key
from config import (
    REDIS_CFG,                  # [🔥 新增] 统一导入 Redis 配置
    STREAM_TRADE_LOG,           
    STREAM_FUSED_MARKET,        
    TRADING_ENABLED)


RISK_FREE_RATE = 0.045

# 资金管理与风控参数 — 统一从 config.py 读取，不再本地重复定义
INITIAL_ACCOUNT = 50000.0
# MAX_POSITIONS, POSITION_RATIO, MAX_TRADE_CAP, GLOBAL_EXPOSURE_LIMIT, COMMISSION_PER_CONTRACT
# 均已在上方 from config import ... 中引入

# Alpha 反转参数
ROLLING_WINDOW = 30
CORR_THRESHOLD = -0.1

# 冷却参数
COOLDOWN_MINUTES = 15

# [联动] 实盘平仓防滑点开关: TRADING_ENABLED=True → LMT (防做市商收割)
#                           TRADING_ENABLED=False → MKT (极速回测)
EXIT_ORDER_TYPE = 'LMT' if TRADING_ENABLED else 'MKT'

 

# =========================================================
# 💰 交易成本模拟 (由 config.py 统一管理)
# =========================================================
# COMMISSION_PER_CONTRACT 已从 config.py 导入


# [BSM 定价函数已废弃]
# def black_scholes_price(S, K, T, r, sigma, option_type='call'):
#     ...

class SymbolState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.prices = deque(maxlen=60)
        self.alpha_history = deque(maxlen=ROLLING_WINDOW + 10)
        self.pct_history = deque(maxlen=ROLLING_WINDOW + 10)
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
        self.last_opt_price = 0.0 # [NEW] 记录最新期权采样价，用于会计盈亏计算
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

        # [🔥 新增] 记录上一分钟的 MACD 柱子，用于计算斜率(导数)
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

    def update_indicators(self, price, raw_alpha_score, is_new_minute=True):
        price = float(price)
        raw_alpha_score = float(raw_alpha_score)
        prev_price = self.prices[-1] if len(self.prices) > 0 else price
        pct_change = 0.0
        if prev_price > 0: pct_change = (price - prev_price) / prev_price

        # 🚀 [核心修复] 只有在跨越逻辑分钟时，才更新滚动历史，保证对齐 1m 离线数据
        if is_new_minute:
            self.prices.append(price)
            self.alpha_history.append(raw_alpha_score)
            self.pct_history.append(pct_change)
        else:
            # 在同一分钟内的高频 Tick，仅更新当前瞬时价格，不推进滚动序列
            if len(self.prices) > 0:
                self.prices[-1] = price
            else:
                self.prices.append(price)

        # 基于分钟级采样序列计算 5min ROC
        roc_5m = 0.0
        if len(self.prices) >= 6:
            prev_5m = self.prices[-6]
            if prev_5m > 0: roc_5m = (price - prev_5m) / prev_5m

        # ==========================================================
        # 🛡️ [修复] 修正 Alpha 反转震荡漏洞，加入迟滞区间 (Hysteresis)
        # 且清理了原来重复写了两遍的代码块
        # ==========================================================
        if len(self.alpha_history) >= ROLLING_WINDOW:
            self.warmup_complete = True  # 标记预热完成
            
            alphas = list(self.alpha_history)[-ROLLING_WINDOW:]
            pcts = list(self.pct_history)[-ROLLING_WINDOW:]
            
            if np.std(alphas) > 1e-6 and np.std(pcts) > 1e-6:
                corr = np.corrcoef(alphas, pcts)[0, 1]
                
                # 设置 0.05 的迟滞缓冲区，防止在 -0.1 附近频繁横跳翻转 Alpha
                if self.correction_mode == "NORMAL" and corr < (CORR_THRESHOLD - 0.05):
                    self.correction_mode = "INVERT"
                elif self.correction_mode == "INVERT" and corr > (CORR_THRESHOLD + 0.05):
                    self.correction_mode = "NORMAL"
            else:
                # 波动过小时，保持原状态，绝不盲目切换回 NORMAL
                pass
        else:
            self.warmup_complete = False
            self.correction_mode = "NORMAL" # 预热期默认Normal，但不应交易

        # EMA / MACD 计算逻辑保持不变
        if self.ema_fast_val is None:
            self.ema_fast_val = price; self.ema_slow_val = price; self.dea_val = 0.0
        else:
            self.ema_fast_val = float(price * self.k_fast + self.ema_fast_val * (1 - self.k_fast))
            self.ema_slow_val = float(price * self.k_slow + self.ema_slow_val * (1 - self.k_slow))
            dif = self.ema_fast_val - self.ema_slow_val
            self.dea_val = float(dif * self.k_sig + self.dea_val * (1 - self.k_sig))
            
        macd_hist = float((self.ema_fast_val - self.ema_slow_val) - self.dea_val)
        
        # 计算 MACD 柱子的变化率 (斜率/动量导数)
        macd_hist_slope = macd_hist - self.prev_macd_hist
        self.prev_macd_hist = macd_hist
        
        return roc_5m, macd_hist, macd_hist_slope, pct_change

    def get_reversal_count(self, window_mins=30, threshold=0.001):
        """
        [NEW] 计算指定分钟窗口内的反转次数
        定义: 当前收益率与上一分钟收益率符号相反，且绝对值均超过阈值
        """
        if len(self.pct_history) < 3: return 0
        
        # 仅截取最近 window_mins 分钟的收益率记录
        recent_pcts = list(self.pct_history)[-window_mins:]
        count = 0
        for i in range(1, len(recent_pcts)):
            p1, p2 = recent_pcts[i-1], recent_pcts[i]
            # 反转判定
            if p1 * p2 < 0 and abs(p1) >= threshold and abs(p2) >= threshold:
                count += 1
        return count

        
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
            'dea_val': self.dea_val,
            'last_opt_price': self.last_opt_price
        }

    def from_dict(self, data):
        """从字典恢复状态 (含 Buffer)"""
        self.position = data.get('position', 0)
        self.qty = data.get('qty', 0)
        self.entry_price = data.get('entry_price', 0.0)
        self.entry_stock = data.get('entry_stock', 0.0)
        self.last_opt_price = data.get('last_opt_price', 0.0)
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

class V8Orchestrator:
    def __init__(self, symbols, mode='realtime', config_paths=None, model_paths=None):
        print(f"DEBUG: V8Orchestrator Initializing... Mode={mode}")
        self.mode = mode
        self.symbols = symbols
        self.states = {s: SymbolState(s) for s in symbols}
        self.symbol_states = self.states # Alias
        self.strategy = StrategyCore(StrategyConfig())
        self.cfg = self.strategy.cfg  # [Fix] 显式暴露配置给组件
        logger.info(f"🧭 Active strategy core: {ACTIVE_STRATEGY_CORE_VERSION}")
        
        # [Refactor] 模块化组件初始化
        self.state_manager = OrchestratorStateManager(self)
        self.accounting = OrchestratorAccounting(self)
        self.execution = OrchestratorExecution(self)
        self.reconciler = OrchestratorReconciler(self)
        
        # Redis Init
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print("DEBUG: Redis Initialized.")

        # Global State Defaults
        # [新增] 全局连败熔断器
        self.last_date = None
        self.mock_cash = INITIAL_ACCOUNT
        self.index_opening_prices = {}     # [NEW] 记录当日 SPY/QQQ 开盘价
        self.consecutive_stop_losses = 0  # 连续止损计数 
        self.global_cooldown_until = 0      # 全局冷却截止时间戳
        self.CIRCUIT_BREAKER_THRESHOLD = 3 # 连续 N 笔止损后触发
        self.CIRCUIT_BREAKER_MINUTES = 30  # 熔断暂停时间(分钟)
        self.MIN_OPTION_PRICE = 2.5        # 最低期权进场价格
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

        # =========================================================
        # 🚀 [新增] 动态 Alpha 归一化追踪器 (Dynamic Alpha Tracker)
        # =========================================================
        self.dynamic_alpha_mean = 0.0
        self.dynamic_alpha_std = 1.0
        self.alpha_count = 0

        # 赋予一个先验初始值 (防止冷启动期间的极端缩放)
        self.dynamic_vol_mean = 0.0739 
        self.dynamic_vol_std = 0.1106


       
        
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
        
        # 3. Mode-Specific Init (IBKR only)
        # 3. Mode-Specific Init (IBKR only)
        if mode == 'realtime' and not IS_SIMULATED:
            logger.info("🚀 V8 Live Mode Init (Real IBKR)...")
            if IBKRConnectorFinal: 
                self.ibkr = IBKRConnectorFinal(client_id=999)
            else: 
                raise ImportError("IBKRConnectorFinal missing!")
        else:
            logger.info("🎞️ V8 Backtest/Replay Mode Init (Mock IBKR)...")
            self.ibkr = MockIBKRHistorical()
        
        # [Shadow Validation] Alpha Only Mode
        from config import ONLY_LOG_ALPHA
        self.only_log_alpha = ONLY_LOG_ALPHA
        if self.only_log_alpha:
            logger.info("🕵️ Shadow System: ONLY_LOG_ALPHA mode enabled. Trading is strictly disabled.")

        # [NEW] 模式启动声明
        mode_str = "✅ [Bid/Ask + BSM 校准模式]" if USE_BID_ASK_PRICING else "⚠️ [成交价模式 (Transaction Price Mode)]"
        logger.info(f"📊 Orchestrator 启动! 当前价格计算模式: {mode_str}")

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
        
        # 简单实现: 如果跨天，只计算每交易日 6.5 小时 (390分钟)
        # 如果是同一天，直接减
        # 为了精确，需要考虑 weekends
        # 这里简化：按日历时间差剔除周末和非交易时间
        
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
    def _get_fair_market_price(base_price: float, bid: float, ask: float, last_price: float = 0.0) -> float:
        """
        [NEW] 统一计算期权公允市价
        1. 如果 USE_BID_ASK_PRICING 为 False, 始终返回成交价 (base_price).
        2. 如果为 True, 优先使用 Bid/Ask 中间价.
        3. 如果 Bid/Ask 缺失, 使用 last_price 作为锚点, 否则 fallback to base_price.
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
        elif last_price > 0.01:
            market_price = last_price
        else:
            market_price = base_price

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
                        'symbol': sym,
                        'ts': payload.get('ts'),
                        'buckets': opt_buckets,
                        'contracts': opt_contracts
                    }
                    self.r.hset(HASH_OPTION_SNAPSHOT, sym, pickle.dumps(snap_payload))
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
            'snap_roc': getattr(st, 'last_snap_roc', 0.0)
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
        import pickle
        
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
            # [兜底降级] 如果 batch 里没有，才去 Redis 查（实盘单股快照可能走这里）
            raw_snap = self.r.hget(HASH_OPTION_SNAPSHOT, sym)
            if not raw_snap: return res
            try:
                snap_data = pickle.loads(raw_snap)
                buckets = snap_data.get('buckets', [])
                contracts = snap_data.get('contracts', [])
                logger.debug(f"📡 [Fallback] Using Redis option data for {sym}")
            except Exception as e:
                logger.warning(f"Failed to parse redis option snapshot for {sym}: {e}")
                return res

        try:
            # 此处不再需要重复 pickle.loads(raw_snap)，因为上面已经处理过了
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
                res['call_price'] = self._get_fair_market_price(_c_last, _c_bid, _c_ask, _c_last)
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
                res['put_price']  = self._get_fair_market_price(_p_last, _p_bid, _p_ask, _p_last)
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
        
        if 'feed_call_price' in batch or 'cheat_call' in batch or 'opt_snapshot' in batch:
            res['has_feed'] = True
            
            # 🔥 [Multi-Gen Support] 同时支持 列表索引格式(Legacy) 和 扁平字典格式(New)
            if 'opt_snapshot' in batch:
                snap = batch['opt_snapshot'][i]
                if isinstance(snap, dict):
                    # 模式 A: 列表索引格式 (Buckets List - 1月2日风格)
                    if 'buckets' in snap and len(snap['buckets']) > 0:
                        contracts = snap.get('contracts', [])
                        buckets = snap['buckets']
                        for idx_b, c_id in enumerate(contracts):
                            b_data = buckets[idx_b]
                            if 'C' in c_id and res['call_price'] == 0:
                                res['call_price'] = _safe_f(b_data[0])
                                res['call_iv']    = _safe_f(b_data[7])
                                res['call_bid']   = _safe_f(b_data[8])
                                res['call_ask']   = _safe_f(b_data[9])
                                res['call_id']    = c_id
                                res['call_k']     = _safe_f(b_data[6])
                            if 'P' in c_id and res['put_price'] == 0:
                                res['put_price']  = _safe_f(b_data[0])
                                res['put_iv']     = _safe_f(b_data[7])
                                res['put_bid']    = _safe_f(b_data[8])
                                res['put_ask']    = _safe_f(b_data[9])
                                res['put_id']     = c_id
                                res['put_k']      = _safe_f(b_data[6])
                        if res['call_bid'] <= 0 and res['put_bid'] <= 0: res['has_feed'] = False
                    
                    # 模式 B: 扁平字典格式 (Flat Dict - 1月8日风格)
                    elif 'call_bid' in snap:
                        if snap.get('has_feed', True):
                            res['call_price'] = _safe_f(snap.get('call_price', 0))
                            res['put_price']  = _safe_f(snap.get('put_price', 0))
                            res['call_bid']   = _safe_f(snap.get('call_bid', res['call_price']))
                            res['call_ask']   = _safe_f(snap.get('call_ask', res['call_price']))
                            res['put_bid']    = _safe_f(snap.get('put_bid', res['put_price']))
                            res['put_ask']    = _safe_f(snap.get('put_ask', res['put_price']))
                            res['call_iv']    = _safe_f(snap.get('call_iv', 0))
                            res['put_iv']     = _safe_f(snap.get('put_iv', 0))
                            res['call_id']    = snap.get('call_id', f"{sym}_C_MOCK")
                        else:
                            res['has_feed'] = False
                    else:
                        res['has_feed'] = False
                else:
                    res['has_feed'] = False
            else:
                # 兼容原始明细列模式 (数组格式)
                res['call_price'] = _safe_f(batch.get('feed_call_price', batch.get('cheat_call', [0]*len(batch['symbols'])))[i])
                res['put_price']  = _safe_f(batch.get('feed_put_price', batch.get('cheat_put', [0]*len(batch['symbols'])))[i])
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

    def _prep_symbol_metrics(self, i, sym, stock_prices, raw_alphas, raw_vols, use_precalc_feed, metrics_batch={}):
        """[Refactor] 统一更新指标与 Alpha 缩放"""
        if sym in NON_TRADABLE_SYMBOLS:
            return None
        if sym not in self.states:
            if self.mode == 'backtest': self.states[sym] = SymbolState(sym)
            else: return None

        st = self.states[sym]
        price = float(stock_prices[i])
        raw_alpha_val = float(raw_alphas[i])
        is_new_minute = metrics_batch.get('is_new_minute', True)
        
        # 指标更新 (传入时间对齐标志)
        roc_5m, macd, macd_slope, snap_roc = st.update_indicators(price, raw_alpha_val, is_new_minute=is_new_minute)
        
        # [🔥 新增] 暂存 snap_roc 供 evaluate 使用
        st.last_snap_roc = snap_roc

        # Alpha & Vol 缩放
        if use_precalc_feed:
            # 🚀 [回测/预计算模式] 直接透传预处理好的 Z-Score
            alpha_z = float(raw_alpha_val)
            vol_z = float(raw_vols[i])
        else:
            # 📡 实盘模式：[对齐 S3 离线逻辑] 使用当前截面 (Current Batch) 的均值/标差进行归一化
            # 彻底抛弃 EWMA 导致的平滑滞后，让实盘信号强度与回测完全一致。
            batch_alpha_mean = metrics_batch.get('alpha_mean', self.dynamic_alpha_mean)
            batch_alpha_std  = metrics_batch.get('alpha_std', self.dynamic_alpha_std)
            
            alpha_z = float((raw_alpha_val - batch_alpha_mean) / (batch_alpha_std + 1e-6))
            alpha_z = max(-5.0, min(5.0, alpha_z)) 
            
            # 🚀 [放宽对齐回测] 抛弃实盘的截面 Vol Z-Score，改用前面算好的独立时序 EWMA vol_z
            vol_z_dict = metrics_batch.get('vol_z_dict', {})
            if sym in vol_z_dict:
                vol_z = float(vol_z_dict[sym])
            else:
                batch_vol_mean   = metrics_batch.get('vol_mean', getattr(self, 'dynamic_vol_mean', 0.0))
                batch_vol_std    = metrics_batch.get('vol_std', getattr(self, 'dynamic_vol_std', 1.0))
                vol_z = float((float(raw_vols[i]) - batch_vol_mean) / (batch_vol_std + 1e-6))
            vol_z = max(-5.0, min(5.0, vol_z))

        final_alpha = float(-alpha_z if st.correction_mode == "INVERT" else alpha_z)
        st.prev_alpha_z = float(st.last_alpha_z)
        st.last_alpha_z = final_alpha
        
        # 缓存供 5s tick 推流使用
        st.last_spy_roc = 0.0 # Will be updated in main loop
        st.last_qqq_roc = 0.0
        st.last_macd_hist = macd
        st.last_macd_hist_slope = macd_slope

        return {
            'st': st, 'price': price, 'alpha_z': alpha_z, 'vol_z': vol_z, 
            'final_alpha': final_alpha, 'roc_5m': roc_5m, 'macd': macd, 'macd_slope': macd_slope, 'snap_roc': snap_roc
        }

    async def _evaluate_symbol_signals(self, i, sym, metrics, opt_data, ny_now, curr_ts, spy_roc, qqq_roc, is_zombie_market, index_trend=0, regime_reversal_count=0):
        """[Refactor] 核心策略评价 logic (平仓与开仓信号收集) - 修复空仓不交易BUG"""
        from config import USE_BID_ASK_PRICING
        st = metrics['st']
        price = metrics['price']
        final_alpha = metrics['final_alpha']
        
        # 1. 更新 IV 状态：消除均值污染，采用流动性择优法则
        if opt_data['has_feed']:
            if st.position == 1: 
                curr_iv = opt_data['call_iv']
            elif st.position == -1: 
                curr_iv = opt_data['put_iv']
            else: 
                # 🚀 [对齐修复] 强制使用均值 IV，确保与 1m 基准 100% 对位
                if opt_data.get('call_iv', 0) > 0 and opt_data.get('put_iv', 0) > 0:
                    curr_iv = (opt_data['call_iv'] + opt_data['put_iv']) / 2.0
                elif opt_data.get('call_iv', 0) > 0:
                    curr_iv = opt_data['call_iv']
                else:
                    curr_iv = opt_data['put_iv']
                    
            if curr_iv > 0.01: st.last_valid_iv = curr_iv

        # 🚀 [终极修复]：动态方向推断 (Dynamic Direction Inference)
        # 如果空仓，我们利用 Alpha 的方向预判策略想看哪个盘口，避免传入 0.0 导致策略的风险校验拒单！
        eval_dir = st.position if st.position != 0 else (1 if final_alpha > 0 else -1)
        
        ctx_bid = opt_data['call_bid'] if eval_dir == 1 else opt_data['put_bid']
        ctx_ask = opt_data['call_ask'] if eval_dir == 1 else opt_data['put_ask']
        market_opt_price = opt_data['call_price'] if eval_dir == 1 else opt_data['put_price']
        
        # 计算 Context 中的公允价
        ctx_curr_price = 0.0
        if opt_data['has_feed']:
            ctx_curr_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask)
        elif st.position != 0:
            ctx_curr_price = max(st.entry_price, 0.01)

        # 1.5 市场状态识别 (Regime Detection) - 已在 process_batch 中预先计算

        # 2. 构建 Context
        ctx = {
            'symbol': sym, 'time': ny_now, 'curr_ts': curr_ts, 'price': price,
            'alpha_z': final_alpha, 'vol_z': metrics['vol_z'], 'stock_roc': metrics['roc_5m'],
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
            'regime_reversal_count': regime_reversal_count
        }
        
        # 补齐 Spread Divergence 与 ROI 更新 (仅持仓时进行计算，防止污染空仓环境)
        if st.position != 0:
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

        # Alpha Log
        self._emit_trade_log({
            'action': 'ALPHA', 'ts': curr_ts, 'symbol': sym,
            'alpha': final_alpha, 'iv': st.last_valid_iv, 'price': price, 'vol_z': metrics['vol_z'],
            'index_trend': index_trend 
        })

        # 3. 执行平仓
        if st.position != 0:
            if curr_ts - st.entry_ts < 59.0:
                return None

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
        if curr_ts < self.global_cooldown_until or is_zombie_market: return None
        
        no_entry_h = self.strategy.cfg.NO_ENTRY_HOUR
        no_entry_m = self.strategy.cfg.NO_ENTRY_MINUTE
        if ny_now.time() >= dt_time(no_entry_h, no_entry_m): return None
        
        entry_sig = self.strategy.decide_entry(ctx)
        if not entry_sig: return None

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

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask, last_price=t_price)
            if fair_p < 0.05 or not t_id: return None
            
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
                    'index_trend': index_trend
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
        
        # 1. 更新 IV 状态
        if opt_data['has_feed']:
            if st.position == 1: curr_iv = opt_data['call_iv']
            elif st.position == -1: curr_iv = opt_data['put_iv']
            else: curr_iv = (opt_data['call_iv'] + opt_data['put_iv']) / 2
            if curr_iv > 0.01: st.last_valid_iv = curr_iv

        

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
            'action': 'ALPHA', 'ts': curr_ts, 'symbol': sym,
            'alpha': final_alpha, 'iv': st.last_valid_iv, 'price': price, 'vol_z': metrics['vol_z'],
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
                
                if st.entry_price > 0:
                    current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price
                    st.max_roi = max(st.max_roi, current_roi)
                    ctx['holding']['max_roi'] = st.max_roi
                     
            else:
                ctx['curr_price'] = market_opt_price
        else:
            if st.position != 0:
                # 🛡️ [防断流核心 3] 彻底缺失数据 (has_feed=False)，直接用开仓价兜底
                effective_price = st.entry_price
                if effective_price <= 0.01: effective_price = 0.01
                
                # 👇 [🔥 把 Debug 改为 Error，强制暴露问题]
                logger.error(f"🚨 [致命盲区] {sym} 期权行情彻底丢失(has_feed=False)！系统被蒙住双眼，当前 ROI 强制归 0.0！")
                # 👆 修复结束)。")
                ctx['curr_price'] = effective_price
            else:
                ctx['curr_price'] = 0.0

        # 4. 执行平仓 (必须在 is_zombie_market 之前，确保 EOD 和止损能随时触发)
        if st.position != 0:
            # 🛡️ [终极修复] 建仓绝对保护期 (Entry Breathing Room)
            # 绝对禁止在建仓后的最初 60 秒内通过常规逻辑平仓 (对应 _process_fast_fused_tick 中的逻辑)
            # 👇 [🔥 终极修复 3：同步防止 60.0 秒整点浮点数死锁，改为 59.0]
            if curr_ts - st.entry_ts < 59.0:
                # [🎯 靶向日志 B1] 暴露 59 秒锁定期状态
                logger.debug(f"🔒 [平仓屏蔽] {sym} 处于 59 秒建仓保护期内，暂不进行平仓评估。")
                return None

            # [🎯 靶向日志 B2] 送入 check_exit 前的最终状态切片
            current_roi = (ctx['curr_price'] - st.entry_price) / st.entry_price if st.entry_price > 0 else 0
            logger.info(f"🔍 [主循环平仓送审] {sym} | 模式: {self.mode} | 当前价: {ctx['curr_price']:.2f} | 成本: {st.entry_price:.2f} | ROI: {current_roi*100:.2f}% | Max ROI: {st.max_roi*100:.2f}%")
          
                
            exit_sig = self.strategy.check_exit(ctx)
            if exit_sig:
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
                else:
                    logger.info(f"📝 [Shadow] {sym} Exit signal detected, but skipping execution (Alpha-Only).")
            else:
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
        if not entry_sig: return None

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

            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask, last_price=t_price)
            if fair_p < 0.05 or not t_id: return None
            
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
                    'index_trend': index_trend
                }
            })
            return entry_sig
        return None

    async def process_batch(self, batch: dict):
        # 🚀 [Surgery 16] 处理同步心跳包：不做任何交易动作，直接返回。
        # 核心在于让调用方 run() 中的 current_ts_to_sync 得到更新，从而触发 sync:orch_done 的 ACK。
        if batch.get('action') == 'HEARTBEAT':
            return

        # 1. 统一时间基准与 EOD 处理
        ny_now, curr_ts = self._prepare_ny_time(batch)
        if ny_now is None:
             return

        # [Moved] 记录行情数据移到 Alpha 计算之后，以便记录 Alpha 信号

        current_time = ny_now.time()
        current_date = ny_now.date()
        
        # EOD 暴力强平 (15:50 - 16:00)
        # if current_time >= dt_time(15, 50) and current_time <= dt_time(16, 0):
        #     await self._force_clear_all(batch, "EOD_HARD_CLEAR", curr_ts, ny_now)
        #     if current_time > dt_time(15, 55): return
        
        # =================================================================
        # 🚀 [新增/修改] 全局主动强平防线 (跨越 Tick 限制，主动扫荡)
        # =================================================================
        close_h = self.strategy.cfg.CLOSE_HOUR
        close_m = self.strategy.cfg.CLOSE_MINUTE
        
        is_eod_time = (ny_now.hour == close_h and ny_now.minute >= close_m) or (ny_now.hour > close_h)
        
        if is_eod_time:
            last_clear_day = getattr(self, 'last_eod_clear_day', None)
            current_day = ny_now.date()
            
            # 每天到达 15:40 的那一刻，主动扫荡一次，强行唤醒所有无 Tick 的僵尸仓位！
            if last_clear_day != current_day:
                logger.warning(f"⏰ [时间风控] 到达策略指定平仓时间 {close_h}:{close_m}，启动全局主动强制清仓！")
                await self._force_clear_all(batch, "EOD_HARD_CLEAR", curr_ts, ny_now)
                self.last_eod_clear_day = current_day


        # 跨日回放/回溯清理 (🚨 仅当 current_date 大于 last_date 且跨度合理时执行)
        if self.last_date is not None and current_date > self.last_date:
            # [🔥 修复] 移除硬编码的 15:55，动态读取策略配置的平仓时间
            close_h = self.strategy.cfg.CLOSE_HOUR
            close_m = self.strategy.cfg.CLOSE_MINUTE
            yesterday_close_ny = ny_now.replace(year=self.last_date.year, month=self.last_date.month, day=self.last_date.day, hour=close_h, minute=close_m, second=0, microsecond=0)
            await self._force_clear_all(batch, "EOD_FORCE", yesterday_close_ny.timestamp(), yesterday_close_ny)
            
            self.consecutive_stop_losses = 0

            self.global_cooldown_until = 0
            self.index_opening_prices = {} # [NEW] 跨日重置开盘价缓存
            self._generate_daily_analysis_report(report_date_str=self.last_date.strftime('%Y%m%d'))
            
            # [🔥 跨日重置] 清空昨日统计数据，但保留 mock_cash 实现滚仓
            self.daily_trades = []
            self.stats_counter_trend_long_count = 0
            self.stats_counter_trend_long_win_count = 0
            self.stats_counter_trend_long_pnl = 0.0
            self.stats_counter_trend_short_count = 0
            self.stats_counter_trend_short_win_count = 0
            self.stats_counter_trend_short_pnl = 0.0
            self.stats_liquidity_drought_liquidations = 0


        self.last_date = current_date

        # 2. 信号与波动率提取
        use_precalc_feed = 'alpha_score' in batch or 'precalc_alpha' in batch
        try:
            symbols = batch['symbols']
            stock_prices = batch['stock_price']
        except KeyError as e:
            logger.error(f"❌ Batch Data Missing Key: {e}")
            return

        if len(symbols) != len(stock_prices):
            logger.error(f"❌ Array Mismatch: symbols({len(symbols)}) != prices({len(stock_prices)})")
            return

        spy_rocs = batch.get('spy_roc_5min', [0.0] * len(symbols))
        qqq_rocs = batch.get('qqq_roc_5min', [0.0] * len(symbols))
        
        # [NEW] 计算当日大盘趋势 (Today's Trend)
        index_trend = 0
        spy_day_roc = 0.0
        qqq_day_roc = 0.0
        
        for i, sym in enumerate(symbols):
            if sym in ['SPY', 'QQQ']:
                p = stock_prices[i]
                if sym not in self.index_opening_prices and p > 1.0: # 排除异常 0 价
                    self.index_opening_prices[sym] = p
                
                if sym in self.index_opening_prices:
                    open_p = self.index_opening_prices[sym]
                    day_roc = (p - open_p) / open_p
                    if sym == 'SPY': spy_day_roc = day_roc
                    else: qqq_day_roc = day_roc
        
        # [🔥 核心优化] 复合动量趋势判定 (Composite Momentum)
        # 1. 提取当前截面均值 5min ROC (敏锐追踪)
        s_5m = np.mean([r for r in spy_rocs if abs(r) > 1e-9]) if any(abs(r) > 1e-9 for r in spy_rocs) else 0.0
        q_5m = np.mean([r for r in qqq_rocs if abs(r) > 1e-9]) if any(abs(r) > 1e-9 for r in qqq_rocs) else 0.0
        
        # 2. 更新 EMA (平滑降噪: alpha=0.4 代表约 2.5 根 5min K线的均值影响力)
        alpha_ema = 0.4
        self.spy_ema_roc = alpha_ema * s_5m + (1 - alpha_ema) * self.spy_ema_roc
        self.qqq_ema_roc = alpha_ema * q_5m + (1 - alpha_ema) * self.qqq_ema_roc
        
        # 3. 趋势判定: 降低门槛至 0.02% (0.0002) 以匹配历史回测敏感度
        #    并增加“惰性”：只要当日累计 ROC 足够强，即使 EMA 出现微小反弹也不立即归 0
        index_trend = 0
        # 判断 SPY 或 QQQ 是否处于明显趋势 (0.02% 门槛)
        is_bull_day = (spy_day_roc > 0.0002 or qqq_day_roc > 0.0002)
        is_bear_day = (spy_day_roc < -0.0002 or qqq_day_roc < -0.0002)
        
        # 动量过滤器：只要 EMA 不处于明显的强力反转（例如正在猛拉），就维持当日趋势
        if is_bull_day and (self.spy_ema_roc > -0.0001 or self.qqq_ema_roc > -0.0001):
            index_trend = 1
        elif is_bear_day and (self.spy_ema_roc < 0.0001 or self.qqq_ema_roc < 0.0001):
            index_trend = -1
        
        self.last_index_trend = index_trend # [NEW] 存储全局最新趋势
        batch['index_trend'] = index_trend
        
        if use_precalc_feed:
            raw_alphas = batch.get('precalc_alpha', batch.get('alpha_score'))
            raw_vols = batch.get('fast_vol', np.zeros(len(symbols)))
        else:
            # 🧠 [Dict-Payload Paradigm] 精准按图索骥，彻底消灭 Index 错位
            features_dict = batch.get('features_dict')
            if features_dict:
                x_stk_list = []
                x_opt_list = []
                
                # 按照 slow_cfg 宪法中定义的特征顺序组装矩阵
                for f in self.slow_cfg['features']:
                    name = f['name']
                    # 排除名单：1. 原始价格列 (OHLV)  2. 静态 Embedding 列 (这些在 s_mock 里处理)
                    if name in {'open', 'high', 'low', 'close', 'volume', 'stock_id', 'sector_id', 'day_of_week'}:
                        continue
                        
                    # 从 Payload 中精准提取 (Shape: [B, 30])
                    feat_data = features_dict.get(name)
                    if feat_data is None:
                        # 容错：如果缺失，补零 (防止模型崩溃)
                        feat_data = np.zeros((len(symbols), 30), dtype=np.float32)
                    
                    # 转换 Tensor
                    t_feat = torch.from_numpy(feat_data).float().to(self.device)
                    
                    # 🛡️ [防弹修复] 针对 Categorical 特征强制 Clamp
                    if f.get('type') == 'categorical':
                        t_feat = torch.clamp(t_feat, 0, 50)
                    
                    if name.startswith('options_'): x_opt_list.append(t_feat)
                    else: x_stk_list.append(t_feat)
                
                # 动态垂直组装 [B, 30, N]
                x_stk = torch.stack(x_stk_list, dim=-1)
                x_opt = torch.stack(x_opt_list, dim=-1)
            else:
                # [Fallback] 回退到旧的 Index-Based 逻辑 (向下兼容)
                if 'x_stock' in batch and 'x_option' in batch:
                    x_stk = torch.from_numpy(batch['x_stock']).float().to(self.device)
                    x_opt = torch.from_numpy(batch['x_option']).float().to(self.device)
                else:
                    x_slow = torch.from_numpy(batch['slow_1m']).float().to(self.device)
                    x_stk = x_slow[..., self.slow_stock_indices]
                    x_opt = x_slow[..., self.slow_option_indices]

                # 兼容性旧版 Clamping
                if hasattr(self, 'slow_stk_cat_indices') and self.slow_stk_cat_indices:
                    x_stk[..., self.slow_stk_cat_indices] = torch.clamp(x_stk[..., self.slow_stk_cat_indices], 0, 50)
                if hasattr(self, 'slow_opt_cat_indices') and self.slow_opt_cat_indices:
                    x_opt[..., self.slow_opt_cat_indices] = torch.clamp(x_opt[..., self.slow_opt_cat_indices], 0, 50)
            
            # [Fix] 检查 NAN (无论哪种方式产生都需防护)
            if torch.isnan(x_stk).any() or torch.isinf(x_stk).any():
                x_stk = torch.nan_to_num(x_stk)
            if torch.isnan(x_opt).any() or torch.isinf(x_opt).any():
                x_opt = torch.nan_to_num(x_opt)

            # [Fix] stock_ids -> stock_id (对齐 Feature Service 产生的 Payload Key)
            s_mock = {
                'stock_id': torch.from_numpy(batch['stock_id']).long().to(self.device) if 'stock_id' in batch else torch.zeros(len(symbols), dtype=torch.long).to(self.device),
                'sector_id': torch.zeros(len(symbols), dtype=torch.long).to(self.device),
                'day_of_week': torch.full((len(symbols),), ny_now.weekday(), dtype=torch.long).to(self.device),
                'hour': torch.full((len(symbols),), ny_now.hour, dtype=torch.long).to(self.device),
                'minute': torch.full((len(symbols),), ny_now.minute, dtype=torch.long).to(self.device)
            }
            with torch.no_grad():
                # 🚀 [Fingerprint Audit] Trace NVDA at 10:00:00 (Baseline Side)
                if "NVDA" in symbols and ny_now.hour == 10 and ny_now.minute == 0 and ny_now.second == 0:
                    idx_nvda = symbols.index("NVDA")
                    stk_sample = x_stk[idx_nvda:idx_nvda+1]
                    logger.info(f"📊 [TRACE-NVDA-BASE] Tensor Fingerprint | Mean: {stk_sample.mean():.6f} | Std: {stk_sample.std():.6f} | Max: {stk_sample.max():.6f}")
                    # 🧪 [NEW] 保存全量基准矩阵，用于像素级对碰
                    np.save("nvda_baseline_1000.npy", stk_sample.cpu().numpy())
                    logger.info("💾 [TRACE-NVDA-BASE] Full Matrix saved to nvda_baseline_1000.npy")
                
                out = self.slow_model(x_stk, x_opt, s_mock)
                raw_alphas = out['rank_score'].cpu().numpy().flatten()

        # =========================================================
        # 🚀 [深度诊断 & 动态自适应归一化]
        # 彻底抛弃硬编码的 GLOBAL_STATS，让系统自己学习当前模型的输出分布！
        # (确保无论实时推断还是预计算模式，都能动态捕捉 Alpha 统计特征)
        # =========================================================
        exclude_indices = {i for i, s in enumerate(symbols) if s in ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS}
        valid_alphas = [a for i, a in enumerate(raw_alphas) if i not in exclude_indices]
        if valid_alphas:
            mean_a = np.mean(valid_alphas)
            std_a = np.std(valid_alphas)
        else:
            mean_a = np.mean(raw_alphas)
            std_a = np.std(raw_alphas)
        
        # 增量平滑更新全局 Alpha 基准 (EWMA)
        alpha_ewma = 0.05
        if self.alpha_count == 0:
            self.dynamic_alpha_mean = float(mean_a)
            self.dynamic_alpha_std = float(std_a if std_a > 1e-5 else 1.0)
        else:
            self.dynamic_alpha_mean = float((1 - alpha_ewma) * self.dynamic_alpha_mean + alpha_ewma * mean_a)
            self.dynamic_alpha_std = float((1 - alpha_ewma) * self.dynamic_alpha_std + alpha_ewma * (std_a if std_a > 1e-5 else 1.0))
        self.alpha_count += 1
        
        # =========================================================
        # 🚀 [修正] 动态波动率 (Volatility) 追踪 (对齐离线时序逻辑)
        # =========================================================
        raw_vols = batch.get('fast_vol', np.zeros(len(symbols)))
        
        # 初始化每个个股的波动率状态机
        if getattr(self, 'sym_vol_mean', None) is None:
            self.sym_vol_mean = {s: 0.0 for s in symbols}
            self.sym_vol_var = {s: 1.0 for s in symbols}
            
        vol_ewma = 2.0 / (1000 + 1) # 对齐离线的回测 1000-span EWMA
        vol_z_dict = {}
        import math
        for idx_v, s in enumerate(symbols):
            r_v = float(raw_vols[idx_v])
            diff = r_v - self.sym_vol_mean.get(s, 0.0)
            mean_cur = self.sym_vol_mean.get(s, 0.0) + vol_ewma * diff
            var_cur = (1 - vol_ewma) * self.sym_vol_var.get(s, 1.0) + vol_ewma * (diff ** 2)
            self.sym_vol_mean[s] = mean_cur
            self.sym_vol_var[s] = var_cur
            v_std = math.sqrt(var_cur) if var_cur > 0 else 1.0
            vz = (r_v - mean_cur) / (v_std + 1e-6)
            vol_z_dict[s] = max(-5.0, min(5.0, vz))

        

        # 僵尸截面检查
        zero_vol_count = sum(1 for v in raw_vols if abs(float(v)) < 1e-5)
        is_zombie_market = (len(symbols) > 0 and (zero_vol_count / len(symbols)) > 0.50)
        
        # 3. 符号处理循环 (指标、解析、评估)
        entry_candidates = []
        metrics_batch = {
            'alpha_mean': mean_a, 'alpha_std': std_a,
            'vol_z_dict': vol_z_dict, # 传递计算好的独立时序 vol_z
            'is_new_minute': batch.get('is_new_minute', True) # 🚀 [新增] 从 Feature Service 提取逻辑分钟标志
        }

        # 👇 [🔥 IC 修复 1/2] 准备一个数组，用来装载真正带反转逻辑的 final_alpha
        final_alphas_for_ic = np.zeros(len(symbols))

        # 🛡️ [NEW] 计算全局市场状态 (Market Regime Guard)
        # 预先统一计算 VIXY 的反转频率，作为全场通用的 "洗盘频率"
        vixy_st = self.states.get('VIXY')
        global_regime_reversal_count = 0
        if vixy_st:
            global_regime_reversal_count = vixy_st.get_reversal_count(
                window_mins=getattr(self.cfg, 'REGIME_WINDOW_MINS', 30),
                threshold=getattr(self.cfg, 'REGIME_REVERSAL_PERCENT', 0.001)
            )
        
        for i, sym in enumerate(symbols):
            metrics = self._prep_symbol_metrics(i, sym, stock_prices, raw_alphas, raw_vols, use_precalc_feed, metrics_batch)
            if not metrics: continue

            # 👇 [🔥 IC 修复 2/2] 把反转、归一化后的终极 Alpha 存起来
            final_alphas_for_ic[i] = metrics['final_alpha']
             
            st = metrics['st']
            # LIVEREPLAY 模式和 realtime 模式都采用 live_options 字典结构
            if self.mode == 'realtime' or os.environ.get('RUN_MODE') == 'LIVEREPLAY':
                opt_data = self._get_opt_data_realtime(sym, st, ny_now, metrics['price'], batch)
            else:
                opt_data = self._get_opt_data_backtest(batch, i, sym, st)

            # 👇 [🔥 记录期权采样价格，用于正确的 Unrealized PnL 计算]
            if st.position != 0 and opt_data['has_feed']:
                # 获取当前持仓方向的公允价 (Mid)
                st.last_opt_price = self._get_fair_market_price(
                    opt_data['call_price'] if st.position == 1 else opt_data['put_price'],
                    opt_data['call_bid'] if st.position == 1 else opt_data['put_bid'],
                    opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']
                )
            # 👆

            st.last_spy_roc = float(spy_rocs[i]) if i < len(spy_rocs) else 0.0
            st.last_qqq_roc = float(qqq_rocs[i]) if i < len(qqq_rocs) else 0.0

            entry_sig = await self._evaluate_symbol_signals(
                i, sym, metrics, opt_data, ny_now, curr_ts, st.last_spy_roc, st.last_qqq_roc, is_zombie_market, 
                index_trend, global_regime_reversal_count
            )
            
            # 🚀 [NEW] 为持仓标的同步最新的期权参考价 (用于会计模块计算准确的 Unrealized PnL)
            # 🚀 [NEW] 为持仓标的同步最新的期权参考价 (用于会计模块计算准确的 Unrealized PnL)
            if st.position != 0:
                if opt_data['has_feed']:
                    mp = opt_data['call_price'] if st.position == 1 else opt_data['put_price']
                    bid = opt_data['call_bid'] if st.position == 1 else opt_data['put_bid']
                    ask = opt_data['call_ask'] if st.position == 1 else opt_data['put_ask']
                    st.last_opt_price = self._get_fair_market_price(mp, bid, ask)
                else:
                    # 遭遇断流时，使用开仓价或 0.01 兜底，防止 PnL 计算崩溃
                    st.last_opt_price = st.entry_price if st.entry_price > 0.01 else 0.01
                    
            if entry_sig:
                # [Optimization] Apply a momentum multiplier to Alpha ranking so dynamically moving stocks squeeze out stagnant high-alpha ones
                raw_rank = abs(metrics['final_alpha']) / max(0.1, st.last_valid_iv)
                mom_multiplier = 1.0 + abs(metrics['roc_5m']) * 100.0  # e.g. 0.5% move = 1.5x rank boost
                entry_candidates.append({
                    'sym': sym, 'sig': entry_sig, 'price': metrics['price'],
                    'curr_ts': curr_ts, 'batch_idx': i,
                    'alpha_strength': raw_rank * mom_multiplier
                })

        if self.mode == 'backtest' and hasattr(self.ibkr, 'record_market_data'):
            self.ibkr.record_market_data(batch, alphas=final_alphas_for_ic)

        # 4. 截面排序与执行
        if entry_candidates:
            if len(symbols) < 10:
                logger.warning(f"🛡️ [截面防线] 在线标的 ({len(symbols)}) 不足 10，放弃开仓！")
                return 

            entry_candidates.sort(key=lambda x: x['alpha_strength'], reverse=True)
            MAX_TOP_K = 3
            for cand in entry_candidates[:MAX_TOP_K]:
                # 🚀 [终极防超限] 在循环内也必须动态盘点，防止一帧开出 10 个
                current_active = sum(1 for s, s_state in self.states.items() if s_state.position != 0 or s_state.is_pending)
                if current_active >= MAX_POSITIONS:
                    logger.warning(f"✋ [及门而止] 达到最大持仓 ({current_active}/{MAX_POSITIONS})，停止继续开仓。")
                    break
                if not self.only_log_alpha:
                    await self._execute_entry(cand['sym'], cand['sig'], cand['price'], cand['curr_ts'], cand['batch_idx'])
                else:
                    logger.info(f"📝 [Shadow] {cand['sym']} Entry candidates found, but skipping execution (Alpha-Only).")

        # 🚀 [Shadow Validation] Trigger synchronous PnL check in replay for 100% determinism
        if IS_LIVEREPLAY:
             await self._report_pnl_status_logic(curr_ts, "LIVEREPLAY_SYNC")

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
        
        if self.mode == 'realtime' and not IS_SIMULATED: 
            await self.ibkr.connect()
            # [NEW] 尝试获取真实账户资金，根据 config 统一同步逻辑
            if hasattr(self.ibkr, 'get_account_balance'):
                real_bal = await self.ibkr.get_account_balance()

                # 🚨 [核心修复] 只有在实盘且真实券商有钱时，才覆盖本地 mock_cash
                if TRADING_ENABLED and real_bal is not None and real_bal > 0:
                    self.mock_cash = float(real_bal)
                    logger.info(f"💰 REAL ACCOUNT BALANCE LOADED: ${self.mock_cash:,.2f}")
                else:
                    # 在 DRY RUN 模式下，绝对不能重置！必须沿用 _load_state 从 DB 恢复出来的虚拟资金
                    logger.info(f"⚠️ 模拟盘 (DRY RUN) 运行中，继续沿用数据库持久化的虚拟资金: ${self.mock_cash:,.2f}")
        elif IS_LIVEREPLAY:
            logger.info(f"🎞️ Live Replay Mode: Skipping IBKR connection.")
        self._ensure_consumer_group()
        
        # [新增] 心跳计时器
        last_heartbeat = time.time()
        # 定义心跳超时阈值 (秒)
        HEARTBEAT_TIMEOUT = 120 
        
        # 🚀 [🔥 核心新增] 启动内存盈亏监视器，作为数据库之外的第二真相源
        # 在 LIVEREPLAY 模式下由 process_batch 同步驱动，不在后台扫描以保证确定性
        if not IS_LIVEREPLAY and not SYNC_EXECUTION:
            asyncio.create_task(self._pnl_monitor_loop())
            
        # 🛡️ [🔥 硬核新增] 启动防掉单真实对账器
        if not IS_LIVEREPLAY and not SYNC_EXECUTION and TRADING_ENABLED:
            asyncio.create_task(self.reconciler.run_reconciliation_loop())
        
        while True:
            try:
                 # 检查心跳 - 只在实盘且真实连接时才监控
                time_since_last = time.time() - last_heartbeat
                is_true_realtime = (self.mode == 'realtime' and hasattr(self, 'ibkr') and hasattr(self.ibkr, 'ib') and self.ibkr.ib.isConnected())
                
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
                    # 🚀 [Surgery 18] 主动追赶逻辑 (杜绝死锁)
                    # 在回放模式下，如果读不到信号，但特征引擎已经跑在前面了，
                    # 说明这一分钟没有信号，或者心跳包漏了，主动同步时轴以防死锁。
                    from config import IS_SIMULATED
                    if IS_SIMULATED:
                        feat_done = self.r.get("sync:feature_calc_done")
                        if feat_done:
                            feat_ts = float(feat_done)
                            # 获取当前 Orch 的水位（本地记录或 Redis 获取）
                            last_orch_done_raw = self.r.get("sync:orch_done")
                            last_orch_done = float(last_orch_done_raw) if last_orch_done_raw else 0
                            
                            if feat_ts > last_orch_done:
                                # logger.info(f"⏭️  [Catchup] Moving Orch Sync to {feat_ts} (Feat: {feat_ts} > Orch: {last_orch_done})")
                                self.r.set("sync:orch_done", str(feat_ts))
                    
                    await asyncio.sleep(0.01)
                    continue
                
                logger.info(f"📩 Orchestrator received {sum(len(msgs) for _, msgs in resp)} messages from {len(resp)} streams")
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

                # 🚀 [核心修复：同步锁统一释放] 
                # 只要处理了数据，且当前是回放模式，无条件发送完成信号！杜绝死锁！
                if current_ts_to_sync and IS_SIMULATED:
                    logger.info(f"🏁 Sync Ack: sync:orch_done = {current_ts_to_sync}")
                    self.r.set("sync:orch_done", str(current_ts_to_sync))

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
