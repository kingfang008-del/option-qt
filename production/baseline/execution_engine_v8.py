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
import uuid
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta
from collections import deque
from queue import Empty
from pytz import timezone
from scipy.stats import norm 
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from entry_risk_rules import evaluate_entry_liquidity

# 引入纯策略核心
from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION, StrategyCore, StrategyConfig


# [Refactor] 引入模块化执行组件
from orchestrator_state_manager import OrchestratorStateManager
from orchestrator_order_state import OrchestratorOrderStateManager
from orchestrator_accounting import OrchestratorAccounting
from orchestrator_execution import OrchestratorExecution
from orchestrator_reconciler import OrchestratorReconciler
from utils import serialization_utils as ser
try:
    from Domain.shadow_router import get_domain_shadow_router
except Exception:  # pragma: no cover
    get_domain_shadow_router = None

 
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
    OMS_SIGNAL_DELAY_BARS,
    OMS_SIGNAL_DELAY_ACTIONS,
    IS_BACKTEST,
    IS_SIMULATED,
    IS_REALTIME_DRY,
    OMS_GUARD_STALE_QUOTES,
    OMS_MAX_QUOTE_STALE_SEC,
    OMS_MAX_QUOTE_WALL_STALE_SEC,
    OMS_BLOCK_ENTRY_ON_STALE,
    OMS_BLOCK_EXIT_ON_STALE,
    OMS_ALLOW_EOD_EXIT_ON_STALE,
    OMS_STATE_NAMESPACE,
    LIVE_TRADING_CAPITAL_LIMIT,
    PURE_ALPHA_REPLAY,
)
from runtime_trading_controls import get_runtime_live_trading_capital_limit


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

# ============================================================================
# [ExecutionWindow] 分钟级执行窗口 — 回测/实盘统一契约 (Step A)
# ----------------------------------------------------------------------------
# 背景:
#   以前散落在回放脚本里的 3 行调用 (process_batch → cache_minute → execute_phase)
#   没有强约束, 容易让未来的 chunked-fill / cancel-replace / tight-exit 改动在脚本侧
#   和 OMS 侧出现实现漂移. 这个 dataclass 把 "一分钟的 alpha + 60 秒行情" 打包成一个
#   首要契约, 使 OMS.execute_window 成为回测和实盘共享的唯一入口.
#
# 语义 (硬约束):
#   minute_ts           — 该分钟左边界 epoch 秒
#   alpha_label_ts      — 该窗口所消费的 alpha 的 label ts, 必定 = minute_ts - 60
#   alpha_available_ts  — alpha 可见时刻, 必定 = minute_ts (10:10 alpha 到 10:11:00 才可见)
#   alpha_frame         — 分钟边界上一次性评估的 signal_packet (strategy 只在此跑一次)
#   quotes_1s           — 该分钟内有序的 1s 行情包 (ts ∈ [minute_ts, minute_ts+60))
#
# 契约验证 (validate) 采取宽松策略:
#   - 硬违约 (alpha_frame.ts 错位 / quotes 越界) → warning 打日志, 调用方选择是否 raise
#   - 软缺失 (某秒数据缺失) → 允许, 不阻断回放
#
# 注意: 这个结构并不拥有策略 / 执行 / 撮合逻辑, 它只是一个有契约的 payload.
#       真正的"分钟边界一次 + 60 秒循环"编排在 ExecutionEngineV8.execute_window 里.
# ============================================================================
@dataclass
class ExecutionWindow:
    minute_ts: int
    alpha_label_ts: int
    alpha_available_ts: int
    alpha_frame: Dict[str, Any]
    quotes_1s: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_packets(
        cls,
        minute_ts: int,
        alpha_frame: Dict[str, Any],
        quotes_1s: List[Dict[str, Any]],
    ) -> "ExecutionWindow":
        """标准构造: 用 minute_ts 自动推导 alpha_label_ts / alpha_available_ts."""
        m = int(minute_ts)
        return cls(
            minute_ts=m,
            alpha_label_ts=m - 60,
            alpha_available_ts=m,
            alpha_frame=alpha_frame,
            quotes_1s=list(quotes_1s or []),
        )

    def validate(self, strict: bool = False) -> List[str]:
        """返回违反契约的条目 (空 list = 通过). strict=True 时 raise ValueError."""
        errors: List[str] = []
        try:
            frame_ts = int(float(self.alpha_frame.get('ts', 0)))
        except Exception:
            frame_ts = -1
        if frame_ts != self.minute_ts:
            errors.append(
                f"alpha_frame.ts={frame_ts} != minute_ts={self.minute_ts}"
            )
        if self.alpha_label_ts != self.minute_ts - 60:
            errors.append(
                f"alpha_label_ts={self.alpha_label_ts} != minute_ts-60={self.minute_ts - 60}"
            )
        if self.alpha_available_ts != self.minute_ts:
            errors.append(
                f"alpha_available_ts={self.alpha_available_ts} != minute_ts={self.minute_ts}"
            )
        if not self.quotes_1s:
            errors.append("quotes_1s 为空")
        else:
            lo, hi = self.minute_ts, self.minute_ts + 60
            for i, q in enumerate(self.quotes_1s):
                try:
                    q_ts = float(q.get('ts', 0.0))
                except Exception:
                    q_ts = -1.0
                if not (lo <= q_ts < hi):
                    errors.append(
                        f"quotes_1s[{i}].ts={q_ts:.0f} 越界 [{lo}, {hi})"
                    )
                    break
        if strict and errors:
            raise ValueError(
                "ExecutionWindow validation failed: " + "; ".join(errors)
            )
        return errors

    def summary(self) -> str:
        return (
            f"win[{self.minute_ts}] alpha={self.alpha_label_ts}→avail={self.alpha_available_ts} "
            f"quotes={len(self.quotes_1s)}"
        )


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
        self.option_tag = ""
        self.latest_call_id = "" 
        self.latest_put_id = ""  
        self.is_pending = False
        self.pending_side = None
        self.pending_action = None # [幽灵 C] 记录当前锁定的动作
        self.pending_ts = 0.0      # [幽灵 C] 记录加锁时间戳
        # Independent OMS slot reservation. Pending flags are transient and may
        # be cleared by async order lifecycles; this one represents capacity.
        self.entry_slot_reserved = False
        self.open_fill_confirmed = False
        self.pending_exit_retry_reason = ""
        self.pending_exit_retry_count = 0
        self.pending_exit_retry_first_ts = 0.0
        self.pending_exit_retry_last_ts = 0.0
        self.second_exit_streak_reason = ""
        self.second_exit_streak_count = 0
        self.second_exit_last_ts = 0.0

        self.prev_macd_hist = 0.0
        
        self.strike_price = 0.0
        self.expiry_date = None 
        self.last_valid_iv = 0.5
        self.opt_type = 'call'

        self.warmup_complete = False
        
        self.last_spread_pct = 0.0
        self.last_snap_roc = 0.0
        self.last_vol_z = 0.0
        self.last_price = 0.0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.cached_min_roc = 0.0
        self.cached_macd_hist = 0.0
        self.cached_macd_hist_slope = 0.0
        self.last_macd_hist = 0.0
        self.last_macd_hist_slope = 0.0
        self.last_opt_price = 0.0
        self.last_tick_price = None
        self.last_tick_opt_data = None
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
            'option_tag': self.option_tag,
            'strike_price': self.strike_price,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'last_valid_iv': self.last_valid_iv,
            'opt_type': self.opt_type,
            'warmup_complete': self.warmup_complete,
            'correction_mode': self.correction_mode, # 🚨 [修复]
            'prev_macd_hist': self.prev_macd_hist,   # 🚨 [修复]
            'last_spread_pct': self.last_spread_pct, # 🚨 [修复]
            'entry_slot_reserved': bool(self.entry_slot_reserved),
            'open_fill_confirmed': bool(self.open_fill_confirmed),
            'pending_exit_retry_reason': self.pending_exit_retry_reason,
            'pending_exit_retry_count': int(self.pending_exit_retry_count),
            'pending_exit_retry_first_ts': float(self.pending_exit_retry_first_ts),
            'pending_exit_retry_last_ts': float(self.pending_exit_retry_last_ts),
            'second_exit_streak_reason': self.second_exit_streak_reason,
            'second_exit_streak_count': int(self.second_exit_streak_count),
            'second_exit_last_ts': float(self.second_exit_last_ts),

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
        # [🛡️ Defensive Coerce] 与 SE 的 from_dict 对齐: 历史脏数据会把 entry_spy_roc /
        # entry_alpha_z 存成 dict, 恢复后 holding payload 透回 SE 会令 check_exit 抛
        # TypeError, 把整批 tick 的 alpha 陪葬. 统一在 load 边界做类型兜底.
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
        self.option_tag = str(data.get('option_tag', data.get('tag', '')) or '').strip().upper()
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
        self.entry_slot_reserved = bool(data.get('entry_slot_reserved', self.position != 0))
        explicit_open_fill = data.get('open_fill_confirmed')
        if explicit_open_fill is None:
            self.open_fill_confirmed = bool(
                self.position != 0 and self.qty > 0 and self.entry_price > 0 and self.entry_ts > 0
            )
        else:
            self.open_fill_confirmed = bool(explicit_open_fill)
        self.pending_exit_retry_reason = str(data.get('pending_exit_retry_reason', '') or '')
        self.pending_exit_retry_count = _coerce_int(data.get('pending_exit_retry_count', 0))
        self.pending_exit_retry_first_ts = _coerce_float(data.get('pending_exit_retry_first_ts', 0.0))
        self.pending_exit_retry_last_ts = _coerce_float(data.get('pending_exit_retry_last_ts', 0.0))
        self.second_exit_streak_reason = str(data.get('second_exit_streak_reason', '') or '')
        self.second_exit_streak_count = _coerce_int(data.get('second_exit_streak_count', 0))
        self.second_exit_last_ts = _coerce_float(data.get('second_exit_last_ts', 0.0))
        
        # [新增] 恢复 Buffer
        if 'prices' in data: self.prices = deque(data['prices'], maxlen=self.prices.maxlen)
        if 'alpha_history' in data: self.alpha_history = deque(data['alpha_history'], maxlen=self.alpha_history.maxlen)
        if 'pct_history' in data: self.pct_history = deque(data['pct_history'], maxlen=self.pct_history.maxlen)
        
        self.ema_fast_val = data.get('ema_fast_val')
        self.ema_slow_val = data.get('ema_slow_val')
        self.dea_val = data.get('dea_val')


def compute_entry_priority_score(
    *,
    alpha: float,
    iv: float,
    roc_5m: float,
    snap_roc: float,
    macd_hist: float,
    entry_dir: int,
    cfg,
    pure_alpha_replay: bool = False,
    trend_net: float = 0.0,
    trend_efficiency: float = 0.0,
    trend_r2: float = 0.0,
    trend_observations: int = 0,
) -> Dict[str, float]:
    alpha_abs = abs(float(alpha or 0.0))
    effective_iv = max(0.1, float(iv or 0.0))
    direction = 1 if int(entry_dir or 0) >= 0 else -1

    alpha_power = float(getattr(cfg, 'ENTRY_RANK_ALPHA_POWER', 1.35))
    iv_penalty_power = float(getattr(cfg, 'ENTRY_RANK_IV_PENALTY_POWER', 0.0))
    base_alpha = alpha_abs ** alpha_power
    base_score = base_alpha if pure_alpha_replay else base_alpha / (effective_iv ** iv_penalty_power)

    high_alpha_bonus = min(
        max(
            0.0,
            alpha_abs - float(getattr(cfg, 'ENTRY_RANK_HIGH_ALPHA_FLOOR', 1.20)),
        )
        * float(getattr(cfg, 'ENTRY_RANK_HIGH_ALPHA_BONUS_SCALE', 0.35)),
        float(getattr(cfg, 'ENTRY_RANK_HIGH_ALPHA_MAX_BONUS', 0.50)),
    )
    alpha_mult = 1.0 + high_alpha_bonus

    abs_roc_mult = 1.0 + abs(float(roc_5m or 0.0)) * float(
        getattr(cfg, 'ENTRY_RANK_ROC_ABS_SCALE', 100.0) or 100.0
    )

    stock_bonus = min(
        max(0.0, float(roc_5m or 0.0) * direction)
        * float(getattr(cfg, 'ENTRY_RANK_STOCK_ROC_SCALE', 120.0) or 120.0),
        float(getattr(cfg, 'ENTRY_RANK_STOCK_ROC_MAX_BONUS', 0.35) or 0.35),
    )
    stock_mult = 1.0 + stock_bonus

    snap_bonus = min(
        max(0.0, float(snap_roc or 0.0) * direction)
        * float(getattr(cfg, 'ENTRY_RANK_SNAP_ROC_SCALE', 200.0) or 200.0),
        float(getattr(cfg, 'ENTRY_RANK_SNAP_ROC_MAX_BONUS', 0.30) or 0.30),
    )
    snap_mult = 1.0 + snap_bonus

    macd_bonus = min(
        max(0.0, float(macd_hist or 0.0) * direction)
        * float(getattr(cfg, 'ENTRY_RANK_MACD_SCALE', 8.0) or 8.0),
        float(getattr(cfg, 'ENTRY_RANK_MACD_MAX_BONUS', 0.30) or 0.30),
    )
    macd_mult = 1.0 + macd_bonus

    stock_ok = float(roc_5m or 0.0) * direction >= float(
        getattr(cfg, 'ENTRY_PRIORITY_STOCK_ROC_FLOOR', 0.0002) or 0.0002
    )
    snap_ok = float(snap_roc or 0.0) * direction >= float(
        getattr(cfg, 'ENTRY_PRIORITY_SNAP_ROC_FLOOR', 0.0) or 0.0
    )
    macd_ok = float(macd_hist or 0.0) * direction >= float(
        getattr(cfg, 'ENTRY_PRIORITY_MACD_FLOOR', 0.01) or 0.01
    )
    confirmation_count = int(stock_ok) + int(snap_ok) + int(macd_ok)
    is_priority_candidate = (
        alpha_abs >= float(getattr(cfg, 'ENTRY_PRIORITY_ALPHA_FLOOR', 0.9) or 0.9)
        and confirmation_count >= int(getattr(cfg, 'ENTRY_PRIORITY_MIN_CONFIRMATIONS', 2) or 2)
    )

    priority_mult = 1.0
    if is_priority_candidate:
        priority_mult += float(getattr(cfg, 'ENTRY_PRIORITY_BOOST', 0.80) or 0.80)
        if stock_ok:
            priority_mult += float(getattr(cfg, 'ENTRY_PRIORITY_STOCK_BONUS', 0.25) or 0.25)
        if snap_ok:
            priority_mult += float(getattr(cfg, 'ENTRY_PRIORITY_SNAP_BONUS', 0.15) or 0.15)
        if macd_ok:
            priority_mult += float(getattr(cfg, 'ENTRY_PRIORITY_MACD_BONUS', 0.20) or 0.20)

    trend_mult = 1.0
    if bool(getattr(cfg, 'ENTRY_RANK_TREND_QUALITY_ENABLED', True)):
        min_obs = int(getattr(cfg, 'ENTRY_RANK_TREND_MIN_OBS', 16))
        if int(trend_observations or 0) >= min_obs:
            net_target = max(1e-6, float(getattr(cfg, 'ENTRY_RANK_TREND_NET_TARGET', 0.012)))
            net_score = min(1.0, max(0.0, float(trend_net or 0.0)) / net_target)
            eff_score = min(1.0, max(0.0, float(trend_efficiency or 0.0)))
            r2_score = min(1.0, max(0.0, float(trend_r2 or 0.0)))
            trend_quality = 0.45 * net_score + 0.30 * eff_score + 0.25 * r2_score
            floor = max(1e-6, float(getattr(cfg, 'ENTRY_RANK_TREND_QUALITY_FLOOR', 0.25)))
            trend_mult += float(getattr(cfg, 'ENTRY_RANK_TREND_QUALITY_BOOST', 0.12)) * trend_quality
            if trend_quality < floor:
                trend_mult -= float(getattr(cfg, 'ENTRY_RANK_TREND_QUALITY_PENALTY', 0.25)) * ((floor - trend_quality) / floor)
            trend_mult = min(
                float(getattr(cfg, 'ENTRY_RANK_TREND_MAX_MULT', 1.12)),
                max(float(getattr(cfg, 'ENTRY_RANK_TREND_MIN_MULT', 0.75)), trend_mult),
            )

    score = base_score * alpha_mult * abs_roc_mult * stock_mult * snap_mult * macd_mult * priority_mult * trend_mult
    return {
        'score': float(score),
        'base_score': float(base_score),
        'alpha_mult': float(alpha_mult),
        'abs_roc_mult': float(abs_roc_mult),
        'stock_mult': float(stock_mult),
        'snap_mult': float(snap_mult),
        'macd_mult': float(macd_mult),
        'priority_mult': float(priority_mult),
        'trend_mult': float(trend_mult),
        'confirmation_count': float(confirmation_count),
        'is_priority_candidate': float(1.0 if is_priority_candidate else 0.0),
    }


def compute_entry_trend_quality(prices: Any, entry_dir: int, window_mins: int = 30) -> Dict[str, float]:
    try:
        price_list = [float(p) for p in list(prices)[-int(window_mins + 1):] if float(p) > 0]
    except Exception:
        price_list = []
    if len(price_list) < 2:
        return {'trend_net': 0.0, 'trend_efficiency': 0.0, 'trend_r2': 0.0, 'trend_observations': float(len(price_list))}

    direction = 1 if int(entry_dir or 0) >= 0 else -1
    returns = [
        (price_list[i] - price_list[i - 1]) / price_list[i - 1]
        for i in range(1, len(price_list))
        if price_list[i - 1] > 0
    ]
    raw_net = (price_list[-1] - price_list[0]) / price_list[0] if price_list[0] > 0 else 0.0
    trend_net = raw_net * direction
    path = sum(abs(x) for x in returns)
    trend_efficiency = max(0.0, trend_net) / path if path > 0 else 0.0

    n = len(price_list)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(price_list) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in price_list)
    if sxx > 0 and syy > 0:
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, price_list))
        trend_r2 = max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))
    else:
        trend_r2 = 0.0

    return {
        'trend_net': float(trend_net),
        'trend_efficiency': float(trend_efficiency),
        'trend_r2': float(trend_r2),
        'trend_observations': float(len(price_list)),
    }


def reserve_priority_entry_slots(entry_candidates: List[Dict[str, Any]], allowed_entries: int, cfg) -> List[Dict[str, Any]]:
    if allowed_entries <= 0 or not entry_candidates:
        return []

    selected = list(entry_candidates[:allowed_entries])
    reserved_slots = max(0, int(getattr(cfg, 'ENTRY_PRIORITY_RESERVED_SLOTS', 1) or 0))
    if reserved_slots <= 0:
        return selected

    def _is_priority_candidate(cand: Dict[str, Any]) -> bool:
        return bool(cand.get('is_priority_candidate', False))

    priority_pool = [cand for cand in entry_candidates if _is_priority_candidate(cand)]
    if not priority_pool:
        return selected

    selected_priority = sum(1 for cand in selected if _is_priority_candidate(cand))
    if selected_priority >= min(reserved_slots, allowed_entries):
        return selected

    missing = min(reserved_slots, allowed_entries) - selected_priority
    selected_keys = {(cand.get('sym'), cand.get('batch_idx')) for cand in selected}
    replacements = [cand for cand in priority_pool if (cand.get('sym'), cand.get('batch_idx')) not in selected_keys][:missing]
    if not replacements:
        return selected

    replaceable_idx = [
        idx for idx in range(len(selected) - 1, -1, -1)
        if not _is_priority_candidate(selected[idx])
    ]
    for cand, idx in zip(replacements, replaceable_idx):
        selected[idx] = cand

    return sorted(selected, key=lambda x: float(x.get('alpha_strength', 0.0) or 0.0), reverse=True)


def select_direction_split_entry_slots(entry_candidates: List[Dict[str, Any]], allowed_entries: int, cfg) -> List[Dict[str, Any]]:
    if allowed_entries <= 0 or not entry_candidates:
        return []

    if not bool(getattr(cfg, 'ENTRY_DIRECTION_SPLIT_POOL_ENABLED', True)):
        return reserve_priority_entry_slots(entry_candidates, allowed_entries, cfg)

    sorted_candidates = sorted(
        entry_candidates,
        key=lambda x: float(x.get('alpha_strength', 0.0) or 0.0),
        reverse=True,
    )
    if allowed_entries == 1:
        return sorted_candidates[:1]

    def _direction(cand: Dict[str, Any]) -> int:
        try:
            return 1 if int((cand.get('sig') or {}).get('dir', 0) or 0) >= 0 else -1
        except Exception:
            return 1

    call_pool = [cand for cand in sorted_candidates if _direction(cand) == 1]
    put_pool = [cand for cand in sorted_candidates if _direction(cand) == -1]
    if not call_pool or not put_pool:
        return reserve_priority_entry_slots(sorted_candidates, allowed_entries, cfg)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def _add(cand: Dict[str, Any]) -> None:
        key = (cand.get('sym'), cand.get('batch_idx'))
        if key not in selected_keys and len(selected) < allowed_entries:
            selected.append(cand)
            selected_keys.add(key)

    # 名额足够时先各保留一个方向槽，避免 CALL/PUT 在同一 abs(alpha) 池里互相挤掉。
    _add(call_pool[0])
    _add(put_pool[0])

    for cand in sorted_candidates:
        _add(cand)
        if len(selected) >= allowed_entries:
            break

    reserved_slots = max(0, int(getattr(cfg, 'ENTRY_PRIORITY_RESERVED_SLOTS', 1) or 0))
    if reserved_slots > 0:
        selected_priority = sum(1 for cand in selected if bool(cand.get('is_priority_candidate', False)))
        missing_priority = min(reserved_slots, allowed_entries) - selected_priority
        priority_pool = [cand for cand in sorted_candidates if bool(cand.get('is_priority_candidate', False))]
        for priority_cand in priority_pool:
            if missing_priority <= 0:
                break
            priority_key = (priority_cand.get('sym'), priority_cand.get('batch_idx'))
            if priority_key in selected_keys:
                continue
            priority_dir = _direction(priority_cand)
            replace_idx = None
            for idx in range(len(selected) - 1, -1, -1):
                if _direction(selected[idx]) == priority_dir and not bool(selected[idx].get('is_priority_candidate', False)):
                    replace_idx = idx
                    break
            if replace_idx is None:
                continue
            old_key = (selected[replace_idx].get('sym'), selected[replace_idx].get('batch_idx'))
            selected_keys.discard(old_key)
            selected[replace_idx] = priority_cand
            selected_keys.add(priority_key)
            missing_priority -= 1

    return sorted(selected, key=lambda x: float(x.get('alpha_strength', 0.0) or 0.0), reverse=True)

class ExecutionEngineV8:
    STALE_GUARD_RISK_EXIT_TOKENS = (
        "TIME_STOP",
        "TIME_STOP15",
        "NO_MOMENTUM",
        "ZOMBIE_STOP",
    )

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
        # OMS owns StrategyCore so decisions share the same position/cash truth.
        self.strategy = StrategyCore(StrategyConfig())
        self.cfg = self.strategy.cfg
        logger.info(f"🧭 [OMS] Active strategy core: {ACTIVE_STRATEGY_CORE_VERSION}")
        self.pending_orders = {}

        self.disable_db_save = True
        
        # [Refactor] 模块化组件初始化
        self.order_state = OrchestratorOrderStateManager(self)
        self.state_manager = OrchestratorStateManager(self)
        self.accounting = OrchestratorAccounting(self)
        self.execution = OrchestratorExecution(self)
        self.reconciler = OrchestratorReconciler(self)
        self.order_state.restore_active_orders()
        
        # Redis Init
        self.r = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']})
        print("DEBUG: Redis Initialized.")
        
        # IBKR backend selection:
        # - REALTIME / REALTIME_DRY: real connector
        # - BACKTEST: mock connector
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

        # Global State Defaults
        # [已统一] 所有的策略、风控参数通过 self.strategy.cfg 访问
        
        self.last_date = None
        self.mock_cash = self.cfg.INITIAL_ACCOUNT
        self._apply_live_trading_capital_limit("init_default")
        self.index_opening_prices = {}
        self.consecutive_stop_losses = 0
        self.global_cooldown_until = 0
        
        # [新增] 延时信号队列
        self.delayed_signal_queue = []
        self._gate_trace_pub_state = {}
        self._gate_counter_pub_state = {}
        self._gate_trace_ttl = 60
        # [Config Snapshot] Dashboard "🧬 策略门禁" tab 读 meta:strategy_config,
        # 由 OMS 发布 (OMS 才持有真正生效的 self.strategy.cfg); 变更时走 fingerprint 去抖.
        self._last_config_fingerprint = None
        self._entry_reject_counts = {}
        self._entry_reject_samples = {}
        self._entry_attempt_count = 0
        self._entry_pass_count = 0
        # Alpha frame readiness is the OMS strategy startup barrier. OMS may
        # restore positions/cash at boot, but strategy trading only starts after
        # a fresh ALPHA_FRAME arrives from SignalEngine/AlphaEngine.
        self.latest_alpha_frame = None
        self.latest_alpha_by_symbol = {}
        self.last_alpha_frame_ts = 0.0
        self.last_alpha_frame_id = None
        self.last_alpha_frame_wall_ts = 0.0
        self._alpha_frame_ready = False
        self._processed_alpha_frame_limit = 512
        self._processed_alpha_frame_ids = deque()
        self._processed_alpha_frame_set = set()
        self._last_alpha_barrier_warn_ts = 0.0
        # Trade signal idempotency guard:
        # 防止同一帧/同一秒重复 BUY/SELL 信号被重复执行（双进程重放、队列抖动、重启边界）。
        self._processed_trade_signal_limit = 4096
        self._processed_trade_signal_ids = deque()
        self._processed_trade_signal_set = set()
        # Seconds-level fused market data is execution-only. It refreshes fill
        # quotes and PnL references, but must never trigger StrategyCore.
        self.latest_execution_quote_by_symbol = {}
        self.last_fused_market_ts = 0.0
        self._last_stale_quote_block_warn_ts = {}
        
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
        self._ledger_seq = 0             # OMS 本地账本序号（随广播递增）
        self._boot_id = uuid.uuid4().hex[:12]
        self._oms_writer_lock_key = f"lock:oms_writer:{RUN_MODE}:db{get_redis_db()}"
        self._oms_writer_lock_value = f"{self._boot_id}:{os.getpid()}"
        self._oms_writer_lock_acquired = False
        self._last_writer_lock_refresh_ts = 0.0

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

    @staticmethod
    def _parse_oms_writer_lock_owner(value):
        """Return (owner_text, pid) from the Redis single-writer lock value."""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        owner = str(value or "")
        try:
            pid = int(owner.rsplit(":", 1)[-1])
        except Exception:
            pid = None
        return owner, pid

    @staticmethod
    def _is_local_pid_alive(pid) -> bool:
        """Best-effort local process liveness check for stale lock recovery."""
        try:
            pid = int(pid)
        except Exception:
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def _delete_oms_writer_lock_if_value_matches(self, expected_owner: str) -> bool:
        """Delete the writer lock only if it still belongs to expected_owner."""
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        try:
            deleted = self.r.eval(script, 1, self._oms_writer_lock_key, str(expected_owner))
            return int(deleted or 0) == 1
        except Exception as e:
            logger.warning(f"⚠️ [OMS Writer Lock] compare-delete failed: {e}")
            return False

    def _acquire_oms_writer_lock(self, ttl_sec: int = 60) -> bool:
        """Ensure only one OMS process can publish the realtime cash ledger."""
        if getattr(self, 'use_shared_mem', False) or self.mode == 'backtest':
            return True
        if os.environ.get("ALLOW_MULTIPLE_OMS", "").strip().lower() in ("1", "true", "yes"):
            logger.warning("⚠️ [OMS Writer Lock] ALLOW_MULTIPLE_OMS enabled; ledger single-writer guard bypassed.")
            return True
        try:
            ok = self.r.set(
                self._oms_writer_lock_key,
                self._oms_writer_lock_value,
                nx=True,
                ex=int(ttl_sec),
            )
            if ok:
                self._oms_writer_lock_acquired = True
                self._last_writer_lock_refresh_ts = time.time()
                logger.info(
                    f"🔐 [OMS Writer Lock] acquired key={self._oms_writer_lock_key} "
                    f"value={self._oms_writer_lock_value}"
                )
                return True
            existing = self.r.get(self._oms_writer_lock_key)
            if isinstance(existing, bytes):
                existing = existing.decode("utf-8", errors="ignore")
            if existing == self._oms_writer_lock_value:
                self._oms_writer_lock_acquired = True
                self._last_writer_lock_refresh_ts = time.time()
                self.r.expire(self._oms_writer_lock_key, int(ttl_sec))
                return True

            owner_text, owner_pid = self._parse_oms_writer_lock_owner(existing)
            try:
                lock_ttl = int(self.r.ttl(self._oms_writer_lock_key))
            except Exception:
                lock_ttl = None
            owner_alive = self._is_local_pid_alive(owner_pid) if owner_pid is not None else None
            if owner_pid is not None and owner_alive is False:
                logger.warning(
                    f"🧹 [OMS Writer Lock] stale owner detected for {self._oms_writer_lock_key}: "
                    f"{owner_text} | ttl={lock_ttl}. Reclaiming lock."
                )
                if self._delete_oms_writer_lock_if_value_matches(owner_text):
                    ok = self.r.set(
                        self._oms_writer_lock_key,
                        self._oms_writer_lock_value,
                        nx=True,
                        ex=int(ttl_sec),
                    )
                    if ok:
                        self._oms_writer_lock_acquired = True
                        self._last_writer_lock_refresh_ts = time.time()
                        logger.info(
                            f"🔐 [OMS Writer Lock] reclaimed key={self._oms_writer_lock_key} "
                            f"value={self._oms_writer_lock_value}"
                        )
                        return True
                logger.critical(
                    f"🚨 [OMS Writer Lock] stale owner reclaim raced/lost for {self._oms_writer_lock_key}: "
                    f"old={owner_text} current={self._oms_writer_lock_value}"
                )
                return False

            logger.critical(
                f"🚨 [OMS Writer Lock] another OMS is active for {self._oms_writer_lock_key}: "
                f"{owner_text}. Current={self._oms_writer_lock_value}. ttl={lock_ttl} "
                f"owner_pid={owner_pid} owner_alive={owner_alive}. Refusing to publish/trade."
            )
            return False
        except Exception as e:
            logger.error(f"🚨 [OMS Writer Lock] acquire failed: {e}")
            return False

    def _refresh_oms_writer_lock(self, ttl_sec: int = 60, min_interval_sec: float = 15.0) -> bool:
        if not getattr(self, '_oms_writer_lock_acquired', False):
            return True
        now = time.time()
        if now - float(getattr(self, '_last_writer_lock_refresh_ts', 0.0) or 0.0) < min_interval_sec:
            return True
        try:
            existing = self.r.get(self._oms_writer_lock_key)
            if isinstance(existing, bytes):
                existing = existing.decode("utf-8", errors="ignore")
            if existing != self._oms_writer_lock_value:
                logger.critical(
                    f"🚨 [OMS Writer Lock] lost ownership key={self._oms_writer_lock_key} "
                    f"existing={existing} current={self._oms_writer_lock_value}. Stop ledger publish."
                )
                self._oms_writer_lock_acquired = False
                return False
            self.r.expire(self._oms_writer_lock_key, int(ttl_sec))
            self._last_writer_lock_refresh_ts = now
            return True
        except Exception as e:
            logger.error(f"🚨 [OMS Writer Lock] refresh failed: {e}")
            return False

    def _reconstruct_market_packet(self, payload_list):
        """[Parity] 将 Pitcher 发来的分品种 Payload 列表重组为 S4 模拟器所需的 Batch 格式"""
        if isinstance(payload_list, dict):
            payload_list = [payload_list]
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
        
        try:
            from config import TAG_TO_INDEX, option_bucket_tag
            idx_p = TAG_TO_INDEX.get(option_bucket_tag(-1), 0)
            idx_c = TAG_TO_INDEX.get(option_bucket_tag(1), 2)
        except Exception:
            idx_p, idx_c = 0, 2

        for p in payload_list:
            reconstructed['symbols'].append(p.get('symbol'))
            reconstructed['stock_price'].append(p.get('stock', {}).get('close', 0.0))
            
            buckets = p.get('option_buckets', [])
            put_bk = buckets[idx_p] if len(buckets) > idx_p else []
            call_bk = buckets[idx_c] if len(buckets) > idx_c else []
            
            # 数值对齐 S4: 0=mid/price, 8=bid, 9=ask
            reconstructed['feed_put_price'].append(float(put_bk[0]) if len(put_bk) > 0 else 0.0)
            reconstructed['feed_put_bid'].append(float(put_bk[8]) if len(put_bk) > 8 else 0.0)
            reconstructed['feed_put_ask'].append(float(put_bk[9]) if len(put_bk) > 9 else 0.0)
            
            reconstructed['feed_call_price'].append(float(call_bk[0]) if len(call_bk) > 0 else 0.0)
            reconstructed['feed_call_bid'].append(float(call_bk[8]) if len(call_bk) > 8 else 0.0)
            reconstructed['feed_call_ask'].append(float(call_bk[9]) if len(call_bk) > 9 else 0.0)
            
        return reconstructed

    def _cache_execution_market_packet(self, market_packet: dict):
        """Cache latest fused quotes for execution/PnL only."""
        if not market_packet:
            return

        symbols = market_packet.get('symbols')
        if symbols is None:
            symbols = []
        ts = float(market_packet.get('ts', 0.0) or 0.0)
        self.last_fused_market_ts = max(float(getattr(self, 'last_fused_market_ts', 0.0) or 0.0), ts)

        def _arr(name):
            value = market_packet.get(name)
            return [] if value is None else value

        stock_prices = _arr('stock_price')
        call_prices = _arr('feed_call_price')
        put_prices = _arr('feed_put_price')
        call_bids = _arr('feed_call_bid')
        call_asks = _arr('feed_call_ask')
        put_bids = _arr('feed_put_bid')
        put_asks = _arr('feed_put_ask')

        for i, sym in enumerate(symbols):
            if not sym:
                continue

            def _safe(arr, default=0.0):
                try:
                    return float(arr[i]) if i < len(arr) else default
                except Exception:
                    return default

            stock_price = _safe(stock_prices)
            c_mid = self._get_fair_market_price(_safe(call_prices), _safe(call_bids), _safe(call_asks))
            p_mid = self._get_fair_market_price(_safe(put_prices), _safe(put_bids), _safe(put_asks))
            now_wall_ts = time.time()
            quote = {
                'ts': ts,
                'wall_ts': now_wall_ts,
                'stock_price': stock_price,
                'call_price': c_mid,
                'put_price': p_mid,
                'call_bid': _safe(call_bids),
                'call_ask': _safe(call_asks),
                'put_bid': _safe(put_bids),
                'put_ask': _safe(put_asks),
            }
            self.latest_execution_quote_by_symbol[sym] = quote
            if get_domain_shadow_router is not None:
                try:
                    st = self.states.get(sym)
                    legacy_position = int(getattr(st, 'position', 0) or 0) if st is not None else None
                    get_domain_shadow_router().on_execution_quote(sym, quote, legacy_position=legacy_position)
                except Exception as e:
                    logger.warning(f"[DomainShadow] execution_quote hook failed for {sym}: {e}")

            # Seconds quotes remain execution-layer facts, but active-position
            # peak ROI must be mode-neutral. Otherwise backtest cannot validate
            # the same profit-protection path used by realtime.
            st = self.states.get(sym)
            if st is not None and int(getattr(st, 'position', 0) or 0) != 0:
                entry_p = float(getattr(st, 'entry_price', 0.0) or 0.0)
                if entry_p > 0.01:
                    px = c_mid if int(st.position) == 1 else p_mid
                    if px > 0.01:
                        roi_now = (px - entry_p) / entry_p
                        st.max_roi = max(float(getattr(st, 'max_roi', -1.0) or -1.0), roi_now)

    def _second_quote_for_symbol(self, sym: str) -> dict:
        return dict(self.latest_execution_quote_by_symbol.get(sym) or {})

    def _reset_second_exit_streak(self, st) -> None:
        st.second_exit_streak_reason = ""
        st.second_exit_streak_count = 0
        st.second_exit_last_ts = 0.0

    def _confirm_second_exit_streak(self, st, reason: str, curr_ts: float, threshold: int) -> bool:
        threshold = max(1, int(threshold or 1))
        reason_key = str(reason or "")
        last_ts = float(getattr(st, 'second_exit_last_ts', 0.0) or 0.0)
        same_reason = reason_key == str(getattr(st, 'second_exit_streak_reason', '') or '')
        continuous = last_ts <= 0.0 or abs(float(curr_ts or 0.0) - last_ts) <= 2.5
        if same_reason and continuous:
            st.second_exit_streak_count = int(getattr(st, 'second_exit_streak_count', 0) or 0) + 1
        else:
            st.second_exit_streak_reason = reason_key
            st.second_exit_streak_count = 1
        st.second_exit_last_ts = float(curr_ts or 0.0)
        return int(st.second_exit_streak_count) >= threshold

    def _build_second_dynamic_exit_signal(self, sym: str, st, quote: dict, curr_ts: float) -> Optional[Dict[str, Any]]:
        side = int(getattr(st, 'position', 0) or 0)
        if side == 0 or bool(getattr(st, 'is_pending', False)):
            self._reset_second_exit_streak(st)
            return None

        entry_p = float(getattr(st, 'entry_price', 0.0) or 0.0)
        if entry_p <= 0.01:
            self._reset_second_exit_streak(st)
            return None

        if side == 1:
            curr_p = float(quote.get('call_price', 0.0) or 0.0)
            bid = float(quote.get('call_bid', 0.0) or 0.0)
            ask = float(quote.get('call_ask', 0.0) or 0.0)
        else:
            curr_p = float(quote.get('put_price', 0.0) or 0.0)
            bid = float(quote.get('put_bid', 0.0) or 0.0)
            ask = float(quote.get('put_ask', 0.0) or 0.0)

        if curr_p <= 0.01 or bid <= 0.01 or ask <= 0.01 or ask < bid:
            self._reset_second_exit_streak(st)
            return None

        spread_pct = (ask - bid) / max(curr_p, 0.01)
        max_exit_spread = float(getattr(self.cfg, 'MAX_SPREAD_PCT_EXIT', 0.20) or 0.20)
        if spread_pct > max_exit_spread:
            self._reset_second_exit_streak(st)
            return None

        roi = (curr_p - entry_p) / entry_p
        st.max_roi = max(float(getattr(st, 'max_roi', -1.0) or -1.0), roi)
        max_roi = float(getattr(st, 'max_roi', roi) or roi)
        held_mins = self._calc_trading_minutes(float(getattr(st, 'entry_ts', 0.0) or 0.0), curr_ts)

        reason = ""
        # EOD still belongs to the fast path because stale holdings are worse
        # than giving the minute loop one more chance near the close.
        try:
            ny_now = datetime.fromtimestamp(float(curr_ts), timezone('America/New_York'))
            close_h = int(getattr(self.cfg, 'CLOSE_HOUR', 15) or 15)
            close_m = int(getattr(self.cfg, 'CLOSE_MINUTE', 40) or 40)
            if ny_now.time() >= dt_time(close_h, close_m):
                reason = f"TIGHT_1S_EOD_CLEAR:{held_mins:.1f}m"
        except Exception:
            reason = ""

        if not reason and held_mins >= 1.0:
            ladder = list(getattr(self.cfg, 'LADDER_TIGHT', []) or [])
            if held_mins < 3.0:
                # During the opening minutes of a position, only protect
                # meaningful profits; tiny early gains are too spread-sensitive.
                ladder = [(trigger, floor) for trigger, floor in ladder if float(trigger) >= 0.08]
            for trigger, floor in sorted(ladder, reverse=True):
                trigger = float(trigger)
                floor = float(floor)
                if max_roi >= trigger:
                    if roi < floor:
                        reason = (
                            f"TIGHT_1S_STEP_PROT:{max_roi:.1%}->{roi:.1%}"
                            f"|T:{trigger:.2f}|held={held_mins:.1f}m"
                        )
                    break

        if not reason and held_mins >= 3.0:
            flash_trigger = float(getattr(self.cfg, 'FLASH_PROTECT_TRIGGER', 0.05) or 0.05)
            flash_exit = float(getattr(self.cfg, 'FLASH_PROTECT_EXIT', 0.02) or 0.02)
            if max_roi >= flash_trigger and roi <= flash_exit:
                reason = f"TIGHT_1S_FLASH_PROT:{max_roi:.1%}->{roi:.1%}|held={held_mins:.1f}m"

        if not reason and held_mins >= 3.0:
            abs_stop = float(getattr(self.cfg, 'ABSOLUTE_STOP_LOSS', -0.15) or -0.15)
            soft_stop = float(getattr(self.cfg, 'STOP_LOSS', -0.10) or -0.10)
            if roi <= abs_stop:
                candidate = f"TIGHT_1S_ABS_STOP:{roi:.1%}|held={held_mins:.1f}m"
                if self._confirm_second_exit_streak(st, "TIGHT_1S_ABS_STOP", curr_ts, threshold=2):
                    reason = candidate
            elif roi <= soft_stop:
                candidate = f"TIGHT_1S_STOP:{roi:.1%}|held={held_mins:.1f}m"
                if self._confirm_second_exit_streak(st, "TIGHT_1S_STOP", curr_ts, threshold=3):
                    reason = candidate
            else:
                self._reset_second_exit_streak(st)
        elif reason:
            self._reset_second_exit_streak(st)
        else:
            self._reset_second_exit_streak(st)

        if not reason:
            return None

        return {
            'action': 'SELL',
            'dir': side,
            'target_side': side,
            'reason': reason,
            'price': curr_p,
            'market_price': curr_p,
            'bid': bid,
            'ask': ask,
            'meta': {
                'source': 'second_dynamic_exit',
                'roi': roi,
                'max_roi': max_roi,
                'held_mins': held_mins,
                'spread_pct': spread_pct,
                'bid': bid,
                'ask': ask,
            },
        }

    async def _evaluate_second_dynamic_exits(self, curr_ts: float):
        for sym, st in self.states.items():
            if int(getattr(st, 'position', 0) or 0) == 0:
                continue
            quote = self._second_quote_for_symbol(sym)
            if not quote:
                continue
            try:
                quote_ts = float(quote.get('ts', 0.0) or 0.0)
            except Exception:
                quote_ts = 0.0
            if quote_ts > 0 and curr_ts > 0 and abs(float(curr_ts) - quote_ts) > max(2.5, float(OMS_MAX_QUOTE_STALE_SEC)):
                continue
            exit_sig = self._build_second_dynamic_exit_signal(sym, st, quote, curr_ts)
            if not exit_sig:
                continue
            if self._should_block_strategy_on_stale_quote(
                sym,
                curr_ts,
                'SELL',
                reason=exit_sig.get('reason', ''),
                frame_has_quote=False,
            ):
                continue
            stock_price = float(quote.get('stock_price', getattr(st, 'last_price', 0.0)) or 0.0)
            logger.warning(
                f"⚡ [OMS 1s Dynamic Exit] {sym} | {exit_sig.get('reason')} "
                f"| px={float(exit_sig.get('price', 0.0) or 0.0):.4f}"
            )
            await self._submit_strategy_order(
                'SELL',
                sym,
                exit_sig,
                stock_price,
                curr_ts,
                -1,
                frame_id=f"1s:{int(float(curr_ts or 0.0))}",
                allow_delay_queue=False,
            )

    def _refresh_signal_from_execution_quote(self, payload: dict):
        """Refresh execution price fields from latest 1s quote before order handling."""
        if not isinstance(payload, dict):
            return
        action = str(payload.get('action', '')).upper()
        if action not in ('BUY', 'SELL'):
            return
        source = payload.get('source')
        # OMS Strategy decisions are minute/ALPHA_FRAME based. For immediate
        # strategy orders, trust the quote snapshot embedded in the frame; the
        # 1s cache is only allowed to refresh delayed/external orders. Otherwise
        # S4 parity can drift by executing a minute signal on the next second's
        # option quote.
        if source == 'oms_strategy_v8' and not payload.get('_delay_released'):
            return
        sym = payload.get('symbol')
        if not sym:
            return
        quote = self.latest_execution_quote_by_symbol.get(sym)
        if not quote:
            return
        sig = payload.get('sig')
        if not isinstance(sig, dict):
            return
        try:
            sig_ts = float(payload.get('_execution_ts', payload.get('ts', 0.0)) or 0.0)
            quote_ts = float(quote.get('ts', 0.0) or 0.0)
        except Exception:
            sig_ts, quote_ts = 0.0, 0.0
        try:
            max_quote_lag = float(payload.get('max_quote_lag_sec', 3.0) or 3.0)
        except Exception:
            max_quote_lag = 3.0
        # Execution quotes may assist fills only if they are tightly anchored to
        # the order execution timestamp. This prevents a stale/future 1s quote
        # from changing the strategy intent that came from a minute ALPHA_FRAME.
        if sig_ts > 0 and quote_ts > 0 and abs(quote_ts - sig_ts) > max_quote_lag:
            return

        side = int(sig.get('dir') or sig.get('target_side') or 0)
        if action == 'SELL':
            st = self.states.get(sym)
            side = int(getattr(st, 'position', 0) or side or 0)
        if side == 0:
            return

        if side == 1:
            px = float(quote.get('call_price', 0.0) or 0.0)
            bid = float(quote.get('call_bid', 0.0) or 0.0)
            ask = float(quote.get('call_ask', 0.0) or 0.0)
        else:
            px = float(quote.get('put_price', 0.0) or 0.0)
            bid = float(quote.get('put_bid', 0.0) or 0.0)
            ask = float(quote.get('put_ask', 0.0) or 0.0)

        if px <= 0.01:
            return
        sig['price'] = px
        sig['market_price'] = px
        sig['bid'] = bid
        sig['ask'] = ask
        meta = dict(sig.get('meta', {}) or {})
        meta['bid'] = bid
        meta['ask'] = ask
        sig['meta'] = meta
        payload['stock_price'] = float(quote.get('stock_price', payload.get('stock_price', 0.0)) or 0.0)

    def _process_fused_market_for_execution(self, payload):
        """Handle seconds-level fused data as execution market data only."""
        market_packet = self._reconstruct_market_packet(payload)
        if not market_packet:
            return False
        self._cache_execution_market_packet(market_packet)
        if self.ibkr and hasattr(self.ibkr, 'record_market_data'):
            self.ibkr.record_market_data(market_packet)
        return True

    # ------------------------------------------------------------------
    # [S2 双引擎 / 秒级回放] 分钟窗口编排 (与 preprocess/.../s4_run_historical_replay_s2_1s 对齐)
    #
    # 语义:
    #   minute_ts (key)     — 该分钟左边界 epoch 秒, 与 signal_packet['ts'] 对齐时标记 is_new_minute
    #   quotes (value)      — 该分钟内全部 1s 行情包 (仅执行层, 不触发 StrategyCore)
    #   上一分钟的 alpha      — 通过 merge_asof backward 对齐到当前秒 (脚本侧完成); 分钟边界上的
    #                         ALPHA/signal 只在 minute_ts 评估一次 (SE 或 OMS ALPHA_FRAME)。
    #
    # 同步契约: 同一分钟内, 不允许在上一笔 BUY/SELL 尚未 process_trade_signal 完毕时推进下一秒 —
    # 通过「每秒末尾 drain_trade_signal_queue」实现; OMS 内 _execute_* 已是 await, 天然顺序化。
    # ------------------------------------------------------------------

    async def drain_trade_signal_queue(self) -> int:
        """排空注入的 asyncio.Queue (SE→OMS 或测试桩), 保证顺序执行所有挂起的交易指令。"""
        q = getattr(self, 'signal_queue', None)
        if q is None:
            return 0
        n = 0
        while True:
            try:
                item = q.get_nowait()
            except (Empty, asyncio.QueueEmpty):
                break
            try:
                await self.process_trade_signal(item)
            finally:
                try:
                    q.task_done()
                except ValueError:
                    pass
                n += 1
        return n

    def _record_market_for_replay(self, market_packet: dict):
        """MockIBKR 行情录制; precalc_alpha 存在则一并写入。"""
        if not market_packet or not self.ibkr or not hasattr(self.ibkr, 'record_market_data'):
            return
        try:
            self.ibkr.record_market_data(
                market_packet,
                alphas=market_packet.get('precalc_alpha'),
            )
        except TypeError:
            self.ibkr.record_market_data(market_packet)

    async def cache_minute_signal_for_execution(self, signal_packet: dict):
        """分钟级信号帧: 只刷新执行侧缓存 + 录制行情 + 排空跨进程队列。

        典型调用顺序 (与 S4 脚本一致):
          1) await signal_engine.process_batch(signal_packet)   # 仅 alpha / 推理
          2) await exec_engine.cache_minute_signal_for_execution(signal_packet)
        """
        if not signal_packet:
            return
        self._cache_execution_market_packet(signal_packet)
        self._record_market_for_replay(signal_packet)
        await self.drain_trade_signal_queue()

    async def ingest_execution_second_sync(self, market_packet: dict):
        """单个 1s 行情包: 执行缓存 → 录制 → SYNC 屏障 → 再次排空队列。

        对应 S4 中每秒循环体的合体版; 不包含策略推理 (StrategyCore 仅在分钟帧运行)。
        """
        if not market_packet:
            return
        ts = float(market_packet.get('ts', 0.0) or 0.0)
        self._cache_execution_market_packet(market_packet)
        self._record_market_for_replay(market_packet)
        await self._evaluate_second_dynamic_exits(ts)
        await self.process_trade_signal({'action': 'SYNC', 'ts': ts, 'payload': {}})
        await self.drain_trade_signal_queue()

    async def execute_minute_execution_phase(self, second_quotes: list):
        """分钟内全部秒级行情 (有序). 必须在分钟 alpha 帧处理完之后调用。"""
        if not second_quotes:
            return
        for pkt in second_quotes:
            await self.ingest_execution_second_sync(pkt)

    # ------------------------------------------------------------------
    # [Step A] 分钟级执行窗口 — 回测/实盘统一入口
    #
    # 契约:
    #   调用顺序:
    #     1) await signal_engine.process_batch(window.alpha_frame)    # 仅 alpha 推理
    #     2) await exec_engine.execute_window(window)                 # 分钟边界 + 60 秒
    #
    #   execute_window 内部职责:
    #     Phase 1 (minute edge): cache → record → drain signal_queue
    #       (strategy 的 decide_entry/check_exit 在此间通过 ALPHA_FRAME 统一跑一次)
    #     Phase 2 (1s loop):     for q in quotes_1s: cache → record → SYNC → drain
    #       (当前实现仅 cache + SYNC, 不触发额外 strategy / fill;
    #        后续 Step B 在此处插秒级 tight-exit, Step C/D 插拆单 fill 与撤改单)
    #
    # 本步骤的目标是 bit-identical: 行为与旧的三行散调用完全一致, 只是把契约显式化.
    # ------------------------------------------------------------------

    async def execute_window(self, window: "ExecutionWindow", validate: bool = True):
        """分钟级执行窗口统一入口. 保证回测/实盘共享同一段 OMS 编排代码."""
        if window is None:
            return
        if validate:
            errs = window.validate(strict=False)
            if errs:
                # 宽松策略: 打警告不阻断, 便于回放数据轻微缺秒时仍能跑完
                logger.warning(
                    f"⚠️ [ExecutionWindow] {window.summary()} 契约告警: "
                    f"{'; '.join(errs[:3])}"
                    + (" ..." if len(errs) > 3 else "")
                )

        # Phase 1: 分钟边界 — 触发 strategy 决策 + 排空因决策产生的交易信号
        await self.cache_minute_signal_for_execution(window.alpha_frame)

        # Phase 2: 秒级循环 — 逐秒 ingest 行情 / SYNC 屏障 / drain (当前行为)
        #   后续步骤 (B/C/D) 将在 ingest_execution_second_sync 内部插入:
        #   - Step B: 秒级 tight exits (X4/X9a/X10.trailing/X10.flash/X1.eod)
        #   - Step C: chunked fill on ask_size depth
        #   - Step D: LMT cancel-replace on timeout
        await self.execute_minute_execution_phase(window.quotes_1s)


       
        
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

    @staticmethod
    def _state_uses_entry_slot(st) -> bool:
        if int(getattr(st, 'position', 0) or 0) != 0:
            return True
        if bool(getattr(st, 'entry_slot_reserved', False)):
            return True
        if bool(getattr(st, 'is_pending', False)):
            side = str(getattr(st, 'pending_action', None) or getattr(st, 'pending_side', '') or '').upper()
            return side == 'BUY'
        return False

    def _count_entry_slots(self) -> int:
        return sum(1 for st in self.states.values() if self._state_uses_entry_slot(st))

    def _apply_live_trading_capital_limit(self, reason: str = "") -> None:
        """Clamp OMS cash ledger to the configured live trading cap.

        This is a deliberate risk budget control for RUN_MODE=REALTIME only.
        It does not affect BACKTEST / REALTIME_DRY, and it does not try to
        mirror the broker's true account size.
        """
        if self.mode != 'realtime' or IS_REALTIME_DRY:
            return
        cap = float(get_runtime_live_trading_capital_limit(default_value=LIVE_TRADING_CAPITAL_LIMIT, r=getattr(self, "r", None)) or 0.0)
        if cap <= 0.0:
            return
        try:
            before = float(self.mock_cash or 0.0)
        except Exception:
            before = 0.0
        after = min(before, cap) if before > 0.0 else cap
        if abs(after - before) <= 1e-9:
            return
        self.mock_cash = after
        logger.warning(
            f"🧮 [Live Capital Cap] {reason or 'apply'} | cash capped from ${before:,.2f} to ${after:,.2f} "
            f"| LIVE_TRADING_CAPITAL_LIMIT=${cap:,.2f}"
        )

    def _should_delay_signal(self, action: str, source: str = None) -> bool:
        if str(source or '').strip().lower() == 'dashboard_manual_close':
            return False
        return OMS_SIGNAL_DELAY_BARS > 0 and str(action).upper() in OMS_SIGNAL_DELAY_ACTIONS

    def _eligible_trade_ts(self, signal_ts: float) -> float:
        return float(signal_ts) + float(OMS_SIGNAL_DELAY_BARS * 60)

    async def _queue_delayed_signal(self, payload: dict):
        sym = payload.get('symbol')
        action = str(payload.get('action', '')).upper()
        curr_ts = float(payload.get('ts', 0.0) or 0.0)
        eligible_ts = self._eligible_trade_ts(curr_ts)
        queued_payload = copy.deepcopy(payload)
        queued_payload['_decision_ts'] = curr_ts
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
            released['_execution_ts'] = float(curr_ts)
            released['ts'] = float(curr_ts)
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
        """确保所需的 Redis 消费者组存在

        [重启行为]
        - 首次创建：回放/回测用 '0'（从头消费）；实盘/DRY 用 '$'（只读新）
        - 已存在组：实盘/DRY 默认 XGROUP SETID $，丢弃启动前的积压，防止
          OMS 重启时把几千条旧信号一次性回放导致重复下单 / 资金异常
          兜底开关：OMS_RESUME_FROM_BACKLOG=1 时保留原 last-delivered-id
        """
        from config import STREAM_FUSED_MARKET, IS_SIMULATED
        
        # 🚀 [核心修复] OMS 必须优先寻找自己的专有组和专有流，防止被通用的 GROUP_FEATURE 带偏
        from config import GROUP_OMS, STREAM_ORCH_SIGNAL
        target_group = REDIS_CFG.get('oms_group') or REDIS_CFG.get('orch_group') or GROUP_OMS
        self._oms_group = target_group
        target_stream = REDIS_CFG.get('signal_stream') or STREAM_ORCH_SIGNAL
        
        streams_to_init = [target_stream]
        if self.mode != 'backtest' or IS_SIMULATED:
            streams_to_init.append(STREAM_FUSED_MARKET)

        # 仅实盘/DRY 场景下默认丢弃积压；回放/回测要求完整重放
        resume_env = os.environ.get("OMS_RESUME_FROM_BACKLOG", "0").strip().lower()
        resume_backlog = resume_env in ("1", "true", "yes")
        skip_backlog_on_restart = (not IS_SIMULATED) and (not resume_backlog)

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
                elif skip_backlog_on_restart:
                    # [🆕 重启丢弃积压] 主动推进 last-delivered-id 到 $，避免重放历史信号
                    try:
                        self.r.xgroup_setid(s, target_group, "$")
                        db_idx = self.r.connection_pool.connection_kwargs.get('db')
                        logger.warning(
                            f"🧹 [OMS] Restart detected → XGROUP SETID {target_group} {s} $ "
                            f"(drop backlog on DB {db_idx}). "
                            f"Set OMS_RESUME_FROM_BACKLOG=1 to keep backlog."
                        )
                    except Exception as e:
                        logger.error(f"[OMS] Failed to SETID on {s}: {e}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e): 
                    logger.error(f"Group Create Error on {s}: {e}")

    # 在 SystemOrchestratorV8 类中添加以下方法

    def _calc_trading_minutes(self, start_ts, end_ts):
        try:
            return max(0.0, (float(end_ts) - float(start_ts)) / 60.0)
        except Exception:
            return 0.0

    def _bump_entry_reject(self, reason, sym=None, extra=None):
        try:
            self._entry_reject_counts[reason] = self._entry_reject_counts.get(reason, 0) + 1
            if sym and reason not in self._entry_reject_samples:
                snap = {'sym': sym}
                if isinstance(extra, dict):
                    for k, v in list(extra.items())[:6]:
                        try:
                            snap[k] = float(v) if isinstance(v, (int, float)) else str(v)[:40]
                        except Exception:
                            pass
                self._entry_reject_samples[reason] = snap
        except Exception:
            pass

    @staticmethod
    def _diff_counter_map(after, before):
        delta = {}
        keys = set((before or {}).keys()) | set((after or {}).keys())
        for key in keys:
            diff = int((after or {}).get(key, 0) or 0) - int((before or {}).get(key, 0) or 0)
            if diff > 0:
                delta[str(key)] = diff
        return delta

    def _publish_entry_diag(
        self,
        *,
        frame: dict,
        items_count: int,
        entry_candidates_count: int,
        min_symbols: int,
        reject_before: dict,
        attempt_before: int,
        pass_before: int,
        diag_reason: str = "",
        diag_extra: dict | None = None,
    ):
        try:
            attempts_delta = int(getattr(self, '_entry_attempt_count', 0) or 0) - int(attempt_before or 0)
            passes_delta = int(getattr(self, '_entry_pass_count', 0) or 0) - int(pass_before or 0)
            reject_delta = self._diff_counter_map(getattr(self, '_entry_reject_counts', {}), reject_before or {})

            top_reject = ""
            top_reject_count = 0
            if reject_delta:
                top_reject, top_reject_count = max(reject_delta.items(), key=lambda kv: kv[1])

            frame_id = str(frame.get('frame_id') or '')
            frame_ts = float(frame.get('ts', 0.0) or 0.0)
            payload = {
                'frame_id': frame_id,
                'frame_ts': f"{frame_ts:.3f}",
                'items': str(int(items_count or 0)),
                'entry_attempts': str(int(attempts_delta)),
                'entry_candidates': str(int(entry_candidates_count or 0)),
                'entry_passes': str(int(passes_delta)),
                'min_symbols': str(int(min_symbols or 0)),
                'diag_reason': str(diag_reason or top_reject or ''),
                'top_reject': str(top_reject or ''),
                'top_reject_count': str(int(top_reject_count or 0)),
                'updated_at': f"{time.time():.3f}",
            }
            if diag_extra:
                for k, v in list(diag_extra.items())[:6]:
                    payload[f"extra_{k}"] = str(v)
            sample = self._entry_reject_samples.get(top_reject or diag_reason, {})
            if sample:
                payload['sample'] = json.dumps(sample, ensure_ascii=False)

            try:
                pipe = self.r.pipeline()
                pipe.delete("meta:oms_entry_diag")
                pipe.hset("meta:oms_entry_diag", mapping=payload)
                pipe.expire("meta:oms_entry_diag", 180)
                pipe.execute()
            except Exception:
                pass

            if attempts_delta > 0 or diag_reason:
                logger.warning(
                    f"⚠️ [OMS-Entry-Diag] frame={frame_id or 'na'} ts={int(frame_ts) if frame_ts > 0 else 0} "
                    f"| items={items_count} candidates={entry_candidates_count} min_symbols={min_symbols} "
                    f"| attempts={attempts_delta} passes={passes_delta} "
                    f"| diag_reason={diag_reason or top_reject or 'none'} "
                    f"| top_reject={top_reject or 'none'}:{top_reject_count} "
                    f"| reject_delta={reject_delta or {}}"
                )
        except Exception as e:
            logger.warning(f"⚠️ [OMS-Entry-Diag] publish failed: {e}")

    def _publish_strategy_config_snapshot(self, force: bool = False):
        """把 OMS 当前持有的 self.strategy.cfg 展平后写入 Redis meta:strategy_config.

        Dashboard "🧬 策略门禁" tab 直接读这张 hash 渲染静态参数表.
        OMS 才是真正跑 StrategyCore 的地方, 所以 snapshot 必须从这里发, 否则 SE 的
        self.cfg (独立 StrategyConfig 实例) 可能跟 OMS 漂移.
        - fingerprint 去抖: 相同内容不重写 (除非 force=True)
        - key:   meta:strategy_config  (hash, TTL=24h)
        - 字段:  <param_name> -> <str(value)>
                 额外: __version__ / __loaded_at__ / __core_version__ 供 Dashboard 识别
        """
        try:
            strat = getattr(self, 'strategy', None)
            cfg_src = getattr(strat, 'cfg', None) or getattr(self, 'cfg', None)
            if cfg_src is None:
                return
            try:
                from dataclasses import asdict
                cfg_dict = asdict(cfg_src)
            except Exception:
                cfg_dict = {k: getattr(cfg_src, k) for k in dir(cfg_src) if not k.startswith('_')}

            try:
                import hashlib as _hl, json as _json
                raw = _json.dumps(cfg_dict, default=str, sort_keys=True)
                fp = _hl.md5(raw.encode('utf-8')).hexdigest()[:10]
            except Exception:
                fp = "unknown"

            if (not force) and self._last_config_fingerprint == fp:
                return
            self._last_config_fingerprint = fp

            mapping = {k: str(v) for k, v in cfg_dict.items()}
            mapping['__version__'] = fp
            mapping['__loaded_at__'] = f"{time.time():.0f}"
            mapping['__core_version__'] = str(globals().get('ACTIVE_STRATEGY_CORE_VERSION', 'v0'))
            pipe = self.r.pipeline()
            pipe.delete("meta:strategy_config")
            pipe.hset("meta:strategy_config", mapping=mapping)
            pipe.expire("meta:strategy_config", 24 * 3600)
            pipe.execute()
            logger.info(f"🧬 [Config Snapshot] 已广播 strategy_config0 | fp={fp} | fields={len(mapping)}")
        except Exception as e:
            logger.warning(f"⚠️ [Config Snapshot] publish failed: {e}")

    def _publish_gate_trace(self, sym: str, kind: str, result_sig, event_ts=None):
        try:
            try:
                trace = self.strategy.get_last_gate_trace()
            except Exception:
                trace = list(getattr(self.strategy, '_last_gate_trace', []) or [])
            if not trace:
                return

            event_ts_safe = None
            if event_ts is not None:
                try:
                    event_ts_safe = float(event_ts)
                    if event_ts_safe <= 0:
                        event_ts_safe = None
                except (TypeError, ValueError):
                    event_ts_safe = None

            if result_sig:
                act = result_sig.get('action')
                if act == 'BUY':
                    result_label = 'BUY'
                elif act == 'SELL':
                    reason = (result_sig.get('reason') or 'SELL').split('|')[0][:32]
                    result_label = f"SELL:{reason}"
                else:
                    result_label = 'PASS'
            else:
                last_block = next((g.get('gate') for g in reversed(trace) if g.get('status') == 'block'), None)
                result_label = f"REJECT:{last_block}" if last_block else "PASS"

            last_block_gate = next((g.get('gate') for g in reversed(trace) if g.get('status') == 'block'), None)
            key = (sym, kind)
            curr = (result_label, last_block_gate)

            if last_block_gate:
                try:
                    ts_for_day = event_ts_safe if event_ts_safe is not None else time.time()
                    ny_date = datetime.fromtimestamp(ts_for_day, timezone('America/New_York')).strftime('%Y%m%d')
                    counter_key = f"meta:gate_counter:{ny_date}"
                    prev_cnt = self._gate_counter_pub_state.get(key)
                    curr_cnt = (ny_date, last_block_gate)
                    if prev_cnt != curr_cnt:
                        self.r.hincrby(counter_key, last_block_gate, 1)
                        self.r.expire(counter_key, 48 * 3600)
                        self._gate_counter_pub_state[key] = curr_cnt
                except Exception:
                    pass

            if self._gate_trace_pub_state.get(key) == curr:
                try:
                    self.r.expire(f"meta:gate_trace:{sym}", self._gate_trace_ttl)
                except Exception:
                    pass
                return
            self._gate_trace_pub_state[key] = curr

            payload = {
                'kind': kind,
                'ts': f"{event_ts_safe:.3f}" if event_ts_safe is not None else f"{time.time():.3f}",
                'result': result_label,
                'last_block': last_block_gate or '',
                'trace_json': json.dumps(trace, ensure_ascii=False),
            }
            pipe = self.r.pipeline()
            pipe.hset(f"meta:gate_trace:{sym}", mapping=payload)
            pipe.expire(f"meta:gate_trace:{sym}", self._gate_trace_ttl)
            pipe.execute()
        except Exception as e:
            if not getattr(self, '_gate_pub_err_logged', False):
                logger.warning(f"⚠️ [OMS Gate Trace Pub] failed: {e}")
                self._gate_pub_err_logged = True

    async def _submit_strategy_order(
        self,
        action,
        sym,
        sig,
        stock_price,
        curr_ts,
        batch_idx,
        frame_id=None,
        allow_delay_queue=True,
    ):
        payload = {
            'ts': curr_ts,
            'symbol': sym,
            'action': action,
            'sig': sig,
            'stock_price': stock_price,
            'batch_idx': batch_idx,
            'frame_id': frame_id,
            'source': 'oms_strategy_v8',
            'prices': {sym: getattr(self.states.get(sym), 'last_opt_price', 0.0)},
        }
        await self._handle_trade_signal(payload, allow_delay_queue=allow_delay_queue)

    def _trade_signal_key(self, payload: dict) -> str:
        """Build a deterministic dedupe key for BUY/SELL handling."""
        action = str(payload.get('action', '')).upper()
        sym = str(payload.get('symbol', '') or '')
        source = str(payload.get('source', '') or '')
        frame_id = str(payload.get('frame_id', '') or '')
        batch_idx = int(payload.get('batch_idx', -1) or -1)
        sig = payload.get('sig') if isinstance(payload.get('sig'), dict) else {}
        reason = str((sig or {}).get('reason', '') or '')
        try:
            ts_bucket = int(float(payload.get('ts', 0.0) or 0.0))
        except Exception:
            ts_bucket = 0
        dir_val = int((sig or {}).get('dir', (sig or {}).get('target_side', 0)) or 0)
        return "|".join([
            action, sym, source, frame_id, str(batch_idx), str(ts_bucket), str(dir_val), reason
        ])

    def _is_duplicate_trade_signal(self, payload: dict) -> bool:
        key = self._trade_signal_key(payload)
        return key in self._processed_trade_signal_set

    def _remember_trade_signal(self, payload: dict):
        key = self._trade_signal_key(payload)
        if not key or key in self._processed_trade_signal_set:
            return
        self._processed_trade_signal_ids.append(key)
        self._processed_trade_signal_set.add(key)
        while len(self._processed_trade_signal_ids) > self._processed_trade_signal_limit:
            old = self._processed_trade_signal_ids.popleft()
            self._processed_trade_signal_set.discard(old)

    def _alpha_frame_key(self, frame: dict, curr_ts: float = None) -> str:
        frame_id = frame.get('frame_id')
        if frame_id not in (None, ''):
            return f"id:{frame_id}"
        ts_val = curr_ts if curr_ts is not None else frame.get('ts')
        try:
            return f"ts:{int(float(ts_val))}"
        except Exception:
            return f"wall:{int(time.time())}"

    def _is_duplicate_alpha_frame(self, frame: dict, curr_ts: float = None) -> bool:
        return self._alpha_frame_key(frame, curr_ts) in self._processed_alpha_frame_set

    def _remember_alpha_frame_key(self, frame_key: str):
        if not frame_key or frame_key in self._processed_alpha_frame_set:
            return
        self._processed_alpha_frame_ids.append(frame_key)
        self._processed_alpha_frame_set.add(frame_key)
        while len(self._processed_alpha_frame_ids) > self._processed_alpha_frame_limit:
            old = self._processed_alpha_frame_ids.popleft()
            self._processed_alpha_frame_set.discard(old)

    def _cache_latest_alpha_frame(self, frame: dict, curr_ts: float):
        """Cache the latest strategy facts without rebuilding alpha history."""
        items = frame.get('items') or []
        self.latest_alpha_frame = frame
        self.latest_alpha_by_symbol = {
            item.get('symbol'): item
            for item in items
            if item.get('symbol')
        }
        self.last_alpha_frame_ts = float(curr_ts or 0.0)
        self.last_alpha_frame_id = frame.get('frame_id')
        self.last_alpha_frame_wall_ts = time.time()
        self._alpha_frame_ready = True

        # Redis copy is deliberately a compact latest snapshot for observability
        # and restart diagnostics, not a historical recovery source.
        try:
            summary = {
                'ts': f"{self.last_alpha_frame_ts:.3f}",
                'frame_id': str(self.last_alpha_frame_id or ''),
                'symbols': str(len(self.latest_alpha_by_symbol)),
                'updated_at': f"{self.last_alpha_frame_wall_ts:.3f}",
                'ready': '1',
            }
            by_symbol = {}
            for sym, item in self.latest_alpha_by_symbol.items():
                by_symbol[sym] = json.dumps({
                    'ts': self.last_alpha_frame_ts,
                    'alpha': float(item.get('alpha', 0.0) or 0.0),
                    'vol_z': float(item.get('vol_z', 0.0) or 0.0),
                    'roc_5m': float(item.get('roc_5m', 0.0) or 0.0),
                    'macd': float(item.get('macd', 0.0) or 0.0),
                    'is_ready': bool(item.get('is_ready', False)),
                }, ensure_ascii=False)
            pipe = self.r.pipeline()
            pipe.delete("meta:oms_latest_alpha_frame")
            pipe.hset("meta:oms_latest_alpha_frame", mapping=summary)
            pipe.expire("meta:oms_latest_alpha_frame", 180)
            pipe.delete("meta:oms_latest_alpha_by_symbol")
            if by_symbol:
                pipe.hset("meta:oms_latest_alpha_by_symbol", mapping=by_symbol)
                pipe.expire("meta:oms_latest_alpha_by_symbol", 180)
            pipe.execute()
        except Exception as e:
            if not getattr(self, '_latest_alpha_cache_err_logged', False):
                logger.warning(f"⚠️ [OMS Alpha Cache] Redis snapshot failed: {e}")
                self._latest_alpha_cache_err_logged = True

    def _strategy_alpha_ready(self, action: str, source: str = None) -> bool:
        """Startup barrier for strategy-originated orders."""
        strategy_sources = {None, '', 'signal_engine_v8', 'oms_strategy_v8'}
        if source not in strategy_sources:
            return True
        if self._alpha_frame_ready and self.last_alpha_frame_ts > 0:
            return True
        now = time.time()
        if now - float(getattr(self, '_last_alpha_barrier_warn_ts', 0.0) or 0.0) >= 10.0:
            logger.warning(
                f"⏳ [OMS Alpha Barrier] Block {action}: no fresh ALPHA_FRAME received since OMS startup."
            )
            self._last_alpha_barrier_warn_ts = now
        return False

    def _execution_quote_freshness(self, sym: str, curr_ts: float):
        """Return (is_fresh, lag_sec, wall_lag_sec, quote)."""
        q = self.latest_execution_quote_by_symbol.get(sym) or {}
        if not q:
            return False, float("inf"), float("inf"), {}

        try:
            q_ts = float(q.get('ts', 0.0) or 0.0)
        except Exception:
            q_ts = 0.0
        try:
            q_wall_ts = float(q.get('wall_ts', 0.0) or 0.0)
        except Exception:
            q_wall_ts = 0.0

        lag = abs(float(curr_ts) - q_ts) if (q_ts > 0 and curr_ts > 0) else float("inf")
        wall_lag = (time.time() - q_wall_ts) if q_wall_ts > 0 else float("inf")
        is_fresh = (
            lag <= float(OMS_MAX_QUOTE_STALE_SEC)
            and wall_lag <= float(OMS_MAX_QUOTE_WALL_STALE_SEC)
        )
        return bool(is_fresh), float(lag), float(wall_lag), q

    @classmethod
    def _allow_risk_exit_on_stale_quote(cls, reason: str) -> bool:
        """Allow time/risk exits to proceed even if the execution quote cache is stale."""
        reason_text = str(reason or "").upper()
        return any(token in reason_text for token in cls.STALE_GUARD_RISK_EXIT_TOKENS)

    def _should_block_strategy_on_stale_quote(
        self,
        sym: str,
        curr_ts: float,
        action: str,
        reason: str = "",
        frame_has_quote: bool = False,
    ) -> bool:
        """Block strategy-side BUY/SELL when quote is stale during realtime modes.

        For strategy exits, the minute ALPHA_FRAME may already carry a fresh,
        symbol-local option quote in ``opt_data`` even when the independent
        execution-quote cache is stale or briefly missing. In that case we
        should trust the frame and allow StrategyCore risk exits (especially
        EOD_CLEAR) to proceed instead of requiring a second cache to refresh.
        """
        if self.mode != 'realtime' or not OMS_GUARD_STALE_QUOTES:
            return False
        action = str(action or '').upper()
        if action == 'BUY' and not OMS_BLOCK_ENTRY_ON_STALE:
            return False
        if action == 'SELL' and not OMS_BLOCK_EXIT_ON_STALE:
            return False
        if action == 'SELL' and frame_has_quote:
            return False

        is_fresh, lag, wall_lag, _ = self._execution_quote_freshness(sym, curr_ts)
        if is_fresh:
            return False

        if action == 'SELL' and self._allow_risk_exit_on_stale_quote(reason):
            return False

        if action == 'SELL' and OMS_ALLOW_EOD_EXIT_ON_STALE:
            r = str(reason or '').upper()
            if r.startswith('X1_') or 'EOD' in r:
                return False

        now = time.time()
        key = f"{sym}:{action}"
        last = float(self._last_stale_quote_block_warn_ts.get(key, 0.0) or 0.0)
        if now - last >= 15.0:
            logger.warning(
                f"🛡️ [OMS StaleGuard] Block {action} {sym}: stale quote "
                f"(event_lag={lag:.1f}s, wall_lag={wall_lag:.1f}s, reason={reason or 'N/A'})"
            )
            self._last_stale_quote_block_warn_ts[key] = now
        return True

    def _build_strategy_ctx(self, item, opt_data, frame, ny_now, curr_ts, spy_roc, qqq_roc):
        sym = item.get('symbol')
        st = self.states[sym]
        price = float(item.get('stock_price', 0.0) or 0.0)
        final_alpha = float(item.get('alpha', 0.0) or 0.0)

        if opt_data.get('has_feed'):
            if st.position == 1:
                curr_iv = float(opt_data.get('call_iv', 0.0) or 0.0)
            elif st.position == -1:
                curr_iv = float(opt_data.get('put_iv', 0.0) or 0.0)
            else:
                c_iv = float(opt_data.get('call_iv', 0.0) or 0.0)
                p_iv = float(opt_data.get('put_iv', 0.0) or 0.0)
                curr_iv = (c_iv + p_iv) / 2.0 if c_iv > 0.01 and p_iv > 0.01 else max(c_iv, p_iv)
            if curr_iv > 0.01:
                st.last_valid_iv = curr_iv

        eval_dir = st.position if st.position != 0 else (1 if final_alpha > 0 else -1)
        ctx_bid = float(opt_data.get('call_bid' if eval_dir == 1 else 'put_bid', 0.0) or 0.0)
        ctx_ask = float(opt_data.get('call_ask' if eval_dir == 1 else 'put_ask', 0.0) or 0.0)
        market_opt_price = float(opt_data.get('call_price' if eval_dir == 1 else 'put_price', 0.0) or 0.0)

        ctx_curr_price = 0.0
        if opt_data.get('has_feed'):
            ctx_curr_price = self._get_fair_market_price(market_opt_price, ctx_bid, ctx_ask, getattr(st, 'last_opt_price', 0.0))
        elif st.position != 0:
            ctx_curr_price = max(float(st.entry_price or 0.0), 0.01)

        # Exit safety fallback:
        # In realtime_dry/live, ALPHA_FRAME may occasionally miss option feed while
        # we still have fresh 1s execution quotes. If we keep curr_price at
        # entry_price, ROI becomes near 0 and hard stop can be silently bypassed.
        # For active positions only, allow latest execution quote to backfill
        # curr_price/bid/ask for exit risk checks.
        if st.position != 0 and (not opt_data.get('has_feed') or ctx_curr_price <= 0.01):
            is_fresh, _, _, q = self._execution_quote_freshness(sym, curr_ts)
            if not is_fresh:
                q = {}
            if st.position == 1:
                q_px = float(q.get('call_price', 0.0) or 0.0)
                q_bid = float(q.get('call_bid', 0.0) or 0.0)
                q_ask = float(q.get('call_ask', 0.0) or 0.0)
            else:
                q_px = float(q.get('put_price', 0.0) or 0.0)
                q_bid = float(q.get('put_bid', 0.0) or 0.0)
                q_ask = float(q.get('put_ask', 0.0) or 0.0)
            if q_px > 0.01:
                ctx_curr_price = q_px
                market_opt_price = q_px
                if q_bid > 0.01:
                    ctx_bid = q_bid
                if q_ask > 0.01:
                    ctx_ask = q_ask

        # For active positions in realtime, prefer fresh 1s execution quote as
        # current mark to avoid minute snapshot staleness delaying ladder exits.
        if st.position != 0 and self.mode == 'realtime':
            is_fresh, _, _, q = self._execution_quote_freshness(sym, curr_ts)
            if is_fresh:
                if st.position == 1:
                    q_px = float(q.get('call_price', 0.0) or 0.0)
                    q_bid = float(q.get('call_bid', 0.0) or 0.0)
                    q_ask = float(q.get('call_ask', 0.0) or 0.0)
                else:
                    q_px = float(q.get('put_price', 0.0) or 0.0)
                    q_bid = float(q.get('put_bid', 0.0) or 0.0)
                    q_ask = float(q.get('put_ask', 0.0) or 0.0)
                if q_px > 0.01:
                    ctx_curr_price = q_px
                    market_opt_price = q_px
                    if q_bid > 0.01:
                        ctx_bid = q_bid
                    if q_ask > 0.01:
                        ctx_ask = q_ask

        st.last_price = price
        st.last_tick_price = price
        st.last_tick_opt_data = opt_data
        if price > 0 and int(getattr(st, 'last_min_ts', 0) or 0) != int(curr_ts):
            st.prices.append(float(price))
            st.alpha_history.append(float(final_alpha))
            if len(st.prices) >= 2 and list(st.prices)[-2] > 0:
                prev_price = list(st.prices)[-2]
                st.pct_history.append((float(price) - prev_price) / prev_price)
            st.last_min_ts = int(curr_ts)
        st.prev_alpha_z = float(getattr(st, 'last_alpha_z', 0.0) or 0.0)
        st.last_alpha_z = final_alpha
        st.last_vol_z = float(item.get('vol_z', 0.0) or 0.0)
        st.last_snap_roc = float(item.get('snap_roc', 0.0) or 0.0)
        st.last_macd_hist = float(item.get('macd', 0.0) or 0.0)
        st.last_macd_hist_slope = float(item.get('macd_slope', 0.0) or 0.0)
        st.warmup_complete = bool(item.get('is_ready', False))
        st.correction_mode = item.get('correction_mode', getattr(st, 'correction_mode', 'NORMAL'))

        ctx = {
            'symbol': sym,
            'time': ny_now,
            'curr_ts': curr_ts,
            'price': price,
            'alpha': final_alpha,
            'alpha_z': final_alpha,
            'cs_alpha_z': float(item.get('cs_alpha_z', final_alpha) or final_alpha),
            'vol_z': st.last_vol_z,
            'stock_roc': float(item.get('roc_5m', 0.0) or 0.0),
            'event_prob': float(item.get('event_prob', 0.0) or 0.0),
            'macd_hist': st.last_macd_hist,
            'macd_hist_slope': st.last_macd_hist_slope,
            'spy_roc': float(spy_roc or 0.0),
            'qqq_roc': float(qqq_roc or 0.0),
            'index_trend': int(frame.get('index_trend', 0) or 0),
            'position': int(getattr(st, 'position', 0) or 0),
            'cooldown_until': float(getattr(st, 'cooldown_until', 0.0) or 0.0),
            'is_ready': bool(getattr(st, 'warmup_complete', False)),
            'is_banned': curr_ts < float(getattr(self, 'global_cooldown_until', 0.0) or 0.0),
            'held_mins': self._calc_trading_minutes(st.entry_ts, curr_ts) if st.position != 0 else 0.0,
            'stock_iv': float(getattr(st, 'last_valid_iv', 0.5) or 0.5),
            'holding': {
                'entry_price': st.entry_price,
                'entry_stock': st.entry_stock,
                'entry_ts': st.entry_ts,
                'dir': st.position,
                'max_roi': st.max_roi,
                'entry_spy_roc': getattr(st, 'entry_spy_roc', 0.0),
                'entry_index_trend': getattr(st, 'entry_index_trend', 0),
            } if st.position != 0 else None,
            'curr_price': ctx_curr_price,
            'curr_stock': price,
            'bid': ctx_bid,
            'ask': ctx_ask,
            'spread_divergence': 0.0,
            'snap_roc': st.last_snap_roc,
            'global_regime_reversal_cnt': int(frame.get('global_regime_reversal_cnt', 0) or 0),
            'regime_reversal_count': int(frame.get('global_regime_reversal_cnt', 0) or 0),
            'is_volatile_regime': bool(frame.get('global_is_volatile_regime', False)),
            'regime_band': str(frame.get('global_regime_band', 'calm') or 'calm'),
            'regime_score': float(frame.get('global_regime_score', 0.0) or 0.0),
            'state': st,
        }

        if st.position != 0:
            stock_ref = float(ctx.get('curr_stock', price) or 0.0)
            valid_opt_price = ctx_curr_price > 0.01 and (stock_ref <= 0 or ctx_curr_price < stock_ref * 0.5)
            if valid_opt_price:
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

        return ctx, market_opt_price, ctx_curr_price, ctx_bid, ctx_ask

    def _clear_pending_exit_retry(self, st, reason: str = ""):
        if not getattr(st, 'pending_exit_retry_reason', ''):
            return
        logger.info(
            f"✅ [Exit Retry Clear] {getattr(st, 'symbol', '?')} | "
            f"reason={reason or 'cleared'} | last={getattr(st, 'pending_exit_retry_reason', '')}"
        )
        st.pending_exit_retry_reason = ""
        st.pending_exit_retry_count = 0
        st.pending_exit_retry_first_ts = 0.0
        st.pending_exit_retry_last_ts = 0.0

    def _build_pending_exit_retry_signal(
        self,
        sym: str,
        st,
        ctx: dict,
        opt_data: dict,
        market_opt_price: float,
        ctx_curr_price: float,
        ctx_bid: float,
        ctx_ask: float,
        curr_ts: float,
    ) -> Optional[Dict[str, Any]]:
        if int(getattr(st, 'position', 0) or 0) == 0:
            self._clear_pending_exit_retry(st, "no_position")
            return None
        if bool(getattr(st, 'is_pending', False)):
            return None

        retry_reason = str(getattr(st, 'pending_exit_retry_reason', '') or '').strip()
        if not retry_reason:
            return None

        max_frames = max(0, int(getattr(self.cfg, 'EXIT_UNFILLED_RETRY_FRAMES', 3) or 0))
        retry_count = int(getattr(st, 'pending_exit_retry_count', 0) or 0)
        if max_frames <= 0 or retry_count > max_frames:
            logger.error(
                f"🚨 [Exit Retry Exhausted] {sym} still open after unfilled exits | "
                f"retry_count={retry_count} max_frames={max_frames} reason={retry_reason}"
            )
            return None

        if ctx_curr_price <= 0.01:
            logger.warning(f"⚠️ [Exit Retry Skip] {sym} invalid option price={ctx_curr_price} reason={retry_reason}")
            return None

        retry_sig = {
            'action': 'SELL',
            'dir': int(getattr(st, 'position', 0) or 0),
            'target_side': int(getattr(st, 'position', 0) or 0),
            'reason': f"{retry_reason}|UNFILLED_RETRY_{retry_count}",
            'retry_root_reason': retry_reason,
            'price': ctx_curr_price,
            'market_price': market_opt_price,
            'original_position': int(getattr(st, 'position', 0) or 0),
            'meta': dict(ctx.get('meta', {}) or {}),
        }
        if opt_data.get('has_feed'):
            retry_sig['bid'] = ctx_bid
            retry_sig['ask'] = ctx_ask
            retry_sig['bid_size'] = opt_data.get('call_bid_size' if st.position == 1 else 'put_bid_size', 0.0)
            retry_sig['ask_size'] = opt_data.get('call_ask_size' if st.position == 1 else 'put_ask_size', 0.0)
        else:
            retry_sig['bid'] = ctx_curr_price
            retry_sig['ask'] = ctx_curr_price
            retry_sig['bid_size'] = 999.0
            retry_sig['ask_size'] = 999.0

        st.pending_exit_retry_last_ts = float(curr_ts or time.time())
        logger.warning(
            f"🔁 [Exit Retry] {sym} retry pending close | "
            f"count={retry_count}/{max_frames} reason={retry_reason} price={ctx_curr_price:.2f}"
        )
        return retry_sig

    async def _process_alpha_frame(self, frame: dict):
        curr_ts = float(frame.get('ts', time.time()) or time.time())
        reject_before = dict(getattr(self, '_entry_reject_counts', {}) or {})
        attempt_before = int(getattr(self, '_entry_attempt_count', 0) or 0)
        pass_before = int(getattr(self, '_entry_pass_count', 0) or 0)
        frame_key = self._alpha_frame_key(frame, curr_ts)
        if self._is_duplicate_alpha_frame(frame, curr_ts):
            logger.warning(f"♻️ [OMS AlphaFrame] Skip duplicated frame {frame_key}")
            return
        self._remember_alpha_frame_key(frame_key)
        self._cache_latest_alpha_frame(frame, curr_ts)

        self.last_curr_ts = curr_ts
        await self._flush_delayed_signals(curr_ts)
        ny_now = datetime.fromtimestamp(curr_ts, timezone('America/New_York'))
        items = frame.get('items') or []
        spy_rocs = frame.get('spy_roc_5min') or []
        qqq_rocs = frame.get('qqq_roc_5min') or []
        logger.info(
            f"📥 [OMS AlphaFrame] accepted {frame_key} ts={curr_ts:.0f} symbols={len(items)}"
        )

        entry_candidates = []
        for item in items:
            sym = item.get('symbol')
            if not sym or sym not in self.states:
                continue
            idx = int(item.get('batch_idx', -1) or -1)
            spy_roc = spy_rocs[idx] if 0 <= idx < len(spy_rocs) else 0.0
            qqq_roc = qqq_rocs[idx] if 0 <= idx < len(qqq_rocs) else 0.0
            opt_data = item.get('opt_data') or {'has_feed': False}
            st = self.states[sym]
            ctx, market_opt_price, ctx_curr_price, ctx_bid, ctx_ask = self._build_strategy_ctx(
                item, opt_data, frame, ny_now, curr_ts, spy_roc, qqq_roc
            )

            if st.position != 0:
                exit_sig = self.strategy.check_exit(ctx)
                self._publish_gate_trace(sym, 'exit', exit_sig, event_ts=curr_ts)
                if exit_sig and getattr(st, 'pending_exit_retry_reason', ''):
                    self._clear_pending_exit_retry(st, "strategy_exit_retriggered")
                if not exit_sig:
                    exit_sig = self._build_pending_exit_retry_signal(
                        sym,
                        st,
                        ctx,
                        opt_data,
                        market_opt_price,
                        ctx_curr_price,
                        ctx_bid,
                        ctx_ask,
                        curr_ts,
                    )
                if exit_sig:
                    frame_has_quote = bool(
                        opt_data.get('has_feed')
                        and ctx_curr_price > 0.01
                        and ctx_bid > 0.0
                        and ctx_ask > 0.0
                    )
                    if self._should_block_strategy_on_stale_quote(
                        sym,
                        curr_ts,
                        'SELL',
                        str(exit_sig.get('reason', '')),
                        frame_has_quote=frame_has_quote,
                    ):
                        continue
                    exit_sig['price'] = ctx_curr_price
                    exit_sig['market_price'] = market_opt_price
                    if opt_data.get('has_feed'):
                        exit_sig['bid'] = ctx_bid
                        exit_sig['ask'] = ctx_ask
                        exit_sig['bid_size'] = opt_data.get('call_bid_size' if st.position == 1 else 'put_bid_size', 0.0)
                        exit_sig['ask_size'] = opt_data.get('call_ask_size' if st.position == 1 else 'put_ask_size', 0.0)
                    else:
                        exit_sig['bid'] = ctx_curr_price
                        exit_sig['ask'] = ctx_curr_price
                        exit_sig['bid_size'] = 999.0
                        exit_sig['ask_size'] = 999.0
                    logger.info(f"🎯 [OMS Strategy Exit] {sym} | {exit_sig.get('reason')}")
                    await self._submit_strategy_order('SELL', sym, exit_sig, ctx['price'], curr_ts, idx, frame_id=frame.get('frame_id'))
                continue

            self._entry_attempt_count += 1
            if curr_ts < float(getattr(self, 'global_cooldown_until', 0.0) or 0.0):
                self._bump_entry_reject('global_cooldown', sym)
                continue
            if frame.get('is_zombie_market', False):
                self._bump_entry_reject('zombie_market', sym)
                continue
            if self._should_block_strategy_on_stale_quote(sym, curr_ts, 'BUY'):
                self._bump_entry_reject('stale_execution_quote', sym)
                continue

            entry_sig = self.strategy.decide_entry(ctx)
            self._publish_gate_trace(sym, 'entry', entry_sig, event_ts=curr_ts)
            if not entry_sig:
                try:
                    sub = self.strategy.get_last_reject_reason() or 'strategy_unspecified'
                except Exception:
                    sub = getattr(self.strategy, '_last_reject_reason', 'strategy_unspecified') or 'strategy_unspecified'
                self._bump_entry_reject(f'strategy:{sub}', sym)
                continue

            if not opt_data.get('has_feed'):
                self._bump_entry_reject('no_option_feed', sym)
                continue

            is_call = entry_sig['dir'] == 1
            t_price = float(opt_data.get('call_price' if is_call else 'put_price', 0.0) or 0.0)
            t_id = str(opt_data.get('call_id' if is_call else 'put_id', '') or '')
            t_k = float(opt_data.get('call_k' if is_call else 'put_k', 0.0) or 0.0)
            t_iv = float(opt_data.get('call_iv' if is_call else 'put_iv', 0.0) or 0.0)
            t_vol = float(opt_data.get('call_vol' if is_call else 'put_vol', 1.0) or 1.0)
            t_bid = float(opt_data.get('call_bid' if is_call else 'put_bid', 0.0) or 0.0)
            t_ask = float(opt_data.get('call_ask' if is_call else 'put_ask', 0.0) or 0.0)
            t_bs = float(opt_data.get('call_bid_size' if is_call else 'put_bid_size', 0.0) or 0.0)
            t_as = float(opt_data.get('call_ask_size' if is_call else 'put_ask_size', 0.0) or 0.0)
            if t_bid <= 0 or t_ask <= 0:
                t_bid = t_price
                t_ask = t_price
            fair_p = self._get_fair_market_price(t_price, t_bid, t_ask)
            liquidity_decision = evaluate_entry_liquidity(
                bid=t_bid,
                ask=t_ask,
                curr_price=fair_p,
                alpha_z=float(entry_sig.get('dir', item.get('alpha', 0.0)) or 0.0),
                spread_divergence=ctx.get('spread_divergence', 0.0),
                cfg=self.cfg,
            )
            if not liquidity_decision['ok']:
                reject_map = {
                    'bidask_invalid': 'opt_bidask_invalid',
                    'min_option_price': 'opt_fair_price_low',
                    'spread_too_wide': 'opt_spread_too_wide',
                    'spread_divergence': 'opt_spread_divergence',
                }
                self._bump_entry_reject(
                    reject_map.get(liquidity_decision['reason'], 'opt_liquidity_guard'),
                    sym,
                    extra={
                        'fair_p': float(fair_p or 0.0),
                        'bid': float(t_bid or 0.0),
                        'ask': float(t_ask or 0.0),
                        'th': float(liquidity_decision.get('spread_threshold', 0.0) or 0.0),
                    },
                )
                continue
            if not t_id:
                self._bump_entry_reject('opt_missing_contract_id', sym)
                continue

            stock_price = float(item.get('stock_price', 0.0) or 0.0)
            strike_valid = t_k > 1.0 and stock_price > 0 and abs(t_k - stock_price) / max(stock_price, 1.0) < 0.80
            if strike_valid:
                intrinsic = max(0.0, stock_price - t_k) if is_call else max(0.0, t_k - stock_price)
                if fair_p < intrinsic * 0.9:
                    self._bump_entry_reject('opt_below_intrinsic', sym)
                    continue

            entry_sig.update({
                'price': fair_p,
                'contract_id': t_id,
                'meta': {
                    'strike': t_k,
                    'iv': t_iv,
                    'contract_id': t_id,
                    'volume': t_vol,
                    'bid': t_bid,
                    'ask': t_ask,
                    'bid_size': t_bs,
                    'ask_size': t_as,
                    'spy_roc': float(spy_roc or 0.0),
                    'alpha_z': float(item.get('alpha', 0.0) or 0.0),
                    'index_trend': int(frame.get('index_trend', 0) or 0),
                    'alpha_label_ts': item.get('alpha_label_ts', 0.0),
                    'alpha_available_ts': item.get('alpha_available_ts', curr_ts),
                }
            })
            entry_dir = int(entry_sig.get('dir', 1 if float(item.get('alpha', 0.0) or 0.0) >= 0 else -1) or 0)
            trend_info = compute_entry_trend_quality(
                getattr(st, 'prices', []),
                entry_dir,
                window_mins=int(getattr(self.cfg, 'ENTRY_RANK_TREND_WINDOW_MINS', 30) or 30),
            )
            rank_info = compute_entry_priority_score(
                alpha=float(item.get('alpha', 0.0) or 0.0),
                iv=float(getattr(st, 'last_valid_iv', 0.0) or 0.0),
                roc_5m=float(item.get('roc_5m', 0.0) or 0.0),
                snap_roc=float(item.get('snap_roc', 0.0) or 0.0),
                macd_hist=float(item.get('macd', 0.0) or 0.0),
                entry_dir=entry_dir,
                cfg=self.cfg,
                pure_alpha_replay=PURE_ALPHA_REPLAY,
                **trend_info,
            )
            entry_candidates.append({
                'sym': sym,
                'sig': entry_sig,
                'price': stock_price,
                'curr_ts': curr_ts,
                'batch_idx': idx,
                'alpha_strength': rank_info['score'],
                'is_priority_candidate': bool(rank_info['is_priority_candidate']),
                'rank_detail': (
                    f"base={rank_info['base_score']:.3f} "
                    f"alpha={rank_info['alpha_mult']:.2f} "
                    f"abs5m={rank_info['abs_roc_mult']:.2f} "
                    f"roc={rank_info['stock_mult']:.2f} "
                    f"snap={rank_info['snap_mult']:.2f} "
                    f"macd={rank_info['macd_mult']:.2f} "
                    f"prio={rank_info['priority_mult']:.2f} "
                    f"trend={rank_info['trend_mult']:.2f} "
                    f"conf={int(rank_info['confirmation_count'])}"
                ),
            })

        min_symbols = 1 if (IS_BACKTEST or IS_SIMULATED) else 10
        diag_reason = ""
        diag_extra = {}
        if entry_candidates and len(items) < min_symbols:
            diag_reason = "min_symbols_gate"
            diag_extra = {
                'items': len(items),
                'candidates': len(entry_candidates),
            }
        if entry_candidates and len(items) >= min_symbols:
            entry_candidates.sort(key=lambda x: x['alpha_strength'], reverse=True)
            max_entries = 3
            active_slots_now = self._count_entry_slots()
            max_slots = int(getattr(self.cfg, 'MAX_POSITIONS', 0) or 0)
            remaining_slots = max(0, max_slots - active_slots_now)
            if remaining_slots <= 0:
                diag_reason = "batch_limit_frame"
                diag_extra = {
                    'active_slots': active_slots_now,
                    'max_slots': max_slots,
                }
                logger.info(
                    f"🚫 [OMS Batch-Limit] frame blocked | active_slots={active_slots_now} >= max={max_slots}"
                )
                self._publish_entry_diag(
                    frame=frame,
                    items_count=len(items),
                    entry_candidates_count=len(entry_candidates),
                    min_symbols=min_symbols,
                    reject_before=reject_before,
                    attempt_before=attempt_before,
                    pass_before=pass_before,
                    diag_reason=diag_reason,
                    diag_extra=diag_extra,
                )
                await self._broadcast_state_to_redis()
                return
            allowed_entries = min(max_entries, remaining_slots)

            selected_candidates = select_direction_split_entry_slots(entry_candidates, allowed_entries, self.cfg)
            if selected_candidates != entry_candidates[:allowed_entries]:
                logger.info(
                    "⭐ [OMS Entry-Slot] adjusted entry slots by direction/priority policy | "
                    f"picked={[cand['sym'] for cand in selected_candidates]}"
                )

            for cand in selected_candidates:
                active_count = self._count_entry_slots()
                if active_count >= int(getattr(self.cfg, 'MAX_POSITIONS', 0)):
                    logger.info(
                        f"🚫 [OMS Batch-Limit] {cand['sym']} blocked | "
                        f"active_count={active_count} >= {self.cfg.MAX_POSITIONS}"
                    )
                    break
                self._entry_pass_count += 1
                logger.info(
                    f"🎯 [OMS Strategy Entry] {cand['sym']} | "
                    f"{cand['sig'].get('reason')} | rank={cand['alpha_strength']:.3f} "
                    f"| {cand.get('rank_detail', '')}"
                )
                await self._submit_strategy_order(
                    'BUY', cand['sym'], cand['sig'], cand['price'], cand['curr_ts'], cand['batch_idx'],
                    frame_id=frame.get('frame_id')
                )

        self._publish_entry_diag(
            frame=frame,
            items_count=len(items),
            entry_candidates_count=len(entry_candidates),
            min_symbols=min_symbols,
            reject_before=reject_before,
            attempt_before=attempt_before,
            pass_before=pass_before,
            diag_reason=diag_reason,
            diag_extra=diag_extra,
        )
        await self._broadcast_state_to_redis()

    async def _evaluate_shadow_signals(self, payload: dict):
        """Legacy no-op: 1s/fused payloads are execution quotes, not strategy signals."""
        logger.warning("🚫 [OMS Shadow] Ignored legacy shadow signal path; StrategyCore runs only on ALPHA_FRAME.")
        return

    async def _broadcast_state_to_redis(self):
        """Publish a read-only OMS projection for dashboard/diagnostics only.

        Trading state authority stays in OMS memory and PG state snapshots.
        Signal Engine must not consume this hash to drive strategy decisions.
        """
        if getattr(self, 'use_shared_mem', False):
            return # 🚀 共享内存模式下，对象修改瞬间双端可见，禁止网络广播！
        if not self._refresh_oms_writer_lock():
            return
        import json
        active_states = {}
        
        for sym, st in self.states.items():
            # 🚀 [核心修复] 如果 sym 是 bytes (来自 Redis)，必须转换为 str 否则 json.dumps 会触发 TypeError
            if isinstance(sym, bytes):
                sym = sym.decode('utf-8')
                
            pos = int(getattr(st, 'position', 0) or 0)
            qty = float(getattr(st, 'qty', 0) or 0)
            open_fill_confirmed = bool(getattr(st, 'open_fill_confirmed', False))

            # `oms:live_positions` 必须只表达真实已成交持仓。
            # pending BUY 仍会占用 entry slot，但应通过 pending_orders 展示，不能污染持仓表。
            if pos != 0 and qty > 0 and open_fill_confirmed:
                opt_type = str(getattr(st, 'opt_type', '') or '').lower()
                raw_tag = str(getattr(st, 'option_tag', '') or '').strip().upper()
                if raw_tag not in {'CALL_ATM', 'CALL_OTM', 'PUT_ATM', 'PUT_OTM'}:
                    raw_tag = 'CALL_ATM' if (pos == 1 or opt_type == 'call') else ('PUT_ATM' if (pos == -1 or opt_type == 'put') else '')
                tag = raw_tag
                active_states[sym] = json.dumps({
                    'projection_only': True,
                    'pos': st.position,
                    'position': st.position,
                    'qty': qty,
                    'price': st.entry_price,
                    'entry_price': st.entry_price,
                    'stock': st.entry_stock,
                    'entry_stock': st.entry_stock,
                    'ts': st.entry_ts,
                    'entry_ts': st.entry_ts,
                    'contract_id': getattr(st, 'contract_id', '') or '',
                    'opt_type': opt_type,
                    'tag': tag,
                    'last_opt_price': getattr(st, 'last_opt_price', 0.0),
                    'max_roi': getattr(st, 'max_roi', 0.0),
                    'is_pending': st.is_pending,
                    'entry_slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'pending_action': getattr(st, 'pending_action', None) or getattr(st, 'pending_side', '') or '',
                    'open_fill_confirmed': open_fill_confirmed,
                    # 仅供 Dashboard/诊断展示，不能作为 SE 策略状态输入。
                    'entry_spy_roc': getattr(st, 'entry_spy_roc', 0.0),
                    'entry_index_trend': getattr(st, 'entry_index_trend', 0),
                    'entry_alpha_z': getattr(st, 'entry_alpha_z', 0.0),
                    'entry_iv': getattr(st, 'entry_iv', getattr(st, 'last_valid_iv', 0.0))
                })
        
        # 👑 同时发布资金池只读投影
        ledger_run_mode = str(RUN_MODE or '').upper()
        ledger_engine_mode = str(getattr(self, 'mode', '') or '').upper()
        active_states['____SYSTEM_CASH____'] = json.dumps({
            'projection_only': True,
            'cash': self.mock_cash,
            'ts': time.time(),
            'mode': ledger_run_mode,
            'engine_mode': ledger_engine_mode,
        })
                
        pipe = self.r.pipeline()
        pipe.delete("oms:live_positions")  # 先清空全量
        if active_states:
            pipe.hset("oms:live_positions", mapping=active_states)
        # [Ledger Projection] Redis 账本快照（Dashboard Remaining Cash 首选口径）
        # 设计目标：
        # - OMS 是唯一账本事实来源；
        # - Dashboard 不再基于 trade log / PG 做二次推导；
        # - 用 seq 提供单调版本，便于排查“金额跳变/回退”。
        self._ledger_seq = int(getattr(self, '_ledger_seq', 0) or 0) + 1
        pipe.hset("meta:oms_ledger", mapping={
            'cash': f"{float(self.mock_cash):.6f}",
            'realized_pnl': f"{float(getattr(self, 'realized_pnl', 0.0) or 0.0):.6f}",
            'total_commission': f"{float(getattr(self, 'total_commission', 0.0) or 0.0):.6f}",
            'trade_count': str(int(getattr(self, 'trade_count', 0) or 0)),
            'win_count': str(int(getattr(self, 'win_count', 0) or 0)),
            'loss_count': str(int(getattr(self, 'loss_count', 0) or 0)),
            'seq': str(self._ledger_seq),
            'updated_at': f"{time.time():.3f}",
            'mode': ledger_run_mode,
            'engine_mode': ledger_engine_mode,
            'projection_only': '1',
            'writer_pid': str(os.getpid()),
            'boot_id': str(getattr(self, '_boot_id', '')),
            'writer_lock': str(getattr(self, '_oms_writer_lock_key', '')),
            'state_namespace': OMS_STATE_NAMESPACE,
            'open_positions': str(self._count_entry_slots()),
        })
        pipe.expire("meta:oms_ledger", 24 * 3600)
        pipe.execute()

        # [Observability Projection] per-symbol cooldown_until.
        # 只给 Dashboard/诊断展示，策略冷却由 OMS 本地 StrategyCore 状态裁决。
        # 禁止 SE 或其他进程把该 Redis hash 当交易状态同步源。
        try:
            now_ts = time.time()
            cooldown_mapping = {}
            for sym, st in self.states.items():
                if isinstance(sym, bytes):
                    sym = sym.decode('utf-8')
                cd = float(getattr(st, 'cooldown_until', 0.0) or 0.0)
                if cd > now_ts:  # 只广播仍在生效的冷却窗口
                    cooldown_mapping[sym] = f"{cd:.3f}"
            cd_pipe = self.r.pipeline()
            cd_pipe.delete("meta:symbol_cooldowns")
            if cooldown_mapping:
                cd_pipe.hset("meta:symbol_cooldowns", mapping=cooldown_mapping)
                cd_pipe.expire("meta:symbol_cooldowns", 6 * 3600)
            cd_pipe.execute()
        except Exception as e:
            logger.warning(f"⚠️ [Cooldown Broadcast] failed: {e}")

        # [Layer A Heartbeat] 节流 30s 打一条策略状态汇总, 让交易员不登录 Dashboard 也能
        # 一眼看到 "资金 / 持仓 / 连败 / 冷却" 的状态条。避免日志刷屏, 每 30s 最多一条。
        try:
            now_hb = time.time()
            last_hb = float(getattr(self, '_last_state_hb_ts', 0.0) or 0.0)
            if now_hb - last_hb >= 30.0:
                self._last_state_hb_ts = now_hb
                streak = int(getattr(self, 'consecutive_stop_losses', 0) or 0)
                cb_th = int(getattr(self, 'CIRCUIT_BREAKER_THRESHOLD', 3))
                cb_until = float(getattr(self, 'global_cooldown_until', 0.0) or 0.0)
                if cb_until > now_hb:
                    cb_tag = f"🔥 CB ON {(cb_until - now_hb) / 60:.0f}m"
                else:
                    cb_tag = "CB OFF"
                pos_cnt = self._count_entry_slots()
                max_pos = int(getattr(self.cfg, 'MAX_POSITIONS', 0))
                cd_list = []
                for sym, st in self.states.items():
                    if isinstance(sym, bytes):
                        sym = sym.decode('utf-8')
                    cd = float(getattr(st, 'cooldown_until', 0.0) or 0.0)
                    if cd > now_hb:
                        cd_list.append(f"{sym}:{(cd - now_hb) / 60:.0f}m")
                cd_tag = f"cd=[{', '.join(cd_list)}]" if cd_list else "cd=[]"
                logger.info(
                    f"🫀 [OMS-State] cash=${self.mock_cash:,.0f} | pos={pos_cnt}/{max_pos} | "
                    f"streak={streak}/{cb_th} | {cb_tag} | {cd_tag}"
                )
        except Exception as _hb_e:
            if not getattr(self, '_hb_log_err_logged', False):
                logger.debug(f"[OMS-State Heartbeat] emit failed: {_hb_e}")
                self._hb_log_err_logged = True

    async def _handle_trade_signal(self, payload: dict, allow_delay_queue: bool = True):
        source = payload.get('source')
        curr_ts = payload.get('ts')
        curr_ts_float = float(curr_ts) if curr_ts is not None else None

        if payload.get('action') == 'ALPHA_FRAME' or source == 'alpha_engine_v8':
            await self._process_alpha_frame(payload)
            return
        
        # 🚀 [Fused Replay Protocol] 下一代高速对齐协议支持
        if source == 'fused_replay_v8':
            if curr_ts_float is not None:
                # Fused/1s replay packets are execution quote frames only.
                # They must not call StrategyCore or shadow-generate BUY/SELL.
                self._cache_execution_market_packet(payload)

                # 🚨 [CRITICAL] 释放屏障：无论是否有交易，必须通知发球机本帧已处理完毕
                self.r.set(f"sync:orch_done:{int(curr_ts_float)}", "1")
                self.r.expire(f"sync:orch_done:{int(curr_ts_float)}", 60)
            return

        action = payload.get('action')

        if curr_ts_float is not None:
            self.last_curr_ts = curr_ts_float
            # Keep delayed strategy release on minute/event clock, not on
            # per-second SYNC heartbeat.
            if action != 'SYNC':
                await self._flush_delayed_signals(curr_ts_float)

        if action in ('BUY', 'SELL') and not self._strategy_alpha_ready(action, source):
            return

        if allow_delay_queue and action in ('BUY', 'SELL') and self._should_delay_signal(action, source):
            await self._queue_delayed_signal(payload)
            return

        if action in ('BUY', 'SELL'):
            if self._is_duplicate_trade_signal(payload):
                logger.warning(
                    f"♻️ [OMS Signal Dedupe] drop duplicated {action} "
                    f"{payload.get('symbol')} key={self._trade_signal_key(payload)}"
                )
                return
            self._remember_trade_signal(payload)

        self._refresh_signal_from_execution_quote(payload)
        
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
                # 冰山开仓会在后台继续成交；此时必须保持 pending，
                # 否则同一 AlphaFrame 的后续候选会忽略这笔锁定资金/名额。
                if not bool(getattr(st, '_async_entry_order_active', False)):
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
        采用非阻塞异步 Redis 流消费，兼容实盘与回测信号流。
        """
        from config import REDIS_CFG, STREAM_ORCH_SIGNAL, STREAM_FUSED_MARKET, SYNC_EXECUTION, TRADING_ENABLED
        import os
        import traceback
        import asyncio
        import redis
        from utils import serialization_utils as ser
        
        logger.info(f"🔥 Execution Engine (OMS) Started (DB: {self.r.connection_pool.connection_kwargs.get('db')})")

        if not self._acquire_oms_writer_lock():
            return
        
        # 1. 确保消费组存在
        self._ensure_consumer_group()

        # 1.1 [Startup Init] 双引擎 OMS 必须自行完成:
        #       (A) 连接 IBKR (REALTIME/REALTIME_DRY, 非 SIMULATED);
        #       (B) 从 PG 恢复当日 mock_cash + 同日持仓 (_load_state);
        #       (C) 从 IBKR 读真实账户净资产只做观测日志, 不覆盖 OMS cash ledger。
        #     之前这段逻辑只存在于 single-process system_orchestrator_v8, 双引擎架构下
        #     OMS 从未 connect / get_balance / _load_state → mock_cash 永远 50000,
        #     REALTIME 甚至下不出单。
        try:
            if not IS_SIMULATED:
                restored_open_positions = 0
                try:
                    sm = getattr(self, 'state_manager', None)
                    if sm is not None:
                        sm._load_state()
                        self._apply_live_trading_capital_limit("post_load_state")
                        restored_open_positions = sum(
                            1 for _st in self.states.values()
                            if int(getattr(_st, 'position', 0) or 0) != 0
                        )
                        logger.info(
                            f"♻️ [OMS-Init] _load_state 完成 | mock_cash=${self.mock_cash:,.2f} "
                            f"| restored_open_positions={restored_open_positions}"
                        )
                except Exception as e:
                    logger.warning(f"⚠️ [OMS-Init] _load_state 失败, 继续使用默认资金: {e}")

                if self.mode == 'realtime' and self.ibkr is not None and hasattr(self.ibkr, 'connect'):
                    try:
                        await self.ibkr.connect()
                        logger.info(
                            "🔌 [OMS-Init] IBKR connected | RUN_MODE=%s | TRADING_ENABLED=%s",
                            RUN_MODE, TRADING_ENABLED
                        )
                    except Exception as e:
                        logger.error(f"❌ [OMS-Init] IBKR connect 失败: {e}")

                    if hasattr(self.ibkr, 'get_account_balance'):
                        try:
                            real_bal = await self.ibkr.get_account_balance()
                        except Exception as e:
                            real_bal = None
                            logger.warning(f"⚠️ [OMS-Init] get_account_balance 异常: {e}")
                        logger.info(
                            f"💰 [OMS-Init] Broker balance observed={real_bal} | "
                            f"OMS cash ledger kept=${self.mock_cash:,.2f} | "
                            f"restored_open_positions={restored_open_positions} | "
                            f"TRADING_ENABLED={TRADING_ENABLED} | IS_REALTIME_DRY={IS_REALTIME_DRY}"
                        )
                else:
                    logger.info(
                        f"🧪 [OMS-Init] mode={self.mode}/ibkr={type(self.ibkr).__name__ if self.ibkr else None} "
                        f"→ 跳过 IBKR connect / balance sync"
                    )
        except Exception as e:
            logger.error(f"❌ [OMS-Init] 启动期初始化异常: {e}")

        # [Startup Projection]
        # OMS 启动后发布一次只读账本投影，供 Dashboard/诊断看到当前状态。
        # SE 不读取该投影，策略交易状态只由 OMS 内存/PG 快照管理。
        try:
            await self._broadcast_state_to_redis()
            logger.info("📡 [OMS-Init] Startup state broadcast completed.")
        except Exception as e:
            logger.warning(f"⚠️ [OMS-Init] Startup broadcast failed: {e}")

        # [Config Snapshot] 启动时强制广播一次 strategy_config0,
        # Dashboard "🧬 策略门禁" tab 读 meta:strategy_config 渲染参数表.
        try:
            self._publish_strategy_config_snapshot(force=True)
        except Exception as _cfg_e:
            logger.warning(f"⚠️ [Config Snapshot] startup publish failed: {_cfg_e}")

        # Strategy startup barrier: restored OMS positions/cash are valid, but
        # strategy trading remains closed until a new ALPHA_FRAME is accepted.
        try:
            self._alpha_frame_ready = False
            self.r.delete("meta:oms_latest_alpha_frame")
            self.r.hset("meta:oms_latest_alpha_frame", mapping={
                'ready': '0',
                'ts': '0',
                'frame_id': '',
                'symbols': '0',
                'updated_at': f"{time.time():.3f}",
                'status': 'waiting_for_alpha_frame',
            })
            self.r.expire("meta:oms_latest_alpha_frame", 180)
            logger.warning("⏳ [OMS-Init] Strategy barrier ON: waiting for first ALPHA_FRAME.")
        except Exception as e:
            logger.warning(f"⚠️ [OMS-Init] Alpha barrier publish failed: {e}")

        # 2. 启动后台监控任务 (同步回测时不启动后台扫描，保持确定性)
        if not SYNC_EXECUTION:
            asyncio.create_task(self._pnl_monitor_loop())
            if TRADING_ENABLED:
                asyncio.create_task(self.reconciler.run_reconciliation_loop())
        
        target_group = getattr(self, '_oms_group', REDIS_CFG.get('oms_group') or REDIS_CFG.get('orch_group') or GROUP_OMS)
        consumer_name = f"oms_consumer_{os.getpid()}"
        
        # 同时监听信号流(指令/ALPHA_FRAME)和行情流(只做执行报价缓存兜底)
        streams = {STREAM_ORCH_SIGNAL: '>', STREAM_FUSED_MARKET: '>'}
        last_stats_log = time.time()
        stats = {'msgs': 0, 'signal': 0, 'fused': 0, 'acks': 0}
        total_msgs_since_boot = 0
        boot_wall_ts = time.time()
        last_health_check_ts = 0.0
        health_guard_warmup_sec = 30.0
        health_check_interval_sec = 10.0

        def _get_group_lag(stream_name: str, group_name: str) -> int:
            """读取指定 stream/group 的 lag；异常时返回 0，避免影响主循环。"""
            try:
                groups = self.r.xinfo_groups(stream_name) or []
                for g in groups:
                    raw_name = g.get(b'name') if isinstance(g, dict) else None
                    if raw_name is None and isinstance(g, dict):
                        raw_name = g.get('name')
                    name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else raw_name
                    if name == group_name:
                        raw_lag = g.get(b'lag') if isinstance(g, dict) else None
                        if raw_lag is None and isinstance(g, dict):
                            raw_lag = g.get('lag')
                        return int(raw_lag or 0)
            except Exception:
                return 0
            return 0
        
        while True:
            try:
                if not self._refresh_oms_writer_lock():
                    return
                now = time.time()
                if now - last_health_check_ts >= health_check_interval_sec:
                    last_health_check_ts = now
                    if now - boot_wall_ts >= health_guard_warmup_sec and total_msgs_since_boot == 0:
                        sig_lag = _get_group_lag(STREAM_ORCH_SIGNAL, target_group)
                        fused_lag = _get_group_lag(STREAM_FUSED_MARKET, target_group)
                        total_lag = sig_lag + fused_lag
                        if total_lag > 0:
                            logger.critical(
                                f"🚨 [OMS-Health-Guard] No consumption after {int(now - boot_wall_ts)}s "
                                f"but lag>0 (signal_lag={sig_lag}, fused_lag={fused_lag}, group={target_group}). "
                                "Likely stream/group mismatch or stuck consumer; exiting OMS loop."
                            )
                            return

                # 3. 优先消费未确认的历史消息 (Pending)
                pending = self.r.xreadgroup(
                    groupname=target_group,
                    consumername=consumer_name,
                    streams={STREAM_ORCH_SIGNAL: '0', STREAM_FUSED_MARKET: '0'},
                    count=100
                )

                # redis-py 在 pending 为空时也可能返回 [(stream, []), ...]；
                # 仅凭 `if pending` 会误判为有消息，导致永远不进入 `>` 新消息分支。
                pending_with_msgs = [(s, msgs) for s, msgs in (pending or []) if msgs]
                if pending_with_msgs:
                    messages = pending_with_msgs
                    pending_msg_count = sum(len(msgs) for _, msgs in pending_with_msgs)
                    logger.debug(f"🔍 [OMS] Found {pending_msg_count} pending messages in streams.")
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
                    total_msgs_since_boot += len(msgs)
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
                                    self._process_fused_market_for_execution(payload)
                                    stats['fused'] += 1
                            
                            try:
                                self.r.xack(stream_str, target_group, msg_id)
                                stats['acks'] += 1
                            except Exception as ack_err:
                                logger.error(f"ACK Error: {ack_err}")
                                
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
