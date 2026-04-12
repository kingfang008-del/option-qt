#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: strategy_core_v1.py
描述: [V15 基准版 - 稳定复刻 + 统一配置]
职责: 
    - 稳定版核心：不包含实验性的 Plan A 或 趋势门控
    - 确保 15.1% 收益基准的逻辑原子性
"""

import logging
logger = logging.getLogger("StrategyCoreV1")

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

from config import TARGET_SYMBOLS
try:
    from strategy_config import StrategyConfig
except ImportError:
    from production.baseline.strategy_config import StrategyConfig

class StrategyCoreV1:
    def __init__(self, config: StrategyConfig = None):
        self.cfg = config if config else StrategyConfig()

    def decide_entry(self, ctx: dict) -> dict:
        # 🧪 [Parity Fix] 在严格对齐模式下，禁用 cs_alpha_z (截面缩放)，只看原始 Alpha
        if getattr(self.cfg, 'PARITY_STRICT_MODE', False):
            alpha_z = ctx.get('alpha_z', 0.0)
        else:
            alpha_z = ctx.get('cs_alpha_z', ctx.get('alpha_z', 0.0))
        
        sig = None
        # Select Channel
        if abs(alpha_z) >= self.cfg.ALPHA_ENTRY_STRICT:
            sig = self._check_channel_a_momentum(ctx)
        
        if not sig: return None
        if self.cfg.ENTRY_LIQUIDITY_GUARD_ENABLED:
            if not self._check_entry_liquidity_guard(ctx): return None
        return sig

    def _check_entry_pre_conditions(self, ctx: dict) -> bool:
        if not ctx.get('is_ready', False): return False
        if ctx.get('is_banned', False): return False
        if ctx['position'] != 0: return False
        if ctx['curr_ts'] < ctx.get('cooldown_until', 0): return False
        
        # 1. 时间检查 (适配 1s/1m)
        curr_time_str = ctx['time'].strftime('%H:%M:%S')
        if curr_time_str < self.cfg.START_TIME:
            return False
            
        # 2. 状态检查
        st = ctx.get('state')
        if not st: return False
        
        # 3. 大盘卫士 (Index Guard)
        if self.cfg.INDEX_GUARD_ENABLED:
            idx_trend = ctx.get('index_trend', 0)
            if idx_trend < 0: return False # 🚀 [Fix] 仅在 Bearish (-1) 时拦截，Neutral (0) 允许尝试
        
        # 4. MACD Hist 确认 (趋势方向)
        if self.cfg.MACD_HIST_CONFIRM_ENABLED:
            hist = ctx.get('macd_hist', 0.0)
            if abs(hist) < self.cfg.MACD_HIST_THRESHOLD: return False
            
        # 5. 市场状态卫士 (Regime Guard - Choppiness Filter)
        if self.cfg.REGIME_GUARD_ENABLED:
            if not self._check_regime_guard(ctx): return False

        return True

    def _check_regime_guard(self, ctx: dict) -> bool:
        """
        [NEW] 核心对齐: 拦截高洗盘频率行情
        通过反转次数统计来识别 Alpha 失效的市场环境
        """
        reversal_count = ctx.get('regime_reversal_count', 0)
        if reversal_count > self.cfg.REGIME_REVERSAL_THRESHOLD:
            logger.info(f"🚫 [REGIME_GUARD] Blocked entry. Current Reversals in window: {reversal_count} (Limit: {self.cfg.REGIME_REVERSAL_THRESHOLD})")
            return False
        return True

    def _check_entry_liquidity_guard(self, ctx: dict) -> bool:
        bid, ask, curr_p = ctx.get('bid', 0.0), ctx.get('ask', 0.0), ctx.get('curr_price', 0.0)
        if bid <= 0.01 or ask <= 0.01 or curr_p <= 0.01: return False
        spread_pct = (ask - bid) / curr_p
        if spread_pct > self.cfg.MAX_SPREAD_PCT_ENTRY: return False
        if ctx.get('spread_divergence', 0.0) > self.cfg.MAX_SPREAD_DIVERGENCE: return False
        return True

    def _check_channel_b_slow_bull(self, ctx: dict) -> dict:
        vol_z, alpha_val = ctx['vol_z'], ctx['alpha_z']
        action = 1 if alpha_val > 0 else -1
        if action != 1: return None
        cond_vol = vol_z <= self.cfg.SLOW_BULL_MAX_VOL_Z
        cond_alpha = alpha_val >= self.cfg.SLOW_BULL_ALPHA_THRESHOLD
        cond_macd = ctx.get('macd_hist', 0.0) >= self.cfg.SLOW_BULL_MACD_THRESHOLD
        cond_index = (ctx.get('spy_roc', 0.0) > self.cfg.SLOW_BULL_MIN_INDEX_ROC) or (ctx.get('qqq_roc', 0.0) > self.cfg.SLOW_BULL_MIN_INDEX_ROC)
        if not (cond_vol and cond_alpha and cond_macd and cond_index): return None
        if ctx['stock_roc'] <= 0.0005: return None
        if ctx.get('snap_roc', 0.0) < self.cfg.MIN_LAST_SNAP_ROC: return None
        return {'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'legacy_tag': 'opt_8', 'score': abs(alpha_val), 'reason': f"CH_B_SLOW|A:{alpha_val:.2f}|V:{vol_z:.1f}"}

    def _check_channel_a_momentum(self, ctx: dict) -> dict:
        alpha = ctx.get('alpha_z', 0.0) # 🚀 [Fix] 对齐 SE 键名 alpha_z
        
        # 1. 动态 Alpha 阈值评估
        alpha_val = alpha
        prob = ctx.get('event_prob', 0.0)
        vol_z = ctx.get('vol_z', 0.0)
        
        # 🧪 [Parity Fix] 如果开启了严格对齐模式，强制使用 Plan A 的门槛逻辑
        if getattr(self.cfg, 'PARITY_STRICT_MODE', False):
            # Plan A 逻辑: 1. 必须 prob >= 0.7  2. 必须 abs(alpha) >= ALPHA_ENTRY_STRICT
            if prob < self.cfg.EVENT_PROB_THRESHOLD: return None
            final_threshold = self.cfg.ALPHA_ENTRY_STRICT
            action = 1 if alpha > 0 else -1
            is_event_hot = True # 默认置为 True 以兼容后缀
        else:
            # S4 高保真模式: 允许 prob=0 的冷启动，并配备动态阈值
            is_event_hot = prob > self.cfg.EVENT_PROB_THRESHOLD or prob == 0
            if is_event_hot:
                if vol_z > self.cfg.VOL_MAX_Z: return None
                final_threshold = 0.5 
                action = 1 if alpha > 0 else -1
            else:
                if not (self.cfg.VOL_MIN_Z < vol_z < self.cfg.VOL_MAX_Z): return None
                base_threshold = self.cfg.ALPHA_ENTRY_THRESHOLD
                dynamic_threshold = self._calculate_dynamic_alpha_threshold(ctx.get('symbol', 'UNK'), vol_z)
                final_threshold = max(base_threshold, dynamic_threshold)
                action = 1 if alpha > 0 else -1

        if abs(alpha_val) < final_threshold: return None
        if self.cfg.ENTRY_MOMENTUM_GUARD_ENABLED:
            if not self._check_stock_momentum_guard(ctx, action): return None
        if self.cfg.INDEX_GUARD_ENABLED:
            if not self._check_index_guard(ctx, action): return None
        if self.cfg.MACD_HIST_CONFIRM_ENABLED:
            if not self._check_macd_confirm(ctx, action): return None
        
        prefix = "[EVENT_HOT]" if is_event_hot else "CH_A_MOM"
        reason = f"{prefix}|A:{alpha_val:.2f}|V:{vol_z:.1f}|EV:{prob:.2f}"
        return {'action': 'BUY', 'dir': action, 'tag': 'CALL_ATM' if action == 1 else 'PUT_ATM', 'legacy_tag': 'opt_8' if action == 1 else 'opt_0', 'score': abs(alpha_val), 'reason': reason}

    def _calculate_dynamic_alpha_threshold(self, symbol: str, vol_z: float) -> float:
        th = self.cfg.ALPHA_ENTRY_THRESHOLD
        if vol_z > 2.0: th += (vol_z - 2.0) * 0.5
        return min(th, 3.0)

    def _check_stock_momentum_guard(self, ctx: dict, action: int) -> bool:
        if action == 1 and ctx['stock_roc'] < -self.cfg.STOCK_MOMENTUM_TOLERANCE: return False
        if action == -1 and ctx['stock_roc'] > self.cfg.STOCK_MOMENTUM_TOLERANCE: return False
        snap_roc = ctx.get('snap_roc', 0.0)
        if abs(snap_roc) > self.cfg.MAX_SNAP_ROC_LIMIT: return False
        if action == 1 and snap_roc < self.cfg.MIN_LAST_SNAP_ROC: return False
        if action == -1 and snap_roc > -self.cfg.MIN_LAST_SNAP_ROC: return False
        return True

    def _check_index_guard(self, ctx: dict, action: int) -> bool:
        if not self.cfg.INDEX_GUARD_ENABLED: return True
        spy_roc, qqq_roc = ctx.get('spy_roc', 0.0), ctx.get('qqq_roc', 0.0)
        if abs(spy_roc) < 1e-9 and abs(qqq_roc) < 1e-9: return True
        if action == -1 and self.cfg.INDEX_GUARD_SHORT_BLOCK_ENABLED:
            if ctx.get('index_trend', 0) == 1 or spy_roc > 0.0001 or qqq_roc > 0.0001: return False
        if action == 1 and (spy_roc < self.cfg.INDEX_ROC_THRESHOLD or qqq_roc < self.cfg.INDEX_ROC_THRESHOLD): return False
        return True

    def _check_macd_confirm(self, ctx: dict, action: int) -> bool:
        if not self.cfg.MACD_HIST_CONFIRM_ENABLED: return True
        macd_hist = ctx.get('macd_hist', 0.0)
        th = self.cfg.MACD_HIST_THRESHOLD
        if action == 1 and macd_hist <= th: return False
        if action == -1 and macd_hist >= -th: return False
        return True

    def check_exit(self, ctx: dict) -> dict:
        pos = ctx.get('holding')
        if not pos: return None
        sig = self._check_exit_pre_conditions(ctx, pos)
        if sig: return sig
        if self.cfg.EXIT_INDEX_REVERSAL_ENABLED:
            sig = self._check_trend_reversal_guard(ctx, pos)
            if sig: return sig
        held_mins = (ctx['curr_ts'] - pos['entry_ts']) / 60.0
        curr_price = ctx['curr_price']
        roi = 0.0 if curr_price <= 0.001 else (curr_price - pos['entry_price']) / pos['entry_price']
        max_roi = pos.get('max_roi', 0.0)
        stock_roi = (ctx['curr_stock'] - pos['entry_stock']) / pos['entry_stock'] if pos['entry_stock'] > 0 else 0.0
        sym = ctx.get('symbol', 'UNK')

        # [像素级诊断] 针对重点品种打印内部状态
        if sym in ['XOM', 'NFLX']:
            print(f"🔬 [Pixel-Trace] {sym} | mins={held_mins:.1f} | roi={roi:.2%}| max_roi={pos.get('max_roi', 0):.2%} | thresh={self.cfg.NO_MOMENTUM_MIN_MAX_ROI:.2%}")

        # --- plan_a 独有的平仓规则 (开关控制) ---
        if self.cfg.EXIT_EARLY_STOP_ENABLED:
            if held_mins <= self.cfg.EARLY_STOP_MINS and roi < self.cfg.EARLY_STOP_ROI:
                if sym in ['XOM', 'NFLX']: print(f"  🚨 {sym} TRIGGERED EARLY STOP")
                return {'action': 'SELL', 'reason': 'EARLY'}
        if self.cfg.EXIT_NO_MOMENTUM_ENABLED:
            if held_mins > self.cfg.NO_MOMENTUM_MINS and max_roi < self.cfg.NO_MOMENTUM_MIN_MAX_ROI:
                if sym in ['XOM', 'NFLX']: print(f"  🚨 {sym} TRIGGERED NO_MOMENTUM (max_roi={max_roi:.4f})")
                return {'action': 'SELL', 'reason': 'NO_MOM'}

        # --- s4 独有的平仓规则 (开关控制) ---
        if self.cfg.EXIT_STOCK_HARD_STOP_ENABLED:
            sig = self._check_stock_hard_stop(ctx, pos, held_mins, stock_roi)
            if sig: return sig
        sig = self._check_time_and_inactivity_stops(ctx, pos, held_mins, roi)
        if sig: return sig
        if self.cfg.EXIT_LIQUIDITY_GUARD_ENABLED:
            sig = self._check_exit_liquidity_guard(ctx, curr_price)
            if sig: return sig
        if self.cfg.EXIT_COND_STOP_ENABLED:
            sig = self._check_stop_loss_guards(ctx, pos, roi, stock_roi)
            if sig: return sig
        # [🔥 PARITY] Plan A 的 LADD 逻辑实际上就是 S4 的 evaluate_profit_guards
        # 之前错误地禁用了它。
        sig = self._evaluate_profit_guards(ctx, pos, held_mins, roi)
        if sig: return sig
        if self.cfg.EXIT_MACD_FADE_ENABLED:
            sig = self._check_macd_fade(ctx, pos, held_mins, roi)
            if sig: return sig
        if self.cfg.EXIT_SIGNAL_FLIP_ENABLED:
            sig = self._check_signal_flip(ctx, pos, held_mins)
            if sig: return sig
        return None

    def _check_exit_pre_conditions(self, ctx: dict, pos: dict) -> dict:
        curr_time_str = ctx['time'].strftime('%H:%M:%S')
        if curr_time_str >= self.cfg.CLOSE_TIME: return {'action': 'SELL', 'reason': 'EOD_CLEAR'}
        if self.cfg.EXIT_COUNTER_TREND_ENABLED:
            if (pos['dir'] == 1 and ctx.get('index_trend', 0) == -1) or (pos['dir'] == -1 and ctx.get('index_trend', 0) == 1):
                if (ctx['curr_ts'] - pos['entry_ts']) / 60.0 >= self.cfg.COUNTER_TREND_MAX_MINS: return {'action': 'SELL', 'reason': 'CT_TIMEOUT'}
        return None

    def _check_trend_reversal_guard(self, ctx: dict, pos: dict) -> dict:
        if not self.cfg.INDEX_REVERSAL_EXIT_ENABLED: return None
        idx_trend, entry_idx_trend = ctx.get('index_trend', 0), pos.get('entry_index_trend', 0)
        if entry_idx_trend >= 0 and pos['dir'] == 1 and idx_trend == -1: return {'action': 'SELL', 'reason': "IDX_REVERSAL_FALL"}
        if entry_idx_trend <= 0 and pos['dir'] == -1 and idx_trend == 1: return {'action': 'SELL', 'reason': "IDX_REVERSAL_RISE"}
        return None

    def _check_stock_hard_stop(self, ctx: dict, pos: dict, held_mins: float, stock_roi: float) -> dict:
        if pos['entry_stock'] <= 0 or ctx['curr_stock'] <= 0: return None
        is_event = ctx.get('event_prob', 0.0) > self.cfg.EVENT_PROB_THRESHOLD
        if is_event: base_th = self.cfg.STOCK_HARD_STOP_EVENT
        else: base_th = self.cfg.STOCK_HARD_STOP_LOOSE if abs(ctx.get('alpha_z', 0.0)) >= self.cfg.HIGH_CONFIDENCE_THRESHOLD else self.cfg.STOCK_HARD_STOP_TIGHT
        is_in_buffer = held_mins < 3.0
        current_th = max(base_th * 3.0, 0.005) if is_in_buffer else base_th
        if (pos['dir'] == 1 and stock_roi < -current_th) or (pos['dir'] == -1 and stock_roi > current_th):
            return {'action': 'SELL', 'reason': f"STOCK_STOP{'(EVENT)' if is_event else ''}:{stock_roi:.2%}"}
        return None

    def _check_time_and_inactivity_stops(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        if self.cfg.EXIT_ZOMBIE_STOP_ENABLED:
            if held_mins >= self.cfg.ZOMBIE_EXIT_MINS and abs(roi) < 0.02: return {'action': 'SELL', 'reason': 'ZOMBIE_STOP'}
        if self.cfg.EXIT_SMALL_GAIN_ENABLED:
            if held_mins >= self.cfg.SMALL_GAIN_MINS and pos['max_roi'] >= self.cfg.SMALL_GAIN_THRESHOLD:
                if roi < self.cfg.SMALL_GAIN_LOCKED_ROI: return {'action': 'SELL', 'reason': 'SMALL_GAIN_P'}
        if held_mins > self.cfg.TIME_STOP_MINS and roi < self.cfg.TIME_STOP_ROI: return {'action': 'SELL', 'reason': 'TIME_STOP'}
        return None

    def _check_exit_liquidity_guard(self, ctx: dict, current_price: float) -> dict:
        bid, ask = ctx.get('bid', 0.0), ctx.get('ask', 0.0)
        if bid > 0 and ask > 0 and current_price > 0.01:
            if (ask - bid) / current_price > self.cfg.MAX_SPREAD_PCT_EXIT: return {'action': 'SELL', 'reason': 'SPREAD_STOP'}
        return None

    def _get_dynamic_ladder(self, pos: dict) -> List[Tuple[float, float]]:
        if not self.cfg.DYNAMIC_LADDER_ENABLED: return self.cfg.LADDER_TIGHT
        init_alpha = abs(pos.get('init_ctx', {}).get('alpha', 0.0))
        if init_alpha >= self.cfg.HIGH_ALPHA_WIDE_THRESHOLD: return self.cfg.LADDER_WIDE
        return self.cfg.LADDER_TIGHT

    def _evaluate_profit_guards(self, ctx: dict, pos: dict, held_mins: float, current_roi: float) -> dict:
        max_roi = pos.get('max_roi', 0.0)
        initial_event_prob = pos.get('init_ctx', {}).get('event_prob', 0.0)
        is_event_trade = initial_event_prob > self.cfg.EVENT_PROB_THRESHOLD
        if is_event_trade and held_mins < 10.0: return None
        if current_roi >= self.cfg.FLASH_PROTECT_TRIGGER:
            if not (is_event_trade and held_mins < self.cfg.EVENT_HODL_MINS):
                if current_roi <= self.cfg.FLASH_PROTECT_EXIT: return {'action': 'SELL', 'reason': f"PROT_L0_FLASH|ROI:{current_roi:.2f}"}
        ladder = self._get_dynamic_ladder(pos)
        for trigger, floor in reversed(ladder):
            effective_trigger = trigger
            if trigger == 0.12 and is_event_trade and ladder == self.cfg.LADDER_TIGHT: effective_trigger = 0.15
            if max_roi >= effective_trigger:
                if current_roi < floor: return {'action': 'SELL', 'reason': f"STEP_PROT({max_roi:.1%}->{current_roi:.1%})|T:{trigger}"}
                break 
        return None

    def _check_stop_loss_guards(self, ctx: dict, pos: dict, roi: float, stock_roi: float) -> dict:
        is_stock_adverse = (pos['dir'] == 1 and stock_roi < -0.0015) or (pos['dir'] == -1 and stock_roi > 0.0015)
        if is_stock_adverse and roi < -0.05: return {'action': 'SELL', 'reason': f"COND_STOP:{roi:.2%}|S:{stock_roi:.2%}"}
        if roi >= self.cfg.STOP_LOSS: return None
        abs_sl = self.cfg.ABSOLUTE_STOP_LOSS
        if ctx.get('stock_iv', 0) > 0.8: abs_sl = -0.12
        if roi < abs_sl: return {'action': 'SELL', 'reason': f"HARD_STOP:{roi:.2%}"}
        return None

    def _check_macd_fade(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        if pos['max_roi'] > self.cfg.MACD_FADE_MIN_ROI and held_mins > 1.0:
            macd_slope = ctx.get('macd_hist_slope', 0.0)
            if (pos['dir'] == 1 and macd_slope < 0) or (pos['dir'] == -1 and macd_slope > 0): return {'action': 'SELL', 'reason': f"MACD_FADE({roi:.1%})"}
        return None

    def _check_signal_flip(self, ctx: dict, pos: dict, held_mins: float) -> dict:
        if held_mins <= 2.0: return None
        alpha_val = ctx['alpha_z']
        if (pos['dir'] == 1 and alpha_val < -self.cfg.ALPHA_FLIP_THRESHOLD) or (pos['dir'] == -1 and alpha_val > self.cfg.ALPHA_FLIP_THRESHOLD): return {'action': 'SELL', 'reason': "FLIP"}
        return None
