#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: strategy_core_v1.py
描述: [V15 策略内核 - 像素级复刻版 + 极速防线升级]
基准: simple_option_backtest_production_v15_stock_double_guard.py
职责: 
    - 接收 Context (包含已修正的 Alpha, Vol, MACD, 正股动量)
    - 输出 严格的买卖信号 (Signal)
    - [Log] 增加平仓决策过程的诊断日志 (Diagnostic Logs)，便于回放排错。
"""

import logging
logger = logging.getLogger("StrategyCore")

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
    
from config import TARGET_SYMBOLS
from strategy_config0 import StrategyConfig


class StrategyCoreV0:
    def __init__(self, config: StrategyConfig = None):
        self.cfg = config if config else StrategyConfig()
        self._last_reject_reason = None

    def decide_entry(self, ctx: dict) -> dict:
        """
        [开仓决策 - 核心路由入口]
        """
        self._last_reject_reason = None
        # 1. 基础状态与时间窗口拦截
        if not self._check_entry_pre_conditions(ctx):
            if not self._last_reject_reason:
                self._last_reject_reason = 'pre_conditions'
            return None

        # 2. 优先判定: 慢牛绿灯通道 (Channel B)
        sig = None
        if self.cfg.SLOW_BULL_CHANNEL_ENABLED:
            sig = self._check_channel_b_slow_bull(ctx)
                
        # 3. 默认判定: 传统高爆动量通道 (Channel A)
        if not sig:
            sig = self._check_channel_a_momentum(ctx)
        
        if not sig:
            if not self._last_reject_reason:
                self._last_reject_reason = 'both_channels_none'
            return None
        
        # 4. 流动性与点差拦截
        if not self._check_entry_liquidity_guard(ctx):
            self._last_reject_reason = 'liquidity_guard'
            return None
                
        return sig

    def _check_entry_pre_conditions(self, ctx: dict) -> bool:
        """检查基础状态、持仓、冷却以及时间窗口"""
        if not ctx.get('is_ready', False): return False
        if ctx.get('is_banned', False): return False
        if ctx['position'] != 0: return False
        if ctx['curr_ts'] < ctx.get('cooldown_until', 0): return False

        t = ctx['time']
        if t.hour == 9 and t.minute < self.cfg.START_MINUTE: return False
        if (t.hour == self.cfg.NO_ENTRY_HOUR and t.minute >= self.cfg.NO_ENTRY_MINUTE) or t.hour > self.cfg.NO_ENTRY_HOUR:
            return False
            
        # 3. 市场状态卫士 (Regime Guard - Choppiness Filter)
        if getattr(self.cfg, 'REGIME_GUARD_ENABLED', False) and getattr(self.cfg, 'REGIME_ENTRY_GUARD_ENABLED', False):
            if not self._check_regime_guard(ctx): return False

        return True

    def _check_regime_guard(self, ctx: dict) -> bool:
        """
        [NEW] 核心对齐: 拦截高洗盘频率行情
        通过反转次数统计来识别 Alpha 失效的市场环境
        """
        if 'is_volatile_regime' in ctx:
            if ctx.get('is_volatile_regime', False):
                logger.info(f"🚫 [REGIME_GUARD] {ctx['symbol']} Blocked. Volatile regime detected.")
                return False
            return True

        reversal_count = ctx.get('regime_reversal_count', 0)
        limit = getattr(self.cfg, 'REGIME_REVERSAL_THRESHOLD', 6)
        if reversal_count > limit:
            logger.info(f"🚫 [REGIME_GUARD] {ctx['symbol']} Blocked. Recent Reversals: {reversal_count} (Limit: {limit})")
            return False
        return True

    def _check_entry_liquidity_guard(self, ctx: dict) -> bool:
        """检查进场时的买卖价差与散度 (必须具备有效的 Bid/Ask)"""
        bid = ctx.get('bid', 0.0)
        ask = ctx.get('ask', 0.0)
        curr_p = ctx.get('curr_price', 0.0)
        
        # 🚨 [终极严密] 如果没有买卖盘口，或者价格异常，绝对禁止开仓！
        if bid <= 0.01 or ask <= 0.01 or curr_p <= 0.01:
            return False
            
        # 动态门槛计算
        if curr_p <= 0.5:
            dynamic_spread_th = 0.20
        elif curr_p >= 5.0:
            dynamic_spread_th = 0.10
        else:
            dynamic_spread_th = 0.20 - (curr_p - 0.5) * (0.10 / 4.5)
        
        spread_pct = (ask - bid) / curr_p
        if spread_pct > dynamic_spread_th: return False
            
        div = ctx.get('spread_divergence', 0.0)
        if div > self.cfg.MAX_SPREAD_DIVERGENCE: return False
            
        return True

    def _check_channel_b_slow_bull(self, ctx: dict) -> dict:
        """
        [通道 B: 慢牛杀波段] 
        针对 VIX 被压制、波动消失但大盘在缓慢爬升的顺风局。
        """
        vol_z = ctx['vol_z']
        alpha_val = ctx['alpha_z']
        action = 1 if alpha_val > 0 else -1
        
        # 慢牛只允许做多接顺风局
        if action != 1: return None
        
        spy_roc = ctx.get('spy_roc', 0.0)
        qqq_roc = ctx.get('qqq_roc', 0.0)
        macd_hist = ctx.get('macd_hist', 0.0)

        # 1. 核心状态校验
        cond_vol = vol_z <= self.cfg.SLOW_BULL_MAX_VOL_Z
        cond_alpha = alpha_val >= self.cfg.SLOW_BULL_ALPHA_THRESHOLD
        cond_macd = macd_hist >= self.cfg.SLOW_BULL_MACD_THRESHOLD
        cond_index = (spy_roc > self.cfg.SLOW_BULL_MIN_INDEX_ROC) or (qqq_roc > self.cfg.SLOW_BULL_MIN_INDEX_ROC)

        if not (cond_vol and cond_alpha and cond_macd and cond_index):
            return None

        # 2. 个股慢牛顺势卫士 (Stock Positive Momentum Guard)
        if ctx['stock_roc'] <= 0.0005:  
            return None
            
        # [🔥 新增] 瞬时动量校验
        if ctx.get('snap_roc', 0.0) < self.cfg.MIN_LAST_SNAP_ROC:
            return None

        return {
            'action': 'BUY',
            'dir': 1,
            'tag': 'CALL_ATM',
            'legacy_tag': 'opt_8',
            'score': abs(alpha_val),
            'reason': f"CH_B_SLOW|A:{alpha_val:.2f}(Th:{self.cfg.SLOW_BULL_ALPHA_THRESHOLD:.1f})|V:{vol_z:.1f}"
        }

    def _check_channel_a_momentum(self, ctx: dict) -> dict:
        """
        [通道 A: 传统高爆动量] 
        """
        vol_z = ctx['vol_z']
        alpha_val = ctx['alpha_z']
        action = 1 if alpha_val > 0 else -1
        
        # 1. 波动率严格过滤
        if not (self.cfg.VOL_MIN_Z < vol_z < self.cfg.VOL_MAX_Z): return None

        # 2. 动态 Alpha 爆发门槛
        symbol = ctx.get('symbol', 'UNKNOWN')
        dynamic_threshold = self._calculate_dynamic_alpha_threshold(symbol, vol_z)
        if abs(alpha_val) < dynamic_threshold: return None

        # 3. 个股爆发顺势校验 (Stock Momentum Guard)
        if not self._check_stock_momentum_guard(ctx, action): return None

        # 4. 大盘实时护栏 (Index Guard)
        if not self._check_index_guard(ctx, action): return None

        # 5. MACD Histogram 绝对方向确认 
        if not self._check_macd_confirm(ctx, action): return None

        return {
            'action': 'BUY',
            'dir': action,
            'tag': 'CALL_ATM' if action == 1 else 'PUT_ATM',
            'legacy_tag': 'opt_8' if action == 1 else 'opt_0',
            'score': abs(alpha_val),
            'reason': f"CH_A_MOM|A:{alpha_val:.2f}(Th:{dynamic_threshold:.1f})|V:{vol_z:.1f}"
        }

    def _calculate_dynamic_alpha_threshold(self, symbol: str, vol_z: float) -> float:
        """根据分级参数和实时波动率动态调整 Alpha 门槛"""
        base_threshold = self.cfg.ALPHA_ENTRY_THRESHOLD
        
        dynamic_threshold = base_threshold
        if vol_z > 2.0:
            dynamic_threshold += (vol_z - 2.0) * 0.5
        return min(dynamic_threshold, 3.0)

    def _check_stock_momentum_guard(self, ctx: dict, action: int) -> bool:
        """双重动量守卫：确保 5min 和 1min 趋势同步顺势"""
        # A. 长期趋势 (5min)
        if action == 1 and ctx['stock_roc'] < -self.cfg.STOCK_MOMENTUM_TOLERANCE: return False
        if action == -1 and ctx['stock_roc'] > self.cfg.STOCK_MOMENTUM_TOLERANCE: return False

        # B. 瞬时趋势 (1min Snap)
        snap_roc = ctx.get('snap_roc', 0.0)
        
        # [垂直尖刺拦截]
        if abs(snap_roc) > self.cfg.MAX_SNAP_ROC_LIMIT: return False

        if action == 1 and snap_roc < self.cfg.MIN_LAST_SNAP_ROC: return False
        if action == -1 and snap_roc > -self.cfg.MIN_LAST_SNAP_ROC: return False
        return True

    def _check_index_guard(self, ctx: dict, action: int) -> bool:
        """大盘卫士：根据回测统计，严格禁止在上升趋势中做空，但允许在下跌趋势中抄底"""
        if not self.cfg.INDEX_GUARD_ENABLED: return True
        
        spy_roc = ctx.get('spy_roc', 0.0)
        qqq_roc = ctx.get('qqq_roc', 0.0)
        if abs(spy_roc) < 1e-9 and abs(qqq_roc) < 1e-9: return True # 无数据跳过

        threshold = self.cfg.INDEX_ROC_THRESHOLD
        idx_trend = ctx.get('index_trend', 0)
        
        # [🔥 终极护栏] 根据回测统计：
        # 上升趋势中顶风做空 (PUT) 胜率极低且长期亏损，无脑拦截！
        if action == -1 and self.cfg.INDEX_GUARD_SHORT_BLOCK_ENABLED:
            # 强化拦截：无论是全天大势向上 (idx_trend==1)，还是短期 5 分钟在拉升 (>0.01%)，统统不准做空！
            if idx_trend == 1 or spy_roc > 0.0001 or qqq_roc > 0.0001:
                return False
            
        # 传统单日 ROC 兜底过滤
        if action == 1:
            if spy_roc < threshold or qqq_roc < threshold: return False 
            
        return True

    def _check_macd_confirm(self, ctx: dict, action: int) -> bool:
        """MACD 柱线确认：确保动能方向与信号一致"""
        if not self.cfg.MACD_HIST_CONFIRM_ENABLED: return True
        
        macd_hist = ctx.get('macd_hist', 0.0)
        th = self.cfg.MACD_HIST_THRESHOLD
        if action == 1 and macd_hist <= th: return False
        if action == -1 and macd_hist >= -th: return False
        return True

    def _get_dynamic_ladder(self, pos: dict) -> List[Tuple[float, float]]:
        """根据初始信号强度动态选择不同的利润阶梯 (V0 默认为 TIGHT)"""
        if not getattr(self.cfg, 'DYNAMIC_LADDER_ENABLED', False):
            return self.cfg.LADDER_TIGHT
        
        # 兼容性设计: 如果以后需要像 V1 一样动态切换
        init_alpha = abs(pos.get('init_ctx', {}).get('alpha_z', 0.0))
        if init_alpha >= getattr(self.cfg, 'HIGH_ALPHA_WIDE_THRESHOLD', 2.5):
            return self.cfg.LADDER_WIDE
        return self.cfg.LADDER_TIGHT

    def _evaluate_profit_guards(self, pos: dict, current_roi: float) -> dict:
        """
        [🔥 抽象重构] 统一利润锁定引擎 (Unified Profit Guard Engine)
        自顶向下拦截，融合了 微利保本、阶梯止盈 和 暴利追踪。
        """
        max_roi = pos.get('max_roi', 0.0)
        
        # 1. 逆势单拦截 (大盘趋势不支持，利润极易回吐)
        is_counter_trend = False
        if 'entry_spy_roc' in pos:
            if pos['dir'] == 1 and pos['entry_spy_roc'] < -0.0001: is_counter_trend = True
            elif pos['dir'] == -1 and pos['entry_spy_roc'] > 0.0001: is_counter_trend = True
                
        if is_counter_trend:
            if max_roi > self.cfg.COUNTER_TREND_PROTECT_TRIGGER and current_roi < self.cfg.COUNTER_TREND_PROTECT_EXIT:
                return {'action': 'SELL', 'reason': f"PROTECT_COUNTER({max_roi:.1%}->{current_roi:.1%})"}

        # 2. 顺势单：动态防线判定
        
        # Level Epic: 暴利追踪 (史诗级别)
        if max_roi >= self.cfg.TRAILING_TRIGGER_ROI:
            trailing_exit = max_roi * self.cfg.TRAILING_KEEP_RATIO
            if current_roi < trailing_exit:
                return {'action': 'SELL', 'reason': f"TRAILING_EPIC({max_roi:.1%}->{current_roi:.1%})"}

        # Step Ladder: 阶梯防线 (按 LADDER 配置循环评估)
        ladder = self._get_dynamic_ladder(pos)
        for trigger, floor in reversed(ladder):
            if max_roi >= trigger:
                if current_roi < floor:
                    return {
                        'action': 'SELL', 
                        'reason': f"STEP_PROT({max_roi:.1%}->{current_roi:.1%})|T:{trigger:.2f}"
                    }
                # 一旦命中最高档位触发器，即使未达成回撤卖出条件，也直接 Break，跳过低档位判定
                break

        # Level 0: 极速保本 (Flash Protect)
        if max_roi >= self.cfg.FLASH_PROTECT_TRIGGER:
            if current_roi <= self.cfg.FLASH_PROTECT_EXIT:
                return {'action': 'SELL', 'reason': f"FLASH_PROT_L0({max_roi:.1%}->{current_roi:.1%})"}
        
        # 3. [Diagnostic] 如果都没触发，输出一行 debug
        if max_roi > 0.03:
             logger.debug(f"⚖️ [Profit Guard Pass] {pos.get('symbol', 'UNK')} | Max ROI: {max_roi:.1%} | Curr ROI: {current_roi:.1%}")
                
        return None

    def check_exit(self, ctx: dict) -> dict:
        """
        [平仓/风控决策 - 核心路由入口]
        """
        pos = ctx.get('holding')
        if not pos: return None
        
        # 1. 基础硬性拦截 (EOD/超时)
        sig = self._check_exit_pre_conditions(ctx, pos)
        if sig: return sig

        # 2. 趋势反转拦截 (指数/动量)
        sig = self._check_trend_reversal_guard(ctx, pos)
        if sig: return sig

        # 3. 价格相关指标预计算
        held_mins = ctx.get('held_mins', (ctx['curr_ts'] - pos['entry_ts']) / 60.0)
        curr_price = ctx['curr_price']
        roi = 0.0 if curr_price <= 0.001 else (curr_price - pos['entry_price']) / pos['entry_price']
        stock_roi = (ctx['curr_stock'] - pos['entry_stock']) / pos['entry_stock'] if pos['entry_stock'] > 0 else 0.0

        # 4. 正股硬止损 (Stock Hard Stop)
        sig = self._check_stock_hard_stop(ctx, pos, held_mins, stock_roi)
        if sig: return sig

        # 5. 时间止损、僵尸单与微利保护
        sig = self._check_time_and_inactivity_stops(ctx, pos, held_mins, roi)
        if sig: return sig

        # 6. 流动性与点差恶化拦截
        sig = self._check_exit_liquidity_guard(ctx, curr_price)
        if sig: return sig

        # 7. 止损防线 (Condition & Hard Stop)
        sig = self._check_stop_loss_guards(ctx, pos, roi, stock_roi)
        if sig: return sig

        # 8. 统一利润锁定引擎 (L0~L4)
        sig = self._evaluate_profit_guards(pos, roi)
        if sig: return sig

        # 9. 动能衰竭极速离场 (MACD Fade)
        sig = self._check_macd_fade(ctx, pos, held_mins, roi)
        if sig: return sig

        # 10. 信号反转 (Signal Flip)
        sig = self._check_signal_flip(ctx, pos, held_mins)
        if sig: return sig

        # [Diagnostic] Final skip log
        if held_mins > 20: # 只针对老僵尸单记录，避免每个tick都刷屏
            logger.debug(f"⚖️ [Exit Check Final Skip] {ctx['symbol']} | Held: {held_mins:.1f}m | ROI: {roi:.2%} | MACD Slope: {ctx.get('macd_hist_slope',0.0):.4f}")

        return None

    def _check_exit_pre_conditions(self, ctx: dict, pos: dict) -> dict:
        """检查收盘强平、逆势超时等基础拦截逻辑"""
        t = ctx['time']
        if (t.hour == self.cfg.CLOSE_HOUR and t.minute >= self.cfg.CLOSE_MINUTE) or (t.hour > self.cfg.CLOSE_HOUR):
            return {'action': 'SELL', 'reason': 'EOD_CLEAR'}

        held_mins = ctx.get('held_mins', 0.0)
        idx_trend = ctx.get('index_trend', 0)
        is_counter_trend = (pos['dir'] == 1 and idx_trend == -1) or (pos['dir'] == -1 and idx_trend == 1)
        
        if is_counter_trend and held_mins >= self.cfg.COUNTER_TREND_MAX_MINS:
            return {'action': 'SELL', 'reason': f"CT_TIMEOUT:{held_mins:.0f}m"}
        return None

    def _check_trend_reversal_guard(self, ctx: dict, pos: dict) -> dict:
        """检查大盘趋势是否发生方向性逆转"""
        if not self.cfg.INDEX_REVERSAL_EXIT_ENABLED: return None
        
        idx_trend = ctx.get('index_trend', 0)
        entry_idx_trend = pos.get('entry_index_trend', 0)
        
        if entry_idx_trend >= 0 and pos['dir'] == 1 and idx_trend == -1:
            return {'action': 'SELL', 'reason': "IDX_REVERSAL_FALL"}
        if entry_idx_trend <= 0 and pos['dir'] == -1 and idx_trend == 1:
            return {'action': 'SELL', 'reason': "IDX_REVERSAL_RISE"}
        return None

    def _check_stock_hard_stop(self, ctx: dict, pos: dict, held_mins: float, stock_roi: float) -> dict:
        """检查正股硬止损，根据 Regime 状态、Alpha 置信度和持仓时间动态调整"""
        if pos['entry_stock'] <= 0 or ctx['curr_stock'] <= 0: return None

        alpha_val = ctx.get('alpha_z', 0.0)
        
        # [🔥 Regime-Adaptive] 根据全局市场状态切换正股止损档位
        # calm     -> 0.003 / 0.005
        # mixed    -> 0.0022 / 0.004
        # volatile -> 0.0015 / 0.003
        is_volatile_regime = False
        regime_band = str(ctx.get('regime_band', 'calm') or 'calm').lower()
        if getattr(self.cfg, 'REGIME_ADAPTIVE_STOCK_STOP_ENABLED', False):
            if regime_band == 'volatile':
                is_volatile_regime = True
                tight_sl = getattr(self.cfg, 'STOCK_HARD_STOP_TIGHT_VOLATILE', self.cfg.STOCK_HARD_STOP_TIGHT)
                loose_sl = getattr(self.cfg, 'STOCK_HARD_STOP_LOOSE_VOLATILE', self.cfg.STOCK_HARD_STOP_LOOSE)
            elif regime_band == 'mixed':
                tight_sl = getattr(self.cfg, 'STOCK_HARD_STOP_TIGHT_MIXED', self.cfg.STOCK_HARD_STOP_TIGHT)
                loose_sl = getattr(self.cfg, 'STOCK_HARD_STOP_LOOSE_MIXED', self.cfg.STOCK_HARD_STOP_LOOSE)
            else:
                tight_sl = self.cfg.STOCK_HARD_STOP_TIGHT
                loose_sl = self.cfg.STOCK_HARD_STOP_LOOSE
        else:
            tight_sl = self.cfg.STOCK_HARD_STOP_TIGHT
            loose_sl = self.cfg.STOCK_HARD_STOP_LOOSE

        # 确定基础门槛：高置信度 Alpha 使用宽松档，否则使用紧缩档
        base_th = loose_sl if abs(alpha_val) >= self.cfg.HIGH_CONFIDENCE_THRESHOLD else tight_sl
        
        # 初始 3 分钟宽限期
        is_in_buffer = held_mins < 3.0
        current_th = max(base_th * 3.0, 0.005) if is_in_buffer else base_th
        tag = "(BUFFER)" if is_in_buffer else ""
        regime_tag = f"[{regime_band.upper()}]"

        if (pos['dir'] == 1 and stock_roi < -current_th) or (pos['dir'] == -1 and stock_roi > current_th):
            return {'action': 'SELL', 'reason': f"STOCK_STOP{regime_tag}{tag}:{stock_roi:.2%}(Th:{current_th:.2%})"}
        return None

    def _check_time_and_inactivity_stops(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        """检查僵尸持仓、小利保本以及长期不涨的时间止损"""
        # 1. 僵尸持仓 (20min 毫无波动)
        if held_mins >= self.cfg.ZOMBIE_EXIT_MINS and abs(roi) < 0.02:
            return {'action': 'SELL', 'reason': f"ZOMBIE_STOP:{held_mins:.0f}m"}
            
        # 2. 小利保护 (曾达到 8%，若跌回 4% 则锁定利润)
        if held_mins >= self.cfg.SMALL_GAIN_MINS and pos['max_roi'] >= self.cfg.SMALL_GAIN_THRESHOLD:
            if roi < self.cfg.SMALL_GAIN_LOCKED_ROI:
                return {'action': 'SELL', 'reason': f"SMALL_GAIN_P:{roi:.1%}"}

        # 3. 长期时间止损 (30min 未达 5% 收益)
        if held_mins > self.cfg.TIME_STOP_MINS and roi < self.cfg.TIME_STOP_ROI:
             return {'action': 'SELL', 'reason': f"TIME_STOP:{held_mins:.0f}m"}
        return None

    def _check_exit_liquidity_guard(self, ctx: dict, current_price: float) -> dict:
        """检查平仓时的流动性是否恶化（点差过大）"""
        bid = ctx.get('bid', 0.0)
        ask = ctx.get('ask', 0.0)
        if bid > 0 and ask > 0 and current_price > 0.01:
            spread_pct = (ask - bid) / current_price
            if spread_pct > self.cfg.MAX_SPREAD_PCT_EXIT:
                return {'action': 'SELL', 'reason': f"SPREAD_STOP:{spread_pct:.2%}"}
        return None

    def _check_stop_loss_guards(self, ctx: dict, pos: dict, roi: float, stock_roi: float) -> dict:
        """检查条件止损与硬性止损防线"""
        # 只有在亏损超过基础 STOP_LOSS (10%) 时才启动
        if roi >= self.cfg.STOP_LOSS: return None
        
        # A. 绝对硬止损 (15%)
        abs_sl = self.cfg.ABSOLUTE_STOP_LOSS
        if ctx.get('stock_iv', 0) > 0.8: abs_sl = -0.12 # 高波动标的更保守
        if roi < abs_sl:
            return {'action': 'SELL', 'reason': f"HARD_STOP:{roi:.2%}"}
        
        # B. 条件止损 (股价也显示逆势)
        is_stock_adverse = (pos['dir'] == 1 and stock_roi < 0) or (pos['dir'] == -1 and stock_roi > 0)
        if is_stock_adverse:
            return {'action': 'SELL', 'reason': f"COND_STOP:{roi:.2%}|S:{stock_roi:.2%}"}
        return None

    def _check_macd_fade(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        """检测 MACD 动能衰竭，实现在微利区间的极速收割"""
        macd_slope = ctx.get('macd_hist_slope', 0.0)
        # 宽限 1 分钟，且浮盈超过 MACD_FADE 门槛 (3%)
        if pos['max_roi'] > self.cfg.MACD_FADE_MIN_ROI and held_mins > 1.0:
            if (pos['dir'] == 1 and macd_slope < 0) or (pos['dir'] == -1 and macd_slope > 0):
                return {'action': 'SELL', 'reason': f"MACD_FADE({roi:.1%})"}
        return None

    def _check_signal_flip(self, ctx: dict, pos: dict, held_mins: float) -> dict:
        """检查 Alpha 信号是否发生彻底反向翻转"""
        if held_mins <= 2.0: return None # 给出 2 分钟冷静期，防止抖动
        
        alpha_val = ctx['alpha_z']
        if (pos['dir'] == 1 and alpha_val < -self.cfg.ALPHA_FLIP_THRESHOLD) or \
           (pos['dir'] == -1 and alpha_val > self.cfg.ALPHA_FLIP_THRESHOLD):
             return {'action': 'SELL', 'reason': "FLIP"}
        return None
