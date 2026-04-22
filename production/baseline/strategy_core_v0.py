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


# ============================================================================
# [Gate Trace Registry]
# 统一的策略门禁编号 ↔ 含义映射, 供 Dashboard 和日志系统使用。
# 维护约束: 此常量任何改动都必须同步更新 dashboard 的渲染表头, 否则 UI 会掉色。
# 命名规则:
#   E*  = Entry Gate (开仓决策链)
#   X*  = Exit Gate  (平仓/风控决策链)
#   G*  = Global Gate (跨标的门禁, 由 SE/OMS 填充, 不在本类内插桩)
# ============================================================================
GATE_REGISTRY = {
    # ---------------- Entry Gates ----------------
    "E1.is_ready":           "预热完成",
    "E2.is_banned":          "标的未被禁入",
    "E3.position_free":      "当前无持仓",
    "E4.cooldown":           "单标的冷却已释放",
    "E5.session_window":     "交易时段窗口",
    "E6.regime_guard":       "市场 Regime 守卫 (洗盘拦截)",
    "E7.ch_b_enabled":       "通道 B 慢牛是否开启",
    "E7.ch_b_direction":     "慢牛仅允许做多",
    "E7.ch_b_vol":           "慢牛 VOL_Z 过滤",
    "E7.ch_b_alpha":         "慢牛 Alpha 门槛",
    "E7.ch_b_macd":          "慢牛 MACD Hist 门槛",
    "E7.ch_b_index":         "慢牛 SPY/QQQ ROC 门槛",
    "E8.ch_b_stock_momentum": "慢牛个股顺势 stock_roc > 0.0005",
    "E8.ch_b_snap_roc":      "慢牛瞬时动量 snap_roc",
    "E9.ch_a_vol_range":     "Ch-A 波动率窗口",
    "E10.ch_a_alpha_dyn":    "Ch-A 动态 Alpha 门槛",
    "E11.ch_a_stock_rocL":   "Ch-A 长期个股动量 (5m stock_roc)",
    "E11.ch_a_snap_spike":   "Ch-A 垂直尖刺拦截 (snap_roc)",
    "E11.ch_a_snap_dir":     "Ch-A 瞬时动量顺势",
    "E12.ch_a_index_guard":  "Ch-A 大盘护栏 (INDEX_GUARD)",
    "E13.ch_a_macd_confirm": "Ch-A MACD Hist 方向确认",
    "E14.liquidity_spread":  "点差 < 动态阈值",
    "E14.liquidity_bidask":  "bid/ask/价格有效",
    "E14.liquidity_div":     "点差散度 spread_divergence",
    # ---------------- Exit Gates -----------------
    "X1.eod_clear":           "EOD 收盘强平",
    "X2.ct_timeout":          "逆势超时 (COUNTER_TREND_MAX_MINS)",
    "X3.idx_reversal":        "大盘方向反转",
    "X4.stock_hard_stop":     "正股硬止损 (Regime-Adaptive)",
    "X5.zombie_stop":         "僵尸持仓 (ZOMBIE_EXIT_MINS)",
    "X6.small_gain":          "小利保护 (SMALL_GAIN)",
    "X7.time_stop":           "长期时间止损 (TIME_STOP)",
    "X8.spread_stop":         "平仓流动性恶化",
    "X9a.hard_stop":          "绝对硬止损 (ABSOLUTE_STOP_LOSS)",
    "X9b.cond_stop":          "条件止损 (个股同向亏损)",
    "X10.protect_counter":    "逆势单利润保护",
    "X10.trailing_epic":      "史诗级暴利追踪",
    "X10.step_protect":       "阶梯止盈 (LADDER)",
    "X10.flash_protect":      "极速保本 (FLASH_PROTECT)",
    "X11.macd_fade":          "动能衰减离场",
    "X12.signal_flip":        "Alpha 信号反转",
}


class StrategyCoreV0:
    def __init__(self, config: StrategyConfig = None):
        self.cfg = config if config else StrategyConfig()
        self._last_reject_reason = None
        # [Gate Trace] 决策链追踪容器。
        # decide_entry / check_exit 入口各自 clear 一次, 内部所有 _check_* 分支
        # 在 pass/block/skip 时各追加一条。SE 在拿到返回值后会消费这个列表写 Redis。
        # 设计要点:
        #   1. 必须保证 "零分配-热路径": 只在插桩点 append dict, 不做任何 deepcopy;
        #   2. 允许上层在任意时点读 (单线程模型, 读写都在同一个 SE tick 循环内);
        #   3. 即便发生异常, _trace 自身不能 raise, 避免污染策略主逻辑。
        self._last_gate_trace: List[dict] = []

    def _trace(self, gate: str, status: str, detail: str = ""):
        """
        记录一次 gate 评估结果。
        status ∈ {"pass", "block", "skip"}:
          - pass : gate 被评估且放行, 信号流继续下一条
          - block: gate 拦截, 决策链在此中止 (本 tick 返回 None)
          - skip : gate 被跳过 (例如 ENABLED=False 或前置条件不满足)
        detail: 触发数值 / 阈值 / 原因, 供 Dashboard 展示 (≤ 120 字符)
        """
        try:
            self._last_gate_trace.append({
                "gate": gate,
                "status": status,
                "detail": detail,
            })
        except Exception:
            # 任何异常都不允许逃逸到策略主逻辑
            pass

    # [Public Accessors]
    # 给上层 (SE / Dashboard Publisher) 一个稳定读取接口，避免直接耦合到私有字段名。
    def get_last_gate_trace(self) -> List[dict]:
        try:
            return list(self._last_gate_trace or [])
        except Exception:
            return []

    def get_last_reject_reason(self) -> str:
        try:
            return str(self._last_reject_reason or "")
        except Exception:
            return ""

    def decide_entry(self, ctx: dict) -> dict:
        """
        [开仓决策 - 核心路由入口]
        """
        self._last_reject_reason = None
        # 每次入口重置 gate 链路, 确保 SE 读到的是当前 tick 的决策快照
        self._last_gate_trace = []

        # 1. 基础状态与时间窗口拦截
        if not self._check_entry_pre_conditions(ctx):
            if not self._last_reject_reason:
                self._last_reject_reason = 'pre_conditions'
            return None

        # 2. 优先判定: 慢牛绿灯通道 (Channel B)
        sig = None
        if self.cfg.SLOW_BULL_CHANNEL_ENABLED:
            sig = self._check_channel_b_slow_bull(ctx)
        else:
            self._trace("E7.ch_b_enabled", "skip", "SLOW_BULL_CHANNEL_ENABLED=False")

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
        if not ctx.get('is_ready', False):
            self._trace("E1.is_ready", "block", "is_ready=False (warmup 未完成)")
            return False
        self._trace("E1.is_ready", "pass", "")

        if ctx.get('is_banned', False):
            self._trace("E2.is_banned", "block", "is_banned=True")
            return False
        self._trace("E2.is_banned", "pass", "")

        if ctx['position'] != 0:
            self._trace("E3.position_free", "block", f"position={ctx['position']}")
            return False
        self._trace("E3.position_free", "pass", "")

        cd = ctx.get('cooldown_until', 0)
        if ctx['curr_ts'] < cd:
            remain = int(max(0, (cd - ctx['curr_ts']) / 60))
            self._trace("E4.cooldown", "block", f"剩余 {remain}m (until={cd:.0f})")
            return False
        self._trace("E4.cooldown", "pass", "")

        t = ctx['time']
        if t.hour == 9 and t.minute < self.cfg.START_MINUTE:
            self._trace("E5.session_window", "block",
                        f"t={t.strftime('%H:%M')} < open({self.cfg.START_HOUR:02d}:{self.cfg.START_MINUTE:02d})")
            return False
        if (t.hour == self.cfg.NO_ENTRY_HOUR and t.minute >= self.cfg.NO_ENTRY_MINUTE) or t.hour > self.cfg.NO_ENTRY_HOUR:
            self._trace("E5.session_window", "block",
                        f"t={t.strftime('%H:%M')} ≥ no_entry({self.cfg.NO_ENTRY_HOUR:02d}:{self.cfg.NO_ENTRY_MINUTE:02d})")
            return False
        self._trace("E5.session_window", "pass", f"t={t.strftime('%H:%M')}")

        # 3. 市场状态卫士 (Regime Guard - Choppiness Filter)
        if getattr(self.cfg, 'REGIME_GUARD_ENABLED', False) and getattr(self.cfg, 'REGIME_ENTRY_GUARD_ENABLED', False):
            if not self._check_regime_guard(ctx):
                return False
        else:
            self._trace("E6.regime_guard", "skip", "REGIME_GUARD_ENABLED=False")

        return True

    def _check_regime_guard(self, ctx: dict) -> bool:
        """
        [NEW] 核心对齐: 拦截高洗盘频率行情
        通过反转次数统计来识别 Alpha 失效的市场环境
        """
        if 'is_volatile_regime' in ctx:
            if ctx.get('is_volatile_regime', False):
                logger.info(f"🚫 [REGIME_GUARD] {ctx['symbol']} Blocked. Volatile regime detected.")
                self._trace("E6.regime_guard", "block", "is_volatile_regime=True")
                return False
            self._trace("E6.regime_guard", "pass", "regime=calm/mixed")
            return True

        reversal_count = ctx.get('regime_reversal_count', 0)
        limit = getattr(self.cfg, 'REGIME_REVERSAL_THRESHOLD', 6)
        if reversal_count > limit:
            logger.info(f"🚫 [REGIME_GUARD] {ctx['symbol']} Blocked. Recent Reversals: {reversal_count} (Limit: {limit})")
            self._trace("E6.regime_guard", "block", f"reversals={reversal_count} > limit={limit}")
            return False
        self._trace("E6.regime_guard", "pass", f"reversals={reversal_count} ≤ {limit}")
        return True

    def _check_entry_liquidity_guard(self, ctx: dict) -> bool:
        """检查进场时的买卖价差与散度 (必须具备有效的 Bid/Ask)"""
        bid = ctx.get('bid', 0.0)
        ask = ctx.get('ask', 0.0)
        curr_p = ctx.get('curr_price', 0.0)
        
        # 🚨 [终极严密] 如果没有买卖盘口，或者价格异常，绝对禁止开仓！
        if bid <= 0.01 or ask <= 0.01 or curr_p <= 0.01:
            self._trace("E14.liquidity_bidask", "block",
                        f"bid={bid:.3f}/ask={ask:.3f}/p={curr_p:.3f}")
            return False
        self._trace("E14.liquidity_bidask", "pass", "")

        # 动态门槛计算
        if curr_p <= 0.5:
            dynamic_spread_th = 0.20
        elif curr_p >= 5.0:
            dynamic_spread_th = 0.10
        else:
            dynamic_spread_th = 0.20 - (curr_p - 0.5) * (0.10 / 4.5)
        
        spread_pct = (ask - bid) / curr_p
        if spread_pct > dynamic_spread_th:
            self._trace("E14.liquidity_spread", "block",
                        f"spread={spread_pct:.2%} > th={dynamic_spread_th:.2%}")
            return False
        self._trace("E14.liquidity_spread", "pass",
                    f"spread={spread_pct:.2%} ≤ {dynamic_spread_th:.2%}")

        div = ctx.get('spread_divergence', 0.0)
        if div > self.cfg.MAX_SPREAD_DIVERGENCE:
            self._trace("E14.liquidity_div", "block",
                        f"div={div:.4f} > {self.cfg.MAX_SPREAD_DIVERGENCE}")
            return False
        self._trace("E14.liquidity_div", "pass", f"div={div:.4f}")

        return True

    def _check_channel_b_slow_bull(self, ctx: dict) -> dict:
        """
        [通道 B: 慢牛杀波段] 
        针对 VIX 被压制、波动消失但大盘在缓慢爬升的顺风局。
        """
        self._trace("E7.ch_b_enabled", "pass", "")

        vol_z = ctx['vol_z']
        alpha_val = ctx['alpha_z']
        action = 1 if alpha_val > 0 else -1

        # 慢牛只允许做多接顺风局
        if action != 1:
            self._trace("E7.ch_b_direction", "block", f"alpha_z={alpha_val:.2f} < 0 (dir=-1 不接)")
            return None
        self._trace("E7.ch_b_direction", "pass", f"alpha_z={alpha_val:.2f} > 0")

        spy_roc = ctx.get('spy_roc', 0.0)
        qqq_roc = ctx.get('qqq_roc', 0.0)
        macd_hist = ctx.get('macd_hist', 0.0)

        # 1. 核心状态校验 (逐条追踪, 不合并, 方便定位到底哪条掉链)
        if not (vol_z <= self.cfg.SLOW_BULL_MAX_VOL_Z):
            self._trace("E7.ch_b_vol", "block",
                        f"vol_z={vol_z:.2f} > max={self.cfg.SLOW_BULL_MAX_VOL_Z}")
            return None
        self._trace("E7.ch_b_vol", "pass", f"vol_z={vol_z:.2f}")

        if not (alpha_val >= self.cfg.SLOW_BULL_ALPHA_THRESHOLD):
            self._trace("E7.ch_b_alpha", "block",
                        f"alpha={alpha_val:.2f} < {self.cfg.SLOW_BULL_ALPHA_THRESHOLD}")
            return None
        self._trace("E7.ch_b_alpha", "pass", f"alpha={alpha_val:.2f}")

        if not (macd_hist >= self.cfg.SLOW_BULL_MACD_THRESHOLD):
            self._trace("E7.ch_b_macd", "block",
                        f"macd_hist={macd_hist:.4f} < {self.cfg.SLOW_BULL_MACD_THRESHOLD}")
            return None
        self._trace("E7.ch_b_macd", "pass", f"macd_hist={macd_hist:.4f}")

        cond_index = (spy_roc > self.cfg.SLOW_BULL_MIN_INDEX_ROC) or (qqq_roc > self.cfg.SLOW_BULL_MIN_INDEX_ROC)
        if not cond_index:
            self._trace("E7.ch_b_index", "block",
                        f"spy={spy_roc:.4f}/qqq={qqq_roc:.4f} 均 ≤ {self.cfg.SLOW_BULL_MIN_INDEX_ROC}")
            return None
        self._trace("E7.ch_b_index", "pass", f"spy={spy_roc:.4f}/qqq={qqq_roc:.4f}")

        # 2. 个股慢牛顺势卫士 (Stock Positive Momentum Guard)
        if ctx['stock_roc'] <= 0.0005:
            self._trace("E8.ch_b_stock_momentum", "block", f"stock_roc={ctx['stock_roc']:.4f} ≤ 0.0005")
            return None
        self._trace("E8.ch_b_stock_momentum", "pass", f"stock_roc={ctx['stock_roc']:.4f}")

        # [🔥 新增] 瞬时动量校验
        snap = ctx.get('snap_roc', 0.0)
        if snap < self.cfg.MIN_LAST_SNAP_ROC:
            self._trace("E8.ch_b_snap_roc", "block",
                        f"snap_roc={snap:.4f} < {self.cfg.MIN_LAST_SNAP_ROC}")
            return None
        self._trace("E8.ch_b_snap_roc", "pass", f"snap_roc={snap:.4f}")

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
        if not (self.cfg.VOL_MIN_Z < vol_z < self.cfg.VOL_MAX_Z):
            self._trace("E9.ch_a_vol_range", "block",
                        f"vol_z={vol_z:.2f} ∉ ({self.cfg.VOL_MIN_Z}, {self.cfg.VOL_MAX_Z})")
            return None
        self._trace("E9.ch_a_vol_range", "pass", f"vol_z={vol_z:.2f}")

        # 2. 动态 Alpha 爆发门槛
        symbol = ctx.get('symbol', 'UNKNOWN')
        dynamic_threshold = self._calculate_dynamic_alpha_threshold(symbol, vol_z)
        if abs(alpha_val) < dynamic_threshold:
            self._trace("E10.ch_a_alpha_dyn", "block",
                        f"|alpha|={abs(alpha_val):.2f} < dyn_th={dynamic_threshold:.2f}")
            return None
        self._trace("E10.ch_a_alpha_dyn", "pass",
                    f"|alpha|={abs(alpha_val):.2f} ≥ dyn_th={dynamic_threshold:.2f}")

        # 3. 个股爆发顺势校验 (Stock Momentum Guard)
        if not self._check_stock_momentum_guard(ctx, action):
            return None

        # 4. 大盘实时护栏 (Index Guard)
        if not self._check_index_guard(ctx, action):
            return None

        # 5. MACD Histogram 绝对方向确认 
        if not self._check_macd_confirm(ctx, action):
            return None

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
        stock_roc = ctx['stock_roc']
        tol = self.cfg.STOCK_MOMENTUM_TOLERANCE

        # A. 长期趋势 (5min)
        if action == 1 and stock_roc < -tol:
            self._trace("E11.ch_a_stock_rocL", "block",
                        f"dir=+1 | stock_roc={stock_roc:.4f} < -{tol}")
            return False
        if action == -1 and stock_roc > tol:
            self._trace("E11.ch_a_stock_rocL", "block",
                        f"dir=-1 | stock_roc={stock_roc:.4f} > {tol}")
            return False
        self._trace("E11.ch_a_stock_rocL", "pass", f"dir={action} | stock_roc={stock_roc:.4f}")

        # B. 瞬时趋势 (1min Snap)
        snap_roc = ctx.get('snap_roc', 0.0)

        # [垂直尖刺拦截]
        if abs(snap_roc) > self.cfg.MAX_SNAP_ROC_LIMIT:
            self._trace("E11.ch_a_snap_spike", "block",
                        f"|snap_roc|={abs(snap_roc):.4f} > max={self.cfg.MAX_SNAP_ROC_LIMIT}")
            return False
        self._trace("E11.ch_a_snap_spike", "pass", f"|snap_roc|={abs(snap_roc):.4f}")

        if action == 1 and snap_roc < self.cfg.MIN_LAST_SNAP_ROC:
            self._trace("E11.ch_a_snap_dir", "block",
                        f"dir=+1 | snap_roc={snap_roc:.4f} < {self.cfg.MIN_LAST_SNAP_ROC}")
            return False
        if action == -1 and snap_roc > -self.cfg.MIN_LAST_SNAP_ROC:
            self._trace("E11.ch_a_snap_dir", "block",
                        f"dir=-1 | snap_roc={snap_roc:.4f} > -{self.cfg.MIN_LAST_SNAP_ROC}")
            return False
        self._trace("E11.ch_a_snap_dir", "pass", f"snap_roc={snap_roc:.4f}")
        return True

    def _check_index_guard(self, ctx: dict, action: int) -> bool:
        """大盘卫士：根据回测统计，严格禁止在上升趋势中做空，但允许在下跌趋势中抄底"""
        if not self.cfg.INDEX_GUARD_ENABLED:
            self._trace("E12.ch_a_index_guard", "skip", "INDEX_GUARD_ENABLED=False")
            return True

        spy_roc = ctx.get('spy_roc', 0.0)
        qqq_roc = ctx.get('qqq_roc', 0.0)
        if abs(spy_roc) < 1e-9 and abs(qqq_roc) < 1e-9:
            self._trace("E12.ch_a_index_guard", "skip", "spy/qqq 无数据")
            return True

        threshold = self.cfg.INDEX_ROC_THRESHOLD
        idx_trend = ctx.get('index_trend', 0)

        # [🔥 终极护栏] 根据回测统计：
        # 上升趋势中顶风做空 (PUT) 胜率极低且长期亏损，无脑拦截！
        if action == -1 and self.cfg.INDEX_GUARD_SHORT_BLOCK_ENABLED:
            if idx_trend == 1 or spy_roc > 0.0001 or qqq_roc > 0.0001:
                self._trace("E12.ch_a_index_guard", "block",
                            f"dir=-1 被上涨大盘拦截 | idx_trend={idx_trend} spy={spy_roc:.4f} qqq={qqq_roc:.4f}")
                return False

        # 传统单日 ROC 兜底过滤
        if action == 1:
            if spy_roc < threshold or qqq_roc < threshold:
                self._trace("E12.ch_a_index_guard", "block",
                            f"dir=+1 指数跌破 th={threshold} | spy={spy_roc:.4f} qqq={qqq_roc:.4f}")
                return False

        self._trace("E12.ch_a_index_guard", "pass",
                    f"dir={action} | spy={spy_roc:.4f} qqq={qqq_roc:.4f} idx_trend={idx_trend}")
        return True

    def _check_macd_confirm(self, ctx: dict, action: int) -> bool:
        """MACD 柱线确认：确保动能方向与信号一致"""
        if not self.cfg.MACD_HIST_CONFIRM_ENABLED:
            self._trace("E13.ch_a_macd_confirm", "skip", "MACD_HIST_CONFIRM_ENABLED=False")
            return True

        macd_hist = ctx.get('macd_hist', 0.0)
        th = self.cfg.MACD_HIST_THRESHOLD
        if action == 1 and macd_hist <= th:
            self._trace("E13.ch_a_macd_confirm", "block",
                        f"dir=+1 | macd_hist={macd_hist:.4f} ≤ th={th}")
            return False
        if action == -1 and macd_hist >= -th:
            self._trace("E13.ch_a_macd_confirm", "block",
                        f"dir=-1 | macd_hist={macd_hist:.4f} ≥ -th={-th}")
            return False
        self._trace("E13.ch_a_macd_confirm", "pass",
                    f"dir={action} | macd_hist={macd_hist:.4f}")
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
        # [🛡️ Defensive] entry_spy_roc 可能来自已污染的状态文件 (例: 历史 bug 存入 dict),
        # 一旦直接和 float 比较会抛 TypeError, 在 SE 主循环里把整批 tick 的 alpha 陪葬.
        # 这里统一转 float, 非法值按 0 处理 (即不触发逆势单保护, 宁可漏保护也不阻断主循环).
        is_counter_trend = False
        if 'entry_spy_roc' in pos:
            raw_spy_roc = pos['entry_spy_roc']
            try:
                spy_roc_val = float(raw_spy_roc) if not isinstance(raw_spy_roc, dict) else 0.0
            except (TypeError, ValueError):
                spy_roc_val = 0.0
            if pos['dir'] == 1 and spy_roc_val < -0.0001: is_counter_trend = True
            elif pos['dir'] == -1 and spy_roc_val > 0.0001: is_counter_trend = True
                
        if is_counter_trend:
            if max_roi > self.cfg.COUNTER_TREND_PROTECT_TRIGGER and current_roi < self.cfg.COUNTER_TREND_PROTECT_EXIT:
                self._trace("X10.protect_counter", "block",
                            f"逆势 max={max_roi:.1%}>trig={self.cfg.COUNTER_TREND_PROTECT_TRIGGER:.0%} cur={current_roi:.1%}<exit={self.cfg.COUNTER_TREND_PROTECT_EXIT:.0%}")
                return {'action': 'SELL', 'reason': f"PROTECT_COUNTER({max_roi:.1%}->{current_roi:.1%})"}
            self._trace("X10.protect_counter", "pass",
                        f"逆势但未触发 max={max_roi:.1%} cur={current_roi:.1%}")
        else:
            self._trace("X10.protect_counter", "skip", "顺势单")

        # 2. 顺势单：动态防线判定
        
        # Level Epic: 暴利追踪 (史诗级别)
        if max_roi >= self.cfg.TRAILING_TRIGGER_ROI:
            trailing_exit = max_roi * self.cfg.TRAILING_KEEP_RATIO
            if current_roi < trailing_exit:
                self._trace("X10.trailing_epic", "block",
                            f"max={max_roi:.1%} cur={current_roi:.1%} < trail={trailing_exit:.1%}")
                return {'action': 'SELL', 'reason': f"TRAILING_EPIC({max_roi:.1%}->{current_roi:.1%})"}
            self._trace("X10.trailing_epic", "pass",
                        f"Epic 就位 max={max_roi:.1%} cur={current_roi:.1%} 高于 trail={trailing_exit:.1%}")
        else:
            self._trace("X10.trailing_epic", "skip",
                        f"max_roi={max_roi:.1%} < trigger={self.cfg.TRAILING_TRIGGER_ROI:.0%}")

        # Step Ladder: 阶梯防线 (按 LADDER 配置循环评估)
        ladder = self._get_dynamic_ladder(pos)
        step_matched = False
        for trigger, floor in reversed(ladder):
            if max_roi >= trigger:
                step_matched = True
                if current_roi < floor:
                    self._trace("X10.step_protect", "block",
                                f"档位T={trigger:.2f} floor={floor:.2f} max={max_roi:.1%} cur={current_roi:.1%}")
                    return {
                        'action': 'SELL',
                        'reason': f"STEP_PROT({max_roi:.1%}->{current_roi:.1%})|T:{trigger:.2f}"
                    }
                self._trace("X10.step_protect", "pass",
                            f"最高档 T={trigger:.2f} floor={floor:.2f} 已守住 cur={current_roi:.1%}")
                break
        if not step_matched:
            self._trace("X10.step_protect", "skip", f"max_roi={max_roi:.1%} 未触达最低档")

        # Level 0: 极速保本 (Flash Protect)
        if max_roi >= self.cfg.FLASH_PROTECT_TRIGGER:
            if current_roi <= self.cfg.FLASH_PROTECT_EXIT:
                self._trace("X10.flash_protect", "block",
                            f"max={max_roi:.1%}≥trig={self.cfg.FLASH_PROTECT_TRIGGER:.0%} cur={current_roi:.1%}≤exit={self.cfg.FLASH_PROTECT_EXIT:.0%}")
                return {'action': 'SELL', 'reason': f"FLASH_PROT_L0({max_roi:.1%}->{current_roi:.1%})"}
            self._trace("X10.flash_protect", "pass",
                        f"max={max_roi:.1%} cur={current_roi:.1%} 高于保本线")
        else:
            self._trace("X10.flash_protect", "skip",
                        f"max_roi={max_roi:.1%} < trigger={self.cfg.FLASH_PROTECT_TRIGGER:.0%}")

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
        # 入口清空 gate 链路, 与 decide_entry 复用同一缓冲区 (SE tick 内串行执行)
        self._last_gate_trace = []
        
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
            self._trace("X1.eod_clear", "block",
                        f"t={t.strftime('%H:%M')} ≥ close({self.cfg.CLOSE_HOUR:02d}:{self.cfg.CLOSE_MINUTE:02d})")
            return {'action': 'SELL', 'reason': 'EOD_CLEAR'}
        self._trace("X1.eod_clear", "pass", f"t={t.strftime('%H:%M')}")

        held_mins = ctx.get('held_mins', 0.0)
        idx_trend = ctx.get('index_trend', 0)
        is_counter_trend = (pos['dir'] == 1 and idx_trend == -1) or (pos['dir'] == -1 and idx_trend == 1)

        if is_counter_trend and held_mins >= self.cfg.COUNTER_TREND_MAX_MINS:
            self._trace("X2.ct_timeout", "block",
                        f"dir={pos['dir']} idx_trend={idx_trend} held={held_mins:.1f}m ≥ {self.cfg.COUNTER_TREND_MAX_MINS}m")
            return {'action': 'SELL', 'reason': f"CT_TIMEOUT:{held_mins:.0f}m"}
        self._trace("X2.ct_timeout", "pass",
                    f"counter_trend={is_counter_trend} held={held_mins:.1f}m")
        return None

    def _check_trend_reversal_guard(self, ctx: dict, pos: dict) -> dict:
        """检查大盘趋势是否发生方向性逆转"""
        if not self.cfg.INDEX_REVERSAL_EXIT_ENABLED:
            self._trace("X3.idx_reversal", "skip", "INDEX_REVERSAL_EXIT_ENABLED=False")
            return None

        idx_trend = ctx.get('index_trend', 0)
        entry_idx_trend = pos.get('entry_index_trend', 0)

        if entry_idx_trend >= 0 and pos['dir'] == 1 and idx_trend == -1:
            self._trace("X3.idx_reversal", "block",
                        f"dir=+1 entry_trend={entry_idx_trend}→now=-1 (FALL)")
            return {'action': 'SELL', 'reason': "IDX_REVERSAL_FALL"}
        if entry_idx_trend <= 0 and pos['dir'] == -1 and idx_trend == 1:
            self._trace("X3.idx_reversal", "block",
                        f"dir=-1 entry_trend={entry_idx_trend}→now=+1 (RISE)")
            return {'action': 'SELL', 'reason': "IDX_REVERSAL_RISE"}
        self._trace("X3.idx_reversal", "pass",
                    f"dir={pos['dir']} entry_trend={entry_idx_trend} now={idx_trend}")
        return None

    def _check_stock_hard_stop(self, ctx: dict, pos: dict, held_mins: float, stock_roi: float) -> dict:
        """检查正股硬止损，根据 Regime 状态、Alpha 置信度和持仓时间动态调整"""
        if pos['entry_stock'] <= 0 or ctx['curr_stock'] <= 0:
            self._trace("X4.stock_hard_stop", "skip", "entry_stock/curr_stock 无效")
            return None

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
            self._trace("X4.stock_hard_stop", "block",
                        f"dir={pos['dir']} stock_roi={stock_roi:.2%} 破阈 ±{current_th:.2%} {regime_tag}{tag}")
            return {'action': 'SELL', 'reason': f"STOCK_STOP{regime_tag}{tag}:{stock_roi:.2%}(Th:{current_th:.2%})"}
        self._trace("X4.stock_hard_stop", "pass",
                    f"stock_roi={stock_roi:.2%} 在 ±{current_th:.2%} 内 {regime_tag}{tag}")
        return None

    def _check_time_and_inactivity_stops(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        """检查僵尸持仓、小利保本以及长期不涨的时间止损"""
        # 1. 僵尸持仓 (20min 毫无波动)
        if held_mins >= self.cfg.ZOMBIE_EXIT_MINS and abs(roi) < 0.02:
            self._trace("X5.zombie_stop", "block",
                        f"held={held_mins:.0f}m ≥ {self.cfg.ZOMBIE_EXIT_MINS}m | |roi|={abs(roi):.2%} < 2%")
            return {'action': 'SELL', 'reason': f"ZOMBIE_STOP:{held_mins:.0f}m"}
        self._trace("X5.zombie_stop", "pass",
                    f"held={held_mins:.1f}m roi={roi:.2%}")

        # 2. 小利保护 (曾达到 8%，若跌回 4% 则锁定利润)
        if held_mins >= self.cfg.SMALL_GAIN_MINS and pos['max_roi'] >= self.cfg.SMALL_GAIN_THRESHOLD:
            if roi < self.cfg.SMALL_GAIN_LOCKED_ROI:
                self._trace("X6.small_gain", "block",
                            f"max_roi={pos['max_roi']:.1%} ≥ {self.cfg.SMALL_GAIN_THRESHOLD:.1%} | roi={roi:.1%} < {self.cfg.SMALL_GAIN_LOCKED_ROI:.1%}")
                return {'action': 'SELL', 'reason': f"SMALL_GAIN_P:{roi:.1%}"}
            self._trace("X6.small_gain", "pass",
                        f"max_roi={pos['max_roi']:.1%} roi={roi:.1%} 仍保持")
        else:
            self._trace("X6.small_gain", "skip",
                        f"held={held_mins:.1f}m / max_roi={pos['max_roi']:.1%} 未触发前置条件")

        # 3. 长期时间止损 (30min 未达 5% 收益)
        if held_mins > self.cfg.TIME_STOP_MINS and roi < self.cfg.TIME_STOP_ROI:
            self._trace("X7.time_stop", "block",
                        f"held={held_mins:.0f}m > {self.cfg.TIME_STOP_MINS}m | roi={roi:.1%} < {self.cfg.TIME_STOP_ROI:.1%}")
            return {'action': 'SELL', 'reason': f"TIME_STOP:{held_mins:.0f}m"}
        self._trace("X7.time_stop", "pass",
                    f"held={held_mins:.1f}m roi={roi:.2%}")
        return None

    def _check_exit_liquidity_guard(self, ctx: dict, current_price: float) -> dict:
        """检查平仓时的流动性是否恶化（点差过大）"""
        bid = ctx.get('bid', 0.0)
        ask = ctx.get('ask', 0.0)
        if bid > 0 and ask > 0 and current_price > 0.01:
            spread_pct = (ask - bid) / current_price
            if spread_pct > self.cfg.MAX_SPREAD_PCT_EXIT:
                self._trace("X8.spread_stop", "block",
                            f"spread={spread_pct:.2%} > {self.cfg.MAX_SPREAD_PCT_EXIT:.0%}")
                return {'action': 'SELL', 'reason': f"SPREAD_STOP:{spread_pct:.2%}"}
            self._trace("X8.spread_stop", "pass", f"spread={spread_pct:.2%}")
        else:
            self._trace("X8.spread_stop", "skip", f"bid={bid:.3f}/ask={ask:.3f}/p={current_price:.3f}")
        return None

    def _check_stop_loss_guards(self, ctx: dict, pos: dict, roi: float, stock_roi: float) -> dict:
        """检查条件止损与硬性止损防线"""
        # 只有在亏损超过基础 STOP_LOSS (10%) 时才启动
        if roi >= self.cfg.STOP_LOSS:
            self._trace("X9a.hard_stop", "skip", f"roi={roi:.1%} ≥ {self.cfg.STOP_LOSS:.0%} 未启动止损层")
            self._trace("X9b.cond_stop", "skip", "父闸未开启")
            return None

        # A. 绝对硬止损 (15%)
        abs_sl = self.cfg.ABSOLUTE_STOP_LOSS
        if ctx.get('stock_iv', 0) > 0.8: abs_sl = -0.12 # 高波动标的更保守
        if roi < abs_sl:
            self._trace("X9a.hard_stop", "block", f"roi={roi:.1%} < abs_sl={abs_sl:.0%}")
            return {'action': 'SELL', 'reason': f"HARD_STOP:{roi:.2%}"}
        self._trace("X9a.hard_stop", "pass", f"roi={roi:.1%} ≥ abs_sl={abs_sl:.0%}")

        # B. 条件止损 (股价也显示逆势)
        is_stock_adverse = (pos['dir'] == 1 and stock_roi < 0) or (pos['dir'] == -1 and stock_roi > 0)
        if is_stock_adverse:
            self._trace("X9b.cond_stop", "block",
                        f"roi={roi:.1%} 且 stock 同向不利 (dir={pos['dir']} stock_roi={stock_roi:.2%})")
            return {'action': 'SELL', 'reason': f"COND_STOP:{roi:.2%}|S:{stock_roi:.2%}"}
        self._trace("X9b.cond_stop", "pass",
                    f"stock 未同向不利 (dir={pos['dir']} stock_roi={stock_roi:.2%})")
        return None

    def _check_macd_fade(self, ctx: dict, pos: dict, held_mins: float, roi: float) -> dict:
        """检测 MACD 动能衰竭，实现在微利区间的极速收割"""
        macd_slope = ctx.get('macd_hist_slope', 0.0)
        # 宽限 1 分钟，且浮盈超过 MACD_FADE 门槛 (3%)
        if pos['max_roi'] > self.cfg.MACD_FADE_MIN_ROI and held_mins > 1.0:
            if (pos['dir'] == 1 and macd_slope < 0) or (pos['dir'] == -1 and macd_slope > 0):
                self._trace("X11.macd_fade", "block",
                            f"dir={pos['dir']} slope={macd_slope:.4f} 反向 | max_roi={pos['max_roi']:.1%} held={held_mins:.1f}m")
                return {'action': 'SELL', 'reason': f"MACD_FADE({roi:.1%})"}
            self._trace("X11.macd_fade", "pass",
                        f"dir={pos['dir']} slope={macd_slope:.4f} 顺向")
        else:
            self._trace("X11.macd_fade", "skip",
                        f"max_roi={pos['max_roi']:.1%}≤{self.cfg.MACD_FADE_MIN_ROI:.0%} 或 held={held_mins:.1f}m≤1m")
        return None

    def _check_signal_flip(self, ctx: dict, pos: dict, held_mins: float) -> dict:
        """检查 Alpha 信号是否发生彻底反向翻转"""
        if held_mins <= 2.0:
            self._trace("X12.signal_flip", "skip", f"held={held_mins:.1f}m ≤ 2m (冷静期)")
            return None

        alpha_val = ctx['alpha_z']
        if (pos['dir'] == 1 and alpha_val < -self.cfg.ALPHA_FLIP_THRESHOLD) or \
           (pos['dir'] == -1 and alpha_val > self.cfg.ALPHA_FLIP_THRESHOLD):
            self._trace("X12.signal_flip", "block",
                        f"dir={pos['dir']} alpha={alpha_val:.2f} 越 ±{self.cfg.ALPHA_FLIP_THRESHOLD}")
            return {'action': 'SELL', 'reason': "FLIP"}
        self._trace("X12.signal_flip", "pass",
                    f"dir={pos['dir']} alpha={alpha_val:.2f}")
        return None
