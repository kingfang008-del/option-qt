#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay 共享类型 —— 避免 replay_harness / replay_session / event_replay 循环依赖。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayConfig:
    entry_threshold: float = 0.015
    entry_delay_bars: int = 1
    max_spread_pct: float = 0.06
    cooldown_bars: int = 5
    # TICK_FAST_HARD 属于异常风险退出；可使用更长的独立冷却，避免刚止损便在
    # 同一失效信号上重新开仓。None=沿用普通 cooldown_bars。
    tick_stop_cooldown_bars: Optional[int] = None
    # 快速止损代表该腿当日交易逻辑失效；锁同腿至收盘，反向腿仍可交易。
    tick_stop_lock_leg_for_day: bool = False
    # 任意亏损平仓后锁同腿至收盘(与 tick 锁共用 stopped_legs)。偏严,会误杀
    # 「早亏午赚」路径(Jul6 10:58 亏后 13:10 +38%)。
    loss_lock_leg_for_day: bool = False
    # 仅当单笔净亏损 <= 该阈值时才锁同腿(更负=更严)。None=任意亏损都锁。
    # -0.15:锁 Jul1 10:47(-18%),不锁 Jul6 早盘(-13%)以便接到午后大 PUT。
    loss_lock_leg_min_loss: Optional[float] = None
    # 同腿亏损后再开:该腿入场阈值 × mult(>1=更紧)。None/1=不抬高。
    loss_reentry_edge_mult: Optional[float] = None
    # 早盘 PUT 因 open30 结构失败被挡后,禁 PUT 直到该 session_bar(含)。
    # 防「挡掉 09:46 阴跌 PUT 后 10:47 同向再开」。None=不延长否决。
    # Jul W1:拉到 sb120 会误杀 Jul2/Jul7 早盘大 PUT,默认关。
    put_structure_veto_end_bar: Optional[int] = None
    # SPOT_THESIS(bounce-cut) 出场后短期禁同腿再开。防 Jul1 10:47 减亏后 11:01 再开更差 PUT。
    # 用 bar 数而非锁全日,以保留 Jul6 午后大 PUT。None/0=关闭。
    thesis_lock_leg_bars: Optional[int] = None
    long_only: bool = True
    entry_threshold_schedule: Optional[tuple] = None
    max_trades_per_day: Optional[int] = None
    daily_loss_stop: Optional[float] = None
    loss_streak_n: Optional[int] = None
    loss_streak_cooldown_bars: int = 30
    # 跨日腿级隔离：若前一交易日 PUT sleeve 的账户贡献
    # prod(1 + position_frac * net_return) - 1 <= 此阈值，下一交易日禁开 PUT。
    # None=关闭；例如 -0.02 表示 acct25 PUT 贡献不高于 -2%。
    next_day_put_quarantine_loss: Optional[float] = None
    # 可选的当日因果 regime 条件：仅当日前 lookback vix_z <= 此值才执行上述隔离。
    # None=不加 regime 条件。该输入应由每个交易日前已完成数据计算，禁止用当日未来值。
    next_day_put_quarantine_vix_z_max: Optional[float] = None
    # 可选的 VX contango 条件：仅当日前已完成 VX 日线的 VX2/VX1-1 >= 此值。
    # 使用无量纲期限结构而非 VIXY ETF 价格水平；None=不加此条件。
    next_day_put_quarantine_vx_slope_min: Optional[float] = None
    # 账户级次日防御：若前一日实际仓位复利贡献 <= 此阈值，次日所有腿进入防御。
    # 与 PUT quarantine 不同，它不锁腿；可组合减仓、延后入场和提高 q10。
    next_day_all_leg_defense_loss: Optional[float] = None
    # 防御日使用的绝对账户仓位比例；None=仍使用 position_frac。
    next_day_all_leg_defense_position_frac: Optional[float] = None
    # 防御日最早允许入场的 session bar；None=不额外延后。
    next_day_all_leg_defense_entry_start_bar: Optional[int] = None
    # 防御日所有腿共用的最低 edge_q10；缺失视为不通过。None=不额外收紧。
    next_day_all_leg_defense_edge_q10_floor: Optional[float] = None
    # 可选 VX contango 条件：仅当前一完成日桶 slope >= 此值才启用全腿防御。
    next_day_all_leg_defense_vx_slope_min: Optional[float] = None
    straddle_entry_threshold: Optional[float] = None
    max_straddles_per_day: Optional[int] = None
    # 会话内 bar 序号(09:30=0)允许新开仓区间;None=不限制
    session_entry_start_bar: Optional[int] = 0
    session_entry_end_bar: Optional[int] = 360
    # 0DTE 权利金 ROI 的经验 p10 约 -10%;高 edge bar 的预测 q10 中位约 -15%。
    # 默认 floor=-0.20(rolling/eval 口径)。QQQ frozen_norm 生产栈见 qqq.config.REPLAY(-0.25)。
    # None=不检查 q10。
    edge_q10_floor: Optional[float] = -0.20
    # 账户仓位比例:权益复利用 (1 + f * option_roi)。
    # 0DTE 单笔 ROI 波动大,f=1 远超 Kelly、波动拖累会把正期望复利打成大亏。
    # 默认 1.0 保持旧口径;QQQ 路径用 ~0.25(半 Kelly)。
    position_frac: float = 1.0
    # --- 滚动分位阈值(None=关闭) ---
    # 入场阈值 = max(静态调度阈值, 近 window 根入场窗 bar 的 edge 分位数)。
    # 动机:模型打分分布随月份漂移(2026-04→06 过阈 bar 607→2113 而均值
    # +0.15→+0.04),固定绝对阈值的选择性失控;分位数把"只做头部 x% 机会"
    # 钉住。取 max = 只收紧不放松,绝对阈值仍然是覆盖成本的地板。
    entry_quantile: Optional[float] = None
    entry_quantile_window: int = 1500   # ~5 个交易日的入场窗 bar
    entry_quantile_min_obs: int = 300   # 观测不足时退回静态阈值
    # True: PUT 腿也用 put_edge 分位抬高门槛。False:仅 CALL 用分位(PUT 仍用静态阈值)。
    # Jul W1 对拍:put_dyn 会把 13:10 大 PUT(edge~0.044)挡在 q80~0.06 外,丢掉 +38% 路径关键腿。
    apply_put_entry_quantile: bool = True
    # 默认 False:仅 CALL 吃 edge_q10_floor(历史 Jul W1 PUT 赢家常带差 q10)。
    # True:PUT 共用同一 floor(OPEN_DEFENSE profile / 高波动防御)。
    apply_put_edge_q10: bool = False
    # --- PUT 腿行情开关(None=不门控) ---
    # PUT 只在恐慌/高波动 regime 有正期望:三时期审计显示 vix_level(归一化)
    # 最高四分位贡献了 PUT 几乎全部利润,低 VIX 时 PUT 持续放血且挤占 CALL
    # 额度(2026Q1: 门槛 0.25 把 PUT 腿从 -31% 变 +99%)。
    # 语义:入场 bar 的 put_gate 信号值 >= put_gate_min 才允许开 PUT。
    # 信号值缺失(NaN)时视为不通过——宁可错过,不可误开。
    put_gate_min: Optional[float] = None
    # --- 早盘冲高回落 PUT 门控(与 put_gate OR 关系) ---
    # open30_max_ret >= morning_fade_min_ret 且 open30_peak_dd <= morning_fade_max_peak_dd
    # 时允许 PUT(典型倒 V)。仅在 session_bar <= morning_fade_session_end_bar 生效。
    morning_fade_min_ret: Optional[float] = None
    morning_fade_max_peak_dd: Optional[float] = None
    morning_fade_session_end_bar: Optional[int] = None
    # --- 急速下跌防护:近 N 分钟现货跌幅过深时禁止新开 CALL ---
    rapid_drop_ret: Optional[float] = None
    rapid_drop_bars: int = 5
    block_call_on_rapid_drop: bool = True
    # 实盘 immediate_entry:信号当根 bar 收盘即成交(不等 pending 下一 bar)
    immediate_entry: bool = False
    # --- PUT 趋势对齐门控(None=关闭) ---
    # 入场 bar 的 30min 拟合趋势收益 > put_trend_max_ret 时禁止新开 PUT。
    # 双时期审计(2026Q2 / 2025H2):趋势向上时买 PUT 系统性亏损
    # (Q2 逆势 PUT 合计 -0.18 vs 顺势 +2.54;H2 逆势单笔均值仅为顺势 1/3)。
    # 趋势值缺失(NaN)时不拦截——该门控是减法保护,数据缺失不应误杀顺势单。
    put_trend_max_ret: Optional[float] = None
    # --- CALL 震荡过滤(None=关闭) ---
    # 入场 bar 的 trend_fit_r2_30m < call_trend_r2_min 时禁止新开 CALL。
    # 6 月审计:低 r2 = 无方向震荡,CALL 头 IC 转负;门控后 Q2 6.94→5.98x /
    # 6 月 1.64→1.95x,连亏 5→3 天。r2 缺失时不拦截。
    call_trend_r2_min: Optional[float] = None
    # --- VIX 洗盘弃权(V0 REGIME_REVERSAL_THRESHOLD 对齐) ---
    # vix_proxy_close 30min 内方向反转次数 > 此值时禁止一切新开仓。
    # 6/10–11 审计:拦 2/4 连亏笔且 6 月账户 +44.6%→+60.9%;计数缺失时不拦截。
    regime_vix_reversal_max: Optional[int] = None
    regime_vix_reversal_window: int = 30
    regime_vix_reversal_pct: float = 0.0015
    # --- PUT 尾盘禁开(None=关闭) ---
    # session_bar > put_late_session_bar 时禁止新开 PUT。6 月规则扫描:拦 3/16 亏单、
    # 1/28 赢单,6 月 +9.5% / Q2 +43%。
    put_late_session_bar: Optional[int] = None
    # --- CALL 追涨洗盘禁开(None=关闭) ---
    # spot_day_ret > call_chase_spot_day_ret_min 且 vix_rev >= call_chase_vix_rev_min
    # 时禁止新开 CALL。6/11 型亏损:日涨+高洗盘仍追 CALL → EARLY_STOP。
    call_chase_vix_rev_min: Optional[int] = None
    call_chase_spot_day_ret_min: float = 0.0
    # --- CALL 局部波动尖刺禁开(None=关闭) ---
    # spot_range_30m >= call_spike_range30_min 时禁 CALL。6/15:低波日局部尖刺 2.77% → EARLY_STOP。
    call_spike_range30_min: Optional[float] = None
    # --- PUT 日涨禁开(None=关闭) ---
    # spot_day_ret > put_spot_day_ret_min 时禁 PUT。6/29:尾盘现货日涨 1.27% 仍开 PUT → EARLY_STOP。
    put_spot_day_ret_min: Optional[float] = None
    # --- CALL 早盘追涨时点门控(6/11 型,None=关闭) ---
    # spot_day_ret > call_timing_spot_min 且 session_bar < call_timing_max_bar
    # 且 vix_rev >= call_timing_vix_min 时禁 CALL。6/11:日涨+洗盘+sb=42/160
    # 而 Oracle 最优 sb=227–230;因果门控拦截早盘追 CALL。
    call_timing_spot_min: Optional[float] = None
    call_timing_max_bar: Optional[int] = None
    call_timing_vix_min: Optional[int] = None
    # --- 早盘 PUT 加强门控(July1 HARD_STOP 型,None=关闭) ---
    # session_bar < put_early_session_bar 时额外约束(不影响午盘/尾盘 PUT):
    #   1) put_early_vix_min: 早盘要求更高 vix_level(或 morning_fade),防 VIX 已回落的假恐慌;
    #   2) put_early_open30_max_min: 要求 open30 曾翻红(结构),挡「阴跌无结构」开盘;
    #   3) put_early_range30_min: 要求 30m 波动结构,挡空洞盘口日。
    # W1 金标网格: vix≥0.6 + open30_max>0 @ sb<30 可去掉 July1 -28.8% 且保留 July7 +69%。
    put_early_session_bar: Optional[int] = None
    put_early_vix_min: Optional[float] = None
    # 早盘中段 vix 禁 PUT: lo <= vix < hi 且无 morning_fade 时拒(July1≈0.84 假恐慌)。
    # 与抬高 put_early_vix_min 不同:不挡高 vix 早盘(Jul8≈1.06),也不挡低 vix+fade(Jul7)。
    put_early_vix_ban_lo: Optional[float] = None
    put_early_vix_ban_hi: Optional[float] = None
    put_early_open30_max_min: Optional[float] = None
    put_early_range30_min: Optional[float] = None
    # 早盘低置信 PUT 的 QQQ/VIXY 15m 反向确认：
    # session_bar < end 且 put-call gap < max 时，若 QQQ 15m>=0 且 VIXY 15m<=0，
    # 说明现货不跌、波动率不升，拒绝 PUT。任一收益缺失时不拦截。
    put_early_cross_confirm_end_bar: Optional[int] = None
    put_early_cross_confirm_edge_gap_max: Optional[float] = None
    # --- 因果盘中 VIXY regime：开盘急跌后低 R² 震荡 ---
    # detect 窗内一旦满足 open30 跌幅/回撤 + 低趋势 R²，状态保持到日切。
    # 状态激活后先禁早盘 PUT；随后仅对低 gap PUT 要求 QQQ 下跌、VIXY 上涨
    # 和负向趋势确认。所有输入均来自已完成分钟。
    vixy_open_shock_regime_enabled: bool = False
    vixy_open_shock_detect_start_bar: int = 30
    vixy_open_shock_detect_end_bar: int = 45
    vixy_open_shock_open30_ret_max: float = 0.0
    vixy_open_shock_peak_dd_max: float = -0.003
    vixy_open_shock_detect_r2_max: float = 0.10
    vixy_open_shock_put_block_end_bar: int = 60
    vixy_open_shock_min_dual_leg_edge_gap: float = 0.001
    vixy_open_shock_low_conf_gap_max: float = 0.005
    vixy_open_shock_spot_ret_15_max: float = -0.0005
    vixy_open_shock_vix_ret_15_min: float = 0.0
    vixy_open_shock_confirm_r2_min: float = 0.15
    # --- CALL TREND_SPENT 禁开(None=关闭) ---
    # 日振幅已走到高位 + 波动压缩 + 午后时点 → 禁 CALL(不碰 PUT)。
    # July10 型:上午慢爬后追涨 CALL 系统性亏;W1+7/10 消融推荐
    #   day_range_pos>=0.85 & bb_width<=0 & session_bar>=210。
    # 字段缺失时不拦截(减法保护)。
    call_spent_day_range_pos_min: Optional[float] = None
    call_spent_bb_width_max: Optional[float] = None
    call_spent_min_session_bar: Optional[int] = None
    # --- 方向头一致性门控(Jul1 09:46 PUT 型;字段缺失时不拦截) ---
    # best_side_none_prob 严格高于 call/put 时整 bar 弃权(模型显式偏好 NONE)。
    block_when_side_none: bool = False
    # PUT 要求 put_prob>call_prob; CALL 要求 call_prob>put_prob。
    require_leg_side_agree: bool = False
    # PUT 要求 spot_down>spot_up; CALL 要求 spot_up>spot_down。
    require_leg_spot_agree: bool = False
    # 双腿 edge 绝对差 < 该值时整 bar 弃权(Jul13 型硬币面;None=关闭)。
    # Apr–Jun 正常 gap 中位~0.05;Jul13 病态日中位~0.0008。
    min_dual_leg_edge_gap: Optional[float] = None
    # CALL 要求 spot_day_ret > eps; PUT 要求 spot_day_ret < -eps(字段缺失不拦)。
    # 比模型 spot_up/down 概率更稳:用当日已实现现货方向确认选腿。
    require_leg_spot_day_agree: bool = False
    spot_day_agree_eps: float = 0.0

    def threshold_at(self, session_bar: Optional[int]) -> float:
        if self.entry_threshold_schedule is None or session_bar is None:
            return self.entry_threshold
        th = self.entry_threshold
        for start, value in self.entry_threshold_schedule:
            if session_bar >= start:
                th = value
            else:
                break
        return th

    def session_allows_entry(self, session_bar: Optional[int]) -> bool:
        """session_bar 在 [start, end] 内才允许新开仓(None 边界=不限制)。"""
        if session_bar is None:
            return True
        if self.session_entry_start_bar is not None and session_bar < self.session_entry_start_bar:
            return False
        if self.session_entry_end_bar is not None and session_bar > self.session_entry_end_bar:
            return False
        return True


@dataclass
class Trade:
    entry_ts: object
    exit_ts: object
    entry_price: float
    exit_price: float
    net_return: float
    exit_reason: str
    bars_held: int
    signal_edge: float
    leg: str = "CALL"
    # 该笔实际账户仓位；允许跨日防御在不改变权利金 ROI 的情况下动态减仓。
    position_frac: float = 1.0


@dataclass
class ReplayResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    position_frac: float = 1.0
    events: List[Any] = field(default_factory=list)

    def summary(self, position_frac: Optional[float] = None) -> dict:
        f = float(self.position_frac if position_frac is None else position_frac)
        if not self.trades:
            return {"trades": 0, "total_net_return": 0.0, "position_frac": f}
        rets = np.array([t.net_return for t in self.trades])
        # equity_curve 已按 f 复利;无曲线时回退到 f * ROI
        if self.equity_curve:
            eq = np.array(self.equity_curve, dtype=float)
        else:
            eq = np.cumprod(1.0 + f * rets)
        peak = np.maximum.accumulate(eq)
        mdd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        reasons = pd.Series([t.exit_reason for t in self.trades]).value_counts().to_dict()
        legs = pd.Series([t.leg for t in self.trades]).value_counts().to_dict()
        return {
            "trades": int(len(rets)),
            "position_frac": f,
            # 账户复利(含仓位比例)
            "total_net_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
            # 权利金 ROI 口径(信号质量,不受 f 影响)
            "avg_net_return": float(rets.mean()),
            "sum_net_return": float(rets.sum()),
            "full_size_compound": float(np.prod(1.0 + rets) - 1.0),
            "hit_rate": float((rets > 0).mean()),
            "profit_factor": profit_factor,
            "max_drawdown_mtm": mdd,
            "avg_bars_held": float(np.mean([t.bars_held for t in self.trades])),
            "worst_trade": float(rets.min()),
            "exit_reasons": reasons,
            "trades_by_leg": legs,
        }

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entry_ts": t.entry_ts,
                    "exit_ts": t.exit_ts,
                    "leg": t.leg,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "net_return": t.net_return,
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                    "signal_edge": t.signal_edge,
                    "position_frac": t.position_frac,
                }
                for t in self.trades
            ]
        )
