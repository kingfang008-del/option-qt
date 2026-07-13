#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QQQ **1DTE 族**（standard_old_v2 / 生产主路径）运行参数。

历史上 docstring 写过「0DTE」,但锁约与特征实际是 trading≈1DTE;
EXIT_RAILS / REPLAY 也按 1DTE 权利金路径标定(hard=-0.25, vol_ref=0.048)。

真正 trading 0DTE 请用 ``qqq_btc.qqq.config_true_0dte``,不要与本模块共用规则。

标签、回放、实盘(1DTE 族)共用本文件;fill 假设只在这里出现一次。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.labels import LabelHorizon
from qqq_btc.common.replay_types import ReplayConfig

SYMBOL = "QQQ"
PROFILE = "1dte_family"  # 与 config_true_0dte.PROFILE 区分;勿混用 EXIT_RAILS/REPLAY

# ---------------------------------------------------------------------------
# 成交模型:全路径唯一的 fill 假设(标签 = 回放 = 实盘审计基准)
# 0.775 = 实盘实测成交 0.75-0.8 点差位的中值。实盘采集到 fill_spread_frac
# 分布后,在这里回填校准,并全链路重跑标签。
# ---------------------------------------------------------------------------
FILL_MODEL = OptionSpreadFillModel(
    entry_frac=0.775,
    exit_frac=0.775,
    commission_per_contract=0.65,
    contract_multiplier=100.0,
)

# 标签 horizon:60s 延迟入场 + 30min 持有
# 与特征视野(trend_fit_30m / seq_len=30)对齐;原 hold=5(6min)对 0DTE 过短、噪声主导
LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=30,
    flat_margin=0.01,  # 30min 期权 ROI 下 1% 内视为盘整
)

# 交易 bucket:2 = 0DTE CALL ATM(与 anchor_qqq_0dte.json label_trade_bucket_id 一致)
TRADE_BUCKET_ID = 2

# ---------------------------------------------------------------------------
# 回放 / 策略入场
# ---------------------------------------------------------------------------
REPLAY = ReplayConfig(
    # 30min 标签下信号呈稀疏脉冲:提高阈值、少做,与 Top edge bar 验收口径一致
    entry_threshold=0.03,
    entry_delay_bars=1,        # 与标签的 60s 延迟一致
    max_spread_pct=0.06,       # 0DTE ATM 常态点差 1-3%,>6% 视为执行环境恶化
    cooldown_bars=10,
    # 双腿开启(2026-07 验证):PUT 腿受 vix_level 行情门控(见 put_gate_min),
    # 三时期回放 CALL单腿 vs 门控双腿:2025H2 +681%→+3138% /
    # 2026Q1 +10%→+100% / 2026Q2 +10%→+102%;fill 压力(0.90)下仍全正。
    long_only=False,
    entry_threshold_schedule=(
        (15, 0.03),      # 09:45 起可新开仓(open30 在 bar29 冻结,早盘用滚动形态)
        (270, 0.036),    # 14:00 之后抬高
        (330, 0.042),    # 15:00 之后(通常已禁开仓)
    ),
    # --- 频率治理:单标的没有截面分散,用日内风险预算代替 ---
    max_trades_per_day=4,
    daily_loss_stop=-0.20,         # 当日累计净收益(权利金 ROI 之和)≤ -20% 停止开新仓
    loss_streak_n=3,               # 连亏 3 笔
    loss_streak_cooldown_bars=60,  # → 冷却 1 小时
    # --- 跨式(双买波动) ---
    # 双份权利金 + 双份 theta,入场门槛 = 单腿基础阈值 2 倍;
    # 跨式是低频武器(事件日/挤压日),日内最多 2 次,防止在盘整日反复买波动
    straddle_entry_threshold=0.030,
    max_straddles_per_day=2,
    # 09:45 起可新开仓;open30 形态在 bar29(10:00)冻结,早盘 fade 门控用滚动值
    session_entry_start_bar=15,
    # 13:30 后禁新开仓:FT56 Jul W1 honest 优化段 +26.9%→+49.0%,
    # 7/9–10 时间外仍 +2.3%;同时减少尾盘 theta 暴露。
    session_entry_end_bar=240,
    # frozen_norm + 1m5m 栈下 CALL 偏好 bar 的 q10 中位约 -26%(rolling/eval 约 -21%)。
    # 原 floor=-0.20 在 rolling 上放行 ~38% call_pref;frozen 上仅 ~3%,CALL 被系统性关掉。
    # 按 2026-06 frozen_1m5m 重标定到 -0.25,使 call_pref 通过率回到 ~38%。
    edge_q10_floor=-0.25,
    # 滚动分位阈值:实际阈值 = max(静态调度, 近 1500 入场窗bar edge 的 p80)。
    # 动机:打分分布漂移(2026-04→06 过阈bar 607→2113 而均值 +0.15→+0.04),
    # 固定绝对阈值选择性失控。q=0.80 在 2025H2 与 2026Q2 双段验证:
    # 亏损月(2025-10/11, 2026-05)全部收窄或翻正,MDD 同步下降。
    entry_quantile=0.80,
    entry_quantile_window=1500,
    entry_quantile_min_obs=300,
    # CALL-only 分位:put_dyn 会系统性挡掉 edge 落在静态阈值之上、q80 之下的大 PUT
    # (Jul6 13:10 +38% 路径)。诚实流式/1m-gate 对拍默认关 PUT 分位。
    apply_put_entry_quantile=False,
    # PUT 腿行情开关:入场 bar 归一化 vix_level >= 0.25 才允许开 PUT。
    # 三时期 PUT 审计:vix_level 最高四分位贡献几乎全部 PUT 利润,低 VIX 时
    # PUT 持续放血(2026Q1 无门控 PUT 腿 -31%,门控后 +95%)。0.2/0.25/0.3
    # 三档门槛全时期均为正,对门槛不敏感;取居中的 0.25。
    put_gate_min=0.25,
    # 早盘 PUT 加强(July1 HARD_STOP 型):session_bar<30 要求更高 vix(或 morning_fade)。
    # W1 网格:拦 09:47 HARD_STOP(-28.8%),保留 July7 大赢家;@25% +35.8%→+51.1%。
    # apr–jun 中性(0pp);2025H2 略正(+3pp)。open30/range 同效但语义弱于 vix。
    put_early_session_bar=30,
    put_early_vix_min=0.6,
    # 早盘冲高回落 PUT:open30_max_ret>=0.4% 且 peak_dd<=-0.3% 时允许 PUT(与 vix 门控 OR)
    morning_fade_min_ret=0.004,
    morning_fade_max_peak_dd=-0.003,
    morning_fade_session_end_bar=60,   # 仅 09:45–10:30 适用 fade 路径
    # 近 5min 现货跌超 0.4% 禁新开 CALL/跨式,防早盘急转下追多
    rapid_drop_ret=-0.004,
    rapid_drop_bars=5,
    block_call_on_rapid_drop=True,
    # PUT 趋势对齐:30min 拟合趋势 > 0 时禁开 PUT(即使 vix/fade 门控通过)。
    # 双时期审计:逆势 PUT 系统性亏损(Q2 -0.18 vs 顺势 +2.54;H2 单笔均值 1/3),
    # 启用后 Q2 4.17x→4.42x / H2 94.7x→146.9x,6 月最长连亏 5 天→3 天
    put_trend_max_ret=0.0,
    # 低 trend r2 震荡市禁 CALL:CALL 头 IC 转负时减法保护。
    # 双时期验证 Q2 6.94→5.98x / 6 月 1.64→1.95x,连亏 5→3 天。
    call_trend_r2_min=0.15,
    # R1 网格: spot_day_ret>0.5% 且 vix_rev>=5 禁 CALL → Q2 565→572% / 6月 86→88%
    call_chase_vix_rev_min=5,
    call_chase_spot_day_ret_min=0.005,
    # R2 网格(strict replay 150组):
    #   put_spot_day_ret>0.8% 禁 PUT → 拦 6/29 尾盘逆势 PUT
    #   spot_range_30m>=2.0% 禁 CALL → 6/15 局部尖刺 -4.5%→-1.4%
    # 合并后 Q2 572→698% / 6月 88→97%(f=0.25,V4 eval)
    put_spot_day_ret_min=0.008,
    call_spike_range30_min=0.020,
    put_late_session_bar=None,
    # 6/11 型早盘追涨时点门控(默认关闭;strict replay 网格见下):
    #   spot>0.3% & sb<200 & vix_rev>=5 → 6/11 -6.5%→+1.9%,但 Q2 -64pp / 6月 -8pp
    #   spot>0.4% 无法拦住 6/11(第二笔 sb=160 刚好在边界)
    call_timing_spot_min=None,
    call_timing_max_bar=None,
    call_timing_vix_min=None,
    # CALL TREND_SPENT:高位+波动压缩+午后禁追涨 CALL。
    # W1+7/10 消融:挡 7/10 两笔与 7/9 小亏 CALL;W1 +51.1%→+51.5%,整段 +48.2%→+51.5%。
    call_spent_day_range_pos_min=0.85,
    call_spent_bb_width_max=0.0,
    call_spent_min_session_bar=210,
    # 半 Kelly(~0.45 的一半):单笔权利金 ROI ±30% 时禁止全仓复利
    position_frac=0.25,
)

# 实盘入场:bar 收盘决策后立即下单(不 pending 下一根 bar,不 OMS 延迟队列)。
# strict replay 仍用 REPLAY.entry_delay_bars=1 与 60s 标签对齐;live 刻意不等满 1min。
LIVE_REPLAY = replace(REPLAY, entry_delay_bars=0, immediate_entry=True)

# 分布头门控列;阈值见 ReplayConfig.edge_q10_floor(不再要求 q10>0)
EDGE_Q10_COL = "net_edge_q10"

# 双腿方向决策(模型 call/put 双头输出列;replay 中两腿各自过阈值取较强者)。
# 前提:LMDB 用 build_dual_leg_net_labels 重建、loss_weights.call_put_edge>0 训练过。
# 开启方式:long_only=False + run_strict_replay(call_edge_col=..., put_edge_col=...)
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
# PUT 腿行情开关信号列(归一化 VIX 代理;live 由特征管线实时产出)
PUT_GATE_COL = "vix_level"
# 跨式头(有符号:预测"同时买两腿"的净收益,大多数日子为负 = 双份 theta)
STRADDLE_EDGE_COL = "straddle_net_edge"

# 与 v4 checkpoint 内嵌特征表对齐(42 列);v2 多 spot_range_30m/trend_strength_30m
FEATURE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "CONFIG" / "slow_feature_qqq_v4.json"
MODEL_MODULE = "qqq_btc.model.backbone"

# 退出轨道(bar = 1 分钟) —— val 上 calibrate_rails(max_hold=45, hold=30) 重标
# 见 /tmp/rails_h30_suggestion.json;孵化期只跑 hard,利润保护延后
EXIT_RAILS = ExitRailsConfig(
    # Jul W1 离线: -0.28→-0.25 收窄开盘毒 PUT 单笔冲击,周 acct@25% 38.3%→40.5%
    hard_stop_roi=-0.25,
    soft_stop_roi=-0.20,         # 赢单 MAE q05 中位;孵化期内也生效(见 exit_rails.check_exit)
    profit_protect_min_bars=15,  # 前 15 bar 孵化:推迟 ladder/trailing/flash,不推迟 soft/hard
    early_stop_bars=15,
    early_stop_roi=-0.12,
    time_stop_bars=30,
    time_stop_min_roi=0.03,
    # 55 是无条件硬上限而非最低持有期；soft/early/time/trailing/step 均可提前退出。
    # FT56 Jul W1 honest:优化段 +49.0%,7/9–10 时间外 +2.3%(f=0.25)。
    max_hold_bars=55,
    trailing_trigger_roi=0.57,   # 大波段才 trailing
    trailing_keep_ratio=0.65,
    ladder=(
        (0.20, 0.12),
        (0.34, 0.22),
        (0.57, 0.40),
    ),
    flash_trigger_roi=0.20,
    flash_exit_roi=0.08,
    eod_close_bar_index=380,   # 09:30 起第 380 分钟 = 15:50 强平
    # tick 级:分钟轨放宽后,闪崩保护仍紧(不污染分钟 max_roi)
    tick_fast_hard_roi=-0.20,
    tick_fast_hard_smooth_n=3,
    disaster_stop_roi=-0.35,
    disaster_smooth_n=3,
    tick_profit_trigger_roi=0.25,
    tick_profit_keep_ratio=0.50,
    tick_profit_smooth_n=3,
    tick_profit_ladder=(
        (0.25, 0.12),
        (0.40, 0.22),
        (0.60, 0.35),
        (1.00, 0.55),
    ),
    # 波动自适应:利润保护阈值按「当日近 60 bar 权利金分钟波动 / 历史参考」缩放。
    # 0.048 = 2025-07~2026-03 日度分钟收益 std 的中位数(189 个交易日)。
    # 动机:2026-06 尾部 |30min ROI| p99 从 2.0 拉到 10.4,静态阈值把右尾剪掉、
    # 左尾留下(同入场点拿满 +17% vs 护栏 -13.6%)。
    # profit_only:2026-04~06 变体回测显示放深止损只加大左尾,
    # 收益来自让 ladder/trailing 档位随波动上移(六月 -13.6% → +12.7%)。
    vol_scale_ref=0.048,
    vol_scale_window=60,
    vol_scale_min_obs=20,
    vol_scale_min=1.0,
    vol_scale_max=2.5,
    vol_scale_profit_only=True,
)

# 交易时段(分钟序号,自 09:30 开盘起算) —— 与 REPLAY.session_entry_* 同步
SESSION_ENTRY_START_BAR = 15    # 09:45;open30 在 bar29 冻结
SESSION_ENTRY_END_BAR = 330     # 15:00 后禁新开仓(hold=30min)
SESSION_FORCE_CLOSE_BAR = 380   # 15:50 强平

# 仓位:单标的同时最多 1 仓;账户下注比例见 REPLAY.position_frac
MAX_POSITIONS = 1
