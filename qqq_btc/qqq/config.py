#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QQQ 0DTE 路径的全部运行参数 —— 标签、回放、实盘共用一份配置对象。

上一代的参数散落在 config.py / strategy_config0.py / slow_feature.json /
环境变量四处且互相矛盾;本路径收敛为一个模块,fill 假设只在这里出现一次。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.labels import LabelHorizon
from qqq_btc.common.replay_types import ReplayConfig

SYMBOL = "QQQ"

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

# 标签 horizon:60s 延迟入场 + 300s 持有(与 New_Pro option_exec_label 对齐)
LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=5,
    flat_margin=0.0005,
)

# 交易 bucket:2 = 0DTE CALL ATM(与 anchor_qqq_0dte.json label_trade_bucket_id 一致)
TRADE_BUCKET_ID = 2

# ---------------------------------------------------------------------------
# 回放 / 策略入场
# ---------------------------------------------------------------------------
REPLAY = ReplayConfig(
    entry_threshold=0.015,     # |net_edge| >= 1.5% 才入场(绝对阈值,无截面)
    entry_delay_bars=1,        # 与标签的 60s 延迟一致
    max_spread_pct=0.06,       # 0DTE ATM 常态点差 1-3%,>6% 视为执行环境恶化
    cooldown_bars=5,
    long_only=True,            # 先做 CALL 多头;PUT 腿(双腿标签模型)验证后开启
    # 0DTE 分时段阈值:下午 theta 燃烧加速,同样 edge 越晚越不值得进。
    # 初值为手拍量级,待 strict replay 按时段收益分布重新标定。
    entry_threshold_schedule=(
        (0, 0.015),      # 09:30-14:00
        (270, 0.020),    # 14:00 之后抬高 1/3
        (330, 0.025),    # 15:00 之后再抬
    ),
    # --- 频率治理:单标的没有截面分散,用日内风险预算代替 ---
    # 初值为手拍量级,待 strict replay 标定。频率不是目标,是规则与市场交互的结果。
    max_trades_per_day=6,          # cooldown=5 + 单持仓下的自然上限附近
    daily_loss_stop=-0.20,         # 当日累计净收益(权利金 ROI 之和)≤ -20% 停止开新仓
    loss_streak_n=3,               # 连亏 3 笔
    loss_streak_cooldown_bars=60,  # → 冷却 1 小时
    # --- 跨式(双买波动) ---
    # 双份权利金 + 双份 theta,入场门槛 = 单腿基础阈值 2 倍;
    # 跨式是低频武器(事件日/挤压日),日内最多 2 次,防止在盘整日反复买波动
    straddle_entry_threshold=0.030,
    max_straddles_per_day=2,
    session_entry_start_bar=0,     # 09:30 起可新开仓(短序列左侧补零,与 infer 一致)
    session_entry_end_bar=360,     # 15:30 后禁新开仓
)

# 分布头门控:回放/实盘要求 net_edge_q10 > 0 才允许 CALL 入场(tft_qqq_v2 输出)
EDGE_Q10_COL = "net_edge_q10"

# 双腿方向决策(模型 call/put 双头输出列;replay 中两腿各自过阈值取较强者)。
# 前提:LMDB 用 build_dual_leg_net_labels 重建、loss_weights.call_put_edge>0 训练过。
# 开启方式:long_only=False + run_strict_replay(call_edge_col=..., put_edge_col=...)
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
# 跨式头(有符号:预测"同时买两腿"的净收益,大多数日子为负 = 双份 theta)
STRADDLE_EDGE_COL = "straddle_net_edge"

# v2 特征配置(含日内时间/趋势特征)与模型(底座已内化,不依赖 New_Pro)
FEATURE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "CONFIG" / "slow_feature_qqq_v2.json"
MODEL_MODULE = "qqq_btc.model.backbone"

# 退出轨道(bar = 1 分钟)
EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.12,
    soft_stop_roi=-0.08,
    early_stop_bars=5,           # 与 LABEL_HORIZON.hold_bars 对齐;calibrate_rails 可重标
    early_stop_roi=-0.05,
    time_stop_bars=15,
    time_stop_min_roi=0.05,
    max_hold_bars=30,
    trailing_trigger_roi=0.30,     # 大波段才启用 trailing,与密 ladder 分工
    trailing_keep_ratio=0.65,
    ladder=(
        (0.08, 0.05),
        (0.12, 0.08),
        (0.18, 0.12),
        (0.25, 0.18),
        (0.35, 0.26),
    ),
    flash_trigger_roi=0.08,
    flash_exit_roi=0.03,
    eod_close_bar_index=380,   # 09:30 起第 380 分钟 = 15:50 强平
    # tick 级风险/浮盈轨(check_tick_stops):不污染分钟 max_roi。
    # fast_hard:略深于分钟 hard(-12%);disaster 只兜闪崩。
    # tick_profit:独立 tick_peak,peak≥20% 后回落到 peak*50% 平仓
    # (吃「一分钟内翻倍再吐光」;比分钟 trailing 触发更早、锁利更紧)。
    tick_fast_hard_roi=-0.15,
    tick_fast_hard_smooth_n=3,
    disaster_stop_roi=-0.25,
    disaster_smooth_n=3,
    tick_profit_trigger_roi=0.20,
    tick_profit_keep_ratio=0.50,
    tick_profit_smooth_n=3,
    tick_profit_ladder=(
        (0.15, 0.08),
        (0.30, 0.18),
        (0.50, 0.30),
        (0.80, 0.50),
    ),
)

# 交易时段(分钟序号,自 09:30 开盘起算) —— 与 REPLAY.session_entry_* 同步
SESSION_ENTRY_START_BAR = 0     # 09:30; 开盘点差由 max_spread_pct 门控
SESSION_ENTRY_END_BAR = 360     # 15:30 后禁新开仓
SESSION_FORCE_CLOSE_BAR = 380   # 15:50 强平

# 仓位
MAX_POSITIONS = 1
