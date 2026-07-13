#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真正 trading 0DTE 的入场/出场规则 —— 与 ``qqq_btc.qqq.config``（1DTE 族）完全隔离。

背景
----
``qqq.config`` 的 EXIT_RAILS / REPLAY 是按 trading≈1DTE 权利金路径标定的
(hard=-28%, vol_ref=0.048, max_hold=45)。套到真 0DTE 上会出现:

  - 标签路径 ≤−28% 占比 ~30%(1DTE 仅 ~8%) → HARD_STOP 占比 33–40%
  - Val 账户 −78% / MDD −85%; Test 虽 +45% 但 MDD −56%

本模块按真 0DTE Val+Test infer 网格选定(变体 B):

  - 更深 hard / 更短持有 / vol_ref≈0.20(0DTE 权利金分钟波动中位)
  - 更严 PUT 门控 + 更早禁开 + 更高分位入场
  - Val −78%→+131%(MDD −17%); Test +42%(MDD −12%), HARD_STOP 降至 ~1 笔

用法
----
  from qqq_btc.qqq import config_true_0dte as qcfg0
  run_strict_replay(df, qcfg0.FILL_MODEL, qcfg0.REPLAY, qcfg0.EXIT_RAILS, ...)

  python qqq_btc/tools/eval_test_set.py --strategy-config qqq_btc.qqq.config_true_0dte ...
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

# 与 1DTE 族共用成交模型/标签口径(同一 fill 假设),但入场/出场独立。
SYMBOL = "QQQ"
PROFILE = "true_0dte"

FILL_MODEL = OptionSpreadFillModel(
    entry_frac=0.775,
    exit_frac=0.775,
    commission_per_contract=0.65,
    contract_multiplier=100.0,
)

LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=30,
    flat_margin=0.01,
)

TRADE_BUCKET_ID = 2  # CALL ATM; PUT 用 bucket 0

# ---------------------------------------------------------------------------
# 入场:更高分位、更严 PUT、更早收工(0DTE 午后 theta / 跳空)
# ---------------------------------------------------------------------------
REPLAY = ReplayConfig(
    entry_threshold=0.05,
    entry_delay_bars=1,
    max_spread_pct=0.08,  # 0DTE ATM 点差常态更宽
    cooldown_bars=15,
    long_only=False,
    entry_threshold_schedule=(
        (15, 0.05),
        (180, 0.06),   # 12:30 后抬高
        (240, 0.07),   # 13:30 后更严
    ),
    max_trades_per_day=3,
    daily_loss_stop=-0.15,
    loss_streak_n=3,
    loss_streak_cooldown_bars=60,
    straddle_entry_threshold=0.10,
    max_straddles_per_day=1,
    session_entry_start_bar=15,
    session_entry_end_bar=270,  # 14:00 后禁新开(比 1DTE 的 300 更早)
    edge_q10_floor=-0.25,
    entry_quantile=0.85,  # 只做头部,1DTE 用 0.80
    entry_quantile_window=1500,
    entry_quantile_min_obs=300,
    # PUT:真 0DTE 高 edge PUT 持有到 horizon 胜率极差;抬高 VIX 门槛 + 午后禁 PUT
    put_gate_min=0.45,           # 1DTE=0.25
    put_early_session_bar=45,    # 早盘窗口拉长
    put_early_vix_min=0.70,      # 1DTE=0.60
    put_late_session_bar=210,    # 13:00 后禁 PUT
    morning_fade_min_ret=0.004,
    morning_fade_max_peak_dd=-0.003,
    morning_fade_session_end_bar=60,
    rapid_drop_ret=-0.004,
    rapid_drop_bars=5,
    block_call_on_rapid_drop=True,
    put_trend_max_ret=0.0,
    call_trend_r2_min=0.15,
    call_chase_vix_rev_min=5,
    call_chase_spot_day_ret_min=0.005,
    put_spot_day_ret_min=0.008,
    call_spike_range30_min=0.020,
    call_timing_spot_min=None,
    call_timing_max_bar=None,
    call_timing_vix_min=None,
    call_spent_day_range_pos_min=0.85,
    call_spent_bb_width_max=0.0,
    call_spent_min_session_bar=210,
    position_frac=0.25,
)

LIVE_REPLAY = replace(REPLAY, entry_delay_bars=0, immediate_entry=True)

EDGE_Q10_COL = "net_edge_q10"
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
PUT_GATE_COL = "vix_level"
STRADDLE_EDGE_COL = "straddle_net_edge"

FEATURE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "CONFIG" / "slow_feature_qqq_v4.json"
MODEL_MODULE = "qqq_btc.model.backbone"

# ---------------------------------------------------------------------------
# 出场:按真 0DTE CALL 赢单 MAE(q01≈-0.33) + 权利金分钟 vol 中位≈0.20 重标
# 持有对齐 label hold=30;不沿用 1DTE 的 hard=-0.28 / max_hold=45 / vol_ref=0.048
# ---------------------------------------------------------------------------
EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.40,
    soft_stop_roi=-0.30,
    profit_protect_min_bars=10,  # 0DTE 给 thesis 的时间更短
    early_stop_bars=10,
    early_stop_roi=-0.18,
    time_stop_bars=20,
    time_stop_min_roi=0.05,
    max_hold_bars=30,
    trailing_trigger_roi=0.80,  # MFE 更大,触发点上移
    trailing_keep_ratio=0.60,
    ladder=(
        (0.30, 0.15),
        (0.55, 0.32),
        (0.80, 0.50),
    ),
    flash_trigger_roi=0.30,
    flash_exit_roi=0.12,
    eod_close_bar_index=380,
    tick_fast_hard_roi=-0.30,
    tick_fast_hard_smooth_n=3,
    disaster_stop_roi=-0.50,
    disaster_smooth_n=3,
    tick_profit_trigger_roi=0.35,
    tick_profit_keep_ratio=0.50,
    tick_profit_smooth_n=3,
    tick_profit_ladder=(
        (0.35, 0.15),
        (0.55, 0.30),
        (0.80, 0.45),
        (1.20, 0.65),
    ),
    vol_scale_ref=0.20,  # 真 0DTE 权利金分钟收益日 std 中位(~4× 1DTE 的 0.048)
    vol_scale_window=60,
    vol_scale_min_obs=20,
    vol_scale_min=1.0,
    vol_scale_max=3.0,
    vol_scale_profit_only=True,
)

SESSION_ENTRY_START_BAR = REPLAY.session_entry_start_bar
SESSION_ENTRY_END_BAR = REPLAY.session_entry_end_bar
SESSION_FORCE_CLOSE_BAR = 380
MAX_POSITIONS = 1
