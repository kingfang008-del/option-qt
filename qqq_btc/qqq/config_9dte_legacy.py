#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Legacy ~9DTE 专用运行参数 —— 与 qqq.config (0DTE V4) 完全隔离。

9DTE 期权 gamma 更低、预测/标签量级更小(正股 forward return ~0.001),
不能用 0DTE 的 entry_threshold=0.03 与紧止损轨道。

用法:
  from qqq_btc.qqq import config_9dte_legacy as cfg9
  run_strict_replay(df, cfg9.FILL_MODEL, cfg9.REPLAY, cfg9.EXIT_RAILS, ...)
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
PROFILE = "legacy_9dte"

# 与 0DTE 相同 fill 假设;9DTE 点差通常更窄,但先保持一致便于对比
FILL_MODEL = OptionSpreadFillModel(
    entry_frac=0.775,
    exit_frac=0.775,
    commission_per_contract=0.65,
    contract_multiplier=100.0,
)

# 训练仍用 feature_merge 正股 forward 标签;flat_margin 与标签量级对齐
LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=30,
    flat_margin=0.0005,
)

# front month CALL ATM (legacy 6-bucket 中 bucket 2)
TRADE_BUCKET_ID = 2
PUT_BUCKET_ID = 0

# 数据源路径(legacy 干净口径)
RAW_1S_ROOT = Path("/mnt/s990/data/raw_1s/options")
RAW_1M_ROOT = Path("/mnt/s990/data/raw_1m/options")
FEATURE_VAL_ROOT = Path.home() / "train_data/quote_features_val_9dte_legacy"
FEATURE_TEST_ROOT = Path.home() / "train_data/quote_features_test_9dte_legacy"
CHECKPOINT_DEFAULT = Path(__file__).resolve().parent.parent.parent / "checkpoints_qqq_9dte_janval_febtest/best.pth"
# 期权 fill 标签重训 (label_pipeline);预测 net_edge 整体偏负,阈值需 <=0
CHECKPOINT_FILL = Path(__file__).resolve().parent.parent.parent / "checkpoints_qqq_9dte_fill_janval/best.pth"
ANCHOR_CONFIG = Path(__file__).resolve().parent.parent / "CONFIG/anchor_qqq_9dte_legacy.json"
FEATURE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "CONFIG/slow_feature_qqq_v2.json"

# 回放 —— stock-return 模型用 0.0008+q85 (optimize_9dte_replay step1/2)
REPLAY_STOCK = ReplayConfig(
    entry_threshold=0.0008,
    entry_delay_bars=1,
    max_spread_pct=0.08,
    cooldown_bars=10,
    long_only=False,
    entry_threshold_schedule=((15, 0.0008), (270, 0.0012), (330, 0.0016)),
    straddle_entry_threshold=0.002,
    max_straddles_per_day=2,
    session_entry_start_bar=15,
    session_entry_end_bar=330,
    edge_q10_floor=-0.05,
    entry_quantile=0.85,
    entry_quantile_window=1500,
    entry_quantile_min_obs=300,
    put_gate_min=0.25,
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
    position_frac=0.25,
    max_trades_per_day=4,
    daily_loss_stop=-0.15,
    loss_streak_n=3,
    loss_streak_cooldown_bars=60,
)

# fill 标签模型: net_edge 整体偏负, 仅用 val 选参 th=0.001 静态阈值 (无 quantile)
REPLAY_FILL = replace(
    REPLAY_STOCK,
    entry_threshold=0.001,
    entry_quantile=None,
    entry_threshold_schedule=((15, 0.001), (270, 0.0015), (330, 0.002)),
)

# 备选: q85 分位门控 (val +16%, test +69% — test 扫参有过拟合风险)
REPLAY_FILL_Q85 = replace(
    REPLAY_STOCK,
    entry_threshold=0.0,
    entry_quantile=0.85,
    entry_threshold_schedule=((15, 0.0), (270, -0.005), (330, -0.01)),
)

REPLAY = REPLAY_FILL

LIVE_REPLAY = replace(REPLAY, entry_delay_bars=0, immediate_entry=True)

EDGE_Q10_COL = "net_edge_q10"
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
PUT_GATE_COL = "vix_level"
STRADDLE_EDGE_COL = "straddle_net_edge"

# 退出轨道 —— 2026-01 val MAE/MFE 标定 (optimize_9dte_replay step2)
EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.13,
    soft_stop_roi=-0.10,
    profit_protect_min_bars=15,
    early_stop_bars=15,
    early_stop_roi=-0.0399,
    time_stop_bars=30,
    time_stop_min_roi=0.03,
    max_hold_bars=45,
    trailing_trigger_roi=0.25,
    trailing_keep_ratio=0.65,
    ladder=(
        (0.0694, 0.0416),
        (0.1154, 0.075),
        (0.1837, 0.1286),
    ),
    flash_trigger_roi=0.0694,
    flash_exit_roi=0.0278,
    eod_close_bar_index=380,
    tick_fast_hard_roi=-0.12,
    tick_fast_hard_smooth_n=3,
    disaster_stop_roi=-0.22,
    disaster_smooth_n=3,
    tick_profit_trigger_roi=0.18,
    tick_profit_keep_ratio=0.50,
    tick_profit_smooth_n=3,
    tick_profit_ladder=(
        (0.18, 0.08),
        (0.30, 0.15),
        (0.45, 0.25),
        (0.70, 0.40),
    ),
    vol_scale_ref=0.025,
    vol_scale_window=60,
    vol_scale_min_obs=20,
    vol_scale_min=1.0,
    vol_scale_max=2.0,
    vol_scale_profit_only=True,
)

SESSION_ENTRY_START_BAR = 15
SESSION_ENTRY_END_BAR = 330
SESSION_FORCE_CLOSE_BAR = 380
MAX_POSITIONS = 1
