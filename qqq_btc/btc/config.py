#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BTC 永续路径参数骨架。

成本结构与期权完全不同(taker 费率 + 滑点 + funding,无点差插值),
但标签构建与 strict replay 复用 common 层同一套代码,输出列命名一致,
TFT 训练侧无需分叉。

当前缺口(见 ARCHITECTURE.md P2):
  - 数据源:永续 1m K 线 + funding 历史尚未接入
  - 特征:New_Pro 的 realtime_feature_engine_btc.py 仍是 equity fallback
    骨架,BTC 专属特征(funding/OI/liquidation/basis)未实现
  - 美股期权特征与 alpha 不可迁移,信号需独立重新验证
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import PerpFillModel
from qqq_btc.common.labels import LabelHorizon
from qqq_btc.common.replay_types import ReplayConfig

SYMBOL = "BTC-PERP"

# taker 5bp + 冲击 2bp,单边;往返固定摩擦 = 14bp(不含 funding)
FILL_MODEL = PerpFillModel(
    taker_fee_bps=5.0,
    slippage_bps=2.0,
    funding_interval_hours=8.0,
)

LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=5,
    flat_margin=0.0002,   # 线性标的的盘整带比期权窄一个量级
)

REPLAY = ReplayConfig(
    entry_threshold=0.002,   # 往返成本 ~14bp,净阈值 20bp 起步(待回放校准)
    entry_delay_bars=1,
    max_spread_pct=1.0,      # 永续无点差门控语义,设为不生效
    cooldown_bars=5,
    long_only=False,         # 永续天然双向
)

EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.006,    # 线性标的止损量级 = 期权的 1/20 左右
    soft_stop_roi=-0.004,
    time_stop_bars=15,
    time_stop_min_roi=0.001,
    max_hold_bars=60,
    trailing_trigger_roi=0.008,
    trailing_keep_ratio=0.60,
    ladder=((0.003, 0.001), (0.006, 0.003), (0.012, 0.008)),
    flash_trigger_roi=0.004,
    flash_exit_roi=0.0002,
    eod_close_bar_index=None,  # 24x7,无 EOD 强平
)

MAX_POSITIONS = 1
