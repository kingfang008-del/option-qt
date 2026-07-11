#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个股 0DTE (周频到期) 专用配置 —— 与 qqq.config / config_9dte_legacy 隔离。

设计约束 (Phase-1):
  - 个股多数只有周五 weekly expiry → 周一~周四 DTE=4/3/2/1, 周五 DTE=0
  - 各 weekday 的 gamma/theta/流动性分布不同, 不可混用
  - 数据有限时: 仅周五训练 + 仅周五推理 (train_weekdays == deploy_weekdays)
  - 后续扩展: 加 dte_norm 特征 + 分 weekday 模型, 或统一模型但部署时匹配 DTE

用法:
  from qqq_btc.stock import config_0dte as cfg
  from qqq_btc.stock.config_0dte import for_symbol
  nvda = for_symbol("NVDA")
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Tuple

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.labels import LabelHorizon
from qqq_btc.common.replay_types import ReplayConfig

PROFILE = "stock_0dte_weekly"

# Phase-1: 仅周五 (0=Mon .. 4=Fri)
TRAIN_WEEKDAYS: Tuple[int, ...] = (4,)
DEPLOY_WEEKDAYS: Tuple[int, ...] = (4,)

FILL_MODEL = OptionSpreadFillModel(
    entry_frac=0.775,
    exit_frac=0.775,
    commission_per_contract=0.65,
    contract_multiplier=100.0,
)

LABEL_HORIZON = LabelHorizon(
    entry_delay_bars=1,
    hold_bars=30,
    flat_margin=0.0005,
)

TRADE_BUCKET_ID = 2
PUT_BUCKET_ID = 0

ANCHOR_CONFIG = _PKG_ROOT / "CONFIG/anchor_stock_0dte_weekly.json"
FEATURE_CONFIG_PATH = _PKG_ROOT / "CONFIG/slow_feature_qqq_v2.json"
SYMBOL_MAP_PATH = _PKG_ROOT / "CONFIG/symbol_map_stock.json"

RAW_1S_ROOT = Path("/mnt/s990/data/raw_1s/options")
RAW_1M_ROOT = Path("/mnt/s990/data/raw_1m/options")

# 默认 NVDA pilot; for_symbol() 按 symbol 覆写路径
DEFAULT_SYMBOL = "NVDA"

REPLAY = ReplayConfig(
    entry_threshold=0.03,
    entry_delay_bars=1,
    max_spread_pct=0.12,
    cooldown_bars=10,
    long_only=False,
    entry_threshold_schedule=((15, 0.03), (270, 0.04), (330, 0.05)),
    session_entry_start_bar=15,
    session_entry_end_bar=330,
    edge_q10_floor=-0.05,
    entry_quantile=0.85,
    entry_quantile_window=1500,
    entry_quantile_min_obs=300,
    position_frac=0.25,
    max_trades_per_day=4,
    daily_loss_stop=-0.15,
)

EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.15,
    soft_stop_roi=-0.12,
    early_stop_bars=15,
    early_stop_roi=-0.05,
    time_stop_bars=30,
    time_stop_min_roi=0.03,
    max_hold_bars=45,
    trailing_trigger_roi=0.25,
    trailing_keep_ratio=0.65,
    eod_close_bar_index=380,
)

EDGE_Q10_COL = "net_edge_q10"
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
PUT_GATE_COL = "vix_level"
MAX_POSITIONS = 1


def _symbol_suffix(symbol: str) -> str:
    return symbol.lower()


def for_symbol(symbol: str) -> dict:
    """返回某标的的路径 bundle (feature/lmdb/checkpoint/anchor patch)。"""
    sym = symbol.upper()
    suf = _symbol_suffix(sym)
    home = Path.home()
    repo = _PKG_ROOT.parent
    return {
        "symbol": sym,
        "profile": PROFILE,
        "train_weekdays": TRAIN_WEEKDAYS,
        "deploy_weekdays": DEPLOY_WEEKDAYS,
        "raw_1s_root": RAW_1S_ROOT / sym,
        "raw_1m_root": RAW_1M_ROOT / sym,
        "day_iv_root": home / f"train_data/quote_options_day_iv_stock_0dte_{suf}",
        "feature_train_root": home / f"train_data/quote_features_train_stock_0dte_{suf}",
        "feature_val_root": home / f"train_data/quote_features_val_stock_0dte_{suf}",
        "feature_test_root": home / f"train_data/quote_features_test_stock_0dte_{suf}",
        "lmdb_train": home / f"train_data/lmdb/train_{suf}_stock_0dte_fri.lmdb",
        "lmdb_val": home / f"train_data/lmdb/val_{suf}_stock_0dte_fri.lmdb",
        "checkpoint": repo / f"checkpoints_{suf}_stock_0dte_fri/best.pth",
        "anchor_config": ANCHOR_CONFIG,
        "feature_config": FEATURE_CONFIG_PATH,
        "fill_model": FILL_MODEL,
        "label_horizon": LABEL_HORIZON,
        "replay": REPLAY,
        "exit_rails": EXIT_RAILS,
    }
