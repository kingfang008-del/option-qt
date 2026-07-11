#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared weekly-DTE configuration for single-stock option pilots."""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO = _PKG_ROOT.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.labels import LabelHorizon
from qqq_btc.common.replay_types import ReplayConfig

PROFILE = "stock_weekly_dte"
ANCHOR_CONFIG = _PKG_ROOT / "CONFIG/anchor_stock_weekly_dte.json"
SYMBOL_MAP_PATH = _PKG_ROOT / "CONFIG/symbol_map_stock.json"

BASE_FEATURE_CONFIG_PATH = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"
FEATURE_CONFIG_PATH = _PKG_ROOT / "CONFIG/slow_feature_stock_weekly_dte.json"

TRAIN_WEEKDAYS: tuple[int, ...] = (0, 1, 2, 3)
DEPLOY_WEEKDAYS: tuple[int, ...] = (0, 1, 2, 3)
ALLOWED_DTE: tuple[int, ...] = (1, 2, 3, 4, 7)

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

# Conservative first pass. These must be recalibrated per symbol and per DTE
# bucket before any live deployment.
BASE_REPLAY = ReplayConfig(
    entry_threshold=0.001,
    entry_delay_bars=1,
    max_spread_pct=0.10,
    cooldown_bars=10,
    long_only=False,
    entry_threshold_schedule=((15, 0.001), (270, 0.0015), (330, 0.002)),
    session_entry_start_bar=15,
    session_entry_end_bar=330,
    edge_q10_floor=-0.05,
    entry_quantile=0.85,
    entry_quantile_window=1500,
    entry_quantile_min_obs=300,
    position_frac=0.20,
    max_trades_per_day=4,
    daily_loss_stop=-0.15,
)

BASE_EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.15,
    soft_stop_roi=-0.12,
    early_stop_bars=15,
    early_stop_roi=-0.05,
    time_stop_bars=30,
    time_stop_min_roi=0.02,
    max_hold_bars=45,
    trailing_trigger_roi=0.20,
    trailing_keep_ratio=0.65,
    eod_close_bar_index=380,
)

EDGE_Q10_COL = "net_edge_q10"
CALL_EDGE_COL = "call_net_edge"
PUT_EDGE_COL = "put_net_edge"
PUT_GATE_COL = "vix_level"
MAX_POSITIONS = 1


@dataclass(frozen=True)
class StockWeeklyDteConfig:
    """Resolved per-symbol paths and runtime parameters."""

    symbol: str
    stock_id: int
    sector_id: int
    profile: str = PROFILE
    train_weekdays: tuple[int, ...] = TRAIN_WEEKDAYS
    deploy_weekdays: tuple[int, ...] = DEPLOY_WEEKDAYS
    allowed_dte: tuple[int, ...] = ALLOWED_DTE
    raw_1s_root: Path = Path("/mnt/s990/data/raw_1s/options")
    raw_1m_root: Path = Path("/mnt/s990/data/raw_1m/options")
    anchor_config: Path = ANCHOR_CONFIG
    feature_config: Path = FEATURE_CONFIG_PATH
    symbol_map: Path = SYMBOL_MAP_PATH
    fill_model: OptionSpreadFillModel = FILL_MODEL
    label_horizon: LabelHorizon = LABEL_HORIZON
    replay: ReplayConfig = BASE_REPLAY
    exit_rails: ExitRailsConfig = BASE_EXIT_RAILS

    @property
    def suffix(self) -> str:
        return self.symbol.lower()

    @property
    def day_iv_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_day_iv_stock_weekly_dte_{self.suffix}"

    @property
    def monthly_iv_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_monthly_iv_stock_weekly_dte_{self.suffix}"

    @property
    def bucketed_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_bucketed_stock_weekly_dte_{self.suffix}"

    @property
    def feature_raw_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_raw_stock_weekly_dte_{self.suffix}"

    @property
    def feature_train_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_train_stock_weekly_dte_{self.suffix}"

    @property
    def feature_val_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_val_stock_weekly_dte_{self.suffix}"

    @property
    def feature_test_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_test_stock_weekly_dte_{self.suffix}"

    @property
    def lmdb_train(self) -> Path:
        return Path.home() / f"train_data/lmdb/train_{self.suffix}_stock_weekly_dte.lmdb"

    @property
    def lmdb_val(self) -> Path:
        return Path.home() / f"train_data/lmdb/val_{self.suffix}_stock_weekly_dte.lmdb"

    @property
    def lmdb_test(self) -> Path:
        return Path.home() / f"train_data/lmdb/test_{self.suffix}_stock_weekly_dte.lmdb"

    @property
    def checkpoint_dir(self) -> Path:
        return _REPO / f"checkpoints_{self.suffix}_stock_weekly_dte"

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pth"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "profile": self.profile,
            "train_weekdays": self.train_weekdays,
            "deploy_weekdays": self.deploy_weekdays,
            "allowed_dte": self.allowed_dte,
            "feature_train_root": self.feature_train_root,
            "feature_val_root": self.feature_val_root,
            "feature_test_root": self.feature_test_root,
            "feature_raw_root": self.feature_raw_root,
            "monthly_iv_root": self.monthly_iv_root,
            "bucketed_root": self.bucketed_root,
            "lmdb_train": self.lmdb_train,
            "lmdb_val": self.lmdb_val,
            "lmdb_test": self.lmdb_test,
            "checkpoint": self.checkpoint,
            "anchor_config": self.anchor_config,
            "feature_config": self.feature_config,
            "symbol_map": self.symbol_map,
        }


def make_config(
    symbol: str,
    *,
    stock_id: int,
    sector_id: int,
    replay: ReplayConfig | None = None,
    exit_rails: ExitRailsConfig | None = None,
) -> StockWeeklyDteConfig:
    """Create a resolved stock weekly-DTE config."""
    return StockWeeklyDteConfig(
        symbol=symbol.upper(),
        stock_id=stock_id,
        sector_id=sector_id,
        replay=replay or BASE_REPLAY,
        exit_rails=exit_rails or BASE_EXIT_RAILS,
    )


def with_threshold(cfg: StockWeeklyDteConfig, threshold: float) -> StockWeeklyDteConfig:
    """Return a copy with a static replay threshold override."""
    replay = replace(cfg.replay, entry_threshold=threshold)
    return replace(cfg, replay=replay)

