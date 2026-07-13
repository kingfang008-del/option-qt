#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stock non-0DTE (trading DTE∈{1,2}) config for V4 TFT — isolated from QQQ 0DTE."""
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

PROFILE = "stock_non0dte"
ANCHOR_CONFIG = _REPO / "preprocess/CONFIG/anchor_stock_non0dte_old_lock.json"
SYMBOL_MAP_PATH = _PKG_ROOT / "CONFIG/symbol_map_stock.json"
BASE_FEATURE_CONFIG_PATH = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"
FEATURE_CONFIG_PATH = _PKG_ROOT / "CONFIG/slow_feature_stock_non0dte.json"

RESEARCH_START = "2026-02-02"
RESEARCH_END = "2026-03-18"
# Thin window: Feb train / early-Mar val / mid-Mar test
SPLIT_TRAIN_END = "2026-02-27"
SPLIT_VAL_END = "2026-03-10"
SPLIT_TEST_END = "2026-03-18"

TRAIN_WEEKDAYS = (0, 1, 2, 3, 4)
DEPLOY_WEEKDAYS = (0, 1, 2, 3, 4)
ALLOWED_DTE = (1, 2)

DEFAULT_LOCKED_MAP = Path.home() / "train_data/locked_targets_map_stock_non0dte_old_lock.parquet"
DEFAULT_DAY_IV_SRC = Path.home() / "train_data/nq_options_day_iv"
BUILD_ROOT = Path.home() / "train_data/builds/stock_non0dte"

FILL_MODEL = OptionSpreadFillModel(
    entry_frac=0.775,
    exit_frac=0.775,
    commission_per_contract=0.65,
    contract_multiplier=100.0,
)
LABEL_HORIZON = LabelHorizon(entry_delay_bars=1, hold_bars=30, flat_margin=0.0005)
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


@dataclass(frozen=True)
class StockNon0DteConfig:
    symbol: str
    stock_id: int
    sector_id: int
    profile: str = PROFILE
    research_start: str = RESEARCH_START
    research_end: str = RESEARCH_END
    train_weekdays: tuple[int, ...] = TRAIN_WEEKDAYS
    deploy_weekdays: tuple[int, ...] = DEPLOY_WEEKDAYS
    allowed_dte: tuple[int, ...] = ALLOWED_DTE
    locked_map: Path = DEFAULT_LOCKED_MAP
    day_iv_src: Path = DEFAULT_DAY_IV_SRC
    build_root: Path = BUILD_ROOT
    anchor_config: Path = ANCHOR_CONFIG
    feature_config: Path = FEATURE_CONFIG_PATH
    symbol_map: Path = SYMBOL_MAP_PATH
    fill_model: OptionSpreadFillModel = FILL_MODEL
    label_horizon: LabelHorizon = LABEL_HORIZON
    replay: ReplayConfig = BASE_REPLAY
    exit_rails: ExitRailsConfig = BASE_EXIT_RAILS
    model_family: str = "tft_dual_stream_v4"
    split_train_end: str = SPLIT_TRAIN_END
    split_val_end: str = SPLIT_VAL_END
    split_test_end: str = SPLIT_TEST_END

    @property
    def suffix(self) -> str:
        return self.symbol.lower()

    @property
    def day_iv_root(self) -> Path:
        return self.build_root / "quote_options_day_iv"

    @property
    def monthly_iv_root(self) -> Path:
        return self.build_root / "quote_options_monthly_iv"

    @property
    def bucketed_root(self) -> Path:
        return self.build_root / "quote_options_bucketed"

    @property
    def feature_raw_root(self) -> Path:
        return self.build_root / "quote_features_raw"

    @property
    def feature_train_root(self) -> Path:
        return self.build_root / "quote_features_train"

    @property
    def feature_val_root(self) -> Path:
        return self.build_root / "quote_features_val"

    @property
    def feature_test_root(self) -> Path:
        return self.build_root / "quote_features_test"

    @property
    def lmdb_train(self) -> Path:
        return Path.home() / f"train_data/lmdb/train_{self.suffix}_stock_non0dte.lmdb"

    @property
    def lmdb_val(self) -> Path:
        return Path.home() / f"train_data/lmdb/val_{self.suffix}_stock_non0dte.lmdb"

    @property
    def lmdb_test(self) -> Path:
        return Path.home() / f"train_data/lmdb/test_{self.suffix}_stock_non0dte.lmdb"

    @property
    def checkpoint_dir(self) -> Path:
        return _REPO / f"checkpoints_{self.suffix}_stock_non0dte_tft_v4"

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pth"

    @property
    def results_dir(self) -> Path:
        return _PKG_ROOT / "results" / f"non0dte_tft_{self.suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "profile": self.profile,
            "model_family": self.model_family,
            "exclude_0dte": True,
            "allowed_dte": self.allowed_dte,
            "research": f"{self.research_start}..{self.research_end}",
            "split": {
                "train": f"{self.research_start}..{self.split_train_end}",
                "val": f"{self.split_train_end}..{self.split_val_end}",
                "test": f"{self.split_val_end}..{self.split_test_end}",
            },
            "locked_map": self.locked_map,
            "build_root": self.build_root,
            "feature_config": self.feature_config,
            "lmdb_train": self.lmdb_train,
            "lmdb_val": self.lmdb_val,
            "checkpoint": self.checkpoint,
        }


def make_config(symbol: str, *, stock_id: int, sector_id: int, replay=None, exit_rails=None) -> StockNon0DteConfig:
    return StockNon0DteConfig(
        symbol=symbol.upper(),
        stock_id=stock_id,
        sector_id=sector_id,
        replay=replay or BASE_REPLAY,
        exit_rails=exit_rails or BASE_EXIT_RAILS,
    )


def with_threshold(cfg: StockNon0DteConfig, threshold: float) -> StockNon0DteConfig:
    return replace(cfg, replay=replace(cfg.replay, entry_threshold=threshold))
