#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared short-DTE (trading dte∈{0,1,2}) config for MAG7 weekday modeling."""
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

PROFILE = "stock_short_dte"
ANCHOR_CONFIG = _PKG_ROOT / "CONFIG/anchor_stock_short_dte.json"
SYMBOL_MAP_PATH = _PKG_ROOT / "CONFIG/symbol_map_stock.json"

# V4 TFT backbone feature schema; stock short-DTE adds weekday/DTE columns on top.
BASE_FEATURE_CONFIG_PATH = _REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"
FEATURE_CONFIG_PATH = _PKG_ROOT / "CONFIG/slow_feature_stock_short_dte.json"

# Mon/Wed expiries begin ~2026-02; Fri weekly existed earlier.
RESEARCH_START = "2026-02-02"
MON_WED_EXPECTED_FROM = "2026-02-01"

# Thin but honest TFT split inside the Mon/Wed window.
SPLIT_TRAIN_END = "2026-04-30"
SPLIT_VAL_END = "2026-05-31"
SPLIT_TEST_END = "2026-06-30"

# All trading weekdays; filter by available expiry×dte at runtime, not by hard skip.
TRAIN_WEEKDAYS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEPLOY_WEEKDAYS: tuple[int, ...] = (0, 1, 2, 3, 4)
ALLOWED_DTE: tuple[int, ...] = (0, 1, 2)
# Primary listed expiries for MAG7 short-DTE skeleton.
EXPIRY_WEEKDAYS_PRIMARY: tuple[int, ...] = (0, 2, 4)  # Mon / Wed / Fri

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

# Wider than QQQ by default; recalibrate per (symbol, dte, weekday).
BASE_REPLAY = ReplayConfig(
    entry_threshold=0.001,
    entry_delay_bars=1,
    max_spread_pct=0.08,
    cooldown_bars=10,
    long_only=False,
    entry_threshold_schedule=((15, 0.001), (270, 0.0015), (330, 0.002)),
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

DEFAULT_LOCKED_MAP = Path.home() / "train_data/locked_targets_map_mag7_short_dte_api_ladder.parquet"
DEFAULT_LOCKED_MAP_WEEKDAY = Path.home() / "train_data/locked_targets_map_mag7_short_dte_weekday.parquet"
DEFAULT_MICRO_ROOT = Path("/mnt/s990/data/microstructure/mag7_short_dte_api_ladder")
DEFAULT_STOCK_1S_ROOT = Path("/mnt/s990/data/raw_1s/stocks")


@dataclass(frozen=True)
class StockShortDteConfig:
    """Resolved per-symbol short-DTE paths and runtime parameters."""

    symbol: str
    stock_id: int
    sector_id: int
    profile: str = PROFILE
    research_start: str = RESEARCH_START
    train_weekdays: tuple[int, ...] = TRAIN_WEEKDAYS
    deploy_weekdays: tuple[int, ...] = DEPLOY_WEEKDAYS
    allowed_dte: tuple[int, ...] = ALLOWED_DTE
    expiry_weekdays_primary: tuple[int, ...] = EXPIRY_WEEKDAYS_PRIMARY
    locked_map: Path = DEFAULT_LOCKED_MAP
    locked_map_weekday: Path = DEFAULT_LOCKED_MAP_WEEKDAY
    micro_root: Path = DEFAULT_MICRO_ROOT
    stock_1s_root: Path = DEFAULT_STOCK_1S_ROOT
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
    def stock_1s(self) -> Path:
        return self.stock_1s_root / self.symbol

    @property
    def micro_contract_1s(self) -> Path:
        return self.micro_root / "contract_1s" / self.symbol

    @property
    def day_iv_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_day_iv_stock_short_dte_{self.suffix}"

    @property
    def monthly_iv_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_monthly_iv_stock_short_dte_{self.suffix}"

    @property
    def bucketed_root(self) -> Path:
        return Path.home() / f"train_data/quote_options_bucketed_stock_short_dte_{self.suffix}"

    @property
    def feature_raw_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_raw_stock_short_dte_{self.suffix}"

    @property
    def feature_train_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_train_stock_short_dte_{self.suffix}"

    @property
    def feature_val_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_val_stock_short_dte_{self.suffix}"

    @property
    def feature_test_root(self) -> Path:
        return Path.home() / f"train_data/quote_features_test_stock_short_dte_{self.suffix}"

    @property
    def lmdb_train(self) -> Path:
        return Path.home() / f"train_data/lmdb/train_{self.suffix}_stock_short_dte.lmdb"

    @property
    def lmdb_val(self) -> Path:
        return Path.home() / f"train_data/lmdb/val_{self.suffix}_stock_short_dte.lmdb"

    @property
    def lmdb_test(self) -> Path:
        return Path.home() / f"train_data/lmdb/test_{self.suffix}_stock_short_dte.lmdb"

    @property
    def checkpoint_dir(self) -> Path:
        return _REPO / f"checkpoints_{self.suffix}_stock_short_dte_tft_v4"

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pth"

    @property
    def results_dir(self) -> Path:
        return _PKG_ROOT / "results" / f"short_dte_tft_{self.suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "profile": self.profile,
            "model_family": self.model_family,
            "research_start": self.research_start,
            "split": {
                "train": f"{self.research_start}..{self.split_train_end}",
                "val": f"{self.split_train_end}..{self.split_val_end}",
                "test": f"{self.split_val_end}..{self.split_test_end}",
            },
            "train_weekdays": self.train_weekdays,
            "deploy_weekdays": self.deploy_weekdays,
            "allowed_dte": self.allowed_dte,
            "expiry_weekdays_primary": self.expiry_weekdays_primary,
            "model_shape": "shared_tft_plus_weekday_dte_features",
            "calibration_axes": ["symbol", "selected_dte", "trade_weekday", "expiry_weekday"],
            "locked_map": self.locked_map,
            "locked_map_weekday": self.locked_map_weekday,
            "micro_root": self.micro_root,
            "micro_contract_1s": self.micro_contract_1s,
            "stock_1s": self.stock_1s,
            "day_iv_root": self.day_iv_root,
            "monthly_iv_root": self.monthly_iv_root,
            "bucketed_root": self.bucketed_root,
            "feature_raw_root": self.feature_raw_root,
            "feature_train_root": self.feature_train_root,
            "feature_val_root": self.feature_val_root,
            "feature_test_root": self.feature_test_root,
            "feature_config": self.feature_config,
            "lmdb_train": self.lmdb_train,
            "lmdb_val": self.lmdb_val,
            "lmdb_test": self.lmdb_test,
            "checkpoint": self.checkpoint,
            "anchor_config": self.anchor_config,
            "results_dir": self.results_dir,
        }


def make_config(
    symbol: str,
    *,
    stock_id: int,
    sector_id: int,
    replay: ReplayConfig | None = None,
    exit_rails: ExitRailsConfig | None = None,
) -> StockShortDteConfig:
    return StockShortDteConfig(
        symbol=symbol.upper(),
        stock_id=stock_id,
        sector_id=sector_id,
        replay=replay or BASE_REPLAY,
        exit_rails=exit_rails or BASE_EXIT_RAILS,
    )


def with_threshold(cfg: StockShortDteConfig, threshold: float) -> StockShortDteConfig:
    replay = replace(cfg.replay, entry_threshold=threshold)
    return replace(cfg, replay=replay)


def enrich_locked_map_weekdays(df: Any) -> Any:
    """Add trade_weekday / expiry_weekday columns to a locked-map DataFrame."""
    import pandas as pd

    out = df.copy()
    trade = pd.to_datetime(out["date_str"], errors="coerce")
    exp = pd.to_datetime(out["expiration"], errors="coerce")
    out["trade_weekday"] = trade.dt.dayofweek.astype("Int64")
    out["trade_weekday_name"] = trade.dt.day_name()
    out["expiry_weekday"] = exp.dt.dayofweek.astype("Int64")
    out["expiry_weekday_name"] = exp.dt.day_name()
    out["is_mon_wed_fri_expiry"] = out["expiry_weekday"].isin(list(EXPIRY_WEEKDAYS_PRIMARY)).astype(int)
    return out
