#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature merge adapter for single-stock weekly-DTE option profiles.

The original ``preprocess.ask_bid.feature_merge_option_raw`` module is kept as
the computation engine. This wrapper isolates stock-option paths and adds DTE
features that are not valid under the QQQ 0DTE assumption.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.weekly_dte_config import (
    ANCHOR_CONFIG,
    BASE_FEATURE_CONFIG_PATH,
    FEATURE_CONFIG_PATH,
)

logger = logging.getLogger("stock_feature_merge")

STOCK_WEEKLY_FEATURES = [
    {
        "name": "stock_dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "Integer days from trade date to selected option expiration.",
    },
    {
        "name": "stock_dte_norm",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "stock_dte normalized by the weekly-DTE max bucket.",
    },
    {
        "name": "stock_trade_weekday",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "Trade weekday, 0=Monday ... 4=Friday.",
    },
    {
        "name": "stock_expiry_weekday",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "Selected option expiration weekday, usually 4 for weekly Friday.",
    },
    {
        "name": "stock_days_to_friday",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "Calendar-day distance from trade date to nearest Friday.",
    },
    {
        "name": "stock_is_weekly_dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "1 if selected DTE is in the weekly-DTE profile bucket.",
    },
    {
        "name": "stock_is_0dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "1 if selected option expires on the trade date.",
    },
]


def load_profile(symbol: str) -> Any:
    module = importlib.import_module(f"stock_options.{symbol.lower()}.config_weekly_dte")
    return module.CONFIG


def ensure_feature_config(
    base_config: Path = BASE_FEATURE_CONFIG_PATH,
    output_config: Path = FEATURE_CONFIG_PATH,
) -> Path:
    """Derive stock weekly-DTE feature config from the QQQ v4 schema."""
    with open(base_config, encoding="utf-8") as f:
        cfg = json.load(f)

    params = cfg.setdefault("parameters", {})
    params["anchor_profile"] = "stock_weekly_dte"
    params["anchor_config_path"] = str(ANCHOR_CONFIG)
    params["stock_weekly_dte"] = {
        "allowed_dte": [1, 2, 3, 4, 7],
        "train_weekdays": [0, 1, 2, 3],
        "deploy_weekdays": [0, 1, 2, 3],
        "exclude_0dte": True,
    }

    features = cfg.setdefault("features", [])
    existing = {f.get("name") for f in features}
    for feature in STOCK_WEEKLY_FEATURES:
        if feature["name"] not in existing:
            features.append(feature)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("feature config ready: %s", output_config)
    return output_config


def _ny_timestamps(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize("America/New_York", ambiguous="infer")
    return ts.dt.tz_convert("America/New_York")


def _session_progress(ts_ny: pd.Series) -> pd.Series:
    minute = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    progress = ((minute - (9 * 60 + 30)).clip(lower=0, upper=390) / 390.0).astype(float)
    return progress


def _coerce_dte(df: pd.DataFrame, ts_ny: pd.Series, fallback_days_to_friday: pd.Series) -> pd.Series:
    for col in ("front_dte", "option_dte", "dte"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").round().fillna(fallback_days_to_friday).astype(int)

    for col in ("expiration", "expiration_date"):
        if col in df.columns:
            exp = pd.to_datetime(df[col], errors="coerce")
            if exp.dt.tz is None:
                exp = exp.dt.tz_localize("America/New_York", ambiguous="infer")
            else:
                exp = exp.dt.tz_convert("America/New_York")
            dte = (exp.dt.normalize() - ts_ny.dt.normalize()).dt.days
            return dte.fillna(fallback_days_to_friday).astype(int)

    return fallback_days_to_friday.astype(int)


def add_weekly_dte_features(df: pd.DataFrame, allowed_dte: tuple[int, ...]) -> pd.DataFrame:
    """Add stock weekly-DTE features and override time_to_expiry_norm."""
    if df.empty or "timestamp" not in df.columns:
        return df

    out = df.copy()
    ts_ny = _ny_timestamps(out)
    trade_weekday = ts_ny.dt.dayofweek.astype(int)
    days_to_friday = ((4 - trade_weekday) % 7).replace(0, 7).astype(int)
    dte = _coerce_dte(out, ts_ny, days_to_friday).clip(lower=0)
    progress = _session_progress(ts_ny)
    max_dte = max(max(allowed_dte), 1)

    out["stock_dte"] = dte.astype(float)
    out["stock_dte_norm"] = (dte / max_dte).clip(lower=0.0, upper=1.5).astype(float)
    out["stock_trade_weekday"] = trade_weekday.astype(float)
    out["stock_expiry_weekday"] = ((trade_weekday + dte) % 7).astype(float)
    out["stock_days_to_friday"] = days_to_friday.astype(float)
    out["stock_is_weekly_dte"] = dte.isin(allowed_dte).astype(float)
    out["stock_is_0dte"] = (dte == 0).astype(float)

    # QQQ 0DTE used 1-session progress. For weekly-DTE, keep the same feature
    # name but encode remaining time to the selected expiry bucket.
    out["time_to_expiry_norm"] = ((dte + (1.0 - progress)) / max_dte).clip(0.0, 1.5)
    return out


def enrich_feature_file(path: Path, allowed_dte: tuple[int, ...], force: bool = False) -> str:
    try:
        df = pd.read_parquet(path)
        needed = {"stock_dte", "stock_is_weekly_dte", "time_to_expiry_norm"}
        if needed.issubset(df.columns) and not force:
            return f"[跳过] {path}"
        out = add_weekly_dte_features(df, allowed_dte)
        out.to_parquet(path, index=False, compression="zstd", compression_level=9)
        return f"[成功] {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[错误] {path}: {exc}"


def enrich_feature_tree(root: Path, allowed_dte: tuple[int, ...], force: bool = False, workers: int = 8) -> None:
    files = sorted(root.glob("*/*/*/*/*.parquet"))
    if not files:
        logger.warning("no feature files under %s", root)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futs = [executor.submit(enrich_feature_file, p, allowed_dte, force) for p in files]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="weekly-dte enrich"):
            msg = fut.result()
            if msg.startswith("[错误]"):
                logger.error(msg)


def configure_feature_merge_module(cfg: Any, stock_root: Path, workers: int, overwrite: bool) -> Any:
    import preprocess.ask_bid.feature_merge_option_raw as fm

    fm.STOCK_RESAMPLED_DIR = stock_root
    fm.OPTION_MONTHLY_DIR = cfg.monthly_iv_root
    fm.AGG_OPTION_MONTHLY_DIR = cfg.bucketed_root
    fm.OUTPUT_FEATURES_DIR = cfg.feature_raw_root
    fm.CONFIG_FILE = str(FEATURE_CONFIG_PATH)
    fm.VIX_BASE_DIR = stock_root / "VIXY/regular/09:30-16:00"
    fm.VIX_PATH_TEMPLATE = str(fm.VIX_BASE_DIR / "{res}" / "{year_month}.parquet")
    fm.MAX_WORKERS = workers
    fm.OVERWRITE_EXISTING = overwrite
    return fm


def run_merge(symbol: str, start_date: str, end_date: str, stock_root: Path, workers: int, overwrite: bool) -> None:
    cfg = load_profile(symbol)
    ensure_feature_config()
    fm = configure_feature_merge_module(cfg, stock_root, workers, overwrite)
    with open(FEATURE_CONFIG_PATH, encoding="utf-8") as f:
        feature_config = json.load(f)

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    tasks = [(cfg.symbol, d.strftime("%Y-%m")) for d in dates]
    logger.info("feature merge tasks=%d symbol=%s output=%s", len(tasks), cfg.symbol, cfg.feature_raw_root)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futs = [executor.submit(fm.process_stock_month, sym, ym, feature_config) for sym, ym in tasks]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="feature merge"):
            msg = fut.result()
            if isinstance(msg, str) and msg.startswith("[错误]"):
                logger.error(msg)
            else:
                logger.info(msg)

    enrich_feature_tree(cfg.feature_raw_root, cfg.allowed_dte, force=True, workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock weekly-DTE feature merge adapter")
    parser.add_argument("--symbol", default="NVDA", choices=["NVDA", "TSLA"])
    parser.add_argument("--step", default="all", choices=["config", "merge", "enrich", "all"])
    parser.add_argument("--start-date", default="2023-04-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--stock-root", type=Path, default=Path.home() / "train_data/spnq_train_resampled")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-enrich", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_profile(args.symbol)

    if args.step in ("config", "all"):
        ensure_feature_config()
    if args.step in ("merge", "all"):
        run_merge(args.symbol, args.start_date, args.end_date, args.stock_root, args.workers, args.overwrite)
    if args.step == "enrich":
        enrich_feature_tree(cfg.feature_raw_root, cfg.allowed_dte, force=args.force_enrich, workers=args.workers)


if __name__ == "__main__":
    main()

