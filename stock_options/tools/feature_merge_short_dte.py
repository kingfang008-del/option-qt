#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature merge adapter for MAG7 short-DTE TFT (V4 backbone).

Reuses ``preprocess.ask_bid.feature_merge_option_raw`` as the engine, then
enriches with trading-DTE / expiry-weekday columns from the short-DTE locked map
so the dual-stream TFT can share one model across Mon/Wed/Fri skeletons.
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

import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.short_dte_config import (
    ANCHOR_CONFIG,
    BASE_FEATURE_CONFIG_PATH,
    FEATURE_CONFIG_PATH,
    RESEARCH_START,
    enrich_locked_map_weekdays,
)

logger = logging.getLogger("stock_short_dte_feature_merge")

STOCK_SHORT_DTE_FEATURES = [
    {
        "name": "stock_dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "Trading DTE of the selected short-DTE expiry (0/1/2).",
    },
    {
        "name": "stock_dte_norm",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "stock_dte / 2, clipped for TFT scale stability.",
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
        "description": "Selected option expiration weekday (Mon/Wed/Fri primary).",
    },
    {
        "name": "stock_is_mon_wed_fri_expiry",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "1 if expiry weekday is Mon/Wed/Fri skeleton.",
    },
    {
        "name": "stock_is_0dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "1 if selected trading DTE is 0.",
    },
    {
        "name": "stock_is_short_dte",
        "type": "real",
        "calc": "raw",
        "resolution": "1min",
        "description": "1 if selected trading DTE is in {0,1,2}.",
    },
]


def load_profile(symbol: str) -> Any:
    module = importlib.import_module(f"stock_options.{symbol.lower()}.config_short_dte")
    return module.CONFIG


def ensure_feature_config(
    base_config: Path = BASE_FEATURE_CONFIG_PATH,
    output_config: Path = FEATURE_CONFIG_PATH,
) -> Path:
    """Derive stock short-DTE TFT feature config from QQQ v4 schema."""
    with open(base_config, encoding="utf-8") as f:
        cfg = json.load(f)

    params = cfg.setdefault("parameters", {})
    params["anchor_profile"] = "stock_short_dte"
    params["anchor_config_path"] = str(ANCHOR_CONFIG)
    params["model_family"] = "tft_dual_stream_v4"
    params["stock_short_dte"] = {
        "allowed_dte": [0, 1, 2],
        "train_weekdays": [0, 1, 2, 3, 4],
        "deploy_weekdays": [0, 1, 2, 3, 4],
        "research_start": RESEARCH_START,
        "expiry_weekdays_primary": [0, 2, 4],
        "prefer_dte": 0,
    }

    features = cfg.setdefault("features", [])
    existing = {f.get("name") for f in features}
    for feature in STOCK_SHORT_DTE_FEATURES:
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
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize("America/New_York", ambiguous="infer")
    return ts.dt.tz_convert("America/New_York")


def _session_progress(ts_ny: pd.Series) -> pd.Series:
    minute = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    return ((minute - (9 * 60 + 30)).clip(lower=0, upper=390) / 390.0).astype(float)


def load_day_dte_lookup(locked_map: Path, symbol: str, prefer_dte: int = 0) -> pd.DataFrame:
    """One row per trade date: prefer available DTE closest to prefer_dte."""
    df = pd.read_parquet(locked_map)
    df = df[df["symbol"] == symbol.upper()].copy()
    if df.empty:
        raise SystemExit(f"no locked-map rows for {symbol} in {locked_map}")
    if "expiry_weekday_name" not in df.columns:
        df = enrich_locked_map_weekdays(df)

    day = (
        df.groupby(["date_str", "selected_dte", "expiration"], as_index=False)
        .agg(
            trade_weekday=("trade_weekday", "first"),
            expiry_weekday=("expiry_weekday", "first"),
            is_mon_wed_fri_expiry=("is_mon_wed_fri_expiry", "first"),
        )
        .sort_values(["date_str", "selected_dte"])
    )
    # Prefer prefer_dte, else nearest lower, else nearest higher.
    picked = []
    for date_str, g in day.groupby("date_str"):
        g = g.copy()
        g["abs_gap"] = (g["selected_dte"] - prefer_dte).abs()
        g = g.sort_values(["abs_gap", "selected_dte"])
        picked.append(g.iloc[0])
    out = pd.DataFrame(picked)
    return out.set_index("date_str")


def add_short_dte_features(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    allowed_dte: tuple[int, ...],
) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df

    out = df.copy()
    ts_ny = _ny_timestamps(out)
    date_str = ts_ny.dt.strftime("%Y-%m-%d")
    trade_weekday = ts_ny.dt.dayofweek.astype(int)
    progress = _session_progress(ts_ny)

    joined = lookup.reindex(date_str.values)
    dte = pd.to_numeric(joined["selected_dte"], errors="coerce")
    # Fallback: if day missing from map, treat as unknown short-DTE (NaN→-1→clip later flags)
    dte = dte.fillna(-1).astype(int)
    expiry_weekday = pd.to_numeric(joined["expiry_weekday"], errors="coerce")
    expiry_weekday = expiry_weekday.fillna(((trade_weekday + dte.clip(lower=0)) % 7)).astype(int)
    is_primary = pd.to_numeric(joined["is_mon_wed_fri_expiry"], errors="coerce").fillna(0).astype(int)

    max_dte = max(max(allowed_dte), 1)
    out["stock_dte"] = dte.clip(lower=0).astype(float)
    out["stock_dte_norm"] = (out["stock_dte"] / max_dte).clip(0.0, 1.5).astype(float)
    out["stock_trade_weekday"] = trade_weekday.astype(float)
    out["stock_expiry_weekday"] = expiry_weekday.astype(float)
    out["stock_is_mon_wed_fri_expiry"] = is_primary.astype(float)
    out["stock_is_0dte"] = (dte == 0).astype(float)
    out["stock_is_short_dte"] = dte.isin(list(allowed_dte)).astype(float)

    # Remaining time to selected expiry in "sessions" units, normalized by max_dte.
    out["time_to_expiry_norm"] = (
        (out["stock_dte"] + (1.0 - progress)) / max_dte
    ).clip(0.0, 1.5)
    return out


def enrich_feature_file(
    path: Path,
    lookup: pd.DataFrame,
    allowed_dte: tuple[int, ...],
    force: bool = False,
) -> str:
    try:
        df = pd.read_parquet(path)
        needed = {"stock_dte", "stock_expiry_weekday", "time_to_expiry_norm"}
        if needed.issubset(df.columns) and not force:
            return f"[跳过] {path}"
        out = add_short_dte_features(df, lookup, allowed_dte)
        out.to_parquet(path, index=False, compression="zstd", compression_level=9)
        return f"[成功] {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[错误] {path}: {exc}"


def enrich_feature_tree(
    root: Path,
    locked_map: Path,
    symbol: str,
    allowed_dte: tuple[int, ...],
    prefer_dte: int = 0,
    force: bool = False,
    workers: int = 8,
) -> None:
    lookup = load_day_dte_lookup(locked_map, symbol, prefer_dte=prefer_dte)
    files = sorted(root.glob("*/*/*/*/*.parquet"))
    if not files:
        logger.warning("no feature files under %s", root)
        return
    # ProcessPool can't pickle DataFrame lookup easily across forks with large maps;
    # keep sequential/threaded for correctness on weekday join.
    for path in tqdm(files, desc="short-dte enrich"):
        msg = enrich_feature_file(path, lookup, allowed_dte, force=force)
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


def run_merge(
    symbol: str,
    start_date: str,
    end_date: str,
    stock_root: Path,
    workers: int,
    overwrite: bool,
) -> None:
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

    enrich_feature_tree(
        cfg.feature_raw_root,
        cfg.locked_map_weekday if cfg.locked_map_weekday.exists() else cfg.locked_map,
        cfg.symbol,
        cfg.allowed_dte,
        prefer_dte=0,
        force=True,
        workers=workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock short-DTE TFT feature merge")
    parser.add_argument("--symbol", default="NVDA", choices=["NVDA", "TSLA"])
    parser.add_argument("--step", default="config", choices=["config", "merge", "enrich", "all"])
    parser.add_argument("--start-date", default=RESEARCH_START)
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
        enrich_feature_tree(
            cfg.feature_raw_root,
            cfg.locked_map_weekday if cfg.locked_map_weekday.exists() else cfg.locked_map,
            cfg.symbol,
            cfg.allowed_dte,
            prefer_dte=0,
            force=args.force_enrich,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
