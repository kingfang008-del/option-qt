#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build MAG7 non-0DTE minute features for V4 TFT — NEW file, does not modify old pipelines.

Steps:
  1) Slice full-chain day_iv by non-0DTE locked map → assign bucket_id
  2) day → monthly
  3) monthly → bucketed (calls existing options_locked_feature.process_single_file)
  4) feature_merge (calls existing feature_merge_option_raw.process_stock_month)
  5) enrich stock_dte / expiry_weekday / time_to_expiry_norm

Usage:
  python stock_options/tools/feature_build_stock_non0dte.py --symbol NVDA --step all
  python stock_options/tools/feature_build_stock_non0dte.py --symbol TSLA --step day-iv
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.non0dte_config import (
    ANCHOR_CONFIG,
    BASE_FEATURE_CONFIG_PATH,
    FEATURE_CONFIG_PATH,
    RESEARCH_END,
    RESEARCH_START,
)

logger = logging.getLogger("stock_non0dte_features")

STOCK_NON0_FEATURES = [
    {"name": "stock_dte", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "Trading DTE of selected non-0DTE expiry (1 or 2)."},
    {"name": "stock_dte_norm", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "stock_dte / 2"},
    {"name": "stock_trade_weekday", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "Trade weekday 0=Mon..4=Fri"},
    {"name": "stock_expiry_weekday", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "Expiry weekday of selected contract"},
    {"name": "stock_is_0dte", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "Always 0 in this profile"},
    {"name": "stock_is_non0dte", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "1 if stock_dte in {1,2}"},
]


def load_profile(symbol: str) -> Any:
    return importlib.import_module(f"stock_options.{symbol.lower()}.config_non0dte").CONFIG


def ensure_feature_config() -> Path:
    with open(BASE_FEATURE_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    params = cfg.setdefault("parameters", {})
    params["anchor_profile"] = "stock_non0dte"
    params["anchor_config_path"] = str(ANCHOR_CONFIG)
    params["model_family"] = "tft_dual_stream_v4"
    params["exclude_0dte"] = True
    params["stock_non0dte"] = {
        "allowed_dte": [1, 2],
        "prefer_dte": 1,
        "research_start": RESEARCH_START,
    }
    features = cfg.setdefault("features", [])
    existing = {f.get("name") for f in features}
    for feat in STOCK_NON0_FEATURES:
        if feat["name"] not in existing:
            features.append(feat)
    FEATURE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", FEATURE_CONFIG_PATH)
    return FEATURE_CONFIG_PATH


def _norm_ticker(x: str) -> str:
    return str(x).replace("O:", "").strip()


def build_day_iv_locked(cfg: Any, force: bool = False) -> None:
    """Filter chain day_iv to locked non-0DTE contracts and attach bucket_id."""
    lock = pd.read_parquet(cfg.locked_map)
    lock = lock[lock["symbol"] == cfg.symbol].copy()
    if lock.empty:
        raise SystemExit(f"no lock rows for {cfg.symbol}")
    lock["contract_norm"] = lock["contract_symbol"].map(_norm_ticker)
    lock = lock[(lock["date_str"] >= cfg.research_start) & (lock["date_str"] <= cfg.research_end)]

    out_dir = cfg.day_iv_root / cfg.symbol / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = cfg.day_iv_src / cfg.symbol

    n_ok = n_skip = n_miss = 0
    for date_str, g in tqdm(lock.groupby("date_str"), desc=f"{cfg.symbol} day-iv"):
        out_fp = out_dir / f"{cfg.symbol}_{date_str}.parquet"
        if out_fp.exists() and not force:
            n_skip += 1
            continue
        src_fp = src_dir / f"{cfg.symbol}_{date_str}.parquet"
        if not src_fp.exists():
            n_miss += 1
            logger.warning("missing day_iv %s", src_fp)
            continue

        raw = pd.read_parquet(src_fp)
        if raw.empty:
            n_miss += 1
            continue
        raw = raw.rename(columns={"ticker": "contract_symbol"} if "ticker" in raw.columns else {})
        if "contract_symbol" not in raw.columns and "ticker" in raw.columns:
            raw["contract_symbol"] = raw["ticker"]
        raw["contract_norm"] = raw["contract_symbol"].map(_norm_ticker)

        want = g[["contract_norm", "bucket_id", "front_dte"]].drop_duplicates("contract_norm")
        merged = raw.merge(want, on="contract_norm", how="inner")
        if merged.empty:
            n_miss += 1
            logger.warning("no locked contracts matched for %s %s", cfg.symbol, date_str)
            continue

        # Micro proxies when chain day_iv has no bid/ask
        close = pd.to_numeric(merged.get("close", merged.get("price", 0)), errors="coerce").fillna(0.0)
        high = pd.to_numeric(merged.get("high", close), errors="coerce").fillna(close)
        low = pd.to_numeric(merged.get("low", close), errors="coerce").fillna(close)
        spread = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged["spread_pct"] = spread.clip(0.0, 1.0)
        merged["volume_imbalance"] = 0.0
        # Synthesize bid/ask so existing label_pipeline (requires bid>0) works
        # without modifying qqq_btc label code.
        half = (merged["spread_pct"].clip(0.0, 0.5) * close / 2.0).fillna(0.0)
        merged["bid"] = (close - half).clip(lower=0.01)
        merged["ask"] = (close + half).clip(lower=merged["bid"] + 0.01)
        # Keep only `ticker` (not also contract_symbol): options_locked_feature
        # renames ticker→contract_symbol and would duplicate columns otherwise.
        merged["ticker"] = merged["contract_norm"]
        merged["front_dte"] = merged["front_dte"].astype(int)
        merged["bucket_id"] = merged["bucket_id"].astype(int)

        keep = [
            c
            for c in [
                "timestamp",
                "ticker",
                "bucket_id",
                "front_dte",
                "expiration_date",
                "contract_type",
                "strike_price",
                "open",
                "high",
                "low",
                "close",
                "bid",
                "ask",
                "volume",
                "iv",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
                "vanna",
                "charm",
                "stock_close",
                "spread_pct",
                "volume_imbalance",
            ]
            if c in merged.columns
        ]
        merged[keep].to_parquet(out_fp, index=False, compression="zstd")
        n_ok += 1

    logger.info("day_iv_locked %s ok=%d skip=%d miss=%d -> %s", cfg.symbol, n_ok, n_skip, n_miss, out_dir)


def build_monthly(cfg: Any) -> None:
    from preprocess.ask_bid.iv_day2month import process_single_symbol

    src = cfg.day_iv_root
    files = sorted((src / cfg.symbol / "standard").glob(f"{cfg.symbol}_*.parquet"))
    if not files:
        raise SystemExit(f"no locked day_iv under {src / cfg.symbol / 'standard'}")
    out = cfg.monthly_iv_root
    out.mkdir(parents=True, exist_ok=True)
    logger.info("monthly %s files=%d -> %s", cfg.symbol, len(files), out)
    msg = process_single_symbol((cfg.symbol, [str(p) for p in files], str(out)))
    logger.info("%s", msg)


def build_bucketed(cfg: Any, workers: int = 8) -> None:
    from preprocess.ask_bid.options_locked_feature import process_single_file

    src = cfg.monthly_iv_root / cfg.symbol / "standard"
    if not src.exists():
        # iv_day2month may write symbol/standard/YYYY-MM.parquet
        alt = cfg.monthly_iv_root / cfg.symbol
        src = alt / "standard" if (alt / "standard").exists() else alt
    out = cfg.bucketed_root
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no monthly files under {src}")
    tasks = [(p, out, cfg.symbol) for p in files]
    logger.info("bucketed tasks=%d -> %s", len(tasks), out)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_single_file, t) for t in tasks]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="bucketed"):
            res = fut.result()
            if res:
                logger.warning("%s", res)


def build_feature_merge(cfg: Any, stock_root: Path, workers: int, overwrite: bool) -> None:
    ensure_feature_config()
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

    with open(FEATURE_CONFIG_PATH, encoding="utf-8") as f:
        feature_config = json.load(f)

    start = pd.Timestamp(cfg.research_start)
    end = pd.Timestamp(cfg.research_end)
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    tasks = [(cfg.symbol, m.strftime("%Y-%m")) for m in months]
    logger.info("feature-merge tasks=%d out=%s", len(tasks), cfg.feature_raw_root)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fm.process_stock_month, sym, ym, feature_config) for sym, ym in tasks]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="feature-merge"):
            msg = fut.result()
            if isinstance(msg, str) and msg.startswith("[错误]"):
                logger.error("%s", msg)
            else:
                logger.info("%s", msg)


def _enrich_one(path: Path, lock_day: pd.DataFrame, force: bool) -> str:
    try:
        df = pd.read_parquet(path)
        needed = {"stock_dte", "stock_expiry_weekday", "time_to_expiry_norm"}
        if needed.issubset(df.columns) and not force:
            return f"skip {path.name}"
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts_ny = ts.dt.tz_localize("America/New_York", ambiguous="infer")
        else:
            ts_ny = ts.dt.tz_convert("America/New_York")
        date_str = ts_ny.dt.strftime("%Y-%m-%d")
        trade_wd = ts_ny.dt.dayofweek.astype(int)
        minute = ts_ny.dt.hour * 60 + ts_ny.dt.minute
        progress = ((minute - (9 * 60 + 30)).clip(0, 390) / 390.0).astype(float)

        dte_map = lock_day["front_dte"].astype(int).to_dict()
        dte = date_str.map(dte_map).fillna(1).astype(int).clip(1, 2)
        expiry_wd = ((trade_wd.to_numpy() + dte.to_numpy()) % 7).astype(int)

        out = df.copy()
        out["stock_dte"] = dte.astype(float).to_numpy()
        out["stock_dte_norm"] = (dte / 2.0).astype(float).to_numpy()
        out["stock_trade_weekday"] = trade_wd.astype(float).to_numpy()
        out["stock_expiry_weekday"] = expiry_wd.astype(float)
        out["stock_is_0dte"] = 0.0
        out["stock_is_non0dte"] = 1.0
        out["time_to_expiry_norm"] = (
            (dte.to_numpy() + (1.0 - progress.to_numpy())) / 2.0
        ).clip(0.0, 1.5)
        out.to_parquet(path, index=False, compression="zstd")
        return f"ok {path.name}"
    except Exception as exc:  # noqa: BLE001
        return f"err {path}: {exc}"


def enrich_features(cfg: Any, force: bool = False) -> None:
    lock = pd.read_parquet(cfg.locked_map)
    lock = lock[lock["symbol"] == cfg.symbol]
    day = (
        lock.groupby("date_str", as_index=True)["front_dte"]
        .first()
        .to_frame()
    )
    root = cfg.feature_raw_root
    files = sorted(root.glob("*/*/*/*/*.parquet"))
    if not files:
        logger.warning("no feature files under %s", root)
        return
    for fp in tqdm(files, desc="enrich"):
        msg = _enrich_one(fp, day, force)
        if msg.startswith("err"):
            logger.error("%s", msg)


def main() -> None:
    p = argparse.ArgumentParser(description="Stock non-0DTE feature builder (isolated)")
    p.add_argument("--symbol", required=True, choices=["NVDA", "TSLA"])
    p.add_argument(
        "--step",
        default="all",
        choices=["config", "day-iv", "monthly", "bucketed", "merge", "enrich", "all"],
    )
    p.add_argument("--stock-root", type=Path, default=Path.home() / "train_data/spnq_train_resampled")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_profile(args.symbol)

    if args.step in ("config", "all"):
        ensure_feature_config()
    if args.step in ("day-iv", "all"):
        build_day_iv_locked(cfg, force=args.force)
    if args.step in ("monthly", "all"):
        build_monthly(cfg)
    if args.step in ("bucketed", "all"):
        build_bucketed(cfg, workers=args.workers)
    if args.step in ("merge", "all"):
        build_feature_merge(cfg, args.stock_root, args.workers, args.overwrite or args.force)
    if args.step in ("enrich", "all"):
        enrich_features(cfg, force=args.force or True)


if __name__ == "__main__":
    main()
