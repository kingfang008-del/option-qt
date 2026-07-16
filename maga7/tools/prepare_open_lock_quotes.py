#!/usr/bin/env python3
"""
Mag7 open-lock 1s quote pipeline (causal 09:30 multi-DTE lock).

Steps:
  1) lock   — build open_locked_map from day_iv (+ option_1m early 0DTE)
  2) seed   — hardlink/copy overlapping tickers from legacy 1s dirs
  3) miss   — write miss map for contracts still absent in quote_1s_root
  4) quotes — step2 sniper → sidecar *_miss (does not skip seeded day files)
  5) merge  — merge sidecar day files into quote_1s_root

Usage:
  export MASSIVE_API_KEY=...   # or POLYGON_API_KEY

  # full pipeline (default Mag7 open_lock profile)
  python -m maga7.tools.prepare_open_lock_quotes --step all

  # add symbols later (keeps profile list + appends)
  python -m maga7.tools.prepare_open_lock_quotes --step all --add-symbols GOOGL,GOOG

  # only download missing after lock/seed already done
  python -m maga7.tools.prepare_open_lock_quotes --step quotes --max-workers 12 --contract-workers 4

  # coverage only
  python -m maga7.tools.prepare_open_lock_quotes --step status
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.open_lock import build_open_lock_map
from maga7.common.replay import month_list
from maga7.common.signals import load_stock_month_files

DEFAULT_PROFILE = ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json"
DEFAULT_SEED_ROOTS = [
    Path("/mnt/s990/data/raw_1s/maga7_mf10_old_lock"),
    Path("/mnt/s990/data/raw_1s/maga7_mf10_signal_atm"),
    Path("/mnt/s990/data/raw_1s/mag7_short_dte_old_lock"),
]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["PYTHONPATH"] = (
        f"{ROOT}:{ROOT}/preprocess/download"
        + (os.pathsep + merged["PYTHONPATH"] if merged.get("PYTHONPATH") else "")
    )
    subprocess.check_call(cmd, cwd=str(ROOT), env=merged)


def _norm_ticker(x: Any) -> str:
    return str(x).replace("O:", "").strip()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _resolve_symbols(profile: dict, symbols: str | None, add_symbols: str | None) -> list[str]:
    base = [str(s).upper() for s in profile.get("symbols") or []]
    if symbols:
        base = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if add_symbols:
        extra = [s.strip().upper() for s in add_symbols.split(",") if s.strip()]
        for s in extra:
            if s not in base:
                base.append(s)
    if not base:
        raise SystemExit("no symbols: set profile.symbols or --symbols")
    return base


def _paths(profile: dict) -> dict[str, Path]:
    p = profile["_paths"]
    quote = Path(p["quote_1s_root"])
    return {
        "open_locked_map": Path(
            p.get("open_locked_map")
            or (Path.home() / "train_data/locked_targets_map_maga7_open_multidte_jan_jul.parquet")
        ),
        "quote_1s_root": quote,
        "quote_miss_root": Path(str(quote) + "_miss"),
        "stock_1s_root": Path(p.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks"),
        "day_iv_root": Path(p["day_iv_root"]),
        "option_1m_root": Path(p.get("option_1m_root") or Path.home() / "data/new_option_data_s3"),
        "stock_root": Path(p["stock_root"]),
        "miss_map": Path.home()
        / "train_data"
        / f"locked_targets_map_{quote.name}_miss_1s.parquet",
        "download_map": Path.home()
        / "train_data"
        / f"locked_targets_map_{quote.name}_download_1s.parquet",
    }


def _date_range(profile: dict, start: str | None, end: str | None) -> tuple[str, str]:
    dr = profile.get("date_range") or {}
    return start or str(dr["start"]), end or str(dr["end"])


def _to_step2_map(df: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    """Densify bucket_id per (date,symbol) so multi-DTE rows stay distinct for step2."""
    if df.empty:
        return df
    out = df.copy()
    out["date_str"] = out["date_str"].astype(str)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["contract_symbol"] = out["contract_symbol"].astype(str).map(
        lambda t: t if t.startswith("O:") else f"O:{t}"
    )
    out = out.drop_duplicates(["date_str", "symbol", "contract_symbol"], keep="first")
    out = out.sort_values(["date_str", "symbol", "front_dte", "bucket_id", "contract_symbol"])
    out["bucket_id"] = out.groupby(["date_str", "symbol"]).cumcount().astype(int)
    out["front_dte"] = out.get("front_dte", -1)
    out["tag"] = tag
    cols = ["date_str", "symbol", "contract_symbol", "bucket_id", "front_dte", "tag"]
    return out[cols]


def _day_file(root: Path, symbol: str, date: str) -> Path:
    return root / symbol / f"{symbol}_{date}.parquet"


def _tickers_in_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        df = pd.read_parquet(path, columns=["ticker"])
    except Exception:
        return set()
    return {_norm_ticker(t) for t in df["ticker"].tolist()}


def step_lock(
    profile: dict,
    *,
    symbols: list[str],
    start: str,
    end: str,
    out_map: Path,
) -> Path:
    paths = profile["_paths"]
    months = month_list(start, end)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            print(f"WARN: no stock bars for {sym}", flush=True)
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        stock_by[sym] = raw

    df = build_open_lock_map(
        day_iv_root=paths["day_iv_root"],
        symbols=symbols,
        start=start,
        end=end,
        allowed_dte=(profile.get("lock") or {}).get("allowed_dte") or [0, 1, 2],
        stock_by=stock_by,
        option_1m_root=paths.get("option_1m_root"),
        otm_rungs=int(
            (profile.get("trade") or {}).get("ladder_otm_rungs")
            or (profile.get("lock") or {}).get("otm_rungs")
            or 1
        ),
    )
    if df.empty:
        raise SystemExit("empty open lock map — check day_iv / option_1m coverage")
    out_map.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_map, index=False)
    meta = {
        "n_rows": int(len(df)),
        "n_day_symbol": int(df.groupby(["date_str", "symbol"]).ngroups),
        "n_dates": int(df["date_str"].nunique()),
        "symbols": symbols,
        "dte_counts": {str(k): int(v) for k, v in df["front_dte"].value_counts().sort_index().items()},
        "bucket_counts": {str(k): int(v) for k, v in df["bucket_id"].value_counts().sort_index().items()},
        "output": str(out_map),
        "start": start,
        "end": end,
    }
    out_map.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return out_map


def step_seed(
    lock_map: Path,
    dest: Path,
    *,
    symbols: list[str] | None,
    seed_roots: Iterable[Path],
    start: str,
    end: str,
) -> dict[str, int]:
    mp = pd.read_parquet(lock_map)
    mp["date_str"] = mp["date_str"].astype(str)
    mp["symbol"] = mp["symbol"].astype(str).str.upper()
    if symbols:
        mp = mp[mp["symbol"].isin(symbols)]
    mp = mp[(mp["date_str"] >= start) & (mp["date_str"] <= end)]
    want = (
        mp.groupby(["date_str", "symbol"])["contract_symbol"]
        .apply(lambda s: {_norm_ticker(x) for x in s})
        .to_dict()
    )
    roots = [Path(r) for r in seed_roots if Path(r).exists()]
    n_files = 0
    n_tickers = 0
    n_days = 0
    for (d, sym), tickers in want.items():
        frames: list[pd.DataFrame] = []
        for root in roots:
            p = _day_file(root, sym, d)
            if not p.is_file():
                continue
            df = pd.read_parquet(p)
            if "ticker" not in df.columns:
                continue
            sub = df[df["ticker"].map(_norm_ticker).isin(tickers)]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp", "ticker"] if "timestamp" in merged.columns else ["ticker"], keep="last")
        out = _day_file(dest, sym, d)
        existing = _tickers_in_file(out)
        # If dest already has a superset of needed tickers, skip rewrite.
        if tickers.issubset(existing):
            continue
        if out.is_file() and existing:
            old = pd.read_parquet(out)
            merged = pd.concat([old, merged], ignore_index=True)
            if "timestamp" in merged.columns:
                merged = merged.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
            else:
                merged = merged.drop_duplicates(subset=["ticker"], keep="last")
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, index=False)
        n_files += 1
        n_tickers += len({_norm_ticker(t) for t in merged["ticker"]})
        n_days += 1
    stats = {"seeded_days_written": n_days, "seed_files_touched": n_files, "tickers_in_written": n_tickers}
    print(json.dumps({"seed": stats, "dest": str(dest), "roots": [str(r) for r in roots]}, indent=2), flush=True)
    return stats


def build_miss_map(
    lock_map: Path,
    quote_root: Path,
    *,
    symbols: list[str] | None,
    start: str,
    end: str,
    out_miss: Path,
) -> pd.DataFrame:
    mp = pd.read_parquet(lock_map)
    mp["date_str"] = mp["date_str"].astype(str)
    mp["symbol"] = mp["symbol"].astype(str).str.upper()
    if symbols:
        mp = mp[mp["symbol"].isin(symbols)]
    mp = mp[(mp["date_str"] >= start) & (mp["date_str"] <= end)].copy()
    mp["_t"] = mp["contract_symbol"].map(_norm_ticker)

    rows: list[dict[str, Any]] = []
    for (d, sym), g in mp.groupby(["date_str", "symbol"]):
        have = _tickers_in_file(_day_file(quote_root, sym, d))
        for _, r in g.iterrows():
            if r["_t"] in have:
                continue
            rows.append(
                {
                    "date_str": d,
                    "symbol": sym,
                    "contract_symbol": r["contract_symbol"],
                    "bucket_id": int(r["bucket_id"]) if pd.notna(r.get("bucket_id")) else 0,
                    "front_dte": int(r["front_dte"]) if pd.notna(r.get("front_dte")) else -1,
                    "tag": "open_miss",
                }
            )
    miss = pd.DataFrame(rows)
    miss = _to_step2_map(miss, tag="open_miss") if not miss.empty else miss
    out_miss.parent.mkdir(parents=True, exist_ok=True)
    if miss.empty:
        if out_miss.exists():
            out_miss.unlink()
        print(json.dumps({"miss_rows": 0, "message": "fully covered"}, indent=2), flush=True)
        return miss
    miss.to_parquet(out_miss, index=False)
    meta = {
        "miss_rows": int(len(miss)),
        "miss_day_symbol": int(miss.groupby(["date_str", "symbol"]).ngroups),
        "symbols": sorted(miss["symbol"].unique().tolist()),
        "output": str(out_miss),
    }
    out_miss.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return miss


def step_quotes(
    miss_map: Path,
    miss_root: Path,
    stock_1s_root: Path,
    *,
    start: str,
    end: str,
    symbols: list[str] | None,
    max_workers: int,
    contract_workers: int,
    window_start: str,
    window_end: str,
    force: bool,
) -> None:
    if not miss_map.is_file():
        print("no miss map — nothing to download", flush=True)
        return
    mp = pd.read_parquet(miss_map)
    if mp.empty:
        print("miss map empty — nothing to download", flush=True)
        return
    if not os.environ.get("MASSIVE_API_KEY") and not os.environ.get("POLYGON_API_KEY"):
        raise SystemExit("set MASSIVE_API_KEY or POLYGON_API_KEY for quote download")

    miss_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "preprocess/download/step2_polygon_second_sniper_v1.py"),
        "--target-map",
        str(miss_map),
        "--output-dir",
        str(miss_root),
        "--stock-output-dir",
        str(stock_1s_root),
        "--start-date",
        start,
        "--end-date",
        end,
        "--max-workers",
        str(max_workers),
        "--contract-workers",
        str(contract_workers),
        "--window-start",
        window_start,
        "--window-end",
        window_end,
        "--no-download-stock",
        "--allow-partial",
        "--global-pool",
    ]
    if symbols:
        cmd.extend(["--symbols", ",".join(symbols)])
    if force:
        cmd.append("--force")
    _run(cmd)


def step_merge(miss_root: Path, dest: Path, *, symbols: list[str] | None) -> dict[str, int]:
    if not miss_root.exists():
        print(f"no miss root {miss_root}", flush=True)
        return {"merged_days": 0, "merged_tickers": 0}
    n_days = 0
    n_tickers = 0
    for sym_dir in sorted(miss_root.iterdir()):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name.upper()
        if symbols and sym not in symbols:
            continue
        for src in sorted(sym_dir.glob(f"{sym}_*.parquet")):
            date = src.name.replace(f"{sym}_", "").replace(".parquet", "")
            dst = _day_file(dest, sym, date)
            add = pd.read_parquet(src)
            if add.empty or "ticker" not in add.columns:
                continue
            if dst.is_file():
                base = pd.read_parquet(dst)
                merged = pd.concat([base, add], ignore_index=True)
            else:
                merged = add
            if "timestamp" in merged.columns:
                merged = merged.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
            else:
                merged = merged.drop_duplicates(subset=["ticker"], keep="last")
            dst.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(dst, index=False)
            n_days += 1
            n_tickers += int(merged["ticker"].nunique())
    stats = {"merged_days": n_days, "merged_tickers": n_tickers, "dest": str(dest), "miss_root": str(miss_root)}
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def step_status(
    lock_map: Path,
    quote_root: Path,
    *,
    symbols: list[str] | None,
    start: str,
    end: str,
) -> dict[str, Any]:
    if not lock_map.is_file():
        raise SystemExit(f"lock map missing: {lock_map}")
    mp = pd.read_parquet(lock_map)
    mp["date_str"] = mp["date_str"].astype(str)
    mp["symbol"] = mp["symbol"].astype(str).str.upper()
    if symbols:
        mp = mp[mp["symbol"].isin(symbols)]
    mp = mp[(mp["date_str"] >= start) & (mp["date_str"] <= end)].copy()
    mp["_t"] = mp["contract_symbol"].map(_norm_ticker)

    per_sym: list[dict[str, Any]] = []
    total_have = total_miss = 0
    for sym, g in mp.groupby("symbol"):
        have = miss = 0
        days_full = days_partial = days_none = 0
        for d, gd in g.groupby("date_str"):
            want = set(gd["_t"])
            got = _tickers_in_file(_day_file(quote_root, sym, d)) & want
            have += len(got)
            miss += len(want - got)
            if len(got) == len(want):
                days_full += 1
            elif got:
                days_partial += 1
            else:
                days_none += 1
        total_have += have
        total_miss += miss
        per_sym.append(
            {
                "symbol": sym,
                "contracts_have": have,
                "contracts_miss": miss,
                "days_full": days_full,
                "days_partial": days_partial,
                "days_none": days_none,
            }
        )
    out = {
        "lock_map": str(lock_map),
        "quote_root": str(quote_root),
        "start": start,
        "end": end,
        "contracts_have": total_have,
        "contracts_miss": total_miss,
        "coverage_pct": round(100.0 * total_have / max(total_have + total_miss, 1), 2),
        "symbols": per_sym,
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mag7 open-lock map + 1s quote download pipeline")
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument(
        "--step",
        choices=["lock", "seed", "miss", "quotes", "merge", "status", "all"],
        default="all",
    )
    ap.add_argument("--symbols", default=None, help="replace profile symbols, e.g. NVDA,TSLA,GOOGL")
    ap.add_argument("--add-symbols", default=None, help="append to profile symbols")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--max-workers", type=int, default=12, help="day-task thread workers")
    ap.add_argument("--contract-workers", type=int, default=4, help="quote streams per day")
    ap.add_argument("--window-start", default="10:00")
    ap.add_argument("--window-end", default="15:00")
    ap.add_argument("--force", action="store_true", help="overwrite miss sidecar day files")
    ap.add_argument(
        "--seed-roots",
        default=",".join(str(p) for p in DEFAULT_SEED_ROOTS),
        help="comma-separated legacy 1s roots to seed from",
    )
    ap.add_argument("--skip-seed", action="store_true")
    ap.add_argument("--skip-download", action="store_true", help="with --step all: stop after miss map")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    symbols = _resolve_symbols(profile, args.symbols, args.add_symbols)
    profile["symbols"] = symbols
    start, end = _date_range(profile, args.start_date, args.end_date)
    paths = _paths(profile)
    seed_roots = [Path(p.strip()) for p in args.seed_roots.split(",") if p.strip()]

    print(
        json.dumps(
            {
                "profile": args.profile,
                "step": args.step,
                "symbols": symbols,
                "start": start,
                "end": end,
                "open_locked_map": str(paths["open_locked_map"]),
                "quote_1s_root": str(paths["quote_1s_root"]),
                "quote_miss_root": str(paths["quote_miss_root"]),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.step in ("lock", "all"):
        step_lock(
            profile,
            symbols=symbols,
            start=start,
            end=end,
            out_map=paths["open_locked_map"],
        )

    if args.step in ("seed", "all") and not args.skip_seed:
        if not paths["open_locked_map"].is_file():
            raise SystemExit(f"need lock map first: {paths['open_locked_map']}")
        step_seed(
            paths["open_locked_map"],
            paths["quote_1s_root"],
            symbols=symbols,
            seed_roots=seed_roots,
            start=start,
            end=end,
        )

    if args.step in ("miss", "all"):
        if not paths["open_locked_map"].is_file():
            raise SystemExit(f"need lock map first: {paths['open_locked_map']}")
        build_miss_map(
            paths["open_locked_map"],
            paths["quote_1s_root"],
            symbols=symbols,
            start=start,
            end=end,
            out_miss=paths["miss_map"],
        )

    if args.step in ("quotes", "all") and not args.skip_download:
        # refresh miss map right before download so seed is reflected
        if paths["open_locked_map"].is_file():
            build_miss_map(
                paths["open_locked_map"],
                paths["quote_1s_root"],
                symbols=symbols,
                start=start,
                end=end,
                out_miss=paths["miss_map"],
            )
        step_quotes(
            paths["miss_map"],
            paths["quote_miss_root"],
            paths["stock_1s_root"],
            start=start,
            end=end,
            symbols=symbols,
            max_workers=args.max_workers,
            contract_workers=args.contract_workers,
            window_start=args.window_start,
            window_end=args.window_end,
            force=args.force,
        )

    if args.step in ("merge", "all") and not args.skip_download:
        step_merge(paths["quote_miss_root"], paths["quote_1s_root"], symbols=symbols)
        step_status(
            paths["open_locked_map"],
            paths["quote_1s_root"],
            symbols=symbols,
            start=start,
            end=end,
        )

    if args.step == "status":
        step_status(
            paths["open_locked_map"],
            paths["quote_1s_root"],
            symbols=symbols,
            start=start,
            end=end,
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()
