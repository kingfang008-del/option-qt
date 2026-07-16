#!/usr/bin/env python3
"""
Prepare Mag7 Jan–Jul data for maga7 mf10 Top2:

  1) (optional) report stock / day_iv gaps
  2) step1_build_target_map_old → locked map
  3) reuse existing May–Jul 1s quotes if present; step2 download missing days

Usage:
  export MASSIVE_API_KEY=...   # or POLYGON_API_KEY
  python -m maga7.tools.prepare_jan_jul_data --step all
  python -m maga7.tools.prepare_jan_jul_data --step lock
  python -m maga7.tools.prepare_jan_jul_data --step quotes --max-workers 12
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def report_gaps(profile: dict) -> None:
    import pandas as pd

    paths = profile["_paths"]
    symbols = profile["symbols"]
    start, end = profile["date_range"]["start"], profile["date_range"]["end"]
    iv_root = paths["day_iv_root"]
    stock_root = paths["stock_root"]
    print(f"=== coverage {start} .. {end} ===")
    for sym in symbols:
        iv_days = sorted(
            p.name.split("_")[1].replace(".parquet", "")
            for p in (iv_root / sym).glob(f"{sym}_*.parquet")
            if "_high_features" not in p.name and start <= p.name.split("_")[1].replace(".parquet", "") <= end
        )
        stock_days = set()
        for p in (stock_root / sym).glob("2026-*.parquet"):
            df = pd.read_parquet(p, columns=["timestamp"])
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
            stock_days |= set(ts.dt.strftime("%Y-%m-%d"))
        stock_days = sorted(d for d in stock_days if start <= d <= end)
        print(
            f"{sym}: day_iv={len(iv_days)} ({iv_days[0] if iv_days else None}..{iv_days[-1] if iv_days else None}) "
            f"stock={len(stock_days)} ({stock_days[0] if stock_days else None}..{stock_days[-1] if stock_days else None})"
        )
    print(
        "NOTE: known stock hole ~2026-03-19..2026-04-30 → no day_iv/lock those days until stock backfilled."
    )


def step_lock(profile: dict) -> Path:
    paths = profile["_paths"]
    out = paths["locked_map"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = profile["lock"]["config"]
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT}/preprocess/download:" + env.get("PYTHONPATH", "")
    _run(
        [
            py,
            "-u",
            str(ROOT / "preprocess/download/step1_build_target_map_old.py"),
            "--config",
            str(ROOT / cfg) if not Path(cfg).is_absolute() else cfg,
            "--dte-mode",
            str(profile["lock"].get("dte_mode", "trading")),
            "--symbols",
            ",".join(profile["symbols"]),
            "--start-date",
            profile["date_range"]["start"],
            "--end-date",
            profile["date_range"]["end"],
            "--raw-dir",
            str(paths["day_iv_root"]),
            "--output",
            str(out),
        ],
        env=env,
    )
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def seed_quotes_from_legacy(profile: dict) -> int:
    """Copy/hardlink May–Jul files from prior mag7_short_dte_old_lock download."""
    legacy = Path("/mnt/s990/data/raw_1s/mag7_short_dte_old_lock")
    dest = profile["_paths"]["quote_1s_root"]
    if not legacy.exists():
        print("no legacy quote dir, skip seed")
        return 0
    n = 0
    for sym in profile["symbols"]:
        sdir = legacy / sym
        if not sdir.exists():
            continue
        for f in sdir.glob(f"{sym}_*.parquet"):
            _link_or_copy(f, dest / sym / f.name)
            n += 1
    print(f"seeded {n} quote day files into {dest}")
    return n


def step_quotes(profile: dict, *, max_workers: int = 12, force: bool = False) -> None:
    paths = profile["_paths"]
    seed_quotes_from_legacy(profile)
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT}/preprocess/download:" + env.get("PYTHONPATH", "")
    if not env.get("MASSIVE_API_KEY") and not env.get("POLYGON_API_KEY"):
        raise SystemExit("set MASSIVE_API_KEY or POLYGON_API_KEY for quote download")
    cmd = [
        py,
        "-u",
        str(ROOT / "preprocess/download/step2_polygon_second_sniper_v1.py"),
        "--target-map",
        str(paths["locked_map"]),
        "--output-dir",
        str(paths["quote_1s_root"]),
        "--stock-output-dir",
        str(paths["stock_1s_root"]),
        "--start-date",
        profile["date_range"]["start"],
        "--end-date",
        profile["date_range"]["end"],
        "--max-workers",
        str(max_workers),
    ]
    if force:
        cmd.append("--force")
    _run(cmd, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--step", choices=["report", "lock", "quotes", "all"], default="all")
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    if args.step in ("report", "all"):
        report_gaps(profile)
    if args.step in ("lock", "all"):
        out = step_lock(profile)
        print("locked_map →", out)
    if args.step in ("quotes", "all"):
        step_quotes(profile, max_workers=args.max_workers, force=args.force)
    print("done")


if __name__ == "__main__":
    main()
