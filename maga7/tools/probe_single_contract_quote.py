#!/usr/bin/env python3
"""Re-download one OCC contract quote with 09:30 window; compare to existing day file.

Proves whether morning NBBO exists upstream (Polygon/Massive) vs local window crop.

Example:
  export MASSIVE_API_KEY=...   # or POLYGON_API_KEY
  /home/kingfang007/anaconda3/envs/ibkr/bin/python -m maga7.tools.probe_single_contract_quote \\
      --ticker AMZN260206P00222500 --date 2026-02-05 \\
      --window-start 09:30 --window-end 11:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_LOCK = Path.home() / "train_data/locked_targets_map_maga7_googl_open_ladder_atm5otm_jan_jul.parquet"
DEFAULT_EXISTING = Path("/mnt/s990/data/raw_1s/maga7_mf10_open_ladder_otm5")
IBKR_PY = Path("/home/kingfang007/anaconda3/envs/ibkr/bin/python")


def _morning_stats(df: pd.DataFrame, ticker: str) -> dict:
    if df is None or df.empty or "ticker" not in df.columns:
        return {"n": 0, "morn_0930_1000": 0, "first": None}
    sub = df[df["ticker"].astype(str).str.replace("O:", "", regex=False) == ticker.replace("O:", "")]
    if sub.empty:
        return {"n": 0, "morn_0930_1000": 0, "first": None}
    ts = pd.to_datetime(sub["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    t = ts.dt.time
    morn = int(((t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("10:00").time())).sum())
    return {"n": int(len(sub)), "morn_0930_1000": morn, "first": str(ts.min())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="AMZN260206P00222500")
    ap.add_argument("--date", default="2026-02-05")
    ap.add_argument("--symbol", default="AMZN")
    ap.add_argument("--lock-map", default=str(DEFAULT_LOCK))
    ap.add_argument("--existing-root", default=str(DEFAULT_EXISTING))
    ap.add_argument("--out-dir", default="/tmp/maga7_quote_probe_0930")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:00")
    ap.add_argument("--skip-download", action="store_true", help="only compare existing roots")
    args = ap.parse_args()

    ticker = str(args.ticker).replace("O:", "")
    occ = f"O:{ticker}"
    date = str(args.date)
    symbol = str(args.symbol).upper()

    # Existing baseline file
    existing_path = Path(args.existing_root) / symbol / f"{symbol}_{date}.parquet"
    existing = pd.read_parquet(existing_path) if existing_path.is_file() else pd.DataFrame()
    before = _morning_stats(existing, ticker)
    print("existing:", json.dumps({"path": str(existing_path), **before}, indent=2))

    # Sibling ATM on same day (often seeded with 09:30)
    if not existing.empty:
        print("same-file siblings (first/morn):")
        for tkr in sorted(existing["ticker"].astype(str).unique()):
            st = _morning_stats(existing, tkr)
            if st["n"]:
                print(f"  {tkr}: first={st['first']} morn0930={st['morn_0930_1000']} n={st['n']}")

    if args.skip_download:
        return

    if not (os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_KEY")):
        raise SystemExit("请先: export MASSIVE_API_KEY=... 或 POLYGON_API_KEY=...")

    lock = pd.read_parquet(args.lock_map)
    row = lock[
        (lock["symbol"].astype(str) == symbol)
        & (lock["date_str"].astype(str) == date)
        & (lock["contract_symbol"].astype(str).str.replace("O:", "") == ticker)
    ].copy()
    if row.empty:
        # synthesize minimal row
        row = pd.DataFrame(
            [
                {
                    "bucket_id": 1,
                    "contract_symbol": occ,
                    "date_str": date,
                    "dte_mode": "trading",
                    "front_dte": 1,
                    "ladder_rung": 1,
                    "lock_spot": float("nan"),
                    "lock_ts": f"{date} 09:30:00",
                    "strike": float(ticker[-8:]) / 1000.0,
                    "symbol": symbol,
                    "tag": "probe_single",
                }
            ]
        )
    map_path = Path("/tmp") / f"maga7_probe_{symbol}_{date}_{ticker}.parquet"
    row.to_parquet(map_path, index=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = str(IBKR_PY if IBKR_PY.is_file() else sys.executable)
    sniper = ROOT / "preprocess/download/step2_polygon_second_sniper_v1.py"
    cmd = [
        py,
        "-u",
        str(sniper),
        "--target-map",
        str(map_path),
        "--output-dir",
        str(out_dir),
        "--start-date",
        date,
        "--end-date",
        date,
        "--symbols",
        symbol,
        "--window-start",
        str(args.window_start),
        "--window-end",
        str(args.window_end),
        "--no-download-stock",
        "--force",
        "--allow-partial",
        "--max-workers",
        "1",
        "--contract-workers",
        "1",
    ]
    print("running:", " ".join(cmd), flush=True)
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)

    probe_path = out_dir / symbol / f"{symbol}_{date}.parquet"
    probe = pd.read_parquet(probe_path) if probe_path.is_file() else pd.DataFrame()
    after = _morning_stats(probe, ticker)
    print("probe:", json.dumps({"path": str(probe_path), **after}, indent=2))
    print(
        json.dumps(
            {
                "ticker": ticker,
                "date": date,
                "existing_morn_0930_1000": before["morn_0930_1000"],
                "probe_morn_0930_1000": after["morn_0930_1000"],
                "verdict": (
                    "UPSTREAM_HAS_MORNING_NBBO"
                    if after["morn_0930_1000"] > 0
                    else (
                        "UPSTREAM_EMPTY_OR_ILLIQUID"
                        if after["n"] > 0
                        else "DOWNLOAD_FAILED_OR_EMPTY"
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
