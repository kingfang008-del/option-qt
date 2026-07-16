#!/usr/bin/env python3
"""Build causal open-window multi-DTE lock map from day_iv (09:30)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.open_lock import build_open_lock_map
from maga7.common.signals import load_stock_month_files
from maga7.common.replay import month_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json"),
    )
    ap.add_argument(
        "--output",
        default=str(Path.home() / "train_data/locked_targets_map_maga7_open_multidte_jan_jul.parquet"),
    )
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument(
        "--otm-rungs",
        type=int,
        default=None,
        help="ATM + N OTM rungs per side (default 1=classic4; 2=ATM+OTM1+OTM2 ladder)",
    )
    args = ap.parse_args()

    profile = load_profile(args.profile)
    start = args.start_date or profile["date_range"]["start"]
    end = args.end_date or profile["date_range"]["end"]
    otm_rungs = args.otm_rungs
    if otm_rungs is None:
        otm_rungs = int(
            (profile.get("trade") or {}).get("ladder_otm_rungs")
            or (profile.get("lock") or {}).get("otm_rungs")
            or 1
        )
    months = month_list(start, end)
    stock_by = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        stock_by[sym] = raw

    df = build_open_lock_map(
        day_iv_root=profile["_paths"]["day_iv_root"],
        symbols=list(profile["symbols"]),
        start=start,
        end=end,
        allowed_dte=profile.get("lock", {}).get("allowed_dte") or [0, 1, 2],
        stock_by=stock_by,
        option_1m_root=profile["_paths"].get("option_1m_root"),
        otm_rungs=otm_rungs,
    )
    if df.empty:
        raise SystemExit("empty open lock map")
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "n_rows": int(len(df)),
        "n_day_symbol": int(df.groupby(["date_str", "symbol"]).ngroups),
        "n_dates": int(df["date_str"].nunique()),
        "otm_rungs": int(otm_rungs),
        "dte_counts": df["front_dte"].value_counts().sort_index().to_dict(),
        "bucket_counts": df["bucket_id"].value_counts().sort_index().to_dict(),
        "output": str(out),
        "start": start,
        "end": end,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
