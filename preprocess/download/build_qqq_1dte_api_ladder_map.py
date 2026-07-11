#!/usr/bin/env python3
"""Build QQQ 1DTE ATM strike ladder map (aligned with 0DTE api_ladder).

Same selection rule as 0DTE / MAG7 short-DTE ladders:
  - resolve expiry with trading_dte == 1
  - lock n_per_side PUT (<= spot) + n_per_side CALL (>= spot) near ATM
  - NO next-month buckets

Default: 4 per side → 8 contracts/day, all selected_dte=1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess.download.build_mag7_short_dte_api_ladder_map import (
    API_KEY,
    main as mag7_main,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="QQQ")
    p.add_argument("--start-date", default="2026-01-02")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--dtes", default="1")
    p.add_argument("--n-per-side", type=int, default=4)
    p.add_argument("--lock-minute", default="09:40")
    p.add_argument(
        "--output",
        default=str(Path.home() / "train_data/locked_targets_map_1dte_api_ladder.parquet"),
    )
    p.add_argument(
        "--report",
        default="qqq_btc/results/locked_targets_map_1dte_api_ladder_report.json",
    )
    p.add_argument("--api-key", default=API_KEY)
    return p.parse_args()


if __name__ == "__main__":
    # Reuse MAG7 builder CLI by injecting argv-compatible namespace via sys.argv rewrite.
    import sys

    args = parse_args()
    sys.argv = [
        sys.argv[0],
        "--symbols",
        args.symbols,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--dtes",
        args.dtes,
        "--n-per-side",
        str(args.n_per_side),
        "--lock-minute",
        args.lock_minute,
        "--output",
        args.output,
        "--report",
        args.report,
        "--api-key",
        args.api_key or "",
    ]
    mag7_main()
