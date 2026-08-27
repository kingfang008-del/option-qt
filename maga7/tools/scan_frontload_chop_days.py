#!/usr/bin/env python3
"""Scan causal FRONTLOAD_CHOP days (H1 priced by 10:30).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_frontload_chop_days \\
    --start-date 2026-05-01 --end-date 2026-07-24 \\
    --tag research_frontload_chop_days
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.frontload_chop import (
    FrontloadChopConfig,
    build_frontload_day_table,
    label_frontload_day,
    parse_frontload_chop,
)
from maga7.common.stock_1s import session_dates

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_frontload_chop_days")
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-24")
    ap.add_argument("--min-med-abs-h1", type=float, default=0.008)
    ap.add_argument("--min-name-abs-h1", type=float, default=0.008)
    ap.add_argument("--min-n-large", type=int, default=4)
    ap.add_argument("--max-quiet-abs-1m", type=float, default=0.00085)
    ap.add_argument("--min-decel-ratio", type=float, default=1.85)
    ap.add_argument("--min-med-abs-first", type=float, default=0.006)
    ap.add_argument("--no-require-quiet", action="store_true")
    ap.add_argument("--no-require-decel", action="store_true")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    root = Path(prof["_paths"]["stock_1s_root"])
    symbols = [str(s).upper() for s in (prof.get("symbols") or [])]
    dates = session_dates(args.start_date, args.end_date)
    cfg = FrontloadChopConfig(
        enabled=True,
        min_med_abs_h1=float(args.min_med_abs_h1),
        min_name_abs_h1=float(args.min_name_abs_h1),
        min_n_large=int(args.min_n_large),
        max_quiet_abs_1m=float(args.max_quiet_abs_1m),
        require_quiet=not bool(args.no_require_quiet),
        require_decel=not bool(args.no_require_decel),
        min_decel_ratio=float(args.min_decel_ratio),
        min_med_abs_first=float(args.min_med_abs_first),
    )

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        by_sym: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            raw = load_stock_1s_day(root, sym, date)
            if raw is None or getattr(raw, "empty", True):
                continue
            by_sym[sym] = raw
        if len(by_sym) < 4:
            rows.append(
                {
                    "date": date,
                    "is_frontload": False,
                    "reason": "insufficient_symbols",
                    "n_names": len(by_sym),
                    "n_large": 0,
                    "med_abs_h1": None,
                    "med_quiet_abs_1m": None,
                    "med_abs_first": None,
                    "med_decel_ratio": None,
                    "med_ret_h1": None,
                }
            )
            continue
        lab = label_frontload_day(by_sym, symbols=list(by_sym.keys()), cfg=cfg)
        rows.append({k: v for k, v in lab.items() if k != "names"} | {"date": date})
        for n in lab.get("names") or []:
            detail_rows.append({"date": date, **n})
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{len(dates)}] {date} fl={lab['is_frontload']} "
                f"med_abs={lab.get('med_abs_h1')} n_large={lab.get('n_large')}",
                flush=True,
            )

    day = pd.DataFrame(rows)
    day.to_csv(out / "frontload_days.csv", index=False)
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(out / "frontload_symbol_h1.csv", index=False)

    fl = day[day["is_frontload"] == True]  # noqa: E712
    summary = {
        "cfg": cfg.__dict__,
        "n_days": int(len(day)),
        "n_frontload": int(len(fl)),
        "frac_frontload": float(len(fl) / max(len(day), 1)),
        "frontload_dates": fl["date"].astype(str).tolist(),
        "week_0720_24": {
            d: bool(day.loc[day.date.astype(str) == d, "is_frontload"].iloc[0])
            if (day.date.astype(str) == d).any()
            else None
            for d in ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24"]
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
