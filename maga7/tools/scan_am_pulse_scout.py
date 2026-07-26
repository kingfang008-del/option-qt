#!/usr/bin/env python3
"""Offline AM pulse scout (alert-only, 09:30–10:30).

No fills / no OMS. Writes ``AM_SCOUT_ALERT`` rows for Mag7 opening impulses.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pulse_scout \\
    --start-date 2026-07-24 --end-date 2026-07-24 \\
    --tag am_pulse_scout_20260724
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

from maga7.common.am_pulse_scout import parse_am_pulse_scout, scan_day
from maga7.common.config import load_profile
from maga7.common.replay import month_list
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_SYMBOLS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--tag", default="am_pulse_scout")
    ap.add_argument("--min-fav-from-open", type=float, default=0.01)
    ap.add_argument("--lookback-bars", type=int, default=2)
    ap.add_argument("--min-lookback-ret", type=float, default=0.008)
    ap.add_argument("--dirs", default="DN,UP")
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cfg = parse_am_pulse_scout(
        {
            "enabled": True,
            "window_start": "09:30",
            "window_end": "10:30",
            "min_fav_from_open": args.min_fav_from_open,
            "lookback_bars": args.lookback_bars,
            "min_lookback_ret": args.min_lookback_ret,
            "dirs": [x.strip().upper() for x in args.dirs.split(",") if x.strip()],
            "symbols": symbols,
            "max_alerts_per_symbol": 1,
        }
    )
    months = month_list(args.start_date, args.end_date)
    stock_root = profile["_paths"]["stock_root"]
    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date
    ]

    alerts: list[dict[str, Any]] = []
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is None or sdf.empty:
            continue
        for date in dates:
            if "date" in sdf.columns:
                day = sdf[sdf["date"].astype(str) == date]
            else:
                ts = pd.to_datetime(sdf["timestamp"])
                day = sdf[ts.dt.strftime("%Y-%m-%d") == date]
            for a in scan_day(day, date=date, symbol=sym, cfg=cfg):
                alerts.append(a.to_dict())

    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(alerts)
    if len(df):
        df = df.sort_values(["date", "ts", "symbol"]).reset_index(drop=True)
        df.to_csv(out / "alerts.csv", index=False)
        (out / "alerts.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in df.to_dict(orient="records"))
            + "\n",
            encoding="utf-8",
        )
    else:
        (out / "alerts.csv").write_text("", encoding="utf-8")
        (out / "alerts.jsonl").write_text("", encoding="utf-8")

    by_day = (
        df.groupby("date").size().to_dict() if len(df) else {}
    )
    summary = {
        "tag": args.tag,
        "start": args.start_date,
        "end": args.end_date,
        "symbols": symbols,
        "n_alerts": int(len(df)),
        "n_days_with_alert": int(len(by_day)),
        "by_day": {str(k): int(v) for k, v in by_day.items()},
        "by_arm": df["arm"].value_counts().to_dict() if len(df) else {},
        "by_dir": df["dir"].value_counts().to_dict() if len(df) else {},
        "cfg": {
            "window": f"{cfg.window_start}-{cfg.window_end}",
            "min_fav_from_open": cfg.min_fav_from_open,
            "lookback_bars": cfg.lookback_bars,
            "min_lookback_ret": cfg.min_lookback_ret,
            "dirs": list(cfg.dirs),
        },
        "note": "ALERT_ONLY — no OMS / no fills",
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if len(df):
        cols = [c for c in ["date", "ts", "symbol", "dir", "arm", "fav_from_open", "chase", "lookback_ret"] if c in df.columns]
        print(df[cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
