#!/usr/bin/env python3
"""Scan Mag7 for PM 5m extension-fade signals (stock path labels).

Example:
  python -m maga7.tools.scan_pm_fade \\
    --start-date 2026-04-01 --end-date 2026-07-22 \\
    --tag research_pm_fade_apr_jul \\
    --ext-min 0.008 --confirm-minutes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.pm_fade import PmFadeConfig, prepare_day, scan_pm_fade_day
from maga7.common.replay import month_list
from maga7.common.signals import load_stock_month_files

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _bdates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _fwd_ret(day: pd.DataFrame, ts: pd.Timestamp, minutes: int) -> float | None:
    if day is None or day.empty:
        return None
    after = day[day["timestamp"] >= ts]
    if after.empty:
        return None
    px0 = float(after.iloc[0]["close"])
    t1 = ts + pd.Timedelta(minutes=int(minutes))
    w = day[(day["timestamp"] >= ts) & (day["timestamp"] <= t1)]
    if w.empty or px0 <= 0:
        return None
    return float(w.iloc[-1]["close"] / px0 - 1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--tag", default="research_pm_fade")
    ap.add_argument("--ext-min", type=float, default=0.008)
    ap.add_argument("--ext-mins", default="", help="comma grid override, e.g. 0.005,0.008,0.012")
    ap.add_argument("--confirm-minutes", type=int, default=5)
    ap.add_argument("--require-confirm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hold-minutes", type=int, default=15)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    months = month_list(args.start_date, args.end_date)
    print(f"loading 1m {args.start_date}..{args.end_date}", flush=True)
    stock_by = {}
    for sym in symbols:
        raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
        if raw.empty:
            continue
        stock_by[sym] = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)]

    ext_list = [float(args.ext_min)]
    if args.ext_mins.strip():
        ext_list = [float(x) for x in args.ext_mins.split(",") if x.strip()]

    all_events: list[pd.DataFrame] = []
    score_rows = []
    dates = _bdates(args.start_date, args.end_date)
    for ext in ext_list:
        cfg = PmFadeConfig(
            enabled=True,
            ext_min=float(ext),
            confirm_minutes=int(args.confirm_minutes),
            require_confirm=bool(args.require_confirm),
            hold_minutes=int(args.hold_minutes),
        )
        rows = []
        for d in dates:
            ev = scan_pm_fade_day(stock_by, date=d, symbols=symbols, cfg=cfg)
            if ev.empty:
                continue
            # stock forward labels
            labs = []
            for r in ev.itertuples():
                day = prepare_day(stock_by.get(str(r.symbol)), str(r.date))
                fwd = _fwd_ret(day, pd.Timestamp(r.ts), int(args.hold_minutes))
                signed = None if fwd is None else (fwd if r.dir == "UP" else -fwd)
                labs.append(
                    {
                        **{c: getattr(r, c) for c in ev.columns},
                        "fwd_ret": fwd,
                        "fwd_ret_signed": signed,
                        "ext_min": float(ext),
                    }
                )
            rows.extend(labs)
        df = pd.DataFrame(rows)
        if df.empty:
            score_rows.append(
                {
                    "ext_min": ext,
                    "n": 0,
                    "win": None,
                    "exp": None,
                }
            )
            continue
        df["ts"] = pd.to_datetime(df["ts"])
        all_events.append(df)
        signed = pd.to_numeric(df["fwd_ret_signed"], errors="coerce")
        score_rows.append(
            {
                "ext_min": ext,
                "n": int(len(df)),
                "n_days": int(df["date"].nunique()),
                "win": float((signed > 0).mean()),
                "exp": float(signed.mean()),
                "mean_abs_ext": float(df["ext_from_anchor"].abs().mean()),
            }
        )
        print(
            f"ext>={ext:.3f}: n={len(df)} win={score_rows[-1]['win']:.1%} exp={score_rows[-1]['exp']*100:+.3f}%",
            flush=True,
        )

    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if all_events:
        ev = pd.concat(all_events, ignore_index=True)
        ev.to_parquet(out / "events.parquet", index=False)
        ev.to_csv(out / "events.csv", index=False)
    board = pd.DataFrame(score_rows)
    board.to_csv(out / "scoreboard.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "start": args.start_date,
                "end": args.end_date,
                "confirm_minutes": args.confirm_minutes,
                "require_confirm": args.require_confirm,
                "hold_minutes": args.hold_minutes,
                "scoreboard": score_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
