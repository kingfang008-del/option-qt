#!/usr/bin/env python3
"""S2 shadow: drive Mag7 scanner from stock **1s** → causal 1m agg → Rule-A audit.

Does not place orders. Prefer this over raw 1m feeds for live-aligned ingest;
Rule-A still evaluates on completed 1m bars only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.live.scanner import Mag7Scanner, write_signal_audit


def _dates(start: str, end: str) -> list[str]:
    days = pd.bdate_range(start, end)
    return [d.strftime("%Y-%m-%d") for d in days]


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 scanner from stock 1s (S2 ingest path)")
    p.add_argument("--profile", default=None)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--stock-1s-root", default=None, help="override paths.stock_1s_root")
    p.add_argument("--out", default=None, help="audit jsonl path")
    args = p.parse_args()

    profile = load_profile(args.profile)
    end = args.end_date or args.start_date
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = end

    stock_1s = Path(args.stock_1s_root) if args.stock_1s_root else profile["_paths"]["stock_1s_root"]
    scanner = Mag7Scanner.from_profile(profile)

    frames = []
    missing = []
    for date in _dates(args.start_date, end):
        for sym in profile["symbols"]:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw.empty:
                missing.append(f"{sym}:{date}")
                continue
            raw = raw.copy()
            raw["symbol"] = sym
            frames.append(raw)

    if not frames:
        raise SystemExit(f"no stock 1s under {stock_1s} for {args.start_date}..{end}")

    all_ticks = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    for r in all_ticks.itertuples(index=False):
        scanner.on_stock_second(
            r.symbol,
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            },
        )
    scanner.flush_seconds()

    out_dir = Path(profile["_paths"]["results_dir"]) / "scanner_shadow_1s"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"signals_{args.start_date}_{end}.jsonl"
    write_signal_audit(scanner.signals, out)

    print(f"ticks={len(all_ticks)} signals={len(scanner.signals)} → {out}")
    if missing:
        print(f"missing_days={len(missing)} (sample {missing[:8]})")
    for s in scanner.signals[:30]:
        print(
            f"  {s.date} rank={s.rank} {s.symbol} {s.direction} "
            f"{s.sig_ts.strftime('%H:%M')} contract={s.contract}"
        )


if __name__ == "__main__":
    main()
