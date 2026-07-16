#!/usr/bin/env python3
"""Shadow Mag7 scanner: replay one (or more) days of 1m bars → signal audit.

Does not place orders. Writes JSONL+CSV under maga7/results/scanner_shadow/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list
from maga7.common.signals import load_stock_month_files
from maga7.live.scanner import Mag7Scanner, write_signal_audit


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 scanner shadow (no OMS)")
    p.add_argument("--profile", default=None)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--out", default=None, help="audit jsonl path")
    args = p.parse_args()

    profile = load_profile(args.profile)
    end = args.end_date or args.start_date
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = end

    scanner = Mag7Scanner.from_profile(profile)
    months = month_list(args.start_date, end)
    frames = []
    for sym in profile["symbols"]:
        raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= args.start_date) & (raw["date"] <= end)].copy()
        raw["symbol"] = sym
        frames.append(raw)
    if not frames:
        raise SystemExit("no stock bars in range")
    import pandas as pd

    all_bars = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    for r in all_bars.itertuples(index=False):
        scanner.on_stock_bar(
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

    out_dir = Path(profile["_paths"]["results_dir"]) / "scanner_shadow"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"signals_{args.start_date}_{end}.jsonl"
    write_signal_audit(scanner.signals, out)
    print(f"signals={len(scanner.signals)} → {out} / {out.with_suffix('.csv')}")
    for s in scanner.signals[:20]:
        print(
            f"  {s.date} rank={s.rank} {s.symbol} {s.direction} "
            f"{s.sig_ts.strftime('%H:%M')} contract={s.contract}"
        )


if __name__ == "__main__":
    main()
