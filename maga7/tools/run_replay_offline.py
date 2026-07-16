#!/usr/bin/env python3
"""Offline Mag7 mf10 Top2 replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def main() -> None:
    p = argparse.ArgumentParser(description="maga7 offline mf10 Top2 replay")
    p.add_argument("--profile", default=None, help="CONFIG json (default mf10_top2_v1.json)")
    p.add_argument(
        "--scheme",
        default="single",
        choices=["single", "m5", "m5_circuit"],
        help="single | m5 reentry | m5 + day circuit",
    )
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--tag", default=None, help="results subfolder tag")
    p.add_argument(
        "--contract-mode",
        default=None,
        choices=["day_lock", "signal_atm", "open_lock", "open_ladder"],
        help="override trade.contract_mode",
    )
    p.add_argument(
        "--quote-source",
        default=None,
        choices=["1s", "day_iv", "auto"],
        help="override trade.quote_source",
    )
    args = p.parse_args()

    profile = load_profile(args.profile)
    if args.start_date:
        profile["date_range"]["start"] = args.start_date
    if args.end_date:
        profile["date_range"]["end"] = args.end_date
    if args.contract_mode:
        profile.setdefault("trade", {})["contract_mode"] = args.contract_mode
    if args.quote_source:
        profile.setdefault("trade", {})["quote_source"] = args.quote_source

    out_dir = Path(profile["_paths"]["results_dir"]) / (args.tag or f"replay_{args.scheme}")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_offline_replay(profile, scheme=args.scheme)
    summary = result["summary"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result["trades"].to_csv(out_dir / "trades.csv", index=False)
    result["daily"].to_csv(out_dir / "daily.csv", index=False)
    result["topk"].to_csv(out_dir / "topk_signals.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
