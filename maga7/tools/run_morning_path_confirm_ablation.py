#!/usr/bin/env python3
"""Ablation: all-weekday entry_confirm vs morning stock path confirm.

Runs causal offline replays (re-priced fills) against the peer3 research profile.
Default window: Feb–Jul (override with --start/--end).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

PATH_CONFIRM = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "tod_start": "10:31",
    "tod_end": "11:00",
    "on_timeout": "block",
}

PATH_CONFIRM_SOFT = {
    **PATH_CONFIRM,
    "on_timeout": "allow",  # only cancel adverse-first; flat path keeps entry
    "delay_on_pos": False,  # veto-only: do not chase +15bp with later fill
}


def _write(out: Path, result: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result["trades"].to_csv(out / "trades.csv", index=False)
    result["daily"].to_csv(out / "daily.csv", index=False)
    if "daily_equity" in result:
        result["daily_equity"].to_csv(out / "daily_equity.csv", index=False)
    elif "equity" in result:
        pass
    # daily_equity may be inside daily already depending on version
    return summary


def _metrics(summary: dict, daily) -> dict:
    import pandas as pd

    d = daily.copy()
    if "day_ret" not in d.columns and "ret" in d.columns:
        d = d.rename(columns={"ret": "day_ret"})
    n_le5 = int((pd.to_numeric(d["day_ret"], errors="coerce") <= -0.05).sum())
    return {
        "n_trades": summary.get("n_trades"),
        "total_ret": summary.get("total_ret"),
        "maxdd": summary.get("maxdd"),
        "day_win": summary.get("day_win"),
        "trade_win": summary.get("trade_win"),
        "n_confirm_block": summary.get("n_confirm_block"),
        "n_stock_path_confirm_block": summary.get("n_stock_path_confirm_block"),
        "n_stock_path_confirm_ok": summary.get("n_stock_path_confirm_ok"),
        "n_day_le_m5": n_le5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument(
        "--variants",
        default="confirm_all_wd,path_confirm_morn",
        help="comma list: confirm_all_wd,path_confirm_morn,combo",
    )
    ap.add_argument("--tag-prefix", default="research_morning_confirm_ablation")
    args = ap.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    results_dir = Path(base["_paths"]["results_dir"])
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    table = []
    for name in variants:
        prof = deepcopy(base)
        trade = prof.setdefault("trade", {})
        if name == "confirm_all_wd":
            trade["entry_confirm_weekdays"] = [0, 1, 2, 3, 4]
            trade.pop("stock_path_confirm", None)
        elif name == "path_confirm_morn":
            # keep Tue/Thu mf confirm; add morning path confirm
            trade["stock_path_confirm"] = dict(PATH_CONFIRM)
        elif name == "path_confirm_soft":
            # cancel only if -30bp first; timeout keeps original fill (no delay)
            trade["stock_path_confirm"] = dict(PATH_CONFIRM_SOFT)
        elif name == "combo":
            trade["entry_confirm_weekdays"] = [0, 1, 2, 3, 4]
            trade["stock_path_confirm"] = dict(PATH_CONFIRM)
        else:
            raise SystemExit(f"unknown variant: {name}")

        tag = f"{args.tag_prefix}_{name}_{args.start_date[5:7]}_{args.end_date[5:7]}"
        out = results_dir / tag
        print(f"=== running {name} → {out} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary = _write(out, result)
        # daily equity path
        daily = result["daily"]
        de = out / "daily_equity.csv"
        if "equity" in daily.columns and not de.exists():
            daily.to_csv(de, index=False)
        m = _metrics(summary, daily)
        m["variant"] = name
        m["out"] = str(out)
        table.append(m)
        print(json.dumps(m, indent=2), flush=True)

    cmp_path = results_dir / f"{args.tag_prefix}_compare.json"
    cmp_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"wrote {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
