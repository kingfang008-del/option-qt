#!/usr/bin/env python3
"""Ablate signal.mf_window on Mag7+GOOGL peer3 causal baseline.

Keeps streak_min / rails / peer3 / delay=60 fixed; only changes mf rolling length.
Note: column is still named mf10; peer_align_mode=mf10 reads that same series.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import attach_mf_features, load_stock_month_files

PEER3 = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)


def _load_raw(profile: dict[str, Any]) -> dict[str, pd.DataFrame]:
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    out: dict[str, pd.DataFrame] = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
        if raw.empty:
            continue
        out[sym] = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--windows", default="5,6,8,10")
    p.add_argument("--tag", default="mf_window_ablation_peer3_may_jul")
    p.add_argument(
        "--exit-mode",
        default="none",
        help="none (hard T30) or hold_extend (use profile hold_extend_* keys)",
    )
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date
    base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
    if args.exit_mode:
        base["trade"]["exit_mode"] = args.exit_mode

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    raw_by = _load_raw(base)
    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    scoreboard: list[dict[str, Any]] = []
    for w in windows:
        prof = deepcopy(base)
        prof["signal"]["mf_window"] = w
        stock_by = {
            sym: attach_mf_features(
                raw,
                mf_window=w,
                vol_ma_window=int(prof["signal"].get("vol_ma_window", 20)),
                mf_confirm_bars=int(prof["signal"].get("mf_confirm_bars", 3)),
            )
            for sym, raw in raw_by.items()
        }
        name = f"mf{w}"
        print(f"==> {name}", flush=True)
        result = run_offline_replay(prof, scheme="single", stock_by=stock_by)
        s = result["summary"]
        row = {
            "name": name,
            "mf_window": w,
            "streak_min": int(prof["signal"].get("streak_min", 8)),
            "exit_mode": s.get("exit_mode"),
            "total_ret": float(s["total_ret"]),
            "maxdd": float(s["maxdd"]),
            "n_trades": int(s["n_trades"]),
            "trade_win": float(s["trade_win"]),
            "trade_exp": float(s["trade_exp"]),
            "n_peer_block": s.get("n_peer_block"),
            "n_regime_block": s.get("n_regime_block"),
            "n_signals_topk": s.get("n_signals_topk"),
        }
        scoreboard.append(row)
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        result["trades"].to_csv(sub / "trades.csv", index=False)
        result["daily"].to_csv(sub / "daily.csv", index=False)
        print(
            f"    ret={row['total_ret']:+.1%} dd={row['maxdd']:.1%} "
            f"n={row['n_trades']} win={row['trade_win']:.1%} "
            f"topk={row['n_signals_topk']} peer_block={row['n_peer_block']}",
            flush=True,
        )

    base_ret = next(r["total_ret"] for r in scoreboard if r["mf_window"] == 10)
    for row in scoreboard:
        row["uplift_vs_mf10_pp"] = float(row["total_ret"] - base_ret)

    pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    best = max(scoreboard, key=lambda r: r["total_ret"])
    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "delay_seconds": 60,
        "profile": str(args.profile),
        "exit_mode": args.exit_mode,
        "streak_min": int(base["signal"].get("streak_min", 8)),
        "baseline_mf10_ret": base_ret,
        "best_by_ret": best,
        "note": "Only mf_window varies; streak_min and peer3 rails unchanged.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
