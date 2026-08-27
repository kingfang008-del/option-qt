#!/usr/bin/env python3
"""Isolated 09:45–10:25 morning sleeve ablation (no 10:30 baseline book).

Completely separate from freeze peer3 10:30+ book:
  - signal window clamped to morning only
  - Watchdog / Hunter OFF (no L1/L2 bleed)
  - optional early_mf5/s6 for faster confirm

Variants (default):
  - morn_mf10_s8_f10       control: same gates, no early_fast, frac=0.10, hold20
  - morn_early_mf5_s6_f10  candidate sleeve
  - morn_early_mf5_s6_f20  same signals, baseline-sized frac=0.20

Research only — does not mutate the frozen baseline profile.
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
from maga7.common.replay import run_offline_replay

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
BASELINE_REF = (
    ROOT
    / "maga7"
    / "results"
    / "research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul"
)


def _apply_morning_isolation(
    profile: dict[str, Any],
    *,
    early: bool,
    mf_fast: int = 5,
    streak_fast: int = 6,
    position_frac: float = 0.10,
    hold_minutes: int = 20,
    hold_extend_minutes: int = 30,
) -> dict[str, Any]:
    p = deepcopy(profile)
    sig = p.setdefault("signal", {})
    sig["window_start"] = "09:45"
    sig["window_end"] = "10:25"
    sig["early_on_mf_fast"] = bool(early)
    if early:
        sig["mf_fast_window"] = int(mf_fast)
        sig["streak_min_fast"] = int(streak_fast)
    else:
        sig["early_on_mf_fast"] = False

    trade = p.setdefault("trade", {})
    trade["position_frac"] = float(position_frac)
    trade["hold_minutes"] = int(hold_minutes)
    trade["hold_extend_minutes"] = int(hold_extend_minutes)

    # Hard isolation from 10:30 stack.
    wd = p.setdefault("watchdog", {})
    wd["enabled"] = False
    wd.setdefault("degrade", {})["enabled"] = False
    wd.setdefault("halt", {})["enabled"] = False
    wd.setdefault("hunter", {})["enabled"] = False
    rr = p.setdefault("regime_router", {})
    rr["enabled"] = False
    return p


def _metrics(summary: dict[str, Any], trades: pd.DataFrame) -> dict[str, Any]:
    entry = None
    if trades is not None and not trades.empty and "entry_ts" in trades.columns:
        ts = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
        local = ts.dt.tz_convert("America/New_York")
        hm = local.dt.hour + local.dt.minute / 60.0
        entry = {
            "n_before_1030": int((hm < 10.5).sum()),
            "n_after_1030": int((hm >= 10.5).sum()),
            "min_entry": str(local.min()),
            "max_entry": str(local.max()),
        }
    return {
        "total_ret": float(summary.get("total_ret") or 0.0),
        "maxdd": float(summary.get("maxdd") or 0.0),
        "n_trades": int(summary.get("n_trades") or 0),
        "n_days": int(summary.get("n_days") or 0),
        "trade_win": float(summary.get("trade_win") or 0.0),
        "day_win": float(summary.get("day_win") or 0.0),
        "trade_exp": float(summary.get("trade_exp") or 0.0),
        "end_equity": float(summary.get("end_equity") or 0.0),
        "n_peer_block": summary.get("n_peer_block"),
        "n_hunt_trades": summary.get("n_hunt_trades"),
        "entry_time": entry,
    }


def _baseline_ref_row() -> dict[str, Any]:
    path = BASELINE_REF / "summary.json"
    if not path.is_file():
        return {"variant": "freeze_10:30_book_REF", "note": "missing"}
    summary = json.loads(path.read_text(encoding="utf-8"))
    trades = pd.read_csv(BASELINE_REF / "trades.csv") if (BASELINE_REF / "trades.csv").is_file() else pd.DataFrame()
    row = {"variant": "freeze_10:30_book_REF", **_metrics(summary, trades)}
    row["note"] = "reference only; not re-run"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE), help="base freeze profile to clone")
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--tag",
        default="research_morn_sleeve_isolated_feb_jul",
        help="results subfolder under maga7/results",
    )
    ap.add_argument(
        "--variants",
        default="morn_mf10_s8_f10,morn_early_mf5_s6_f10,morn_early_mf5_s6_f20",
        help="comma list",
    )
    args = ap.parse_args()

    base = load_profile(args.profile)
    base["date_range"] = {"start": args.start_date, "end": args.end_date}
    out_root = Path(base["_paths"]["results_dir"]) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    catalog: dict[str, dict[str, Any]] = {
        "morn_mf10_s8_f10": {
            "early": False,
            "position_frac": 0.10,
            "hold_minutes": 20,
            "hold_extend_minutes": 30,
        },
        "morn_early_mf5_s6_f10": {
            "early": True,
            "mf_fast": 5,
            "streak_fast": 6,
            "position_frac": 0.10,
            "hold_minutes": 20,
            "hold_extend_minutes": 30,
        },
        "morn_early_mf5_s6_f20": {
            "early": True,
            "mf_fast": 5,
            "streak_fast": 6,
            "position_frac": 0.20,
            "hold_minutes": 20,
            "hold_extend_minutes": 30,
        },
    }
    wanted = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    rows: list[dict[str, Any]] = [_baseline_ref_row()]

    for name in wanted:
        if name not in catalog:
            raise SystemExit(f"unknown variant {name}; choose from {sorted(catalog)}")
        cfg = catalog[name]
        prof = _apply_morning_isolation(base, **cfg)
        prof["profile"] = f"morning_sleeve_{name}"
        prof["result_tag"] = f"{args.tag}_{name}"
        print(f"=== {name} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary = result["summary"]
        trades = result["trades"]
        daily = result["daily"]
        sub = out_root / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        trades.to_csv(sub / "trades.csv", index=False)
        daily.to_csv(sub / "daily.csv", index=False)
        row = {"variant": name, **_metrics(summary, trades)}
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    scoreboard = pd.DataFrame(rows)
    scoreboard.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("\n=== scoreboard ===")
    print(scoreboard.to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
