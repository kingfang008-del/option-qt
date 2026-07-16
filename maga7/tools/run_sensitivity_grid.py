#!/usr/bin/env python3
"""Rule-A sensitivity grid: streak / from_prev / vol_z on Mag7 offline replay.

Loads stock features once, then sweeps thresholds (single scheme by default).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import attach_mf_features, load_stock_month_files


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_stock_by(profile: dict) -> dict[str, pd.DataFrame]:
    paths = profile["_paths"]
    sig = profile["signal"]
    start, end = profile["date_range"]["start"], profile["date_range"]["end"]
    months = month_list(start, end)
    out = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        out[sym] = attach_mf_features(
            raw,
            mf_window=int(sig.get("mf_window", 10)),
            vol_ma_window=int(sig.get("vol_ma_window", 20)),
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 Rule-A sensitivity grid")
    p.add_argument("--profile", default=None)
    p.add_argument("--start-date", default="2026-01-02")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--scheme", default="single", choices=["single", "m5", "m5_circuit"])
    p.add_argument("--streak", default="6,8,10")
    p.add_argument("--from-prev", default="0.015,0.02,0.025")
    p.add_argument("--vol-z", default="0.5,1.0,1.5")
    p.add_argument("--tag", default="sensitivity_jan_jul")
    args = p.parse_args()

    profile = load_profile(args.profile)
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = args.end_date

    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading stock features once…")
    stock_by = load_stock_by(profile)
    print(f"symbols={list(stock_by)} bars={sum(len(v) for v in stock_by.values())}")

    streaks = _parse_ints(args.streak)
    fps = _parse_floats(args.from_prev)
    vzs = _parse_floats(args.vol_z)

    rows = []
    baseline = None
    for streak, fp, vz in itertools.product(streaks, fps, vzs):
        cfg = copy.deepcopy(profile)
        cfg["signal"]["streak_min"] = streak
        cfg["signal"]["from_prev_abs"] = fp
        cfg["signal"]["vol_z_min"] = vz
        result = run_offline_replay(cfg, scheme=args.scheme, stock_by=stock_by)
        s = result["summary"]
        row = {
            "streak_min": streak,
            "from_prev_abs": fp,
            "vol_z_min": vz,
            "is_baseline": streak == 8 and abs(fp - 0.02) < 1e-12 and abs(vz - 1.0) < 1e-12,
            **s,
        }
        rows.append(row)
        mark = " ★" if row["is_baseline"] else ""
        print(
            f"streak={streak} fp={fp:.3f} vz={vz:.1f} → "
            f"ret={s['total_ret']*100:+.0f}% dd={s['maxdd']*100:.1f}% "
            f"n={s['n_trades']} win={s['trade_win']*100:.0f}% exp={s['trade_exp']*100:+.1f}%{mark}"
        )
        if row["is_baseline"]:
            baseline = row

    df = pd.DataFrame(rows).sort_values(["total_ret", "maxdd"], ascending=[False, False])
    df.to_csv(out_dir / "scoreboard.csv", index=False)
    (out_dir / "scoreboard.json").write_text(
        json.dumps({"baseline": baseline, "rows": rows}, indent=2), encoding="utf-8"
    )

    # score: ret / |dd| with floor, favor baseline neighborhood
    scored = df.copy()
    scored["calmar_like"] = scored["total_ret"] / scored["maxdd"].abs().clip(lower=0.05)
    best = scored.sort_values("calmar_like", ascending=False).iloc[0]
    print("\n=== top by calmar-like (ret/|dd|) ===")
    print(
        f"streak={int(best.streak_min)} fp={best.from_prev_abs:.3f} vz={best.vol_z_min:.1f} "
        f"ret={best.total_ret*100:+.0f}% dd={best.maxdd*100:.1f}% calmar={best.calmar_like:.2f}"
    )
    if baseline:
        print(
            f"baseline streak=8 fp=0.02 vz=1.0 "
            f"ret={baseline['total_ret']*100:+.0f}% dd={baseline['maxdd']*100:.1f}%"
        )
    print(f"wrote {out_dir / 'scoreboard.csv'}")


if __name__ == "__main__":
    main()
