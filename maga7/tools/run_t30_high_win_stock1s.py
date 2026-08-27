#!/usr/bin/env python3
"""Hunt high trade_win Rule-A variants on stock 1s (short hold / short window).

Question: can we reach ~80% trade_win on dual windows by shortening hold,
narrowing the entry clock, and/or allowing more reentries — without killing EV?

Example:
  PYTHONPATH=. python -m maga7.tools.run_t30_high_win_stock1s \\
    --tag research_t30_high_win_stock1s_dual
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import build_stock_by_from_1s, session_dates
from maga7.tools.run_t30_sensitivity_stock1s import WINDOWS, _slice_stock_by

T30 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=T30)
    ap.add_argument("--tag", default="research_t30_high_win_stock1s_dual")
    ap.add_argument("--target-win", type=float, default=0.80)
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    profile["trade"]["exit_mode"] = "none"
    profile["trade"]["bar_availability_delay_seconds"] = 60

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    profile["date_range"]["start"] = start_all
    profile["date_range"]["end"] = end_all

    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start_all, end_all)
    print(f"building stock_by 1s {start_all}..{end_all} …", flush=True)
    stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
    print(f"bars={sum(len(v) for v in stock_by.values())}", flush=True)

    # Compact structure grid (peer3 + fp2%/vz1 fixed). ~72 cells.
    holds = [10, 15, 20]
    windows = [
        ("1030_1200", "10:30", "12:00"),
        ("1030_1400", "10:30", "14:00"),
    ]
    streaks = [6, 8]
    schemes = ["single", "m5"]
    books = [
        ("tp16_sl04", 1.6, 0.4),  # baseline book
        ("tp12_sl06", 1.2, 0.6),  # tighter TP
        ("tp11_sl07", 1.1, 0.7),  # win-rate biased
    ]

    rows: list[dict[str, Any]] = []
    cells = list(itertools.product(holds, windows, streaks, schemes, books))
    print(f"cells={len(cells)}", flush=True)

    for i, (hold, (wlabel, w0, w1), streak, scheme, (bname, tp, sl)) in enumerate(
        cells, 1
    ):
        win_stats: dict[str, dict[str, Any]] = {}
        for win_name, d0, d1 in WINDOWS:
            cfg = copy.deepcopy(profile)
            cfg["date_range"]["start"] = d0
            cfg["date_range"]["end"] = d1
            cfg["signal"]["window_start"] = w0
            cfg["signal"]["window_end"] = w1
            cfg["signal"]["streak_min"] = int(streak)
            # keep baseline fp/vz/peer — structure ablation first
            cfg["signal"]["from_prev_abs"] = 0.02
            cfg["signal"]["vol_z_min"] = 1.0
            cfg["signal"]["peer_align_min"] = 3
            cfg["trade"]["hold_minutes"] = int(hold)
            cfg["trade"]["tp_mult"] = float(tp)
            cfg["trade"]["sl_mult"] = float(sl)
            cfg["trade"]["exit_mode"] = "none"
            if scheme == "m5":
                cfg["trade"]["max_entries_per_symbol"] = 5
                cfg["trade"]["cooldown_minutes"] = 5
            else:
                cfg["trade"]["max_entries_per_symbol"] = 1
            sb = _slice_stock_by(stock_by, d0, d1)
            s = run_offline_replay(cfg, scheme=scheme, stock_by=sb)["summary"]
            win_stats[win_name] = s

        def _g(w: str, k: str, default=None):
            return win_stats[w].get(k, default)

        jm_win = float(_g("jan_mar", "trade_win") or 0)
        mj_win = float(_g("may_jul", "trade_win") or 0)
        jm_n = int(_g("jan_mar", "n_trades") or 0)
        mj_n = int(_g("may_jul", "n_trades") or 0)
        jm_ret = float(_g("jan_mar", "total_ret") or 0)
        mj_ret = float(_g("may_jul", "total_ret") or 0)
        jm_exp = float(_g("jan_mar", "trade_exp") or 0)
        mj_exp = float(_g("may_jul", "trade_exp") or 0)
        hit80 = (
            jm_win >= float(args.target_win)
            and mj_win >= float(args.target_win)
            and jm_n >= int(args.min_n)
            and mj_n >= int(args.min_n)
        )
        dual_ev = jm_exp > 0 and mj_exp > 0 and jm_ret > 0 and mj_ret > 0
        row = {
            "hold_minutes": hold,
            "entry_window": wlabel,
            "window_start": w0,
            "window_end": w1,
            "streak_min": streak,
            "scheme": scheme,
            "book": bname,
            "tp_mult": tp,
            "sl_mult": sl,
            "jan_mar_n": jm_n,
            "jan_mar_win": jm_win,
            "jan_mar_ret": jm_ret,
            "jan_mar_exp": jm_exp,
            "jan_mar_maxdd": _g("jan_mar", "maxdd"),
            "may_jul_n": mj_n,
            "may_jul_win": mj_win,
            "may_jul_ret": mj_ret,
            "may_jul_exp": mj_exp,
            "may_jul_maxdd": _g("may_jul", "maxdd"),
            "min_win": min(jm_win, mj_win),
            "sum_n": jm_n + mj_n,
            "hit80": hit80,
            "dual_ev_pos": dual_ev,
        }
        rows.append(row)
        mark = " *** HIT80" if hit80 else ""
        if dual_ev:
            mark += " EV+"
        print(
            f"[{i}/{len(cells)}] H{hold} {wlabel} s{streak} {scheme} {bname} "
            f"JM n={jm_n} win={jm_win:.0%} ret={jm_ret*100:+.0f}% | "
            f"MJ n={mj_n} win={mj_win:.0%} ret={mj_ret*100:+.0f}%{mark}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "scoreboard.csv", index=False)

    hit = df[df["hit80"]].sort_values(["dual_ev_pos", "min_win", "sum_n"], ascending=[False, False, False])
    near = df.sort_values(["min_win", "dual_ev_pos", "sum_n"], ascending=[False, False, False])

    # also: best win among EV+ dual
    ev = df[df["dual_ev_pos"]].sort_values("min_win", ascending=False)

    summary = {
        "target_win": float(args.target_win),
        "min_n": int(args.min_n),
        "n_cells": int(len(df)),
        "n_hit80": int(len(hit)),
        "n_hit80_ev_pos": int(len(hit[hit["dual_ev_pos"]])) if len(hit) else 0,
        "verdict": (
            "PASS_80_EV"
            if len(hit[hit["dual_ev_pos"]]) > 0
            else ("PASS_80_NEG_EV" if len(hit) else "NO_80")
        ),
        "hit80": hit.head(20).to_dict(orient="records"),
        "best_ev_by_min_win": ev.head(15).to_dict(orient="records"),
        "closest_by_min_win": near.head(20).to_dict(orient="records"),
        "note": (
            "fp=2% vz=1 peer3 fixed. Vary hold/window/streak/scheme/TP-SL. "
            "80% win with negative EV is rejected for promotion."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== hit80={len(hit)} (ev+={summary['n_hit80_ev_pos']}) verdict={summary['verdict']} ===", flush=True)
    if len(hit):
        print(hit.head(10).to_string(index=False), flush=True)
    print("\n=== best EV+ by min_win ===", flush=True)
    cols = [
        "hold_minutes",
        "entry_window",
        "streak_min",
        "scheme",
        "book",
        "min_win",
        "jan_mar_n",
        "jan_mar_win",
        "jan_mar_exp",
        "may_jul_n",
        "may_jul_win",
        "may_jul_exp",
        "jan_mar_ret",
        "may_jul_ret",
    ]
    if len(ev):
        print(ev[cols].head(15).to_string(index=False), flush=True)
    print("\n=== closest overall by min_win ===", flush=True)
    print(near[cols].head(15).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
