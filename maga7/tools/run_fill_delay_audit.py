#!/usr/bin/env python3
"""Execution audit: entry_frac × bar_availability_delay on freeze stack.

Does **not** mutate freeze. Answers: how much of the baseline edge survives
worse fills / faster (more optimistic) bar delay.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    # keep router off for pure execution audit
    rr = dict(p.get("regime_router") or {})
    rr["enabled"] = False
    p["regime_router"] = rr
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    d0717 = None
    hit = res["daily"][res["daily"]["date"].astype(str) == "2026-07-17"]
    if len(hit):
        d0717 = float(hit.iloc[0]["day_ret"])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "avg_trade_ret": s.get("avg_trade_ret") or s.get("mean_ret"),
        "day_ret_0717": d0717,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--entry-fracs", default="0.6,0.8,1.0")
    ap.add_argument("--delays", default="30,60")
    ap.add_argument(
        "--exit-frac-mode",
        choices=["match_entry", "freeze"],
        default="match_entry",
        help="match_entry: exit_frac=entry_frac; freeze: keep profile exit_frac",
    )
    ap.add_argument("--out", default="maga7/results/exec_audit/fill_delay_grid")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    entry_fracs = [float(x) for x in str(args.entry_fracs).split(",") if x.strip()]
    delays = [int(x) for x in str(args.delays).split(",") if x.strip()]
    freeze_entry = float((base.get("fill") or {}).get("entry_frac", 0.8))
    freeze_exit = float((base.get("fill") or {}).get("exit_frac", 0.8))
    freeze_delay = int((base.get("trade") or {}).get("bar_availability_delay_seconds", 60) or 60)

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    rows: list[dict] = []

    for ef in entry_fracs:
        for delay in delays:
            xf = float(ef) if args.exit_frac_mode == "match_entry" else freeze_exit
            prof = copy.deepcopy(base)
            prof.setdefault("fill", {})["entry_frac"] = float(ef)
            prof.setdefault("fill", {})["exit_frac"] = float(xf)
            prof.setdefault("trade", {})["bar_availability_delay_seconds"] = int(delay)
            vname = f"ef{ef:g}_xf{xf:g}_d{delay}"
            is_freeze = (
                abs(ef - freeze_entry) < 1e-12
                and abs(xf - freeze_exit) < 1e-12
                and int(delay) == int(freeze_delay)
            )
            for wname, start, end in windows:
                tag = f"{vname}__{wname}"
                print(f"=== {tag} ===", flush=True)
                r = _run(prof, start=start, end=end, tag=tag, out=out)
                r.update(
                    {
                        "variant": vname,
                        "window": wname,
                        "entry_frac": float(ef),
                        "exit_frac": float(xf),
                        "delay_sec": int(delay),
                        "is_freeze_point": bool(is_freeze),
                    }
                )
                rows.append(r)
                print(
                    f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                    f"n={r['n_trades']} win={r['trade_win']} 0717={r['day_ret_0717']}",
                    flush=True,
                )

    board = pd.DataFrame(rows)
    # vs freeze point within each window
    freeze_ret = {
        r["window"]: r["total_ret"]
        for r in rows
        if r.get("is_freeze_point")
    }
    board["vs_freeze"] = board.apply(
        lambda r: (r["total_ret"] / freeze_ret[r["window"]])
        if freeze_ret.get(r["window"]) not in (None, 0)
        else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    # pivot convenience
    piv = board.pivot_table(
        index=["entry_frac", "delay_sec"],
        columns="window",
        values=["total_ret", "maxdd", "vs_freeze"],
        aggfunc="first",
    )
    piv.to_csv(out / "scoreboard_pivot.csv")

    summary = {
        "profile": args.profile,
        "freeze_point": {
            "entry_frac": freeze_entry,
            "exit_frac": freeze_exit,
            "delay_sec": freeze_delay,
        },
        "exit_frac_mode": args.exit_frac_mode,
        "grid": {"entry_fracs": entry_fracs, "delays": delays},
        "windows": windows,
        "scoreboard": board.to_dict(orient="records"),
        "note": "Execution robustness audit; freeze profile untouched.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    cols = [
        "entry_frac",
        "exit_frac",
        "delay_sec",
        "window",
        "total_ret",
        "maxdd",
        "n_trades",
        "trade_win",
        "day_ret_0717",
        "vs_freeze",
        "is_freeze_point",
    ]
    print(board[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
