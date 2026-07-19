#!/usr/bin/env python3
"""Dual-window ablation: trade-mark toxic early cut on L2+TT1_05+sl55.

Signal on option trade last (MFE bypass + cut); exit fill still quote-based.
Does not mutate the freeze profile (research switch only).
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

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    trades = res["trades"]
    daily = res["daily"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    trades.to_csv(sub / "trades.csv", index=False)
    daily.to_csv(sub / "daily.csv", index=False)
    n_tox = int((trades["reason"] == "TRADE_TOX").sum()) if len(trades) else 0
    focus = []
    if len(trades):
        for d in ["2026-05-06", "2026-05-11", "2026-06-11", "2026-06-24", "2026-05-15"]:
            hit = trades[trades["date"].astype(str) == d]
            for r in hit.itertuples(index=False):
                focus.append(
                    {
                        "date": d,
                        "symbol": r.symbol,
                        "dir": r.dir,
                        "ret": float(r.ret),
                        "reason": r.reason,
                    }
                )
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "n_trade_tox": n_tox,
        "n_trade_path": s.get("n_trade_path"),
        "n_trade_path_miss": s.get("n_trade_path_miss"),
        "worst_trade": float(trades["ret"].min()) if len(trades) else None,
        "worst_day": float(daily["day_ret"].min()) if len(daily) else None,
        "end_equity": float(s["end_equity"]),
        "focus": focus,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/trade_toxic_ablation_peer3_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-16")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    # ensure trades root
    base.setdefault("paths", {}).setdefault(
        "option_trades_root", "/mnt/s990/new_option_data_s3_trades"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def tt(**kw):
        return {"trade_toxic": {"enabled": True, "min_hold_seconds": 60, **kw}}

    variants = [
        ("00_off", {"trade_toxic": {"enabled": False}}),
        ("01_cut20_mfe05", tt(cut_ret=0.20, mfe_bypass=0.05)),
        ("02_cut25_mfe05", tt(cut_ret=0.25, mfe_bypass=0.05)),
        ("03_cut30_mfe05", tt(cut_ret=0.30, mfe_bypass=0.05)),
        ("04_cut25_mfe08", tt(cut_ret=0.25, mfe_bypass=0.08)),
    ]

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    board = []
    for wname, start, end in windows:
        base_ret = None
        for vname, patch in variants:
            p = copy.deepcopy(base)
            p.setdefault("trade", {}).update(patch)
            print(f"==> {wname} / {vname}", flush=True)
            row = _run(p, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "00_off":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_baseline"] = (row["total_ret"] / base_ret) if base_ret else None
            board.append(row)
            print(
                f"    ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                f"vs={row['vs_baseline']:.1%} tox={row['n_trade_tox']} "
                f"path={row['n_trade_path']}/{row['n_trade_path_miss']} "
                f"worst_t={row['worst_trade']:.2%}",
                flush=True,
            )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k != "focus"} for r in board]
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "n_trades",
        "vs_baseline",
        "n_trade_tox",
        "worst_trade",
        "worst_day",
        "trade_win",
    ]
    print(pd.DataFrame(flat)[cols].to_string(index=False))
    print("--- focus ---")
    for r in board:
        if r["variant"] in {"00_off", "02_cut25_mfe05"} and r.get("focus"):
            print(r["window"], r["variant"], r["focus"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
