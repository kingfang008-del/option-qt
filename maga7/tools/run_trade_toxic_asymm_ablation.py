#!/usr/bin/env python3
"""Dual-window ablation: asymmetric trade_toxic guards (persist / max_cut).

Compares against L2+TT1_05+sl55 with base cut25. Does not mutate freeze profile.
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
FOCUS_DATES = ["2026-05-06", "2026-05-11", "2026-05-15", "2026-06-11", "2026-06-24"]


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    # Ablation patches trade_toxic; ensure baseline-off for fair vs unless patch enables.
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
        for d in FOCUS_DATES:
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
    worst_day = None
    for d in ["2026-05-06", "2026-06-11"]:
        if len(daily) and "date" in daily.columns:
            row = daily[daily["date"].astype(str) == d]
            if len(row):
                worst_day = worst_day or {}
                worst_day[d] = float(row["day_ret"].iloc[0])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "n_trade_tox": n_tox,
        "n_trade_path": s.get("n_trade_path"),
        "n_trade_path_miss": s.get("n_trade_path_miss"),
        "worst_trade": float(trades["ret"].min()) if len(trades) else None,
        "worst_day": float(daily["day_ret"].min()) if len(daily) else None,
        "focus_days": worst_day,
        "end_equity": float(s["end_equity"]),
        "focus": focus,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/trade_toxic_asymm_ablation_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-16")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    base.setdefault("paths", {}).setdefault(
        "option_trades_root", "/mnt/s990/new_option_data_s3_trades"
    )
    # Start from tox OFF so variants are explicit (profile may have tox on).
    base.setdefault("trade", {})["trade_toxic"] = {"enabled": False}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def tt(**kw):
        return {
            "trade_toxic": {
                "enabled": True,
                "cut_ret": 0.25,
                "mfe_bypass": 0.05,
                "min_hold_seconds": 60,
                **kw,
            }
        }

    variants = [
        ("00_off", {"trade_toxic": {"enabled": False}}),
        ("01_base_cut25", tt()),
        ("02_persist30", tt(persist_seconds=30)),
        ("03_persist60", tt(persist_seconds=60)),
        ("04_max600", tt(max_cut_seconds=600)),
        ("05_max720", tt(max_cut_seconds=720)),
        ("06_persist30_max600", tt(persist_seconds=30, max_cut_seconds=600)),
        ("07_persist30_max720", tt(persist_seconds=30, max_cut_seconds=720)),
        ("08_qconf20", tt(quote_confirm_ret=0.20)),
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
            row["vs_off"] = (row["total_ret"] / base_ret) if base_ret else None
            board.append(row)
            fd = row.get("focus_days") or {}
            print(
                f"    ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                f"vs_off={row['vs_off']:.1%} tox={row['n_trade_tox']} "
                f"worst_t={row['worst_trade']:.2%} "
                f"d0506={fd.get('2026-05-06')} d0611={fd.get('2026-06-11')}",
                flush=True,
            )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in {"focus", "focus_days"}} for r in board]
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "vs_off",
        "n_trade_tox",
        "worst_trade",
        "worst_day",
    ]
    print(pd.DataFrame(flat)[cols].to_string(index=False))
    print("--- focus ---")
    for r in board:
        if r["window"] == "strong_may_jul" and r.get("focus"):
            print(r["variant"], r["focus"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
