#!/usr/bin/env python3
"""Dual-window ablation: trade_toxic + stock-divergence soft MFE bypass.

Base: cut25 / mfe05 / max_cut=600. Softens MFE to div_mfe when underlying
adverse move < div_stock_adverse_max (targets 05-06-style option dig / flat stock).
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
FOCUS = ["2026-05-06", "2026-05-11", "2026-06-11", "2026-06-24"]


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
    focus = []
    if len(trades):
        for d in FOCUS:
            for r in trades[trades["date"].astype(str) == d].itertuples(index=False):
                focus.append(
                    {
                        "date": d,
                        "symbol": r.symbol,
                        "dir": r.dir,
                        "ret": float(r.ret),
                        "reason": r.reason,
                    }
                )
    day_focus = {}
    if len(daily):
        for d in ["2026-05-06", "2026-06-11"]:
            row = daily[daily["date"].astype(str) == d]
            if len(row):
                day_focus[d] = float(row["day_ret"].iloc[0])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_trade_tox": int((trades["reason"] == "TRADE_TOX").sum()) if len(trades) else 0,
        "worst_trade": float(trades["ret"].min()) if len(trades) else None,
        "worst_day": float(daily["day_ret"].min()) if len(daily) else None,
        "day_focus": day_focus,
        "focus": focus,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/trade_toxic_div_ablation_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-16")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    base.setdefault("paths", {}).setdefault(
        "option_trades_root", "/mnt/s990/new_option_data_s3_trades"
    )
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
                "max_cut_seconds": 600,
                **kw,
            }
        }

    variants = [
        ("00_off", {"trade_toxic": {"enabled": False}}),
        ("01_max600", tt()),
        ("02_div08_adv05", tt(div_mfe_bypass=0.08, div_stock_adverse_max=0.005)),
        ("03_div08_adv03", tt(div_mfe_bypass=0.08, div_stock_adverse_max=0.003)),
        ("04_div06_adv05", tt(div_mfe_bypass=0.06, div_stock_adverse_max=0.005)),
        ("05_div10_adv05", tt(div_mfe_bypass=0.10, div_stock_adverse_max=0.005)),
        # Global mfe08 without stock gate (control — previously weak on strong window)
        ("06_mfe08_nogate", tt(mfe_bypass=0.08, max_cut_seconds=600)),
    ]

    board = []
    for wname, start, end in [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]:
        base_ret = None
        max600_ret = None
        for vname, patch in variants:
            p = copy.deepcopy(base)
            p.setdefault("trade", {}).update(patch)
            print(f"==> {wname} / {vname}", flush=True)
            row = _run(p, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "00_off":
                base_ret = row["total_ret"]
            if vname == "01_max600":
                max600_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_off"] = (row["total_ret"] / base_ret) if base_ret else None
            row["vs_max600"] = (row["total_ret"] / max600_ret) if max600_ret else None
            board.append(row)
            fd = row.get("day_focus") or {}
            print(
                f"    ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                f"vs_off={row['vs_off']:.1%} vs600={row['vs_max600']} "
                f"tox={row['n_trade_tox']} d0506={fd.get('2026-05-06')} "
                f"d0611={fd.get('2026-06-11')}",
                flush=True,
            )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in {"focus", "day_focus"}} for r in board]
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)
    print(
        pd.DataFrame(flat)[
            ["window", "variant", "total_ret", "maxdd", "vs_off", "vs_max600", "n_trade_tox", "worst_day"]
        ].to_string(index=False)
    )
    print("--- strong focus ---")
    for r in board:
        if r["window"] == "strong_may_jul":
            print(r["variant"], r.get("focus"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
