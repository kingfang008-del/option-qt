#!/usr/bin/env python3
"""Dual-window: quote_fallback (+ optional max_cut / qf_cut) on current spine.

Case: 2026-02-17 GOOGL hard SL — GOOGL OPRA trades missing for Feb–mid-Apr, so
print toxic never arms. Even on quotes, cut25 hits after max_cut=600 (~T+671s);
cut20 hits ~T+422s (inside max600).

Arms keep print toxic unchanged; quote_fallback only when prints missing.
Windows: weak Feb–Apr / strong May–Jul23.
"""
from __future__ import annotations

import argparse
import copy
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

BASE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("weak_feb_apr", "2026-02-01", "2026-04-30"),
    ("strong_may_jul23", "2026-05-01", "2026-07-23"),
)
FOCUS = [
    "2026-02-17",
    "2026-05-11",
    "2026-06-11",
    "2026-06-24",
]


def _base_tt(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "cut_ret": 0.25,
        "mfe_bypass": 0.05,
        "min_hold_seconds": 60,
        "persist_seconds": 0,
        "max_cut_seconds": 600,
        "quote_confirm_ret": None,
        "div_mfe_bypass": 0.06,
        "div_stock_adverse_max": 0.005,
        "quote_fallback": False,
        "quote_fallback_cut_ret": None,
    }
    cfg.update(kw)
    return cfg


ARMS = {
    "OFF": None,  # keep profile trade_toxic as-is (prints-only)
    "QF_MAX600": _base_tt(quote_fallback=True),
    "QF_MAX720": _base_tt(quote_fallback=True, max_cut_seconds=720),
    "QF_MAX900": _base_tt(quote_fallback=True, max_cut_seconds=900),
    "QF_C20_MAX600": _base_tt(quote_fallback=True, quote_fallback_cut_ret=0.20),
    "QF_C20_MAX720": _base_tt(
        quote_fallback=True, quote_fallback_cut_ret=0.20, max_cut_seconds=720
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/trade_toxic_quote_fallback_dual_v1")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            print(f"skip unknown arm {arm}", flush=True)
            continue
        for wname, start, end in WINDOWS:
            p = copy.deepcopy(base)
            p["date_range"] = {"start": start, "end": end}
            if ARMS[arm] is not None:
                p.setdefault("trade", {})["trade_toxic"] = copy.deepcopy(ARMS[arm])
            tag = f"{arm}__{wname}"
            print(f"=== {tag} ===", flush=True)
            res = run_offline_replay(p, scheme="single")
            sub = out / tag
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "summary.json").write_text(
                json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
            )
            res["daily"].to_csv(sub / "daily.csv", index=False)
            res["trades"].to_csv(sub / "trades.csv", index=False)
            s = res["summary"]
            tr = res["trades"]
            n_tox = int((tr["reason"] == "TRADE_TOX").sum()) if len(tr) else 0
            focus_hits = []
            if len(tr):
                for d in FOCUS:
                    hit = tr[tr["date"].astype(str) == d]
                    for r in hit.itertuples(index=False):
                        focus_hits.append(
                            {
                                "date": d,
                                "symbol": r.symbol,
                                "ret": float(r.ret),
                                "reason": r.reason,
                            }
                        )
            (sub / "focus.json").write_text(
                json.dumps(focus_hits, indent=2), encoding="utf-8"
            )
            r = {
                "arm": arm,
                "window": wname,
                "total_ret": float(s.get("total_ret") or 0),
                "maxdd": float(s.get("maxdd") or 0),
                "n_trades": int(s.get("n_trades") or 0),
                "n_trade_tox": n_tox,
                "n_trade_path": s.get("n_trade_path"),
                "n_trade_path_miss": s.get("n_trade_path_miss"),
                "worst_trade": float(tr["ret"].min()) if len(tr) else None,
            }
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} "
                f"tox={n_tox} miss={r['n_trade_path_miss']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    off = {r["window"]: r["total_ret"] for r in rows if r["arm"] == "OFF"}
    board["vs_OFF"] = board.apply(
        lambda r: (r["total_ret"] / off[r["window"]]) if off.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    verdict, best, best_score = "DUAL_FAIL", None, -1.0
    for arm in [a for a in ARMS if a != "OFF"]:
        if arm not in set(board.arm):
            continue
        sub = board[board.arm == arm]
        weak = float(sub[sub.window == "weak_feb_apr"].vs_OFF.iloc[0])
        strong = float(sub[sub.window == "strong_may_jul23"].vs_OFF.iloc[0])
        score = weak * strong
        print(f"{arm}: weak={weak:.3f} strong={strong:.3f} score={score:.3f}", flush=True)
        # Survives-first: strong keep ≥0.97; wire weak ≥0.95 / research ≥0.85.
        # Prefer highest weak×strong among passers (skip no-op keep=1/1 if better exists).
        if strong >= 0.97 and weak >= 0.95 and score > best_score:
            verdict, best, best_score = "DUAL_PASS_WIRE", arm, score
        elif (
            strong >= 0.97
            and weak >= 0.85
            and verdict == "DUAL_FAIL"
            and score > best_score
        ):
            verdict, best, best_score = "DUAL_PASS_RESEARCH", arm, score

    summary = {
        "verdict": verdict,
        "best": best,
        "rule": (
            "quote_fallback when OPRA prints missing; optional quote_fallback_cut_ret "
            "only on quote mark; print cut_ret unchanged"
        ),
        "scoreboard": board.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(board.to_string(index=False), flush=True)
    print(f"verdict={verdict} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
