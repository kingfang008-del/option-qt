#!/usr/bin/env python3
"""Three-window ablation: event/news blackout vs peer3_v1 (Jul toxic rescue).

Context: Jul10 META / Jul20–21 AMD / Jul22 MSFT are **not** on Finnhub earnings
or live calendar (META/MSFT report 07-29; 07-22 has GOOGL/TSLA AH). This run
measures (1) oracle upper bound and (2) whether real calendar rows help.

Windows:
  weak   Jan–Mar
  mid    May1–Jul9
  jul    Jul10–23

Arms:
  PRE              peer3_v1 as-is (feb_jul_aapl_ceo already on; Jul≈no-op)
  CAL_OFF          event_calendar_block=false
  ORACLE_SYM       blackout tox symbols on tox days (upper bound)
  ORACLE_FULL      full-day blackout on tox dates
  FH_JUL22         Finnhub-true 07-22 GOOGL+TSLA (wrong names vs MSFT)
  FH_EARN_JUL      Finnhub Jul Mag7 earnings rows (07-22 peers + 07-29 META/MSFT)
  GAP04_BLOCK      causal overnight fav_gap≥4% block (META/AMD proxy; misses MSFT)
  GAP04_SCALE      same, scale×0.5
  GAP035_BLOCK     fav_gap≥3.5% block

Pass: jul vs_PRE>1 and weak/mid keep≥0.85. Oracle PASS only proves ceiling —
still need a causal proxy before wiring.

Example:
  PYTHONPATH=. python -m maga7.tools.run_event_news_triple_ablation \\
    --out maga7/results/event_news_triple_v1
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
    ("weak_jan_mar", "2026-01-02", "2026-03-31"),
    ("mid_may_jul9", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

# Post-hoc labels from research_baseline Jul autopsy (NOT a live source).
ORACLE_SYM = {
    "2026-07-10": ["META"],
    "2026-07-20": ["AMD"],
    "2026-07-21": ["AMD"],
    "2026-07-22": ["MSFT"],
}
ORACLE_FULL_DATES = sorted(ORACLE_SYM.keys())

FH_JUL22 = {"2026-07-22": ["GOOGL", "TSLA"]}
FH_EARN_JUL = {
    "2026-07-22": ["GOOGL", "TSLA"],
    "2026-07-29": ["META", "MSFT"],  # outside jul pocket; mid/weak noop
}


def _regime_patch(**kw: Any) -> dict[str, Any]:
    return dict(kw)


def _gap(**kw: Any) -> dict[str, Any]:
    cfg = {"enabled": True, "max_fav_gap": 0.04, "mode": "block", "scale": 0.5}
    cfg.update(kw)
    return {"_trade": {"overnight_gap_gate": cfg}}


ARMS: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "CAL_OFF": _regime_patch(event_calendar_block=False),
    "ORACLE_SYM": _regime_patch(event_symbol_blackout=ORACLE_SYM),
    "ORACLE_FULL": _regime_patch(
        event_dates=ORACLE_FULL_DATES,
        # keep named calendar too; event_dates merges in resolve
    ),
    "FH_JUL22": _regime_patch(event_symbol_blackout=FH_JUL22),
    "FH_EARN_JUL": _regime_patch(event_symbol_blackout=FH_EARN_JUL),
    "GAP04_BLOCK": _gap(max_fav_gap=0.04, mode="block"),
    "GAP04_SCALE": _gap(max_fav_gap=0.04, mode="scale", scale=0.5),
    "GAP035_BLOCK": _gap(max_fav_gap=0.035, mode="block"),
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    patch = ARMS[arm]
    if patch is None:
        return p
    if "_trade" in patch:
        trade = p.setdefault("trade", {})
        trade.update(copy.deepcopy(patch["_trade"]))
    reg = p.setdefault("regime", {})
    # Fresh blackout maps per arm (avoid leaking prior keys when reusing profile).
    if "event_symbol_blackout" in patch:
        reg["event_symbol_blackout"] = copy.deepcopy(patch["event_symbol_blackout"])
    if "event_dates" in patch:
        # Merge with existing preset dates so we don't drop FOMC/earnings research days.
        from maga7.common.event_calendar import event_dates_from_cfg, event_cfg_from_profile

        existing = set(event_dates_from_cfg(event_cfg_from_profile(p)))
        reg["event_dates"] = sorted(existing | set(patch["event_dates"]))
    if "event_calendar_block" in patch:
        reg["event_calendar_block"] = bool(patch["event_calendar_block"])
    return p


def _run(prof: dict[str, Any], *, start: str, end: str, tag: str, out: Path) -> dict[str, Any]:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(
        json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
    )
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    s = res["summary"]
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "day_win": s.get("day_win"),
        "n_event_block": s.get("n_event_block") or s.get("n_event_skip"),
        "n_event_symbol_block": s.get("n_event_symbol_block"),
        "n_overnight_gap_block": s.get("n_overnight_gap_block"),
        "n_overnight_gap_scale": s.get("n_overnight_gap_scale"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--out", default="maga7/results/event_news_triple_v1")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--windows", default=",".join(w[0] for w in WINDOWS))
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    want_w = {w.strip() for w in args.windows.split(",") if w.strip()}
    windows = [w for w in WINDOWS if w[0] in want_w]

    rows: list[dict[str, Any]] = []
    for arm in arm_names:
        if arm not in ARMS:
            print(f"skip unknown arm {arm}", flush=True)
            continue
        prof = _apply(base, arm)
        for wname, start, end in windows:
            tag = f"{arm}__{wname}"
            print(f"=== {tag} ===", flush=True)
            r = _run(prof, start=start, end=end, tag=tag, out=out)
            r["arm"] = arm
            r["window"] = wname
            rows.append(r)
            print(
                f"  ret={r['total_ret']:.3f} maxdd={r['maxdd']:.3f} n={r['n_trades']} "
                f"day_blk={r['n_event_block']} sym_blk={r['n_event_symbol_block']} "
                f"gap_blk={r['n_overnight_gap_block']} gap_sc={r['n_overnight_gap_scale']}",
                flush=True,
            )

    board = pd.DataFrame(rows)
    pre = {r["window"]: float(r["total_ret"]) for r in rows if r["arm"] == "PRE"}
    board["vs_PRE"] = board.apply(
        lambda r: (float(r["total_ret"]) / pre[r["window"]]) if pre.get(r["window"]) else None,
        axis=1,
    )
    board.to_csv(out / "scoreboard.csv", index=False)

    jul = board[board.window == "jul10_23"].copy()
    best = None
    verdict = "NO_LIFT"
    for _, r in jul.sort_values("total_ret", ascending=False).iterrows():
        if r["arm"] == "PRE":
            continue
        weak = board[(board.arm == r["arm"]) & (board.window == "weak_jan_mar")]
        mid = board[(board.arm == r["arm"]) & (board.window == "mid_may_jul9")]
        if weak.empty or mid.empty:
            continue
        kw = float(weak.iloc[0]["vs_PRE"] or 0)
        km = float(mid.iloc[0]["vs_PRE"] or 0)
        if float(r["vs_PRE"] or 0) > 1.0 and kw >= 0.85 and km >= 0.85:
            best = r.to_dict()
            arm_s = str(r["arm"])
            if arm_s.startswith("ORACLE"):
                verdict = "ORACLE_CEILING"
            elif arm_s.startswith("GAP"):
                verdict = "GAP_TRAP_LIFT"
            else:
                verdict = "EVENT_NEWS_LIFT"
            break
    if best is None and len(jul):
        cand = jul[jul.arm != "PRE"].sort_values("total_ret", ascending=False)
        if len(cand) and float(cand.iloc[0]["vs_PRE"] or 0) > 1.0:
            best = cand.iloc[0].to_dict()
            verdict = "JUL_ONLY_PARTIAL"

    summary = {
        "verdict": verdict,
        "best": best,
        "oracle_sym": ORACLE_SYM,
        "windows": [list(w) for w in windows],
        "arms": arm_names,
        "scoreboard": board.to_dict(orient="records"),
        "note": (
            "Jul tox days lack Finnhub/live calendar hits. "
            "ORACLE_CEILING ≠ wireable; need causal proxy or park."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== scoreboard ===", flush=True)
    cols = [
        c
        for c in [
            "arm",
            "window",
            "total_ret",
            "maxdd",
            "n_trades",
            "vs_PRE",
            "n_event_block",
            "n_event_symbol_block",
        ]
        if c in board.columns
    ]
    print(board[cols].to_string(index=False), flush=True)
    print(f"\nverdict={verdict} best={best.get('arm') if best else None}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
