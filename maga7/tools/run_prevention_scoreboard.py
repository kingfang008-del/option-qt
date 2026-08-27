#!/usr/bin/env python3
"""Dual-window scoreboard: predictive prevention hard vs soft vs off.

On research baseline (L1+L2 hunter kept): only varies ``watchdog.prevention``.
Does not mutate freeze. Gate suggestion: strong ≥90% of prev-off (freeze bar 95%).
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

# Days known UP-toxic / wash-dispersion stress (within Feb–Jul 2026 when present).
FOCUS_DATES = (
    "2026-05-06",
    "2026-05-11",
    "2026-06-11",
    "2026-06-24",
    "2026-07-07",
    "2026-07-17",
)


def _patch_prevention(profile: dict, *, mode: str) -> dict:
    p = copy.deepcopy(profile)
    wd = dict(p.get("watchdog") or {})
    prev = dict(wd.get("prevention") or {})
    prev.update(
        {
            "rule": "mixed_wash_up",
            "expert": "up_toxic",
            "risk_off_expert": "up_toxic_block",
            "wash_drop_min": 0.008,
            "washout_breadth_min": 3,
            "wash_window_end": "10:00",
            "frac_above_min": 0.35,
            "frac_above_max": 0.70,
            "ttl_minutes": None,
        }
    )
    if mode == "off":
        prev["enabled"] = False
    elif mode == "soft":
        prev["enabled"] = True
        prev["prefer_risk_off"] = False
    elif mode == "hard":
        prev["enabled"] = True
        prev["prefer_risk_off"] = True
    else:
        raise ValueError(mode)
    wd["prevention"] = prev
    wd["enabled"] = True
    p["watchdog"] = wd
    return p


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"].copy()
    trades = res["trades"].copy()
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    daily.to_csv(sub / "daily.csv", index=False)
    trades.to_csv(sub / "trades.csv", index=False)

    counts = dict(s.get("router_day_counts") or {})
    n_soft = int(counts.get("up_toxic", 0) or 0)
    n_hard = int(counts.get("up_toxic_block", 0) or 0)
    # prevention reason days from daily if present
    n_prev_reason = 0
    if "watchdog_reason" in daily.columns:
        n_prev_reason = int(
            daily["watchdog_reason"].astype(str).str.startswith("prevention:").sum()
        )

    focus = []
    if len(daily):
        for d in FOCUS_DATES:
            hit = daily[daily["date"].astype(str) == d]
            if hit.empty:
                continue
            focus.append({"date": d, "day_ret": float(hit.iloc[0]["day_ret"])})

    bad = daily[daily["day_ret"].astype(float) <= -0.03] if len(daily) else daily
    neg = daily[daily["day_ret"].astype(float) < 0] if len(daily) else daily
    up_tr = trades[trades["dir"].astype(str).str.upper() == "UP"] if len(trades) else trades
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "end_equity": float(s.get("end_equity") or 0.0),
        "n_watchdog_days": int(s.get("n_watchdog_days") or 0),
        "router_day_counts": counts,
        "n_up_toxic_days": n_soft,
        "n_up_toxic_block_days": n_hard,
        "n_prevention_reason_days": n_prev_reason,
        "n_regime_block": int(s.get("n_regime_block") or 0),
        "n_bad_days": int(len(bad)),
        "sum_bad": float(bad["day_ret"].sum()) if len(bad) else 0.0,
        "n_neg_days": int(len(neg)),
        "sum_neg": float(neg["day_ret"].sum()) if len(neg) else 0.0,
        "worst_day": float(daily["day_ret"].min()) if len(daily) else None,
        "n_up_trades": int(len(up_tr)),
        "up_trade_exp": float(up_tr["ret"].mean()) if len(up_tr) else None,
        "focus": focus,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/prevention_scoreboard_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument("--gate-strong", type=float, default=0.90, help="min vs prev-off on strong")
    ap.add_argument("--gate-freeze", type=float, default=0.95, help="freeze bar vs prev-off")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    variants = [
        ("00_off", "off"),
        ("01_soft", "soft"),
        ("02_hard", "hard"),
    ]
    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]

    board: list[dict] = []
    for wname, start, end in windows:
        base_ret = None
        for vname, mode in variants:
            print(f"==> {wname} / {vname} ({mode})", flush=True)
            prof = _patch_prevention(base, mode=mode)
            row = _run(prof, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "00_off":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["mode"] = mode
            row["vs_prev_off"] = (
                (row["total_ret"] / base_ret) if base_ret not in (None, 0.0) else None
            )
            board.append(row)
            print(
                f"    ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                f"vs_off={row['vs_prev_off']:.1%} "
                f"trades={row['n_trades']} "
                f"block_days={row['n_up_toxic_block_days']} "
                f"soft_days={row['n_up_toxic_days']} "
                f"bad={row['n_bad_days']} sum_bad={row['sum_bad']:+.2%} "
                f"up_exp={row['up_trade_exp']}",
                flush=True,
            )

    (out / "scoreboard.json").write_text(
        json.dumps(board, indent=2, default=str), encoding="utf-8"
    )
    flat = [{k: v for k, v in r.items() if k not in {"focus", "router_day_counts"}} for r in board]
    pd.DataFrame(flat).to_csv(out / "scoreboard.csv", index=False)

    # Verdict
    verdict: dict = {
        "gate_strong": args.gate_strong,
        "gate_freeze": args.gate_freeze,
        "candidates": {},
        "recommend": None,
        "notes": [],
    }
    for mode_name, vkey in (("soft", "01_soft"), ("hard", "02_hard")):
        strong = next(r for r in board if r["window"] == "strong_may_jul" and r["variant"] == vkey)
        weak = next(r for r in board if r["window"] == "weak_feb_apr" and r["variant"] == vkey)
        ok_strong = (strong["vs_prev_off"] or 0) >= args.gate_strong
        ok_freeze = (strong["vs_prev_off"] or 0) >= args.gate_freeze
        # Prefer bad-day sum not worse than off by much; and weak not collapse
        off_s = next(r for r in board if r["window"] == "strong_may_jul" and r["variant"] == "00_off")
        off_w = next(r for r in board if r["window"] == "weak_feb_apr" and r["variant"] == "00_off")
        bad_improve = strong["sum_bad"] >= off_s["sum_bad"] - 1e-12  # less negative is better
        weak_ok = (weak["vs_prev_off"] or 0) >= 0.85
        verdict["candidates"][mode_name] = {
            "strong_vs_off": strong["vs_prev_off"],
            "weak_vs_off": weak["vs_prev_off"],
            "pass_research_gate": bool(ok_strong and weak_ok),
            "pass_freeze_bar": bool(ok_freeze and weak_ok),
            "strong_bad_sum": strong["sum_bad"],
            "off_strong_bad_sum": off_s["sum_bad"],
            "bad_not_worse": bool(bad_improve or strong["sum_bad"] > off_s["sum_bad"]),
            "n_trigger_strong": strong["n_up_toxic_block_days"] + strong["n_up_toxic_days"],
            "n_trigger_weak": weak["n_up_toxic_block_days"] + weak["n_up_toxic_days"],
        }

    hard = verdict["candidates"]["hard"]
    soft = verdict["candidates"]["soft"]
    if hard["pass_freeze_bar"] and hard["bad_not_worse"]:
        verdict["recommend"] = "hard"
        verdict["notes"].append("hard passes freeze bar and bad-day sum not worse → keep prefer_risk_off=true")
    elif soft["pass_freeze_bar"] and soft["bad_not_worse"]:
        verdict["recommend"] = "soft"
        verdict["notes"].append("only soft clears freeze bar → set prefer_risk_off=false")
    elif hard["pass_research_gate"] and hard["bad_not_worse"]:
        verdict["recommend"] = "hard_research"
        verdict["notes"].append("hard OK for research (≥90%) but below freeze 95% — keep research, not freeze")
    elif soft["pass_research_gate"] and soft["bad_not_worse"]:
        verdict["recommend"] = "soft_research"
        verdict["notes"].append("soft OK for research; hard fails gates")
    else:
        verdict["recommend"] = "off"
        verdict["notes"].append("neither arm clears dual-window gates — leave prevention off / retune rule")

    # Prefer hard over soft when both research-OK and hard bad-days better
    if (
        hard["pass_research_gate"]
        and soft["pass_research_gate"]
        and hard["strong_bad_sum"] > soft["strong_bad_sum"] + 1e-9
        and verdict["recommend"] in {"soft", "soft_research"}
    ):
        verdict["recommend"] = "hard_research" if not hard["pass_freeze_bar"] else "hard"
        verdict["notes"].append("both pass research; hard improves bad-day sum more")

    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "n_trades",
        "vs_prev_off",
        "n_up_toxic_block_days",
        "n_up_toxic_days",
        "n_bad_days",
        "sum_bad",
        "up_trade_exp",
        "worst_day",
    ]
    print(pd.DataFrame(flat)[cols].to_string(index=False))
    print("--- focus day_ret ---")
    for r in board:
        if r.get("focus"):
            print(r["window"], r["variant"], r["focus"])
    print("--- verdict ---")
    print(json.dumps(verdict, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
