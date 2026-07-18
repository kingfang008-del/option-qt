#!/usr/bin/env python3
"""Scoreboard: freeze vs Watchdog(Degrade+Halt) vs +Hunter washout_reclaim.

Does not mutate freeze. Writes under ``--out``.
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


WINDOWS = [
    ("strong", "2026-05-01", "2026-07-17"),
    ("weak", "2026-02-01", "2026-04-30"),
]


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
    return {"summary": s, "daily": daily, "trades": trades}


def _row(window: str, variant: str, s: dict, base_ret: float | None) -> dict:
    tr = float(s.get("total_ret") or s.get("total_return") or 0.0)
    # summary key varies across versions
    if "total_ret" not in s and "equity" in s:
        tr = float(s["equity"]) / 100.0 - 1.0 if float(s["equity"]) > 2 else float(s.get("total_ret") or 0)
    # prefer explicit total_ret from replay summary
    for k in ("total_ret", "cum_ret", "portfolio_ret"):
        if k in s and s[k] is not None:
            tr = float(s[k])
            break
    # offline summary often uses ending equity path — read from daily if needed later
    vs = (tr / base_ret) if base_ret and abs(base_ret) > 1e-12 else None
    return {
        "window": window,
        "variant": variant,
        "total_ret": tr,
        "maxdd": s.get("maxdd") if s.get("maxdd") is not None else s.get("max_drawdown"),
        "n_trades": s.get("n_trades"),
        "n_hunt_signals": s.get("n_hunt_signals"),
        "n_hunt_trades": s.get("n_hunt_trades"),
        "n_hunt_mutex_skip": s.get("n_hunt_mutex_skip"),
        "watchdog_state_counts": s.get("watchdog_state_counts") or {},
        "vs_base": vs,
        "trade_exp": s.get("trade_exp"),
        "trade_win": s.get("trade_win"),
    }


def _total_ret_from_daily(daily: pd.DataFrame) -> float:
    if daily is None or daily.empty or "day_ret" not in daily.columns:
        return 0.0
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument(
        "--watchdog",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_v1.json",
    )
    ap.add_argument(
        "--hunter",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_hunter_washout_reclaim_v1.json",
    )
    ap.add_argument("--out", default="maga7/results/watchdog/hunter_washout_reclaim")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    base = load_profile(args.baseline)
    wdog = load_profile(args.watchdog)
    hunt = load_profile(args.hunter)

    board = []
    for wname, start, end in WINDOWS:
        print(f"== {wname} {start}..{end} baseline", flush=True)
        b = _run(base, start=start, end=end, tag=f"{wname}_baseline", out=out)
        b_ret = _total_ret_from_daily(b["daily"])
        b["summary"]["total_ret"] = b_ret
        board.append(_row(wname, "baseline", b["summary"], b_ret))

        print(f"== {wname} watchdog", flush=True)
        w = _run(wdog, start=start, end=end, tag=f"{wname}_watchdog", out=out)
        w_ret = _total_ret_from_daily(w["daily"])
        w["summary"]["total_ret"] = w_ret
        board.append(_row(wname, "watchdog", w["summary"], b_ret))

        print(f"== {wname} hunter_washout_reclaim", flush=True)
        h = _run(hunt, start=start, end=end, tag=f"{wname}_hunter", out=out)
        h_ret = _total_ret_from_daily(h["daily"])
        h["summary"]["total_ret"] = h_ret
        board.append(_row(wname, "hunter_washout_reclaim", h["summary"], b_ret))

    df = pd.DataFrame(board)
    df.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")

    # verdict: hunter must not destroy strong window (<95% of freeze)
    strong_h = next(r for r in board if r["window"] == "strong" and r["variant"] == "hunter_washout_reclaim")
    weak_h = next(r for r in board if r["window"] == "weak" and r["variant"] == "hunter_washout_reclaim")
    ok_strong = (strong_h["vs_base"] or 0) >= 0.95
    ok_weak = (weak_h["vs_base"] or 0) >= 0.95
    verdict = "ACCEPT_RESEARCH" if (ok_strong and ok_weak) else "REJECT"
    if ok_strong and ok_weak and (strong_h["vs_base"] or 0) < 1.0 and (weak_h["vs_base"] or 0) < 1.0:
        verdict = "ACCEPT_NEUTRAL"  # survives but no edge
    summary = {
        "verdict": verdict,
        "rule": "both windows vs freeze >= 95%; prefer incremental hunt edge",
        "scoreboard": board,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md = [
        "# Hunter washout→reclaim scoreboard",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "| window | variant | total_ret | MaxDD | n_trades | n_hunt | vs_base |",
        "|--------|---------|-----------|-------|----------|--------|---------|",
    ]
    for r in board:
        md.append(
            f"| {r['window']} | {r['variant']} | {r['total_ret']:.3f} | {r['maxdd']} | "
            f"{r['n_trades']} | {r['n_hunt_trades']} | {r['vs_base']} |"
        )
    md.append("")
    md.append("Profile: `…_watchdog_hunter_washout_reclaim_v1.json` (freeze untouched).")
    (out / "SCOREBOARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
