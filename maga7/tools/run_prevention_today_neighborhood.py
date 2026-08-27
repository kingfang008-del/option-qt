#!/usr/bin/env python3
"""Test mixed_wash_up prevention knobs on one live session fused stream (today).

Does not claim dual-window acceptance — only whether today's UP book is blocked.
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

import maga7.tools.run_live_fused_replay as fr
from maga7.common.config import load_profile
from maga7.tools.run_live_fused_replay import run_fused_replay


def _grid() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = [
        ("00_off", {"enabled": False}),
        (
            "01_b3_f70_hard",
            {
                "enabled": True,
                "prefer_risk_off": True,
                "washout_breadth_min": 3,
                "wash_drop_min": 0.008,
                "frac_above_min": 0.35,
                "frac_above_max": 0.70,
            },
        ),
        (
            "02_b3_f70_soft",
            {
                "enabled": True,
                "prefer_risk_off": False,
                "washout_breadth_min": 3,
                "wash_drop_min": 0.008,
                "frac_above_min": 0.35,
                "frac_above_max": 0.70,
            },
        ),
    ]
    for b in (4, 5):
        for fmax in (0.70, 0.65, 0.60):
            for drop in (0.008, 0.010):
                for hard in (True, False):
                    tag = (
                        f"b{b}_f{int(fmax * 100)}_d{int(drop * 1000)}_"
                        f"{'hard' if hard else 'soft'}"
                    )
                    out.append(
                        (
                            tag,
                            {
                                "enabled": True,
                                "prefer_risk_off": hard,
                                "washout_breadth_min": b,
                                "wash_drop_min": drop,
                                "frac_above_min": 0.35,
                                "frac_above_max": fmax,
                            },
                        )
                    )
    seen: set[str] = set()
    uniq: list[tuple[str, dict]] = []
    for tag, cfg in out:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append((tag, cfg))
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--session-dir",
        default="/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e",
    )
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--scheme", default="m5_circuit")
    args = ap.parse_args()

    session = Path(args.session_dir)
    out = session / "prevention_today_neighborhood"
    out.mkdir(parents=True, exist_ok=True)
    base = load_profile(args.profile)
    orig_load = fr.load_profile
    rows: list[dict] = []

    try:
        for tag, cfg in _grid():
            print(f"==> {tag}", flush=True)

            def _load(path, _cfg=cfg):  # noqa: ARG001
                p = copy.deepcopy(base)
                wd = dict(p.get("watchdog") or {})
                prev = dict(wd.get("prevention") or {})
                prev.update(
                    {
                        "rule": "mixed_wash_up",
                        "expert": "up_toxic",
                        "risk_off_expert": "up_toxic_block",
                        "wash_window_end": "10:00",
                        "ttl_minutes": None,
                        **_cfg,
                    }
                )
                wd["prevention"] = prev
                wd["enabled"] = True
                p["watchdog"] = wd
                return p

            fr.load_profile = lambda path, _f=_load: _f(path)  # type: ignore[assignment]
            summary = run_fused_replay(
                session,
                scheme=args.scheme,
                disable_prevention=False,
                tag=f"prev_{tag}",
            )
            src = session / f"fused_replay_prev_{tag}"
            sig_n = int(summary.get("n_scanner_signals") or 0)
            trades_n = int(summary.get("n_trades") or 0)
            reason = str(summary.get("watchdog_reason") or "")
            blocked = trades_n == 0 and reason.startswith("prevention:")
            symbols = ""
            tp = src / "trades.csv"
            if tp.is_file() and tp.stat().st_size > 10:
                try:
                    df = pd.read_csv(tp)
                    if len(df):
                        symbols = ",".join(
                            f"{r.symbol}:{r.direction}:{float(r.ret):.0%}"
                            for r in df.itertuples()
                        )
                except Exception:
                    symbols = ""
            row = {
                "tag": tag,
                **cfg,
                "watchdog_reason": reason,
                "watchdog_state": summary.get("watchdog_state"),
                "n_signals": sig_n,
                "n_trades": trades_n,
                "total_ret": summary.get("total_ret"),
                "blocked_today_up": blocked,
                "symbols": symbols,
            }
            rows.append(row)
            print(
                f"  wd={reason} sig={sig_n} trades={trades_n} "
                f"ret={row['total_ret']} blocked={blocked}",
                flush=True,
            )
    finally:
        fr.load_profile = orig_load

    board = pd.DataFrame(rows)
    board.to_csv(out / "neighborhood.csv", index=False)
    (out / "neighborhood.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    cols = [
        c
        for c in (
            "tag",
            "washout_breadth_min",
            "wash_drop_min",
            "frac_above_max",
            "prefer_risk_off",
            "n_trades",
            "total_ret",
            "blocked_today_up",
            "symbols",
        )
        if c in board.columns
    ]
    print("\n=== BLOCKED today's UP ===")
    catch = board[board["blocked_today_up"] == True]
    print(catch[cols].to_string(index=False) if len(catch) else "(none)")
    print("\n=== NOT blocked ===")
    miss = board[board["blocked_today_up"] == False]
    print(miss[cols].to_string(index=False) if len(miss) else "(none)")
    # soft that reduced but didn't zero
    soft_hit = board[
        (board.get("prefer_risk_off") == False)
        & (board["n_trades"] > 0)
        & (board["n_trades"] < board.loc[board["tag"] == "00_off", "n_trades"].iloc[0])
    ] if "00_off" in set(board["tag"]) else board.iloc[0:0]
    if len(soft_hit):
        print("\n=== soft reduced size/count (still traded) ===")
        print(soft_hit[cols].to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
