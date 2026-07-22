#!/usr/bin/env python3
"""Dual-window accept: S1 soft path + WAVE_ABORT UP-only on peer3 baseline.

Windows: strong May–Jul21, weak Feb–Apr.
Pass: strong keep>=0.85 vs L0 AND (weak improved OR MaxDD improved) AND July toxic not worse.
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

PATH_SOFT = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "on_timeout": "allow",
    "delay_on_pos": False,
    "tod_start": "10:30",
    "tod_end": "14:00",
}

WA_UP = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "revoke_seconds": 1800,
    "on_timeout": "allow",
    "only_directions": ["UP"],
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    trade = p.setdefault("trade", {})
    trade["hold_watchdog"] = {"enabled": False}
    trade.pop("wave_abort", None)
    trade.pop("stock_path_confirm", None)
    p["state_gate"] = {"enabled": False}
    rr = dict(p.get("regime_router") or {})
    rr["enabled"] = False
    p["regime_router"] = rr

    if arm == "L0":
        return p
    if arm == "S1":
        trade["stock_path_confirm"] = dict(PATH_SOFT)
        return p
    if arm == "WA_UP":
        trade["wave_abort"] = dict(WA_UP)
        return p
    if arm == "S1_WA_UP":
        trade["stock_path_confirm"] = dict(PATH_SOFT)
        trade["wave_abort"] = dict(WA_UP)
        return p
    raise SystemExit(f"unknown arm {arm}")


def _run(prof: dict[str, Any], *, start: str, end: str, tag: str, out: Path) -> dict[str, Any]:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(res["summary"], indent=2, default=str), encoding="utf-8")
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    s = res["summary"]
    tr = res["trades"]
    reasons = tr["reason"].value_counts().to_dict() if not tr.empty else {}
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "n_WAVE_ABORT": int(reasons.get("WAVE_ABORT", 0)),
        "n_stock_path_confirm_block": s.get("n_stock_path_confirm_block"),
        "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/s1_wa_up_dual_window_v1")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-21")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument("--july-start", default="2026-07-01")
    ap.add_argument("--july-end", default="2026-07-21")
    args = ap.parse_args(argv)

    base = load_profile(BASE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = ["L0", "S1", "WA_UP", "S1_WA_UP"]
    windows = [
        ("strong", args.strong_start, args.strong_end),
        ("weak", args.weak_start, args.weak_end),
        ("july", args.july_start, args.july_end),
    ]
    rows: list[dict[str, Any]] = []
    for wname, start, end in windows:
        for arm in arms:
            tag = f"{wname}_{arm}"
            print(f"[run] {tag} {start}→{end}", flush=True)
            prof = _apply(base, arm)
            row = _run(prof, start=start, end=end, tag=tag, out=out)
            row["window"] = wname
            row["arm"] = arm
            rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']} "
                f"wa={row['n_WAVE_ABORT']} path_blk={row.get('n_stock_path_confirm_block')}",
                flush=True,
            )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)

    def cell(window: str, arm: str, col: str = "total_ret") -> float | None:
        hit = rdf[(rdf["window"] == window) & (rdf["arm"] == arm)]
        if hit.empty:
            return None
        v = hit.iloc[0][col]
        return None if pd.isna(v) else float(v)

    strong_l0 = cell("strong", "L0")
    weak_l0 = cell("weak", "L0")
    july_l0 = cell("july", "L0")
    pack = "S1_WA_UP"
    strong_p = cell("strong", pack)
    weak_p = cell("weak", pack)
    july_p = cell("july", pack)
    strong_keep = (strong_p / strong_l0) if strong_l0 and strong_l0 > 1e-9 and strong_p is not None else None
    weak_lift = weak_p is not None and weak_l0 is not None and weak_p > weak_l0
    weak_dd_ok = (
        cell("weak", pack, "maxdd") is not None
        and cell("weak", "L0", "maxdd") is not None
        and cell("weak", pack, "maxdd") > cell("weak", "L0", "maxdd")
    )
    july_ok = july_p is not None and july_l0 is not None and july_p + 1e-9 >= july_l0 * 0.95

    flags = []
    if strong_keep is not None and strong_keep >= 0.85:
        flags.append("strong_retain_ok")
    else:
        flags.append("strong_retain_fail")
    if weak_lift:
        flags.append("weak_ret_improved")
    if weak_dd_ok:
        flags.append("weak_dd_improved")
    if july_ok:
        flags.append("july_keep_ok")
    else:
        flags.append("july_keep_fail")

    if "strong_retain_ok" in flags and ("weak_ret_improved" in flags or "weak_dd_improved" in flags) and "july_keep_ok" in flags:
        decision = "PROMOTE_TO_RESEARCH_CANDIDATE"
    elif "strong_retain_ok" in flags and "july_keep_ok" in flags:
        decision = "PROMOTE_SOFT_PACK"
    else:
        decision = "REJECT_OR_ITERATE"

    summary = {
        "decision": decision,
        "flags": flags,
        "pack": pack,
        "metrics": {
            "strong_l0": strong_l0,
            "strong_pack": strong_p,
            "strong_keep": strong_keep,
            "weak_l0": weak_l0,
            "weak_pack": weak_p,
            "july_l0": july_l0,
            "july_pack": july_p,
            "july_keep": (july_p / july_l0) if july_l0 and july_l0 > 1e-9 and july_p is not None else None,
        },
        "by_arm": {
            arm: {
                "strong": cell("strong", arm),
                "weak": cell("weak", arm),
                "july": cell("july", arm),
                "strong_keep_vs_l0": (
                    cell("strong", arm) / strong_l0
                    if strong_l0 and strong_l0 > 1e-9 and cell("strong", arm) is not None
                    else None
                ),
            }
            for arm in arms
        },
        "rows": rows,
        "pass_rule": "strong keep>=0.85 AND (weak ret or dd improve) AND july keep>=0.95",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# S1 + WAVE_ABORT UP-only Dual Window",
                "",
                f"**Decision: `{decision}`**",
                "",
                "```json",
                json.dumps(summary["metrics"], indent=2, default=str),
                "```",
                "",
                rdf.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "flags": flags, "metrics": summary["metrics"]}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
