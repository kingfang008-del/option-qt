#!/usr/bin/env python3
"""Offline accept for S1 soft path on research_baseline (correct dual windows).

Windows (docs/research_full_day_peer3_baseline.md):
  strong = Apr–Jul, weak = Jan–Mar, july = toxic slice.

Arms (full research stack intact; only toggle stock_path_confirm):
  PRE = S1 off
  S1  = current research_baseline (S1 on)
  S1_WA_UP = S1 + WAVE_ABORT only_directions=UP (overlay, not default)

Pass (S1 vs PRE):
  strong keep>=0.85 AND (weak ret↑ OR weak MaxDD↑) AND july keep>=0.95
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
from maga7.common.provenance import code_fingerprint
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
    trade.pop("wave_abort", None)
    if arm == "PRE":
        trade["stock_path_confirm"] = {"enabled": False}
        return p
    if arm == "S1":
        trade["stock_path_confirm"] = dict(PATH_SOFT)
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
    (sub / "summary.json").write_text(
        json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
    )
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
        "n_TRADE_TOX": int(reasons.get("TRADE_TOX", 0)),
        "n_stock_path_confirm_block": s.get("n_stock_path_confirm_block"),
        "n_stock_path_confirm_ok": s.get("n_stock_path_confirm_ok"),
        "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/s1_research_baseline_accept_apr_jul_jan_mar_v1",
    )
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-21")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--july-start", default="2026-07-01")
    ap.add_argument("--july-end", default="2026-07-21")
    ap.add_argument(
        "--arms",
        default="PRE,S1,S1_WA_UP",
        help="comma list: PRE,S1,S1_WA_UP",
    )
    args = ap.parse_args(argv)

    base = load_profile(BASE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = code_fingerprint(base.get("_profile_path") or BASE)

    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
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
                f"tox={row['n_TRADE_TOX']} path_ok={row.get('n_stock_path_confirm_ok')} "
                f"path_blk={row.get('n_stock_path_confirm_block')}",
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

    def decide(pack: str) -> dict[str, Any]:
        strong_pre = cell("strong", "PRE")
        weak_pre = cell("weak", "PRE")
        july_pre = cell("july", "PRE")
        strong_p = cell("strong", pack)
        weak_p = cell("weak", pack)
        july_p = cell("july", pack)
        strong_keep = (
            (strong_p / strong_pre)
            if strong_pre and strong_pre > 1e-9 and strong_p is not None
            else None
        )
        weak_lift = weak_p is not None and weak_pre is not None and weak_p > weak_pre
        weak_dd_ok = (
            cell("weak", pack, "maxdd") is not None
            and cell("weak", "PRE", "maxdd") is not None
            and cell("weak", pack, "maxdd") > cell("weak", "PRE", "maxdd")
        )
        july_ok = (
            july_p is not None
            and july_pre is not None
            and july_p + 1e-9 >= july_pre * 0.95
        )
        flags: list[str] = []
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
        if (
            "strong_retain_ok" in flags
            and ("weak_ret_improved" in flags or "weak_dd_improved" in flags)
            and "july_keep_ok" in flags
        ):
            decision = "KEEP_S1_RESEARCH_BASELINE"
        elif "strong_retain_ok" in flags and "july_keep_ok" in flags:
            decision = "KEEP_SOFT"
        else:
            decision = "RECONSIDER_S1"
        return {
            "pack": pack,
            "decision": decision,
            "flags": flags,
            "metrics": {
                "strong_pre": strong_pre,
                "strong_pack": strong_p,
                "strong_keep": strong_keep,
                "weak_pre": weak_pre,
                "weak_pack": weak_p,
                "weak_maxdd_pre": cell("weak", "PRE", "maxdd"),
                "weak_maxdd_pack": cell("weak", pack, "maxdd"),
                "july_pre": july_pre,
                "july_pack": july_p,
                "july_keep": (
                    july_p / july_pre
                    if july_pre and july_pre > 1e-9 and july_p is not None
                    else None
                ),
            },
        }

    s1 = decide("S1") if "S1" in arms and "PRE" in arms else None
    s1_wa = decide("S1_WA_UP") if "S1_WA_UP" in arms and "PRE" in arms else None

    summary = {
        "profile": base.get("profile_id") or base.get("profile"),
        "research_revision": base.get("research_revision"),
        "strategy_fingerprint": fp,
        "windows": {
            "strong": f"{args.strong_start}..{args.strong_end}",
            "weak": f"{args.weak_start}..{args.weak_end}",
            "july": f"{args.july_start}..{args.july_end}",
        },
        "pass_rule": "strong keep>=0.85 vs PRE AND (weak ret or dd improve) AND july keep>=0.95",
        "s1_vs_pre": s1,
        "s1_wa_up_vs_pre": s1_wa,
        "note": (
            "PRE = research stack with stock_path_confirm off; "
            "S1 = research_baseline default; S1_WA_UP = overlay only."
        ),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# S1 research_baseline offline accept (Apr–Jul / Jan–Mar)",
        "",
        f"**Fingerprint:** `{fp}`",
        f"**Revision:** `{base.get('research_revision')}`",
        "",
    ]
    if s1:
        lines += [
            f"**S1 vs PRE: `{s1['decision']}`**",
            "",
            "```json",
            json.dumps(s1["metrics"], indent=2, default=str),
            "```",
            "",
            f"flags: {', '.join(s1['flags'])}",
            "",
        ]
    if s1_wa:
        lines += [
            f"**S1_WA_UP vs PRE (overlay): `{s1_wa['decision']}`**",
            "",
            "```json",
            json.dumps(s1_wa["metrics"], indent=2, default=str),
            "```",
            "",
        ]
    lines += [rdf.to_markdown(index=False), ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "s1": s1["decision"] if s1 else None,
                "s1_wa_up": s1_wa["decision"] if s1_wa else None,
                "s1_metrics": (s1 or {}).get("metrics"),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
