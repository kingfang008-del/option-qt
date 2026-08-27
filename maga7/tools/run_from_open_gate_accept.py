#!/usr/bin/env python3
"""Dual-window accept: session from_open chase gate (hard block vs soft size).

Arms (on current S1 research baseline, incl. dvol soft):
  PRE       = baseline (gate off)
  HARD_035  = block when |from_open| > 3.5% (same-sign)
  HARD_040  = block @ 4.0%
  HARD_045  = block @ 4.5%
  SOFT_035  = scale×0.5 @ 3.5%
  SOFT_040  = scale×0.5 @ 4.0%
  SOFT_045  = scale×0.5 @ 4.5%

Pass (pack vs PRE): strong keep>=0.85 AND weak (ret↑ OR MaxDD↑) AND july keep>=0.95
Also report 2026-07-22 AMD skip / day_ret.
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

ARMS: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "HARD_035": {"enabled": True, "max_abs": 0.035, "mode": "block", "same_sign_only": True},
    "HARD_040": {"enabled": True, "max_abs": 0.040, "mode": "block", "same_sign_only": True},
    "HARD_045": {"enabled": True, "max_abs": 0.045, "mode": "block", "same_sign_only": True},
    "SOFT_035": {
        "enabled": True,
        "max_abs": 0.035,
        "mode": "scale",
        "scale": 0.5,
        "same_sign_only": True,
    },
    "SOFT_040": {
        "enabled": True,
        "max_abs": 0.040,
        "mode": "scale",
        "scale": 0.5,
        "same_sign_only": True,
    },
    "SOFT_045": {
        "enabled": True,
        "max_abs": 0.045,
        "mode": "scale",
        "scale": 0.5,
        "same_sign_only": True,
    },
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    trade = p.setdefault("trade", {})
    cfg = ARMS[arm]
    if cfg is None:
        trade.pop("from_open_gate", None)
    else:
        trade["from_open_gate"] = dict(cfg)
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
    tr = res["trades"]
    d22 = tr[tr["date"].astype(str) == "2026-07-22"] if not tr.empty else tr
    d22_syms = (
        ",".join(f"{a}/{b}" for a, b in zip(d22["symbol"], d22["dir"])) if len(d22) else ""
    )
    amd_up = bool(len(d22) and ((d22["symbol"] == "AMD") & (d22["dir"] == "UP")).any())
    d22_day = res["daily"]
    d22_day = d22_day[d22_day["date"].astype(str) == "2026-07-22"] if not d22_day.empty else d22_day
    d22_ret = float(d22_day.iloc[0]["day_ret"]) if len(d22_day) else None
    fo_col = "sig_from_open" if (not tr.empty and "sig_from_open" in tr.columns) else None
    n_fo_ge04 = (
        int((pd.to_numeric(tr[fo_col], errors="coerce").abs() > 0.04).sum()) if fo_col else None
    )
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_exp": s.get("trade_exp"),
        "n_from_open_block": s.get("n_from_open_block"),
        "n_from_open_scale": s.get("n_from_open_scale"),
        "d22_syms": d22_syms,
        "d22_has_amd_up": amd_up,
        "d22_day_ret": d22_ret,
        "n_trades_fo_gt04": n_fo_ge04,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/from_open_gate_accept_s1_apr_jul_jan_mar_v1",
    )
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--july-start", default="2026-07-01")
    ap.add_argument("--july-end", default="2026-07-22")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args(argv)

    base = load_profile(BASE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = code_fingerprint(base.get("_profile_path") or BASE)
    arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {list(ARMS)}")
    windows = [
        ("strong", args.strong_start, args.strong_end),
        ("weak", args.weak_start, args.weak_end),
        ("july", args.july_start, args.july_end),
        ("d22", "2026-07-22", "2026-07-22"),
    ]
    rows: list[dict[str, Any]] = []
    for wname, start, end in windows:
        for arm in arms:
            tag = f"{wname}_{arm}"
            print(f"[run] {tag} {start}→{end}", flush=True)
            row = _run(_apply(base, arm), start=start, end=end, tag=tag, out=out)
            row["window"] = wname
            row["arm"] = arm
            rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']} "
                f"fo_block={row.get('n_from_open_block')} fo_scale={row.get('n_from_open_scale')} "
                f"d22={row.get('d22_syms')!r} amd={row.get('d22_has_amd_up')} "
                f"d22_ret={row.get('d22_day_ret')}",
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
        strong_pre, weak_pre, july_pre = cell("strong", "PRE"), cell("weak", "PRE"), cell("july", "PRE")
        strong_p, weak_p, july_p = cell("strong", pack), cell("weak", pack), cell("july", pack)
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
        flags.append(
            "strong_retain_ok" if strong_keep is not None and strong_keep >= 0.85 else "strong_retain_fail"
        )
        if weak_lift:
            flags.append("weak_ret_improved")
        if weak_dd_ok:
            flags.append("weak_dd_improved")
        flags.append("july_keep_ok" if july_ok else "july_keep_fail")
        d22_amd = bool(rdf[(rdf.window == "d22") & (rdf.arm == pack)]["d22_has_amd_up"].any())
        d22_ret = cell("d22", pack, "d22_day_ret")
        d22_pre = cell("d22", "PRE", "d22_day_ret")
        if not d22_amd:
            flags.append("d22_amd_filtered")
        if d22_ret is not None and d22_pre is not None and d22_ret > d22_pre + 1e-9:
            flags.append("d22_day_improved")
        if (
            "strong_retain_ok" in flags
            and ("weak_ret_improved" in flags or "weak_dd_improved" in flags)
            and "july_keep_ok" in flags
        ):
            decision = "PROMOTE_FROM_OPEN_RESEARCH"
        elif "strong_retain_ok" in flags and "july_keep_ok" in flags:
            decision = "OVERLAY_ONLY"
        else:
            decision = "REJECT_FOR_BASELINE"
        return {
            "pack": pack,
            "decision": decision,
            "flags": flags,
            "strong_keep": strong_keep,
            "strong_ret": strong_p,
            "weak_ret": weak_p,
            "july_ret": july_p,
            "d22_day_ret": d22_ret,
            "d22_has_amd_up": d22_amd,
        }

    decisions = [decide(a) for a in arms if a != "PRE"]
    summary = {
        "fingerprint": fp,
        "base_profile": BASE,
        "windows": {
            "strong": [args.strong_start, args.strong_end],
            "weak": [args.weak_start, args.weak_end],
            "july": [args.july_start, args.july_end],
        },
        "pass_rule": "strong_keep>=0.85 AND weak(ret↑|MaxDD↑) AND july_keep>=0.95",
        "decisions": decisions,
        "scoreboard": rows,
    }
    (out / "accept_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== decisions ===", flush=True)
    for d in decisions:
        print(
            f"{d['pack']}: {d['decision']} keep={d['strong_keep']} "
            f"flags={d['flags']} d22_ret={d['d22_day_ret']} amd={d['d22_has_amd_up']}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
