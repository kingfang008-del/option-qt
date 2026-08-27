#!/usr/bin/env python3
"""Dual-window accept: chop_gate calm mixed-tape overlay (RTH stock_noise).

Arms on S1 research baseline:
  PRE         = gate off
  SOFT_NOISE  = med_abs≥1% & |q_am|≤0.5% & frac∈[0.35,0.50] → scale×0.5
  HARD_NOISE  = same features, block
  SOFT_NOISE65= soft with frac_hi=0.65 (wider breadth band)
  SOFT_AM6    = soft with q_am_max=0.6%

Pass vs PRE: strong keep≥0.85 AND weak(ret↑|MaxDD↑) AND july keep≥0.95
Extra flag: chop window (Jul10–22) compound improves.
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


def _noise(**kw: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "asof": "10:30",
        "rule": "stock_noise",
        "mode": "scale",
        "scale": 0.5,
        "q_am_max": 0.005,
        "q_rng_min": 0.0,
        "frac_above_lo": 0.35,
        "frac_above_hi": 0.50,
        "med_abs_min": 0.01,
    }
    cfg.update(kw)
    return cfg


ARMS: dict[str, dict[str, Any] | None] = {
    "PRE": None,
    "SOFT_NOISE": _noise(),
    "HARD_NOISE": _noise(mode="block"),
    "SOFT_NOISE65": _noise(frac_above_hi=0.65),
    "SOFT_AM6": _noise(q_am_max=0.006),
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    cfg = ARMS[arm]
    if cfg is None:
        p.pop("chop_gate", None)
    else:
        p["chop_gate"] = dict(cfg)
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
        "trade_exp": s.get("trade_exp"),
        "n_chop_gate_block": s.get("n_chop_gate_block"),
        "n_chop_gate_scale": s.get("n_chop_gate_scale"),
        "chop_gate_day_counts": s.get("chop_gate_day_counts"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/chop_gate_accept_s1_apr_jul_jan_mar_v1",
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
        ("chop", "2026-07-10", "2026-07-22"),
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
                f"chop_block={row.get('n_chop_gate_block')} chop_scale={row.get('n_chop_gate_scale')} "
                f"days={row.get('chop_gate_day_counts')}",
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
        chop_pre = cell("chop", "PRE")
        strong_p, weak_p, july_p = cell("strong", pack), cell("weak", pack), cell("july", pack)
        chop_p = cell("chop", pack)
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
        if chop_p is not None and chop_pre is not None and chop_p > chop_pre + 1e-9:
            flags.append("chop_window_improved")
        if (
            "strong_retain_ok" in flags
            and ("weak_ret_improved" in flags or "weak_dd_improved" in flags)
            and "july_keep_ok" in flags
        ):
            decision = "PROMOTE_CHOP_RESEARCH"
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
            "chop_ret": chop_p,
        }

    decisions = [decide(a) for a in arms if a != "PRE"]
    summary = {
        "fingerprint": fp,
        "base_profile": BASE,
        "windows": {
            "strong": [args.strong_start, args.strong_end],
            "weak": [args.weak_start, args.weak_end],
            "july": [args.july_start, args.july_end],
            "chop": ["2026-07-10", "2026-07-22"],
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
            f"flags={d['flags']} chop={d['chop_ret']}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
