#!/usr/bin/env python3
"""Ablate soft DN sizing when QQQ > today's open (vs hard block)."""
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


def _run(prof: dict, *, start: str, end: str, tag: str, out: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["trades"].to_csv(sub / "trades.csv", index=False)
    res["daily"].to_csv(sub / "daily.csv", index=False)
    d0717 = None
    daily = res["daily"]
    if not daily.empty and "date" in daily.columns:
        hit = daily[daily["date"].astype(str) == "2026-07-17"]
        if not hit.empty and "day_ret" in hit.columns:
            d0717 = float(hit.iloc[0]["day_ret"])
        elif not hit.empty and "ret" in hit.columns:
            d0717 = float(hit.iloc[0]["ret"])
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_regime_scale": s.get("n_regime_scale"),
        "n_regime_block": s.get("n_regime_block"),
        "day_ret_0717": d0717,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--out", default="maga7/results/dn_qqq_scale_ablation_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    args = ap.parse_args()

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    variants = [
        ("baseline", {}),
        ("scale50", {"scale_dn_if_qqq_above_open": 0.5}),
        ("scale25", {"scale_dn_if_qqq_above_open": 0.25}),
        ("block", {"block_dn_if_qqq_above_open": True}),
    ]
    board = []
    for wname, start, end in windows:
        base_ret = None
        for vname, reg in variants:
            p = copy.deepcopy(base)
            p.setdefault("regime", {}).update(reg)
            # ensure hard-block off when testing scale
            if "scale_dn_if_qqq_above_open" in reg:
                p["regime"]["block_dn_if_qqq_above_open"] = False
            row = _run(p, start=start, end=end, tag=f"{wname}__{vname}", out=out)
            if vname == "baseline":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_baseline"] = (row["total_ret"] / base_ret) if base_ret else None
            board.append(row)

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(board).to_csv(out / "scoreboard.csv", index=False)
    cols = ["window", "variant", "total_ret", "maxdd", "n_trades", "vs_baseline", "day_ret_0717", "n_regime_scale"]
    print(pd.DataFrame(board)[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
