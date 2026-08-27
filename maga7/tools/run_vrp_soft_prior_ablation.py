#!/usr/bin/env python3
"""Ablate buyer-side VRP-lite soft prior on research_baseline (dual window).

Stock fact: ``paths.stock_1s_root`` → 1s→1m (not left-labeled spnq_train).
IV: QQQ bucketed_v7 surface @ 10:30. RV: QQQ RTH last print from 1s.

Variants:
  off     — no VRP prior
  scale50 — rich IV−RV → size ×0.5
  skip    — rich IV−RV → skip entry
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
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    regime_gate_from_1s,
    session_dates,
)
from maga7.common.vrp_prior import DEFAULT_STOCK_1S_ROOT, build_vrp_day_table, parse_vrp_size_scale


VRP_SCALE50 = {
    "enabled": True,
    "asof": "10:30",
    "rv_lookback_days": 5,
    "mode": "scale",
    "scale": 0.5,
    "rich_pctile": 0.70,
    "rich_min": 0.0,
    "missing": "passthrough",
    "stock_1s_root": str(DEFAULT_STOCK_1S_ROOT),
}
VRP_SKIP = {**VRP_SCALE50, "mode": "skip"}


def _slice_stock_by(
    stock_by: dict[str, pd.DataFrame], start: str, end: str
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, df in stock_by.items():
        if df is None or df.empty:
            continue
        m = (df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)
        sub = df.loc[m].copy()
        if not sub.empty:
            out[sym] = sub
    return out


def _run(
    prof: dict,
    *,
    start: str,
    end: str,
    tag: str,
    out: Path,
    stock_by: dict[str, pd.DataFrame],
    regime_gate: Any,
) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    # Causal completed 1m from 1s aggregate
    p.setdefault("trade", {})["bar_availability_delay_seconds"] = int(
        p.get("trade", {}).get("bar_availability_delay_seconds") or 60
    )
    sb = _slice_stock_by(stock_by, start, end)
    res = run_offline_replay(p, scheme="single", stock_by=sb, regime_gate=regime_gate)
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["trades"].to_csv(sub / "trades.csv", index=False)
    res["daily"].to_csv(sub / "daily.csv", index=False)
    return {
        "tag": tag,
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_vrp_size_scale": s.get("n_vrp_size_scale"),
        "n_vrp_skip": s.get("n_vrp_skip"),
    }


def _rich_day_count(prof: dict, start: str, end: str) -> dict:
    paths = prof.get("_paths") or {}
    s1s = paths.get("stock_1s_root") or DEFAULT_STOCK_1S_ROOT
    cfg = parse_vrp_size_scale({**VRP_SCALE50, "stock_1s_root": str(s1s)})
    tab = build_vrp_day_table(
        qqq_df=None, start=start, end=end, cfg=cfg, stock_1s_root=s1s
    )
    if tab is None or tab.empty:
        return {"n_days": 0, "n_rich": 0, "n_vrp_ok": 0, "stock_1s_root": str(s1s)}
    return {
        "n_days": int(len(tab)),
        "n_rich": int(tab["rich"].fillna(False).sum()),
        "n_vrp_ok": int(tab["vrp"].notna().sum()),
        "vrp_mean": float(tab["vrp"].mean(skipna=True)) if tab["vrp"].notna().any() else None,
        "stock_1s_root": str(s1s),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/research_vrp_soft_prior_dual_v1",
    )
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-09")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument(
        "--warm-days",
        type=int,
        default=40,
        help="Extra calendar pad before window for mf/regime/VRP warm-up",
    )
    args = ap.parse_args()

    base = load_profile(args.profile)
    base.setdefault("trade", {}).pop("vrp_size_scale", None)
    # Point stock fact to 1s (profile already has stock_1s_root)
    s1s = (base.get("_paths") or {}).get("stock_1s_root") or DEFAULT_STOCK_1S_ROOT
    base.setdefault("_paths", {})["stock_1s_root"] = Path(s1s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = [
        ("strong_may_jul9", args.strong_start, args.strong_end),
        ("weak_jan_mar", args.weak_start, args.weak_end),
    ]
    variants = [
        ("off", None),
        ("scale50", VRP_SCALE50),
        ("skip", VRP_SKIP),
    ]

    diag = {}
    for wname, start, end in windows:
        print(f"[diag] {wname} …", flush=True)
        diag[wname] = _rich_day_count(base, start, end)
    (out / "vrp_diag.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    print("VRP diag:", json.dumps(diag, indent=2, default=str))
    if args.diag_only:
        return

    board = []
    for wname, start, end in windows:
        warm0 = str(pd.Timestamp(start) - pd.Timedelta(days=int(args.warm_days) * 2))[:10]
        dates = session_dates(warm0, end)
        print(
            f"[build 1s→1m] {wname} {warm0}..{end} sessions={len(dates)} root={s1s}",
            flush=True,
        )
        build_prof = copy.deepcopy(base)
        build_prof["date_range"] = {"start": warm0, "end": end}
        stock_by = build_stock_by_from_1s(build_prof, dates=dates, include_refs=True)
        regime_gate = regime_gate_from_1s(build_prof, stock_by)
        print(
            f"  bars={sum(len(v) for v in stock_by.values())} symbols={sorted(stock_by)}",
            flush=True,
        )

        base_ret = None
        for vname, vrp in variants:
            p = copy.deepcopy(base)
            if vrp is None:
                p.setdefault("trade", {}).pop("vrp_size_scale", None)
            else:
                p.setdefault("trade", {})["vrp_size_scale"] = dict(vrp)
            print(f"=== {wname} / {vname} {start}→{end} ===", flush=True)
            row = _run(
                p,
                start=start,
                end=end,
                tag=f"{wname}__{vname}",
                out=out,
                stock_by=stock_by,
                regime_gate=regime_gate,
            )
            if vname == "off":
                base_ret = row["total_ret"]
            row["window"] = wname
            row["variant"] = vname
            row["vs_off"] = (row["total_ret"] - base_ret) if base_ret is not None else None
            row["rich_days"] = diag.get(wname, {}).get("n_rich")
            board.append(row)

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(board).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        "window",
        "variant",
        "total_ret",
        "maxdd",
        "n_trades",
        "vs_off",
        "n_vrp_size_scale",
        "n_vrp_skip",
        "rich_days",
    ]
    print(pd.DataFrame(board)[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
