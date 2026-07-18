#!/usr/bin/env python3
"""Scoreboard: rebound Router (LGBM / causal rules) vs baseline / oracle-rebound-only."""
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
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    d0717 = None
    hit = res["daily"][res["daily"]["date"].astype(str) == "2026-07-17"]
    if len(hit):
        d0717 = float(hit.iloc[0]["day_ret"])
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_router_expert_days": s.get("n_router_expert_days"),
        "router_day_counts": s.get("router_day_counts"),
        "day_ret_0717": d0717,
    }


def _lgbm_labels(dataset: Path, model: Path, p_min: float) -> dict[str, str]:
    import lightgbm as lgb

    meta = json.loads(model.with_suffix(model.suffix + ".meta.json").read_text(encoding="utf-8"))
    feat_cols = list(meta["feature_cols"])
    df = pd.read_parquet(dataset)
    booster = lgb.Booster(model_file=str(model))
    p = booster.predict(df[feat_cols].astype(float).fillna(0.0).to_numpy())
    out = {}
    for i, r in enumerate(df.itertuples(index=False)):
        if float(p[i]) >= float(p_min):
            out[str(r.date)] = "rebound_trap_dn"
    return out


def _rule_labels(dataset: Path, rule: str) -> dict[str, str]:
    df = pd.read_parquet(dataset)
    if rule == "low_open_reclaim":
        m = (df["qqq_low_open_reclaim"] >= 0.5) & (df["qqq_bounce_lod"] >= 0.008)
    elif rule == "reclaim_bounce012":
        m = (df["qqq_low_open_reclaim"] >= 0.5) & (df["qqq_bounce_lod"] >= 0.012)
    elif rule == "above_bounce012":
        m = (df["qqq_above_open"] >= 0.5) & (df["qqq_bounce_lod"] >= 0.012)
    else:
        raise SystemExit(f"unknown rule {rule}")
    return {str(d): "rebound_trap_dn" for d in df.loc[m, "date"].astype(str)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset_v2.parquet")
    ap.add_argument("--model", default="maga7/results/regime_router/router_rebound_v1.txt")
    ap.add_argument("--experts", default="maga7/CONFIG/regime_router/experts_v1.json")
    ap.add_argument("--oracle-labels", default="maga7/results/regime_router/day_type_labels.csv")
    ap.add_argument("--p-min", type=float, nargs="+", default=[0.20, 0.30, 0.40])
    ap.add_argument("--out", default="maga7/results/regime_router/rebound_scoreboard")
    args = ap.parse_args()

    # rebound-only oracle labels
    lab = pd.read_csv(args.oracle_labels)
    oracle_reb = {
        str(r.date): "rebound_trap_dn"
        for r in lab.itertuples(index=False)
        if str(r.day_type) == "rebound_trap_dn"
    }

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = [
        ("strong_may_jul", "2026-05-01", "2026-07-17"),
        ("weak_feb_apr", "2026-02-01", "2026-04-30"),
    ]
    board = []
    for wname, start, end in windows:
        row_b = _run(base, start=start, end=end, tag=f"{wname}__baseline", out=out)
        br = row_b["total_ret"]
        board.append({**row_b, "window": wname, "variant": "baseline", "vs_baseline": 1.0, "n_labels": 0})

        p_o = copy.deepcopy(base)
        p_o["regime_router"] = {
            "enabled": True,
            "labels": {d: t for d, t in oracle_reb.items() if start <= d <= end},
            "experts_path": str(args.experts),
        }
        row_o = _run(p_o, start=start, end=end, tag=f"{wname}__oracle_reb", out=out)
        board.append(
            {
                **row_o,
                "window": wname,
                "variant": "oracle_rebound",
                "vs_baseline": row_o["total_ret"] / br,
                "n_labels": len(p_o["regime_router"]["labels"]),
            }
        )

        for rule in ("low_open_reclaim", "reclaim_bounce012", "above_bounce012"):
            labs = {d: t for d, t in _rule_labels(Path(args.dataset), rule).items() if start <= d <= end}
            p = copy.deepcopy(base)
            p["regime_router"] = {"enabled": True, "labels": labs, "experts_path": str(args.experts)}
            row = _run(p, start=start, end=end, tag=f"{wname}__rule_{rule}", out=out)
            board.append(
                {
                    **row,
                    "window": wname,
                    "variant": f"rule_{rule}",
                    "vs_baseline": row["total_ret"] / br,
                    "n_labels": len(labs),
                }
            )

        if Path(args.model).is_file():
            for thr in args.p_min:
                labs = {
                    d: t
                    for d, t in _lgbm_labels(Path(args.dataset), Path(args.model), thr).items()
                    if start <= d <= end
                }
                p = copy.deepcopy(base)
                p["regime_router"] = {"enabled": True, "labels": labs, "experts_path": str(args.experts)}
                row = _run(p, start=start, end=end, tag=f"{wname}__lgbm_p{int(thr*100):02d}", out=out)
                board.append(
                    {
                        **row,
                        "window": wname,
                        "variant": f"lgbm_p{thr:.2f}",
                        "vs_baseline": row["total_ret"] / br,
                        "n_labels": len(labs),
                    }
                )

    (out / "scoreboard.json").write_text(json.dumps(board, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(board).to_csv(out / "scoreboard.csv", index=False)
    cols = [
        c
        for c in (
            "window",
            "variant",
            "total_ret",
            "maxdd",
            "vs_baseline",
            "day_ret_0717",
            "n_router_expert_days",
            "n_labels",
        )
        if c in pd.DataFrame(board).columns
    ]
    print(pd.DataFrame(board)[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
