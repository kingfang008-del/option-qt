#!/usr/bin/env python3
"""Scoreboard: causal Router predictions → expert overlays (vs baseline / oracle)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _predict_labels(
    *,
    dataset: Path,
    model: Path,
    p_min: float,
) -> dict[str, str]:
    import lightgbm as lgb

    meta = json.loads(model.with_suffix(model.suffix + ".meta.json").read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    feat_cols = list(meta["feature_cols"])
    df = pd.read_parquet(dataset)
    booster = lgb.Booster(model_file=str(model))
    X = df[feat_cols].astype(float).to_numpy()
    proba = booster.predict(X)
    out = {}
    for i, r in enumerate(df.itertuples(index=False)):
        mx = float(proba[i].max())
        cls = int(proba[i].argmax())
        name = classes[cls]
        if name == "baseline" or mx < float(p_min):
            continue  # omit → replay treats as baseline
        out[str(r.date)] = name
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset.parquet")
    ap.add_argument("--model", default="maga7/results/regime_router/router_lgbm_v1.txt")
    ap.add_argument("--experts", default="maga7/CONFIG/regime_router/experts_v1.json")
    ap.add_argument("--oracle-labels", default="maga7/results/regime_router/day_type_labels.csv")
    ap.add_argument("--p-min", type=float, nargs="+", default=[0.40, 0.45, 0.50])
    ap.add_argument("--out", default="maga7/results/regime_router/predicted_scoreboard")
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
    board = []
    for wname, start, end in windows:
        row_b = _run(base, start=start, end=end, tag=f"{wname}__baseline", out=out)
        base_ret = row_b["total_ret"]
        board.append({**row_b, "window": wname, "variant": "baseline", "vs_baseline": 1.0, "p_min": None})

        # oracle reference
        p_o = copy.deepcopy(base)
        p_o["regime_router"] = {
            "enabled": True,
            "labels_path": str(args.oracle_labels),
            "experts_path": str(args.experts),
        }
        row_o = _run(p_o, start=start, end=end, tag=f"{wname}__oracle", out=out)
        board.append(
            {
                **row_o,
                "window": wname,
                "variant": "oracle",
                "vs_baseline": row_o["total_ret"] / base_ret if base_ret else None,
                "p_min": None,
            }
        )

        for thr in args.p_min:
            pred = _predict_labels(dataset=Path(args.dataset), model=Path(args.model), p_min=thr)
            # restrict labels to window for cleanliness (optional)
            pred_w = {d: t for d, t in pred.items() if start <= d <= end}
            p = copy.deepcopy(base)
            p["regime_router"] = {
                "enabled": True,
                "labels": pred_w,
                "experts_path": str(args.experts),
            }
            row = _run(p, start=start, end=end, tag=f"{wname}__pred_p{int(thr*100):02d}", out=out)
            board.append(
                {
                    **row,
                    "window": wname,
                    "variant": f"pred_p{thr:.2f}",
                    "vs_baseline": row["total_ret"] / base_ret if base_ret else None,
                    "p_min": thr,
                    "n_pred_labels": len(pred_w),
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
            "n_trades",
            "vs_baseline",
            "day_ret_0717",
            "n_router_expert_days",
            "n_pred_labels",
        )
        if c in pd.DataFrame(board).columns
    ]
    print(pd.DataFrame(board)[cols].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
