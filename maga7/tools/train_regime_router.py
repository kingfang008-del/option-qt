#!/usr/bin/env python3
"""Train LightGBM regime Router (multiclass day_type) with walk-forward split.

Saves model + meta loadable by predicted-route scoreboard.
Inference: argmax class if max_proba >= p_min else baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = ["baseline", "rebound_trap_dn", "dn_toxic", "up_toxic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset.parquet")
    ap.add_argument("--out", default="maga7/results/regime_router/router_lgbm_v1.txt")
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--valid-start", default="2026-05-01")
    ap.add_argument("--num-leaves", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import lightgbm as lgb

    ds = Path(args.dataset)
    meta = json.loads(ds.with_suffix(".meta.json").read_text(encoding="utf-8"))
    feat_cols = list(meta["feature_cols"])
    df = pd.read_parquet(ds)
    df["y_type"] = df["y_type"].astype(str)
    # map unknown → baseline
    df.loc[~df["y_type"].isin(CLASSES), "y_type"] = "baseline"
    class_to_id = {c: i for i, c in enumerate(CLASSES)}
    y = df["y_type"].map(class_to_id).astype(int).to_numpy()
    X = df[feat_cols].astype(float).to_numpy()
    dates = df["date"].astype(str)

    tr = dates <= str(args.train_end)
    va = dates >= str(args.valid_start)
    if int(tr.sum()) < 50 or int(va.sum()) < 20:
        raise SystemExit(f"split too small train={tr.sum()} valid={va.sum()}")

    # class weights inverse frequency on train
    y_tr = y[tr.to_numpy()]
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(float)
    counts = np.maximum(counts, 1.0)
    w_per_class = counts.sum() / (len(CLASSES) * counts)
    w_tr = w_per_class[y_tr]

    train_set = lgb.Dataset(
        X[tr.to_numpy()],
        label=y_tr,
        weight=w_tr,
        feature_name=feat_cols,
    )
    valid_set = lgb.Dataset(
        X[va.to_numpy()],
        label=y[va.to_numpy()],
        feature_name=feat_cols,
        reference=train_set,
    )
    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": int(args.seed),
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(args.n_estimators),
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    def _eval(mask: pd.Series) -> dict:
        proba = booster.predict(X[mask.to_numpy()])
        pred = proba.argmax(axis=1)
        yt = y[mask.to_numpy()]
        acc = float((pred == yt).mean())
        # expert recall: among true expert days, fraction predicted as any expert
        expert_ids = {1, 2, 3}
        true_exp = np.isin(yt, list(expert_ids))
        pred_exp = np.isin(pred, list(expert_ids))
        recall = float((pred_exp & true_exp).sum() / max(true_exp.sum(), 1))
        precision = float((pred_exp & true_exp).sum() / max(pred_exp.sum(), 1))
        # per-class
        per = {}
        for c, i in class_to_id.items():
            m = yt == i
            if m.sum() == 0:
                per[c] = {"n": 0, "recall": None}
            else:
                per[c] = {"n": int(m.sum()), "recall": float((pred[m] == i).sum() / m.sum())}
        # threshold sweeps for "route if maxp>=thr and class!=baseline"
        sweeps = []
        for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            mx = proba.max(axis=1)
            cls = proba.argmax(axis=1)
            route = np.where((mx >= thr) & (cls != 0), cls, 0)
            n_route = int((route != 0).sum())
            hit = int(((route != 0) & true_exp & (route == yt)).sum())
            false_route = int(((route != 0) & (~true_exp)).sum())
            sweeps.append(
                {
                    "p_min": thr,
                    "n_route": n_route,
                    "expert_exact_hit": hit,
                    "false_route_on_baseline": false_route,
                    "expert_recall_any": float(((route != 0) & true_exp).sum() / max(true_exp.sum(), 1)),
                }
            )
        return {
            "n": int(mask.sum()),
            "acc": acc,
            "expert_recall_any": recall,
            "expert_precision_any": precision,
            "per_class": per,
            "threshold_sweep": sweeps,
        }

    metrics = {"train": _eval(tr), "valid": _eval(va)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    out_meta = {
        "feature_cols": feat_cols,
        "classes": CLASSES,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "best_iteration": int(booster.best_iteration or args.n_estimators),
        "class_counts_train": {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
        "metrics": metrics,
        "dataset": str(ds),
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(out_meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
