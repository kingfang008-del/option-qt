#!/usr/bin/env python3
"""Train binary need_expert Router + optional subtype heuristic metadata."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset.parquet")
    ap.add_argument("--out", default="maga7/results/regime_router/router_binary_v1.txt")
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--valid-start", default="2026-05-01")
    args = ap.parse_args()

    import lightgbm as lgb

    ds = Path(args.dataset)
    meta = json.loads(ds.with_suffix(".meta.json").read_text(encoding="utf-8"))
    feat_cols = list(meta["feature_cols"])
    df = pd.read_parquet(ds)
    y = df["y_need_expert"].astype(int).to_numpy()
    X = df[feat_cols].astype(float).to_numpy()
    dates = df["date"].astype(str)
    tr = dates <= str(args.train_end)
    va = dates >= str(args.valid_start)

    y_tr = y[tr.to_numpy()]
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int(len(y_tr) - y_tr.sum()), 1)
    scale = n_neg / n_pos

    train_set = lgb.Dataset(X[tr.to_numpy()], label=y_tr, feature_name=feat_cols)
    valid_set = lgb.Dataset(X[va.to_numpy()], label=y[va.to_numpy()], feature_name=feat_cols, reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 16,
        "min_data_in_leaf": 12,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": float(scale),
        "verbosity": -1,
        "seed": 7,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=250,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    def _auc(yt, ys):
        order = np.argsort(ys)
        y = yt[order].astype(float)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos <= 0 or n_neg <= 0:
            return None
        ranks = np.arange(1, len(y) + 1, dtype=float)
        return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    p_va = booster.predict(X[va.to_numpy()])
    y_va = y[va.to_numpy()]
    sweeps = []
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        pred = p_va >= thr
        tp = int(((pred) & (y_va == 1)).sum())
        fp = int(((pred) & (y_va == 0)).sum())
        fn = int(((~pred) & (y_va == 1)).sum())
        sweeps.append(
            {
                "p_min": thr,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "n_flag": int(pred.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    out = Path(args.out)
    booster.save_model(str(out))
    out_meta = {
        "mode": "binary_need_expert",
        "feature_cols": feat_cols,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "best_iteration": int(booster.best_iteration or 250),
        "auc_valid": _auc(y_va, p_va),
        "auc_train": _auc(y_tr, booster.predict(X[tr.to_numpy()])),
        "threshold_sweep_valid": sweeps,
        "subtype_heuristic": "if need: rebound if qqq_above_open&bounce>=0.5%; elif frac_dn>=0.5 -> dn_toxic; else up_toxic",
        "dataset": str(ds),
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(out_meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
