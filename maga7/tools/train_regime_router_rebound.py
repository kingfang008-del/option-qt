#!/usr/bin/env python3
"""Train binary Router: rebound_trap_dn vs other (enriched v2 features)."""
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


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = y_true.astype(float)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(y_score)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return None
    ranks = np.arange(1, len(y) + 1, dtype=float)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="maga7/results/regime_router/router_dataset_v2.parquet")
    ap.add_argument("--out", default="maga7/results/regime_router/router_rebound_v1.txt")
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--valid-start", default="2026-05-01")
    args = ap.parse_args()

    import lightgbm as lgb

    ds = Path(args.dataset)
    meta = json.loads(ds.with_suffix(".meta.json").read_text(encoding="utf-8"))
    feat_cols = [c for c in meta["feature_cols"] if c in pd.read_parquet(ds).columns]
    df = pd.read_parquet(ds)
    y = df["y_rebound"].astype(int).to_numpy()
    X = df[feat_cols].astype(float).fillna(0.0).to_numpy()
    dates = df["date"].astype(str)
    tr = dates <= str(args.train_end)
    va = dates >= str(args.valid_start)

    y_tr = y[tr.to_numpy()]
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    scale = n_neg / n_pos

    train_set = lgb.Dataset(X[tr.to_numpy()], label=y_tr, feature_name=feat_cols)
    valid_set = lgb.Dataset(X[va.to_numpy()], label=y[va.to_numpy()], feature_name=feat_cols, reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.03,
        "num_leaves": 12,
        "min_data_in_leaf": 8,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "scale_pos_weight": float(scale),
        "verbosity": -1,
        "seed": 7,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    p_tr = booster.predict(X[tr.to_numpy()])
    p_va = booster.predict(X[va.to_numpy()])
    y_va = y[va.to_numpy()]
    sweeps = []
    for thr in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        pred = p_va >= thr
        tp = int((pred & (y_va == 1)).sum())
        fp = int((pred & (y_va == 0)).sum())
        fn = int((~pred & (y_va == 1)).sum())
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

    # also evaluate causal rule baselines on valid
    va_df = df.loc[va].copy()
    rules = {
        "low_open_reclaim": (va_df["qqq_low_open_reclaim"] >= 0.5) & (va_df["qqq_bounce_lod"] >= 0.008),
        "reclaim_bounce012": (va_df["qqq_low_open_reclaim"] >= 0.5) & (va_df["qqq_bounce_lod"] >= 0.012),
        "above_bounce012": (va_df["qqq_above_open"] >= 0.5) & (va_df["qqq_bounce_lod"] >= 0.012),
    }
    rule_eval = {}
    for name, m in rules.items():
        m = m.to_numpy()
        tp = int((m & (y_va == 1)).sum())
        fp = int((m & (y_va == 0)).sum())
        fn = int((~m & (y_va == 1)).sum())
        rule_eval[name] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "n_flag": int(m.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    imp = dict(zip(feat_cols, [float(x) for x in booster.feature_importance("gain")]))
    out = Path(args.out)
    booster.save_model(str(out))
    out_meta = {
        "mode": "binary_rebound",
        "feature_cols": feat_cols,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_pos_train": int(y_tr.sum()),
        "n_pos_valid": int(y_va.sum()),
        "best_iteration": int(booster.best_iteration or 300),
        "auc_train": _auc(y_tr, p_tr),
        "auc_valid": _auc(y_va, p_va),
        "threshold_sweep_valid": sweeps,
        "rule_baselines_valid": rule_eval,
        "feature_importance_top": dict(sorted(imp.items(), key=lambda x: -x[1])[:12]),
        "dataset": str(ds),
        "expert_on_fire": "rebound_trap_dn",
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(out_meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
