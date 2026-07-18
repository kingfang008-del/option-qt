#!/usr/bin/env python3
"""Train LightGBM Smart Bouncer (P(allow) = 1 - P(toxic)).

Walk-forward: ``--train-end`` / ``--valid-start``. Prefer option-labeled rows
when enough; otherwise use all rows (option + underlying fallback).
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

from maga7.common.lgbm_bouncer import FEATURE_COLS, save_lgbm_model


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
    # Mann–Whitney
    sum_ranks_pos = ranks[y > 0.5].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="maga7/results/lgbm_bouncer/dataset_rule_a.parquet")
    ap.add_argument("--out", default="maga7/results/lgbm_bouncer/lgbm_bouncer_v1.txt")
    ap.add_argument("--train-end", default="2026-04-30")
    ap.add_argument("--valid-start", default="2026-05-01")
    ap.add_argument("--option-only", action="store_true", help="train/eval on option labels only")
    ap.add_argument(
        "--direction",
        default=None,
        help="filter to one direction, e.g. DN (case-insensitive)",
    )
    ap.add_argument(
        "--toxic-mae",
        type=float,
        default=None,
        help="relabel toxic from opt_mae >= thresh (option rows); keep y_ternary fallback otherwise",
    )
    ap.add_argument(
        "--target",
        default="toxic",
        choices=["allow", "toxic"],
        help="toxic=P(toxic) then p_allow=1-p; allow=direct P(allow)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--min-data-in-leaf", type=int, default=20)
    ap.add_argument("--no-early-stop", action="store_true")
    args = ap.parse_args()

    import lightgbm as lgb

    ds = Path(args.dataset)
    df = pd.read_parquet(ds)
    if args.option_only:
        df = df[df["label_src"] == "option"].copy()
        if df.empty:
            raise SystemExit("no option-labeled rows")
    if args.direction:
        df = df[df["direction"].astype(str).str.upper() == str(args.direction).upper()].copy()
        if df.empty:
            raise SystemExit(f"no rows for direction={args.direction}")
    if args.toxic_mae is not None:
        mae = pd.to_numeric(df.get("opt_mae"), errors="coerce")
        toxic = mae >= float(args.toxic_mae)
        # option rows: override; non-option keep original ternary toxic
        is_opt = df["label_src"].astype(str) == "option"
        y_allow = df["y_allow"].astype(int).to_numpy().copy()
        y_allow[is_opt.to_numpy()] = (~toxic[is_opt]).astype(int).to_numpy()
        df = df.copy()
        df["y_allow"] = y_allow
        df["y_ternary"] = np.where(is_opt & toxic, -1, df["y_ternary"])

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    dates = df["date"].astype(str)
    tr = dates <= str(args.train_end)
    va = dates >= str(args.valid_start)
    if int(tr.sum()) < 30 or int(va.sum()) < 10:
        raise SystemExit(f"split too small train={tr.sum()} valid={va.sum()}")

    y_allow_tr = df.loc[tr, "y_allow"].astype(int).to_numpy()
    y_allow_va = df.loc[va, "y_allow"].astype(int).to_numpy()
    if args.target == "toxic":
        y_tr = 1 - y_allow_tr
        y_va = 1 - y_allow_va
    else:
        y_tr = y_allow_tr
        y_va = y_allow_va

    X_tr = df.loc[tr, feat_cols].astype(float).to_numpy()
    X_va = df.loc[va, feat_cols].astype(float).to_numpy()

    n_pos = int((y_tr == 1).sum())  # minority of interest when target=toxic
    n_neg = int((y_tr == 0).sum())
    scale_pos = (n_neg / max(n_pos, 1)) if n_pos else 1.0

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=list(feat_cols))
    valid_set = lgb.Dataset(X_va, label=y_va, feature_name=list(feat_cols), reference=train_set)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": int(args.seed),
        "scale_pos_weight": float(scale_pos),
    }
    callbacks = [lgb.log_evaluation(period=0)]
    if not args.no_early_stop:
        callbacks.insert(0, lgb.early_stopping(60, verbose=False))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(args.n_estimators),
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    raw_tr = booster.predict(X_tr)
    raw_va = booster.predict(X_va)
    # Always report / save in p_allow space for replay gate.
    if args.target == "toxic":
        p_tr = 1.0 - raw_tr
        p_va = 1.0 - raw_va
    else:
        p_tr = raw_tr
        p_va = raw_va
    auc_tr = _auc(y_allow_tr, p_tr)
    auc_va = _auc(y_allow_va, p_va)
    auc_toxic_va = _auc(1 - y_allow_va, 1.0 - p_va)

    sweeps = []
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        keep = p_va >= thr
        if keep.sum() == 0:
            sweeps.append({"p_min": thr, "keep_rate": 0.0, "allow_rate_kept": None})
            continue
        sweeps.append(
            {
                "p_min": thr,
                "keep_rate": float(keep.mean()),
                "allow_rate_kept": float(y_allow_va[keep].mean()),
                "toxic_catch_rate": float(((~keep) & (y_allow_va == 0)).sum() / max((y_allow_va == 0).sum(), 1)),
            }
        )

    out = Path(args.out)
    meta = {
        "feature_cols": feat_cols,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "option_only": bool(args.option_only),
        "direction_filter": str(args.direction).upper() if args.direction else None,
        "toxic_mae_relabel": args.toxic_mae,
        "target": args.target,
        "predict_space": "p_allow",
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_pos_train": n_pos,
        "n_neg_train": n_neg,
        "scale_pos_weight": scale_pos,
        "best_iteration": int(getattr(booster, "best_iteration", None) or args.n_estimators),
        "auc_train_allow": auc_tr,
        "auc_valid_allow": auc_va,
        "auc_valid_toxic": auc_toxic_va,
        "p_allow_valid_stats": {
            "min": float(np.min(p_va)),
            "p50": float(np.median(p_va)),
            "max": float(np.max(p_va)),
            "std": float(np.std(p_va)),
        },
        "threshold_sweep_valid": sweeps,
        "feature_importance": dict(
            zip(feat_cols, [float(x) for x in booster.feature_importance("gain")])
        ),
        "dataset": str(ds),
    }
    # Wrap toxic models so load_lgbm_bouncer still sees P(allow).
    if args.target == "toxic":
        meta["model_outputs"] = "p_toxic"
        meta["inference_note"] = "load_lgbm_bouncer expects p_allow; use LgbmToxicWrapper or retrain --target allow"

    save_lgbm_model(booster, out, meta=meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
