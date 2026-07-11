#!/usr/bin/env python3
"""Validate learnability of 0DTE side/bucket/horizon routing from option-edge labels.

Uses microstructure minute features to predict routing targets derived from
strict bid/ask option-edge labels. This is the model-learnability gate before
building a heavier TFT routing stack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from factor_lab.tools.option_edge_routing_common import (
    build_routing_dataset,
    month_values,
    routing_feature_matrix,
    top_fraction_metrics,
)


def evaluate_regression(pred: np.ndarray, y: np.ndarray, months: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(y)
    pred, y, months = pred[m], y[m], months[m]
    ic = float(spearmanr(pred, y).statistic) if len(y) > 2 else 0.0
    out = {
        "n": int(len(y)),
        "ic": ic,
        "top20": top_fraction_metrics(np.abs(pred), y, frac=0.2),
        "top10": top_fraction_metrics(np.abs(pred), y, frac=0.1),
    }
    per_month = {}
    for mon in np.unique(months):
        mm = months == mon
        if mm.sum() < 50:
            continue
        per_month[str(mon)] = top_fraction_metrics(np.abs(pred[mm]), y[mm], frac=0.2)
    out["per_month"] = per_month
    return out


def evaluate_classifier(pred: np.ndarray, y: np.ndarray, months: np.ndarray) -> dict:
    m = np.isfinite(y)
    pred, y, months = pred[m], y[m], months[m]
    out = {
        "n": int(len(y)),
        "acc": float(np.mean(pred == y)),
        "top20": top_fraction_metrics((pred == y).astype(float), y.astype(float), frac=0.2),
    }
    per_month = {}
    for mon in np.unique(months):
        mm = months == mon
        if mm.sum() < 50:
            continue
        per_month[str(mon)] = {"acc": float(np.mean(pred[mm] == y[mm]))}
    out["per_month"] = per_month
    return out


def fit_side_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], horizon: int) -> dict:
    target = f"gap_{horizon}m"
    call_col = f"best_call_ret_{horizon}m"
    put_col = f"best_put_ret_{horizon}m"
    tr = train.dropna(subset=[target, call_col, put_col]).copy()
    te = test.dropna(subset=[target, call_col, put_col]).copy()
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    X_tr = routing_feature_matrix(tr, features)
    model.fit(X_tr, tr[target].values)
    out = {"horizon": horizon, "train_rows": int(len(tr)), "test_rows": int(len(te))}
    for name, df in [("train", tr), ("test", te)]:
        pred = model.predict(routing_feature_matrix(df, features))
        months = month_values(df)
        side_true = (df[target].values > 0).astype(int)
        side_pred = (pred > 0).astype(int)
        chosen = np.where(pred > 0, df[call_col].values, df[put_col].values)
        eval_out = evaluate_regression(pred, df[target].values, months)
        eval_out["side_acc"] = float(np.mean(side_pred == side_true))
        top = np.abs(pred) >= np.quantile(np.abs(pred), 0.8)
        eval_out["top20_chosen_mean"] = float(np.mean(chosen[top]))
        eval_out["top20_chosen_hit"] = float(np.mean(chosen[top] > 0))
        out[name] = eval_out
    return out


def fit_bucket_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    tr = train.dropna(subset=["label_bucket"]).copy()
    te = test.dropna(subset=["label_bucket"]).copy()
    y_tr = tr["label_bucket"].astype(int).values
    y_te = te["label_bucket"].astype(int).values
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(routing_feature_matrix(tr, features), y_tr)
    pred_tr = model.predict(routing_feature_matrix(tr, features))
    pred_te = model.predict(routing_feature_matrix(te, features))
    return {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "train": evaluate_classifier(pred_tr, y_tr, month_values(tr)),
        "test": evaluate_classifier(pred_te, y_te, month_values(te)),
    }


def fit_horizon_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    tr = train.dropna(subset=["label_horizon"]).copy()
    te = test.dropna(subset=["label_horizon"]).copy()
    y_tr = tr["label_horizon"].astype(int).values
    y_te = te["label_horizon"].astype(int).values
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(routing_feature_matrix(tr, features), y_tr)
    pred_tr = model.predict(routing_feature_matrix(tr, features))
    pred_te = model.predict(routing_feature_matrix(te, features))
    return {
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "train": evaluate_classifier(pred_tr, y_tr, month_values(tr)),
        "test": evaluate_classifier(pred_te, y_te, month_values(te)),
    }


def fit_edge_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    tr = train.dropna(subset=["label_edge"]).copy()
    te = test.dropna(subset=["label_edge"]).copy()
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(routing_feature_matrix(tr, features), tr["label_edge"].values)
    out = {"train_rows": int(len(tr)), "test_rows": int(len(te))}
    for name, df in [("train", tr), ("test", te)]:
        pred = model.predict(routing_feature_matrix(df, features))
        out[name] = evaluate_regression(pred, df["label_edge"].values, month_values(df))
    return out


def fit_joint_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], horizon: int) -> dict:
    """Joint routing quality: side + bucket + horizon at a fixed execution horizon."""
    tr = train.dropna(subset=["label_side", "label_bucket", "label_horizon", "label_edge"]).copy()
    te = test.dropna(subset=["label_side", "label_bucket", "label_horizon", "label_edge"]).copy()
    side_model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    bucket_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=43,
    )
    horizon_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=44,
    )
    edge_model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=45,
    )
    X_tr = routing_feature_matrix(tr, features)
    X_te = routing_feature_matrix(te, features)
    side_model.fit(X_tr, tr[f"gap_{horizon}m"].fillna(0.0).values)
    bucket_model.fit(X_tr, tr["label_bucket"].astype(int).values)
    horizon_model.fit(X_tr, tr["label_horizon"].astype(int).values)
    edge_model.fit(X_tr, tr["label_edge"].values)

    side_pred = side_model.predict(X_te)
    bucket_pred = bucket_model.predict(X_te)
    horizon_pred = horizon_model.predict(X_te)
    edge_pred = edge_model.predict(X_te)

    side_true = np.where(tr[f"gap_{horizon}m"].fillna(0.0).values > 0, 1, 0)
    side_true_te = np.where(te[f"gap_{horizon}m"].fillna(0.0).values > 0, 1, 0)
    side_pred_bin = (side_pred > 0).astype(int)

    realized = []
    for row, sp, bp, hp in zip(te.itertuples(index=False), side_pred_bin, bucket_pred, horizon_pred):
        side = "CALL" if sp == 1 else "PUT"
        h = int(hp)
        ret_col = f"best_{side.lower()}_ret_{h}m"
        bucket_col = f"best_{side.lower()}_bucket_{h}m"
        val = getattr(row, ret_col, np.nan)
        realized.append(float(val) if pd.notna(val) else np.nan)
    realized = np.asarray(realized, dtype=float)

    return {
        "horizon_for_side": horizon,
        "test_rows": int(len(te)),
        "side_acc": float(np.mean(side_pred_bin == side_true_te)),
        "bucket_acc": float(np.mean(bucket_pred == te["label_bucket"].astype(int).values)),
        "horizon_acc": float(np.mean(horizon_pred == te["label_horizon"].astype(int).values)),
        "edge_ic": float(spearmanr(edge_pred, te["label_edge"].values).statistic),
        "top20": top_fraction_metrics(
            edge_pred,
            realized,
            side_pred=side_pred_bin,
            side_true=side_true_te,
            bucket_pred=bucket_pred,
            bucket_true=te["label_bucket"].astype(int).values,
            horizon_pred=horizon_pred,
            horizon_true=te["label_horizon"].astype(int).values,
            frac=0.2,
        ),
        "top10": top_fraction_metrics(edge_pred, realized, frac=0.1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--train-start", default="2026-01-01")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--test-start", default="2026-04-01")
    p.add_argument("--test-end", default="2026-06-30")
    p.add_argument("--horizons", default="1,3,5,10")
    p.add_argument("--output", default="qqq_btc/results/validate_0dte_routing_model_2026H1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    micro_root = Path(args.micro_root)
    label_dir = Path(args.label_dir).expanduser()
    train = build_routing_dataset(
        micro_root=micro_root,
        label_dir=label_dir,
        symbol=args.symbol,
        start=args.train_start,
        end=args.train_end,
        horizons=horizons,
    )
    test = build_routing_dataset(
        micro_root=micro_root,
        label_dir=label_dir,
        symbol=args.symbol,
        start=args.test_start,
        end=args.test_end,
        horizons=horizons,
    )
    features = train.attrs["features"]
    payload = {
        "micro_root": str(micro_root),
        "label_dir": str(label_dir),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": len(features),
        "side_by_horizon": [fit_side_model(train, test, features, h) for h in horizons],
        "bucket": fit_bucket_model(train, test, features),
        "horizon": fit_horizon_model(train, test, features),
        "edge": fit_edge_model(train, test, features),
        "joint": [fit_joint_model(train, test, features, h) for h in horizons],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"train={len(train)} test={len(test)} features={len(features)}")
    for r in payload["side_by_horizon"]:
        t = r["test"]
        print(
            f"side h={r['horizon']}m IC={t['ic']:.4f} side_acc={t['side_acc']:.3f} "
            f"top20_chosen_mean={t['top20_chosen_mean']:.4f} top20_hit={t['top20_chosen_hit']:.3f}"
        )
    print(f"bucket test_acc={payload['bucket']['test']['acc']:.3f}")
    print(f"horizon test_acc={payload['horizon']['test']['acc']:.3f}")
    print(f"edge test_ic={payload['edge']['test']['ic']:.4f}")
    best_joint = max(payload["joint"], key=lambda x: x["top20"].get("mean_realized", -999.0))
    print(
        f"joint best h={best_joint['horizon_for_side']} side_acc={best_joint['side_acc']:.3f} "
        f"bucket_acc={best_joint['bucket_acc']:.3f} horizon_acc={best_joint['horizon_acc']:.3f} "
        f"top20_mean={best_joint['top20'].get('mean_realized', 0.0):.4f}"
    )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
