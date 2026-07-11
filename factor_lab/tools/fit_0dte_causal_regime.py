#!/usr/bin/env python3
"""Causal slow-regime baseline for QQQ 0DTE candidate events.

This is deliberately lighter than a neural sequence model:

1. Fit a diagonal Gaussian mixture on prior-month entry-time state features.
2. Order components from low to high risk so fold labels remain comparable.
3. Estimate within-day transition probabilities on the training period.
4. Apply forward-only Bayesian filtering to each unseen test day.

It is an HMM-like baseline with persistent state probabilities and uncertainty.
No return label is used to form regimes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REGIME_FEATURES = [
    "vol_score",
    "state_activity_q",
    "stock_rv_60s",
    "state_stock_abs_mom_q",
    "state_spread_q",
    "liquidity_score",
    "stock_vwap_dev",
    "state_stock_volume_q",
]


@dataclass
class RegimeArtifact:
    features: list[str]
    medians: pd.Series
    scaler: StandardScaler
    mixture: GaussianMixture
    component_order: np.ndarray
    transition: np.ndarray
    initial: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades",
        default=(
            "qqq_btc/results/"
            "0dte_state_gate_curated_noconfirm_statehold_jan_jun/"
            "trades_all.parquet"
        ),
    )
    parser.add_argument("--min-train-trades", type=int, default=30)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_causal_regime_h1",
    )
    return parser.parse_args()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    if "month" not in out.columns:
        out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    if "date_str" not in out.columns:
        out["date_str"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out


def numeric_matrix(
    frame: pd.DataFrame,
    features: list[str],
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = matrix.median(numeric_only=True).reindex(features).fillna(0.0)
    return matrix.fillna(medians).fillna(0.0), medians


def ordered_component_indices(
    mixture: GaussianMixture, features: list[str]
) -> np.ndarray:
    """Stable semantic ordering by standardized vol/activity/spread risk."""
    risk_features = {
        "vol_score": 1.0,
        "state_activity_q": 0.8,
        "stock_rv_60s": 1.0,
        "state_stock_abs_mom_q": 0.8,
        "state_spread_q": 0.5,
        "liquidity_score": -0.3,
    }
    weights = np.array([risk_features.get(col, 0.0) for col in features])
    if np.allclose(weights, 0.0):
        weights = np.ones(len(features))
    scores = mixture.means_ @ weights
    return np.argsort(scores)


def estimate_transition(
    labels: np.ndarray,
    dates: pd.Series,
    components: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    counts = np.full((components, components), smoothing, dtype=float)
    date_values = dates.astype(str).to_numpy()
    for idx in range(1, len(labels)):
        if date_values[idx] != date_values[idx - 1]:
            continue
        counts[labels[idx - 1], labels[idx]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def fit_regime_artifact(
    train: pd.DataFrame,
    *,
    max_components: int = 3,
) -> RegimeArtifact:
    features = [col for col in REGIME_FEATURES if col in train.columns]
    if len(features) < 3:
        raise ValueError(f"insufficient regime features: {features}")
    matrix, medians = numeric_matrix(train, features)
    scaler = StandardScaler()
    values = scaler.fit_transform(matrix)
    components = min(max_components, max(2, len(train) // 25))
    mixture = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=10,
        max_iter=500,
        random_state=42,
    )
    mixture.fit(values)
    order = ordered_component_indices(mixture, features)
    raw_to_ordered = np.empty(components, dtype=int)
    raw_to_ordered[order] = np.arange(components)
    raw_labels = mixture.predict(values)
    labels = raw_to_ordered[raw_labels]
    transition = estimate_transition(
        labels, train["date_str"], components=components
    )
    raw_initial = mixture.weights_[order]
    initial = raw_initial / raw_initial.sum()
    return RegimeArtifact(
        features=features,
        medians=medians,
        scaler=scaler,
        mixture=mixture,
        component_order=order,
        transition=transition,
        initial=initial,
    )


def component_emission(
    artifact: RegimeArtifact, frame: pd.DataFrame
) -> np.ndarray:
    matrix, _ = numeric_matrix(frame, artifact.features, artifact.medians)
    values = artifact.scaler.transform(matrix)
    mixture = artifact.mixture
    means = mixture.means_
    variances = mixture.covariances_
    log_det = np.log(2.0 * np.pi * variances).sum(axis=1)
    diff = values[:, None, :] - means[None, :, :]
    quadratic = (diff * diff / variances[None, :, :]).sum(axis=2)
    log_emission = -0.5 * (quadratic + log_det[None, :])
    log_emission = log_emission[:, artifact.component_order]
    log_emission -= log_emission.max(axis=1, keepdims=True)
    emission = np.exp(log_emission)
    return emission / emission.sum(axis=1, keepdims=True)


def attach_causal_regime(
    artifact: RegimeArtifact,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.sort_values("timestamp").copy()
    emission = component_emission(artifact, out)
    components = emission.shape[1]
    posterior = np.zeros_like(emission)
    prior_predictive = np.zeros_like(emission)
    dates = out["date_str"].astype(str).to_numpy()
    previous = artifact.initial.copy()
    for idx in range(len(out)):
        if idx == 0 or dates[idx] != dates[idx - 1]:
            predictive = artifact.initial.copy()
        else:
            predictive = previous @ artifact.transition
        unnormalized = predictive * emission[idx]
        total = float(unnormalized.sum())
        current = (
            unnormalized / total
            if total > 1e-12
            else np.full(components, 1.0 / components)
        )
        prior_predictive[idx] = predictive
        posterior[idx] = current
        previous = current

    names = (
        ["low", "high"]
        if components == 2
        else ["low", "mid", "high"]
    )
    for idx, name in enumerate(names):
        out[f"regime_p_{name}"] = posterior[:, idx]
    if components == 2:
        out["regime_p_mid"] = 0.0
    out["regime_id"] = np.argmax(posterior, axis=1)
    entropy = -np.sum(
        posterior * np.log(np.clip(posterior, 1e-12, 1.0)), axis=1
    )
    out["regime_entropy"] = entropy / np.log(components)
    out["regime_confidence"] = posterior.max(axis=1)
    out["regime_transition_surprise"] = 1.0 - prior_predictive.max(axis=1)
    return out.sort_index()


def regime_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (month, regime), group in frame.groupby(
        ["month", "regime_id"], sort=True
    ):
        returns = pd.to_numeric(
            group["path_exec_ret"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "month": month,
                "regime_id": int(regime),
                "trades": int(len(returns)),
                "days": int(group["date_str"].nunique()),
                "avg_return": float(returns.mean()),
                "hit_rate": float((returns > 0).mean()),
                "avg_confidence": float(group["regime_confidence"].mean()),
                "avg_entropy": float(group["regime_entropy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades = prepare_frame(pd.read_parquet(args.trades))
    months = sorted(trades["month"].unique())

    scored_parts: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for test_month in months[1:]:
        train = trades[trades["month"] < test_month].copy()
        test = trades[trades["month"] == test_month].copy()
        if len(train) < args.min_train_trades or test.empty:
            continue
        artifact = fit_regime_artifact(
            train, max_components=args.max_components
        )
        scored = attach_causal_regime(artifact, test)
        scored["regime_fit_through"] = max(train["month"])
        scored_parts.append(scored)
        folds.append(
            {
                "test_month": test_month,
                "train_months": sorted(train["month"].unique()),
                "train_trades": int(len(train)),
                "test_trades": int(len(test)),
                "components": int(artifact.mixture.n_components),
                "features": artifact.features,
                "transition": artifact.transition.tolist(),
                "initial": artifact.initial.tolist(),
            }
        )

    if not scored_parts:
        raise ValueError("no eligible expanding-window regime folds")
    scored_all = pd.concat(scored_parts, ignore_index=True).sort_values(
        "timestamp"
    )
    scored_all.to_parquet(
        output_dir / "walk_forward_regime_trades.parquet", index=False
    )
    attribution = regime_attribution(scored_all)
    attribution.to_csv(output_dir / "regime_month_attribution.csv", index=False)

    summary = {
        "experiment_type": (
            "unsupervised expanding-window causal regime diagnostic; "
            "regimes use no return labels"
        ),
        "source": args.trades,
        "folds": folds,
        "mean_regime_confidence": float(
            scored_all["regime_confidence"].mean()
        ),
        "mean_regime_entropy": float(scored_all["regime_entropy"].mean()),
        "high_uncertainty_ratio": float(
            (scored_all["regime_entropy"] >= 0.80).mean()
        ),
        "limitations": [
            "Filtering occurs at candidate-event times, not every market second.",
            "The mixture emissions are Gaussian approximations with diagonal covariance.",
            "Regime usefulness must be judged only after adding it to a causal selector.",
        ],
        "files": {
            "scored_trades": str(
                output_dir / "walk_forward_regime_trades.parquet"
            ),
            "attribution": str(
                output_dir / "regime_month_attribution.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {output_dir}")


if __name__ == "__main__":
    main()
