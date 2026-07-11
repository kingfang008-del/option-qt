#!/usr/bin/env python3
"""Causal expanding-window meta selector for QQQ 0DTE curated candidates.

The input must contain *pre-confirm* State Gate candidates with path-level
ask-to-bid returns.  The selector never creates a direction signal; it decides
whether an already-triggered curated rule should be executed.

Important: the current curated rule identities and base scorer were discovered
with Apr-Jun research.  Therefore Jan-Jun selector results are a chronological
shadow test of the meta layer, not a clean end-to-end strategy OOS claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONTINUOUS_FEATURES = [
    "tree_edge_score",
    "trend_score",
    "flow_score",
    "gamma_score",
    "vol_score",
    "liquidity_score",
    "time_score",
    "spread_pct",
    "quote_imbalance",
    "stock_ret_10s",
    "stock_ret_30s",
    "stock_ret_60s",
    "stock_abs_ret_30s",
    "stock_abs_ret_60s",
    "stock_rv_60s",
    "stock_vwap_dev",
    "stock_volume_z_60s",
    "state_activity_q",
    "state_quote_q",
    "state_spread_q",
    "state_abs_momentum_q",
    "state_stock_abs_mom_q",
    "state_stock_volume_q",
    "flow_imbalance_5s",
    "flow_toxicity_5s",
    "quote_event_intensity",
    "trade_notional_log1p",
    "tod_frac",
    "entry_ask",
]

BINARY_FEATURES = [
    "is_vol_expansion",
    "is_liquidity_stress",
    "is_range_pin_proxy",
    "is_put_trend_proxy",
    "is_call_trend_proxy",
    "is_stock_trend_up",
    "is_stock_trend_down",
    "is_stock_vwap_extension",
    "is_high_vol_proxy",
    "is_low_vol_proxy",
    "is_positive_gamma_proxy",
    "is_negative_gamma_proxy",
    "is_put_flow_continuation",
    "is_put_flow_exhaustion",
    "is_opening",
    "is_lunch",
    "is_power_hour",
]

REGIME_FEATURES = [
    "regime_p_low",
    "regime_p_mid",
    "regime_p_high",
    "regime_entropy",
    "regime_confidence",
    "regime_transition_surprise",
]

THRESHOLD_GRID = np.round(np.arange(0.40, 0.81, 0.05), 2).tolist()


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
    parser.add_argument("--position-frac", type=float, default=0.25)
    parser.add_argument("--min-train-trades", type=int, default=20)
    parser.add_argument("--min-validation-trades", type=int, default=8)
    parser.add_argument("--default-threshold", type=float, default=0.55)
    parser.add_argument(
        "--no-regime-features",
        action="store_true",
        help="ignore causal regime columns even when they are present",
    )
    parser.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_rule_selector_h1",
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
    out["is_recovering_state"] = out["active_state"].eq(
        "is_qqq_recovering"
    ).astype(float)
    out["is_lunch_state"] = out["active_state"].eq(
        "is_stock_trend_down__and__is_lunch"
    ).astype(float)
    out["is_call_side"] = out["side"].eq("CALL").astype(float)
    out["is_put_side"] = out["side"].eq("PUT").astype(float)
    target = pd.to_numeric(out["path_exec_ret"], errors="coerce")
    out["rule_valid"] = (target > 0.0).astype(int)
    return out[np.isfinite(target)].copy()


def feature_columns(
    frame: pd.DataFrame, *, use_regime_features: bool = True
) -> list[str]:
    preferred = [
        *CONTINUOUS_FEATURES,
        *BINARY_FEATURES,
        "is_recovering_state",
        "is_lunch_state",
        "is_call_side",
        "is_put_side",
    ]
    if use_regime_features:
        preferred.extend(REGIME_FEATURES)
    return [col for col in preferred if col in frame.columns]


def feature_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = matrix.median(numeric_only=True).reindex(columns).fillna(0.0)
    matrix = matrix.fillna(medians).fillna(0.0)
    return matrix, medians


def fit_selector(
    train: pd.DataFrame, columns: list[str]
) -> tuple[Pipeline | None, pd.Series, float]:
    matrix, medians = feature_matrix(train, columns)
    target = train["rule_valid"].astype(int)
    prior = float(target.mean()) if len(target) else 0.5
    if len(train) < 8 or target.nunique() < 2:
        return None, medians, prior
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.20,
                    penalty="l2",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(matrix, target)
    return model, medians, prior


def predict_selector(
    model: Pipeline | None,
    frame: pd.DataFrame,
    columns: list[str],
    medians: pd.Series,
    prior: float,
) -> pd.Series:
    matrix, _ = feature_matrix(frame, columns, medians)
    if model is None:
        values = np.full(len(frame), prior, dtype=float)
    else:
        values = model.predict_proba(matrix)[:, 1]
    return pd.Series(values, index=frame.index, name="rule_valid_probability")


def account_metrics(
    frame: pd.DataFrame,
    *,
    label: str,
    position_frac: float,
) -> dict[str, object]:
    returns = pd.to_numeric(frame["path_exec_ret"], errors="coerce").dropna()
    if returns.empty:
        return {
            "label": label,
            "trades": 0,
            "days": 0,
            "position_frac": position_frac,
            "total_return_position": 0.0,
            "max_drawdown_from_initial": 0.0,
        }
    equity = np.cumprod(1.0 + position_frac * returns.to_numpy())
    equity_with_initial = np.r_[1.0, equity]
    drawdown = equity_with_initial / np.maximum.accumulate(equity_with_initial) - 1.0
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    return {
        "label": label,
        "trades": int(len(returns)),
        "days": int(frame.loc[returns.index, "date_str"].nunique()),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "hit_rate": float((returns > 0).mean()),
        "profit_factor": gains / losses if losses > 0 else float("inf"),
        "position_frac": position_frac,
        "total_return_position": float(equity[-1] - 1.0),
        "max_drawdown_from_initial": float(drawdown.min()),
    }


def threshold_objective(
    validation: pd.DataFrame,
    threshold: float,
    position_frac: float,
    min_trades: int,
) -> tuple[float, dict[str, object]]:
    selected = validation[
        validation["rule_valid_probability"] >= threshold
    ].copy()
    metrics = account_metrics(
        selected,
        label=f"validation_p_ge_{threshold:.2f}",
        position_frac=position_frac,
    )
    if metrics["trades"] < min_trades:
        return float("-inf"), metrics
    total_return = float(metrics["total_return_position"])
    drawdown = abs(float(metrics["max_drawdown_from_initial"]))
    # Economic validation objective; a losing validation policy must abstain.
    objective = total_return - 0.50 * drawdown
    return objective, metrics


def select_threshold(
    history: pd.DataFrame,
    columns: list[str],
    *,
    default_threshold: float,
    min_validation_trades: int,
    position_frac: float,
) -> tuple[float, dict[str, object]]:
    months = sorted(history["month"].unique())
    if len(months) < 2:
        return default_threshold, {
            "mode": "fixed_default_insufficient_months",
            "threshold": default_threshold,
        }
    validation_month = months[-1]
    inner_train = history[history["month"] < validation_month]
    validation = history[history["month"] == validation_month].copy()
    if len(inner_train) < 12 or inner_train["rule_valid"].nunique() < 2:
        return default_threshold, {
            "mode": "fixed_default_insufficient_inner_train",
            "threshold": default_threshold,
            "validation_month": validation_month,
        }
    model, medians, prior = fit_selector(inner_train, columns)
    validation["rule_valid_probability"] = predict_selector(
        model, validation, columns, medians, prior
    )
    candidates: list[dict[str, object]] = []
    best_threshold = 1.01
    best_objective = 0.0
    for threshold in THRESHOLD_GRID:
        objective, metrics = threshold_objective(
            validation,
            threshold,
            position_frac,
            min_validation_trades,
        )
        candidates.append(
            {
                "threshold": threshold,
                "objective": objective,
                **metrics,
            }
        )
        if objective > best_objective:
            best_objective = objective
            best_threshold = threshold
    return best_threshold, {
        "mode": "prior_month_economic_validation",
        "validation_month": validation_month,
        "threshold": best_threshold,
        "best_objective": best_objective,
        "abstain_selected": best_threshold > 1.0,
        "candidates": candidates,
    }


def safe_auc(target: pd.Series, score: pd.Series) -> float:
    if target.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = prepare_frame(pd.read_parquet(args.trades))
    columns = feature_columns(
        trades, use_regime_features=not args.no_regime_features
    )
    months = sorted(trades["month"].unique())
    if len(months) < 2:
        raise ValueError("need at least two chronological months")

    scored_parts: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for test_month in months[1:]:
        history = trades[trades["month"] < test_month].copy()
        test = trades[trades["month"] == test_month].copy()
        if len(history) < args.min_train_trades or test.empty:
            continue
        threshold, threshold_meta = select_threshold(
            history,
            columns,
            default_threshold=args.default_threshold,
            min_validation_trades=args.min_validation_trades,
            position_frac=args.position_frac,
        )
        model, medians, prior = fit_selector(history, columns)
        test["rule_valid_probability"] = predict_selector(
            model, test, columns, medians, prior
        )
        test["selector_threshold"] = threshold
        test["selector_passed"] = test["rule_valid_probability"] >= threshold
        test["test_month"] = test_month

        target = test["rule_valid"].astype(int)
        probability = test["rule_valid_probability"].clip(1e-6, 1 - 1e-6)
        selected = test[test["selector_passed"]]
        fold = {
            "test_month": test_month,
            "train_months": sorted(history["month"].unique()),
            "train_trades": int(len(history)),
            "test_trades": int(len(test)),
            "train_positive_rate": float(history["rule_valid"].mean()),
            "test_positive_rate": float(target.mean()),
            "threshold": threshold,
            "auc": safe_auc(target, probability),
            "brier": float(brier_score_loss(target, probability)),
            "baseline": account_metrics(
                test,
                label="all_candidates",
                position_frac=args.position_frac,
            ),
            "selector": account_metrics(
                selected,
                label="selector",
                position_frac=args.position_frac,
            ),
            "threshold_selection": threshold_meta,
        }
        folds.append(fold)
        scored_parts.append(test)

    if not scored_parts:
        raise ValueError("no eligible expanding-window folds")
    scored = pd.concat(scored_parts, ignore_index=True).sort_values("timestamp")
    scored.to_parquet(output_dir / "walk_forward_scored_trades.parquet", index=False)

    baseline = account_metrics(
        scored,
        label="walk_forward_all_candidates",
        position_frac=args.position_frac,
    )
    selected = scored[scored["selector_passed"]]
    selector = account_metrics(
        selected,
        label="walk_forward_selector",
        position_frac=args.position_frac,
    )
    fixed_thresholds = []
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        policy = scored[scored["rule_valid_probability"] >= threshold]
        fixed_thresholds.append(
            {
                "threshold": threshold,
                **account_metrics(
                    policy,
                    label=f"fixed_p_ge_{threshold:.2f}",
                    position_frac=args.position_frac,
                ),
            }
        )

    summary = {
        "experiment_type": (
            "causal meta-layer expanding-window shadow test; not clean "
            "end-to-end OOS because curated rule identities/base scorer were "
            "discovered using later Apr-Jun research"
        ),
        "source": args.trades,
        "position_frac": args.position_frac,
        "feature_columns": columns,
        "regime_features_enabled": bool(
            not args.no_regime_features
            and any(col in columns for col in REGIME_FEATURES)
        ),
        "label": "path_exec_ret > 0 after ask-entry, bid-exit and commission",
        "threshold_policy": (
            "fit on months before the latest history month; choose on latest "
            "history month by return-minus-drawdown objective; abstain if no "
            "threshold has positive validation objective"
        ),
        "folds": folds,
        "combined_baseline": baseline,
        "combined_selector": selector,
        "fixed_threshold_sensitivity": fixed_thresholds,
        "limitations": [
            "Rule and base-score discovery leakage prevents a production claim.",
            "The earliest fold has only one training month and uses a fixed threshold.",
            "Rows within a day are dependent; all splits are chronological by month.",
            "Probability calibration is fragile at this sample size.",
        ],
        "files": {
            "scored_trades": str(
                output_dir / "walk_forward_scored_trades.parquet"
            ),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {output_dir}")


if __name__ == "__main__":
    main()
