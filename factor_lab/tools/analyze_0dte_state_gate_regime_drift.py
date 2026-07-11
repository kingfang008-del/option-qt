#!/usr/bin/env python3
"""Diagnose QQQ 0DTE State Gate drift across chronological periods.

This script intentionally starts from *executed, confirmed* curated trades.
It answers three narrower questions before a Rule Selector is trained:

1. Did the entry-time feature distribution move?
2. Did feature-to-executable-return relationships move (concept drift)?
3. Which rule/state/side/month cohorts caused the PnL change?

The Apr-fitted curated gate replayed on Jan-Mar is a reverse-time portability
test, not forward OOS.  Outputs preserve that distinction explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "tree_edge_score",
    "trend_score",
    "flow_score",
    "gamma_score",
    "vol_score",
    "liquidity_score",
    "time_score",
    "spread_pct",
    "quote_imbalance",
    "stock_ret_30s",
    "stock_ret_60s",
    "stock_abs_ret_60s",
    "stock_rv_60s",
    "stock_vwap_dev",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades",
        default=(
            "qqq_btc/results/"
            "0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/"
            "trades_all.parquet"
        ),
    )
    parser.add_argument("--early-months", default="2026-01,2026-02,2026-03")
    parser.add_argument("--late-months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--position-frac", type=float, default=0.25)
    parser.add_argument("--psi-bins", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_regime_drift_h1",
    )
    return parser.parse_args()


def month_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def finite_numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 8 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    left = left.dropna()
    right = right.dropna()
    if left.empty or right.empty:
        return float("nan")
    pooled = np.sqrt((left.var(ddof=0) + right.var(ddof=0)) / 2.0)
    if not np.isfinite(pooled) or pooled <= 1e-12:
        return 0.0 if np.isclose(left.mean(), right.mean()) else float("inf")
    return float((right.mean() - left.mean()) / pooled)


def population_stability_index(
    left: pd.Series, right: pd.Series, bins: int
) -> float:
    """Symmetric PSI using pooled quantile bins.

    Pooled bins avoid implying that either reverse-time period is the production
    reference.  PSI is descriptive here, not a hypothesis test.
    """
    left = left.dropna()
    right = right.dropna()
    pooled = pd.concat([left, right], ignore_index=True)
    if left.empty or right.empty or pooled.nunique() < 2:
        return 0.0
    edges = np.unique(
        pooled.quantile(np.linspace(0.0, 1.0, bins + 1)).to_numpy(dtype=float)
    )
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    left_share = (
        pd.cut(left, bins=edges, include_lowest=True).value_counts(
            sort=False, normalize=True
        )
    )
    right_share = (
        pd.cut(right, bins=edges, include_lowest=True).value_counts(
            sort=False, normalize=True
        )
    )
    eps = 1e-6
    p = left_share.to_numpy(dtype=float) + eps
    q = right_share.to_numpy(dtype=float) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def empirical_ks(left: pd.Series, right: pd.Series) -> float:
    left = np.sort(left.dropna().to_numpy(dtype=float))
    right = np.sort(right.dropna().to_numpy(dtype=float))
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    points = np.unique(np.concatenate([left, right]))
    cdf_left = np.searchsorted(left, points, side="right") / len(left)
    cdf_right = np.searchsorted(right, points, side="right") / len(right)
    return float(np.max(np.abs(cdf_left - cdf_right)))


def compound_return(returns: pd.Series, position_frac: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(np.prod(1.0 + position_frac * values.to_numpy()) - 1.0)


def max_drawdown(returns: pd.Series, position_frac: float) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    equity = np.cumprod(1.0 + position_frac * values.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    return float(np.min(equity / peaks - 1.0))


def return_summary(
    group: pd.DataFrame, *, label: str, position_frac: float
) -> dict[str, object]:
    ret = finite_numeric(group, "path_exec_ret").dropna()
    gains = float(ret[ret > 0].sum())
    losses = float(-ret[ret < 0].sum())
    return {
        "label": label,
        "trades": int(len(ret)),
        "days": int(group["date_str"].nunique()),
        "avg_return": float(ret.mean()) if len(ret) else float("nan"),
        "median_return": float(ret.median()) if len(ret) else float("nan"),
        "hit_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
        "profit_factor": gains / losses if losses > 0 else float("inf"),
        "total_return_position": compound_return(ret, position_frac),
        "max_drawdown": max_drawdown(ret, position_frac),
    }


def feature_drift_table(
    early: pd.DataFrame,
    late: pd.DataFrame,
    features: list[str],
    target: str,
    psi_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_early = finite_numeric(early, target)
    y_late = finite_numeric(late, target)
    for col in features:
        x_early = finite_numeric(early, col)
        x_late = finite_numeric(late, col)
        ic_early = safe_corr(x_early, y_early)
        ic_late = safe_corr(x_late, y_late)
        rows.append(
            {
                "feature": col,
                "early_non_null": int(x_early.notna().sum()),
                "late_non_null": int(x_late.notna().sum()),
                "early_mean": float(x_early.mean()),
                "late_mean": float(x_late.mean()),
                "early_median": float(x_early.median()),
                "late_median": float(x_late.median()),
                "smd_late_minus_early": standardized_mean_difference(
                    x_early, x_late
                ),
                "psi_symmetric": population_stability_index(
                    x_early, x_late, psi_bins
                ),
                "ks_distance": empirical_ks(x_early, x_late),
                "early_spearman_to_return": ic_early,
                "late_spearman_to_return": ic_late,
                "spearman_delta": (
                    ic_late - ic_early
                    if np.isfinite(ic_early) and np.isfinite(ic_late)
                    else float("nan")
                ),
                "sign_flip": bool(
                    np.isfinite(ic_early)
                    and np.isfinite(ic_late)
                    and np.sign(ic_early) != np.sign(ic_late)
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["drift_priority"] = (
        out["psi_symmetric"].fillna(0).clip(upper=5)
        + out["ks_distance"].fillna(0)
        + out["spearman_delta"].abs().fillna(0)
        + out["sign_flip"].astype(float) * 0.25
    )
    return out.sort_values("drift_priority", ascending=False).reset_index(drop=True)


def grouped_attribution(
    frame: pd.DataFrame, columns: list[str], position_frac: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(columns, key_tuple))
        record.update(
            return_summary(group, label="segment", position_frac=position_frac)
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_parquet(args.trades).copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades.sort_values("timestamp").reset_index(drop=True)
    if "month" not in trades.columns:
        trades["month"] = trades["timestamp"].dt.strftime("%Y-%m")
    if "date_str" not in trades.columns:
        trades["date_str"] = trades["timestamp"].dt.strftime("%Y-%m-%d")

    early_months = month_list(args.early_months)
    late_months = month_list(args.late_months)
    early = trades[trades["month"].isin(early_months)].copy()
    late = trades[trades["month"].isin(late_months)].copy()
    if early.empty or late.empty:
        raise ValueError(
            f"empty comparison period: early={len(early)} late={len(late)}"
        )

    features = [col for col in DEFAULT_FEATURES if col in trades.columns]
    drift = feature_drift_table(
        early, late, features, "path_exec_ret", args.psi_bins
    )
    drift.to_csv(output_dir / "feature_concept_drift.csv", index=False)

    monthly = grouped_attribution(trades, ["month"], args.position_frac)
    monthly.to_csv(output_dir / "monthly_attribution.csv", index=False)

    segment_cols = [
        col
        for col in ["month", "active_state", "side", "hold_s"]
        if col in trades.columns
    ]
    segments = grouped_attribution(trades, segment_cols, args.position_frac)
    segments.to_csv(output_dir / "rule_state_side_month.csv", index=False)

    period_frames = []
    for name, frame in [("early_reverse_portability", early), ("late", late)]:
        period = frame.copy()
        period["comparison_period"] = name
        period_frames.append(period)
    period_data = pd.concat(period_frames, ignore_index=True)
    period_segments = grouped_attribution(
        period_data,
        ["comparison_period", "active_state", "side"],
        args.position_frac,
    )
    period_segments.to_csv(output_dir / "period_state_side.csv", index=False)

    top_distribution = drift.nlargest(10, "psi_symmetric")[
        ["feature", "psi_symmetric", "ks_distance", "smd_late_minus_early"]
    ].to_dict("records")
    top_concept = drift.nlargest(10, "spearman_delta", keep="all")[
        [
            "feature",
            "early_spearman_to_return",
            "late_spearman_to_return",
            "spearman_delta",
            "sign_flip",
        ]
    ].to_dict("records")
    sign_flips = drift[drift["sign_flip"]].sort_values(
        "spearman_delta", key=lambda s: s.abs(), ascending=False
    )

    summary = {
        "experiment_type": (
            "reverse_time_portability_diagnostic; not forward OOS because the "
            "curated rules/scorer/confirm policy were selected or fit in Apr-May"
        ),
        "source": args.trades,
        "position_frac": args.position_frac,
        "early_months": early_months,
        "late_months": late_months,
        "early": return_summary(
            early,
            label="early_reverse_portability",
            position_frac=args.position_frac,
        ),
        "late": return_summary(
            late, label="late", position_frac=args.position_frac
        ),
        "feature_count": len(features),
        "distribution_drift": {
            "psi_ge_0_10": int((drift["psi_symmetric"] >= 0.10).sum()),
            "psi_ge_0_25": int((drift["psi_symmetric"] >= 0.25).sum()),
            "top": top_distribution,
        },
        "concept_drift": {
            "spearman_sign_flips": int(drift["sign_flip"].sum()),
            "top_absolute_sign_flips": sign_flips.head(10).to_dict("records"),
            "top_late_minus_early": top_concept,
        },
        "limitations": [
            "Only confirmed executed trades are present; rejected confirmations "
            "cannot yet show whether the confirm gate was too strict or too loose.",
            "The two curated rules were selected with later-period information, so "
            "Jan-Mar results are diagnostic rather than deployable walk-forward OOS.",
            "Trade-level rows within a day are dependent; model validation must split "
            "and embargo by day/month, never randomly by row.",
        ],
        "next_step": (
            "Build pre-confirm candidate-event data, then train a day-grouped "
            "expanding-window Rule Selector with an explicit No-Trade action."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {output_dir}")


if __name__ == "__main__":
    main()
