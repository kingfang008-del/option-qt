#!/usr/bin/env python3
"""Six-score factor fusion loop for QQQ 0DTE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset
from factor_lab.tools.eval_0dte_tradeprint_factor_gates import add_composite_scores
from factor_lab.tools.run_0dte_minimal_five_layer_loop import (
    add_market_state_raw,
    apply_market_state_thresholds,
    fit_state_thresholds,
    load_stock_state_features,
    replay_daily_topk,
)


SCORE_COLS = ["trend_score", "gamma_score", "flow_score", "liquidity_score", "vol_score", "time_score"]
STATE_COLS = [
    "is_vol_expansion",
    "is_range_pin_proxy",
    "is_put_trend_proxy",
    "is_call_trend_proxy",
    "is_stock_trend_up",
    "is_stock_trend_down",
    "is_stock_vwap_extension",
]


def rank01(df: pd.DataFrame, col: str, by: list[str] | None = None, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if by:
        return s.groupby([df[k] for k in by]).rank(pct=True, ascending=ascending).fillna(0.5)
    return s.rank(pct=True, ascending=ascending).fillna(0.5)


def add_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = add_composite_scores(df)
    by = ["date_str", "side"]
    out["trend_score"] = (
        0.35 * rank01(out, "stock_ret_30s", ["date_str"])
        + 0.25 * rank01(out, "stock_ret_60s", ["date_str"])
        + 0.20 * rank01(out, "stock_vwap_dev", ["date_str"])
        + 0.20 * rank01(out, "state_call_minus_put_mom", ["date_str"])
    )
    put = out["side"].eq("PUT")
    out.loc[put, "trend_score"] = (
        0.35 * (1.0 - rank01(out.loc[put], "stock_ret_30s", ["date_str"]))
        + 0.25 * (1.0 - rank01(out.loc[put], "stock_ret_60s", ["date_str"]))
        + 0.20 * (1.0 - rank01(out.loc[put], "stock_vwap_dev", ["date_str"]))
        + 0.20 * rank01(out.loc[put], "state_put_minus_call_mom", ["date_str"])
    )
    out["flow_score"] = (
        0.30 * rank01(out, "trade_notional_sum_60s", by)
        + 0.25 * rank01(out, "quote_events_sum_10s", by)
        + 0.20 * rank01(out, "quote_imbalance", by)
        + 0.15 * rank01(out, "flow_imbalance_5s", by)
        + 0.10 * rank01(out, "net_buy_sum_5s", by)
    )
    out["liquidity_score"] = (
        0.35 * rank01(out, "spread_pct", by, ascending=False)
        + 0.25 * rank01(out, "rolling_trades", by)
        + 0.25 * rank01(out, "rolling_notional", by)
        + 0.15 * rank01(out, "universe_rank", by, ascending=False)
    )
    out["vol_score"] = (
        0.35 * rank01(out, "stock_rv_60s", ["date_str"])
        + 0.25 * rank01(out, "stock_abs_ret_30s", ["date_str"])
        + 0.20 * rank01(out, "panel_abs_mom_10s", ["date_str"])
        + 0.20 * rank01(out, "flow_toxicity_10s", by)
    )
    out["gamma_score"] = (
        0.35 * (1.0 - rank01(out, "universe_rank", by))
        + 0.25 * rank01(out, "rank_spread_tight", by)
        + 0.20 * rank01(out, "state_stock_abs_mom_q", ["date_str"], ascending=False)
        + 0.20 * rank01(out, "stock_vwap_dev", ["date_str"]).sub(0.5).abs().mul(2.0)
    )
    tod = pd.to_numeric(out["tod_frac"], errors="coerce").clip(0, 1)
    out["time_score"] = pd.Series(
        np.maximum(1.0 - tod * 3.0, 0.0) + np.maximum((tod - 0.78) / 0.22, 0.0),
        index=out.index,
    ).clip(0, 1)
    for col in SCORE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(0.0, 1.0).fillna(0.5)
    return out


def build_score_dataset(raw: pd.DataFrame, stock: pd.DataFrame, target: str, state_thresholds: dict | None) -> tuple[pd.DataFrame, dict]:
    merged = raw.merge(stock, on="timestamp", how="inner")
    work = add_market_state_raw(add_composite_scores(merged))
    if state_thresholds is None:
        state_thresholds = fit_state_thresholds(work)
    work = apply_market_state_thresholds(work, state_thresholds)
    work = add_factor_scores(work)
    keep = SCORE_COLS + STATE_COLS + ["side_code", target, "timestamp", "date_str", "side", "ticker"]
    clean = work.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
    return clean, state_thresholds


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    sample = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 100 or sample["x"].nunique() < 3:
        return 0.0
    return float(sample["x"].rank().corr(sample["y"].rank()))


def fit_ic_weights(train: pd.DataFrame, target: str) -> dict[str, float]:
    raw = {c: max(0.0, spearman_ic(train[c], train[target])) for c in SCORE_COLS}
    total = sum(raw.values())
    return {c: (v / total if total > 0 else 1.0 / len(SCORE_COLS)) for c, v in raw.items()}


def apply_ic_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        score += weight * pd.to_numeric(df[col], errors="coerce").fillna(0.5)
    return score


def model_features() -> list[str]:
    return SCORE_COLS + STATE_COLS + ["side_code"]


def fit_tree(train: pd.DataFrame, target: str) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=500,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=23,
    )
    model.fit(
        train[model_features()].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(),
        pd.to_numeric(train[target], errors="coerce").fillna(0.0).to_numpy(),
    )
    return model


def predict_tree(model: HistGradientBoostingRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[model_features()].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())


def quantile_report(df: pd.DataFrame, score: str, target: str) -> dict:
    out = {"n": int(len(df)), "base_mean": float(df[target].mean()), "base_pos": float((df[target] > 0).mean())}
    for q, tag in [(0.90, "top10"), (0.95, "top5"), (0.99, "top1"), (0.995, "top0_5")]:
        sel = df[df[score] >= df[score].quantile(q)]
        out[f"{tag}_n"] = int(len(sel))
        out[f"{tag}_mean"] = float(sel[target].mean()) if len(sel) else 0.0
        out[f"{tag}_pos"] = float((sel[target] > 0).mean()) if len(sel) else 0.0
    return out


def replay_grid(df: pd.DataFrame, score: str, target: str, cooldown_s: int) -> pd.DataFrame:
    rows = []
    states = [None, *STATE_COLS]
    work = df.copy()
    work["edge_score"] = work[score]
    for side in [None, "CALL", "PUT"]:
        for state in states:
            for topk in (1, 2, 3, 5):
                rows.append(
                    {
                        "score": score,
                        **replay_daily_topk(
                            work,
                            score_col="edge_score",
                            target=target,
                            topk=topk,
                            cooldown_s=cooldown_s,
                            side=side,
                            state_filter=state,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--train-start", default="2026-04-13")
    p.add_argument("--train-end", default="2026-05-29")
    p.add_argument("--test-start", default="2026-06-01")
    p.add_argument("--test-end", default="2026-06-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_factor_score_loop")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[score-loop] loading panels", flush=True)
    raw_train = load_factor_dataset(Path(args.micro_root), args.train_start, args.train_end, (args.horizon_s,), top_n=args.top_n, lookback_s=args.lookback_s, per_side=False, commission=args.commission_per_contract, max_spread_pct=args.max_spread_pct, min_ask=args.min_ask)
    raw_test = load_factor_dataset(Path(args.micro_root), args.test_start, args.test_end, (args.horizon_s,), top_n=args.top_n, lookback_s=args.lookback_s, per_side=False, commission=args.commission_per_contract, max_spread_pct=args.max_spread_pct, min_ask=args.min_ask)
    stock_train = load_stock_state_features(Path(args.stock_root), args.train_start, args.train_end)
    stock_test = load_stock_state_features(Path(args.stock_root), args.test_start, args.test_end)
    train, thresholds = build_score_dataset(raw_train, stock_train, target, None)
    test, _ = build_score_dataset(raw_test, stock_test, target, thresholds)
    print(f"[score-loop] train={len(train)} test={len(test)}", flush=True)

    weights = fit_ic_weights(train, target)
    train["ic_edge_score"] = apply_ic_score(train, weights)
    test["ic_edge_score"] = apply_ic_score(test, weights)
    model = fit_tree(train, target)
    train["tree_edge_score"] = predict_tree(model, train)
    test["tree_edge_score"] = predict_tree(model, test)
    train["hot_score"] = train["score_hot_quote_tight"]
    test["hot_score"] = test["score_hot_quote_tight"]

    replay = pd.concat(
        [
            replay_grid(test, "ic_edge_score", target, args.cooldown_s),
            replay_grid(test, "tree_edge_score", target, args.cooldown_s),
            replay_grid(test, "hot_score", target, args.cooldown_s),
        ],
        ignore_index=True,
    )
    replay.to_csv(out_dir / "oos_replay_grid.csv", index=False)
    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "score_ic_train": {c: spearman_ic(train[c], train[target]) for c in SCORE_COLS},
        "ic_weights": weights,
        "tree_model": {
            "train_r2": float(r2_score(train[target], train["tree_edge_score"])),
            "test_r2": float(r2_score(test[target], test["tree_edge_score"])),
        },
        "quantiles": {
            "ic_weight_test": quantile_report(test, "ic_edge_score", target),
            "tree_test": quantile_report(test, "tree_edge_score", target),
            "hot_score_test": quantile_report(test, "hot_score", target),
        },
        "best_oos_replays": replay[replay["trades"].fillna(0) >= 10].sort_values(["avg_return", "profit_factor"], ascending=False).head(30).to_dict("records"),
        "files": {"oos_replay_grid": str(out_dir / "oos_replay_grid.csv")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
