#!/usr/bin/env python3
"""Dynamic top-dollar-volume universe validation for 0DTE trade-print triggers."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from factor_lab.tools.validate_0dte_1s_triple_barrier import (
    BarrierConfig,
    replay_predictions,
    summarize_trades,
)
from factor_lab.tools.validate_0dte_tradeprint_trigger import (
    FEATURES,
    add_labels_and_features,
)


def add_dynamic_universe(
    df: pd.DataFrame,
    *,
    top_n: int,
    lookback_s: int,
    per_side: bool,
) -> pd.DataFrame:
    work = df.sort_values(["ticker", "timestamp"]).copy()
    work["rolling_notional"] = (
        work.groupby("ticker")["trade_notional"]
        .rolling(lookback_s, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    # Avoid selecting contracts before any real trade has appeared.
    work["rolling_trades"] = (
        work.groupby("ticker")["trade_count"]
        .rolling(lookback_s, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    work = work[work["rolling_trades"] > 0].copy()
    if work.empty:
        return work
    if per_side:
        work["universe_rank"] = work.groupby(["timestamp", "side"])["rolling_notional"].rank(
            method="first", ascending=False
        )
    else:
        work["universe_rank"] = work.groupby("timestamp")["rolling_notional"].rank(method="first", ascending=False)
    return work[work["universe_rank"] <= top_n].copy()


def load_dynamic_dataset(
    micro_root: Path,
    start: str,
    end: str,
    cfg: BarrierConfig,
    *,
    top_n: int,
    lookback_s: int,
    per_side: bool,
) -> pd.DataFrame:
    files = sorted((micro_root / "contract_1s/QQQ").glob("QQQ_*.parquet"))
    files = [p for p in files if start <= p.stem.replace("QQQ_", "") <= end]
    frames = []
    for p in files:
        raw = pd.read_parquet(p)
        if raw.empty:
            continue
        day = add_labels_and_features(raw, cfg)
        if day.empty:
            continue
        day = add_dynamic_universe(day, top_n=top_n, lookback_s=lookback_s, per_side=per_side)
        if day.empty:
            continue
        day["date_str"] = p.stem.replace("QQQ_", "")
        frames.append(day)
    if not frames:
        raise SystemExit(f"no dynamic-universe data for {start}..{end}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def eval_probs(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "pos_rate": float(np.mean(y))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    for q, tag in [(0.90, "top10"), (0.95, "top5"), (0.975, "top2_5"), (0.99, "top1"), (0.995, "top0_5"), (0.999, "top0_1")]:
        th = float(np.quantile(p, q))
        sel = p >= th
        out[f"{tag}_n"] = int(sel.sum())
        out[f"{tag}_pos_rate"] = float(np.mean(y[sel])) if sel.any() else 0.0
    return out


def fit_predict(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.dropna(subset=FEATURES + ["label"]).copy()
    ev = eval_df.dropna(subset=FEATURES + ["label"]).copy()
    X = tr[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = tr["label"].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=31,
    )
    model.fit(X, y)
    p_tr = model.predict_proba(X)[:, 1]
    p_ev = model.predict_proba(ev[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())[:, 1]
    return p_tr, p_ev


def anomaly_eval(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    cols = [
        "trade_count",
        "trade_volume",
        "trade_notional",
        "buy_volume",
        "sell_volume",
        "net_buy_volume",
        "flow_toxicity",
        "trade_volume_sum_10s",
        "flow_toxicity_10s",
        "quote_events",
        "spread_pct",
        "rolling_notional",
        "rolling_trades",
    ]
    tr = train.dropna(subset=cols + ["label"]).copy()
    ev = eval_df.dropna(subset=cols + ["label"]).copy()
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=37, n_jobs=-1)
    model.fit(tr[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    score = -model.score_samples(ev[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    return eval_probs(ev["label"].astype(int).to_numpy(), score)


def replay_grid(val: pd.DataFrame, p_val: np.ndarray, cfg: BarrierConfig) -> dict:
    val_clean = val.dropna(subset=FEATURES + ["label"]).copy()
    rows = []
    best = None
    for q in (0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999):
        threshold = float(np.quantile(p_val, q))
        summary, _ = replay_predictions(val_clean, p_val, threshold, cfg)
        score = summary.get("total_net_return", 0.0) - 0.5 * abs(summary.get("max_drawdown", 0.0))
        row = {"quantile": q, "threshold": threshold, "score": score, **summary}
        rows.append(row)
        if summary["trades"] >= 5 and (best is None or row["score"] > best["score"]):
            best = row
    return best or max(rows, key=lambda x: x["score"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_tradeprint_ladder8_202604")
    p.add_argument("--train-start", default="2026-04-13")
    p.add_argument("--train-end", default="2026-04-22")
    p.add_argument("--val-start", default="2026-04-23")
    p.add_argument("--val-end", default="2026-04-24")
    p.add_argument("--test-start", default="2026-04-27")
    p.add_argument("--test-end", default="2026-04-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--per-side", action="store_true")
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--take-profit", type=float, default=0.05)
    p.add_argument("--stop-loss", type=float, default=-0.03)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_dynamic_tradeprint_universe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BarrierConfig(
        horizon_s=args.horizon_s,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    root = Path(args.micro_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_dynamic_dataset(
        root, args.train_start, args.train_end, cfg, top_n=args.top_n, lookback_s=args.lookback_s, per_side=args.per_side
    )
    val = load_dynamic_dataset(
        root, args.val_start, args.val_end, cfg, top_n=args.top_n, lookback_s=args.lookback_s, per_side=args.per_side
    )
    test = load_dynamic_dataset(
        root, args.test_start, args.test_end, cfg, top_n=args.top_n, lookback_s=args.lookback_s, per_side=args.per_side
    )
    train_full = pd.concat([train, val], ignore_index=True)
    p_train, p_val = fit_predict(train, val)
    _, p_test = fit_predict(train_full, test)
    train_clean = train.dropna(subset=FEATURES + ["label"]).copy()
    val_clean = val.dropna(subset=FEATURES + ["label"]).copy()
    test_clean = test.dropna(subset=FEATURES + ["label"]).copy()

    selected = replay_grid(val, p_val, cfg)
    test_summary, test_trades = replay_predictions(test_clean, p_test, float(selected["threshold"]), cfg)
    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize_trades(g, cfg)

    payload = {
        "config": asdict(cfg),
        "dynamic_universe": {
            "top_n": args.top_n,
            "lookback_s": args.lookback_s,
            "per_side": bool(args.per_side),
        },
        "split": {
            "train": [args.train_start, args.train_end],
            "val": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "rows": {"train": int(len(train_clean)), "val": int(len(val_clean)), "test": int(len(test_clean))},
        "base_pos_rate": {
            "train": float(train_clean["label"].mean()),
            "val": float(val_clean["label"].mean()),
            "test": float(test_clean["label"].mean()),
        },
        "learnability": {
            "train": eval_probs(train_clean["label"].astype(int).to_numpy(), p_train),
            "val": eval_probs(val_clean["label"].astype(int).to_numpy(), p_val),
            "test": eval_probs(test_clean["label"].astype(int).to_numpy(), p_test),
        },
        "anomaly_detection_test": anomaly_eval(train_full, test),
        "selected_val": selected,
        "test": test_summary,
        "test_monthly": monthly,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    test_trades.to_parquet(out_dir / "test_trades.parquet", index=False)
    print(json.dumps(payload, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
