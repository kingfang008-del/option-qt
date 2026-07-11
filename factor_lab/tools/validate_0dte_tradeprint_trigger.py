#!/usr/bin/env python3
"""Validate option trade-print flow features against 0DTE 1s triple-barrier labels."""
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
    first_barrier_return,
    replay_predictions,
    summarize_trades,
)


FEATURES = [
    "side_code",
    "bucket_id",
    "tod_frac",
    "ask",
    "mid",
    "spread_pct",
    "quote_imbalance",
    "quote_events",
    "trade_count",
    "trade_volume",
    "trade_notional",
    "buy_volume",
    "sell_volume",
    "unknown_volume",
    "net_buy_volume",
    "buy_ratio",
    "signed_net_buy_volume",
    "signed_buy_ratio",
    "signed_trade_volume",
    "flow_toxicity",
    "net_buy_sum_3s",
    "net_buy_sum_5s",
    "net_buy_sum_10s",
    "trade_volume_sum_5s",
    "trade_volume_sum_10s",
    "quote_events_sum_5s",
    "quote_events_sum_10s",
    "flow_toxicity_10s",
    "mid_ret_1s",
    "mid_ret_3s",
    "mid_ret_5s",
    "spread_chg_3s",
]


def add_labels_and_features(day: pd.DataFrame, cfg: BarrierConfig) -> pd.DataFrame:
    df = day.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    for c in [
        "bid", "ask", "mid", "spread_pct", "quote_imbalance", "quote_events",
        "trade_count", "trade_volume", "trade_notional", "buy_volume", "sell_volume",
        "unknown_volume", "net_buy_volume", "buy_ratio", "bucket_id",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0
    df["side_code"] = df["side"].map({"PUT": -1.0, "CALL": 1.0}).fillna(0.0)
    df["tod_frac"] = (
        df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second - (9 * 3600 + 30 * 60)
    ) / (6.5 * 3600)
    frames = []
    for _, g0 in df.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=False):
        g = g0.drop_duplicates("timestamp", keep="last").copy().reset_index(drop=True)
        for w in (1, 3, 5):
            g[f"mid_ret_{w}s"] = g["mid"] / g["mid"].shift(w) - 1.0
        g["spread_chg_3s"] = g["spread_pct"] - g["spread_pct"].shift(3)
        denom = (g["buy_volume"] + g["sell_volume"]).replace(0, np.nan)
        g["flow_toxicity"] = (g["buy_volume"] - g["sell_volume"]).abs() / denom
        g["signed_net_buy_volume"] = g["side_code"] * g["net_buy_volume"]
        g["signed_buy_ratio"] = np.where(g["side_code"] > 0, g["buy_ratio"], 1.0 - g["buy_ratio"])
        g["signed_trade_volume"] = g["side_code"] * g["trade_volume"]
        for w in (3, 5, 10):
            g[f"net_buy_sum_{w}s"] = g["net_buy_volume"].rolling(w, min_periods=1).sum()
        for w in (5, 10):
            g[f"trade_volume_sum_{w}s"] = g["trade_volume"].rolling(w, min_periods=1).sum()
            g[f"quote_events_sum_{w}s"] = g["quote_events"].rolling(w, min_periods=1).sum()
        buy10 = g["buy_volume"].rolling(10, min_periods=1).sum()
        sell10 = g["sell_volume"].rolling(10, min_periods=1).sum()
        g["flow_toxicity_10s"] = (buy10 - sell10).abs() / (buy10 + sell10).replace(0, np.nan)

        bids = g["bid"].to_numpy(dtype=float)
        asks = g["ask"].to_numpy(dtype=float)
        labels = np.zeros(len(g), dtype=np.int8)
        exit_rets = np.full(len(g), np.nan, dtype=float)
        bars = np.zeros(len(g), dtype=np.int16)
        reasons = ["INVALID"] * len(g)
        for i in range(len(g)):
            entry_idx = i + cfg.latency_s
            if entry_idx >= len(g):
                continue
            entry_ask = asks[entry_idx]
            if not np.isfinite(entry_ask) or entry_ask <= 0:
                continue
            cost_frac = 2.0 * cfg.commission_per_contract / (entry_ask * 100.0)
            b, ret, reason = first_barrier_return(
                bids, entry_idx, entry_ask, cfg.horizon_s, cfg.take_profit, cfg.stop_loss, cost_frac
            )
            if b <= 0 or not np.isfinite(ret):
                continue
            labels[i] = 1 if reason == "TAKE_PROFIT" else 0
            exit_rets[i] = ret
            bars[i] = b
            reasons[i] = reason
        g["label"] = labels
        g["exit_return"] = exit_rets
        g["bars_held"] = bars
        g["exit_reason"] = reasons
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    tradable = (
        out["exit_return"].notna()
        & (out["ask"] >= cfg.min_ask)
        & (out["bid"] > 0)
        & (out["spread_pct"] <= cfg.max_spread_pct)
        & out["bucket_id"].notna()
    )
    return out[tradable].copy()


def load_dataset(micro_root: Path, start: str, end: str, cfg: BarrierConfig) -> pd.DataFrame:
    files = sorted((micro_root / "contract_1s/QQQ").glob("QQQ_*.parquet"))
    files = [p for p in files if start <= p.stem.replace("QQQ_", "") <= end]
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        day = add_labels_and_features(df, cfg)
        if day.empty:
            continue
        day["date_str"] = p.stem.replace("QQQ_", "")
        frames.append(day)
    if not frames:
        raise SystemExit(f"no tradeprint dataset for {start}..{end}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def eval_probs(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "pos_rate": float(np.mean(y))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    for q, tag in [(0.90, "top10"), (0.95, "top5"), (0.975, "top2_5"), (0.99, "top1"), (0.999, "top0_1")]:
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
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=19,
    )
    model.fit(X, y)
    p_tr = model.predict_proba(X)[:, 1]
    p_ev = model.predict_proba(ev[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())[:, 1]
    return p_tr, p_ev


def anomaly_scores(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cols = [
        "trade_count", "trade_volume", "trade_notional", "buy_volume", "sell_volume",
        "net_buy_volume", "flow_toxicity", "trade_volume_sum_10s", "flow_toxicity_10s",
        "quote_events", "spread_pct",
    ]
    tr = train.dropna(subset=cols + ["label"]).copy()
    ev = eval_df.dropna(subset=cols + ["label"]).copy()
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=23, n_jobs=-1)
    model.fit(tr[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    score = -model.score_samples(ev[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    return ev, score


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
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_tradeprint_smoke")
    p.add_argument("--train-start", default="2026-04-13")
    p.add_argument("--train-end", default="2026-04-15")
    p.add_argument("--val-start", default="2026-04-16")
    p.add_argument("--val-end", default="2026-04-16")
    p.add_argument("--test-start", default="2026-04-17")
    p.add_argument("--test-end", default="2026-04-17")
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--take-profit", type=float, default=0.05)
    p.add_argument("--stop-loss", type=float, default=-0.03)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_tradeprint_trigger_smoke")
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
    train = load_dataset(root, args.train_start, args.train_end, cfg)
    val = load_dataset(root, args.val_start, args.val_end, cfg)
    test = load_dataset(root, args.test_start, args.test_end, cfg)
    train_full = pd.concat([train, val], ignore_index=True)
    p_train, p_val = fit_predict(train, val)
    _, p_test = fit_predict(train_full, test)
    train_clean = train.dropna(subset=FEATURES + ["label"]).copy()
    val_clean = val.dropna(subset=FEATURES + ["label"]).copy()
    test_clean = test.dropna(subset=FEATURES + ["label"]).copy()
    selected = replay_grid(val, p_val, cfg)
    test_summary, test_trades = replay_predictions(test_clean, p_test, float(selected["threshold"]), cfg)
    ev_anom, s_anom = anomaly_scores(train_full, test)
    anomaly_eval = eval_probs(ev_anom["label"].astype(int).to_numpy(), s_anom)
    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize_trades(g, cfg)
    payload = {
        "config": asdict(cfg),
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
        "anomaly_detection_test": anomaly_eval,
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
