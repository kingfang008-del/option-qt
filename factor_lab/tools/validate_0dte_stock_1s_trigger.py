#!/usr/bin/env python3
"""Validate stock 1s flow/momentum trigger -> QQQ 0DTE option burst labels.

This is the "predict the underlying, execute the option" feasibility check.
It intentionally uses only QQQ stock 1s OHLCV-derived features plus option side
metadata, then evaluates strict option triple-barrier labels from raw option 1s.
"""
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
    build_dataset,
    replay_predictions,
    summarize_trades,
)


NY = "America/New_York"


STOCK_FEATURES = [
    "side_code",
    "bucket_id",
    "tod_frac",
    "signed_ret_1s",
    "signed_ret_3s",
    "signed_ret_5s",
    "signed_ret_10s",
    "signed_ret_30s",
    "abs_ret_1s",
    "abs_ret_5s",
    "abs_ret_10s",
    "abs_ret_30s",
    "ret_accel_3s",
    "range_1s",
    "range_5s",
    "volume_z_10s",
    "volume_z_30s",
    "dollar_volume_z_30s",
    "signed_volume_imb_5s",
    "signed_volume_imb_15s",
]


def load_stock_features(stock_root: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted(stock_root.glob("QQQ_*.parquet"))
    files = [p for p in files if start <= p.stem.replace("QQQ_", "") <= end]
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        df = df[(df["timestamp"].dt.time >= pd.Timestamp("09:30").time()) & (df["timestamp"].dt.time < pd.Timestamp("16:00").time())]
        if df.empty:
            continue
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date_str"] = p.stem.replace("QQQ_", "")
        df["ret_1s"] = df["close"] / df["close"].shift(1) - 1.0
        for w in (3, 5, 10, 30):
            df[f"ret_{w}s"] = df["close"] / df["close"].shift(w) - 1.0
            df[f"abs_ret_{w}s"] = df[f"ret_{w}s"].abs()
        df["abs_ret_1s"] = df["ret_1s"].abs()
        df["ret_accel_3s"] = df["ret_3s"] - df["ret_3s"].shift(3)
        df["range_1s"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["range_5s"] = df["range_1s"].rolling(5, min_periods=3).sum()
        vol_mean_10 = df["volume"].rolling(60, min_periods=20).mean()
        vol_std_10 = df["volume"].rolling(60, min_periods=20).std()
        df["volume_z_10s"] = (df["volume"].rolling(10, min_periods=5).sum() - vol_mean_10 * 10) / (vol_std_10 * np.sqrt(10)).replace(0, np.nan)
        vol_mean_30 = df["volume"].rolling(180, min_periods=60).mean()
        vol_std_30 = df["volume"].rolling(180, min_periods=60).std()
        vol30 = df["volume"].rolling(30, min_periods=10).sum()
        df["volume_z_30s"] = (vol30 - vol_mean_30 * 30) / (vol_std_30 * np.sqrt(30)).replace(0, np.nan)
        dollar = df["close"] * df["volume"]
        dv_mean = dollar.rolling(180, min_periods=60).mean()
        dv_std = dollar.rolling(180, min_periods=60).std()
        df["dollar_volume_z_30s"] = (dollar.rolling(30, min_periods=10).sum() - dv_mean * 30) / (dv_std * np.sqrt(30)).replace(0, np.nan)
        signed_vol = np.sign(df["ret_1s"].fillna(0.0)) * df["volume"].fillna(0.0)
        df["signed_volume_imb_5s"] = signed_vol.rolling(5, min_periods=3).sum() / df["volume"].rolling(5, min_periods=3).sum().replace(0, np.nan)
        df["signed_volume_imb_15s"] = signed_vol.rolling(15, min_periods=5).sum() / df["volume"].rolling(15, min_periods=5).sum().replace(0, np.nan)
        keep = ["timestamp", "date_str"] + [c for c in df.columns if c in {
            "ret_1s", "ret_3s", "ret_5s", "ret_10s", "ret_30s",
            "abs_ret_1s", "abs_ret_5s", "abs_ret_10s", "abs_ret_30s",
            "ret_accel_3s", "range_1s", "range_5s", "volume_z_10s",
            "volume_z_30s", "dollar_volume_z_30s", "signed_volume_imb_5s",
            "signed_volume_imb_15s",
        }]
        frames.append(df[keep])
    if not frames:
        raise SystemExit(f"no stock 1s files for {start}..{end}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def merge_stock_option(option_df: pd.DataFrame, stock_feat: pd.DataFrame) -> pd.DataFrame:
    out = option_df.merge(stock_feat.drop(columns=["date_str"], errors="ignore"), on="timestamp", how="inner")
    for w in (1, 3, 5, 10, 30):
        out[f"signed_ret_{w}s"] = out["side_code"] * pd.to_numeric(out[f"ret_{w}s"], errors="coerce")
    return out


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


def fit_classifier(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.dropna(subset=STOCK_FEATURES + ["label"]).copy()
    ev = eval_df.dropna(subset=STOCK_FEATURES + ["label"]).copy()
    X = tr[STOCK_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = tr["label"].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=7,
    )
    model.fit(X, y)
    p_tr = model.predict_proba(X)[:, 1]
    p_ev = model.predict_proba(ev[STOCK_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())[:, 1]
    return p_tr, p_ev


def anomaly_score(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [
        "abs_ret_1s", "abs_ret_5s", "abs_ret_10s", "abs_ret_30s",
        "range_1s", "range_5s", "volume_z_10s", "volume_z_30s",
        "dollar_volume_z_30s", "signed_volume_imb_5s", "signed_volume_imb_15s",
    ]
    tr = train.dropna(subset=cols).copy()
    ev = eval_df.dropna(subset=cols).copy()
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=11, n_jobs=-1)
    model.fit(tr[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    # sklearn returns larger = more normal, invert it.
    s_tr = -model.score_samples(tr[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    s_ev = -model.score_samples(ev[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy())
    return s_tr, s_ev


def run_replay_grid(val: pd.DataFrame, p_val: np.ndarray, cfg: BarrierConfig) -> dict:
    rows = []
    best = None
    val_clean = val.dropna(subset=STOCK_FEATURES + ["label"]).copy()
    for q in (0.90, 0.95, 0.975, 0.99, 0.995, 0.999):
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
    p.add_argument("--option-root", default="/mnt/s990/data/raw_1s/dte0_options/QQQ")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--train-start", default="2026-01-01")
    p.add_argument("--train-end", default="2026-02-28")
    p.add_argument("--val-start", default="2026-03-01")
    p.add_argument("--val-end", default="2026-03-31")
    p.add_argument("--test-start", default="2026-04-01")
    p.add_argument("--test-end", default="2026-06-30")
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--take-profit", type=float, default=0.05)
    p.add_argument("--stop-loss", type=float, default=-0.03)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_stock_1s_trigger")
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
    option_root = Path(args.option_root)
    stock_root = Path(args.stock_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building option labels...")
    opt_train = build_dataset(option_root, args.train_start, args.train_end, cfg)
    opt_val = build_dataset(option_root, args.val_start, args.val_end, cfg)
    opt_test = build_dataset(option_root, args.test_start, args.test_end, cfg)
    print("building stock features...")
    st_train = load_stock_features(stock_root, args.train_start, args.train_end)
    st_val = load_stock_features(stock_root, args.val_start, args.val_end)
    st_test = load_stock_features(stock_root, args.test_start, args.test_end)

    train = merge_stock_option(opt_train, st_train)
    val = merge_stock_option(opt_val, st_val)
    test = merge_stock_option(opt_test, st_test)
    train_full = pd.concat([train, val], ignore_index=True)

    print("training stock-trigger classifier...")
    p_train, p_val = fit_classifier(train, val)
    _, p_test = fit_classifier(train_full, test)
    train_clean = train.dropna(subset=STOCK_FEATURES + ["label"]).copy()
    val_clean = val.dropna(subset=STOCK_FEATURES + ["label"]).copy()
    test_clean = test.dropna(subset=STOCK_FEATURES + ["label"]).copy()

    selected = run_replay_grid(val, p_val, cfg)
    test_summary, test_trades = replay_predictions(test_clean, p_test, float(selected["threshold"]), cfg)
    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize_trades(g, cfg)

    # Unsupervised anomaly detector sanity check: does "rare stock flow" enrich labels?
    _, s_val = anomaly_score(train, val)
    _, s_test = anomaly_score(train_full, test)
    anomaly_val_eval = eval_probs(val.dropna(subset=[
        "abs_ret_1s", "abs_ret_5s", "abs_ret_10s", "abs_ret_30s",
        "range_1s", "range_5s", "volume_z_10s", "volume_z_30s",
        "dollar_volume_z_30s", "signed_volume_imb_5s", "signed_volume_imb_15s",
    ])["label"].astype(int).to_numpy(), s_val)
    anomaly_test_eval = eval_probs(test.dropna(subset=[
        "abs_ret_1s", "abs_ret_5s", "abs_ret_10s", "abs_ret_30s",
        "range_1s", "range_5s", "volume_z_10s", "volume_z_30s",
        "dollar_volume_z_30s", "signed_volume_imb_5s", "signed_volume_imb_15s",
    ])["label"].astype(int).to_numpy(), s_test)

    payload = {
        "config": asdict(cfg),
        "features": STOCK_FEATURES,
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
        "anomaly_detection": {"val": anomaly_val_eval, "test": anomaly_test_eval},
        "selected_val": selected,
        "test": test_summary,
        "test_monthly": monthly,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    test_trades.to_parquet(out_dir / "test_trades.parquet", index=False)

    print("SUMMARY", json.dumps({
        "rows": payload["rows"],
        "base_pos_rate": payload["base_pos_rate"],
        "learnability_test": payload["learnability"]["test"],
        "anomaly_test": payload["anomaly_detection"]["test"],
        "selected_val": payload["selected_val"],
        "test": payload["test"],
        "output_dir": str(out_dir),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
