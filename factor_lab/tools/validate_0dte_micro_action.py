#!/usr/bin/env python3
"""Validate 0DTE microstructure action signal.

Labels are computed directly from the downloaded 0DTE contract quotes:
long return = future bid / entry ask - 1.

For each timestamp, the side label is:
  max CALL return across call ladder - max PUT return across put ladder

This answers the architecture question: do option quote/trade flow features help
route CALL vs PUT on a liquid 0DTE universe?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor


MICRO_COLS = [
    "call_trade_volume",
    "call_buy_volume",
    "call_sell_volume",
    "call_net_buy_volume",
    "call_weighted_net_buy",
    "call_quote_events",
    "call_buy_ratio",
    "call_quote_imbalance",
    "call_spread_pct",
    "put_trade_volume",
    "put_buy_volume",
    "put_sell_volume",
    "put_net_buy_volume",
    "put_weighted_net_buy",
    "put_quote_events",
    "put_buy_ratio",
    "put_quote_imbalance",
    "put_spread_pct",
    "cp_net_buy_diff",
    "cp_weighted_net_buy_diff",
    "cp_buy_ratio_diff",
    "cp_quote_imbalance_diff",
    "cp_quote_event_diff",
    "cp_spread_diff",
]


def minute_aggregate_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp")
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    cols = [c for c in MICRO_COLS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    sum_cols = [c for c in cols if not any(k in c for k in ("ratio", "imbalance", "spread_pct"))]
    mean_cols = [c for c in cols if c not in sum_cols]
    parts = []
    if sum_cols:
        parts.append(df.groupby("minute_ts")[sum_cols].sum())
    if mean_cols:
        parts.append(df.groupby("minute_ts")[mean_cols].mean())
    out = pd.concat(parts, axis=1).sort_index().reset_index().rename(columns={"minute_ts": "timestamp"})
    return out


def add_rolling(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values("timestamp").copy()
    ts = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York")
    out["_day"] = ts.dt.date
    feats: list[str] = []
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        feats.append(c)
        for w in (3, 5, 10, 15):
            name = f"{c}_sum{w}"
            out[name] = out.groupby("_day")[c].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            feats.append(name)
    return out, feats


def quote_label_file(path: Path, horizons: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values(["ticker", "timestamp"])
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    rows = []
    for (_, side), g in df.groupby(["ticker", "side"]):
        g = g.sort_values("timestamp").drop_duplicates("minute_ts", keep="last").copy()
        g = g[["minute_ts", "side", "ask", "bid"]]
        for h in horizons:
            entry_ask = pd.to_numeric(g["ask"], errors="coerce").replace(0, np.nan)
            future_bid = pd.to_numeric(g["bid"], errors="coerce").shift(-h)
            ret = future_bid / entry_ask - 1.0
            tmp = pd.DataFrame({"timestamp": g["minute_ts"], "side": side, f"ret_{h}m": ret})
            rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    all_ret = pd.concat(rows, ignore_index=True)
    labels = None
    for h in horizons:
        pivot = (
            all_ret.dropna(subset=[f"ret_{h}m"])
            .groupby(["timestamp", "side"])[f"ret_{h}m"]
            .max()
            .unstack("side")
            .reset_index()
        )
        for side in ("CALL", "PUT"):
            if side not in pivot.columns:
                pivot[side] = np.nan
        pivot = pivot.rename(columns={"CALL": f"best_call_ret_{h}m", "PUT": f"best_put_ret_{h}m"})
        pivot[f"gap_{h}m"] = pivot[f"best_call_ret_{h}m"] - pivot[f"best_put_ret_{h}m"]
        keep = ["timestamp", f"best_call_ret_{h}m", f"best_put_ret_{h}m", f"gap_{h}m"]
        labels = pivot[keep] if labels is None else labels.merge(pivot[keep], on="timestamp", how="outer")
    return labels.sort_values("timestamp")


def build_dataset(root: Path, start: str, end: str, horizons: list[int]) -> pd.DataFrame:
    feat_files = sorted((root / "features_1s/QQQ").glob("QQQ_*.parquet"))
    feat_files = [f for f in feat_files if start <= f.stem.replace("QQQ_", "") <= end]
    frames = []
    for fp in feat_files:
        date_str = fp.stem.replace("QQQ_", "")
        cp = root / "contract_1s/QQQ" / fp.name
        if not cp.exists():
            continue
        feat = minute_aggregate_features(fp)
        lab = quote_label_file(cp, horizons)
        if feat.empty or lab.empty:
            continue
        merged = feat.merge(lab, on="timestamp", how="inner")
        merged["date_str"] = date_str
        frames.append(merged)
    if not frames:
        raise SystemExit(f"no merged data for {start}..{end}")
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    micro_cols = [c for c in MICRO_COLS if c in df.columns]
    df, feats = add_rolling(df, micro_cols)
    df.attrs["features"] = feats
    return df


def evaluate(pred: np.ndarray, y: np.ndarray, call_r: np.ndarray, put_r: np.ndarray, months: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(y) & np.isfinite(call_r) & np.isfinite(put_r)
    pred, y, call_r, put_r, months = pred[m], y[m], call_r[m], put_r[m], months[m]
    out = {"n": int(len(y)), "ic": float(spearmanr(pred, y).statistic), "side_acc": float(((pred > 0) == (y > 0)).mean())}
    for frac, tag in [(0.2, "top20"), (0.1, "top10"), (0.05, "top5")]:
        thr = np.quantile(np.abs(pred), 1 - frac)
        sel = np.abs(pred) >= thr
        chosen = np.where(pred[sel] > 0, call_r[sel], put_r[sel])
        out[f"{tag}_n"] = int(sel.sum())
        out[f"{tag}_side_acc"] = float(((pred[sel] > 0) == (y[sel] > 0)).mean())
        out[f"{tag}_chosen_mean"] = float(np.mean(chosen))
        out[f"{tag}_chosen_hit"] = float(np.mean(chosen > 0))
        out[f"{tag}_call_rate"] = float(np.mean(pred[sel] > 0))
    per_month = {}
    for mon in np.unique(months):
        mm = months == mon
        if mm.sum() < 50:
            continue
        thr = np.quantile(np.abs(pred[mm]), 0.8)
        sel = mm & (np.abs(pred) >= thr)
        chosen = np.where(pred[sel] > 0, call_r[sel], put_r[sel])
        per_month[str(mon)] = {
            "n": int(mm.sum()),
            "top20_n": int(sel.sum()),
            "top20_side_acc": float(((pred[sel] > 0) == (y[sel] > 0)).mean()),
            "top20_chosen_mean": float(np.mean(chosen)),
            "top20_chosen_hit": float(np.mean(chosen > 0)),
            "top20_call_rate": float(np.mean(pred[sel] > 0)),
        }
    out["per_month"] = per_month
    return out


def train_eval(train: pd.DataFrame, test: pd.DataFrame, features: list[str], horizon: int) -> dict:
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
    X = tr[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    model.fit(X, tr[target].values)
    out = {"horizon": horizon, "train_rows": int(len(tr)), "test_rows": int(len(te)), "features": len(features)}
    for name, df in [("train", tr), ("test", te)]:
        pred = model.predict(df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values)
        months = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m").values
        out[name] = evaluate(pred, df[target].values, df[call_col].values, df[put_col].values, months)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--train-start", default="2026-01-01")
    p.add_argument("--train-end", default="2026-03-31")
    p.add_argument("--test-start", default="2026-04-01")
    p.add_argument("--test-end", default="2026-06-30")
    p.add_argument("--horizons", default="5,10,15,30")
    p.add_argument("--output", default="qqq_btc/results/validate_0dte_micro_action_2026.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    root = Path(args.micro_root)
    train = build_dataset(root, args.train_start, args.train_end, horizons)
    test = build_dataset(root, args.test_start, args.test_end, horizons)
    features = train.attrs["features"]
    results = [train_eval(train, test, features, h) for h in horizons]
    payload = {
        "micro_root": str(root),
        "train_rows_raw": int(len(train)),
        "test_rows_raw": int(len(test)),
        "feature_count": len(features),
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in results:
        t = r["test"]
        print(
            f"h={r['horizon']}m IC={t['ic']:.4f} side_acc={t['side_acc']:.3f} "
            f"top20_acc={t['top20_side_acc']:.3f} top20_mean={t['top20_chosen_mean']:.4f} "
            f"top20_hit={t['top20_chosen_hit']:.3f} call_rate={t['top20_call_rate']:.3f}"
        )
        for mon, m in t["per_month"].items():
            print(
                f"  {mon}: top20_acc={m['top20_side_acc']:.3f} mean={m['top20_chosen_mean']:.4f} "
                f"hit={m['top20_chosen_hit']:.3f} call_rate={m['top20_call_rate']:.3f}"
            )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
