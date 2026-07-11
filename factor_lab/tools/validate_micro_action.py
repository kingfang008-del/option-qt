#!/usr/bin/env python3
"""Validate whether short-DTE option microstructure improves CALL/PUT action routing.

The experiment is intentionally simple:
  - aggregate downloaded 1s microstructure features into causal 1-minute features
  - join labels from the existing 2DTE ladder feature files
  - train on Jan-Mar 2026 and test on Apr-Jun 2026

This is an architecture validation, not a production training script.
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

BASE_COLS = [
    "close_log_return",
    "vwap_log_return",
    "volume_ratio",
    "garman_klass_vol",
    "bb_width",
    "adx_smooth_10",
    "options_vw_spread",
    "options_vw_imbalance",
    "options_vw_iv",
    "options_pcr_volume",
    "options_iv_momentum",
    "options_flow_skew",
    "time_session_progress",
    "trend_fit_ret_30m",
    "trend_fit_r2_30m",
    "day_range_pos",
    "drawdown_from_day_high",
    "drawup_from_day_low",
    "open30_ret",
    "open30_reversal",
    "vix_level",
]

LABEL_COLS = ["label_call_return_fwd_net", "label_put_return_fwd_net", "label_return_fwd_net"]


def read_feature_stage(root: Path) -> pd.DataFrame:
    files = sorted((root / "QQQ/regular/09:30-16:00/1min").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no feature files under {root}")
    frames = []
    keep = ["timestamp"] + BASE_COLS + LABEL_COLS
    for f in files:
        df = pd.read_parquet(f)
        cols = [c for c in keep if c in df.columns]
        frames.append(df[cols].copy())
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return out.sort_values("timestamp").reset_index(drop=True)


def aggregate_micro_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp")
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    cols = [c for c in MICRO_COLS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Per-minute sums for flow/rate fields, means for ratios/spreads/imbalance fields.
    sum_cols = [c for c in cols if not any(k in c for k in ["ratio", "imbalance", "spread_pct"])]
    mean_cols = [c for c in cols if c not in sum_cols]
    parts = []
    if sum_cols:
        parts.append(df.groupby("minute_ts")[sum_cols].sum())
    if mean_cols:
        parts.append(df.groupby("minute_ts")[mean_cols].mean())
    out = pd.concat(parts, axis=1).sort_index().reset_index().rename(columns={"minute_ts": "timestamp"})
    return out


def read_micro(root: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted((root / "features_1s/QQQ").glob("QQQ_*.parquet"))
    files = [f for f in files if start <= f.stem.replace("QQQ_", "") <= end]
    if not files:
        raise SystemExit(f"no micro files under {root} for {start}..{end}")
    frames = [aggregate_micro_file(f) for f in files]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, micro_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values("timestamp").copy()
    ts = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York")
    out["_day"] = ts.dt.date
    feats = []
    for col in micro_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        feats.append(col)
        for win in (5, 15, 30):
            name = f"{col}_sum{win}"
            out[name] = out.groupby("_day")[col].rolling(win, min_periods=1).sum().reset_index(level=0, drop=True)
            feats.append(name)
    return out, feats


def build_stage(feature_root: Path, micro_root: Path, start: str, end: str) -> pd.DataFrame:
    feat = read_feature_stage(feature_root)
    micro = read_micro(micro_root, start, end)
    merged = pd.merge_asof(
        feat.sort_values("timestamp"),
        micro.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("59s"),
    )
    micro_cols = [c for c in MICRO_COLS if c in merged.columns]
    for c in micro_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    merged, micro_feats = add_rolling_features(merged, micro_cols)
    merged["_micro_has_flow"] = (merged[[c for c in ["call_quote_events", "put_quote_events"] if c in merged.columns]].sum(axis=1) > 0).astype(int)
    merged.attrs["micro_features"] = micro_feats + ["_micro_has_flow"]
    return merged


def evaluate(pred: np.ndarray, y: np.ndarray, call_r: np.ndarray, put_r: np.ndarray, months: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(y) & np.isfinite(call_r) & np.isfinite(put_r)
    pred, y, call_r, put_r, months = pred[m], y[m], call_r[m], put_r[m], months[m]
    out = {
        "n": int(len(y)),
        "ic": float(spearmanr(pred, y).statistic),
        "side_acc": float(((pred > 0) == (y > 0)).mean()),
    }
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
        if mm.sum() == 0:
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


def run_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], name: str) -> dict:
    call_col = "label_call_return_fwd_net"
    put_col = "label_put_return_fwd_net"
    for df in (train, test):
        df["target_leg_gap"] = pd.to_numeric(df[call_col], errors="coerce") - pd.to_numeric(df[put_col], errors="coerce")
    tr = train.dropna(subset=["target_leg_gap"]).copy()
    te = test.dropna(subset=["target_leg_gap"]).copy()
    X = tr[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = tr["target_leg_gap"].values
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X, y)
    out = {"name": name, "features": len(features), "train_rows": int(len(tr)), "test_rows": int(len(te))}
    for stage_name, df in [("train", tr), ("test", te)]:
        pred = model.predict(df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values)
        months = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m").values
        out[stage_name] = evaluate(
            pred,
            df["target_leg_gap"].values,
            pd.to_numeric(df[call_col], errors="coerce").values,
            pd.to_numeric(df[put_col], errors="coerce").values,
            months,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_short_dte")
    p.add_argument("--val-feature-root", default="/home/kingfang007/train_data/quote_features_val_dte2_ladder_spotroute_202606")
    p.add_argument("--test-feature-root", default="/home/kingfang007/train_data/quote_features_test_dte2_ladder_spotroute_202606")
    p.add_argument("--output", default="qqq_btc/results/micro_action_validation_2026.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    micro_root = Path(args.micro_root)
    train = build_stage(Path(args.val_feature_root), micro_root, "2026-01-01", "2026-03-31")
    test = build_stage(Path(args.test_feature_root), micro_root, "2026-04-01", "2026-06-30")
    micro_features = train.attrs["micro_features"]
    base_features = [c for c in BASE_COLS if c in train.columns]
    results = [
        run_model(train, test, base_features, "base_only"),
        run_model(train, test, micro_features, "micro_only"),
        run_model(train, test, base_features + micro_features, "base_plus_micro"),
    ]
    payload = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "micro_feature_count": len(micro_features),
        "base_feature_count": len(base_features),
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in results:
        t = r["test"]
        print(
            f"{r['name']}: IC={t['ic']:.4f} side_acc={t['side_acc']:.3f} "
            f"top20_acc={t['top20_side_acc']:.3f} top20_mean={t['top20_chosen_mean']:.4f} "
            f"top20_hit={t['top20_chosen_hit']:.3f} call_rate={t['top20_call_rate']:.3f}"
        )
        for mon, m in t["per_month"].items():
            print(
                f"  {mon}: top20_acc={m['top20_side_acc']:.3f} "
                f"mean={m['top20_chosen_mean']:.4f} hit={m['top20_chosen_hit']:.3f} call_rate={m['top20_call_rate']:.3f}"
            )
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
