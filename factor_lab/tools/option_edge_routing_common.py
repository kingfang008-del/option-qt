"""Shared helpers for 0DTE option-edge routing learnability + replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.validate_0dte_micro_action import (
    MICRO_COLS,
    add_rolling,
    minute_aggregate_features,
)


def load_contract_labels(label_dir: Path, symbol: str, start: str, end: str) -> pd.DataFrame:
    files = sorted((label_dir / symbol).glob(f"{symbol}_*.parquet"))
    files = [p for p in files if start <= p.stem.replace(f"{symbol}_", "") <= end]
    if not files:
        raise FileNotFoundError(f"no label files under {label_dir / symbol} for {start}..{end}")
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return df.sort_values(["timestamp", "bucket_id", "ticker"]).reset_index(drop=True)


def aggregate_routing_labels(contract_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Minute-level routing labels from contract-level option-edge rows."""
    rows: list[dict] = []
    for ts, g in contract_df.groupby("timestamp", sort=False):
        row: dict = {
            "timestamp": ts,
            "date_str": str(g["date_str"].iloc[0]),
        }
        for h in horizons:
            col = f"ret_{h}m"
            valid = g.dropna(subset=[col])
            if valid.empty:
                continue
            best = valid.loc[valid[col].idxmax()]
            row[f"best_ret_{h}m"] = float(best[col])
            row[f"best_side_{h}m"] = str(best["side"])
            row[f"best_bucket_{h}m"] = int(best["bucket_id"])
            row[f"best_ticker_{h}m"] = str(best["ticker"])
            for side in ("CALL", "PUT"):
                sub = g[(g["side"] == side) & g[col].notna()]
                if sub.empty:
                    row[f"best_{side.lower()}_ret_{h}m"] = np.nan
                    row[f"best_{side.lower()}_bucket_{h}m"] = np.nan
                else:
                    b = sub.loc[sub[col].idxmax()]
                    row[f"best_{side.lower()}_ret_{h}m"] = float(b[col])
                    row[f"best_{side.lower()}_bucket_{h}m"] = int(b["bucket_id"])
            call_r = row.get(f"best_call_ret_{h}m", np.nan)
            put_r = row.get(f"best_put_ret_{h}m", np.nan)
            if pd.notna(call_r) and pd.notna(put_r):
                row[f"gap_{h}m"] = float(call_r) - float(put_r)

        best_h = None
        best_ret = -np.inf
        for h in horizons:
            val = row.get(f"best_ret_{h}m")
            if val is not None and pd.notna(val) and float(val) > best_ret:
                best_ret = float(val)
                best_h = h
        if best_h is not None:
            row["label_horizon"] = int(best_h)
            row["label_side"] = row[f"best_side_{best_h}m"]
            row["label_bucket"] = int(row[f"best_bucket_{best_h}m"])
            row["label_edge"] = float(row[f"best_ret_{best_h}m"])
            row["label_ticker"] = row[f"best_ticker_{best_h}m"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    out["label_side_code"] = out["label_side"].map({"PUT": 0, "CALL": 1}).astype("Int64")
    return out


OPTION_NATIVE_COLS = [
    "call_mean_spread",
    "put_mean_spread",
    "call_mean_ask",
    "put_mean_ask",
    "call_mean_bid",
    "put_mean_bid",
    "call_n",
    "put_n",
    "cp_spread_diff",
    "cp_ask_diff",
    "bucket0_mean_spread",
    "bucket1_mean_spread",
    "bucket2_mean_spread",
    "bucket3_mean_spread",
]


def aggregate_option_native_features(day_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ts, grp in day_df.groupby("timestamp", sort=False):
        row: dict = {"timestamp": ts}
        for side in ("CALL", "PUT"):
            s = grp[grp["side"] == side]
            if s.empty:
                row[f"{side.lower()}_mean_spread"] = np.nan
                row[f"{side.lower()}_mean_ask"] = np.nan
                row[f"{side.lower()}_mean_bid"] = np.nan
                row[f"{side.lower()}_n"] = 0
            else:
                row[f"{side.lower()}_mean_spread"] = float(pd.to_numeric(s["spread_pct"], errors="coerce").mean())
                row[f"{side.lower()}_mean_ask"] = float(pd.to_numeric(s["ask"], errors="coerce").mean())
                row[f"{side.lower()}_mean_bid"] = float(pd.to_numeric(s["bid"], errors="coerce").mean())
                row[f"{side.lower()}_n"] = int(len(s))
        row["cp_spread_diff"] = row["call_mean_spread"] - row["put_mean_spread"]
        row["cp_ask_diff"] = row["call_mean_ask"] - row["put_mean_ask"]
        for bid in (0, 1, 2, 3):
            b = grp[pd.to_numeric(grp["bucket_id"], errors="coerce") == bid]
            row[f"bucket{bid}_mean_spread"] = (
                float(pd.to_numeric(b["spread_pct"], errors="coerce").mean()) if not b.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp")


def build_routing_dataset_option_native(
    *,
    label_dir: Path,
    symbol: str,
    start: str,
    end: str,
    horizons: list[int],
) -> pd.DataFrame:
    files = sorted((label_dir / symbol).glob(f"{symbol}_*.parquet"))
    files = [p for p in files if start <= p.stem.replace(f"{symbol}_", "") <= end]
    if not files:
        raise FileNotFoundError(f"no label files under {label_dir / symbol} for {start}..{end}")
    frames: list[pd.DataFrame] = []
    for fp in files:
        date_str = fp.stem.replace(f"{symbol}_", "")
        day = pd.read_parquet(fp)
        if day.empty:
            continue
        day["timestamp"] = pd.to_datetime(day["timestamp"], utc=True).dt.tz_convert("America/New_York")
        labels = aggregate_routing_labels(day, horizons)
        feat = aggregate_option_native_features(day)
        merged = feat.merge(labels, on="timestamp", how="inner")
        merged["date_str"] = date_str
        frames.append(merged)
    if not frames:
        raise SystemExit(f"no option-native routing dataset for {start}..{end}")
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    feat_cols = [c for c in OPTION_NATIVE_COLS if c in df.columns]
    df, feats = add_rolling(df, feat_cols)
    df.attrs["features"] = feats
    df.attrs["feature_source"] = "option_native"
    return df


def build_routing_dataset(
    *,
    micro_root: Path,
    label_dir: Path,
    symbol: str,
    start: str,
    end: str,
    horizons: list[int],
) -> pd.DataFrame:
    feat_files = sorted((micro_root / "features_1s/QQQ").glob("QQQ_*.parquet"))
    feat_files = [f for f in feat_files if start <= f.stem.replace("QQQ_", "") <= end]
    labels = aggregate_routing_labels(load_contract_labels(label_dir, symbol, start, end), horizons)
    frames: list[pd.DataFrame] = []
    for fp in feat_files:
        date_str = fp.stem.replace("QQQ_", "")
        day_labels = labels[labels["date_str"] == date_str]
        if day_labels.empty:
            continue
        feat = minute_aggregate_features(fp)
        if feat.empty:
            continue
        merged = feat.merge(day_labels, on="timestamp", how="inner")
        merged["date_str"] = date_str
        frames.append(merged)
    if not frames:
        raise SystemExit(f"no merged routing dataset for {start}..{end}")
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    micro_cols = [c for c in MICRO_COLS if c in df.columns]
    df, feats = add_rolling(df, micro_cols)
    df.attrs["features"] = feats
    df.attrs["feature_source"] = "micro"
    return df


def routing_feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values


def month_values(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m").values


def top_fraction_metrics(
    score: np.ndarray,
    realized: np.ndarray,
    *,
    side_pred: np.ndarray | None = None,
    side_true: np.ndarray | None = None,
    bucket_pred: np.ndarray | None = None,
    bucket_true: np.ndarray | None = None,
    horizon_pred: np.ndarray | None = None,
    horizon_true: np.ndarray | None = None,
    frac: float = 0.2,
) -> dict:
    m = np.isfinite(score) & np.isfinite(realized)
    score, realized = score[m], realized[m]
    if side_pred is not None:
        side_pred = side_pred[m]
    if side_true is not None:
        side_true = side_true[m]
    if bucket_pred is not None:
        bucket_pred = bucket_pred[m]
    if bucket_true is not None:
        bucket_true = bucket_true[m]
    if horizon_pred is not None:
        horizon_pred = horizon_pred[m]
    if horizon_true is not None:
        horizon_true = horizon_true[m]
    if len(score) == 0:
        return {"n": 0}
    thr = np.quantile(score, 1.0 - frac)
    sel = score >= thr
    out = {
        "n": int(sel.sum()),
        "mean_realized": float(np.mean(realized[sel])),
        "hit_rate": float(np.mean(realized[sel] > 0)),
    }
    if side_pred is not None and side_true is not None:
        out["side_acc"] = float(np.mean(side_pred[sel] == side_true[sel]))
    if bucket_pred is not None and bucket_true is not None:
        out["bucket_acc"] = float(np.mean(bucket_pred[sel] == bucket_true[sel]))
    if horizon_pred is not None and horizon_true is not None:
        out["horizon_acc"] = float(np.mean(horizon_pred[sel] == horizon_true[sel]))
    return out
