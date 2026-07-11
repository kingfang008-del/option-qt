#!/usr/bin/env python3
"""实验: bucketed 层对 PUT volume/greeks 做分钟 ffill，对齐 7/5 gold delta."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from preprocess.ask_bid.options_locked_feature import calculate_locked_features


def calculate_locked_features_vol_ffill(df: pd.DataFrame, vol_ffill_limit: int = 30) -> pd.DataFrame:
    """在 pivot 后对 volume 也做 ffill（原逻辑 volume 严格填 0）。"""
    if df.empty or "bucket_id" not in df.columns:
        return pd.DataFrame()

    from preprocess.ask_bid.options_locked_feature import safe_convert_to_ny_time

    epsilon = 1e-9
    df = df.copy()
    df["timestamp"] = safe_convert_to_ny_time(df["timestamp"]).dt.ceil("1min")
    df["vanna"] = df["vega"] / (df["stock_close"] + epsilon)
    df["charm"] = df["theta"] / (df["stock_close"] + epsilon)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "bucket_id"], keep="last")

    pivot_cols = [
        "volume", "iv", "delta", "gamma", "vega", "theta", "vanna", "charm",
        "spread_pct", "volume_imbalance",
    ]
    df_wide = df.pivot(index="timestamp", columns="bucket_id", values=pivot_cols)
    df_wide.columns = [f"{c[0]}_{int(c[1])}" for c in df_wide.columns]

    full_idx = pd.date_range(start=df_wide.index.min(), end=df_wide.index.max(), freq="1min")
    df_wide = df_wide.reindex(full_idx)
    df_wide.index.name = "timestamp"

    state_cols = [c for c in df_wide.columns if "volume" not in c or "volume_imbalance" in c]
    df_wide[state_cols] = df_wide[state_cols].replace(0.0, np.nan).ffill(limit=vol_ffill_limit).fillna(0.0)

    vol_cols = [c for c in df_wide.columns if c.startswith("volume_") and "imbalance" not in c]
    df_wide[vol_cols] = df_wide[vol_cols].replace(0.0, np.nan).ffill(limit=vol_ffill_limit).fillna(0.0)

    stock_prices = df.groupby("timestamp")["stock_close"].last().reindex(full_idx).ffill()
    df_wide["stock_close"] = stock_prices
    df_wide = df_wide.fillna(0.0)

    v = {i: df_wide.get(f"volume_{i}", 0) for i in range(6)}
    iv = {i: df_wide.get(f"iv_{i}", 0) for i in range(6)}
    vega = {i: df_wide.get(f"vega_{i}", 0) for i in range(6)}
    gamma = {i: df_wide.get(f"gamma_{i}", 0) for i in range(6)}
    delta = {i: df_wide.get(f"delta_{i}", 0) for i in range(6)}
    theta = {i: df_wide.get(f"theta_{i}", 0) for i in range(6)}

    total_vol_front = v[0] + v[1] + v[2] + v[3]
    mask_no_vol = total_vol_front < 1.0

    df_wide["options_vw_iv"] = (iv[0] + iv[2]) / 2.0
    net_delta_vol = delta[0] * v[0] + delta[1] * v[1] + delta[2] * v[2] + delta[3] * v[3]
    atm_delta = (delta[0] + delta[2]) / 2.0
    df_wide["options_vw_delta"] = np.where(
        mask_no_vol, atm_delta, net_delta_vol / (total_vol_front + epsilon)
    )

    net_gamma = gamma[0] * v[0] + gamma[1] * v[1] + gamma[2] * v[2] + gamma[3] * v[3]
    net_vega = vega[0] * v[0] + vega[1] * v[1] + vega[2] * v[2] + vega[3] * v[3]
    net_theta = theta[0] * v[0] + theta[1] * v[1] + theta[2] * v[2] + theta[3] * v[3]
    df_wide["options_vw_gamma"] = np.where(mask_no_vol, (gamma[0] + gamma[2]) / 2.0, net_gamma / (total_vol_front + epsilon))
    df_wide["options_vw_vega"] = np.where(mask_no_vol, (vega[0] + vega[2]) / 2.0, net_vega / (total_vol_front + epsilon))
    df_wide["options_vw_theta"] = np.where(mask_no_vol, (theta[0] + theta[2]) / 2.0, net_theta / (total_vol_front + epsilon))

    denom_call_vol = v[2] + v[3]
    df_wide["options_pcr_volume"] = np.where(denom_call_vol > 0, (v[0] + v[1]) / denom_call_vol, 1.0)

    final_cols = ["stock_close", "options_vw_iv", "options_vw_delta", "options_pcr_volume"]
    return df_wide[final_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index()


def compare_to_gold(feat: pd.DataFrame, gold: pd.DataFrame, label: str) -> dict:
    feat = feat.copy()
    gold = gold.copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"]).dt.tz_convert("America/New_York")
    gold["timestamp"] = pd.to_datetime(gold["timestamp"]).dt.tz_convert("America/New_York")
    m = gold.merge(feat, on="timestamp", suffixes=("_gold", "_new"), how="inner")
    out = {"label": label, "n": len(m)}
    for col in ("options_vw_delta", "options_pcr_volume", "options_vw_iv"):
        a = m[f"{col}_gold"]
        b = m[f"{col}_new"]
        mask = a.notna() & b.notna()
        out[f"{col}_corr"] = float(a[mask].corr(b[mask])) if mask.sum() > 10 else None
        out[f"{col}_mad"] = float((a[mask] - b[mask]).abs().mean()) if mask.sum() > 10 else None
    return out


def main() -> None:
    months = ["2026-04", "2026-05", "2026-06"]
    monthly_root = Path.home() / "train_data/quote_options_monthly_iv/QQQ/standard"
    gold_root = Path.home() / "train_data/_bak_pre4c/quote_features_test_QQQ/regular/09:30-16:00/1min"

    rows = []
    for month in months:
        mon = pd.read_parquet(monthly_root / f"{month}.parquet")
        gold = pd.read_parquet(gold_root / f"{month}.parquet")

        baseline = calculate_locked_features(mon)
        vol_ffill = calculate_locked_features_vol_ffill(mon, vol_ffill_limit=30)

        for feat, tag in [(baseline, "baseline"), (vol_ffill, "vol_ffill30")]:
            r = compare_to_gold(feat, gold, f"{month}/{tag}")
            rows.append(r)
            print(
                f"{r['label']:22} n={r['n']:5d}  "
                f"delta={r.get('options_vw_delta_corr', 0):.4f}  "
                f"pcr={r.get('options_pcr_volume_corr', 0):.4f}  "
                f"iv={r.get('options_vw_iv_corr', 0):.4f}"
            )

    out = Path(REPO / "qqq_btc/results/put_ffill_bucketed_experiment.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
