#!/usr/bin/env python3
"""QQQ Deep Anchor scaffold: causal stock Regime + micro-window Tradeable hook.

This is NOT an end-to-end trading model.

Layer A (this script, runnable now):
  - Build daily regime features from QQQ stock 1s (~2022-03 onward).
  - Expanding-window fit of a shallow MLP on past days → next-month regime probs.
  - Regime labels are unsupervised risk/trend buckets from *past* features only
    (no future return used to define the state).

Layer B (wired, needs micro-era trade labels):
  - Optional join to curated trade days to measure whether regime probs separate
    good vs bad days inside the 2026 micro window.

Protocol:
  train on all days strictly before month M; evaluate on month M.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rv_20d",
    "rv_60d",
    "vwap_dev_eod",
    "volume_z_20d",
    "range_20d",
    "trend_strength_20d",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument(
        "--trade-days",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/"
            "trades_all.parquet"
        ),
        help="optional curated trades for regime×PnL diagnostics in micro era",
    )
    p.add_argument("--min-train-days", type=int, default=252)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_qqq_deep_anchor_scaffold")
    return p.parse_args()


def load_daily_stock_features(stock_root: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted(stock_root.glob("QQQ_*.parquet"))
    rows = []
    for fp in files:
        d = fp.stem.replace("QQQ_", "")
        if d < start or d > end:
            continue
        raw = pd.read_parquet(fp, columns=["timestamp", "close", "high", "low", "volume"])
        if raw.empty:
            continue
        ts = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("America/New_York")
        raw = raw.assign(_ts=ts)
        raw = raw[(raw["_ts"].dt.time >= pd.Timestamp("09:30").time()) & (raw["_ts"].dt.time < pd.Timestamp("16:00").time())]
        if raw.empty:
            continue
        raw = raw.sort_values("_ts")
        px = pd.to_numeric(raw["close"], errors="coerce")
        hi = pd.to_numeric(raw["high"], errors="coerce")
        lo = pd.to_numeric(raw["low"], errors="coerce")
        vol = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)
        # session VWAP approx
        vwap = (px * vol).sum() / vol.replace(0, np.nan).sum()
        rows.append(
            {
                "date_str": d,
                "close": float(px.iloc[-1]),
                "high": float(hi.max()),
                "low": float(lo.min()),
                "volume": float(vol.sum()),
                "vwap_dev_eod": float(px.iloc[-1] / vwap - 1.0) if np.isfinite(vwap) and vwap > 0 else 0.0,
            }
        )
    if not rows:
        raise SystemExit(f"no stock days in {stock_root} for {start}..{end}")
    df = pd.DataFrame(rows).sort_values("date_str").reset_index(drop=True)
    c = df["close"]
    df["ret_1d"] = c.pct_change(1)
    df["ret_5d"] = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    logret = np.log(c / c.shift(1))
    df["rv_20d"] = logret.rolling(20, min_periods=10).std() * np.sqrt(252)
    df["rv_60d"] = logret.rolling(60, min_periods=20).std() * np.sqrt(252)
    vol_mean = df["volume"].rolling(20, min_periods=10).mean()
    vol_std = df["volume"].rolling(20, min_periods=10).std().replace(0, np.nan)
    df["volume_z_20d"] = (df["volume"] - vol_mean) / vol_std
    df["range_20d"] = ((df["high"] - df["low"]) / c).rolling(20, min_periods=10).mean()
    df["trend_strength_20d"] = df["ret_20d"].abs() / (df["rv_20d"].replace(0, np.nan) / np.sqrt(252) * np.sqrt(20))
    df["month"] = df["date_str"].str.slice(0, 7)
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def assign_regime_buckets(train: pd.DataFrame) -> pd.Series:
    """Unsupervised 3-way risk/trend buckets from training distribution only."""
    rv = pd.to_numeric(train["rv_20d"], errors="coerce")
    tr = pd.to_numeric(train["ret_20d"], errors="coerce")
    rv_hi = rv.quantile(0.67)
    rv_lo = rv.quantile(0.33)
    # 0=calm, 1=trend, 2=stress
    lab = pd.Series(1, index=train.index, dtype=int)
    lab[(rv <= rv_lo) & (tr.abs() <= tr.abs().quantile(0.5))] = 0
    lab[(rv >= rv_hi) | (tr.abs() >= tr.abs().quantile(0.8))] = 2
    return lab


def causal_month_loop(daily: pd.DataFrame, min_train_days: int) -> pd.DataFrame:
    months = sorted(daily["month"].unique())
    out_rows = []
    for i, month in enumerate(months):
        test = daily[daily["month"] == month].copy()
        train = daily[daily["month"] < month].copy()
        if len(train) < min_train_days or test.empty:
            continue
        y = assign_regime_buckets(train)
        x_train = train[FEATURE_COLS].to_numpy()
        x_test = test[FEATURE_COLS].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=200,
            random_state=23,
            early_stopping=True,
            validation_fraction=0.15,
        )
        # need >=2 classes present
        if y.nunique() < 2:
            continue
        clf.fit(x_train, y.to_numpy())
        proba = clf.predict_proba(x_test)
        classes = list(clf.classes_)
        pred = clf.predict(x_test)
        for j, (_, row) in enumerate(test.iterrows()):
            rec = {
                "date_str": row["date_str"],
                "month": month,
                "regime_pred": int(pred[j]),
                "n_train_days": int(len(train)),
            }
            for k, cls in enumerate(classes):
                rec[f"p_regime_{int(cls)}"] = float(proba[j, k])
            for c in (0, 1, 2):
                rec.setdefault(f"p_regime_{c}", 0.0)
            out_rows.append(rec)
        print(f"[deep-anchor] month={month} train={len(train)} test={len(test)}", flush=True)
    return pd.DataFrame(out_rows)


def join_trade_diagnostics(regime: pd.DataFrame, trade_path: Path) -> dict:
    if not trade_path.exists() or regime.empty:
        return {"enabled": False, "reason": f"missing {trade_path}"}
    tr = pd.read_parquet(trade_path)
    if "date_str" not in tr.columns:
        tr["date_str"] = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m-%d")
    ret_col = "path_exec_ret" if "path_exec_ret" in tr.columns else None
    if ret_col is None:
        for c in tr.columns:
            if "exec_ret" in c or c.startswith("target_exec"):
                ret_col = c
                break
    if ret_col is None:
        return {"enabled": False, "reason": "no return column on trades"}
    day = (
        tr.groupby("date_str", as_index=False)
        .agg(n_trades=(ret_col, "size"), avg_ret=(ret_col, "mean"), sum_ret=(ret_col, "sum"))
    )
    m = day.merge(regime, on="date_str", how="inner")
    if m.empty:
        return {"enabled": False, "reason": "no overlap between trades and regime panel"}
    by = (
        m.groupby("regime_pred")
        .agg(days=("date_str", "nunique"), trades=("n_trades", "sum"), avg_day_ret=("avg_ret", "mean"))
        .reset_index()
    )
    # stress probability vs day return correlation
    if "p_regime_2" in m.columns:
        corr = float(pd.Series(m["p_regime_2"]).corr(m["avg_ret"]))
    else:
        corr = float("nan")
    return {
        "enabled": True,
        "overlap_days": int(m["date_str"].nunique()),
        "by_regime_pred": by.to_dict(orient="records"),
        "corr_p_stress_vs_day_avg_ret": corr,
        "note": "diagnostic only; negative corr would mean higher stress prob on worse days",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[deep-anchor] building daily stock features", flush=True)
    daily = load_daily_stock_features(Path(args.stock_root), args.start_date, args.end_date)
    daily.to_parquet(out_dir / "daily_stock_regime_features.parquet", index=False)
    print(f"[deep-anchor] days={len(daily)} {daily['date_str'].iloc[0]}..{daily['date_str'].iloc[-1]}", flush=True)

    preds = causal_month_loop(daily, args.min_train_days)
    preds.to_parquet(out_dir / "causal_regime_predictions.parquet", index=False)

    diag = join_trade_diagnostics(preds, Path(args.trade_days))
    summary = {
        "experiment": "qqq_deep_anchor_scaffold",
        "role": "Regime engine on stock 1s; NOT end-to-end trading",
        "data_note": {
            "stock_1s": "usable ~2022-03 onward (~3y+)",
            "option_micro": "currently ~2026 only; Tradeable/timing stay micro-window until backfill",
        },
        "config": vars(args),
        "n_feature_days": int(len(daily)),
        "n_pred_days": int(len(preds)),
        "months_scored": sorted(preds["month"].unique().tolist()) if not preds.empty else [],
        "trade_diagnostics": diag,
        "files": {
            "features": str(out_dir / "daily_stock_regime_features.parquet"),
            "predictions": str(out_dir / "causal_regime_predictions.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
        "next": [
            "gate curated / MAG7 TopK with p_regime_2 (stress) on Jul OOS",
            "backfill QQQ 2025 micro to expand Tradeable training",
            "upgrade MLP→temporal model only after micro >= 2 calendar years",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["n_feature_days", "n_pred_days", "months_scored", "trade_diagnostics", "next"]}, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
