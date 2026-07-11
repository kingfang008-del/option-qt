#!/usr/bin/env python3
"""Minimal five-layer QQQ 0DTE loop.

The goal is not to build a production system.  It is to test whether separating
market state, microstructure factors, edge scoring, execution, and weight
learning gives a cleaner OOS result than direct price prediction/replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from factor_lab.tools.analyze_0dte_tradeprint_factors import CORE_FACTORS, load_factor_dataset
from factor_lab.tools.eval_0dte_tradeprint_factor_gates import add_composite_scores


STATE_FEATURES = [
    "panel_notional_60s",
    "panel_quote_10s",
    "panel_spread",
    "panel_abs_mom_10s",
    "state_activity_q",
    "state_quote_q",
    "state_spread_q",
    "state_abs_momentum_q",
    "state_put_minus_call_mom",
    "state_call_minus_put_mom",
    "state_stock_abs_mom_q",
    "state_stock_volume_q",
    "stock_ret_10s",
    "stock_ret_30s",
    "stock_ret_60s",
    "stock_vwap_dev",
    "stock_rv_60s",
    "stock_volume_z_60s",
    "is_vol_expansion",
    "is_liquidity_stress",
    "is_range_pin_proxy",
    "is_put_trend_proxy",
    "is_call_trend_proxy",
    "is_stock_trend_up",
    "is_stock_trend_down",
    "is_stock_vwap_extension",
]


EXTRA_FACTORS = [
    "rank_notional_60s",
    "rank_quote_10s",
    "rank_quote_imbalance",
    "rank_spread_tight",
    "score_hot_quote",
    "score_hot_quote_tight",
    "score_hot_quote_imb",
    "universe_rank",
    "rolling_notional",
    "rolling_trades",
    "side_code",
]


def load_stock_state_features(stock_root: Path, start: str, end: str, symbol: str = "QQQ") -> pd.DataFrame:
    sym = str(symbol).upper()
    files = sorted(stock_root.glob(f"{sym}_*.parquet"))
    prefix = f"{sym}_"
    files = [p for p in files if start <= p.stem.replace(prefix, "") <= end]
    frames = []
    for fp in files:
        raw = pd.read_parquet(fp)
        if raw.empty:
            continue
        df = raw.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df = df[
            (df["timestamp"].dt.time >= pd.Timestamp("09:30").time())
            & (df["timestamp"].dt.time < pd.Timestamp("16:00").time())
        ].copy()
        if df.empty:
            continue
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date_str"] = fp.stem.replace(prefix, "")
        px = df["close"]
        vol = df["volume"].fillna(0.0)
        for w in (10, 30, 60):
            df[f"stock_ret_{w}s"] = px / px.shift(w) - 1.0
        df["stock_abs_ret_30s"] = df["stock_ret_30s"].abs()
        df["stock_abs_ret_60s"] = df["stock_ret_60s"].abs()
        cum_dollar = (px * vol).groupby(df["date_str"]).cumsum()
        cum_vol = vol.groupby(df["date_str"]).cumsum().replace(0, np.nan)
        df["stock_vwap"] = cum_dollar / cum_vol
        df["stock_vwap_dev"] = px / df["stock_vwap"] - 1.0
        df["stock_rv_60s"] = np.log(px / px.shift(1)).rolling(60, min_periods=20).std()
        vol_mean = vol.rolling(600, min_periods=120).mean()
        vol_std = vol.rolling(600, min_periods=120).std().replace(0, np.nan)
        df["stock_volume_z_60s"] = (vol.rolling(60, min_periods=20).sum() - vol_mean * 60) / (
            vol_std * np.sqrt(60)
        )
        keep = [
            "timestamp",
            "stock_ret_10s",
            "stock_ret_30s",
            "stock_ret_60s",
            "stock_abs_ret_30s",
            "stock_abs_ret_60s",
            "stock_vwap_dev",
            "stock_rv_60s",
            "stock_volume_z_60s",
        ]
        frames.append(df[keep])
    if not frames:
        raise SystemExit(f"no stock state files for {start}..{end}: {stock_root}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def add_market_state_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple, observable market-state proxies from the option panel."""
    work = df.copy()
    work["abs_mid_mom_10s"] = pd.to_numeric(work["mid_ret_past_10s"], errors="coerce").abs()
    panel = (
        work.groupby("timestamp")
        .agg(
            panel_notional_60s=("trade_notional_sum_60s", "mean"),
            panel_quote_10s=("quote_events_sum_10s", "mean"),
            panel_spread=("spread_pct", "mean"),
            panel_abs_mom_10s=("abs_mid_mom_10s", "mean"),
            call_mom_10s=("mid_ret_past_10s", lambda x: x[work.loc[x.index, "side"].eq("CALL")].mean()),
            put_mom_10s=("mid_ret_past_10s", lambda x: x[work.loc[x.index, "side"].eq("PUT")].mean()),
        )
        .reset_index()
    )
    panel["state_put_minus_call_mom"] = panel["put_mom_10s"].fillna(0.0) - panel["call_mom_10s"].fillna(0.0)
    panel["state_call_minus_put_mom"] = -panel["state_put_minus_call_mom"]
    state_cols = [
        "timestamp",
        "panel_notional_60s",
        "panel_quote_10s",
        "panel_spread",
        "panel_abs_mom_10s",
        "state_put_minus_call_mom",
        "state_call_minus_put_mom",
    ]
    return work.merge(panel[state_cols], on="timestamp", how="left")


def fit_state_thresholds(train: pd.DataFrame) -> dict[str, list[float]]:
    cols = [
        "panel_notional_60s",
        "panel_quote_10s",
        "panel_spread",
        "panel_abs_mom_10s",
        "state_put_minus_call_mom",
        "state_call_minus_put_mom",
        "stock_abs_ret_30s",
        "stock_volume_z_60s",
        "stock_vwap_dev",
    ]
    thresholds: dict[str, list[float]] = {}
    for col in cols:
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        thresholds[col] = [float(s.quantile(q)) for q in np.linspace(0.1, 0.9, 9)] if len(s) else [0.0] * 9
    return thresholds


def percentile_from_thresholds(values: pd.Series, cuts: list[float]) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.searchsorted(np.asarray(cuts, dtype=float), arr, side="right") / (len(cuts) + 1.0)
    return pd.Series(out, index=values.index).fillna(0.0)


def apply_market_state_thresholds(df: pd.DataFrame, thresholds: dict[str, list[float]]) -> pd.DataFrame:
    out = df.copy()
    out["state_activity_q"] = percentile_from_thresholds(out["panel_notional_60s"], thresholds["panel_notional_60s"])
    out["state_quote_q"] = percentile_from_thresholds(out["panel_quote_10s"], thresholds["panel_quote_10s"])
    out["state_spread_q"] = percentile_from_thresholds(out["panel_spread"], thresholds["panel_spread"])
    out["state_abs_momentum_q"] = percentile_from_thresholds(out["panel_abs_mom_10s"], thresholds["panel_abs_mom_10s"])
    put_trend_q = percentile_from_thresholds(out["state_put_minus_call_mom"], thresholds["state_put_minus_call_mom"])
    call_trend_q = percentile_from_thresholds(out["state_call_minus_put_mom"], thresholds["state_call_minus_put_mom"])
    stock_abs_q = percentile_from_thresholds(out["stock_abs_ret_30s"], thresholds["stock_abs_ret_30s"])
    stock_volume_q = percentile_from_thresholds(out["stock_volume_z_60s"], thresholds["stock_volume_z_60s"])
    stock_vwap_q = percentile_from_thresholds(out["stock_vwap_dev"], thresholds["stock_vwap_dev"])
    stock_vwap_down_q = percentile_from_thresholds(-out["stock_vwap_dev"], [-x for x in reversed(thresholds["stock_vwap_dev"])])
    out["state_stock_abs_mom_q"] = stock_abs_q
    out["state_stock_volume_q"] = stock_volume_q
    out["is_vol_expansion"] = (
        (out["state_activity_q"] >= 0.70)
        & (out["state_quote_q"] >= 0.70)
        & ((out["state_abs_momentum_q"] >= 0.60) | (out["state_stock_abs_mom_q"] >= 0.70))
    ).astype(float)
    out["is_liquidity_stress"] = (
        (out["state_spread_q"] >= 0.80)
        & ((out["state_quote_q"] >= 0.70) | (out["state_activity_q"] >= 0.70))
    ).astype(float)
    out["is_range_pin_proxy"] = (
        (out["state_abs_momentum_q"] <= 0.35)
        & (out["state_stock_abs_mom_q"] <= 0.50)
        & (out["state_spread_q"] <= 0.50)
        & (out["state_quote_q"] >= 0.50)
    ).astype(float)
    out["is_put_trend_proxy"] = ((put_trend_q >= 0.75) & (out["state_abs_momentum_q"] >= 0.60)).astype(float)
    out["is_call_trend_proxy"] = ((call_trend_q >= 0.75) & (out["state_abs_momentum_q"] >= 0.60)).astype(float)
    out["is_stock_trend_up"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce") > 0)
        & (stock_abs_q >= 0.70)
        & (stock_volume_q >= 0.50)
    ).astype(float)
    out["is_stock_trend_down"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce") < 0)
        & (stock_abs_q >= 0.70)
        & (stock_volume_q >= 0.50)
    ).astype(float)
    out["is_stock_vwap_extension"] = ((stock_vwap_q >= 0.80) | (stock_vwap_down_q >= 0.80)).astype(float)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CORE_FACTORS if c in df.columns]
    cols.extend(c for c in EXTRA_FACTORS if c in df.columns)
    cols.extend(c for c in STATE_FEATURES if c in df.columns)
    return sorted(set(cols))


def build_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    work = add_market_state_raw(add_composite_scores(df))
    features = feature_columns(work)
    need = [c for c in features if c in work.columns] + [target, "timestamp", "date_str", "side", "ticker"]
    clean = work.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    return clean, [c for c in features if c in clean.columns]


def fit_edge_model(train: pd.DataFrame, features: list[str], target: str, max_train_rows: int) -> HistGradientBoostingRegressor:
    sample = train
    if max_train_rows > 0 and len(train) > max_train_rows:
        sample = train.sample(max_train_rows, random_state=17)
    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=17,
    )
    X = sample[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(sample[target], errors="coerce").fillna(0.0).to_numpy()
    model.fit(X, y)
    return model


def predict_edge(model: HistGradientBoostingRegressor, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    return model.predict(X)


def quantile_report(df: pd.DataFrame, score: str, target: str) -> dict:
    out = {
        "n": int(len(df)),
        "target_mean": float(df[target].mean()),
        "target_pos_rate": float((df[target] > 0).mean()),
    }
    for q, tag in [(0.90, "top10"), (0.95, "top5"), (0.99, "top1"), (0.995, "top0_5")]:
        th = df[score].quantile(q)
        sel = df[df[score] >= th]
        out[f"{tag}_n"] = int(len(sel))
        out[f"{tag}_mean"] = float(sel[target].mean()) if len(sel) else 0.0
        out[f"{tag}_pos_rate"] = float((sel[target] > 0).mean()) if len(sel) else 0.0
    return out


def replay_daily_topk(
    df: pd.DataFrame,
    *,
    score_col: str,
    target: str,
    topk: int,
    cooldown_s: int,
    side: str | None,
    state_filter: str | None,
) -> dict:
    work = df.copy()
    if side:
        work = work[work["side"].eq(side)].copy()
    if state_filter:
        work = work[pd.to_numeric(work[state_filter], errors="coerce").fillna(0.0) > 0.5].copy()
    trades = []
    for _, g in work.sort_values(["date_str", score_col], ascending=[True, False]).groupby("date_str"):
        last_ts = None
        chosen = 0
        for row in g.sort_values(score_col, ascending=False).itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "timestamp"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            trades.append(row._asdict())
            last_ts = ts
            chosen += 1
            if chosen >= topk:
                break
    tr = pd.DataFrame(trades)
    if tr.empty:
        return {"trades": 0}
    r = pd.to_numeric(tr[target], errors="coerce").fillna(0.0)
    eq = (1.0 + 0.10 * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        "trades": int(len(tr)),
        "days": int(tr["date_str"].nunique()),
        "side": side or "ALL",
        "state_filter": state_filter or "none",
        "topk_per_day": int(topk),
        "avg_return": float(r.mean()),
        "sum_return": float(r.sum()),
        "total_return_10pct_position": float(eq.iloc[-1] - 1.0),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
        "side_counts": tr["side"].value_counts().to_dict(),
        "state_counts": {
            c: int(pd.to_numeric(tr[c], errors="coerce").fillna(0.0).gt(0.5).sum())
            for c in [
                "is_vol_expansion",
                "is_range_pin_proxy",
                "is_put_trend_proxy",
                "is_call_trend_proxy",
                "is_liquidity_stress",
                "is_stock_trend_up",
                "is_stock_trend_down",
                "is_stock_vwap_extension",
            ]
            if c in tr.columns
        },
    }


def evaluate_replays(df: pd.DataFrame, target: str, cooldown_s: int) -> pd.DataFrame:
    rows = []
    states = [
        None,
        "is_vol_expansion",
        "is_range_pin_proxy",
        "is_put_trend_proxy",
        "is_call_trend_proxy",
        "is_stock_trend_up",
        "is_stock_trend_down",
        "is_stock_vwap_extension",
    ]
    for side in [None, "PUT", "CALL"]:
        for state in states:
            for topk in (1, 2, 3, 5):
                rows.append(
                    replay_daily_topk(
                        df,
                        score_col="edge_score",
                        target=target,
                        topk=topk,
                        cooldown_s=cooldown_s,
                        side=side,
                        state_filter=state,
                    )
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
    p.add_argument("--max-train-rows", type=int, default=500_000)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_minimal_five_layer_loop")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon_s)
    target = f"target_exec_ret_{horizon}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[five-layer] loading train", flush=True)
    raw_train = load_factor_dataset(
        Path(args.micro_root),
        args.train_start,
        args.train_end,
        (horizon,),
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    print("[five-layer] loading test", flush=True)
    raw_test = load_factor_dataset(
        Path(args.micro_root),
        args.test_start,
        args.test_end,
        (horizon,),
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    print("[five-layer] loading stock state", flush=True)
    stock_train = load_stock_state_features(Path(args.stock_root), args.train_start, args.train_end)
    stock_test = load_stock_state_features(Path(args.stock_root), args.test_start, args.test_end)
    raw_train = raw_train.merge(stock_train, on="timestamp", how="inner")
    raw_test = raw_test.merge(stock_test, on="timestamp", how="inner")
    train, _ = build_dataset(raw_train, target)
    test, _ = build_dataset(raw_test, target)
    state_thresholds = fit_state_thresholds(train)
    train = apply_market_state_thresholds(train, state_thresholds)
    test = apply_market_state_thresholds(test, state_thresholds)
    features = [c for c in feature_columns(train) if c in test.columns]
    keep = features + [target, "timestamp", "date_str", "side", "ticker"]
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
    print(f"[five-layer] train={len(train)} test={len(test)} features={len(features)}", flush=True)

    model = fit_edge_model(train, features, target, args.max_train_rows)
    train["edge_score"] = predict_edge(model, train, features)
    test["edge_score"] = predict_edge(model, test, features)
    train["baseline_hot_score"] = train["score_hot_quote_tight"]
    test["baseline_hot_score"] = test["score_hot_quote_tight"]

    train_r2 = r2_score(train[target], train["edge_score"])
    test_r2 = r2_score(test[target], test["edge_score"])
    model_quantile_train = quantile_report(train, "edge_score", target)
    model_quantile_test = quantile_report(test, "edge_score", target)
    baseline_quantile_test = quantile_report(test, "baseline_hot_score", target)
    replay = evaluate_replays(test, target, args.cooldown_s)
    replay.to_csv(out_dir / "oos_replay_grid.csv", index=False)
    test[
        [
            "timestamp",
            "date_str",
            "ticker",
            "side",
            target,
            "edge_score",
            "baseline_hot_score",
            *[c for c in STATE_FEATURES if c in test.columns],
        ]
    ].to_parquet(out_dir / "test_edge_panel.parquet", index=False)

    best = replay[replay["trades"].fillna(0) >= 10].sort_values(
        ["avg_return", "profit_factor"], ascending=False
    ).head(20)
    summary = {
        "config": vars(args),
        "target": target,
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "features": features,
        "state_thresholds": state_thresholds,
        "model": {
            "type": "HistGradientBoostingRegressor",
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "max_train_rows": int(args.max_train_rows),
        },
        "quantiles": {
            "train_model": model_quantile_train,
            "test_model": model_quantile_test,
            "test_baseline_hot_score": baseline_quantile_test,
        },
        "best_oos_replays": best.to_dict("records"),
        "files": {
            "oos_replay_grid": str(out_dir / "oos_replay_grid.csv"),
            "test_edge_panel": str(out_dir / "test_edge_panel.parquet"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
