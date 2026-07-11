#!/usr/bin/env python3
"""Strict replay for 0DTE microstructure action models.

Train on Jan-Feb, tune entry threshold on Mar, then replay Apr-Jun OOS.
Execution is intentionally strict:
  - signal at minute t
  - entry at t + entry_delay using contract ask
  - exit using future contract bid
  - contract selection uses only entry-time same-side quotes
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from factor_lab.tools.validate_0dte_micro_action import build_dataset


@dataclass(frozen=True)
class ReplayParams:
    horizon: int
    entry_quantile: float
    entry_delay: int = 1
    max_hold: int = 30
    take_profit: float = 0.35
    stop_loss: float = -0.18
    max_spread_pct: float = 0.20
    cooldown: int = 5
    max_trades_per_day: int = 4
    position_frac: float = 0.25


def month_key(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("America/New_York").strftime("%Y-%m")


def load_contract_minutes(root: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    files = sorted((root / "contract_1s/QQQ").glob("QQQ_*.parquet"))
    for f in files:
        date_str = f.stem.replace("QQQ_", "")
        if not (start <= date_str <= end):
            continue
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df["minute_ts"] = df["timestamp"].dt.floor("min")
        keep = [
            "minute_ts",
            "ticker",
            "bucket_id",
            "side",
            "bid",
            "ask",
            "spread_pct",
            "trade_volume",
            "quote_events",
            "buy_ratio",
        ]
        cols = [c for c in keep if c in df.columns]
        q = df.sort_values("timestamp").drop_duplicates(["ticker", "minute_ts"], keep="last")[cols].copy()
        for c in ["bid", "ask", "spread_pct", "trade_volume", "quote_events", "buy_ratio"]:
            if c in q.columns:
                q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0.0)
        out[date_str] = q.sort_values(["minute_ts", "bucket_id", "ticker"]).reset_index(drop=True)
    return out


def choose_contract(day_quotes: pd.DataFrame, entry_ts: pd.Timestamp, side: str, max_spread_pct: float) -> pd.Series | None:
    candidates = day_quotes[(day_quotes["minute_ts"] == entry_ts) & (day_quotes["side"].astype(str).str.upper() == side)].copy()
    if candidates.empty:
        return None
    candidates = candidates[
        (pd.to_numeric(candidates["ask"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["bid"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["spread_pct"], errors="coerce") <= max_spread_pct)
    ].copy()
    if candidates.empty:
        return None
    # Entry-time liquidity score only: tight spread first, then active contracts.
    vol = (
        pd.to_numeric(candidates["trade_volume"], errors="coerce").fillna(0.0)
        if "trade_volume" in candidates.columns
        else 0.0
    )
    qev = (
        pd.to_numeric(candidates["quote_events"], errors="coerce").fillna(0.0)
        if "quote_events" in candidates.columns
        else 0.0
    )
    candidates["liq_score"] = (
        -pd.to_numeric(candidates["spread_pct"], errors="coerce").fillna(9.0)
        + 1e-5 * vol
        + 1e-6 * qev
    )
    return candidates.sort_values(["liq_score", "bucket_id"], ascending=[False, True]).iloc[0]


def exit_trade(
    day_quotes: pd.DataFrame,
    ticker: str,
    entry_ts: pd.Timestamp,
    entry_ask: float,
    params: ReplayParams,
) -> tuple[pd.Timestamp, float, int, str]:
    q = day_quotes[day_quotes["ticker"] == ticker].sort_values("minute_ts")
    future = q[q["minute_ts"] > entry_ts].head(params.max_hold)
    if future.empty:
        return entry_ts, -1.0, 0, "NO_FUTURE"
    last_ret = -1.0
    last_ts = entry_ts
    bars = 0
    for i, row in enumerate(future.itertuples(index=False), start=1):
        bid = float(getattr(row, "bid", 0.0))
        if bid <= 0 or entry_ask <= 0:
            continue
        ret = bid / entry_ask - 1.0
        last_ret = ret
        last_ts = getattr(row, "minute_ts")
        bars = i
        if ret <= params.stop_loss:
            return last_ts, ret, bars, "STOP"
        if ret >= params.take_profit:
            return last_ts, ret, bars, "TAKE_PROFIT"
    return last_ts, last_ret, bars, "TIME"


def replay(df: pd.DataFrame, pred: np.ndarray, quote_map: dict[str, pd.DataFrame], params: ReplayParams) -> tuple[dict, pd.DataFrame]:
    work = df.copy()
    work["pred"] = pred
    work["abs_pred"] = np.abs(pred)
    threshold = float(work["abs_pred"].quantile(params.entry_quantile))
    work = work.sort_values("timestamp").reset_index(drop=True)
    last_exit_by_day: dict[str, pd.Timestamp] = {}
    trades = []
    trades_by_day: dict[str, int] = {}
    for row in work.itertuples(index=False):
        sig_ts = pd.Timestamp(getattr(row, "timestamp"))
        date_str = getattr(row, "date_str")
        if date_str not in quote_map:
            continue
        if trades_by_day.get(date_str, 0) >= params.max_trades_per_day:
            continue
        if abs(float(getattr(row, "pred"))) < threshold:
            continue
        last_exit = last_exit_by_day.get(date_str)
        if last_exit is not None and sig_ts <= last_exit + pd.Timedelta(minutes=params.cooldown):
            continue
        side = "CALL" if float(getattr(row, "pred")) > 0 else "PUT"
        entry_ts = sig_ts + pd.Timedelta(minutes=params.entry_delay)
        day_quotes = quote_map[date_str]
        chosen = choose_contract(day_quotes, entry_ts, side, params.max_spread_pct)
        if chosen is None:
            continue
        entry_ask = float(chosen["ask"])
        exit_ts, net_ret, bars, reason = exit_trade(day_quotes, str(chosen["ticker"]), entry_ts, entry_ask, params)
        if bars <= 0:
            continue
        trades.append(
            {
                "signal_ts": sig_ts,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "date_str": date_str,
                "month": month_key(entry_ts),
                "side": side,
                "ticker": str(chosen["ticker"]),
                "bucket_id": int(chosen["bucket_id"]),
                "pred": float(getattr(row, "pred")),
                "entry_ask": entry_ask,
                "net_return": float(net_ret),
                "bars_held": int(bars),
                "exit_reason": reason,
            }
        )
        trades_by_day[date_str] = trades_by_day.get(date_str, 0) + 1
        last_exit_by_day[date_str] = exit_ts
    trades_df = pd.DataFrame(trades)
    return summarize(trades_df, params), trades_df


def summarize(trades: pd.DataFrame, params: ReplayParams) -> dict:
    if trades.empty:
        return {**asdict(params), "trades": 0, "total_net_return": 0.0}
    r = pd.to_numeric(trades["net_return"], errors="coerce").fillna(0.0)
    eq = (1.0 + params.position_frac * r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        **asdict(params),
        "trades": int(len(trades)),
        "total_net_return": float(eq.iloc[-1] - 1.0),
        "sum_net_return": float(r.sum()),
        "avg_net_return": float(r.mean()),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
        "worst_trade": float(r.min()),
        "trades_by_side": trades["side"].value_counts().to_dict(),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def fit_predict(train: pd.DataFrame, eval_df: pd.DataFrame, features: list[str], horizon: int) -> np.ndarray:
    target = f"gap_{horizon}m"
    tr = train.dropna(subset=[target]).copy()
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
    return model.predict(eval_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_micro_replay")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.micro_root)
    horizons = [5, 10, 15, 30]
    train = build_dataset(root, "2026-01-01", "2026-02-28", horizons)
    val = build_dataset(root, "2026-03-01", "2026-03-31", horizons)
    test = build_dataset(root, "2026-04-01", "2026-06-30", horizons)
    features = train.attrs["features"]
    quote_val = load_contract_minutes(root, "2026-03-01", "2026-03-31")
    quote_test = load_contract_minutes(root, "2026-04-01", "2026-06-30")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    best_by_h = []
    for h in horizons:
        pred_val = fit_predict(train, val, features, h)
        for q in [0.80, 0.85, 0.90, 0.95, 0.975]:
            for tp in [0.20, 0.35, 0.50]:
                for sl in [-0.12, -0.18, -0.25]:
                    params = ReplayParams(horizon=h, max_hold=h, entry_quantile=q, take_profit=tp, stop_loss=sl)
                    summary, _ = replay(val, pred_val, quote_val, params)
                    score = summary.get("total_net_return", 0.0) - 0.5 * abs(summary.get("max_drawdown", 0.0))
                    grid_rows.append({"stage": "val", "score": score, **summary})
        h_grid = pd.DataFrame([r for r in grid_rows if r["horizon"] == h])
        elig = h_grid[h_grid["trades"].between(10, 160)]
        if elig.empty:
            elig = h_grid[h_grid["trades"] > 0]
        best = elig.sort_values(["score", "total_net_return"], ascending=[False, False]).iloc[0].to_dict()
        best_by_h.append(best)
        print(f"best val h={h}: trades={best['trades']} total={best['total_net_return']:.4f} pf={best['profit_factor']:.3f} q={best['entry_quantile']} tp={best['take_profit']} sl={best['stop_loss']}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "val_grid.csv", index=False)
    best_overall = pd.DataFrame(best_by_h).sort_values(["score", "total_net_return"], ascending=[False, False]).iloc[0]
    h = int(best_overall["horizon"])
    # Final model: train on Jan-Mar after selecting params, test on Apr-Jun.
    train_full = build_dataset(root, "2026-01-01", "2026-03-31", horizons)
    pred_test = fit_predict(train_full, test, features, h)
    params = ReplayParams(
        horizon=h,
        max_hold=h,
        entry_quantile=float(best_overall["entry_quantile"]),
        take_profit=float(best_overall["take_profit"]),
        stop_loss=float(best_overall["stop_loss"]),
    )
    test_summary, test_trades = replay(test, pred_test, quote_test, params)
    test_trades.to_parquet(out_dir / "test_trades.parquet", index=False)
    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize(g, params)
    payload = {
        "selected_val": best_overall.to_dict(),
        "test": test_summary,
        "test_monthly": monthly,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("SELECTED", json.dumps(payload["selected_val"], indent=2, default=str))
    print("TEST", json.dumps(test_summary, indent=2, default=str))
    print("MONTHLY", json.dumps(monthly, indent=2, default=str))


if __name__ == "__main__":
    main()
