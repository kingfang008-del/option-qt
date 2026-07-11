#!/usr/bin/env python3
"""Strict replay for 0DTE side/bucket/horizon routing models.

Train Jan-Feb, tune on Mar, replay Apr-Jun OOS using option-edge routing labels
and microstructure minute features.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from factor_lab.tools.option_edge_routing_common import (
    build_routing_dataset,
    routing_feature_matrix,
)
from factor_lab.tools.replay_0dte_micro_action import (
    ReplayParams,
    choose_contract,
    exit_trade,
    load_contract_minutes,
    month_key,
    summarize,
)


@dataclass(frozen=True)
class RoutingReplayParams(ReplayParams):
    use_predicted_bucket: bool = True
    use_predicted_horizon: bool = True


class RoutingModelBundle:
    def __init__(self, side_h: int):
        self.side_h = side_h
        self.side_model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=42,
        )
        self.bucket_model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=43,
        )
        self.horizon_model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=44,
        )
        self.edge_model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=5, min_samples_leaf=250,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=45,
        )

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        tr = train.dropna(subset=[f"gap_{self.side_h}m", "label_bucket", "label_horizon", "label_edge"]).copy()
        X = routing_feature_matrix(tr, features)
        self.side_model.fit(X, tr[f"gap_{self.side_h}m"].fillna(0.0).values)
        self.bucket_model.fit(X, tr["label_bucket"].astype(int).values)
        self.horizon_model.fit(X, tr["label_horizon"].astype(int).values)
        self.edge_model.fit(X, tr["label_edge"].values)

    def predict_frame(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        X = routing_feature_matrix(df, features)
        out = df.copy()
        out["pred_gap"] = self.side_model.predict(X)
        out["pred_side"] = np.where(out["pred_gap"] > 0, "CALL", "PUT")
        out["pred_bucket"] = self.bucket_model.predict(X).astype(int)
        out["pred_horizon"] = self.horizon_model.predict(X).astype(int)
        out["pred_edge"] = self.edge_model.predict(X)
        return out


def choose_contract_with_bucket(
    day_quotes: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    bucket_id: int | None,
    max_spread_pct: float,
) -> pd.Series | None:
    candidates = day_quotes[
        (day_quotes["minute_ts"] == entry_ts) & (day_quotes["side"].astype(str).str.upper() == side)
    ].copy()
    if candidates.empty:
        return None
    if bucket_id is not None:
        bucketed = candidates[pd.to_numeric(candidates["bucket_id"], errors="coerce") == bucket_id]
        if not bucketed.empty:
            candidates = bucketed
    candidates = candidates[
        (pd.to_numeric(candidates["ask"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["bid"], errors="coerce") > 0)
        & (pd.to_numeric(candidates["spread_pct"], errors="coerce") <= max_spread_pct)
    ].copy()
    if candidates.empty:
        return None
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


def replay_routing(
    df: pd.DataFrame,
    quote_map: dict[str, pd.DataFrame],
    params: RoutingReplayParams,
) -> tuple[dict, pd.DataFrame]:
    work = df.sort_values("timestamp").reset_index(drop=True)
    threshold = float(work["pred_edge"].quantile(params.entry_quantile))
    last_exit_by_day: dict[str, pd.Timestamp] = {}
    trades_by_day: dict[str, int] = {}
    trades = []
    for row in work.itertuples(index=False):
        sig_ts = pd.Timestamp(getattr(row, "timestamp"))
        date_str = getattr(row, "date_str")
        if date_str not in quote_map:
            continue
        if trades_by_day.get(date_str, 0) >= params.max_trades_per_day:
            continue
        if float(getattr(row, "pred_edge")) < threshold:
            continue
        last_exit = last_exit_by_day.get(date_str)
        if last_exit is not None and sig_ts <= last_exit + pd.Timedelta(minutes=params.cooldown):
            continue
        side = str(getattr(row, "pred_side"))
        bucket_id = int(getattr(row, "pred_bucket")) if params.use_predicted_bucket else None
        hold = int(getattr(row, "pred_horizon")) if params.use_predicted_horizon else params.horizon
        entry_ts = sig_ts + pd.Timedelta(minutes=params.entry_delay)
        day_quotes = quote_map[date_str]
        chosen = choose_contract_with_bucket(day_quotes, entry_ts, side, bucket_id, params.max_spread_pct)
        if chosen is None:
            continue
        entry_ask = float(chosen["ask"])
        rp = ReplayParams(
            horizon=hold,
            entry_quantile=params.entry_quantile,
            entry_delay=params.entry_delay,
            max_hold=hold,
            take_profit=params.take_profit,
            stop_loss=params.stop_loss,
            max_spread_pct=params.max_spread_pct,
            cooldown=params.cooldown,
            max_trades_per_day=params.max_trades_per_day,
            position_frac=params.position_frac,
        )
        exit_ts, net_ret, bars, reason = exit_trade(day_quotes, str(chosen["ticker"]), entry_ts, entry_ask, rp)
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
                "pred_bucket": bucket_id,
                "pred_horizon": hold,
                "actual_bucket": int(chosen["bucket_id"]),
                "ticker": str(chosen["ticker"]),
                "pred_edge": float(getattr(row, "pred_edge")),
                "pred_gap": float(getattr(row, "pred_gap")),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    p.add_argument("--output-dir", default="factor_lab/results/0dte_routing_replay")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.micro_root)
    label_dir = Path(args.label_dir).expanduser()
    horizons = [1, 3, 5, 10]
    train = build_routing_dataset(micro_root=root, label_dir=label_dir, symbol="QQQ", start="2026-01-01", end="2026-02-28", horizons=horizons)
    val = build_routing_dataset(micro_root=root, label_dir=label_dir, symbol="QQQ", start="2026-03-01", end="2026-03-31", horizons=horizons)
    test = build_routing_dataset(micro_root=root, label_dir=label_dir, symbol="QQQ", start="2026-04-01", end="2026-06-30", horizons=horizons)
    features = train.attrs["features"]
    quote_val = load_contract_minutes(root, "2026-03-01", "2026-03-31")
    quote_test = load_contract_minutes(root, "2026-04-01", "2026-06-30")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []
    best_by_side_h = []
    for side_h in horizons:
        bundle = RoutingModelBundle(side_h=side_h)
        bundle.fit(train, features)
        pred_val = bundle.predict_frame(val, features)
        for q in [0.80, 0.85, 0.90, 0.95]:
            for tp in [0.15, 0.25, 0.35]:
                for sl in [-0.10, -0.15, -0.20]:
                    params = RoutingReplayParams(
                        horizon=side_h,
                        max_hold=side_h,
                        entry_quantile=q,
                        take_profit=tp,
                        stop_loss=sl,
                        max_trades_per_day=8,
                        cooldown=3,
                    )
                    summary, _ = replay_routing(pred_val, quote_val, params)
                    score = summary.get("total_net_return", 0.0) - 0.5 * abs(summary.get("max_drawdown", 0.0))
                    grid_rows.append({"stage": "val", "side_h": side_h, "score": score, **summary})
        h_grid = pd.DataFrame([r for r in grid_rows if r.get("side_h") == side_h])
        elig = h_grid[h_grid["trades"].between(15, 400)]
        if elig.empty:
            elig = h_grid[h_grid["trades"] > 0]
        if elig.empty:
            continue
        best = elig.sort_values(["score", "total_net_return"], ascending=[False, False]).iloc[0].to_dict()
        best_by_side_h.append(best)
        print(
            f"best val side_h={side_h}: trades={best['trades']} total={best['total_net_return']:.4f} "
            f"pf={best['profit_factor']:.3f} q={best['entry_quantile']}"
        )

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "val_grid.csv", index=False)
    if not best_by_side_h:
        raise SystemExit("no valid val replay configs")
    best_overall = pd.DataFrame(best_by_side_h).sort_values(["score", "total_net_return"], ascending=[False, False]).iloc[0]
    side_h = int(best_overall["side_h"])
    train_full = build_routing_dataset(micro_root=root, label_dir=label_dir, symbol="QQQ", start="2026-01-01", end="2026-03-31", horizons=horizons)
    bundle = RoutingModelBundle(side_h=side_h)
    bundle.fit(train_full, features)
    pred_test = bundle.predict_frame(test, features)
    params = RoutingReplayParams(
        horizon=side_h,
        max_hold=side_h,
        entry_quantile=float(best_overall["entry_quantile"]),
        take_profit=float(best_overall["take_profit"]),
        stop_loss=float(best_overall["stop_loss"]),
        max_trades_per_day=8,
        cooldown=3,
    )
    test_summary, test_trades = replay_routing(pred_test, quote_test, params)
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
