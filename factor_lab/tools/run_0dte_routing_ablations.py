#!/usr/bin/env python3
"""Run 0DTE routing ablations: fixed horizon + side-only vs full joint routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from factor_lab.tools.option_edge_routing_common import (
    build_routing_dataset,
    routing_feature_matrix,
)
from factor_lab.tools.replay_0dte_routing_model import (
    RoutingModelBundle,
    RoutingReplayParams,
    replay_routing,
)
from factor_lab.tools.replay_0dte_micro_action import (
    ReplayParams,
    choose_contract,
    exit_trade,
    load_contract_minutes,
    month_key,
    summarize,
)


class SideOnlyBundle:
    def __init__(self, side_h: int):
        self.side_h = side_h
        self.side_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=250,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

    def fit(self, train: pd.DataFrame, features: list[str]) -> None:
        target = f"gap_{self.side_h}m"
        tr = train.dropna(subset=[target]).copy()
        self.side_model.fit(routing_feature_matrix(tr, features), tr[target].values)

    def predict_frame(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = df.copy()
        out["pred_gap"] = self.side_model.predict(routing_feature_matrix(out, features))
        out["pred_side"] = np.where(out["pred_gap"] > 0, "CALL", "PUT")
        out["pred_edge"] = np.abs(out["pred_gap"])
        return out


def replay_side_only(
    df: pd.DataFrame,
    quote_map: dict[str, pd.DataFrame],
    params: ReplayParams,
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
        entry_ts = sig_ts + pd.Timedelta(minutes=params.entry_delay)
        day_quotes = quote_map[date_str]
        chosen = choose_contract(day_quotes, entry_ts, side, params.max_spread_pct)
        if chosen is None:
            continue
        entry_ask = float(chosen["ask"])
        exit_ts, net_ret, bars, reason = exit_trade(
            day_quotes, str(chosen["ticker"]), entry_ts, entry_ask, params
        )
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


def grid_search_val(
    pred_val: pd.DataFrame,
    quote_val: dict[str, pd.DataFrame],
    *,
    mode: str,
    exec_h: int,
    side_h: int,
    use_bucket: bool,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for q in [0.85, 0.90, 0.95, 0.975]:
        for tp in [0.15, 0.25, 0.35, 0.50]:
            for sl in [-0.10, -0.15, -0.20, -0.25]:
                if mode == "side_only":
                    params = ReplayParams(
                        horizon=exec_h,
                        max_hold=exec_h,
                        entry_quantile=q,
                        take_profit=tp,
                        stop_loss=sl,
                        max_trades_per_day=8,
                        cooldown=3,
                    )
                    summary, _ = replay_side_only(pred_val, quote_val, params)
                else:
                    params = RoutingReplayParams(
                        horizon=exec_h,
                        max_hold=exec_h,
                        entry_quantile=q,
                        take_profit=tp,
                        stop_loss=sl,
                        max_trades_per_day=8,
                        cooldown=3,
                        use_predicted_bucket=use_bucket,
                        use_predicted_horizon=False,
                    )
                    summary, _ = replay_routing(pred_val, quote_val, params)
                score = summary.get("total_net_return", 0.0) - 0.5 * abs(summary.get("max_drawdown", 0.0))
                rows.append(
                    {
                        "mode": mode,
                        "side_h": side_h,
                        "exec_h": exec_h,
                        "use_bucket": use_bucket,
                        "score": score,
                        **summary,
                    }
                )
    grid = pd.DataFrame(rows)
    elig = grid[grid["trades"].between(10, 400)]
    if elig.empty:
        elig = grid[grid["trades"] > 0]
    if elig.empty:
        return {}, rows
    best = elig.sort_values(["score", "total_net_return"], ascending=[False, False]).iloc[0].to_dict()
    return best, rows


def run_variant(
    *,
    name: str,
    mode: str,
    side_h: int,
    exec_h: int,
    use_bucket: bool,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_full: pd.DataFrame,
    features: list[str],
    quote_val: dict[str, pd.DataFrame],
    quote_test: dict[str, pd.DataFrame],
) -> dict:
    if mode == "side_only":
        bundle = SideOnlyBundle(side_h=side_h)
    else:
        bundle = RoutingModelBundle(side_h=side_h)
    bundle.fit(train, features)
    pred_val = bundle.predict_frame(val, features)
    best, grid_rows = grid_search_val(
        pred_val, quote_val, mode=mode, exec_h=exec_h, side_h=side_h, use_bucket=use_bucket
    )
    if not best:
        return {"name": name, "error": "no val trades"}

    bundle.fit(train_full, features)
    pred_test = bundle.predict_frame(test, features)
    if mode == "side_only":
        params = ReplayParams(
            horizon=exec_h,
            max_hold=exec_h,
            entry_quantile=float(best["entry_quantile"]),
            take_profit=float(best["take_profit"]),
            stop_loss=float(best["stop_loss"]),
            max_trades_per_day=8,
            cooldown=3,
        )
        test_summary, test_trades = replay_side_only(pred_test, quote_test, params)
    else:
        params = RoutingReplayParams(
            horizon=exec_h,
            max_hold=exec_h,
            entry_quantile=float(best["entry_quantile"]),
            take_profit=float(best["take_profit"]),
            stop_loss=float(best["stop_loss"]),
            max_trades_per_day=8,
            cooldown=3,
            use_predicted_bucket=use_bucket,
            use_predicted_horizon=False,
        )
        test_summary, test_trades = replay_routing(pred_test, quote_test, params)

    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize(g, params)

    return {
        "name": name,
        "mode": mode,
        "side_h": side_h,
        "exec_h": exec_h,
        "use_bucket": use_bucket,
        "selected_val": best,
        "test": test_summary,
        "test_monthly": monthly,
        "grid_rows": len(grid_rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    p.add_argument("--output", default="factor_lab/results/0dte_routing_ablations.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.micro_root)
    label_dir = Path(args.label_dir).expanduser()
    horizons = [1, 3, 5, 10]

    train = build_routing_dataset(
        micro_root=root, label_dir=label_dir, symbol="QQQ",
        start="2026-01-01", end="2026-02-28", horizons=horizons,
    )
    val = build_routing_dataset(
        micro_root=root, label_dir=label_dir, symbol="QQQ",
        start="2026-03-01", end="2026-03-31", horizons=horizons,
    )
    test = build_routing_dataset(
        micro_root=root, label_dir=label_dir, symbol="QQQ",
        start="2026-04-01", end="2026-06-30", horizons=horizons,
    )
    train_full = build_routing_dataset(
        micro_root=root, label_dir=label_dir, symbol="QQQ",
        start="2026-01-01", end="2026-03-31", horizons=horizons,
    )
    features = train.attrs["features"]
    quote_val = load_contract_minutes(root, "2026-03-01", "2026-03-31")
    quote_test = load_contract_minutes(root, "2026-04-01", "2026-06-30")

    variants = [
        ("joint_h1", "joint", 1, 1, True),
        ("joint_h5", "joint", 5, 5, True),
        ("joint_h10", "joint", 10, 10, True),
        ("joint_h5_side1", "joint", 1, 5, True),
        ("joint_h10_side5", "joint", 5, 10, True),
        ("side_only_h5", "side_only", 5, 5, False),
        ("side_only_h10", "side_only", 10, 10, False),
        ("side_only_h5_side1", "side_only", 1, 5, False),
        ("side_only_h10_side5", "side_only", 5, 10, False),
        ("joint_h5_no_bucket", "joint", 5, 5, False),
        ("joint_h10_no_bucket", "joint", 10, 10, False),
    ]

    results = []
    for name, mode, side_h, exec_h, use_bucket in variants:
        print(f"=== {name} mode={mode} side_h={side_h} exec_h={exec_h} bucket={use_bucket} ===")
        out = run_variant(
            name=name,
            mode=mode,
            side_h=side_h,
            exec_h=exec_h,
            use_bucket=use_bucket,
            train=train,
            val=val,
            test=test,
            train_full=train_full,
            features=features,
            quote_val=quote_val,
            quote_test=quote_test,
        )
        results.append(out)
        if "error" in out:
            print(f"  {name}: {out['error']}")
            continue
        t = out["test"]
        print(
            f"  val={out['selected_val']['total_net_return']:.3f} "
            f"test trades={t['trades']} total={t['total_net_return']:.3f} pf={t['profit_factor']:.3f}"
        )

    ranked = sorted(
        [r for r in results if "test" in r],
        key=lambda x: x["test"].get("total_net_return", -999.0),
        reverse=True,
    )
    payload = {
        "baseline_joint_h1_test_total": -0.7316,
        "variants": results,
        "ranked_by_test_total": [
            {
                "name": r["name"],
                "test_total": r["test"]["total_net_return"],
                "test_trades": r["test"]["trades"],
                "test_pf": r["test"]["profit_factor"],
                "val_total": r["selected_val"]["total_net_return"],
                "exec_h": r["exec_h"],
                "side_h": r["side_h"],
            }
            for r in ranked
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n=== RANKED BY TEST TOTAL ===")
    for row in payload["ranked_by_test_total"]:
        print(
            f"{row['name']:22s} test={row['test_total']:+.3f} "
            f"trades={row['test_trades']:3d} pf={row['test_pf']:.3f} "
            f"val={row['val_total']:+.3f} exec_h={row['exec_h']}"
        )
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
