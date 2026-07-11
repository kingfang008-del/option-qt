#!/usr/bin/env python3
"""Rule x State x Month stability analysis for QQQ 0DTE.

The goal is to stop selecting rules by one lucky month.  Each alpha rule is
scored by how consistently it works inside a state across months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_state_alpha_attribution import (
    TIME_STATE_COLS,
    add_pair_states,
    add_time_states,
    choose_daily_topk,
    replay_metrics,
)
from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset
from factor_lab.tools.run_0dte_factor_score_loop import (
    SCORE_COLS,
    STATE_COLS,
    apply_ic_score,
    build_score_dataset,
    fit_ic_weights,
    fit_tree,
    predict_tree,
    spearman_ic,
)
from factor_lab.tools.run_0dte_minimal_five_layer_loop import load_stock_state_features


RULE_SCORES = ["ic_edge_score", "tree_edge_score", "hot_score", *SCORE_COLS]


def month_starts(start: str, end: str) -> list[tuple[str, str, str]]:
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    out = []
    for p in periods:
        s = max(pd.Timestamp(start), p.start_time).strftime("%Y-%m-%d")
        e = min(pd.Timestamp(end), p.end_time).strftime("%Y-%m-%d")
        out.append((str(p), s, e))
    return out


def add_regime_proxy_states(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["is_high_vol_proxy"] = (
        pd.to_numeric(out["state_stock_abs_mom_q"], errors="coerce").fillna(0) >= 0.70
    ).astype(float)
    out["is_low_vol_proxy"] = (
        (pd.to_numeric(out["state_stock_abs_mom_q"], errors="coerce").fillna(1) <= 0.35)
        & (pd.to_numeric(out["state_spread_q"], errors="coerce").fillna(1) <= 0.50)
    ).astype(float)
    out["is_positive_gamma_proxy"] = (
        (out["is_range_pin_proxy"].fillna(0) > 0.5)
        | ((out["is_low_vol_proxy"].fillna(0) > 0.5) & (out["liquidity_score"].fillna(0) >= 0.55))
    ).astype(float)
    out["is_negative_gamma_proxy"] = (
        (out["is_vol_expansion"].fillna(0) > 0.5)
        & (pd.to_numeric(out["state_spread_q"], errors="coerce").fillna(0) >= 0.60)
    ).astype(float)
    out["is_qqq_recovering"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out["stock_ret_60s"], errors="coerce").fillna(0) < 0)
    ).astype(float)
    # Symbol-agnostic aliases (same logic; keep qqq_* for backward compatibility).
    out["is_underlying_recovering"] = out["is_qqq_recovering"]
    out["is_qqq_breaking_down"] = (
        (pd.to_numeric(out["stock_ret_30s"], errors="coerce").fillna(0) < 0)
        & (pd.to_numeric(out["stock_ret_60s"], errors="coerce").fillna(0) < 0)
        & (pd.to_numeric(out["stock_vwap_dev"], errors="coerce").fillna(0) < 0)
    ).astype(float)
    out["is_underlying_breaking_down"] = out["is_qqq_breaking_down"]
    put_flow = out["side"].eq("PUT") & (pd.to_numeric(out["flow_score"], errors="coerce").fillna(0) >= 0.70)
    put_flow_by_ts = put_flow.groupby(out["timestamp"]).transform("max")
    out["is_put_flow_continuation"] = put_flow_by_ts.astype(float)
    out["is_put_flow_exhaustion"] = (
        (put_flow_by_ts.astype(float) > 0.5)
        & (pd.to_numeric(out["stock_ret_30s"], errors="coerce").fillna(0) > 0)
    ).astype(float)
    states = [
        "is_high_vol_proxy",
        "is_low_vol_proxy",
        "is_positive_gamma_proxy",
        "is_negative_gamma_proxy",
        "is_qqq_recovering",
        "is_underlying_recovering",
        "is_qqq_breaking_down",
        "is_underlying_breaking_down",
        "is_put_flow_continuation",
        "is_put_flow_exhaustion",
    ]
    return out, states


def load_or_build_month(
    args: argparse.Namespace,
    month: str,
    start: str,
    end: str,
    target: str,
    thresholds: dict,
    cache_dir: Path,
) -> pd.DataFrame:
    fp = cache_dir / f"score_dataset_{month}.parquet"
    if fp.exists() and not args.refresh_cache:
        return pd.read_parquet(fp)
    raw = load_factor_dataset(
        Path(args.micro_root),
        start,
        end,
        (args.horizon_s,),
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
        symbol=getattr(args, "symbol", "QQQ"),
    )
    stock = load_stock_state_features(
        Path(args.stock_root),
        start,
        end,
        symbol=getattr(args, "symbol", "QQQ"),
    )
    data, _ = build_score_dataset(raw, stock, target, thresholds)
    data["month"] = month
    data.to_parquet(fp, index=False)
    return data


def load_fit_period(args: argparse.Namespace, target: str):
    raw = load_factor_dataset(
        Path(args.micro_root),
        args.fit_start,
        args.fit_end,
        (args.horizon_s,),
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
        symbol=getattr(args, "symbol", "QQQ"),
    )
    stock = load_stock_state_features(
        Path(args.stock_root),
        args.fit_start,
        args.fit_end,
        symbol=getattr(args, "symbol", "QQQ"),
    )
    return build_score_dataset(raw, stock, target, None)


def fit_rule_scorers(fit_data: pd.DataFrame, target: str):
    weights = fit_ic_weights(fit_data, target)
    model = fit_tree(fit_data, target)
    scored = apply_rule_scorers(fit_data, weights, model)
    return scored, weights, model


def apply_rule_scorers(df: pd.DataFrame, weights: dict[str, float], model) -> pd.DataFrame:
    out = df.copy()
    out["ic_edge_score"] = apply_ic_score(out, weights)
    out["tree_edge_score"] = predict_tree(model, out)
    out["hot_score"] = out["score_hot_quote_tight"]
    return out


def attach_all_states(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = add_time_states(df)
    out, regime_states = add_regime_proxy_states(out)
    base_states = [*STATE_COLS, *TIME_STATE_COLS, *regime_states]
    out, pair_states = add_pair_states(out, base_states)
    selected_pairs = [
        c
        for c in pair_states
        if any(key in c for key in ["is_stock_trend_down", "is_call_trend_proxy", "is_vol_expansion"])
    ]
    return out, [*base_states, *selected_pairs]


def monthly_rule_rows(df: pd.DataFrame, target: str, states: list[str], cooldown_s: int, min_rows: int) -> pd.DataFrame:
    rows = []
    all_states = ["ALL", *states]
    for month, month_df in df.groupby("month", sort=True):
        for score in RULE_SCORES:
            if score not in month_df.columns:
                continue
            for state in all_states:
                state_df = month_df if state == "ALL" else month_df[pd.to_numeric(month_df[state], errors="coerce").fillna(0) > 0.5]
                if len(state_df) < min_rows:
                    continue
                for side in ["ALL", "CALL", "PUT"]:
                    side_df = state_df if side == "ALL" else state_df[state_df["side"].eq(side)]
                    if len(side_df) < min_rows:
                        continue
                    picks = choose_daily_topk(side_df, score, max_topk=5, cooldown_s=cooldown_s)
                    base = {
                        "month": month,
                        "rule": score,
                        "state": state,
                        "side": side,
                        "rows": int(len(side_df)),
                        "conditional_ic": spearman_ic(side_df[score], side_df[target]),
                        "base_mean": float(side_df[target].mean()),
                    }
                    for topk in (1, 2, 3, 5):
                        rows.append({**base, **replay_metrics(picks, target, topk)})
    return pd.DataFrame(rows)


def stability_score(monthly: pd.DataFrame, min_months: int) -> pd.DataFrame:
    keys = ["rule", "state", "side", "topk_per_day"]
    rows = []
    for key, g in monthly.groupby(keys, dropna=False):
        if len(g) < min_months:
            continue
        avg = pd.to_numeric(g["avg_return"], errors="coerce").fillna(0)
        pf = pd.to_numeric(g["profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(5).clip(0, 5)
        hit = pd.to_numeric(g["hit_rate"], errors="coerce").fillna(0)
        trades = pd.to_numeric(g["trades"], errors="coerce").fillna(0)
        dd = pd.to_numeric(g["max_drawdown"], errors="coerce").fillna(-1).abs()
        ic = pd.to_numeric(g["conditional_ic"], errors="coerce").fillna(0)
        positive_month_ratio = float((avg > 0).mean())
        state_consistency = float(max(0.0, ic.mean()) / (ic.std(ddof=0) + 0.02))
        sample_size_penalty = float(min(1.0, trades.mean() / 40.0))
        drawdown_penalty = float(1.0 / (1.0 + 8.0 * dd.mean()))
        rule_score = float(
            max(0.0, avg.mean())
            * max(0.0, hit.mean())
            * positive_month_ratio
            * min(2.0, state_consistency)
            * sample_size_penalty
            * drawdown_penalty
        )
        rows.append(
            {
                "rule": key[0],
                "state": key[1],
                "side": key[2],
                "topk_per_day": key[3],
                "months": int(len(g)),
                "positive_month_ratio": positive_month_ratio,
                "mean_return": float(avg.mean()),
                "median_return": float(avg.median()),
                "mean_hit_rate": float(hit.mean()),
                "mean_profit_factor": float(pf.mean()),
                "mean_trades": float(trades.mean()),
                "mean_drawdown": float(-dd.mean()),
                "mean_conditional_ic": float(ic.mean()),
                "state_consistency": state_consistency,
                "sample_size_penalty": sample_size_penalty,
                "drawdown_penalty": drawdown_penalty,
                "rule_score": rule_score,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rule_score", "positive_month_ratio", "mean_return"], ascending=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--start", default="2026-04-13")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--min-months", type=int, default=2)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_rule_state_stability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[rule-stability] fitting thresholds and scores", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    fit_data, weights, model = fit_rule_scorers(fit_data, target)
    months = []
    for month, start, end in month_starts(args.start, args.end):
        print(f"[rule-stability] loading {month}", flush=True)
        data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        data = apply_rule_scorers(data, weights, model)
        months.append(data)
    panel = pd.concat(months, ignore_index=True)
    panel, states = attach_all_states(panel)
    print(f"[rule-stability] rows={len(panel)} months={panel['month'].nunique()} states={len(states)}", flush=True)
    monthly = monthly_rule_rows(panel, target, states, args.cooldown_s, args.min_rows)
    stable = stability_score(monthly, args.min_months)
    monthly.to_csv(out_dir / "rule_state_month_matrix.csv", index=False)
    stable.to_csv(out_dir / "rule_stability_score.csv", index=False)
    summary = {
        "config": vars(args),
        "rows": int(len(panel)),
        "months": sorted(panel["month"].unique().tolist()),
        "states": states,
        "ic_weights": weights,
        "matrix_rows": int(len(monthly)),
        "stable_rules": int(len(stable)),
        "best_stable_rules": stable.head(40).to_dict("records"),
        "files": {
            "rule_state_month_matrix": str(out_dir / "rule_state_month_matrix.csv"),
            "rule_stability_score": str(out_dir / "rule_stability_score.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
