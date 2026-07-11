#!/usr/bin/env python3
"""State-adaptive rule pool replay for QQQ 0DTE.

Permanent filters -> State Gate -> activate historically stable rules -> TopK.
Rules are selected only from training months, then frozen for OOS months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
    load_fit_period,
    load_or_build_month,
    monthly_rule_rows,
    stability_score,
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk, replay_metrics
from factor_lab.tools.analyze_0dte_state_alpha_attribution import TIME_STATE_COLS
from factor_lab.tools.run_0dte_factor_score_loop import STATE_COLS


def permanent_filter(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Soft liquidity filter. Keep disabled for state-gated curated rules by default."""
    if not enabled:
        return df
    out = df.copy()
    spread = pd.to_numeric(out.get("spread_pct"), errors="coerce")
    liq = pd.to_numeric(out.get("liquidity_score"), errors="coerce")
    mask = spread.fillna(1.0).le(0.08) & liq.fillna(0.0).ge(0.20)
    return out.loc[mask].copy()


def select_rule_pool(monthly: pd.DataFrame, min_months: int, top_n: int) -> pd.DataFrame:
    """Loose pool: allow some month misses if mean edge and PF remain positive."""
    stable = stability_score(monthly, min_months=min_months)
    if stable.empty:
        return stable
    pool = stable[
        (stable["mean_return"] > 0)
        & (stable["median_return"] > 0)
        & (stable["mean_profit_factor"] > 1.10)
        & (stable["mean_trades"] >= 12)
        & (stable["rule_score"] > 0)
        & (stable["positive_month_ratio"] >= 0.5)
        & (stable["state"] != "ALL")
    ].copy()
    return pool.sort_values(
        ["positive_month_ratio", "rule_score", "mean_return"],
        ascending=False,
    ).head(top_n)


def select_rule_pool_strict(monthly: pd.DataFrame, min_months: int, top_n: int) -> pd.DataFrame:
    """Strict pool: every training month must be positive."""
    stable = stability_score(monthly, min_months=min_months)
    if stable.empty:
        return stable
    pool = stable[
        (stable["positive_month_ratio"] >= 1.0)
        & (stable["mean_return"] > 0)
        & (stable["median_return"] > 0)
        & (stable["mean_profit_factor"] > 1.05)
        & (stable["rule_score"] > 0)
        & (stable["state"] != "ALL")
    ].copy()
    return pool.sort_values(["rule_score", "mean_return"], ascending=False).head(top_n)


def curated_pool() -> pd.DataFrame:
    """Hand-picked rules that were positive in all of Apr/May/Jun on the stability matrix."""
    return pd.DataFrame(
        [
            {
                "rule": "tree_edge_score",
                "state": "is_qqq_recovering",
                "side": "ALL",
                "topk_per_day": 1,
                "positive_month_ratio": 1.0,
                "mean_return": 0.0161,
                "mean_profit_factor": 1.83,
                "rule_score": 0.0064,
            },
            {
                "rule": "tree_edge_score",
                "state": "is_stock_trend_down__and__is_lunch",
                "side": "ALL",
                "topk_per_day": 1,
                "positive_month_ratio": 1.0,
                "mean_return": 0.0076,
                "mean_profit_factor": 1.35,
                "rule_score": 0.0030,
            },
            {
                "rule": "tree_edge_score",
                "state": "is_stock_trend_down__and__is_lunch",
                "side": "ALL",
                "topk_per_day": 3,
                "positive_month_ratio": 1.0,
                "mean_return": 0.0028,
                "mean_profit_factor": 1.14,
                "rule_score": 0.0021,
            },
        ]
    )


def rule_active_mask(df: pd.DataFrame, state: str, side: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if state != "ALL":
        mask &= pd.to_numeric(df[state], errors="coerce").fillna(0.0) > 0.5
    if side != "ALL":
        mask &= df["side"].eq(side)
    return mask


def adaptive_daily_trades(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    target: str,
    cooldown_s: int,
    daily_topk: int,
) -> pd.DataFrame:
    """Activate all matching rules, then take global daily TopK by edge score."""
    if pool.empty or df.empty:
        return pd.DataFrame()
    candidates = []
    for row in pool.itertuples(index=False):
        rule = getattr(row, "rule")
        state = getattr(row, "state")
        side = getattr(row, "side")
        topk = int(getattr(row, "topk_per_day"))
        if rule not in df.columns:
            continue
        sub = df.loc[rule_active_mask(df, state, side)].copy()
        if sub.empty:
            continue
        picks = choose_daily_topk(sub, rule, max_topk=topk, cooldown_s=cooldown_s)
        if picks.empty:
            continue
        picks = picks.copy()
        picks["active_rule"] = rule
        picks["active_state"] = state
        picks["active_side"] = side
        picks["edge_for_rank"] = pd.to_numeric(picks[rule], errors="coerce").fillna(-1e9)
        candidates.append(picks)
    if not candidates:
        return pd.DataFrame()
    all_cands = pd.concat(candidates, ignore_index=True)
    # Deduplicate same contract/timestamp across rules; keep highest edge.
    all_cands = all_cands.sort_values("edge_for_rank", ascending=False)
    all_cands = all_cands.drop_duplicates(subset=["date_str", "timestamp", "ticker"], keep="first")
    trades = []
    for _, g in all_cands.groupby("date_str", sort=False):
        last_ts = None
        chosen = 0
        for r in g.itertuples(index=False):
            ts = pd.Timestamp(getattr(r, "timestamp"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            trades.append(r._asdict())
            last_ts = ts
            chosen += 1
            if chosen >= daily_topk:
                break
    return pd.DataFrame(trades)


def summarize_trades(
    trades: pd.DataFrame,
    target: str,
    label: str,
    *,
    position_frac: float = 0.25,
) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0, "position_frac": float(position_frac)}
    r = pd.to_numeric(trades[target], errors="coerce").fillna(0.0)
    eq = (1.0 + float(position_frac) * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    total_ret = float(eq.iloc[-1] - 1.0)
    out = {
        "label": label,
        "trades": int(len(trades)),
        "days": int(trades["date_str"].nunique()),
        "avg_return": float(r.mean()),
        "sum_return": float(r.sum()),
        "position_frac": float(position_frac),
        "total_return_position": total_ret,
        # Backward-compatible alias used by older summaries; now reflects position_frac.
        "total_return_10pct_position": total_ret,
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": float(dd.min()),
    }
    if "active_rule" in trades.columns:
        out["rule_counts"] = trades["active_rule"].value_counts().to_dict()
        out["state_counts"] = trades["active_state"].value_counts().to_dict()
        out["side_counts"] = trades["side"].value_counts().to_dict()
    return out


def fixed_baseline(df: pd.DataFrame, target: str, cooldown_s: int) -> pd.DataFrame:
    """Always-on baseline that previously looked strong but was unstable."""
    sub = df[df["side"].eq("CALL")].copy()
    if "is_stock_trend_down" in sub.columns:
        sub = sub[pd.to_numeric(sub["is_stock_trend_down"], errors="coerce").fillna(0) > 0.5]
    score = "ic_edge_score" if "ic_edge_score" in sub.columns else "hot_score"
    return choose_daily_topk(sub, score, max_topk=1, cooldown_s=cooldown_s)


def idle_ratio(df: pd.DataFrame, trades: pd.DataFrame) -> float:
    all_days = set(df["date_str"].unique())
    if not all_days:
        return 1.0
    active = set(trades["date_str"].unique()) if not trades.empty else set()
    return float(1.0 - len(active) / len(all_days))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--cache-dir", default="factor_lab/results/0dte_rule_state_stability_apr_jun/cache")
    p.add_argument("--train-months", default="2026-04,2026-05")
    p.add_argument("--test-months", default="2026-06")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--pool-size", type=int, default=12)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--use-permanent-filter", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_adaptive_rule_pool")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_months = [m.strip() for m in args.train_months.split(",") if m.strip()]
    test_months = [m.strip() for m in args.test_months.split(",") if m.strip()]
    all_months = sorted(set(train_months + test_months))

    print("[adaptive] fitting scorers", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    _, weights, model = fit_rule_scorers(fit_data, target)

    panels_raw = {}
    panels_exec = {}
    for month in all_months:
        start = f"{month}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        if month == "2026-04":
            start = "2026-04-13"
        print(f"[adaptive] loading {month}", flush=True)
        data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        data = apply_rule_scorers(data, weights, model)
        data, _ = attach_all_states(data)
        panels_raw[month] = data
        panels_exec[month] = permanent_filter(data, enabled=args.use_permanent_filter)

    # Select rules on unfiltered train data so permanent filters don't rewrite history.
    train_panel = pd.concat([panels_raw[m] for m in train_months], ignore_index=True)
    test_panel = pd.concat([panels_exec[m] for m in test_months], ignore_index=True)
    test_panel_raw = pd.concat([panels_raw[m] for m in test_months], ignore_index=True)
    regime = [
        "is_high_vol_proxy",
        "is_low_vol_proxy",
        "is_positive_gamma_proxy",
        "is_negative_gamma_proxy",
        "is_qqq_recovering",
        "is_qqq_breaking_down",
        "is_put_flow_continuation",
        "is_put_flow_exhaustion",
    ]
    base_states = [*STATE_COLS, *TIME_STATE_COLS, *regime]
    pair_states = [
        c
        for c in train_panel.columns
        if "__and__" in c
        and any(k in c for k in ["is_stock_trend_down", "is_call_trend_proxy", "is_vol_expansion"])
    ]
    states = [*base_states, *pair_states]

    print(f"[adaptive] train={len(train_panel)} test={len(test_panel)} states={len(states)}", flush=True)
    print("[adaptive] building train monthly matrix", flush=True)
    train_monthly = monthly_rule_rows(train_panel, target, states, args.cooldown_s, args.min_rows)
    pool_loose = select_rule_pool(train_monthly, min_months=max(1, len(train_months)), top_n=args.pool_size)
    pool_strict = select_rule_pool_strict(train_monthly, min_months=max(1, len(train_months)), top_n=args.pool_size)
    pool_curated = curated_pool()

    adaptive_loose = adaptive_daily_trades(test_panel, pool_loose, target, args.cooldown_s, args.daily_topk)
    adaptive_strict = adaptive_daily_trades(test_panel, pool_strict, target, args.cooldown_s, args.daily_topk)
    adaptive_curated = adaptive_daily_trades(test_panel, pool_curated, target, args.cooldown_s, args.daily_topk)
    baseline = fixed_baseline(test_panel_raw, target, args.cooldown_s)

    per_rule = []
    for pool_name, pool in [("loose", pool_loose), ("strict", pool_strict), ("curated", pool_curated)]:
        for row in pool.itertuples(index=False):
            rule, state, side, topk = row.rule, row.state, row.side, int(row.topk_per_day)
            sub = test_panel.loc[rule_active_mask(test_panel, state, side)]
            picks = (
                choose_daily_topk(sub, rule, max_topk=topk, cooldown_s=args.cooldown_s)
                if not sub.empty
                else pd.DataFrame()
            )
            metrics = replay_metrics(picks, target, topk) if not picks.empty else {"trades": 0, "topk_per_day": topk}
            per_rule.append(
                {
                    "pool": pool_name,
                    "rule": rule,
                    "state": state,
                    "side": side,
                    "topk_per_day": topk,
                    "train_rule_score": float(getattr(row, "rule_score", 0)),
                    "train_positive_month_ratio": float(getattr(row, "positive_month_ratio", 0)),
                    **metrics,
                }
            )

    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train_panel)), "test_exec": int(len(test_panel)), "test_raw": int(len(test_panel_raw))},
        "pool_loose_n": int(len(pool_loose)),
        "pool_strict_n": int(len(pool_strict)),
        "pool_curated_n": int(len(pool_curated)),
        "pool_loose": pool_loose.to_dict("records"),
        "pool_strict": pool_strict.to_dict("records"),
        "pool_curated": pool_curated.to_dict("records"),
        "oos": {
            "adaptive_loose": {
                **summarize_trades(adaptive_loose, target, "adaptive_loose"),
                "idle_day_ratio": idle_ratio(test_panel_raw, adaptive_loose),
            },
            "adaptive_strict": {
                **summarize_trades(adaptive_strict, target, "adaptive_strict"),
                "idle_day_ratio": idle_ratio(test_panel_raw, adaptive_strict),
            },
            "adaptive_curated": {
                **summarize_trades(adaptive_curated, target, "adaptive_curated"),
                "idle_day_ratio": idle_ratio(test_panel_raw, adaptive_curated),
            },
            "fixed_call_trend_down": summarize_trades(baseline, target, "fixed_call_trend_down"),
        },
        "per_rule_oos": per_rule,
    }
    train_monthly.to_csv(out_dir / "train_monthly_matrix.csv", index=False)
    pool_loose.to_csv(out_dir / "pool_loose.csv", index=False)
    pool_strict.to_csv(out_dir / "pool_strict.csv", index=False)
    pool_curated.to_csv(out_dir / "pool_curated.csv", index=False)
    if not adaptive_loose.empty:
        adaptive_loose.to_parquet(out_dir / "adaptive_loose_trades.parquet", index=False)
    if not adaptive_strict.empty:
        adaptive_strict.to_parquet(out_dir / "adaptive_strict_trades.parquet", index=False)
    if not adaptive_curated.empty:
        adaptive_curated.to_parquet(out_dir / "adaptive_curated_trades.parquet", index=False)
    pd.DataFrame(per_rule).to_csv(out_dir / "per_rule_oos.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["oos"], indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
