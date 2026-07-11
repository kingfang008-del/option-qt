#!/usr/bin/env python3
"""Causal expanding-window recalibration for QQQ 0DTE State Gate.

For each test month m:
  1. Fit state thresholds + IC/tree scorers on months < m only
  2. Rebuild train Rule×State×Month matrix under that scorer
  3. Select loose/strict rule pools from train only
  4. Optionally fit confirm thresholds and per-state hold clocks on train
  5. Replay month m with path-level ask→bid returns

This is forward walk-forward. It does NOT reuse Apr-selected curated rules.
Existing score_dataset caches are used only as bar-level feature stores; state
flags and scorers are refit causally each fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    RULE_SCORES,
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
    monthly_rule_rows,
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import TIME_STATE_COLS
from factor_lab.tools.run_0dte_adaptive_rule_pool import (
    adaptive_daily_trades,
    idle_ratio,
    select_rule_pool,
    select_rule_pool_strict,
    summarize_trades,
)
from factor_lab.tools.run_0dte_factor_score_loop import SCORE_COLS, STATE_COLS, add_factor_scores
from factor_lab.tools.run_0dte_minimal_five_layer_loop import (
    apply_market_state_thresholds,
    fit_state_thresholds,
)
from factor_lab.tools.run_0dte_state_gate_curated import (
    CONFIRM_SPECS,
    CURATED_RULES,
    filter_trades_by_confirm,
    fit_confirm_thresholds,
    path_exec_return,
)


DEFAULT_HOLD_S = 45
HOLD_CANDIDATES = (45, 180)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        default="factor_lab/results/0dte_state_gate_h1_cache",
        help="bar-level score_dataset_{month}.parquet store",
    )
    p.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06")
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--pool-size", type=int, default=8)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--min-train-months", type=int, default=1)
    p.add_argument("--enable-confirm", action="store_true")
    p.add_argument("--fit-state-hold", action="store_true", help="choose 45/180 per state on train")
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_causal_rolling_recal_h1",
    )
    return p.parse_args()


def month_list(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def load_cached_month(cache_dir: Path, month: str) -> pd.DataFrame:
    fp = cache_dir / f"score_dataset_{month}.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"missing {fp}; build with run_0dte_state_gate_curated.py first"
        )
    data = pd.read_parquet(fp)
    data["month"] = month
    if "date_str" not in data.columns:
        data["date_str"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d")
    return data


def refit_panel(
    raw: pd.DataFrame,
    *,
    thresholds: dict,
    weights: dict[str, float],
    model,
    target: str,
) -> pd.DataFrame:
    """Re-apply causal thresholds/scorers on a cached bar panel."""
    work = apply_market_state_thresholds(raw, thresholds)
    # Factor scores are within-day ranks; recompute for consistency after any
    # column repair, then drop incomplete state/score rows.
    work = add_factor_scores(work)
    keep = SCORE_COLS + STATE_COLS + ["side_code", target, "timestamp", "date_str", "side", "ticker"]
    missing = [c for c in keep if c not in work.columns]
    if missing:
        raise KeyError(f"panel missing required columns: {missing}")
    clean = work.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
    clean = apply_rule_scorers(clean, weights, model)
    clean, _ = attach_all_states(clean)
    return clean


def collect_states(df: pd.DataFrame) -> list[str]:
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
    base = [*STATE_COLS, *TIME_STATE_COLS, *regime]
    pairs = [
        c
        for c in df.columns
        if "__and__" in c
        and any(k in c for k in ["is_stock_trend_down", "is_call_trend_proxy", "is_vol_expansion"])
    ]
    return [*base, *pairs]


def fit_hold_map(
    train_panel: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    cooldown_s: int,
    daily_topk: int,
    commission: float,
) -> dict[str, int]:
    """Pick 45s vs 180s per active state using train path returns only."""
    if pool.empty:
        return {}
    raw = adaptive_daily_trades(
        train_panel, pool, "tree_edge_score", cooldown_s, daily_topk
    )
    if raw.empty or "active_state" not in raw.columns:
        return {}
    hold_map: dict[str, int] = {}
    for state, g in raw.groupby("active_state"):
        best_hold = DEFAULT_HOLD_S
        best_avg = -np.inf
        for hold in HOLD_CANDIDATES:
            path = path_exec_return(
                train_panel,
                g,
                hold_s=hold,
                commission=commission,
                use_state_hold=False,
            )
            if path.empty:
                continue
            avg = float(pd.to_numeric(path["path_exec_ret"], errors="coerce").mean())
            if np.isfinite(avg) and avg > best_avg:
                best_avg = avg
                best_hold = int(hold)
        hold_map[str(state)] = best_hold
    return hold_map


def evaluate_policy(
    panel: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    cooldown_s: int,
    daily_topk: int,
    commission: float,
    confirm_thresholds: dict[str, float],
    enable_confirm: bool,
    state_hold: dict[str, int],
    use_state_hold: bool,
    default_hold_s: int,
    position_frac: float,
    label: str,
) -> tuple[dict, pd.DataFrame]:
    if pool.empty:
        return {"label": label, "trades": 0, "position_frac": position_frac}, pd.DataFrame()
    trades = adaptive_daily_trades(
        panel, pool, "tree_edge_score", cooldown_s, daily_topk
    )
    trades = filter_trades_by_confirm(
        trades, thresholds=confirm_thresholds, enabled=enable_confirm
    )
    trades = path_exec_return(
        panel,
        trades,
        hold_s=default_hold_s,
        commission=commission,
        state_hold=state_hold,
        use_state_hold=use_state_hold,
    )
    metrics = summarize_trades(
        trades, "path_exec_ret", label, position_frac=position_frac
    )
    metrics["idle_day_ratio"] = idle_ratio(panel, trades)
    metrics["active_day_ratio"] = 1.0 - float(metrics["idle_day_ratio"])
    return metrics, trades


def pool_records(pool: pd.DataFrame) -> list[dict]:
    if pool.empty:
        return []
    cols = [
        c
        for c in [
            "rule",
            "state",
            "side",
            "topk_per_day",
            "rule_score",
            "positive_month_ratio",
            "mean_return",
            "mean_profit_factor",
            "months",
        ]
        if c in pool.columns
    ]
    return pool[cols].to_dict("records")


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    months = month_list(args.months)

    print("[rolling] loading cached months", flush=True)
    raw_months = {m: load_cached_month(cache_dir, m) for m in months}

    folds: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    pool_frames: list[pd.DataFrame] = []

    for i, test_month in enumerate(months):
        train_months = months[:i]
        if len(train_months) < args.min_train_months:
            print(f"[rolling] skip {test_month}: need {args.min_train_months} train months", flush=True)
            continue

        print(
            f"[rolling] fold test={test_month} train={train_months}",
            flush=True,
        )
        train_raw = pd.concat([raw_months[m] for m in train_months], ignore_index=True)
        thresholds = fit_state_thresholds(train_raw)
        # Fit scorers on thresholded train panel without leaking test month.
        train_for_fit = apply_market_state_thresholds(train_raw, thresholds)
        train_for_fit = add_factor_scores(train_for_fit)
        keep = SCORE_COLS + STATE_COLS + ["side_code", target]
        train_for_fit = (
            train_for_fit.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
        )
        _, weights, model = fit_rule_scorers(train_for_fit, target)

        train_panels = {
            m: refit_panel(
                raw_months[m],
                thresholds=thresholds,
                weights=weights,
                model=model,
                target=target,
            )
            for m in train_months
        }
        test_panel = refit_panel(
            raw_months[test_month],
            thresholds=thresholds,
            weights=weights,
            model=model,
            target=target,
        )
        train_panel = pd.concat(list(train_panels.values()), ignore_index=True)
        states = collect_states(train_panel)

        print(
            f"[rolling] {test_month}: train_rows={len(train_panel)} "
            f"test_rows={len(test_panel)} states={len(states)}",
            flush=True,
        )
        train_monthly = monthly_rule_rows(
            train_panel, target, states, args.cooldown_s, args.min_rows
        )
        min_months = max(1, min(len(train_months), 2 if len(train_months) >= 2 else 1))
        pool_loose = select_rule_pool(
            train_monthly, min_months=min_months, top_n=args.pool_size
        )
        pool_strict = select_rule_pool_strict(
            train_monthly, min_months=min_months, top_n=args.pool_size
        )

        confirm_thresholds: dict[str, float] = {}
        if args.enable_confirm:
            fit_pool = pool_strict if not pool_strict.empty else pool_loose
            if not fit_pool.empty:
                fit_trades = adaptive_daily_trades(
                    train_panel,
                    fit_pool,
                    "tree_edge_score",
                    args.cooldown_s,
                    args.daily_topk,
                )
                active_states = set(fit_pool["state"].astype(str))
                specs = {
                    state: spec
                    for state, spec in CONFIRM_SPECS.items()
                    if state in active_states
                }
                if specs and not fit_trades.empty:
                    confirm_thresholds = fit_confirm_thresholds(
                        fit_trades, specs=specs
                    )

        hold_map: dict[str, int] = {}
        if args.fit_state_hold:
            hold_source = pool_strict if not pool_strict.empty else pool_loose
            hold_map = fit_hold_map(
                train_panel,
                hold_source,
                cooldown_s=args.cooldown_s,
                daily_topk=args.daily_topk,
                commission=args.commission_per_contract,
            )

        policies = {}
        fold_trades = []
        policy_specs = [
            (
                "strict",
                pool_strict,
                args.enable_confirm,
                bool(hold_map),
                hold_map,
                confirm_thresholds,
            ),
            (
                "loose",
                pool_loose,
                args.enable_confirm,
                bool(hold_map),
                hold_map,
                confirm_thresholds,
            ),
            # Frozen curated: same rule identities every month, no confirm/hold overlay.
            ("frozen_curated", CURATED_RULES, False, False, {}, {}),
        ]
        for name, pool, use_confirm, use_hold, holds, confirms in policy_specs:
            metrics, trades = evaluate_policy(
                test_panel,
                pool,
                cooldown_s=args.cooldown_s,
                daily_topk=args.daily_topk,
                commission=args.commission_per_contract,
                confirm_thresholds=confirms,
                enable_confirm=use_confirm,
                state_hold=holds,
                use_state_hold=use_hold,
                default_hold_s=DEFAULT_HOLD_S,
                position_frac=args.position_frac,
                label=name,
            )
            policies[name] = metrics
            if not trades.empty:
                trades = trades.copy()
                trades["test_month"] = test_month
                trades["policy"] = name
                fold_trades.append(trades)
            print(
                f"[rolling] {test_month} {name}: "
                f"trades={metrics.get('trades', 0)} "
                f"avg={metrics.get('avg_return', float('nan')):.4f} "
                f"acct={metrics.get('total_return_position', float('nan')):.3f}",
                flush=True,
            )

        if fold_trades:
            trade_frames.extend(fold_trades)
        for pname, pool in [("strict", pool_strict), ("loose", pool_loose)]:
            if pool.empty:
                continue
            tmp = pool.copy()
            tmp["test_month"] = test_month
            tmp["pool"] = pname
            pool_frames.append(tmp)

        folds.append(
            {
                "test_month": test_month,
                "train_months": train_months,
                "train_rows": int(len(train_panel)),
                "test_rows": int(len(test_panel)),
                "test_days": int(test_panel["date_str"].nunique()),
                "pool_strict_n": int(len(pool_strict)),
                "pool_loose_n": int(len(pool_loose)),
                "pool_strict": pool_records(pool_strict),
                "pool_loose": pool_records(pool_loose),
                "confirm_thresholds": confirm_thresholds,
                "state_hold_s": hold_map,
                "ic_weights": weights,
                "policies": policies,
            }
        )

    if not folds:
        raise RuntimeError("no eligible expanding-window folds")

    all_trades = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )
    if not all_trades.empty:
        all_trades.to_parquet(out_dir / "walk_forward_trades.parquet", index=False)

    if pool_frames:
        pd.concat(pool_frames, ignore_index=True).to_csv(
            out_dir / "selected_pools_by_fold.csv", index=False
        )

    combined = {}
    for policy in ["strict", "loose", "frozen_curated"]:
        sub = (
            all_trades[all_trades["policy"].eq(policy)]
            if not all_trades.empty
            else pd.DataFrame()
        )
        combined[policy] = summarize_trades(
            sub, "path_exec_ret", f"all_{policy}", position_frac=args.position_frac
        )

    summary = {
        "experiment_type": (
            "causal expanding-window recalibration; scorers, state thresholds, "
            "and rule pools are fit only on months before each test month"
        ),
        "config": vars(args),
        "available_rule_scores": RULE_SCORES,
        "folds": folds,
        "combined": combined,
        "limitations": [
            "Cached bar panels were originally materialized by an earlier pipeline; "
            "this script refits thresholds/scorers/rules causally on top of those bars.",
            "Rule selection still uses target_exec_ret_30s inside train months; OOS PnL "
            "uses path ask→bid execution.",
            "Frozen curated comparison disables confirm/state-hold to isolate rule identity leakage.",
        ],
        "files": {
            "trades": str(out_dir / "walk_forward_trades.parquet"),
            "pools": str(out_dir / "selected_pools_by_fold.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"combined": combined, "n_folds": len(folds)}, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
