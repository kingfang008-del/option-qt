#!/usr/bin/env python3
"""Low-memory causal expanding-window recalibration for QQQ 0DTE.

Previous bar-level Rule×State matrix on concatenated months blew past RAM.
This rewrite:

1. Never concatenates multi-month bar panels
2. Fits thresholds/scorers on sampled train bars only
3. Selects rules from month-by-month *trade-level* path PnL
4. Keeps only slim columns after scoring
5. Loads one month at a time

Forward walk-forward only: for test month m, fit on months < m.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
)
from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk
from factor_lab.tools.run_0dte_adaptive_rule_pool import (
    adaptive_daily_trades,
    idle_ratio,
    rule_active_mask,
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
)


DEFAULT_HOLD_S = 45
HOLD_CANDIDATES = (45, 180)

# Focused candidate set — enough to rediscover recovering/lunch and nearby states.
CANDIDATE_STATES = [
    "is_qqq_recovering",
    "is_stock_trend_down__and__is_lunch",
    "is_stock_trend_down",
    "is_stock_trend_up",
    "is_vol_expansion",
    "is_put_trend_proxy",
    "is_call_trend_proxy",
    "is_opening",
    "is_lunch",
    "is_power_hour",
    "is_qqq_breaking_down",
    "is_high_vol_proxy",
    "is_low_vol_proxy",
    "is_negative_gamma_proxy",
    "is_positive_gamma_proxy",
]

# Columns required to rebuild thresholds / factor scores / states / path PnL.
KEEP_RAW = [
    "timestamp",
    "date_str",
    "month",
    "ticker",
    "side",
    "side_code",
    "bid",
    "ask",
    "spread_pct",
    "quote_imbalance",
    "rolling_trades",
    "rolling_notional",
    "universe_rank",
    "tod_frac",
    "mid_ret_past_10s",
    "trade_notional_sum_60s",
    "quote_events_sum_10s",
    "flow_imbalance_5s",
    "net_buy_sum_5s",
    "flow_toxicity_10s",
    "stock_ret_10s",
    "stock_ret_30s",
    "stock_ret_60s",
    "stock_abs_ret_30s",
    "stock_abs_ret_60s",
    "stock_vwap_dev",
    "stock_rv_60s",
    "stock_volume_z_60s",
    "panel_notional_60s",
    "panel_quote_10s",
    "panel_spread",
    "panel_abs_mom_10s",
    "state_put_minus_call_mom",
    "state_call_minus_put_mom",
    "score_hot_quote_tight",
]

SLIM_AFTER_SCORE = [
    "timestamp",
    "date_str",
    "month",
    "ticker",
    "side",
    "bid",
    "ask",
    "spread_pct",
    "tree_edge_score",
    "ic_edge_score",
    "flow_score",
    "liquidity_score",
    "vol_score",
    "trend_score",
    "is_put_trend_proxy",
    *CANDIDATE_STATES,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="factor_lab/results/0dte_state_gate_h1_cache")
    p.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06")
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--pool-size", type=int, default=6)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--min-train-months", type=int, default=1)
    p.add_argument("--fit-sample-per-month", type=int, default=120_000)
    p.add_argument("--fit-max-rows", type=int, default=350_000)
    p.add_argument("--min-month-trades", type=int, default=8)
    p.add_argument("--enable-confirm", action="store_true")
    p.add_argument("--fit-state-hold", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_causal_rolling_recal_lean_h1")
    return p.parse_args()


def month_list(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def load_month_raw(cache_dir: Path, month: str, target: str) -> pd.DataFrame:
    fp = cache_dir / f"score_dataset_{month}.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    cols = [c for c in KEEP_RAW + [target] if True]
    # Read only existing columns.
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(fp).schema.names)
    use = [c for c in cols if c in available]
    if target not in use:
        raise KeyError(f"{fp} missing {target}")
    data = pd.read_parquet(fp, columns=use)
    data["month"] = month
    if "date_str" not in data.columns:
        data["date_str"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d")
    return data


def score_month(
    raw: pd.DataFrame,
    *,
    thresholds: dict,
    weights: dict[str, float],
    model,
    target: str,
) -> pd.DataFrame:
    work = apply_market_state_thresholds(raw, thresholds)
    work = add_factor_scores(work)
    keep = SCORE_COLS + STATE_COLS + ["side_code", target, "timestamp", "date_str", "side", "ticker"]
    clean = work.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
    del work
    clean = apply_rule_scorers(clean, weights, model)
    clean, _ = attach_all_states(clean)
    slim_cols = [c for c in dict.fromkeys(SLIM_AFTER_SCORE) if c in clean.columns]
    # Always keep bid/ask for path exec.
    for c in ("bid", "ask", "tree_edge_score"):
        if c not in slim_cols and c in clean.columns:
            slim_cols.append(c)
    out = clean.loc[:, slim_cols].copy()
    # Guard against accidental duplicate labels.
    out = out.loc[:, ~out.columns.duplicated()].copy()
    del clean
    return out


def path_exec_light(
    panel: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    hold_s: int,
    commission: float,
    state_hold: dict[str, int] | None = None,
    use_state_hold: bool = False,
    by_ticker: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Ask→bid path return; only indexes tickers present in trades."""
    if trades.empty:
        return trades
    if by_ticker is None:
        needed = set(trades["ticker"].astype(str))
        sub = panel[panel["ticker"].astype(str).isin(needed)][
            ["timestamp", "ticker", "bid", "ask"]
        ].copy()
        by_ticker = {
            t: g.sort_values("timestamp").reset_index(drop=True)
            for t, g in sub.groupby("ticker", sort=False)
        }
    rows = []
    for tr in trades.itertuples(index=False):
        rec = tr._asdict()
        path = by_ticker.get(str(getattr(tr, "ticker")))
        if path is None or path.empty:
            continue
        ts = pd.Timestamp(getattr(tr, "timestamp"))
        ts_ns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
        pos = int(np.searchsorted(ts_ns, ts.value, side="left"))
        if pos >= len(path):
            continue
        if abs(int(ts_ns[pos]) - ts.value) > 1_500_000_000:
            exact = np.where(ts_ns == ts.value)[0]
            if len(exact) == 0:
                continue
            pos = int(exact[0])
        trade_hold = int(hold_s)
        if use_state_hold and state_hold:
            state = str(getattr(tr, "active_state", "") or "")
            if state in state_hold:
                trade_hold = int(state_hold[state])
        exit_pos = pos + trade_hold
        if exit_pos >= len(path):
            continue
        entry_ask = float(pd.to_numeric(path["ask"].iloc[pos], errors="coerce"))
        exit_bid = float(pd.to_numeric(path["bid"].iloc[exit_pos], errors="coerce"))
        if not np.isfinite(entry_ask) or not np.isfinite(exit_bid) or entry_ask <= 0:
            continue
        cost = 2.0 * commission / (entry_ask * 100.0)
        rec["hold_s"] = trade_hold
        rec["entry_ask"] = entry_ask
        rec["exit_bid"] = exit_bid
        rec["path_exec_ret"] = float(exit_bid / entry_ask - 1.0 - cost)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_ticker_paths(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sub = panel[["timestamp", "ticker", "bid", "ask"]].copy()
    return {
        t: g.sort_values("timestamp").reset_index(drop=True)
        for t, g in sub.groupby("ticker", sort=False)
    }


def candidate_specs() -> list[dict]:
    specs = []
    for state in CANDIDATE_STATES:
        for side in ("ALL", "CALL", "PUT"):
            for topk in (1, 2, 3):
                specs.append(
                    {
                        "rule": "tree_edge_score",
                        "state": state,
                        "side": side,
                        "topk_per_day": topk,
                    }
                )
    return specs


def eval_rule_on_month(
    panel: pd.DataFrame,
    spec: dict,
    *,
    cooldown_s: int,
    commission: float,
    hold_s: int,
    by_ticker: dict[str, pd.DataFrame] | None = None,
) -> dict | None:
    state = spec["state"]
    if state not in panel.columns:
        return None
    sub = panel.loc[rule_active_mask(panel, state, spec["side"])]
    if len(sub) < 50:
        return None
    picks = choose_daily_topk(
        sub, spec["rule"], max_topk=int(spec["topk_per_day"]), cooldown_s=cooldown_s
    )
    if picks.empty:
        return None
    path = path_exec_light(
        panel,
        picks,
        hold_s=hold_s,
        commission=commission,
        by_ticker=by_ticker,
    )
    if path.empty:
        return None
    r = pd.to_numeric(path["path_exec_ret"], errors="coerce").dropna()
    if r.empty:
        return None
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    return {
        "month": str(panel["month"].iloc[0]),
        "rule": spec["rule"],
        "state": spec["state"],
        "side": spec["side"],
        "topk_per_day": int(spec["topk_per_day"]),
        "trades": int(len(r)),
        "days": int(path["date_str"].nunique()),
        "avg_return": float(r.mean()),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "sum_return": float(r.sum()),
    }


def select_pool_from_monthly(
    monthly: pd.DataFrame,
    *,
    min_months: int,
    top_n: int,
    strict: bool,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    keys = ["rule", "state", "side", "topk_per_day"]
    rows = []
    for key, g in monthly.groupby(keys, dropna=False):
        if len(g) < min_months:
            continue
        avg = pd.to_numeric(g["avg_return"], errors="coerce")
        pf = pd.to_numeric(g["profit_factor"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(5.0).clip(0, 5)
        hit = pd.to_numeric(g["hit_rate"], errors="coerce")
        trades = pd.to_numeric(g["trades"], errors="coerce")
        pos_ratio = float((avg > 0).mean())
        if strict and pos_ratio < 1.0:
            continue
        if (not strict) and pos_ratio < 0.5:
            continue
        if float(avg.mean()) <= 0 or float(pf.mean()) <= 1.05:
            continue
        if float(trades.mean()) < 8:
            continue
        score = float(
            max(0.0, avg.mean())
            * max(0.0, hit.mean())
            * pos_ratio
            * min(1.0, trades.mean() / 20.0)
        )
        rows.append(
            {
                "rule": key[0],
                "state": key[1],
                "side": key[2],
                "topk_per_day": int(key[3]),
                "months": int(len(g)),
                "positive_month_ratio": pos_ratio,
                "mean_return": float(avg.mean()),
                "mean_hit_rate": float(hit.mean()),
                "mean_profit_factor": float(pf.mean()),
                "mean_trades": float(trades.mean()),
                "rule_score": score,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["positive_month_ratio", "rule_score", "mean_return"], ascending=False
    ).head(top_n)


def fit_hold_map_from_pool(
    train_months: list[str],
    cache_dir: Path,
    *,
    thresholds: dict,
    weights: dict[str, float],
    model,
    target: str,
    pool: pd.DataFrame,
    cooldown_s: int,
    daily_topk: int,
    commission: float,
) -> dict[str, int]:
    if pool.empty:
        return {}
    # Aggregate train path returns by state under each hold.
    by_state: dict[str, dict[int, list[float]]] = {}
    for month in train_months:
        raw = load_month_raw(cache_dir, month, target)
        panel = score_month(
            raw, thresholds=thresholds, weights=weights, model=model, target=target
        )
        del raw
        trades = adaptive_daily_trades(
            panel, pool, "tree_edge_score", cooldown_s, daily_topk
        )
        if trades.empty:
            del panel
            continue
        for hold in HOLD_CANDIDATES:
            path = path_exec_light(panel, trades, hold_s=hold, commission=commission)
            if path.empty or "active_state" not in path.columns:
                continue
            for state, g in path.groupby("active_state"):
                by_state.setdefault(str(state), {}).setdefault(hold, [])
                by_state[str(state)][hold].extend(
                    pd.to_numeric(g["path_exec_ret"], errors="coerce").dropna().tolist()
                )
        del panel
        gc.collect()
    hold_map: dict[str, int] = {}
    for state, holds in by_state.items():
        best_h, best_avg = DEFAULT_HOLD_S, -np.inf
        for h, vals in holds.items():
            if len(vals) < 5:
                continue
            avg = float(np.mean(vals))
            if avg > best_avg:
                best_avg = avg
                best_h = int(h)
        hold_map[state] = best_h
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
    trades = path_exec_light(
        panel,
        trades,
        hold_s=DEFAULT_HOLD_S,
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


def build_fit_sample(
    cache_dir: Path,
    train_months: list[str],
    target: str,
    *,
    per_month: int,
    max_rows: int,
) -> pd.DataFrame:
    parts = []
    rng = np.random.default_rng(42)
    for month in train_months:
        raw = load_month_raw(cache_dir, month, target)
        n = min(len(raw), per_month)
        if n <= 0:
            continue
        idx = rng.choice(len(raw), size=n, replace=False)
        parts.append(raw.iloc[idx].copy())
        del raw
        gc.collect()
    if not parts:
        raise RuntimeError("empty fit sample")
    sample = pd.concat(parts, ignore_index=True)
    del parts
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=42)
    return sample.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    months = month_list(args.months)
    specs = candidate_specs()

    folds: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    pool_frames: list[pd.DataFrame] = []

    for i, test_month in enumerate(months):
        train_months = months[:i]
        if len(train_months) < args.min_train_months:
            print(f"[lean] skip {test_month}: need prior months", flush=True)
            continue

        print(f"[lean] fold test={test_month} train={train_months}", flush=True)
        fit_raw = build_fit_sample(
            cache_dir,
            train_months,
            target,
            per_month=args.fit_sample_per_month,
            max_rows=args.fit_max_rows,
        )
        thresholds = fit_state_thresholds(fit_raw)
        fit_work = apply_market_state_thresholds(fit_raw, thresholds)
        del fit_raw
        fit_work = add_factor_scores(fit_work)
        keep = SCORE_COLS + STATE_COLS + ["side_code", target]
        fit_work = fit_work.replace([np.inf, -np.inf], np.nan).dropna(subset=keep).copy()
        _, weights, model = fit_rule_scorers(fit_work, target)
        del fit_work
        gc.collect()
        print(f"[lean] {test_month}: scorers fit", flush=True)

        # Month-by-month trade-level rule matrix on train only.
        monthly_rows: list[dict] = []
        for month in train_months:
            raw = load_month_raw(cache_dir, month, target)
            panel = score_month(
                raw, thresholds=thresholds, weights=weights, model=model, target=target
            )
            del raw
            by_ticker = build_ticker_paths(panel)
            for spec in specs:
                row = eval_rule_on_month(
                    panel,
                    spec,
                    cooldown_s=args.cooldown_s,
                    commission=args.commission_per_contract,
                    hold_s=DEFAULT_HOLD_S,
                    by_ticker=by_ticker,
                )
                if row is not None and row["trades"] >= args.min_month_trades:
                    monthly_rows.append(row)
            del panel, by_ticker
            gc.collect()
            print(f"[lean] scored train month {month}", flush=True)

        monthly = pd.DataFrame(monthly_rows)
        min_months = 1 if len(train_months) == 1 else 2
        pool_strict = select_pool_from_monthly(
            monthly, min_months=min_months, top_n=args.pool_size, strict=True
        )
        pool_loose = select_pool_from_monthly(
            monthly, min_months=min_months, top_n=args.pool_size, strict=False
        )
        print(
            f"[lean] {test_month}: strict={len(pool_strict)} loose={len(pool_loose)}",
            flush=True,
        )

        confirm_thresholds: dict[str, float] = {}
        hold_map: dict[str, int] = {}
        active_pool = pool_strict if not pool_strict.empty else pool_loose

        if args.enable_confirm and not active_pool.empty:
            # Fit confirm on last train month only to limit memory.
            last = train_months[-1]
            raw = load_month_raw(cache_dir, last, target)
            panel = score_month(
                raw, thresholds=thresholds, weights=weights, model=model, target=target
            )
            del raw
            fit_trades = adaptive_daily_trades(
                panel, active_pool, "tree_edge_score", args.cooldown_s, args.daily_topk
            )
            active_states = set(active_pool["state"].astype(str))
            specs_confirm = {
                s: spec for s, spec in CONFIRM_SPECS.items() if s in active_states
            }
            if specs_confirm and not fit_trades.empty:
                confirm_thresholds = fit_confirm_thresholds(
                    fit_trades, specs=specs_confirm
                )
            del panel
            gc.collect()

        if args.fit_state_hold and not active_pool.empty:
            hold_map = fit_hold_map_from_pool(
                train_months,
                cache_dir,
                thresholds=thresholds,
                weights=weights,
                model=model,
                target=target,
                pool=active_pool,
                cooldown_s=args.cooldown_s,
                daily_topk=args.daily_topk,
                commission=args.commission_per_contract,
            )

        # Test month replay.
        raw = load_month_raw(cache_dir, test_month, target)
        test_panel = score_month(
            raw, thresholds=thresholds, weights=weights, model=model, target=target
        )
        del raw
        gc.collect()

        policies = {}
        fold_trades = []
        for name, pool, use_confirm, use_hold in [
            ("strict", pool_strict, args.enable_confirm, bool(hold_map)),
            ("loose", pool_loose, args.enable_confirm, bool(hold_map)),
            ("frozen_curated", CURATED_RULES, False, False),
        ]:
            metrics, trades = evaluate_policy(
                test_panel,
                pool,
                cooldown_s=args.cooldown_s,
                daily_topk=args.daily_topk,
                commission=args.commission_per_contract,
                confirm_thresholds=confirm_thresholds if use_confirm else {},
                enable_confirm=use_confirm,
                state_hold=hold_map if use_hold else {},
                use_state_hold=use_hold,
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
                f"[lean] {test_month} {name}: trades={metrics.get('trades', 0)} "
                f"avg={metrics.get('avg_return', float('nan')):.4f} "
                f"acct={metrics.get('total_return_position', float('nan')):.3f}",
                flush=True,
            )

        del test_panel
        gc.collect()

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
                "pool_strict_n": int(len(pool_strict)),
                "pool_loose_n": int(len(pool_loose)),
                "pool_strict": pool_strict.to_dict("records") if not pool_strict.empty else [],
                "pool_loose": pool_loose.to_dict("records") if not pool_loose.empty else [],
                "confirm_thresholds": confirm_thresholds,
                "state_hold_s": hold_map,
                "ic_weights": weights,
                "policies": policies,
            }
        )
        if not monthly.empty:
            monthly.to_csv(out_dir / f"train_rule_month_{test_month}.csv", index=False)

    if not folds:
        raise RuntimeError("no folds completed")

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
            "low-memory causal expanding-window recalibration; "
            "sampled scorer fit + month-streamed trade-level rule selection"
        ),
        "config": vars(args),
        "candidate_states": CANDIDATE_STATES,
        "folds": folds,
        "combined": combined,
        "limitations": [
            "Scorers are fit on sampled bars, not the full train panel.",
            "Rule search uses a focused state set and tree_edge_score only.",
            "Train rule metrics use fixed 45s path returns; state-hold is optional overlay.",
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
