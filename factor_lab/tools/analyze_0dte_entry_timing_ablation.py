#!/usr/bin/env python3
"""Entry-timing ablation inside curated State Gate states.

Hypothesis: trade count is already tiny; the bottleneck is *when* we fire
inside an active state, not how often we No-Trade.

Current baseline picks daily nlargest(tree_edge_score) inside the state, which
can chase late/extended edge.  This script compares timing filters that keep
the same states/rules/confirm/exit, only changing entry eligibility.
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
)
from factor_lab.tools.run_0dte_adaptive_rule_pool import (
    adaptive_daily_trades,
    rule_active_mask,
    summarize_trades,
)
from factor_lab.tools.run_0dte_state_gate_curated import (
    CURATED_RULES,
    STATE_HOLD_S,
    filter_trades_by_confirm,
    path_exec_return,
)


FROZEN_CONFIRM = {"is_stock_trend_down__and__is_lunch::flow_score": 0.6389275887503352}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--panel-cache-dirs",
        default=(
            "factor_lab/results/0dte_state_gate_h1_cache,"
            "factor_lab/results/0dte_state_gate_jul_w1_cache"
        ),
    )
    p.add_argument("--months", default="2026-04,2026-05,2026-06,2026-07")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_entry_timing_ablation",
    )
    return p.parse_args()


def resolve_panel(cache_dirs: list[Path], month: str) -> Path | None:
    for d in cache_dirs:
        fp = d / f"score_dataset_{month}.parquet"
        if fp.exists():
            return fp
    return None


def attach_timing_features(df: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    """Add state-age and edge-slope features used by timing filters."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if "date_str" not in out.columns:
        out["date_str"] = out["timestamp"].dt.strftime("%Y-%m-%d")

    # Edge slope per ticker (5s / 10s lookback on sorted path).
    out["edge_slope_5s"] = np.nan
    out["edge_slope_10s"] = np.nan
    edge = pd.to_numeric(out.get("tree_edge_score"), errors="coerce")
    out["_edge"] = edge
    pieces = []
    for _, g in out.groupby(["date_str", "ticker"], sort=False):
        g = g.sort_values("timestamp")
        e = g["_edge"]
        # approximate by shift on contiguous rows (1s panel rows are dense per contract)
        g = g.copy()
        g["edge_slope_5s"] = e - e.shift(5)
        g["edge_slope_10s"] = e - e.shift(10)
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)

    for state in state_cols:
        age_col = f"age_{state}"
        out[age_col] = np.nan
        if state not in out.columns:
            continue
        age_maps = []
        for date, g in out.groupby("date_str", sort=False):
            ts_state = (
                g.groupby("timestamp", sort=True)[state]
                .max()
                .sort_index()
            )
            on = pd.to_numeric(ts_state, errors="coerce").fillna(0.0) > 0.5
            ages = {}
            episode_start = None
            for ts, flag in on.items():
                ts = pd.Timestamp(ts)
                if flag:
                    if episode_start is None:
                        episode_start = ts
                    ages[ts] = float((ts - episode_start).total_seconds())
                else:
                    episode_start = None
                    ages[ts] = np.nan
            age_maps.append(pd.DataFrame({"timestamp": list(ages.keys()), age_col: list(ages.values()), "date_str": date}))
        if age_maps:
            am = pd.concat(age_maps, ignore_index=True)
            out = out.drop(columns=[age_col], errors="ignore").merge(
                am, on=["date_str", "timestamp"], how="left"
            )
    return out.drop(columns=["_edge"], errors="ignore")


def timing_mask(df: pd.DataFrame, state: str, policy: str) -> pd.Series:
    """Boolean mask on rows already inside `state`."""
    n = len(df)
    ok = pd.Series(True, index=df.index)
    age = pd.to_numeric(df.get(f"age_{state}"), errors="coerce")
    slope5 = pd.to_numeric(df.get("edge_slope_5s"), errors="coerce")
    slope10 = pd.to_numeric(df.get("edge_slope_10s"), errors="coerce")
    edge = pd.to_numeric(df.get("tree_edge_score"), errors="coerce")

    if policy == "baseline":
        return ok
    if policy == "fresh_le_30":
        return age.notna() & (age <= 30)
    if policy == "fresh_le_60":
        return age.notna() & (age <= 60)
    if policy == "fresh_le_120":
        return age.notna() & (age <= 120)
    if policy == "onset_5_60":
        return age.notna() & (age >= 5) & (age <= 60)
    if policy == "onset_10_90":
        return age.notna() & (age >= 10) & (age <= 90)
    if policy == "delay_ge_30":
        return age.notna() & (age >= 30)
    if policy == "mid_30_180":
        return age.notna() & (age >= 30) & (age <= 180)
    if policy == "edge_rising5":
        return slope5.fillna(-1e9) > 0
    if policy == "edge_rising10":
        return slope10.fillna(-1e9) > 0
    if policy == "fresh60_rising5":
        return age.notna() & (age <= 60) & (slope5.fillna(-1e9) > 0)
    if policy == "onset5_60_rising5":
        return age.notna() & (age >= 5) & (age <= 60) & (slope5.fillna(-1e9) > 0)
    if policy == "edge_not_extended":
        # within-day state rows: keep edge below 70th percentile (avoid chasing peak edge)
        thr = edge.where(ok).quantile(0.70)
        if not np.isfinite(thr):
            return ok
        return edge.notna() & (edge <= thr)
    if policy == "fresh60_not_extended":
        thr = edge.quantile(0.70)
        return age.notna() & (age <= 60) & edge.notna() & (edge <= thr)
    raise ValueError(policy)


def select_with_timing(
    panel: pd.DataFrame,
    *,
    policy: str,
    cooldown_s: int,
    daily_topk: int,
) -> pd.DataFrame:
    """Like adaptive_daily_trades, but applies timing mask per state first."""
    if policy == "baseline":
        return adaptive_daily_trades(panel, CURATED_RULES, "tree_edge_score", cooldown_s, daily_topk)

    candidates = []
    for row in CURATED_RULES.itertuples(index=False):
        rule = row.rule
        state = row.state
        side = row.side
        topk = int(row.topk_per_day)
        if rule not in panel.columns or state not in panel.columns:
            continue
        sub = panel.loc[rule_active_mask(panel, state, side)].copy()
        if sub.empty:
            continue
        # timing filter inside this state
        # edge_not_extended threshold should be computed on state-active rows of the day;
        # apply per day for stability.
        kept = []
        for _, day in sub.groupby("date_str", sort=False):
            m = timing_mask(day, state, policy)
            part = day.loc[m]
            if not part.empty:
                kept.append(part)
        if not kept:
            continue
        sub2 = pd.concat(kept, ignore_index=True)
        from factor_lab.tools.analyze_0dte_state_alpha_attribution import choose_daily_topk

        picks = choose_daily_topk(sub2, rule, max_topk=topk, cooldown_s=cooldown_s)
        if picks.empty:
            continue
        picks = picks.copy()
        picks["active_rule"] = rule
        picks["active_state"] = state
        picks["active_side"] = side
        picks["edge_for_rank"] = pd.to_numeric(picks[rule], errors="coerce").fillna(-1e9)
        picks["timing_policy"] = policy
        if f"age_{state}" in picks.columns:
            picks["state_age_s"] = pd.to_numeric(picks[f"age_{state}"], errors="coerce")
        candidates.append(picks)
    if not candidates:
        return pd.DataFrame()
    all_cands = pd.concat(candidates, ignore_index=True)
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


def diagnose_baseline_ages(trades: pd.DataFrame) -> dict:
    if trades.empty or "state_age_s" not in trades.columns:
        # try reconstruct from age_* columns
        ages = []
        for _, r in trades.iterrows():
            st = str(r.get("active_state", ""))
            col = f"age_{st}"
            if col in trades.columns:
                ages.append(float(r[col]))
        if not ages and "state_age_s" in trades.columns:
            ages = pd.to_numeric(trades["state_age_s"], errors="coerce").dropna().tolist()
        s = pd.Series(ages, dtype=float)
    else:
        s = pd.to_numeric(trades["state_age_s"], errors="coerce").dropna()
    if s.empty:
        return {"n": int(len(trades))}
    return {
        "n": int(len(trades)),
        "age_mean": float(s.mean()),
        "age_median": float(s.median()),
        "age_p25": float(s.quantile(0.25)),
        "age_p75": float(s.quantile(0.75)),
        "pct_age_le_30": float((s <= 30).mean()),
        "pct_age_le_60": float((s <= 60).mean()),
        "pct_age_gt_180": float((s > 180).mean()),
    }


POLICIES = [
    "baseline",
    "fresh_le_30",
    "fresh_le_60",
    "fresh_le_120",
    "onset_5_60",
    "onset_10_90",
    "delay_ge_30",
    "mid_30_180",
    "edge_rising5",
    "edge_rising10",
    "fresh60_rising5",
    "onset5_60_rising5",
    "edge_not_extended",
    "fresh60_not_extended",
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [Path(x.strip()) for x in args.panel_cache_dirs.split(",") if x.strip()]
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    state_cols = CURATED_RULES["state"].tolist()

    print("[timing] fitting scorers on Apr fit window", flush=True)
    target = f"target_exec_ret_{args.horizon_s}s"
    fit_data, _thresholds = load_fit_period(args, target)
    _, weights, model = fit_rule_scorers(fit_data, target)

    panels: dict[str, pd.DataFrame] = {}
    for month in months:
        fp = resolve_panel(cache_dirs, month)
        if fp is None:
            print(f"[timing] missing {month}", flush=True)
            continue
        print(f"[timing] load+score+feature {month}", flush=True)
        df = pd.read_parquet(fp)
        df["month"] = month
        if "date_str" not in df.columns:
            df["date_str"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        df = apply_rule_scorers(df, weights, model)
        df, _ = attach_all_states(df)
        panels[month] = attach_timing_features(df, state_cols)
        print(
            f"  rows={len(panels[month])} recovering={(panels[month]['is_qqq_recovering']>0.5).sum()} "
            f"lunch={(panels[month]['is_stock_trend_down__and__is_lunch']>0.5).sum()}",
            flush=True,
        )

    results = {}
    trade_frames = []
    for policy in POLICIES:
        print(f"[timing] policy={policy}", flush=True)
        month_stats = {}
        frames = []
        for month, panel in panels.items():
            raw = select_with_timing(
                panel,
                policy=policy,
                cooldown_s=args.cooldown_s,
                daily_topk=args.daily_topk,
            )
            if raw.empty:
                month_stats[month] = {"trades": 0}
                continue
            # attach age for diagnostics on baseline-like picks
            if "state_age_s" not in raw.columns:
                ages = []
                for r in raw.itertuples(index=False):
                    col = f"age_{getattr(r, 'active_state')}"
                    ages.append(getattr(r, col) if hasattr(r, col) else np.nan)
                raw = raw.copy()
                raw["state_age_s"] = ages
            confirmed = filter_trades_by_confirm(raw, thresholds=FROZEN_CONFIRM, enabled=True)
            path = path_exec_return(
                panel,
                confirmed,
                hold_s=45,
                commission=args.commission_per_contract,
                state_hold=STATE_HOLD_S,
                use_state_hold=True,
            )
            if not path.empty:
                path = path.copy()
                path["month"] = month
                path["timing_policy"] = policy
                frames.append(path)
            month_stats[month] = {
                **summarize_trades(path, "path_exec_ret", policy, position_frac=args.position_frac),
                "age_diag": diagnose_baseline_ages(path if not path.empty else confirmed),
            }
            print(
                f"  {month}: trades={month_stats[month].get('trades', 0)} "
                f"avg={month_stats[month].get('avg_return', 0):.4f}",
                flush=True,
            )
        all_tr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not all_tr.empty:
            trade_frames.append(all_tr)
        apr_jun = all_tr[all_tr["month"].isin(["2026-04", "2026-05", "2026-06"])] if not all_tr.empty else all_tr
        july = all_tr[all_tr["month"].eq("2026-07")] if not all_tr.empty else all_tr
        results[policy] = {
            "monthly": month_stats,
            "apr_jun": summarize_trades(apr_jun, "path_exec_ret", policy, position_frac=args.position_frac),
            "july": summarize_trades(july, "path_exec_ret", policy, position_frac=args.position_frac),
            "age_apr_jun": diagnose_baseline_ages(apr_jun) if not apr_jun.empty else {},
            "age_july": diagnose_baseline_ages(july) if not july.empty else {},
        }

    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(out_dir / "all_policy_trades.parquet", index=False)

    # ranking: prefer Jul improvement without destroying Apr-Jun too much
    rows = []
    base_aj = results["baseline"]["apr_jun"].get("total_return_position", 0.0)
    base_jul = results["baseline"]["july"].get("total_return_position", 0.0)
    for policy, res in results.items():
        aj = res["apr_jun"]
        jul = res["july"]
        rows.append(
            {
                "policy": policy,
                "apr_jun_trades": aj.get("trades", 0),
                "apr_jun_avg": aj.get("avg_return", 0.0),
                "apr_jun_acct": aj.get("total_return_position", 0.0),
                "apr_jun_lift": aj.get("total_return_position", 0.0) - base_aj,
                "july_trades": jul.get("trades", 0),
                "july_avg": jul.get("avg_return", 0.0),
                "july_acct": jul.get("total_return_position", 0.0),
                "july_lift": jul.get("total_return_position", 0.0) - base_jul,
                "age_median_aj": res.get("age_apr_jun", {}).get("age_median"),
                "age_median_jul": res.get("age_july", {}).get("age_median"),
            }
        )
    rank = pd.DataFrame(rows).sort_values(
        ["july_lift", "apr_jun_lift", "july_avg"], ascending=False
    )
    rank.to_csv(out_dir / "policy_rank.csv", index=False)

    # decision heuristic
    viable = rank[
        (rank["apr_jun_lift"] >= -0.50)  # don't destroy AJ too much
        & (rank["july_lift"] > 0)
        & (rank["july_trades"] >= 4)
        & (rank["policy"] != "baseline")
    ]
    recommendation = viable.iloc[0].to_dict() if not viable.empty else {
        "policy": None,
        "reason": "no timing policy beat July without large Apr-Jun damage under current thresholds",
    }

    summary = {
        "experiment_type": "entry timing ablation inside curated states",
        "hypothesis": "selection timing inside state matters more than No-Trade frequency cuts",
        "config": vars(args),
        "confirm_thresholds": FROZEN_CONFIRM,
        "policies": results,
        "policy_rank": rank.to_dict(orient="records"),
        "recommendation": recommendation,
        "files": {
            "rank": str(out_dir / "policy_rank.csv"),
            "trades": str(out_dir / "all_policy_trades.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("recommendation:", json.dumps(recommendation, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
