#!/usr/bin/env python3
"""Generic State Gate skeleton for QQQ/stock × 0DTE/1DTE/2DTE.

This is the portable entrypoint. It reuses the curated-gate machinery but:
  - loads symbol/DTE paths from state_gate_profiles.json
  - does NOT freeze QQQ-0DTE curated rules by default
  - first discovers Rule×State stability, then optionally replays a rule JSON

Typical flow:
  1) Ensure micro exists (build_micro_from_raw_1s.py or download_short_dte_microstructure.py)
  2) python -m qqq_btc.tools.run_state_gate_skeleton --profile qqq_1dte --mode stability
  3) Inspect ranked rules, write curated_rules.json
  4) python -m qqq_btc.tools.run_state_gate_skeleton --profile qqq_1dte --mode replay \\
       --rules-json ... --hold-s 45
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
from factor_lab.tools.run_0dte_adaptive_rule_pool import (
    adaptive_daily_trades,
    idle_ratio,
    rule_active_mask,
    summarize_trades,
)
from factor_lab.tools.run_0dte_state_gate_curated import path_exec_return


PROFILES_PATH = Path("factor_lab/CONFIG/state_gate_profiles.json")


def load_profile(name: str, profiles_path: Path = PROFILES_PATH) -> dict:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    if name not in profiles:
        raise SystemExit(f"unknown profile {name}; available={sorted(profiles)}")
    return profiles[name]


def month_bounds(month: str, profile: dict) -> tuple[str, str]:
    start = f"{month}-01"
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    # Optional profile override for partial first month
    overrides = profile.get("month_start_overrides", {})
    if month in overrides:
        start = overrides[month]
    return start, end


def apply_profile_to_args(args: argparse.Namespace, profile: dict) -> argparse.Namespace:
    args.symbol = profile.get("symbol", args.symbol)
    if not getattr(args, "micro_root", None) or args.micro_root == "":
        args.micro_root = profile["micro_root"]
    if not getattr(args, "stock_root", None) or args.stock_root == "":
        args.stock_root = profile["stock_root"]
    if args.months == "":
        args.months = profile.get("default_months", args.months)
    if args.fit_start == "":
        args.fit_start = profile.get("fit_start", args.fit_start)
    if args.fit_end == "":
        args.fit_end = profile.get("fit_end", args.fit_end)
    if args.max_spread_pct is None:
        args.max_spread_pct = float(profile.get("max_spread_pct", 0.05))
    if args.min_ask is None:
        args.min_ask = float(profile.get("min_ask", 0.20))
    return args


def filter_weekdays(df: pd.DataFrame, weekdays: list[int] | None) -> pd.DataFrame:
    if not weekdays or df.empty or "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"])
    keep = ts.dt.weekday.isin(weekdays)
    return df.loc[keep].copy()


def load_rules(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    fp = Path(path)
    if fp.suffix == ".json":
        rows = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(rows, dict) and "rules" in rows:
            rows = rows["rules"]
        return pd.DataFrame(rows)
    return pd.read_csv(fp)


def run_stability(args: argparse.Namespace, profile: dict, out_dir: Path) -> dict:
    target = f"target_exec_ret_{args.horizon_s}s"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    print(f"[skeleton] fitting scorers symbol={args.symbol}", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    fit_data = filter_weekdays(fit_data, profile.get("weekday_filter"))
    _, weights, model = fit_rule_scorers(fit_data, target)

    frames = []
    for month in months:
        start, end = month_bounds(month, profile)
        print(f"[skeleton] stability panel {month} {start}..{end}", flush=True)
        panel = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        panel = apply_rule_scorers(panel, weights, model)
        panel, states = attach_all_states(panel)
        panel = filter_weekdays(panel, profile.get("weekday_filter"))
        panel["month"] = month
        frames.append(panel)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        raise SystemExit("no panels loaded")

    monthly = monthly_rule_rows(
        data,
        target,
        states,
        cooldown_s=args.cooldown_s,
        min_rows=args.min_rows,
    )
    ranked = stability_score(monthly, min_months=args.min_months)
    monthly.to_csv(out_dir / "rule_state_month_matrix.csv", index=False)
    ranked.to_csv(out_dir / "rule_state_stability_ranked.csv", index=False)

    # Suggest top portable candidates (exclude ALL-state noise optionally)
    suggest = ranked.copy()
    if not args.include_all_state:
        suggest = suggest[suggest["state"].ne("ALL")]
    suggest = suggest.head(args.top_rules)
    suggest.to_csv(out_dir / "suggested_curated_rules.csv", index=False)
    suggest_json = []
    for i, row in suggest.iterrows():
        suggest_json.append(
            {
                "rule": row["rule"],
                "state": row["state"],
                "side": row["side"],
                "topk_per_day": int(row["topk_per_day"]),
                "name": f"{row['rule']}__{row['state']}__{row['side']}__top{int(row['topk_per_day'])}",
                "rule_score": float(row["rule_score"]),
                "positive_month_ratio": float(row["positive_month_ratio"]),
                "mean_return": float(row["mean_return"]),
            }
        )
    (out_dir / "suggested_curated_rules.json").write_text(
        json.dumps({"profile": args.profile, "symbol": args.symbol, "rules": suggest_json}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "mode": "stability",
        "profile": args.profile,
        "symbol": args.symbol,
        "dte": profile.get("dte"),
        "months": months,
        "n_rows": int(len(data)),
        "n_rules_ranked": int(len(ranked)),
        "top_rules": suggest_json[:10],
        "files": {
            "matrix": str(out_dir / "rule_state_month_matrix.csv"),
            "ranked": str(out_dir / "rule_state_stability_ranked.csv"),
            "suggested": str(out_dir / "suggested_curated_rules.json"),
        },
        "next": "Inspect suggested rules, optionally edit, then rerun with --mode replay --rules-json ...",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_replay(args: argparse.Namespace, profile: dict, out_dir: Path) -> dict:
    rules = load_rules(args.rules_json)
    if rules.empty:
        raise SystemExit("--mode replay requires --rules-json with at least one rule")
    for col in ["rule", "state", "side", "topk_per_day"]:
        if col not in rules.columns:
            raise SystemExit(f"rules missing column {col}")

    target = f"target_exec_ret_{args.horizon_s}s"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    print(f"[skeleton] fitting scorers for replay symbol={args.symbol}", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    fit_data = filter_weekdays(fit_data, profile.get("weekday_filter"))
    _, weights, model = fit_rule_scorers(fit_data, target)

    state_hold = {}
    if args.state_hold_json:
        state_hold = json.loads(Path(args.state_hold_json).read_text(encoding="utf-8"))

    monthly = {}
    trade_frames = []
    for month in months:
        start, end = month_bounds(month, profile)
        print(f"[skeleton] replay {month}", flush=True)
        panel = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        panel = apply_rule_scorers(panel, weights, model)
        panel, _ = attach_all_states(panel)
        panel = filter_weekdays(panel, profile.get("weekday_filter"))

        trades = adaptive_daily_trades(panel, rules, "tree_edge_score", args.cooldown_s, args.daily_topk)
        # If rules use their own score column names, re-rank already done inside adaptive via rule col.
        # Prefer each rule's own score when present.
        if not trades.empty and "active_rule" in trades.columns:
            # already set by adaptive_daily_trades
            pass
        trades = path_exec_return(
            panel,
            trades,
            hold_s=args.hold_s,
            commission=args.commission_per_contract,
            state_hold=state_hold,
            use_state_hold=bool(state_hold),
        )
        if not trades.empty:
            trades = trades.copy()
            trades["month"] = month
            trade_frames.append(trades)
            trades.to_parquet(out_dir / f"trades_{month}.parquet", index=False)
        monthly[month] = {
            "days": int(panel["date_str"].nunique()),
            "rows": int(len(panel)),
            "state_gate": {
                **summarize_trades(trades, "path_exec_ret", "state_gate"),
                "idle_day_ratio": idle_ratio(panel, trades),
                "active_day_ratio": 1.0 - idle_ratio(panel, trades),
            },
        }
        print(
            f"[skeleton] {month} avg={monthly[month]['state_gate'].get('avg_return', 0):.4f} "
            f"trades={monthly[month]['state_gate'].get('trades', 0)}",
            flush=True,
        )

    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if not all_trades.empty:
        all_trades.to_parquet(out_dir / "trades_all.parquet", index=False)
    combined = summarize_trades(all_trades, "path_exec_ret", "state_gate_all") if not all_trades.empty else {"trades": 0}
    summary = {
        "mode": "replay",
        "profile": args.profile,
        "symbol": args.symbol,
        "dte": profile.get("dte"),
        "rules": rules.to_dict("records"),
        "hold_s": args.hold_s,
        "state_hold": state_hold,
        "monthly": monthly,
        "combined": combined,
        "files": {"trades_all": str(out_dir / "trades_all.parquet"), "summary": str(out_dir / "summary.json")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, help="key in state_gate_profiles.json")
    p.add_argument("--profiles-path", default=str(PROFILES_PATH))
    p.add_argument("--mode", choices=["stability", "replay", "smoke"], default="stability")
    p.add_argument("--symbol", default="QQQ")
    p.add_argument("--micro-root", default="")
    p.add_argument("--stock-root", default="")
    p.add_argument("--months", default="")
    p.add_argument("--fit-start", default="")
    p.add_argument("--fit-end", default="")
    p.add_argument("--confirm-fit-months", default="")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--hold-s", type=int, default=45)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=None)
    p.add_argument("--min-ask", type=float, default=None)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--min-rows", type=int, default=80)
    p.add_argument("--min-months", type=int, default=2)
    p.add_argument("--top-rules", type=int, default=20)
    p.add_argument("--include-all-state", action="store_true")
    p.add_argument("--rules-json", default="")
    p.add_argument("--state-hold-json", default="")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile, Path(args.profiles_path))
    args = apply_profile_to_args(args, profile)
    if not args.cache_dir:
        args.cache_dir = f"qqq_btc/results/state_gate_{args.profile}/cache"
    if not args.output_dir:
        args.output_dir = f"qqq_btc/results/state_gate_{args.profile}/{args.mode}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    micro = Path(args.micro_root) / "contract_1s" / args.symbol
    if not micro.exists():
        raise SystemExit(
            f"micro missing: {micro}\n"
            f"Build it first, e.g.\n"
            f"  PYTHONPATH=. python qqq_btc/tools/build_micro_from_raw_1s.py \\\n"
            f"    --raw-root {profile.get('raw_1s_root', '<raw>')} --symbol {args.symbol} \\\n"
            f"    --selected-dte {profile.get('dte', '')} --output-dir {args.micro_root}\n"
            f"or download_short_dte_microstructure.py --locked-map {profile.get('locked_map')}"
        )

    print(
        json.dumps(
            {
                "profile": args.profile,
                "symbol": args.symbol,
                "dte": profile.get("dte"),
                "mode": args.mode,
                "micro_root": args.micro_root,
                "stock_root": args.stock_root,
                "months": args.months,
                "notes": profile.get("notes"),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.mode == "stability":
        summary = run_stability(args, profile, out_dir)
    elif args.mode == "replay":
        summary = run_replay(args, profile, out_dir)
    else:
        # smoke: load one month only and print row counts / state coverage
        months = [m.strip() for m in args.months.split(",") if m.strip()][:1]
        args.months = ",".join(months)
        target = f"target_exec_ret_{args.horizon_s}s"
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fit_data, thresholds = load_fit_period(args, target)
        fit_data = filter_weekdays(fit_data, profile.get("weekday_filter"))
        _, weights, model = fit_rule_scorers(fit_data, target)
        month = months[0]
        start, end = month_bounds(month, profile)
        panel = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        panel = apply_rule_scorers(panel, weights, model)
        panel, states = attach_all_states(panel)
        panel = filter_weekdays(panel, profile.get("weekday_filter"))
        summary = {
            "mode": "smoke",
            "profile": args.profile,
            "symbol": args.symbol,
            "month": month,
            "rows": int(len(panel)),
            "days": int(panel["date_str"].nunique()),
            "n_states": len(states),
            "state_hit_rates": {
                s: float(pd.to_numeric(panel[s], errors="coerce").fillna(0).gt(0.5).mean())
                for s in states
                if s in panel.columns
            },
            "score_cols_present": [c for c in ["tree_edge_score", "ic_edge_score", "hot_score"] if c in panel.columns],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str)[:4000])
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
