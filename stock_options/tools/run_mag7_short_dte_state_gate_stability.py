#!/usr/bin/env python3
"""MAG7 short-DTE Rule×State×Month stability (per symbol × trading-dte).

Reuses QQQ State Gate primitives but:
  - does NOT copy QQQ curated rules
  - filters micro contracts to one selected_dte bucket before universe selection
  - writes under stock_options/results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_lab.tools.analyze_0dte_rule_state_stability import (
    apply_rule_scorers,
    attach_all_states,
    fit_rule_scorers,
    month_starts,
    monthly_rule_rows,
    stability_score,
)
from factor_lab.tools.analyze_0dte_tradeprint_factors import (
    add_contract_factors,
    add_dynamic_universe,
    normalize_day,
)
from factor_lab.tools.run_0dte_factor_score_loop import build_score_dataset
from factor_lab.tools.run_0dte_minimal_five_layer_loop import load_stock_state_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, choices=["NVDA", "TSLA", "QQQ"])
    p.add_argument("--selected-dte", type=int, required=True, choices=[0, 1, 2])
    p.add_argument(
        "--micro-root",
        default="/mnt/s990/data/microstructure/mag7_short_dte_api_ladder",
    )
    p.add_argument("--stock-root", default="", help="default: /mnt/s990/data/raw_1s/stocks/{SYMBOL}")
    p.add_argument("--fit-start", default="2026-02-02")
    p.add_argument("--fit-end", default="2026-03-31")
    p.add_argument("--start", default="2026-02-02")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.08)
    p.add_argument("--min-ask", type=float, default=0.15)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--min-rows", type=int, default=80)
    p.add_argument("--min-months", type=int, default=2)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def load_raw(args: argparse.Namespace, start: str, end: str) -> pd.DataFrame:
    """Load factors for one symbol, filtering selected_dte *before* dynamic universe."""
    sym = args.symbol.upper()
    micro_root = Path(args.micro_root)
    files = sorted((micro_root / f"contract_1s/{sym}").glob(f"{sym}_*.parquet"))
    prefix = f"{sym}_"
    files = [p for p in files if start <= p.stem.replace(prefix, "") <= end]
    frames: list[pd.DataFrame] = []
    for fp in files:
        raw = pd.read_parquet(fp)
        if raw.empty:
            continue
        if "selected_dte" in raw.columns:
            raw = raw[pd.to_numeric(raw["selected_dte"], errors="coerce") == int(args.selected_dte)]
        elif "target_dte" in raw.columns:
            raw = raw[pd.to_numeric(raw["target_dte"], errors="coerce") == int(args.selected_dte)]
        else:
            raise SystemExit("micro missing selected_dte/target_dte")
        if raw.empty:
            continue
        day = add_contract_factors(normalize_day(raw), (args.horizon_s,), args.commission_per_contract)
        day = add_dynamic_universe(day, top_n=args.top_n, lookback_s=args.lookback_s, per_side=False)
        tradable = (
            day["side"].isin(["CALL", "PUT"])
            & (day["ask"] >= args.min_ask)
            & (day["bid"] > 0)
            & (day["spread_pct"] <= args.max_spread_pct)
            & day["bucket_id"].notna()
        )
        day = day[tradable].copy()
        if day.empty:
            continue
        day["date_str"] = fp.stem.replace(prefix, "")
        day["underlying"] = sym
        day["selected_dte"] = int(args.selected_dte)
        frames.append(day)
    if not frames:
        raise SystemExit(f"no factor rows for {sym} dte={args.selected_dte} {start}..{end}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


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
    raw = load_raw(args, start, end)
    stock = load_stock_state_features(Path(args.stock_root), start, end, symbol=args.symbol)
    data, _ = build_score_dataset(raw, stock, target, thresholds)
    data["month"] = month
    data["symbol"] = args.symbol
    data["selected_dte"] = int(args.selected_dte)
    data.to_parquet(fp, index=False)
    return data


def main() -> None:
    args = parse_args()
    if not args.stock_root:
        args.stock_root = f"/mnt/s990/data/raw_1s/stocks/{args.symbol}"
    if not args.output_dir:
        args.output_dir = (
            f"stock_options/results/mag7_state_gate_{args.symbol.lower()}_dte{args.selected_dte}"
        )
    out_dir = Path(args.output_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = f"target_exec_ret_{args.horizon_s}s"

    print(
        f"[mag7-stability] {args.symbol} dte={args.selected_dte} "
        f"fit={args.fit_start}..{args.fit_end} eval={args.start}..{args.end}",
        flush=True,
    )
    fit_raw = load_raw(args, args.fit_start, args.fit_end)
    fit_stock = load_stock_state_features(
        Path(args.stock_root), args.fit_start, args.fit_end, symbol=args.symbol
    )
    fit_data, thresholds = build_score_dataset(fit_raw, fit_stock, target, None)
    fit_data, weights, model = fit_rule_scorers(fit_data, target)

    months = []
    for month, start, end in month_starts(args.start, args.end):
        print(f"[mag7-stability] loading {month}", flush=True)
        try:
            data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        except SystemExit as exc:
            print(f"  skip {month}: {exc}", flush=True)
            continue
        data = apply_rule_scorers(data, weights, model)
        months.append(data)
    if not months:
        raise SystemExit("no monthly panels built")
    panel = pd.concat(months, ignore_index=True)
    panel, states = attach_all_states(panel)
    print(
        f"[mag7-stability] rows={len(panel)} months={panel['month'].nunique()} states={len(states)}",
        flush=True,
    )
    monthly = monthly_rule_rows(panel, target, states, args.cooldown_s, args.min_rows)
    stable = stability_score(monthly, args.min_months)
    monthly.to_csv(out_dir / "rule_state_month_matrix.csv", index=False)
    stable.to_csv(out_dir / "rule_stability_score.csv", index=False)

    curated_candidates = stable[
        (stable["state"] != "ALL")
        & (stable["mean_return"] > 0)
        & (stable["positive_month_ratio"] >= 0.5)
    ].head(30)

    summary = {
        "experiment": "mag7_short_dte_rule_state_stability",
        "symbol": args.symbol,
        "selected_dte": int(args.selected_dte),
        "note": "framework shared with QQQ; params/thresholds fit independently; no QQQ curated copy",
        "config": vars(args),
        "rows": int(len(panel)),
        "months": sorted(panel["month"].astype(str).unique().tolist()),
        "states": states,
        "ic_weights": weights,
        "matrix_rows": int(len(monthly)),
        "stable_rules": int(len(stable)),
        "best_stable_rules": stable.head(40).to_dict("records"),
        "curated_candidates": curated_candidates.to_dict("records"),
        "files": {
            "rule_state_month_matrix": str(out_dir / "rule_state_month_matrix.csv"),
            "rule_stability_score": str(out_dir / "rule_stability_score.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("top curated candidates:")
    if curated_candidates.empty:
        print("  (none)")
    else:
        cols = [
            "rule",
            "state",
            "side",
            "topk_per_day",
            "rule_score",
            "mean_return",
            "positive_month_ratio",
            "mean_trades",
        ]
        print(curated_candidates[cols].head(15).to_string(index=False))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
