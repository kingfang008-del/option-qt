#!/usr/bin/env python3
"""Execution-price sensitivity for validated 0DTE trade-print factor gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset
from factor_lab.tools.eval_0dte_tradeprint_factor_gates import (
    add_composite_scores,
    gate_mask,
    thresholds,
)


GATES = [
    "score_hot_quote_tight95",
    "score_hot_quote95",
    "score_hot_quote_imb95",
    "notional90_quote90_tight",
    "notional90_quote90",
    "notional90",
    "quote90",
]


def add_execution_targets(df: pd.DataFrame, horizons: tuple[int, ...], commission: float) -> pd.DataFrame:
    out = df.copy()
    ask = pd.to_numeric(out["ask"], errors="coerce")
    bid = pd.to_numeric(out["bid"], errors="coerce")
    mid = pd.to_numeric(out["mid"], errors="coerce")
    half_spread = (ask - mid).clip(lower=0.0)
    for h in horizons:
        cost_ask = 2.0 * commission / (ask * 100.0).replace(0, np.nan)
        future_bid = ask * (1.0 + pd.to_numeric(out[f"target_exec_ret_{h}s"], errors="coerce") + cost_ask)
        future_mid = mid * (1.0 + pd.to_numeric(out[f"target_mid_ret_{h}s"], errors="coerce"))
        for improve in (0.0, 0.25, 0.50, 0.75, 1.0):
            entry = ask - improve * half_spread
            cost = 2.0 * commission / (entry * 100.0).replace(0, np.nan)
            tag = f"entry_improve_{int(improve * 100)}"
            out[f"{tag}_exit_bid_ret_{h}s"] = future_bid / entry - 1.0 - cost
            out[f"{tag}_exit_mid_ret_{h}s"] = future_mid / entry - 1.0 - cost
    return out


def summarize(df: pd.DataFrame, ret_cols: list[str]) -> dict:
    if df.empty:
        return {"n": 0}
    out = {
        "n": int(len(df)),
        "dates": int(df["date_str"].nunique()),
        "avg_spread_pct": float(pd.to_numeric(df["spread_pct"], errors="coerce").mean()),
        "avg_ask": float(pd.to_numeric(df["ask"], errors="coerce").mean()),
        "side_counts": df["side"].value_counts().to_dict(),
    }
    for col in ret_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        out[f"{col}_mean"] = float(s.mean())
        out[f"{col}_pos_rate"] = float((s > 0).mean())
        out[f"{col}_p50"] = float(s.quantile(0.50))
        out[f"{col}_p90"] = float(s.quantile(0.90))
        out[f"{col}_p95"] = float(s.quantile(0.95))
        out[f"{col}_p99"] = float(s.quantile(0.99))
    return out


def daily_topk(df: pd.DataFrame, score_col: str, k: int, ret_cols: list[str], cooldown_s: int) -> dict:
    rows = []
    for _, g in df.sort_values(["date_str", score_col], ascending=[True, False]).groupby("date_str"):
        last_ts = None
        chosen = 0
        for row in g.sort_values(score_col, ascending=False).itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "timestamp"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            rows.append(row._asdict())
            last_ts = ts
            chosen += 1
            if chosen >= k:
                break
    out = summarize(pd.DataFrame(rows), ret_cols)
    out["topk_per_day"] = int(k)
    return out


def evaluate(test: pd.DataFrame, th: dict, horizons: tuple[int, ...], cooldown_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_cols = []
    for h in horizons:
        for improve in (0, 25, 50, 75, 100):
            ret_cols.append(f"entry_improve_{improve}_exit_bid_ret_{h}s")
            ret_cols.append(f"entry_improve_{improve}_exit_mid_ret_{h}s")

    rows = []
    topk_rows = []
    for side in ("CALL", "PUT"):
        for gate in GATES:
            selected = test[gate_mask(test, side, th, gate)].copy()
            rows.append({"side": side, "gate": gate, **summarize(selected, ret_cols)})
            for k in (1, 2, 3, 5):
                if selected.empty:
                    continue
                topk_rows.append(
                    {
                        "side": side,
                        "gate": gate,
                        **daily_topk(selected, "score_hot_quote_tight", k, ret_cols, cooldown_s),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(topk_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--train-start", default="2026-04-13")
    p.add_argument("--train-end", default="2026-05-29")
    p.add_argument("--test-start", default="2026-06-01")
    p.add_argument("--test-end", default="2026-06-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizons", default="10,30")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_tradeprint_execution_sensitivity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[exec-sensitivity] loading train", flush=True)
    train = load_factor_dataset(
        Path(args.micro_root),
        args.train_start,
        args.train_end,
        horizons,
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    print("[exec-sensitivity] loading test", flush=True)
    test = load_factor_dataset(
        Path(args.micro_root),
        args.test_start,
        args.test_end,
        horizons,
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=False,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    train = add_composite_scores(train)
    test = add_composite_scores(test)
    test = add_execution_targets(test, horizons, args.commission_per_contract)
    th = thresholds(train, (0.25, 0.50, 0.75, 0.90, 0.95))
    gate_summary, topk_summary = evaluate(test, th, horizons, args.cooldown_s)
    gate_summary.to_csv(out_dir / "execution_gate_summary.csv", index=False)
    topk_summary.to_csv(out_dir / "execution_daily_topk_summary.csv", index=False)

    key_cols = [
        "entry_improve_0_exit_bid_ret_30s_mean",
        "entry_improve_50_exit_bid_ret_30s_mean",
        "entry_improve_100_exit_bid_ret_30s_mean",
        "entry_improve_100_exit_mid_ret_30s_mean",
    ]
    top = gate_summary[gate_summary["n"] >= 500].sort_values(key_cols[1], ascending=False).head(20)
    topk = topk_summary[topk_summary["n"] >= 5].sort_values(key_cols[1], ascending=False).head(20)
    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "top_gate_by_entry50_exit_bid_30s": top.to_dict("records"),
        "top_daily_topk_by_entry50_exit_bid_30s": topk.to_dict("records"),
        "files": {
            "gate_summary": str(out_dir / "execution_gate_summary.csv"),
            "daily_topk_summary": str(out_dir / "execution_daily_topk_summary.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
