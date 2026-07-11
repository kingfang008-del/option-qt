#!/usr/bin/env python3
"""Evaluate interpretable trade-print factor gates for 0DTE options.

The thresholds are learned from a training window and applied unchanged to an
out-of-sample window.  This keeps the analysis closer to factor validation than
to model fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset


FACTOR_COLS = [
    "trade_notional_sum_60s",
    "trade_notional_sum_10s",
    "quote_events_sum_10s",
    "quote_events_sum_5s",
    "quote_event_intensity",
    "quote_imbalance",
    "spread_pct",
    "flow_imbalance_1s",
    "buy_ratio",
    "net_buy_sum_5s",
    "spread_compress_3s",
]


def target_cols(horizons: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for h in horizons:
        out.extend([f"target_mid_ret_{h}s", f"target_exec_ret_{h}s"])
    return out


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = ["date_str", "side"]
    out["rank_notional_60s"] = out.groupby(keys)["trade_notional_sum_60s"].rank(pct=True)
    out["rank_quote_10s"] = out.groupby(keys)["quote_events_sum_10s"].rank(pct=True)
    out["rank_quote_imbalance"] = out.groupby(keys)["quote_imbalance"].rank(pct=True)
    out["rank_spread_tight"] = 1.0 - out.groupby(keys)["spread_pct"].rank(pct=True)
    out["score_hot_quote"] = 0.5 * out["rank_notional_60s"] + 0.5 * out["rank_quote_10s"]
    out["score_hot_quote_tight"] = (
        0.4 * out["rank_notional_60s"] + 0.4 * out["rank_quote_10s"] + 0.2 * out["rank_spread_tight"]
    )
    out["score_hot_quote_imb"] = (
        0.4 * out["rank_notional_60s"] + 0.35 * out["rank_quote_10s"] + 0.25 * out["rank_quote_imbalance"]
    )
    return out


def thresholds(train: pd.DataFrame, qs: tuple[float, ...]) -> dict:
    cols = [
        "trade_notional_sum_60s",
        "trade_notional_sum_10s",
        "quote_events_sum_10s",
        "quote_events_sum_5s",
        "quote_event_intensity",
        "quote_imbalance",
        "spread_pct",
        "score_hot_quote",
        "score_hot_quote_tight",
        "score_hot_quote_imb",
    ]
    out: dict[str, dict[str, float]] = {}
    for side, sub in train.groupby("side"):
        side_thresholds: dict[str, float] = {}
        for c in cols:
            s = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            for q in qs:
                side_thresholds[f"{c}_q{int(q * 100)}"] = float(s.quantile(q))
        out[str(side)] = side_thresholds
    return out


def gate_mask(df: pd.DataFrame, side: str, th: dict[str, dict[str, float]], name: str) -> pd.Series:
    t = th[side]
    base = df["side"].eq(side)
    if name == "rank1":
        return base & (df["universe_rank"] <= 1)
    if name == "notional90":
        return base & (df["trade_notional_sum_60s"] >= t["trade_notional_sum_60s_q90"])
    if name == "quote90":
        return base & (df["quote_events_sum_10s"] >= t["quote_events_sum_10s_q90"])
    if name == "notional90_quote90":
        return (
            base
            & (df["trade_notional_sum_60s"] >= t["trade_notional_sum_60s_q90"])
            & (df["quote_events_sum_10s"] >= t["quote_events_sum_10s_q90"])
        )
    if name == "notional75_quote90":
        return (
            base
            & (df["trade_notional_sum_60s"] >= t["trade_notional_sum_60s_q75"])
            & (df["quote_events_sum_10s"] >= t["quote_events_sum_10s_q90"])
        )
    if name == "notional90_quote75":
        return (
            base
            & (df["trade_notional_sum_60s"] >= t["trade_notional_sum_60s_q90"])
            & (df["quote_events_sum_10s"] >= t["quote_events_sum_10s_q75"])
        )
    if name == "notional90_quote90_tight":
        return (
            base
            & (df["trade_notional_sum_60s"] >= t["trade_notional_sum_60s_q90"])
            & (df["quote_events_sum_10s"] >= t["quote_events_sum_10s_q90"])
            & (df["spread_pct"] <= t["spread_pct_q25"])
        )
    if name == "score_hot_quote95":
        return base & (df["score_hot_quote"] >= t["score_hot_quote_q95"])
    if name == "score_hot_quote_tight95":
        return base & (df["score_hot_quote_tight"] >= t["score_hot_quote_tight_q95"])
    if name == "score_hot_quote_imb95":
        return base & (df["score_hot_quote_imb"] >= t["score_hot_quote_imb_q95"])
    raise ValueError(f"unknown gate: {name}")


def summarize_slice(df: pd.DataFrame, targets: list[str]) -> dict:
    if df.empty:
        return {"n": 0}
    out = {
        "n": int(len(df)),
        "dates": int(df["date_str"].nunique()),
        "side_counts": df["side"].value_counts().to_dict(),
        "avg_spread_pct": float(pd.to_numeric(df["spread_pct"], errors="coerce").mean()),
        "avg_ask": float(pd.to_numeric(df["ask"], errors="coerce").mean()),
    }
    for target in targets:
        s = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        out[f"{target}_mean"] = float(s.mean())
        out[f"{target}_pos_rate"] = float((s > 0).mean())
        out[f"{target}_p50"] = float(s.quantile(0.50))
        out[f"{target}_p90"] = float(s.quantile(0.90))
        out[f"{target}_p95"] = float(s.quantile(0.95))
        out[f"{target}_p99"] = float(s.quantile(0.99))
    return out


def daily_topk(df: pd.DataFrame, score_col: str, k: int, target: str, cooldown_s: int) -> dict:
    rows = []
    for date_str, g in df.sort_values(["date_str", score_col], ascending=[True, False]).groupby("date_str"):
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
    chosen_df = pd.DataFrame(rows)
    out = summarize_slice(chosen_df, [target])
    out["score_col"] = score_col
    out["topk_per_day"] = int(k)
    return out


def evaluate_gates(train: pd.DataFrame, test: pd.DataFrame, th: dict, targets: list[str], cooldown_s: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_names = [
        "rank1",
        "notional90",
        "quote90",
        "notional90_quote90",
        "notional75_quote90",
        "notional90_quote75",
        "notional90_quote90_tight",
        "score_hot_quote95",
        "score_hot_quote_tight95",
        "score_hot_quote_imb95",
    ]
    rows = []
    topk_rows = []
    for side in ["CALL", "PUT"]:
        for gate in gate_names:
            for split, df in [("train", train), ("test", test)]:
                selected = df[gate_mask(df, side, th, gate)]
                rows.append({"split": split, "side": side, "gate": gate, **summarize_slice(selected, targets)})
            for k in (1, 2, 3, 5):
                selected = test[gate_mask(test, side, th, gate)]
                if selected.empty:
                    continue
                topk_rows.append(
                    {
                        "split": "test",
                        "side": side,
                        "gate": gate,
                        **daily_topk(selected, "score_hot_quote_tight", k, "target_exec_ret_10s", cooldown_s),
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
    p.add_argument("--horizons", default="5,10,30")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_tradeprint_factor_gates")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[gate-analysis] loading train", flush=True)
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
    print("[gate-analysis] loading test", flush=True)
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
    th = thresholds(train, (0.25, 0.50, 0.75, 0.90, 0.95))
    targets = target_cols(horizons)
    print(f"[gate-analysis] train_rows={len(train)} test_rows={len(test)}", flush=True)
    gates, topk = evaluate_gates(train, test, th, targets, args.cooldown_s)
    gates.to_csv(out_dir / "gate_summary.csv", index=False)
    topk.to_csv(out_dir / "daily_topk_summary.csv", index=False)
    (out_dir / "thresholds.json").write_text(json.dumps(th, indent=2, default=str), encoding="utf-8")

    metric = "target_exec_ret_10s_mean"
    test_gates = gates[(gates["split"] == "test") & (gates["n"] >= 500)].copy()
    top = test_gates.sort_values(metric, ascending=False).head(20)
    topk_best = topk[topk["n"] >= 5].sort_values("target_exec_ret_10s_mean", ascending=False).head(20)
    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "base": {
            "train": summarize_slice(train, targets),
            "test": summarize_slice(test, targets),
        },
        "top_test_gates_exec_10s": top.to_dict("records"),
        "top_test_daily_topk_exec_10s": topk_best.to_dict("records"),
        "files": {
            "gate_summary": str(out_dir / "gate_summary.csv"),
            "daily_topk_summary": str(out_dir / "daily_topk_summary.csv"),
            "thresholds": str(out_dir / "thresholds.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
