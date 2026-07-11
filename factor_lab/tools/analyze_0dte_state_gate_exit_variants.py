#!/usr/bin/env python3
"""State-specific hold and light trailing exits for curated State Gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(rets: pd.Series, label: str, *, position_frac: float = 0.25) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {"label": label, "trades": 0, "position_frac": float(position_frac)}
    eq = (1.0 + float(position_frac) * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    total_ret = float(eq.iloc[-1] - 1.0)
    return {
        "label": label,
        "trades": int(len(r)),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "sum_return": float(r.sum()),
        "position_frac": float(position_frac),
        "total_return_position": total_ret,
        "total_return_10pct_position": total_ret,
        "max_drawdown": float(dd.min()),
    }


def apply_policy(df: pd.DataFrame, policy: str) -> pd.Series:
    """Map each trade to an exit return under a named policy."""
    rec = df["active_state"].eq("is_qqq_recovering")
    lunch = df["active_state"].eq("is_stock_trend_down__and__is_lunch")
    out = pd.Series(np.nan, index=df.index, dtype=float)

    if policy == "fixed_30":
        out = df["fixed_30s"]
    elif policy == "fixed_45":
        out = df["fixed_45s"]
    elif policy == "fixed_60":
        out = df["fixed_60s"]
    elif policy == "rec45_lunch60":
        out = np.where(rec, df["fixed_45s"], np.where(lunch, df["fixed_60s"], df["fixed_45s"]))
        out = pd.Series(out, index=df.index)
    elif policy == "rec60_lunch45":
        out = np.where(rec, df["fixed_60s"], np.where(lunch, df["fixed_45s"], df["fixed_45s"]))
        out = pd.Series(out, index=df.index)
    elif policy == "rec45_lunch45":
        out = df["fixed_45s"]
    elif policy == "trail5_50":
        out = df["exit_trail5_50"]
    elif policy == "trail3_40":
        out = df["exit_trail3_40"]
    elif policy == "rec45_lunch_trail5":
        out = np.where(rec, df["fixed_45s"], np.where(lunch, df["exit_trail5_50"], df["fixed_45s"]))
        out = pd.Series(out, index=df.index)
    elif policy == "rec_trail5_lunch45":
        out = np.where(rec, df["exit_trail5_50"], np.where(lunch, df["fixed_45s"], df["fixed_45s"]))
        out = pd.Series(out, index=df.index)
    elif policy == "rec45_lunch_trail3":
        out = np.where(rec, df["fixed_45s"], np.where(lunch, df["exit_trail3_40"], df["fixed_45s"]))
        out = pd.Series(out, index=df.index)
    elif policy == "best_of_45_60":
        # oracle upper bound, not tradable
        out = df[["fixed_45s", "fixed_60s"]].max(axis=1)
    else:
        raise ValueError(policy)
    return pd.to_numeric(out, errors="coerce")


def walk_forward_select(train: pd.DataFrame, test: pd.DataFrame, cands: list[str]) -> dict:
    """Select one global policy and per-state atomic exits on train, evaluate on test."""
    global_best = max(cands, key=lambda p: apply_policy(train, p).mean())
    atomic = ["fixed_30", "fixed_45", "fixed_60", "trail5_50", "trail3_40"]
    per_state = {}
    for st, g in train.groupby("active_state"):
        per_state[st] = max(atomic, key=lambda p: apply_policy(g, p).mean())

    parts = []
    for st, pol in per_state.items():
        te = test[test["active_state"].eq(st)]
        parts.append(apply_policy(te, pol))
    selected = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return {
        "global_best_policy": global_best,
        "global_best_train": summarize(apply_policy(train, global_best), f"train_{global_best}"),
        "global_best_test": summarize(apply_policy(test, global_best), f"test_{global_best}"),
        "per_state_policies": per_state,
        "per_state_test": summarize(selected, "test_per_state_selected"),
        "baseline_test_fixed45": summarize(apply_policy(test, "fixed_45"), "test_fixed_45"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mfe-path",
        default="factor_lab/results/0dte_state_gate_mfe_diag_apr_jun/trade_mfe_paths.parquet",
    )
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_exit_variants_apr_jun")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.mfe_path)
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m")

    policies = [
        "fixed_30",
        "fixed_45",
        "fixed_60",
        "rec45_lunch60",
        "rec60_lunch45",
        "trail5_50",
        "trail3_40",
        "rec45_lunch_trail5",
        "rec_trail5_lunch45",
        "rec45_lunch_trail3",
        "best_of_45_60",
    ]

    overall = {p: summarize(apply_policy(df, p), p) for p in policies}
    by_month = {}
    for month, g in df.groupby("month"):
        by_month[month] = {p: summarize(apply_policy(g, p), p) for p in policies}

    by_state = {}
    for st, g in df.groupby("active_state"):
        by_state[st] = {p: summarize(apply_policy(g, p), p) for p in ["fixed_30", "fixed_45", "fixed_60", "trail5_50", "trail3_40"]}

    train = df[df["month"].isin(["2026-04", "2026-05"])].copy()
    test = df[df["month"].eq("2026-06")].copy()
    # Only tradable policies for WF selection (exclude oracle best_of)
    wf_cands = [
        "fixed_30",
        "fixed_45",
        "fixed_60",
        "rec45_lunch60",
        "rec60_lunch45",
        "trail5_50",
        "trail3_40",
        "rec45_lunch_trail5",
        "rec_trail5_lunch45",
        "rec45_lunch_trail3",
    ]
    wf = walk_forward_select(train, test, wf_cands)

    # Rank overall by account return, excluding oracle
    ranked = sorted(
        ((k, v) for k, v in overall.items() if k != "best_of_45_60"),
        key=lambda kv: kv[1].get("total_return_10pct_position", -1e9),
        reverse=True,
    )

    summary = {
        "n_trades": int(len(df)),
        "overall": overall,
        "by_month": by_month,
        "by_state": by_state,
        "walk_forward_apr_may_to_jun": wf,
        "ranked_by_account_return": [{"policy": k, **v} for k, v in ranked],
        "note": (
            "Policies reuse path-level exits from MFE diagnosis. "
            "rec45_lunch60 = recovering fixed 45s + lunch fixed 60s. "
            "Trailing uses existing trail5_50 / trail3_40 path exits."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "ranked": summary["ranked_by_account_return"][:6],
        "walk_forward": wf,
        "jun_key": {p: by_month["2026-06"][p] for p in ["fixed_45", "rec45_lunch60", "rec60_lunch45", "trail5_50", "rec45_lunch_trail5"]},
    }, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
