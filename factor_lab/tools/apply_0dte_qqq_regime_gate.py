#!/usr/bin/env python3
"""Apply shadow Regime gate block_pred_1 to curated trade parquets.

Reads causal_regime_predictions + trades, drops regime_pred==1, writes filtered
trades + before/after account metrics. Does not re-run State Gate discovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True)
    p.add_argument(
        "--regime-preds",
        default="factor_lab/results/0dte_qqq_deep_anchor_scaffold/causal_regime_predictions.parquet",
    )
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def account(returns: pd.Series, position_frac: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {"n": 0, "avg_ret": None, "account_ret": 0.0, "max_dd": 0.0, "win_rate": None}
    eq = np.cumprod(1.0 + position_frac * r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_rate": float((r > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(args.trades)
    if "date_str" not in tr.columns:
        tr["date_str"] = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m-%d")
    reg = pd.read_parquet(args.regime_preds)
    m = tr.merge(reg[["date_str", "regime_pred", "p_regime_0", "p_regime_1", "p_regime_2"]], on="date_str", how="left")
    miss = int(m["regime_pred"].isna().sum())
    kept = m[m["regime_pred"].fillna(-1).astype(int) != 1].copy()
    summary = {
        "policy": "block_pred_1",
        "status": "shadow",
        "missing_regime_rows": miss,
        "baseline": account(m["path_exec_ret"], args.position_frac),
        "gated": account(kept["path_exec_ret"], args.position_frac),
        "dropped_n": int(len(m) - len(kept)),
    }
    kept.to_parquet(out / "trades_block_pred_1.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
