#!/usr/bin/env python3
"""Ablate QQQ Deep Anchor regime gates on curated path trades (Jan–Jun + Jul OOS).

Gate forms (applied to already-selected curated trades):
  - baseline: keep all
  - block_pred_k: drop regime_pred == k
  - keep_pred_k: keep only regime_pred == k
  - block_p_stress_ge_q: drop if p_regime_2 >= quantile(q) on train months
  - keep_p_stress_ge_q: keep if p_regime_2 >= quantile(q)
  - size_by_inv_stress: position_frac *= (1 - p_regime_2)  [downweight stress]
  - size_by_stress: position_frac *= p_regime_2            [upweight stress]

Protocol:
  - Regime probs from causal expanding stock model (train months < eval month).
  - Quantile thresholds for p_* gates fit on Jan–May only when scoring Jun/Jul,
    or on all months strictly before the eval month for monthly loops.
  - Jul is hard OOS for curated alpha; Jan–Jun is diagnostic stratification.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--regime-dir",
        default="factor_lab/results/0dte_qqq_deep_anchor_scaffold",
        help="dir with causal_regime_predictions.parquet; will refresh if missing Jul",
    )
    p.add_argument(
        "--jan-jun-trades",
        default="factor_lab/results/0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/trades_all.parquet",
    )
    p.add_argument(
        "--jul-trades",
        default="factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/trades_all.parquet",
    )
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--refresh-regime", action="store_true")
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_qqq_deep_anchor_gate_ablation",
    )
    return p.parse_args()


def ensure_regime(regime_dir: Path, refresh: bool) -> pd.DataFrame:
    pred_path = regime_dir / "causal_regime_predictions.parquet"
    need = refresh or (not pred_path.exists())
    if pred_path.exists() and not refresh:
        pred = pd.read_parquet(pred_path)
        if pred["date_str"].max() < "2026-07-01":
            need = True
    if need:
        cmd = [
            sys.executable,
            "qqq_btc/tools/train_0dte_qqq_deep_anchor.py",
            "--start-date",
            "2023-01-01",
            "--end-date",
            "2026-07-09",
            "--output-dir",
            str(regime_dir),
        ]
        print("[gate] refreshing regime predictions through Jul...", flush=True)
        subprocess.check_call(cmd)
    pred = pd.read_parquet(pred_path)
    for c in ("p_regime_0", "p_regime_1", "p_regime_2"):
        if c not in pred.columns:
            pred[c] = 0.0
    return pred


def load_trades(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date_str" not in df.columns:
        df["date_str"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.copy()
    df["path_exec_ret"] = pd.to_numeric(df["path_exec_ret"], errors="coerce")
    df["split"] = tag
    df["month"] = df["date_str"].str.slice(0, 7)
    return df.dropna(subset=["path_exec_ret"])


def account_stats(returns: pd.Series, position_frac: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "n": 0,
            "avg_ret": float("nan"),
            "win_rate": float("nan"),
            "account_ret": 0.0,
            "max_dd": 0.0,
        }
    eq = np.cumprod(1.0 + position_frac * r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    dd = eq / peaks - 1.0
    return {
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float(dd.min()),
    }


def apply_gate(
    trades: pd.DataFrame,
    *,
    name: str,
    mask: pd.Series | None = None,
    size_mult: pd.Series | None = None,
    base_frac: float = 0.25,
) -> dict:
    if mask is None:
        kept = trades
    else:
        kept = trades.loc[mask.fillna(False)]
    if size_mult is None:
        stats = account_stats(kept["path_exec_ret"], base_frac)
        stats.update({"gate": name, "mean_size_mult": 1.0})
        return stats
    # trade-level variable size: compound with per-trade frac
    r = pd.to_numeric(kept["path_exec_ret"], errors="coerce")
    m = pd.to_numeric(size_mult.loc[kept.index], errors="coerce").clip(lower=0.0, upper=1.5)
    valid = r.notna() & m.notna()
    r = r[valid]
    m = m[valid]
    if r.empty:
        return {
            "gate": name,
            "n": 0,
            "avg_ret": float("nan"),
            "win_rate": float("nan"),
            "account_ret": 0.0,
            "max_dd": 0.0,
            "mean_size_mult": float("nan"),
        }
    fracs = base_frac * m.to_numpy()
    eq = np.cumprod(1.0 + fracs * r.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    dd = eq / peaks - 1.0
    return {
        "gate": name,
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float(dd.min()),
        "mean_size_mult": float(m.mean()),
    }


def quantile_thr(train: pd.DataFrame, col: str, q: float) -> float:
    s = pd.to_numeric(train[col], errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def eval_split(trades: pd.DataFrame, thr_train: pd.DataFrame, base_frac: float) -> list[dict]:
    """Evaluate gates; continuous thresholds from thr_train distribution."""
    rows = []
    p2 = pd.to_numeric(trades["p_regime_2"], errors="coerce")
    p1 = pd.to_numeric(trades["p_regime_1"], errors="coerce")
    pred = trades["regime_pred"]

    rows.append(apply_gate(trades, name="baseline", base_frac=base_frac))
    for k in (0, 1, 2):
        rows.append(apply_gate(trades, name=f"block_pred_{k}", mask=pred != k, base_frac=base_frac))
        rows.append(apply_gate(trades, name=f"keep_pred_{k}", mask=pred == k, base_frac=base_frac))

    for q in (0.33, 0.50, 0.67):
        thr = quantile_thr(thr_train, "p_regime_2", q)
        rows.append(
            apply_gate(
                trades,
                name=f"block_p2_ge_q{q:.2f}",
                mask=p2 < thr,
                base_frac=base_frac,
            )
        )
        rows.append(
            apply_gate(
                trades,
                name=f"keep_p2_ge_q{q:.2f}",
                mask=p2 >= thr,
                base_frac=base_frac,
            )
        )

    # size multipliers (always keep trades)
    rows.append(
        apply_gate(
            trades,
            name="size_inv_stress",
            size_mult=(1.0 - p2).clip(0.15, 1.0),
            base_frac=base_frac,
        )
    )
    rows.append(
        apply_gate(
            trades,
            name="size_by_stress",
            size_mult=p2.clip(0.15, 1.0),
            base_frac=base_frac,
        )
    )
    rows.append(
        apply_gate(
            trades,
            name="size_by_trendish",
            size_mult=p1.clip(0.15, 1.0),
            base_frac=base_frac,
        )
    )
    # hybrid: keep only pred!=1 (drop mid bucket that looked weak in scaffold), full size
    rows.append(apply_gate(trades, name="block_pred_1_only", mask=pred != 1, base_frac=base_frac))
    return rows


def by_regime_table(trades: pd.DataFrame, base_frac: float) -> pd.DataFrame:
    rows = []
    for k, g in trades.groupby("regime_pred"):
        s = account_stats(g["path_exec_ret"], base_frac)
        s["regime_pred"] = int(k)
        s["days"] = int(g["date_str"].nunique())
        rows.append(s)
    return pd.DataFrame(rows).sort_values("regime_pred")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    regime = ensure_regime(Path(args.regime_dir), args.refresh_regime)
    jan_jun = load_trades(Path(args.jan_jun_trades), "jan_jun")
    jul = load_trades(Path(args.jul_trades), "jul_oos")
    all_tr = pd.concat([jan_jun, jul], ignore_index=True)
    merged = all_tr.merge(regime, on="date_str", how="left", suffixes=("", "_reg"))
    if "month_reg" in merged.columns:
        merged = merged.drop(columns=["month_reg"])
    miss = merged["regime_pred"].isna().sum()
    if miss:
        print(f"[gate] WARNING: {miss} trades missing regime pred", flush=True)
    merged = merged.dropna(subset=["regime_pred"]).copy()
    merged["regime_pred"] = merged["regime_pred"].astype(int)
    merged.to_parquet(out / "trades_with_regime.parquet", index=False)

    # threshold train: months before Jul for Jul; for jan_jun diagnostic use months before each trade's month
    # For simplicity: global thr from Jan–May for both Jun diagnostic slice and Jul OOS
    thr_train = merged[merged["month"].isin(["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"])]
    if thr_train.empty:
        thr_train = merged[merged["split"] == "jan_jun"]

    strat_rows = []
    gate_rows = []
    for split, g in merged.groupby("split"):
        strat = by_regime_table(g, args.position_frac)
        strat["split"] = split
        strat_rows.append(strat)
        for row in eval_split(g, thr_train, args.position_frac):
            row["split"] = split
            gate_rows.append(row)

    # also Apr–Jun strong window diagnostic
    strong = merged[merged["month"].isin(["2026-04", "2026-05", "2026-06"])]
    if not strong.empty:
        strat = by_regime_table(strong, args.position_frac)
        strat["split"] = "apr_jun"
        strat_rows.append(strat)
        for row in eval_split(strong, thr_train, args.position_frac):
            row["split"] = "apr_jun"
            gate_rows.append(row)

    strat_df = pd.concat(strat_rows, ignore_index=True)
    gate_df = pd.DataFrame(gate_rows)
    strat_df.to_csv(out / "by_regime_pred.csv", index=False)
    gate_df.to_csv(out / "gate_ablation.csv", index=False)

    # pick candidates: Jul OOS improve DD or account vs baseline without killing n too hard
    jul_gates = gate_df[gate_df["split"] == "jul_oos"].copy()
    base = jul_gates[jul_gates["gate"] == "baseline"].iloc[0]
    jul_gates["d_account"] = jul_gates["account_ret"] - base["account_ret"]
    jul_gates["d_dd"] = jul_gates["max_dd"] - base["max_dd"]  # less negative = better
    jul_gates["n_keep_ratio"] = jul_gates["n"] / max(base["n"], 1)

    # also check apr_jun not destroyed
    apr = gate_df[gate_df["split"] == "apr_jun"].set_index("gate")
    jul_gates["apr_jun_account"] = jul_gates["gate"].map(
        lambda g: float(apr.loc[g, "account_ret"]) if g in apr.index else float("nan")
    )
    jul_gates["apr_jun_n"] = jul_gates["gate"].map(
        lambda g: float(apr.loc[g, "n"]) if g in apr.index else float("nan")
    )
    base_apr_n = float(apr.loc["baseline", "n"]) if "baseline" in apr.index else float("nan")
    jul_gates["apr_jun_n_ratio"] = jul_gates["apr_jun_n"] / base_apr_n if base_apr_n else float("nan")

    ranked = jul_gates.sort_values(["d_account", "d_dd"], ascending=False)
    ranked.to_csv(out / "jul_oos_gate_ranked.csv", index=False)

    # promotion heuristic (shadow only): Jul account up, DD not worse by >2pp, keep >=50% Jul trades,
    # and Apr–Jun keep >=70% trades with account not worse than 50% of baseline
    promote = ranked[
        (ranked["d_account"] > 0)
        & (ranked["d_dd"] >= -0.02)
        & (ranked["n_keep_ratio"] >= 0.5)
        & (ranked["apr_jun_n_ratio"] >= 0.7)
        & (ranked["apr_jun_account"] >= 0.5 * float(apr.loc["baseline", "account_ret"]))
    ]

    summary = {
        "experiment": "qqq_deep_anchor_gate_ablation",
        "position_frac": args.position_frac,
        "n_trades_merged": int(len(merged)),
        "jul_baseline": base.to_dict(),
        "by_regime_pred": strat_df.to_dict(orient="records"),
        "jul_top_gates": ranked.head(8).to_dict(orient="records"),
        "promotion_candidates": promote.head(5).to_dict(orient="records"),
        "verdict": (
            "promote_shadow"
            if not promote.empty
            else "no_promote_keep_sizing_research"
        ),
        "note": (
            "If stress days are better on Jan–Jun, block_stress will fail Jul too; "
            "prefer size_by_stress / keep_pred_2 over no-trade-on-stress."
        ),
        "files": {
            "trades_with_regime": str(out / "trades_with_regime.parquet"),
            "by_regime": str(out / "by_regime_pred.csv"),
            "gates": str(out / "gate_ablation.csv"),
            "jul_ranked": str(out / "jul_oos_gate_ranked.csv"),
            "summary": str(out / "summary.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "verdict": summary["verdict"],
        "jul_baseline": {k: base[k] for k in ("n", "avg_ret", "account_ret", "max_dd")},
        "jul_top_gates": summary["jul_top_gates"][:5],
        "promotion_candidates": summary["promotion_candidates"],
        "by_regime_pred": summary["by_regime_pred"],
    }, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
