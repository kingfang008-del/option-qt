#!/usr/bin/env python3
"""C-class Exit / trailing ablation for curated State Gate trades.

Compares fixed state-clock exits against trailing / early-lock rules on the
same entry set.  Focus:
  - C_mfe_but_exit_fail trades (had MFE, finished red)
  - full Apr-Jun confirm + July OOS impact (do not destroy winners)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_state_gate_mfe_exit import (
    build_ticker_index,
    exec_path_returns,
)
from factor_lab.tools.analyze_0dte_july_failure_attribution import classify_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--apr-jun-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/"
            "trades_all.parquet"
        ),
    )
    p.add_argument(
        "--july-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/"
            "trades_all.parquet"
        ),
    )
    p.add_argument(
        "--panel-cache-dirs",
        default=(
            "factor_lab/results/0dte_state_gate_h1_cache,"
            "factor_lab/results/0dte_state_gate_jul_w1_cache"
        ),
    )
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_c_class_exit_trailing",
    )
    return p.parse_args()


def account_metrics(returns: pd.Series, *, label: str, position_frac: float) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {
            "label": label,
            "trades": 0,
            "avg_return": 0.0,
            "hit_rate": 0.0,
            "total_return_position": 0.0,
            "max_drawdown_from_initial": 0.0,
        }
    equity = np.cumprod(1.0 + position_frac * r.to_numpy())
    eq0 = np.r_[1.0, equity]
    dd = eq0 / np.maximum.accumulate(eq0) - 1.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    return {
        "label": label,
        "trades": int(len(r)),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total_return_position": float(equity[-1] - 1.0),
        "max_drawdown_from_initial": float(dd.min()),
    }


def resolve_panel(cache_dirs: list[Path], month: str) -> Path | None:
    for d in cache_dirs:
        fp = d / f"score_dataset_{month}.parquet"
        if fp.exists():
            return fp
    return None


def simulate_exits(
    bids: np.ndarray,
    entry_ask: float,
    *,
    commission: float,
    clock_hold: int,
) -> dict[str, float]:
    """Return realized exec returns under multiple exit policies."""
    rets = exec_path_returns(bids, entry_ask, commission)
    n = len(rets)
    if n < 2 or not np.isfinite(rets).any():
        return {}
    clock = min(int(clock_hold), n - 1)
    valid = np.isfinite(rets)

    def at(i: int) -> float:
        i = min(max(i, 0), n - 1)
        return float(rets[i]) if np.isfinite(rets[i]) else float("nan")

    mfe = float(np.nanmax(rets))
    mae = float(np.nanmin(rets))
    mfe_t = int(np.nanargmax(np.where(valid, rets, -np.inf)))
    profit_hits = np.where(valid & (rets > 0))[0]
    ttp = int(profit_hits[0]) if len(profit_hits) else -1

    out = {
        "mfe": mfe,
        "mae": mae,
        "mfe_t": mfe_t,
        "time_to_profit": ttp,
        "clock": at(clock),
        "oracle_mfe": mfe,  # upper bound if exit exactly at peak
        "fixed_30s": at(min(30, n - 1)),
        "fixed_45s": at(min(45, n - 1)),
        "fixed_60s": at(min(60, n - 1)),
        "fixed_90s": at(min(90, n - 1)),
        "fixed_120s": at(min(120, n - 1)),
        "fixed_180s": at(min(180, n - 1)),
    }

    # First green after min hold
    for min_h, tag in [(3, "tp_after3"), (5, "tp_after5"), (10, "tp_after10")]:
        hits = np.where(valid & (np.arange(n) >= min_h) & (rets > 0))[0]
        idx = int(hits[0]) if len(hits) else clock
        idx = min(idx, clock)
        out[tag] = at(idx)
        out[f"{tag}_t"] = idx

    # Trailing: arm after MFE>=trigger, exit when giveback >= trail * peak
    for trigger, trail, tag in [
        (0.02, 0.50, "trail2_50"),
        (0.03, 0.40, "trail3_40"),
        (0.03, 0.50, "trail3_50"),
        (0.05, 0.50, "trail5_50"),
        (0.05, 0.30, "trail5_30"),
        (0.08, 0.40, "trail8_40"),
    ]:
        peak = -np.inf
        armed = False
        exit_i = clock
        for i, r in enumerate(rets[: clock + 1]):
            if not np.isfinite(r):
                continue
            if r > peak:
                peak = r
            if peak >= trigger:
                armed = True
            if armed and peak > 0 and r <= peak * (1.0 - trail) and i > 0:
                exit_i = i
                break
        out[tag] = at(exit_i)
        out[f"{tag}_t"] = int(exit_i)

    # Lock: once +X reached, exit immediately (take-profit)
    for tp, tag in [(0.03, "tp_lock3"), (0.05, "tp_lock5"), (0.08, "tp_lock8"), (0.10, "tp_lock10")]:
        hits = np.where(valid & (rets >= tp))[0]
        idx = int(hits[0]) if len(hits) else clock
        idx = min(idx, clock)
        out[tag] = at(idx)
        out[f"{tag}_t"] = idx

    # Hybrid: arm at +3%, then trail 50% giveback, else clock
    peak = -np.inf
    armed = False
    exit_i = clock
    for i, r in enumerate(rets[: clock + 1]):
        if not np.isfinite(r):
            continue
        if r > peak:
            peak = r
        if peak >= 0.03:
            armed = True
        if armed and peak > 0 and r <= peak - 0.5 * peak and i > 0:
            exit_i = i
            break
    out["hybrid_lock3_trail50"] = at(exit_i)
    out["hybrid_lock3_trail50_t"] = int(exit_i)

    return out


def diagnose_set(
    trades: pd.DataFrame,
    cache_dirs: list[Path],
    *,
    commission: float,
) -> pd.DataFrame:
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    if "month" not in work.columns:
        work["month"] = work["timestamp"].dt.strftime("%Y-%m")
    if "date_str" not in work.columns:
        work["date_str"] = work["timestamp"].dt.strftime("%Y-%m-%d")
    work["hold_s"] = pd.to_numeric(work.get("hold_s"), errors="coerce").fillna(45).astype(int)

    rows = []
    for month, g in work.groupby("month", sort=True):
        fp = resolve_panel(cache_dirs, str(month))
        if fp is None:
            print(f"[exit] missing panel for {month}", flush=True)
            continue
        panel = pd.read_parquet(fp)
        index = build_ticker_index(panel)
        for tr in g.itertuples(index=False):
            path = index.get(str(tr.ticker))
            if path is None or path.empty:
                continue
            ts = pd.Timestamp(tr.timestamp)
            tns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
            pos = int(np.searchsorted(tns, ts.value, side="left"))
            if pos >= len(path):
                continue
            if abs(int(tns[pos]) - ts.value) > 1_500_000_000:
                exact = np.where(tns == ts.value)[0]
                if len(exact) == 0:
                    continue
                pos = int(exact[0])
            hold = int(getattr(tr, "hold_s", 45))
            end = min(len(path) - 1, pos + max(hold, 180))
            seg = path.iloc[pos : end + 1]
            entry_ask = float(pd.to_numeric(seg["ask"].iloc[0], errors="coerce"))
            if not np.isfinite(entry_ask) or entry_ask <= 0:
                continue
            bids = pd.to_numeric(seg["bid"], errors="coerce").to_numpy(dtype=float)
            sim = simulate_exits(bids, entry_ask, commission=commission, clock_hold=hold)
            if not sim:
                continue
            rec = {
                "month": str(month),
                "date_str": getattr(tr, "date_str"),
                "timestamp": ts,
                "ticker": str(tr.ticker),
                "side": getattr(tr, "side"),
                "active_state": getattr(tr, "active_state", ""),
                "hold_s": hold,
                "path_exec_ret": float(getattr(tr, "path_exec_ret", sim.get("clock", np.nan))),
                "entry_ask": entry_ask,
            }
            # classify with path stats for bucket
            class_row = pd.Series(
                {
                    **rec,
                    "mfe": sim["mfe"],
                    "mae": sim["mae"],
                    "mfe_t": sim["mfe_t"],
                    "time_to_profit": sim["time_to_profit"],
                    "stock_ret_hold": np.nan,
                    "final_mid": np.nan,
                    "half_spread_cost": np.nan,
                    "commission_cost": 2.0 * commission / (entry_ask * 100.0),
                }
            )
            # Prefer existing fail_bucket if present
            bucket = getattr(tr, "fail_bucket", None)
            if bucket is None or (isinstance(bucket, float) and np.isnan(bucket)) or str(bucket) == "":
                # crude bucket without stock: C if mfe>=3% and clock<=0; F if clock>0
                if sim["clock"] > 0:
                    bucket = "F_winner"
                elif sim["mfe"] >= 0.03:
                    bucket = "C_mfe_but_exit_fail"
                else:
                    bucket = "G_other"
            rec["fail_bucket"] = str(bucket)
            rec.update(sim)
            rows.append(rec)
    return pd.DataFrame(rows)


EXIT_COLS = [
    "clock",
    "oracle_mfe",
    "fixed_30s",
    "fixed_45s",
    "fixed_60s",
    "fixed_90s",
    "fixed_120s",
    "fixed_180s",
    "tp_after3",
    "tp_after5",
    "tp_after10",
    "trail2_50",
    "trail3_40",
    "trail3_50",
    "trail5_50",
    "trail5_30",
    "trail8_40",
    "tp_lock3",
    "tp_lock5",
    "tp_lock8",
    "tp_lock10",
    "hybrid_lock3_trail50",
]


def summarize_exits(df: pd.DataFrame, position_frac: float) -> dict:
    if df.empty:
        return {"n": 0}
    out = {
        "n": int(len(df)),
        "bucket_counts": df["fail_bucket"].value_counts().to_dict() if "fail_bucket" in df.columns else {},
        "avg_mfe": float(df["mfe"].mean()),
        "avg_clock": float(df["clock"].mean()),
        "exits": {},
    }
    for col in EXIT_COLS:
        if col not in df.columns:
            continue
        metrics = account_metrics(df[col], label=col, position_frac=position_frac)
        # lift vs clock
        metrics["lift_vs_clock_avg"] = float(df[col].mean() - df["clock"].mean())
        out["exits"][col] = metrics
    ranked = sorted(
        out["exits"].items(),
        key=lambda kv: (kv[1]["avg_return"], kv[1]["total_return_position"]),
        reverse=True,
    )
    out["best_exits"] = [{"exit": k, **v} for k, v in ranked[:10]]
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [Path(x.strip()) for x in args.panel_cache_dirs.split(",") if x.strip()]

    apr_jun_raw = pd.read_parquet(args.apr_jun_trades)
    apr_jun_raw = apr_jun_raw[apr_jun_raw["month"].isin(["2026-04", "2026-05", "2026-06"])].copy()
    # attach fail_bucket from scored file if available
    scored = Path("factor_lab/results/0dte_state_gate_no_trade_gate_h1/apr_jun_confirm_scored.parquet")
    if scored.exists():
        sc = pd.read_parquet(scored)[["timestamp", "ticker", "fail_bucket", "mfe", "mae"]]
        sc["timestamp"] = pd.to_datetime(sc["timestamp"])
        apr_jun_raw["timestamp"] = pd.to_datetime(apr_jun_raw["timestamp"])
        apr_jun_raw = apr_jun_raw.merge(sc, on=["timestamp", "ticker"], how="left", suffixes=("", "_sc"))

    july_raw = pd.read_parquet(args.july_trades)
    july_attr = Path("factor_lab/results/0dte_state_gate_july_w1_failure_attribution/trade_attribution.parquet")
    if july_attr.exists():
        ja = pd.read_parquet(july_attr)[["timestamp", "ticker", "fail_bucket"]]
        ja["timestamp"] = pd.to_datetime(ja["timestamp"])
        july_raw["timestamp"] = pd.to_datetime(july_raw["timestamp"])
        july_raw = july_raw.merge(ja, on=["timestamp", "ticker"], how="left")
    july_raw["month"] = "2026-07"

    print("[exit] diagnosing Apr-Jun", flush=True)
    apr_jun = diagnose_set(apr_jun_raw, cache_dirs, commission=args.commission_per_contract)
    print("[exit] diagnosing July", flush=True)
    july = diagnose_set(july_raw, cache_dirs, commission=args.commission_per_contract)

    apr_jun.to_parquet(out_dir / "apr_jun_exit_paths.parquet", index=False)
    july.to_parquet(out_dir / "july_exit_paths.parquet", index=False)

    summary = {
        "config": vars(args),
        "apr_jun_all": summarize_exits(apr_jun, args.position_frac),
        "apr_jun_C": summarize_exits(apr_jun[apr_jun["fail_bucket"].eq("C_mfe_but_exit_fail")], args.position_frac),
        "apr_jun_F": summarize_exits(apr_jun[apr_jun["fail_bucket"].eq("F_winner")], args.position_frac),
        "july_all": summarize_exits(july, args.position_frac),
        "july_C": summarize_exits(july[july["fail_bucket"].eq("C_mfe_but_exit_fail")], args.position_frac),
        "july_F": summarize_exits(july[july["fail_bucket"].eq("F_winner")], args.position_frac),
    }

    # Recommended policy: pick best trailing that improves C without collapsing F on Apr-Jun
    candidates = [
        "trail3_50",
        "trail5_50",
        "hybrid_lock3_trail50",
        "tp_lock5",
        "tp_after5",
        "fixed_60s",
        "fixed_90s",
    ]
    rec = []
    for col in candidates:
        if col not in apr_jun.columns:
            continue
        aj_all = account_metrics(apr_jun[col], label=col, position_frac=args.position_frac)
        aj_c = account_metrics(
            apr_jun.loc[apr_jun["fail_bucket"].eq("C_mfe_but_exit_fail"), col],
            label=col,
            position_frac=args.position_frac,
        )
        aj_clock = account_metrics(apr_jun["clock"], label="clock", position_frac=args.position_frac)
        jul_all = account_metrics(july[col], label=col, position_frac=args.position_frac)
        jul_c = account_metrics(
            july.loc[july["fail_bucket"].eq("C_mfe_but_exit_fail"), col],
            label=col,
            position_frac=args.position_frac,
        )
        jul_clock = account_metrics(july["clock"], label="clock", position_frac=args.position_frac)
        rec.append(
            {
                "exit": col,
                "apr_jun_avg": aj_all["avg_return"],
                "apr_jun_acct": aj_all["total_return_position"],
                "apr_jun_lift_acct": aj_all["total_return_position"] - aj_clock["total_return_position"],
                "apr_jun_C_avg": aj_c["avg_return"],
                "july_avg": jul_all["avg_return"],
                "july_acct": jul_all["total_return_position"],
                "july_lift_acct": jul_all["total_return_position"] - jul_clock["total_return_position"],
                "july_C_avg": jul_c["avg_return"],
            }
        )
    # Prefer positive Jul C rescue and non-negative Apr-Jun lift (or small damage)
    rec_sorted = sorted(
        rec,
        key=lambda x: (
            x["july_C_avg"],
            x["july_lift_acct"],
            x["apr_jun_lift_acct"],
            x["apr_jun_acct"],
        ),
        reverse=True,
    )
    summary["candidate_compare"] = rec_sorted
    summary["recommendation"] = rec_sorted[0] if rec_sorted else {}
    summary["files"] = {
        "apr_jun": str(out_dir / "apr_jun_exit_paths.parquet"),
        "july": str(out_dir / "july_exit_paths.parquet"),
        "summary": str(out_dir / "summary.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "candidate_compare.csv").write_text(
        pd.DataFrame(rec_sorted).to_csv(index=False), encoding="utf-8"
    )

    print(json.dumps({
        "apr_jun_clock": summary["apr_jun_all"]["exits"].get("clock"),
        "july_clock": summary["july_all"]["exits"].get("clock"),
        "july_C_best": summary["july_C"].get("best_exits", [])[:5],
        "apr_jun_C_best": summary["apr_jun_C"].get("best_exits", [])[:5],
        "recommendation": summary["recommendation"],
        "candidate_compare": rec_sorted[:8],
    }, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
