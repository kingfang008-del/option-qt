#!/usr/bin/env python3
"""No-Trade / Tradeable gate for curated QQQ 0DTE State Gate.

This layer does not create direction.  It decides whether an already-triggered
candidate should be executed.

Two complementary controls:
  1) tradeable logit  — entry-time probability that the setup is worth taking
  2) day_risk_halt    — causal halt after severe losing day(s)

Labels (when path panels are available):
  tradeable = 1  if winner OR mfe >= 3%
  tradeable = 0  if direction-wrong / never-green deep MAE / option-dead with weak mfe
  fallback       path_exec_ret > 0

Causal protocol (default):
  train months = Apr+May
  validation   = Jun  (threshold / halt params)
  forward OOS  = Jul confirm trades
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_july_failure_attribution import (
    classify_row,
    load_stock_close_series,
    option_path_stats,
    stock_return_over,
)
from factor_lab.tools.train_0dte_state_gate_rule_selector import (
    THRESHOLD_GRID,
    account_metrics,
    feature_columns,
    feature_matrix,
    fit_selector,
    predict_selector,
    prepare_frame,
    safe_auc,
)


POSITIVE_BUCKETS = {"F_winner", "C_mfe_but_exit_fail"}
NEGATIVE_BUCKETS = {
    "A_direction_wrong",
    "B_direction_ok_option_dead",
    "D_spread_execution",
    "E_should_no_trade",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_noconfirm_statehold_jan_jun/"
            "trades_all.parquet"
        ),
        help="candidate pool used to fit tradeable model (prefer noconfirm)",
    )
    p.add_argument(
        "--eval-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jan_jun_pos25/"
            "trades_all.parquet"
        ),
        help="confirm-policy trades for chronological evaluation",
    )
    p.add_argument(
        "--oos-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/"
            "trades_all.parquet"
        ),
        help="true forward OOS trades (July W1)",
    )
    p.add_argument(
        "--panel-cache-dirs",
        default=(
            "factor_lab/results/0dte_state_gate_h1_cache,"
            "factor_lab/results/0dte_state_gate_jul_w1_cache"
        ),
    )
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--train-months", default="2026-04,2026-05")
    p.add_argument("--val-month", default="2026-06")
    p.add_argument("--default-threshold", type=float, default=0.55)
    p.add_argument("--severe-day-avg", type=float, default=-0.08)
    p.add_argument("--halt-after-severe-days", type=int, default=1)
    p.add_argument("--min-val-trades", type=int, default=6)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_no_trade_gate_h1",
    )
    return p.parse_args()


def resolve_panel(cache_dirs: list[Path], month: str) -> Path | None:
    for d in cache_dirs:
        fp = d / f"score_dataset_{month}.parquet"
        if fp.exists():
            return fp
    return None


def attach_tradeable_labels(
    trades: pd.DataFrame,
    *,
    cache_dirs: list[Path],
    stock_root: Path,
    commission: float,
) -> pd.DataFrame:
    """Add mfe/mae/stock path stats + tradeable label when panels exist."""
    out = prepare_frame(trades)
    out["mfe"] = np.nan
    out["mae"] = np.nan
    out["mfe_t"] = -1
    out["time_to_profit"] = -1
    out["stock_ret_hold"] = np.nan
    out["final_mid"] = np.nan
    out["half_spread_cost"] = np.nan
    out["commission_cost"] = np.nan
    out["fail_bucket"] = ""
    out["tradeable"] = out["rule_valid"].astype(int)

    months = sorted(out["month"].astype(str).unique())
    stock_start = str(out["date_str"].min())
    stock_end = str(out["date_str"].max())
    stock = load_stock_close_series(stock_root, stock_start, stock_end)

    for month in months:
        fp = resolve_panel(cache_dirs, month)
        if fp is None:
            continue
        panel = pd.read_parquet(fp)
        idx = out.index[out["month"].astype(str) == month]
        for i in idx:
            tr = out.loc[i]
            opt = option_path_stats(panel, tr, commission=commission)
            stk = stock_return_over(stock, tr["timestamp"], int(tr.get("hold_s", 45)))
            for k, v in {**opt, **stk}.items():
                if k in out.columns:
                    out.at[i, k] = v
            rec = out.loc[i]
            bucket, _ = classify_row(rec)
            out.at[i, "fail_bucket"] = bucket
            if bucket in POSITIVE_BUCKETS:
                out.at[i, "tradeable"] = 1
            elif bucket in NEGATIVE_BUCKETS:
                out.at[i, "tradeable"] = 0
            else:
                out.at[i, "tradeable"] = int(float(rec["path_exec_ret"]) > 0)
    return out


def day_severity(day_trades: pd.DataFrame, severe_day_avg: float) -> bool:
    if day_trades.empty:
        return False
    rets = pd.to_numeric(day_trades["path_exec_ret"], errors="coerce").dropna()
    if rets.empty:
        return False
    all_red = bool((rets <= 0).all()) and len(rets) >= 2
    avg_bad = float(rets.mean()) <= float(severe_day_avg)
    return all_red or avg_bad


def apply_day_risk_halt(
    trades: pd.DataFrame,
    *,
    severe_day_avg: float,
    halt_after_severe_days: int,
) -> pd.DataFrame:
    """Causal day halt with live-consistent severity updates.

    - At day open, if skip_remaining > 0, halt today.
    - Severity is computed only on days that were actually traded
      (not halted).  Halted days do not extend the skip window.
    - After a traded severe day, skip the next `halt_after_severe_days` days.
    """
    out = trades.sort_values(["date_str", "timestamp"]).copy()
    out["day_halt"] = False
    out["severe_streak_before"] = 0
    dates = sorted(out["date_str"].unique())
    skip_remaining = 0
    traded_severe_streak = 0
    for d in dates:
        mask = out["date_str"] == d
        halt_today = skip_remaining > 0
        out.loc[mask, "day_halt"] = bool(halt_today)
        out.loc[mask, "severe_streak_before"] = int(traded_severe_streak)
        if halt_today:
            skip_remaining -= 1
            continue
        day = out.loc[mask]
        severe = day_severity(day, severe_day_avg)
        if severe:
            traded_severe_streak += 1
            skip_remaining = max(skip_remaining, int(halt_after_severe_days))
        else:
            traded_severe_streak = 0
    return out


def choose_threshold(
    val: pd.DataFrame,
    *,
    position_frac: float,
    min_trades: int,
    default_threshold: float,
    prob_col: str = "tradeable_probability",
) -> tuple[float, dict]:
    candidates = []
    best_thr = default_threshold
    best_obj = float("-inf")
    # Always include abstain-all and take-all extremes for transparency.
    for thr in [*THRESHOLD_GRID, 0.0, 1.01]:
        selected = val[val[prob_col] >= thr] if thr <= 1.0 else val.iloc[0:0]
        metrics = account_metrics(selected, label=f"val_p>={thr}", position_frac=position_frac)
        n = int(metrics["trades"])
        if thr <= 1.0 and n < min_trades and thr > 0:
            obj = float("-inf")
        elif thr > 1.0:
            # abstain-all: objective 0 (neutral), prefer only if all trading is worse
            obj = 0.0
        else:
            ret = float(metrics["total_return_position"])
            dd = abs(float(metrics["max_drawdown_from_initial"]))
            obj = ret - 0.5 * dd
        candidates.append({"threshold": thr, "objective": obj, **metrics})
        if obj > best_obj:
            best_obj = obj
            best_thr = thr
    return float(best_thr), {
        "threshold": best_thr,
        "best_objective": best_obj,
        "candidates": candidates,
    }


def setup_veto_mask(frame: pd.DataFrame, mode: str = "none") -> pd.Series:
    """Optional research vetoes. Default none — static vetoes destroy Apr-Jun alpha."""
    ok = pd.Series(True, index=frame.index)
    if mode == "none":
        return ok
    lunch = frame["active_state"].astype(str).str.contains("lunch", na=False)
    call = frame["side"].astype(str).eq("CALL")
    if mode == "veto_lunch_call":
        return ~(lunch & call)
    if mode == "veto_lunch_call_stock60":
        stock60 = pd.to_numeric(frame.get("stock_ret_60s"), errors="coerce").fillna(0.0)
        return ~(lunch & call & (stock60 < 0))
    raise ValueError(f"unknown setup veto mode: {mode}")


def apply_no_trade_policy(
    trades: pd.DataFrame,
    *,
    tradeable_probability: pd.Series | None = None,
    threshold: float = 0.0,
    severe_day_avg: float = -0.08,
    halt_after_severe_days: int = 1,
    setup_veto: str = "none",
) -> pd.DataFrame:
    """Apply the No-Trade layer to already-triggered curated trades.

    Returns a copy with:
      day_halt, tradeable_passed, setup_veto_passed, no_trade_passed
    """
    out = trades.copy()
    if tradeable_probability is None:
        out["tradeable_probability"] = 1.0
    else:
        out["tradeable_probability"] = pd.to_numeric(tradeable_probability, errors="coerce").fillna(0.0)
    out = apply_day_risk_halt(
        out,
        severe_day_avg=severe_day_avg,
        halt_after_severe_days=halt_after_severe_days,
    )
    out["tradeable_passed"] = out["tradeable_probability"] >= float(threshold)
    out["setup_veto_passed"] = setup_veto_mask(out, setup_veto)
    out["no_trade_passed"] = (
        out["tradeable_passed"] & (~out["day_halt"].astype(bool)) & out["setup_veto_passed"]
    )
    return out


def apply_gates(
    frame: pd.DataFrame,
    *,
    threshold: float,
    use_day_halt: bool,
    severe_day_avg: float = -0.08,
    halt_after_severe_days: int = 1,
) -> pd.DataFrame:
    out = apply_no_trade_policy(
        frame,
        tradeable_probability=(
            frame["tradeable_probability"] if "tradeable_probability" in frame.columns else None
        ),
        threshold=threshold,
        severe_day_avg=severe_day_avg,
        halt_after_severe_days=halt_after_severe_days,
        setup_veto="none",
    )
    if not use_day_halt:
        out["no_trade_passed"] = out["tradeable_passed"] & out["setup_veto_passed"]
    return out


def policy_metrics(frame: pd.DataFrame, passed_col: str, label: str, position_frac: float) -> dict:
    selected = frame[frame[passed_col]].copy()
    base = account_metrics(selected, label=label, position_frac=position_frac)
    base["blocked_trades"] = int((~frame[passed_col]).sum())
    base["pass_rate"] = float(frame[passed_col].mean()) if len(frame) else 0.0
    if "fail_bucket" in frame.columns and frame["fail_bucket"].astype(str).str.len().gt(0).any():
        blocked = frame.loc[~frame[passed_col], "fail_bucket"].value_counts().to_dict()
        kept = frame.loc[frame[passed_col], "fail_bucket"].value_counts().to_dict()
        base["blocked_buckets"] = blocked
        base["kept_buckets"] = kept
    return base


def _daily_gate_table(frame: pd.DataFrame) -> dict:
    rows = {}
    for d, g in frame.groupby("date_str", sort=True):
        rows[str(d)] = {
            "n": int(len(g)),
            "avg": float(g["path_exec_ret"].mean()),
            "day_halt": bool(g["day_halt"].iloc[0]),
            "severe_streak_before": int(g["severe_streak_before"].iloc[0]),
            "passed": int(g["no_trade_passed"].sum()),
            "blocked": int((~g["no_trade_passed"]).sum()),
            "buckets": g["fail_bucket"].value_counts().to_dict(),
        }
    return rows


def score_with_model(
    model,
    medians: pd.Series,
    prior: float,
    columns: list[str],
    frame: pd.DataFrame,
) -> pd.Series:
    # Reuse selector predict; it reads rule_valid_probability name — rename after.
    probs = predict_selector(model, frame, columns, medians, prior)
    return probs.rename("tradeable_probability")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [Path(x.strip()) for x in args.panel_cache_dirs.split(",") if x.strip()]
    train_months = [m.strip() for m in args.train_months.split(",") if m.strip()]
    val_month = args.val_month.strip()

    print("[no-trade] labeling train pool", flush=True)
    train_pool = attach_tradeable_labels(
        pd.read_parquet(args.train_trades),
        cache_dirs=cache_dirs,
        stock_root=Path(args.stock_root),
        commission=args.commission_per_contract,
    )
    print("[no-trade] labeling eval confirm pool", flush=True)
    eval_pool = attach_tradeable_labels(
        pd.read_parquet(args.eval_trades),
        cache_dirs=cache_dirs,
        stock_root=Path(args.stock_root),
        commission=args.commission_per_contract,
    )
    print("[no-trade] labeling July OOS", flush=True)
    oos = attach_tradeable_labels(
        pd.read_parquet(args.oos_trades),
        cache_dirs=cache_dirs,
        stock_root=Path(args.stock_root),
        commission=args.commission_per_contract,
    )
    # Force July month tag
    oos["month"] = "2026-07"

    # Fit tradeable model on noconfirm Apr+May (more samples), target=tradeable
    fit = train_pool[train_pool["month"].isin(train_months)].copy()
    # Temporarily map tradeable -> rule_valid for reuse of fit_selector
    fit = fit.copy()
    fit["rule_valid"] = fit["tradeable"].astype(int)
    columns = feature_columns(fit, use_regime_features=False)
    model, medians, prior = fit_selector(fit, columns)
    print(
        f"[no-trade] fit months={train_months} n={len(fit)} "
        f"tradeable_rate={fit['tradeable'].mean():.3f} features={len(columns)}",
        flush=True,
    )

    # Score validation (Jun confirm) and choose threshold
    val = eval_pool[eval_pool["month"] == val_month].copy()
    val["tradeable_probability"] = score_with_model(model, medians, prior, columns, val)
    val = apply_day_risk_halt(
        val,
        severe_day_avg=args.severe_day_avg,
        halt_after_severe_days=args.halt_after_severe_days,
    )
    thr, thr_meta = choose_threshold(
        val,
        position_frac=args.position_frac,
        min_trades=args.min_val_trades,
        default_threshold=args.default_threshold,
    )
    print(f"[no-trade] val={val_month} chosen_threshold={thr}", flush=True)

    # Also score Apr-Jun confirm for in-window shadow + Jul OOS
    def decorate(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["tradeable_probability"] = score_with_model(model, medians, prior, columns, x)
        x = apply_day_risk_halt(
            x,
            severe_day_avg=args.severe_day_avg,
            halt_after_severe_days=args.halt_after_severe_days,
        )
        x = apply_gates(x, threshold=thr, use_day_halt=True)
        x["threshold"] = thr
        return x

    apr_jun = decorate(eval_pool[eval_pool["month"].isin(["2026-04", "2026-05", "2026-06"])].copy())
    jul = decorate(oos)

    # Expanding-window shadow on confirm Jan-Jun (diagnostic only)
    expand_rows = []
    months = sorted(eval_pool["month"].unique())
    for test_m in months:
        if test_m <= min(train_months):
            continue
        hist = train_pool[train_pool["month"] < test_m].copy()
        if hist.empty:
            continue
        hist["rule_valid"] = hist["tradeable"].astype(int)
        if hist["rule_valid"].nunique() < 2 or len(hist) < 20:
            continue
        m, med, pr = fit_selector(hist, columns)
        te = eval_pool[eval_pool["month"] == test_m].copy()
        if te.empty:
            continue
        te["tradeable_probability"] = score_with_model(m, med, pr, columns, te)
        te = apply_day_risk_halt(
            te,
            severe_day_avg=args.severe_day_avg,
            halt_after_severe_days=args.halt_after_severe_days,
        )
        # use fixed thr from Jun protocol for comparability on later months;
        # for months before Jun, re-pick on previous month if available.
        local_thr = thr
        prev = hist[hist["month"] == sorted(hist["month"].unique())[-1]].copy()
        if not prev.empty and test_m <= val_month:
            prev["tradeable_probability"] = score_with_model(m, med, pr, columns, prev)
            local_thr, _ = choose_threshold(
                prev,
                position_frac=args.position_frac,
                min_trades=max(4, args.min_val_trades // 2),
                default_threshold=args.default_threshold,
            )
        te = apply_gates(te, threshold=local_thr, use_day_halt=True)
        expand_rows.append(te)
        print(
            f"[no-trade] expand test={test_m} thr={local_thr:.2f} "
            f"pass={te['no_trade_passed'].mean():.2f}",
            flush=True,
        )
    expand = pd.concat(expand_rows, ignore_index=True) if expand_rows else pd.DataFrame()

    summary = {
        "experiment_type": "no-trade / tradeable gate over curated state gate",
        "config": vars(args),
        "label_definition": {
            "positive_buckets": sorted(POSITIVE_BUCKETS),
            "negative_buckets": sorted(NEGATIVE_BUCKETS),
            "fallback": "path_exec_ret > 0",
        },
        "fit": {
            "train_months": train_months,
            "n": int(len(fit)),
            "tradeable_rate": float(fit["tradeable"].mean()),
            "n_features": int(len(columns)),
            "features": columns,
            "prior": float(prior),
        },
        "validation": {
            "month": val_month,
            "threshold": thr,
            "threshold_meta": thr_meta,
            "baseline": account_metrics(val, label="jun_all", position_frac=args.position_frac),
            "tradeable_only": policy_metrics(
                apply_gates(val, threshold=thr, use_day_halt=False),
                "tradeable_passed",
                "jun_tradeable",
                args.position_frac,
            ),
            "no_trade_combo": policy_metrics(
                apply_gates(val, threshold=thr, use_day_halt=True),
                "no_trade_passed",
                "jun_combo",
                args.position_frac,
            ),
            "auc": safe_auc(val["tradeable"], val["tradeable_probability"]),
        },
        "apr_jun_confirm_shadow": {
            "baseline": account_metrics(apr_jun, label="apr_jun_all", position_frac=args.position_frac),
            "tradeable_only": policy_metrics(
                apply_gates(apr_jun, threshold=thr, use_day_halt=False),
                "tradeable_passed",
                "apr_jun_tradeable",
                args.position_frac,
            ),
            "day_halt_only": policy_metrics(
                apr_jun.assign(no_trade_passed=~apr_jun["day_halt"].astype(bool)),
                "no_trade_passed",
                "apr_jun_day_halt",
                args.position_frac,
            ),
            "combo": policy_metrics(apr_jun, "no_trade_passed", "apr_jun_combo", args.position_frac),
        },
        "july_oos": {
            "baseline": account_metrics(jul, label="jul_all", position_frac=args.position_frac),
            "tradeable_only": policy_metrics(
                apply_gates(jul, threshold=thr, use_day_halt=False),
                "tradeable_passed",
                "jul_tradeable",
                args.position_frac,
            ),
            "day_halt_only": policy_metrics(
                jul.assign(no_trade_passed=~jul["day_halt"].astype(bool)),
                "no_trade_passed",
                "jul_day_halt",
                args.position_frac,
            ),
            "combo": policy_metrics(jul, "no_trade_passed", "jul_combo", args.position_frac),
            "auc": safe_auc(jul["tradeable"], jul["tradeable_probability"]),
            "daily": _daily_gate_table(jul),
        },
    }
    if not expand.empty:
        summary["expanding_confirm_shadow"] = {
            "baseline": account_metrics(expand, label="expand_all", position_frac=args.position_frac),
            "combo": policy_metrics(expand, "no_trade_passed", "expand_combo", args.position_frac),
        }

    train_pool.to_parquet(out_dir / "train_pool_labeled.parquet", index=False)
    apr_jun.to_parquet(out_dir / "apr_jun_confirm_scored.parquet", index=False)
    jul.to_parquet(out_dir / "july_oos_scored.parquet", index=False)
    if not expand.empty:
        expand.to_parquet(out_dir / "expanding_confirm_scored.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "threshold": thr,
        "jun": summary["validation"]["no_trade_combo"],
        "apr_jun_combo": summary["apr_jun_confirm_shadow"]["combo"],
        "july_baseline": summary["july_oos"]["baseline"],
        "july_day_halt": summary["july_oos"]["day_halt_only"],
        "july_tradeable": summary["july_oos"]["tradeable_only"],
        "july_combo": summary["july_oos"]["combo"],
    }, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
