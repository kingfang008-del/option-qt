#!/usr/bin/env python3
"""Confirmation filters for curated State Gate trades (hold=45s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(
    rets: pd.Series,
    label: str,
    n_days: int | None = None,
    *,
    position_frac: float = 0.25,
) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {"label": label, "trades": 0, "position_frac": float(position_frac)}
    eq = (1.0 + float(position_frac) * r).cumprod()
    dd = eq / eq.cummax() - 1.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    total_ret = float(eq.iloc[-1] - 1.0)
    out = {
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
    if n_days is not None:
        out["days"] = int(n_days)
        out["trades_per_day"] = float(len(r) / max(n_days, 1))
    return out


def flag(df: pd.DataFrame, col: str, thr: float = 0.5) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0) > thr


def qflag(df: pd.DataFrame, col: str, q: float, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    # threshold learned later; here just return series for train-fit
    return s


def build_filter_masks(df: pd.DataFrame, train: pd.DataFrame) -> dict[str, pd.Series]:
    """Build candidate confirmation masks. Continuous thresholds fit on train only."""
    masks: dict[str, pd.Series] = {}

    # binary state confirms
    bin_cols = [
        "is_put_flow_exhaustion",
        "is_put_flow_continuation",
        "is_positive_gamma_proxy",
        "is_negative_gamma_proxy",
        "is_high_vol_proxy",
        "is_low_vol_proxy",
        "is_vol_expansion",
        "is_call_trend_proxy",
        "is_put_trend_proxy",
        "is_stock_vwap_extension",
        "is_opening",
        "is_power_hour",
    ]
    for c in bin_cols:
        if c in df.columns:
            masks[c] = flag(df, c)

    # continuous score confirms: keep top half / top tercile by train quantile
    score_cols = [
        ("flow_score", True),
        ("liquidity_score", True),
        ("vol_score", True),
        ("gamma_score", True),
        ("hot_score", True),
        ("score_hot_quote_tight", True),
        ("tree_edge_score", True),
        ("spread_pct", False),  # tighter better
        ("stock_vwap_dev", False),  # for recovering, less extension maybe better; keep both later
    ]
    for col, higher_better in score_cols:
        if col not in df.columns or col not in train.columns:
            continue
        s_train = pd.to_numeric(train[col], errors="coerce").dropna()
        if s_train.empty:
            continue
        for q, tag in [(0.50, "p50"), (0.67, "p67")]:
            thr = float(s_train.quantile(q if higher_better else 1.0 - q))
            s = pd.to_numeric(df[col], errors="coerce")
            if higher_better:
                masks[f"{col}_ge_{tag}"] = s >= thr
            else:
                masks[f"{col}_le_{tag}"] = s <= thr

    # side-aware / state-aware specials
    if "stock_ret_30s" in df.columns:
        s = pd.to_numeric(df["stock_ret_30s"], errors="coerce")
        masks["stock_ret30_pos"] = s > 0
        masks["stock_ret30_neg"] = s < 0
    if "quote_imbalance" in df.columns:
        s = pd.to_numeric(df["quote_imbalance"], errors="coerce")
        thr = float(pd.to_numeric(train["quote_imbalance"], errors="coerce").quantile(0.67))
        masks["quote_imbalance_ge_p67"] = s >= thr
    if "flow_imbalance_5s" in df.columns:
        s = pd.to_numeric(df["flow_imbalance_5s"], errors="coerce")
        thr = float(pd.to_numeric(train["flow_imbalance_5s"], errors="coerce").quantile(0.67))
        masks["flow_imbalance5_ge_p67"] = s >= thr
        thr_lo = float(pd.to_numeric(train["flow_imbalance_5s"], errors="coerce").quantile(0.33))
        masks["flow_imbalance5_le_p33"] = s <= thr_lo

    # combinations tailored to reversal thesis
    if "is_put_flow_exhaustion" in masks and "is_positive_gamma_proxy" in masks:
        masks["exhaust_and_pos_gamma"] = masks["is_put_flow_exhaustion"] & masks["is_positive_gamma_proxy"]
    if "is_put_flow_exhaustion" in masks and "liquidity_score_ge_p50" in masks:
        masks["exhaust_and_liq_p50"] = masks["is_put_flow_exhaustion"] & masks["liquidity_score_ge_p50"]
    if "is_positive_gamma_proxy" in masks and "liquidity_score_ge_p50" in masks:
        masks["pos_gamma_and_liq_p50"] = masks["is_positive_gamma_proxy"] & masks["liquidity_score_ge_p50"]
    if "flow_score_ge_p50" in masks and "liquidity_score_ge_p50" in masks:
        masks["flow_and_liq_p50"] = masks["flow_score_ge_p50"] & masks["liquidity_score_ge_p50"]
    if "hot_score_ge_p50" in masks and "spread_pct_le_p50" in masks:
        masks["hot_and_tight_p50"] = masks["hot_score_ge_p50"] & masks["spread_pct_le_p50"]
    if "is_put_flow_exhaustion" in masks and "stock_ret30_pos" in masks:
        masks["exhaust_and_stock_up"] = masks["is_put_flow_exhaustion"] & masks["stock_ret30_pos"]
    if "is_call_trend_proxy" in masks and "is_put_flow_exhaustion" in masks:
        masks["call_trend_and_exhaust"] = masks["is_call_trend_proxy"] & masks["is_put_flow_exhaustion"]

    return masks


def evaluate_filters(df: pd.DataFrame, masks: dict[str, pd.Series], min_trades: int) -> pd.DataFrame:
    rows = []
    days = int(df["date_str"].nunique()) if "date_str" in df.columns else None
    base = summarize(df["path_exec_ret"], "baseline_none", days)
    rows.append({"filter": "none", "state": "ALL", **base})
    for state, g in df.groupby("active_state"):
        st_days = int(g["date_str"].nunique()) if "date_str" in g.columns else None
        rows.append({"filter": "none", "state": state, **summarize(g["path_exec_ret"], "none", st_days)})

    for name, mask in masks.items():
        # ALL trades with filter
        sub = df.loc[mask.reindex(df.index).fillna(False)]
        if len(sub) >= min_trades:
            rows.append({"filter": name, "state": "ALL", **summarize(sub["path_exec_ret"], name, int(sub["date_str"].nunique()) if "date_str" in sub.columns else None)})
        # per-state
        for state, g in df.groupby("active_state"):
            m = mask.reindex(g.index).fillna(False)
            sub = g.loc[m]
            if len(sub) >= max(5, min_trades // 2):
                rows.append(
                    {
                        "filter": name,
                        "state": state,
                        **summarize(sub["path_exec_ret"], name, int(sub["date_str"].nunique()) if "date_str" in sub.columns else None),
                    }
                )
    return pd.DataFrame(rows)


def select_filters(train_eval: pd.DataFrame, baseline_by_state: dict[str, float], min_trades: int) -> pd.DataFrame:
    """Keep filters that beat baseline avg_return on train with enough trades and PF>1."""
    rows = []
    for state, base_avg in baseline_by_state.items():
        sub = train_eval[train_eval["state"].eq(state) & train_eval["filter"].ne("none")].copy()
        sub = sub[sub["trades"] >= min_trades]
        sub = sub[(sub["avg_return"] > base_avg) & (sub["profit_factor"] > 1.05) & (sub["avg_return"] > 0)]
        sub = sub.sort_values(["avg_return", "profit_factor", "hit_rate"], ascending=False)
        rows.append(sub.head(10))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def apply_selected(df: pd.DataFrame, masks: dict[str, pd.Series], selected: pd.DataFrame) -> pd.DataFrame:
    """Apply best filter per state; if a state has no selected filter, keep all."""
    keep = pd.Series(False, index=df.index)
    used = {}
    for state, g in df.groupby("active_state"):
        cands = selected[selected["state"].eq(state)]
        if cands.empty:
            keep.loc[g.index] = True
            used[state] = "none"
            continue
        best = cands.iloc[0]["filter"]
        used[state] = best
        m = masks[best].reindex(g.index).fillna(False)
        keep.loc[g.index] = m.to_numpy()
    out = df.loc[keep].copy()
    return out, used


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trades",
        default="factor_lab/results/0dte_state_gate_hold45_apr_jun/trades_all.parquet",
    )
    p.add_argument("--min-train-trades", type=int, default=8)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_confirm_filters_apr_jun")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.trades)
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m")

    train = df[df["month"].isin(["2026-04", "2026-05"])].copy()
    test = df[df["month"].eq("2026-06")].copy()

    train_masks = build_filter_masks(train, train)
    # rebuild masks on full/test using train thresholds via same function with train reference
    all_masks = build_filter_masks(df, train)
    test_masks = {k: v.reindex(test.index) for k, v in all_masks.items()}

    train_eval = evaluate_filters(train, train_masks, min_trades=args.min_train_trades)
    test_eval = evaluate_filters(test, test_masks, min_trades=5)

    baseline_by_state = {
        st: float(g["path_exec_ret"].mean())
        for st, g in train.groupby("active_state")
    }
    baseline_by_state["ALL"] = float(train["path_exec_ret"].mean())
    selected = select_filters(train_eval, baseline_by_state, min_trades=args.min_train_trades)

    # Also select top ALL-level filters
    all_sel = train_eval[
        train_eval["state"].eq("ALL")
        & train_eval["filter"].ne("none")
        & (train_eval["trades"] >= args.min_train_trades)
        & (train_eval["avg_return"] > baseline_by_state["ALL"])
        & (train_eval["profit_factor"] > 1.05)
    ].sort_values(["avg_return", "profit_factor"], ascending=False).head(10)

    filtered_train, used_train = apply_selected(train, train_masks, selected[selected["state"].ne("ALL")])
    filtered_test, used_test = apply_selected(test, test_masks, selected[selected["state"].ne("ALL")])

    # single best ALL filter applied globally
    best_all = all_sel.iloc[0]["filter"] if not all_sel.empty else None
    if best_all:
        m_train = train_masks[best_all]
        m_test = test_masks[best_all].fillna(False)
        global_train = train.loc[m_train]
        global_test = test.loc[m_test]
    else:
        global_train = train.iloc[0:0]
        global_test = test.iloc[0:0]

    summary = {
        "baseline": {
            "train": summarize(train["path_exec_ret"], "train_baseline", int(train["date_str"].nunique())),
            "test": summarize(test["path_exec_ret"], "test_baseline", int(test["date_str"].nunique())),
        },
        "selected_per_state_filters": selected.to_dict("records"),
        "selected_all_filters": all_sel.to_dict("records"),
        "applied_filters": {"train": used_train, "test": used_test},
        "per_state_filtered": {
            "train": summarize(filtered_train["path_exec_ret"], "train_per_state_filtered", int(filtered_train["date_str"].nunique()) if len(filtered_train) else 0),
            "test": summarize(filtered_test["path_exec_ret"], "test_per_state_filtered", int(filtered_test["date_str"].nunique()) if len(filtered_test) else 0),
        },
        "best_global_filter": best_all,
        "global_filtered": {
            "train": summarize(global_train["path_exec_ret"], "train_global_filtered", int(global_train["date_str"].nunique()) if len(global_train) else 0),
            "test": summarize(global_test["path_exec_ret"], "test_global_filtered", int(global_test["date_str"].nunique()) if len(global_test) else 0),
        },
        "files": {
            "train_filter_matrix": str(out_dir / "train_filter_matrix.csv"),
            "test_filter_matrix": str(out_dir / "test_filter_matrix.csv"),
            "selected_filters": str(out_dir / "selected_filters.csv"),
        },
    }

    train_eval.to_csv(out_dir / "train_filter_matrix.csv", index=False)
    test_eval.to_csv(out_dir / "test_filter_matrix.csv", index=False)
    selected.to_csv(out_dir / "selected_filters.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # print concise comparison + top train filters that stay positive on test
    print(json.dumps({
        "baseline": summary["baseline"],
        "per_state_filtered": summary["per_state_filtered"],
        "global_filtered": summary["global_filtered"],
        "applied_filters": summary["applied_filters"],
        "best_global_filter": best_all,
    }, indent=2, default=str))

    # OOS survivors: train-selected filters with test avg>0
    survivors = []
    for _, row in selected.iterrows():
        te = test_eval[(test_eval["filter"].eq(row["filter"])) & (test_eval["state"].eq(row["state"]))]
        if te.empty:
            continue
        trow = te.iloc[0]
        survivors.append({
            "state": row["state"],
            "filter": row["filter"],
            "train_avg": row["avg_return"],
            "train_trades": row["trades"],
            "test_avg": trow["avg_return"],
            "test_trades": trow["trades"],
            "test_pf": trow["profit_factor"],
            "test_hit": trow["hit_rate"],
        })
    surv = pd.DataFrame(survivors)
    if not surv.empty:
        surv = surv.sort_values(["test_avg", "train_avg"], ascending=False)
        surv.to_csv(out_dir / "selected_filters_oos.csv", index=False)
        print("\nOOS survivors (train-selected):")
        print(surv.head(15).to_string(index=False))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
