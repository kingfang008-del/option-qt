#!/usr/bin/env python3
"""RightTailScore v0 + MFE Capture attribution for confirmed State Gate trades.

Goal:
  - Measure how much MFE is left on the table after fixed 45s exits
  - Build a simple entry-time RightTailScore from available state/score features
  - Walk-forward (Apr+May fit -> Jun OOS): only leave a runner when score is high

Runner policy (conservative):
  - RightTailScore low  -> 100% exit at 45s
  - RightTailScore high -> 80% exit at 45s, 20% runner to longer hold / trail
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
)
from factor_lab.tools.analyze_0dte_state_gate_confirm_filters import summarize as summarize_rets
from factor_lab.tools.analyze_0dte_state_gate_mfe_exit import (
    build_ticker_index,
    exec_path_returns,
    month_bounds,
)


HOLD_GRID = (45, 60, 90, 120, 180)
MFE_THRESHOLDS = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


def locate_entry(path: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    ts_ns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
    pos = int(np.searchsorted(ts_ns, ts.value, side="left"))
    if pos >= len(path):
        return None
    if abs(int(ts_ns[pos]) - ts.value) > 1_500_000_000:
        exact = np.where(ts_ns == ts.value)[0]
        if len(exact) == 0:
            return None
        pos = int(exact[0])
    return pos


def simulate_right_tail_path(
    path: pd.DataFrame,
    entry_pos: int,
    *,
    commission: float,
    max_hold_s: int,
) -> dict | None:
    if entry_pos < 0 or entry_pos >= len(path):
        return None
    entry_ask = float(pd.to_numeric(path["ask"].iloc[entry_pos], errors="coerce"))
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return None
    end = min(len(path) - 1, entry_pos + max_hold_s)
    if end <= entry_pos:
        return None
    segment = path.iloc[entry_pos : end + 1]
    bids = pd.to_numeric(segment["bid"], errors="coerce").to_numpy(dtype=float)
    rets = exec_path_returns(bids, entry_ask, commission)
    edges = pd.to_numeric(segment.get("tree_edge_score"), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(rets)
    if not valid.any():
        return None

    def at(h: int) -> float:
        if h >= len(rets) or not np.isfinite(rets[h]):
            return float("nan")
        return float(rets[h])

    mfe = float(np.nanmax(rets))
    mae = float(np.nanmin(rets))
    mfe_t = int(np.nanargmax(np.where(valid, rets, -np.inf)))
    ret_45 = at(45)
    capture_45 = float(ret_45 / mfe) if np.isfinite(ret_45) and mfe > 1e-8 else float("nan")
    left_on_table = float(mfe - ret_45) if np.isfinite(ret_45) else float("nan")

    # Simple runner trails from peak after arming at +X
    trail_rets = {}
    for trigger, trail, tag in [(0.05, 0.40, "trail5_40"), (0.08, 0.40, "trail8_40"), (0.10, 0.50, "trail10_50")]:
        peak = -np.inf
        armed = False
        exit_i = min(max_hold_s, len(rets) - 1)
        for i, r in enumerate(rets):
            if not np.isfinite(r):
                continue
            if r > peak:
                peak = r
            if peak >= trigger:
                armed = True
            if armed and peak > 0 and r <= peak * (1.0 - trail) and i > 0:
                exit_i = i
                break
            if i >= max_hold_s:
                exit_i = i
                break
        trail_rets[f"runner_{tag}"] = float(rets[exit_i]) if np.isfinite(rets[exit_i]) else float("nan")
        trail_rets[f"runner_{tag}_t"] = int(exit_i)

    # Edge-decay runner: hold at least 45s, then exit if edge drops hard
    entry_edge = float(edges[0]) if len(edges) and np.isfinite(edges[0]) else float("nan")
    edge_exit = min(max_hold_s, len(rets) - 1)
    if np.isfinite(entry_edge):
        for i in range(45, min(max_hold_s, len(rets) - 1) + 1):
            if not np.isfinite(rets[i]):
                continue
            e = edges[i] if i < len(edges) else np.nan
            if np.isfinite(e) and e < entry_edge - 0.25 * max(abs(entry_edge), 1e-4):
                edge_exit = i
                break
    edge_ret = float(rets[edge_exit]) if np.isfinite(rets[edge_exit]) else float("nan")

    fixed = {f"fixed_{h}s": at(h) for h in HOLD_GRID if h <= max_hold_s}
    labels = {f"mfe_ge_{int(t * 100)}": bool(mfe >= t) for t in MFE_THRESHOLDS}
    # Doc-style right-tail label (scaled to our short horizon): big MFE, controlled MAE
    labels["right_tail_docish"] = bool(mfe >= 0.15 and mae > -0.25)
    labels["right_tail_soft"] = bool(mfe >= 0.08 and mae > -0.20)

    return {
        "entry_ask": entry_ask,
        "entry_edge": entry_edge,
        "mfe": mfe,
        "mae": mae,
        "mfe_t": mfe_t,
        "ret_45s": ret_45,
        "mfe_capture_45s": capture_45,
        "mfe_left_after_45s": left_on_table,
        "runner_edge_decay": edge_ret,
        "runner_edge_decay_t": int(edge_exit),
        **fixed,
        **trail_rets,
        **labels,
    }


FEATURE_SPEC = [
    # (col, higher_better, weight)
    ("vol_score", True, 1.0),
    ("flow_score", True, 1.0),
    ("gamma_score", True, 0.75),
    ("liquidity_score", True, 0.75),
    ("hot_score", True, 0.50),
    ("tree_edge_score", True, 0.75),
    ("is_vol_expansion", True, 1.0),
    ("is_negative_gamma_proxy", True, 1.0),
    ("is_power_hour", True, 0.50),
    ("is_opening", True, 0.50),
    ("stock_abs_ret_60s", True, 0.75),  # jump / range expansion proxy
    ("spread_pct", False, 1.0),  # tighter better
    ("is_range_pin_proxy", False, 1.0),  # pin risk penalty
]


def _rank01(s: pd.Series) -> pd.Series:
    r = s.rank(method="average", pct=True)
    return r.fillna(0.5)


def build_right_tail_score(df: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Rule-based RightTailScore in [0,1], ranks fit on train distribution."""
    parts = []
    weights = []
    for col, higher_better, w in FEATURE_SPEC:
        if col not in df.columns or col not in train.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # binary flags already ~0/1
        if set(pd.to_numeric(train[col], errors="coerce").dropna().unique()).issubset({0.0, 1.0}):
            x = s.fillna(0.0).clip(0, 1)
            if not higher_better:
                x = 1.0 - x
        else:
            # map via train ranks: use train values to define empirical CDF
            tr = pd.to_numeric(train[col], errors="coerce").dropna().sort_values().to_numpy()
            if len(tr) < 5:
                continue
            # percentile of each value within train
            vals = s.to_numpy(dtype=float)
            pct = np.full(len(vals), 0.5)
            ok = np.isfinite(vals)
            pct[ok] = np.searchsorted(tr, vals[ok], side="right") / max(len(tr), 1)
            x = pd.Series(pct, index=df.index)
            if not higher_better:
                x = 1.0 - x
        parts.append(x.astype(float) * float(w))
        weights.append(float(w))
    if not parts:
        return pd.Series(0.5, index=df.index)
    score = sum(parts) / max(sum(weights), 1e-9)
    return score.clip(0.0, 1.0)


def attach_path_metrics(
    trades: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    *,
    commission: float,
    max_hold_s: int,
) -> pd.DataFrame:
    indexes = {m: build_ticker_index(df) for m, df in panels.items()}
    rows = []
    for tr in trades.itertuples(index=False):
        month = getattr(tr, "month", None)
        if month is None:
            month = str(pd.Timestamp(getattr(tr, "timestamp")).strftime("%Y-%m"))
        ticker = str(getattr(tr, "ticker"))
        ts = pd.Timestamp(getattr(tr, "timestamp"))
        path = indexes.get(month, {}).get(ticker)
        if path is None or path.empty:
            continue
        pos = locate_entry(path, ts)
        if pos is None:
            continue
        sim = simulate_right_tail_path(path, pos, commission=commission, max_hold_s=max_hold_s)
        if sim is None:
            continue
        base = {c: getattr(tr, c) for c in trades.columns}
        rows.append({**base, **sim})
    return pd.DataFrame(rows)


def blended_return(core: pd.Series, runner: pd.Series, core_w: float = 0.80) -> pd.Series:
    c = pd.to_numeric(core, errors="coerce")
    r = pd.to_numeric(runner, errors="coerce")
    out = core_w * c + (1.0 - core_w) * r
    # if runner missing, fall back to core
    miss = ~np.isfinite(r) & np.isfinite(c)
    out = out.where(~miss, c)
    return out


def policy_matrix(df: pd.DataFrame, score_col: str, q_thr: float) -> pd.DataFrame:
    high = pd.to_numeric(df[score_col], errors="coerce") >= q_thr
    rows = []
    policies = {
        "always_45s": df["ret_45s"],
        "always_60s": df.get("fixed_60s"),
        "always_90s": df.get("fixed_90s"),
        "always_120s": df.get("fixed_120s"),
        "always_180s": df.get("fixed_180s"),
        "high_rt_120_else_45": np.where(high, df["fixed_120s"], df["ret_45s"]),
        "high_rt_180_else_45": np.where(high, df["fixed_180s"], df["ret_45s"]),
        "high_rt_runner80_120": blended_return(
            df["ret_45s"],
            np.where(high, df["fixed_120s"], df["ret_45s"]),
            0.80,
        ),
        "high_rt_runner80_180": blended_return(
            df["ret_45s"],
            np.where(high, df["fixed_180s"], df["ret_45s"]),
            0.80,
        ),
        "high_rt_runner80_trail8": blended_return(
            df["ret_45s"],
            np.where(high, df["runner_trail8_40"], df["ret_45s"]),
            0.80,
        ),
        "high_rt_runner80_edge": blended_return(
            df["ret_45s"],
            np.where(high, df["runner_edge_decay"], df["ret_45s"]),
            0.80,
        ),
        # only trade when high RT (skip low) — quality filter, not runner
        "only_high_rt_45s": df.loc[high, "ret_45s"] if high.any() else pd.Series(dtype=float),
    }
    days = int(df["date_str"].nunique()) if "date_str" in df.columns else None
    for name, rets in policies.items():
        if rets is None:
            continue
        s = pd.Series(rets).dropna() if not isinstance(rets, pd.Series) else pd.to_numeric(rets, errors="coerce").dropna()
        if name == "only_high_rt_45s":
            n_days = int(df.loc[high, "date_str"].nunique()) if high.any() and "date_str" in df.columns else 0
            rows.append({"policy": name, "n_high": int(high.sum()), **summarize_rets(s, name, n_days)})
        else:
            rows.append(
                {
                    "policy": name,
                    "n_high": int(high.sum()),
                    "high_ratio": float(high.mean()),
                    **summarize_rets(pd.Series(rets, index=df.index), name, days),
                }
            )
    return pd.DataFrame(rows)


def attribution_table(df: pd.DataFrame, score_col: str) -> dict:
    out = {
        "overall": {
            "n": int(len(df)),
            "avg_mfe": float(df["mfe"].mean()),
            "avg_mae": float(df["mae"].mean()),
            "avg_ret_45s": float(df["ret_45s"].mean()),
            "avg_mfe_capture_45s": float(pd.to_numeric(df["mfe_capture_45s"], errors="coerce").dropna().mean()),
            "avg_mfe_left_after_45s": float(pd.to_numeric(df["mfe_left_after_45s"], errors="coerce").dropna().mean()),
            "pct_mfe_ge_8": float((df["mfe"] >= 0.08).mean()),
            "pct_mfe_ge_15": float((df["mfe"] >= 0.15).mean()),
            "pct_right_tail_soft": float(df["right_tail_soft"].mean()),
            "pct_right_tail_docish": float(df["right_tail_docish"].mean()),
            "avg_mfe_t": float(df["mfe_t"].mean()),
            "pct_mfe_after_45s": float((df["mfe_t"] > 45).mean()),
        }
    }
    # score quintiles vs future MFE
    s = pd.to_numeric(df[score_col], errors="coerce")
    try:
        q = pd.qcut(s, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    except ValueError:
        q = pd.Series(["ALL"] * len(df), index=df.index)
    rows = []
    for label, g in df.groupby(q):
        rows.append(
            {
                "bucket": str(label),
                "n": int(len(g)),
                "avg_score": float(pd.to_numeric(g[score_col], errors="coerce").mean()),
                "avg_mfe": float(g["mfe"].mean()),
                "avg_ret_45s": float(g["ret_45s"].mean()),
                "avg_capture_45s": float(pd.to_numeric(g["mfe_capture_45s"], errors="coerce").dropna().mean())
                if g["mfe_capture_45s"].notna().any()
                else float("nan"),
                "pct_mfe_ge_8": float((g["mfe"] >= 0.08).mean()),
                "pct_mfe_ge_15": float((g["mfe"] >= 0.15).mean()),
                "avg_fixed_120s": float(pd.to_numeric(g["fixed_120s"], errors="coerce").mean()),
                "avg_left_after_45s": float(pd.to_numeric(g["mfe_left_after_45s"], errors="coerce").mean()),
            }
        )
    out["score_quintiles"] = rows
    # by state
    by_state = []
    for st, g in df.groupby("active_state"):
        by_state.append(
            {
                "state": st,
                "n": int(len(g)),
                "avg_mfe": float(g["mfe"].mean()),
                "avg_ret_45s": float(g["ret_45s"].mean()),
                "avg_capture_45s": float(pd.to_numeric(g["mfe_capture_45s"], errors="coerce").dropna().mean())
                if g["mfe_capture_45s"].notna().any()
                else float("nan"),
                "pct_mfe_after_45s": float((g["mfe_t"] > 45).mean()),
                "pct_right_tail_soft": float(g["right_tail_soft"].mean()),
            }
        )
    out["by_state"] = by_state
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--cache-dir", default="factor_lab/results/0dte_rule_state_stability_apr_jun/cache")
    p.add_argument(
        "--trades",
        default="factor_lab/results/0dte_state_gate_curated_confirm_apr_jun/trades_all.parquet",
    )
    p.add_argument("--months", default="2026-04,2026-05,2026-06")
    p.add_argument("--fit-months", default="2026-04,2026-05")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--max-hold-s", type=int, default=180)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_right_tail_apr_jun")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    fit_months = [m.strip() for m in args.fit_months.split(",") if m.strip()]

    print("[right-tail] fitting scorers", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    _, weights, model = fit_rule_scorers(fit_data, target)

    panels: dict[str, pd.DataFrame] = {}
    for month in months:
        print(f"[right-tail] loading panel {month}", flush=True)
        start, end = month_bounds(month)
        data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        data = apply_rule_scorers(data, weights, model)
        data, _ = attach_all_states(data)
        panels[month] = data

    trades = pd.read_parquet(args.trades)
    if "month" not in trades.columns:
        trades["month"] = pd.to_datetime(trades["timestamp"]).dt.strftime("%Y-%m")

    print(f"[right-tail] path metrics for {len(trades)} confirmed trades", flush=True)
    diag = attach_path_metrics(
        trades,
        panels,
        commission=args.commission_per_contract,
        max_hold_s=args.max_hold_s,
    )
    if diag.empty:
        raise SystemExit("no diagnosable trades")

    train = diag[diag["month"].isin(fit_months)].copy()
    test = diag[diag["month"].eq("2026-06")].copy()
    diag["right_tail_score"] = build_right_tail_score(diag, train)
    train = diag[diag["month"].isin(fit_months)].copy()
    test = diag[diag["month"].eq("2026-06")].copy()

    # threshold = train median / p67 of RightTailScore
    thr_p50 = float(train["right_tail_score"].quantile(0.50))
    thr_p67 = float(train["right_tail_score"].quantile(0.67))

    attr_all = attribution_table(diag, "right_tail_score")
    attr_train = attribution_table(train, "right_tail_score")
    attr_test = attribution_table(test, "right_tail_score") if not test.empty else {"overall": {"n": 0}}

    pol_train_p50 = policy_matrix(train, "right_tail_score", thr_p50)
    pol_train_p67 = policy_matrix(train, "right_tail_score", thr_p67)
    pol_test_p50 = policy_matrix(test, "right_tail_score", thr_p50) if not test.empty else pd.DataFrame()
    pol_test_p67 = policy_matrix(test, "right_tail_score", thr_p67) if not test.empty else pd.DataFrame()
    pol_all_p50 = policy_matrix(diag, "right_tail_score", thr_p50)
    pol_all_p67 = policy_matrix(diag, "right_tail_score", thr_p67)

    # score vs label IC
    def ic(a: pd.Series, b: pd.Series) -> float:
        x = pd.to_numeric(a, errors="coerce")
        y = pd.to_numeric(b, errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 5:
            return float("nan")
        return float(x[m].corr(y[m], method="spearman"))

    score_quality = {
        "train": {
            "ic_mfe": ic(train["right_tail_score"], train["mfe"]),
            "ic_left_after_45": ic(train["right_tail_score"], train["mfe_left_after_45s"]),
            "ic_right_tail_soft": ic(train["right_tail_score"], train["right_tail_soft"].astype(float)),
            "ic_fixed_120_minus_45": ic(train["right_tail_score"], train["fixed_120s"] - train["ret_45s"]),
        },
        "jun": {
            "ic_mfe": ic(test["right_tail_score"], test["mfe"]) if not test.empty else float("nan"),
            "ic_left_after_45": ic(test["right_tail_score"], test["mfe_left_after_45s"]) if not test.empty else float("nan"),
            "ic_right_tail_soft": ic(test["right_tail_score"], test["right_tail_soft"].astype(float))
            if not test.empty
            else float("nan"),
            "ic_fixed_120_minus_45": ic(test["right_tail_score"], test["fixed_120s"] - test["ret_45s"])
            if not test.empty
            else float("nan"),
        },
    }

    diag.to_parquet(out_dir / "trade_right_tail.parquet", index=False)
    pol_train_p50.to_csv(out_dir / "policies_train_p50.csv", index=False)
    pol_train_p67.to_csv(out_dir / "policies_train_p67.csv", index=False)
    pol_test_p50.to_csv(out_dir / "policies_jun_p50.csv", index=False)
    pol_test_p67.to_csv(out_dir / "policies_jun_p67.csv", index=False)
    pol_all_p50.to_csv(out_dir / "policies_all_p50.csv", index=False)
    pol_all_p67.to_csv(out_dir / "policies_all_p67.csv", index=False)

    # pick best Jun policy among those that stay train-positive vs always_45s
    def pick(train_pol: pd.DataFrame, test_pol: pd.DataFrame) -> dict:
        if train_pol.empty or test_pol.empty:
            return {}
        base_tr = float(train_pol.loc[train_pol["policy"].eq("always_45s"), "avg_return"].iloc[0])
        base_te = float(test_pol.loc[test_pol["policy"].eq("always_45s"), "avg_return"].iloc[0])
        merged = train_pol.merge(test_pol, on="policy", suffixes=("_train", "_test"))
        cand = merged[
            merged["policy"].ne("always_45s")
            & (merged["avg_return_train"] >= base_tr - 1e-9)
            & (merged["avg_return_test"] > base_te)
            & (merged["profit_factor_test"] > 1.0)
        ].copy()
        if cand.empty:
            # fallback: best jun among runner policies even if train flat
            cand = merged[merged["policy"].str.contains("high_rt|always_120|always_180")].copy()
        cand = cand.sort_values(["avg_return_test", "total_return_10pct_position_test"], ascending=False)
        return {
            "baseline_train_avg": base_tr,
            "baseline_jun_avg": base_te,
            "best": cand.head(5).to_dict("records") if not cand.empty else [],
        }

    recommendation = {
        "thr_p50": thr_p50,
        "thr_p67": thr_p67,
        "p50": pick(pol_train_p50, pol_test_p50),
        "p67": pick(pol_train_p67, pol_test_p67),
    }

    summary = {
        "config": vars(args),
        "n_trades": int(len(diag)),
        "thresholds": {"right_tail_score_p50": thr_p50, "right_tail_score_p67": thr_p67},
        "score_quality": score_quality,
        "attribution": {"all": attr_all, "train": attr_train, "jun": attr_test},
        "recommendation": recommendation,
        "note": (
            "RightTailScore v0 is a rank-weighted blend of vol/flow/gamma/liquidity/"
            "jump/time features minus spread and pin risk. Runner policies only activate "
            "when score >= train quantile; core 80% still exits at 45s."
        ),
        "files": {
            "trade_right_tail": str(out_dir / "trade_right_tail.parquet"),
            "policies_jun_p50": str(out_dir / "policies_jun_p50.csv"),
            "policies_jun_p67": str(out_dir / "policies_jun_p67.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "n": len(diag),
                "thresholds": summary["thresholds"],
                "score_quality": score_quality,
                "attr_all": attr_all["overall"],
                "attr_jun": attr_test.get("overall", {}),
                "quintiles_all": attr_all["score_quintiles"],
                "jun_policies_p50": pol_test_p50.sort_values("avg_return", ascending=False).head(8).to_dict("records")
                if not pol_test_p50.empty
                else [],
                "jun_policies_p67": pol_test_p67.sort_values("avg_return", ascending=False).head(8).to_dict("records")
                if not pol_test_p67.empty
                else [],
                "recommendation": recommendation,
            },
            indent=2,
            default=str,
        )
    )
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
