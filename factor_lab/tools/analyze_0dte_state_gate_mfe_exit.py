#!/usr/bin/env python3
"""MFE / Edge-decay / timed-exit diagnosis for curated State Gate trades."""
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
from factor_lab.tools.run_0dte_adaptive_rule_pool import adaptive_daily_trades
from factor_lab.tools.run_0dte_state_gate_curated import CURATED_RULES


HORIZONS = (5, 10, 15, 20, 30, 45, 60)


def month_bounds(month: str) -> tuple[str, str]:
    start = f"{month}-01"
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    if month == "2026-04":
        start = "2026-04-13"
    return start, end


def exec_path_returns(bids: np.ndarray, entry_ask: float, commission: float) -> np.ndarray:
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return np.full(len(bids), np.nan)
    cost = 2.0 * commission / (entry_ask * 100.0)
    out = bids / entry_ask - 1.0 - cost
    out[~np.isfinite(bids)] = np.nan
    return out


def simulate_trade_path(
    path: pd.DataFrame,
    entry_pos: int,
    *,
    commission: float,
    max_hold_s: int,
) -> dict | None:
    if entry_pos < 0 or entry_pos >= len(path):
        return None
    entry_ask = float(path["ask"].iloc[entry_pos])
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return None
    end = min(len(path) - 1, entry_pos + max_hold_s)
    if end <= entry_pos:
        return None
    segment = path.iloc[entry_pos : end + 1].copy()
    bids = pd.to_numeric(segment["bid"], errors="coerce").to_numpy(dtype=float)
    rets = exec_path_returns(bids, entry_ask, commission)
    edges = pd.to_numeric(segment.get("tree_edge_score"), errors="coerce").to_numpy(dtype=float)
    recovering = pd.to_numeric(segment.get("is_qqq_recovering"), errors="coerce").fillna(0).to_numpy(dtype=float)
    lunch_down = pd.to_numeric(segment.get("is_stock_trend_down__and__is_lunch"), errors="coerce").fillna(0).to_numpy(dtype=float)
    hold_idx = np.arange(len(rets))
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
    # time to first profit
    profit_hits = np.where(valid & (rets > 0))[0]
    ttp = int(profit_hits[0]) if len(profit_hits) else -1

    # Exit rule helpers
    def first_where(mask: np.ndarray, default: int) -> int:
        hits = np.where(mask)[0]
        return int(hits[0]) if len(hits) else default

    entry_edge = float(edges[0]) if np.isfinite(edges[0]) else np.nan
    fixed = {f"fixed_{h}s": at(h) for h in HORIZONS if h <= max_hold_s}

    # Edge decay: exit when edge falls below frac * entry_edge, else max hold
    decay_exits = {}
    if np.isfinite(entry_edge):
        for frac, tag in [(0.85, "edge085"), (0.70, "edge070"), (0.50, "edge050")]:
            thr = entry_edge * frac if entry_edge > 0 else entry_edge - abs(entry_edge) * (1 - frac)
            # for regression scores, use absolute drop from entry
            drop = edges <= (entry_edge - max(1e-6, abs(entry_edge)) * (1.0 - frac))
            # simpler: relative rank-ish decay using absolute threshold below entry
            drop = edges < (entry_edge - (0.15 if tag == "edge085" else 0.30 if tag == "edge070" else 0.50) * max(abs(entry_edge), 1e-4))
            idx = first_where(drop & (hold_idx > 0) & valid, max_hold_s)
            idx = min(idx, max_hold_s, len(rets) - 1)
            decay_exits[f"exit_{tag}"] = float(rets[idx]) if np.isfinite(rets[idx]) else float("nan")
            decay_exits[f"exit_{tag}_t"] = int(idx)

    # Trailing: lock once MFE >= trigger, exit if giveback from peak exceeds trail
    trail_exits = {}
    for trigger, trail, tag in [(0.02, 0.5, "trail2_50"), (0.03, 0.4, "trail3_40"), (0.05, 0.5, "trail5_50")]:
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
                # giveback of trail fraction of peak (for positive peak)
                if r <= peak - trail * peak:
                    exit_i = i
                    break
            if i >= max_hold_s:
                exit_i = i
                break
        trail_exits[f"exit_{tag}"] = float(rets[exit_i]) if np.isfinite(rets[exit_i]) else float("nan")
        trail_exits[f"exit_{tag}_t"] = int(exit_i)

    # State end: leave when activating state turns off for 3s
    state_exits = {}
    for state_arr, tag, min_hold in [
        (recovering, "state_end_recovering", 5),
        (lunch_down, "state_end_lunch_down", 5),
    ]:
        off = (state_arr < 0.5) & (hold_idx >= min_hold)
        idx = first_where(off & valid, min(max_hold_s, len(rets) - 1))
        state_exits[f"exit_{tag}"] = float(rets[idx]) if np.isfinite(rets[idx]) else float("nan")
        state_exits[f"exit_{tag}_t"] = int(idx)

    # First profit after min hold, else max hold
    first_profit = {}
    for min_h, tag in [(3, "tp_after3"), (5, "tp_after5"), (10, "tp_after10")]:
        hits = np.where(valid & (hold_idx >= min_h) & (rets > 0))[0]
        idx = int(hits[0]) if len(hits) else min(max_hold_s, len(rets) - 1)
        first_profit[f"exit_{tag}"] = float(rets[idx]) if np.isfinite(rets[idx]) else float("nan")
        first_profit[f"exit_{tag}_t"] = int(idx)

    capture_30 = float(at(30) / mfe) if np.isfinite(at(30)) and mfe > 1e-8 else float("nan")
    return {
        "entry_ask": entry_ask,
        "entry_edge": entry_edge,
        "mfe": mfe,
        "mae": mae,
        "mfe_t": mfe_t,
        "time_to_profit": ttp,
        "final_30s": at(30),
        "mfe_capture_30s": capture_30,
        **fixed,
        **decay_exits,
        **trail_exits,
        **state_exits,
        **first_profit,
    }


def build_ticker_index(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker, g in panel.sort_values("timestamp").groupby("ticker", sort=False):
        out[str(ticker)] = g.reset_index(drop=True)
    return out


def diagnose_trades(
    trades: pd.DataFrame,
    panel_by_month: dict[str, pd.DataFrame],
    *,
    commission: float,
    max_hold_s: int,
) -> pd.DataFrame:
    rows = []
    indexes = {m: build_ticker_index(df) for m, df in panel_by_month.items()}
    for tr in trades.itertuples(index=False):
        month = getattr(tr, "month", None)
        if month is None:
            month = str(pd.Timestamp(getattr(tr, "timestamp")).strftime("%Y-%m"))
        ticker = str(getattr(tr, "ticker"))
        ts = pd.Timestamp(getattr(tr, "timestamp"))
        path = indexes.get(month, {}).get(ticker)
        if path is None or path.empty:
            continue
        # locate entry
        ts_ns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
        pos = int(np.searchsorted(ts_ns, ts.value, side="left"))
        if pos >= len(path):
            continue
        # exact or nearest same second
        if abs(int(ts_ns[pos]) - ts.value) > 1_500_000_000:  # >1.5s
            # try exact match
            exact = np.where(ts_ns == ts.value)[0]
            if len(exact) == 0:
                continue
            pos = int(exact[0])
        sim = simulate_trade_path(path, pos, commission=commission, max_hold_s=max_hold_s)
        if sim is None:
            continue
        rows.append(
            {
                "month": month,
                "date_str": getattr(tr, "date_str"),
                "ticker": ticker,
                "side": getattr(tr, "side"),
                "active_state": getattr(tr, "active_state", ""),
                "active_rule": getattr(tr, "active_rule", ""),
                "timestamp": ts,
                **sim,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    out = {
        "n": int(len(df)),
        "avg_mfe": float(df["mfe"].mean()),
        "avg_mae": float(df["mae"].mean()),
        "avg_final_30s": float(df["final_30s"].mean()),
        "avg_mfe_capture_30s": float(df["mfe_capture_30s"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        if df["mfe_capture_30s"].notna().any()
        else 0.0,
        "median_mfe": float(df["mfe"].median()),
        "median_final_30s": float(df["final_30s"].median()),
        "pct_mfe_gt_3pct": float((df["mfe"] >= 0.03).mean()),
        "pct_mfe_gt_5pct": float((df["mfe"] >= 0.05).mean()),
        "pct_ttp_le_10s": float(((df["time_to_profit"] >= 0) & (df["time_to_profit"] <= 10)).mean()),
        "avg_mfe_t": float(df["mfe_t"].mean()),
    }
    exit_cols = [c for c in df.columns if c.startswith("exit_") and not c.endswith("_t") and not c.startswith("exit_edge") or c.startswith("exit_edge")]
    # cleaner: all exit_* without _t, plus fixed_*
    metrics = {}
    for c in sorted([x for x in df.columns if x.startswith("fixed_") or (x.startswith("exit_") and not x.endswith("_t"))]):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        gains = s[s > 0].sum()
        losses = -s[s < 0].sum()
        metrics[c] = {
            "avg_return": float(s.mean()),
            "hit_rate": float((s > 0).mean()),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "median_return": float(s.median()),
        }
    out["exits"] = metrics
    # best exits
    ranked = sorted(metrics.items(), key=lambda kv: (kv[1]["avg_return"], kv[1]["profit_factor"]), reverse=True)
    out["best_exits"] = [{"exit": k, **v} for k, v in ranked[:12]]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--fit-start", default="2026-04-13")
    p.add_argument("--fit-end", default="2026-04-30")
    p.add_argument("--cache-dir", default="factor_lab/results/0dte_rule_state_stability_apr_jun/cache")
    p.add_argument("--trades", default="factor_lab/results/0dte_state_gate_curated_apr_jun/trades_all.parquet")
    p.add_argument("--months", default="2026-04,2026-05,2026-06")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--max-hold-s", type=int, default=60)
    p.add_argument("--daily-topk", type=int, default=2)
    p.add_argument("--rebuild-trades", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_mfe_diag")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = f"target_exec_ret_{args.horizon_s}s"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    print("[mfe] fitting scorers", flush=True)
    fit_data, thresholds = load_fit_period(args, target)
    _, weights, model = fit_rule_scorers(fit_data, target)

    panels = {}
    for month in months:
        print(f"[mfe] loading panel {month}", flush=True)
        start, end = month_bounds(month)
        data = load_or_build_month(args, month, start, end, target, thresholds, cache_dir)
        data = apply_rule_scorers(data, weights, model)
        data, _ = attach_all_states(data)
        panels[month] = data

    trades_path = Path(args.trades)
    if args.rebuild_trades or not trades_path.exists():
        print("[mfe] rebuilding curated trades", flush=True)
        frames = []
        for month, panel in panels.items():
            t = adaptive_daily_trades(panel, CURATED_RULES, target, cooldown_s=30, daily_topk=args.daily_topk)
            if not t.empty:
                t = t.copy()
                t["month"] = month
                frames.append(t)
        trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        trades = pd.read_parquet(trades_path)
        if "month" not in trades.columns:
            trades["month"] = pd.to_datetime(trades["timestamp"]).dt.strftime("%Y-%m")

    print(f"[mfe] diagnosing {len(trades)} trades", flush=True)
    diag = diagnose_trades(trades, panels, commission=args.commission_per_contract, max_hold_s=args.max_hold_s)
    diag.to_parquet(out_dir / "trade_mfe_paths.parquet", index=False)

    overall = summarize(diag)
    by_month = {m: summarize(diag[diag["month"].eq(m)]) for m in months}
    by_state = {
        s: summarize(diag[diag["active_state"].eq(s)])
        for s in sorted(diag["active_state"].dropna().unique())
    } if not diag.empty else {}

    summary = {
        "config": vars(args),
        "n_trades_input": int(len(trades)),
        "n_trades_diagnosed": int(len(diag)),
        "overall": overall,
        "by_month": by_month,
        "by_state": by_state,
        "files": {"trade_mfe_paths": str(out_dir / "trade_mfe_paths.parquet")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"overall": overall, "by_month": {k: {"n": v.get("n"), "avg_mfe": v.get("avg_mfe"), "avg_final_30s": v.get("avg_final_30s"), "best": v.get("best_exits", [])[:5]} for k, v in by_month.items()}}, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
