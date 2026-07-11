#!/usr/bin/env python3
"""Feature-based Exit Model for curated QQQ 0DTE State Gate.

Does not change entries.  Along each trade path, decides whether to exit now
or continue to the state clock.

Causal protocol:
  train path-seconds from Apr+May confirm trades
  validate threshold on Jun
  forward OOS on July

Label at path second t (t >= min_hold):
  y = 1 if ret[t] >= ret[clock] + margin   # exiting now beats finishing at clock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from factor_lab.tools.analyze_0dte_state_gate_mfe_exit import (
    build_ticker_index,
    exec_path_returns,
)


FEATURE_COLS = [
    "unrealized_ret",
    "mfe_so_far",
    "mae_so_far",
    "giveback",
    "giveback_frac",
    "ret_from_peak_sign",
    "edge_now",
    "entry_edge",
    "edge_delta",
    "edge_decay_frac",
    "state_on",
    "state_off_streak",
    "spread_pct",
    "hold_progress",
    "hold_s_log",
    "is_lunch_state",
    "is_recovering_state",
    "ret_slope_5s",
    "t_since_mfe",
    "in_profit",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trades",
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
    p.add_argument("--train-months", default="2026-04,2026-05")
    p.add_argument("--val-month", default="2026-06")
    p.add_argument("--sample-every-s", type=int, default=5)
    p.add_argument("--min-hold-s", type=int, default=5)
    p.add_argument("--label-margin", type=float, default=0.005)
    p.add_argument("--default-threshold", type=float, default=0.55)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_state_gate_exit_model_h1",
    )
    return p.parse_args()


def resolve_panel(cache_dirs: list[Path], month: str) -> Path | None:
    for d in cache_dirs:
        fp = d / f"score_dataset_{month}.parquet"
        if fp.exists():
            return fp
    return None


def account_metrics(returns: pd.Series | np.ndarray, *, label: str, position_frac: float) -> dict:
    r = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce").dropna(), dtype=float)
    if len(r) == 0:
        return {
            "label": label,
            "trades": 0,
            "avg_return": 0.0,
            "hit_rate": 0.0,
            "total_return_position": 0.0,
            "max_drawdown_from_initial": 0.0,
        }
    equity = np.cumprod(1.0 + position_frac * r)
    eq0 = np.r_[1.0, equity]
    dd = eq0 / np.maximum.accumulate(eq0) - 1.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    return {
        "label": label,
        "trades": int(len(r)),
        "avg_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total_return_position": float(equity[-1] - 1.0),
        "max_drawdown_from_initial": float(dd.min()),
    }


def _state_series(seg: pd.DataFrame, active_state: str) -> np.ndarray:
    col = str(active_state)
    if col in seg.columns:
        return pd.to_numeric(seg[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # fallback: recovering / lunch names
    if "recovering" in col and "is_qqq_recovering" in seg.columns:
        return pd.to_numeric(seg["is_qqq_recovering"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "lunch" in col and "is_stock_trend_down__and__is_lunch" in seg.columns:
        return pd.to_numeric(
            seg["is_stock_trend_down__and__is_lunch"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
    return np.ones(len(seg), dtype=float)


def build_trade_path_rows(
    trade: pd.Series,
    path: pd.DataFrame,
    *,
    commission: float,
    sample_every_s: int,
    min_hold_s: int,
    label_margin: float,
) -> tuple[list[dict], dict | None]:
    ts = pd.Timestamp(trade["timestamp"])
    tns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
    pos = int(np.searchsorted(tns, ts.value, side="left"))
    if pos >= len(path):
        return [], None
    if abs(int(tns[pos]) - ts.value) > 1_500_000_000:
        exact = np.where(tns == ts.value)[0]
        if len(exact) == 0:
            return [], None
        pos = int(exact[0])

    hold_s = int(pd.to_numeric(trade.get("hold_s"), errors="coerce") or 45)
    end = min(len(path) - 1, pos + hold_s)
    if end <= pos:
        return [], None
    seg = path.iloc[pos : end + 1].reset_index(drop=True)
    entry_ask = float(pd.to_numeric(seg["ask"].iloc[0], errors="coerce"))
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return [], None

    bids = pd.to_numeric(seg["bid"], errors="coerce").to_numpy(dtype=float)
    rets = exec_path_returns(bids, entry_ask, commission)
    edges = (
        pd.to_numeric(seg["tree_edge_score"], errors="coerce").to_numpy(dtype=float)
        if "tree_edge_score" in seg.columns
        else np.full(len(seg), np.nan)
    )
    spreads = (
        pd.to_numeric(seg["spread_pct"], errors="coerce").to_numpy(dtype=float)
        if "spread_pct" in seg.columns
        else np.full(len(seg), np.nan)
    )
    active_state = str(trade.get("active_state", ""))
    state = _state_series(seg, active_state)
    entry_edge = float(edges[0]) if np.isfinite(edges[0]) else 0.0
    clock_i = len(rets) - 1
    clock_ret = float(rets[clock_i]) if np.isfinite(rets[clock_i]) else np.nan
    if not np.isfinite(clock_ret):
        return [], None

    is_lunch = 1.0 if "lunch" in active_state else 0.0
    is_rec = 1.0 if "recovering" in active_state else 0.0

    # running stats
    peak = -np.inf
    trough = np.inf
    peak_t = 0
    off_streak = 0
    rows: list[dict] = []
    for t in range(len(rets)):
        r = rets[t]
        if not np.isfinite(r):
            continue
        if r > peak:
            peak = r
            peak_t = t
        if r < trough:
            trough = r
        if state[t] < 0.5:
            off_streak += 1
        else:
            off_streak = 0

        if t < min_hold_s:
            continue
        if t % sample_every_s != 0 and t != clock_i:
            continue

        giveback = float(peak - r)
        giveback_frac = float(giveback / peak) if peak > 1e-6 else 0.0
        edge_now = float(edges[t]) if np.isfinite(edges[t]) else entry_edge
        edge_delta = edge_now - entry_edge
        denom = max(abs(entry_edge), 1e-4)
        edge_decay_frac = float(-edge_delta / denom)
        slope = float(r - rets[max(0, t - 5)]) if t >= 5 and np.isfinite(rets[max(0, t - 5)]) else 0.0

        y = int(r >= clock_ret + label_margin)
        rows.append(
            {
                "trade_id": f"{trade.get('date_str')}|{trade.get('ticker')}|{ts.value}",
                "month": str(trade.get("month") or pd.Timestamp(ts).strftime("%Y-%m")),
                "date_str": str(trade.get("date_str")),
                "side": str(trade.get("side")),
                "active_state": active_state,
                "hold_s": hold_s,
                "t": int(t),
                "clock_ret": clock_ret,
                "path_exec_ret": float(trade.get("path_exec_ret", clock_ret)),
                "y_exit_beats_clock": y,
                "unrealized_ret": float(r),
                "mfe_so_far": float(peak),
                "mae_so_far": float(trough),
                "giveback": giveback,
                "giveback_frac": giveback_frac,
                "ret_from_peak_sign": float(np.sign(r - peak)),
                "edge_now": edge_now,
                "entry_edge": entry_edge,
                "edge_delta": float(edge_delta),
                "edge_decay_frac": edge_decay_frac,
                "state_on": float(state[t] >= 0.5),
                "state_off_streak": float(off_streak),
                "spread_pct": float(spreads[t]) if np.isfinite(spreads[t]) else 0.0,
                "hold_progress": float(t / max(hold_s, 1)),
                "hold_s_log": float(np.log1p(hold_s)),
                "is_lunch_state": is_lunch,
                "is_recovering_state": is_rec,
                "ret_slope_5s": slope,
                "t_since_mfe": float(t - peak_t),
                "in_profit": float(r > 0),
            }
        )

    meta = {
        "trade_id": f"{trade.get('date_str')}|{trade.get('ticker')}|{ts.value}",
        "month": str(trade.get("month") or pd.Timestamp(ts).strftime("%Y-%m")),
        "date_str": str(trade.get("date_str")),
        "side": str(trade.get("side")),
        "active_state": active_state,
        "hold_s": hold_s,
        "clock_ret": clock_ret,
        "path_exec_ret": float(trade.get("path_exec_ret", clock_ret)),
        "fail_bucket": str(trade.get("fail_bucket", "")),
        "rets": rets,
        "edges": edges,
        "state": state,
        "spreads": spreads,
        "entry_edge": entry_edge,
        "is_lunch": is_lunch,
        "is_rec": is_rec,
    }
    return rows, meta


def collect_dataset(
    trades: pd.DataFrame,
    cache_dirs: list[Path],
    *,
    commission: float,
    sample_every_s: int,
    min_hold_s: int,
    label_margin: float,
) -> tuple[pd.DataFrame, list[dict]]:
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    if "month" not in work.columns:
        work["month"] = work["timestamp"].dt.strftime("%Y-%m")
    if "date_str" not in work.columns:
        work["date_str"] = work["timestamp"].dt.strftime("%Y-%m-%d")

    all_rows: list[dict] = []
    metas: list[dict] = []
    for month, g in work.groupby("month", sort=True):
        fp = resolve_panel(cache_dirs, str(month))
        if fp is None:
            print(f"[exit-model] missing panel {month}", flush=True)
            continue
        print(f"[exit-model] building paths {month} n={len(g)}", flush=True)
        panel = pd.read_parquet(fp)
        index = build_ticker_index(panel)
        for _, tr in g.iterrows():
            path = index.get(str(tr["ticker"]))
            if path is None:
                continue
            rows, meta = build_trade_path_rows(
                tr,
                path,
                commission=commission,
                sample_every_s=sample_every_s,
                min_hold_s=min_hold_s,
                label_margin=label_margin,
            )
            if meta is None:
                continue
            all_rows.extend(rows)
            metas.append(meta)
    return pd.DataFrame(all_rows), metas


def fit_model(train: pd.DataFrame) -> tuple[Pipeline | None, pd.Series, float]:
    x = train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    med = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(med).fillna(0.0)
    y = train["y_exit_beats_clock"].astype(int)
    prior = float(y.mean()) if len(y) else 0.5
    if len(train) < 50 or y.nunique() < 2:
        return None, med, prior
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(C=0.3, penalty="l2", max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(x, y)
    return model, med, prior


def predict_proba(model: Pipeline | None, frame: pd.DataFrame, med: pd.Series, prior: float) -> np.ndarray:
    if frame.empty:
        return np.array([])
    x = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.0)
    if model is None:
        return np.full(len(frame), prior, dtype=float)
    return model.predict_proba(x)[:, 1]


def simulate_policy_on_meta(
    meta: dict,
    model: Pipeline | None,
    med: pd.Series,
    prior: float,
    *,
    threshold: float,
    min_hold_s: int,
    sample_every_s: int,
) -> dict:
    rets = meta["rets"]
    edges = meta["edges"]
    state = meta["state"]
    spreads = meta["spreads"]
    hold_s = int(meta["hold_s"])
    entry_edge = float(meta["entry_edge"])
    clock_i = len(rets) - 1
    clock_ret = float(meta["clock_ret"])

    peak = -np.inf
    trough = np.inf
    peak_t = 0
    off_streak = 0
    exit_t = clock_i
    exit_ret = clock_ret
    fired = False

    for t in range(len(rets)):
        r = rets[t]
        if not np.isfinite(r):
            continue
        if r > peak:
            peak = r
            peak_t = t
        if r < trough:
            trough = r
        if state[t] < 0.5:
            off_streak += 1
        else:
            off_streak = 0
        if t < min_hold_s:
            continue
        if t % sample_every_s != 0 and t != clock_i:
            continue
        if t >= clock_i:
            break

        giveback = float(peak - r)
        giveback_frac = float(giveback / peak) if peak > 1e-6 else 0.0
        edge_now = float(edges[t]) if np.isfinite(edges[t]) else entry_edge
        edge_delta = edge_now - entry_edge
        row = pd.DataFrame(
            [
                {
                    "unrealized_ret": float(r),
                    "mfe_so_far": float(peak),
                    "mae_so_far": float(trough),
                    "giveback": giveback,
                    "giveback_frac": giveback_frac,
                    "ret_from_peak_sign": float(np.sign(r - peak)),
                    "edge_now": edge_now,
                    "entry_edge": entry_edge,
                    "edge_delta": float(edge_delta),
                    "edge_decay_frac": float(-edge_delta / max(abs(entry_edge), 1e-4)),
                    "state_on": float(state[t] >= 0.5),
                    "state_off_streak": float(off_streak),
                    "spread_pct": float(spreads[t]) if np.isfinite(spreads[t]) else 0.0,
                    "hold_progress": float(t / max(hold_s, 1)),
                    "hold_s_log": float(np.log1p(hold_s)),
                    "is_lunch_state": float(meta["is_lunch"]),
                    "is_recovering_state": float(meta["is_rec"]),
                    "ret_slope_5s": float(r - rets[max(0, t - 5)]) if t >= 5 else 0.0,
                    "t_since_mfe": float(t - peak_t),
                    "in_profit": float(r > 0),
                }
            ]
        )
        p = float(predict_proba(model, row, med, prior)[0])
        if p >= threshold:
            exit_t = t
            exit_ret = float(r)
            fired = True
            break

    return {
        "trade_id": meta["trade_id"],
        "month": meta["month"],
        "date_str": meta["date_str"],
        "side": meta["side"],
        "active_state": meta["active_state"],
        "fail_bucket": meta.get("fail_bucket", ""),
        "hold_s": hold_s,
        "clock_ret": clock_ret,
        "model_ret": exit_ret,
        "exit_t": int(exit_t),
        "early_exit": bool(fired),
        "lift_vs_clock": float(exit_ret - clock_ret),
    }


def choose_threshold(
    val_metas: list[dict],
    model: Pipeline | None,
    med: pd.Series,
    prior: float,
    *,
    position_frac: float,
    min_hold_s: int,
    sample_every_s: int,
    default_threshold: float,
) -> tuple[float, dict]:
    grid = [round(x, 2) for x in np.arange(0.45, 0.86, 0.05)] + [1.01]
    best_thr = default_threshold
    best_obj = float("-inf")
    rows = []
    for thr in grid:
        sims = [
            simulate_policy_on_meta(
                m,
                model,
                med,
                prior,
                threshold=thr,
                min_hold_s=min_hold_s,
                sample_every_s=sample_every_s,
            )
            for m in val_metas
        ]
        df = pd.DataFrame(sims)
        if thr > 1.0:
            # force clock-only
            metrics = account_metrics(df["clock_ret"], label="clock", position_frac=position_frac)
            obj = 0.0  # neutral reference
        else:
            metrics = account_metrics(df["model_ret"], label=f"p>={thr}", position_frac=position_frac)
            clock = account_metrics(df["clock_ret"], label="clock", position_frac=position_frac)
            # require not much worse than clock on val; maximize lift - 0.25*extra dd
            lift = metrics["total_return_position"] - clock["total_return_position"]
            dd_pen = max(
                0.0,
                abs(metrics["max_drawdown_from_initial"]) - abs(clock["max_drawdown_from_initial"]),
            )
            obj = lift - 0.25 * dd_pen
            # hard floor: if avg much worse than clock, discard
            if metrics["avg_return"] < clock["avg_return"] - 0.01:
                obj = float("-inf")
        rows.append({"threshold": thr, "objective": obj, **metrics})
        if obj > best_obj:
            best_obj = obj
            best_thr = thr
    if best_obj == float("-inf"):
        best_thr = 1.01  # abstain to clock
        best_obj = 0.0
    return float(best_thr), {"threshold": best_thr, "best_objective": best_obj, "grid": rows}


def rule_baseline_exit(meta: dict, *, min_hold_s: int = 5) -> dict:
    """Interpretable baseline: exit when giveback>=50% of peak after peak>=5% and state off."""
    rets = meta["rets"]
    state = meta["state"]
    clock_i = len(rets) - 1
    clock_ret = float(meta["clock_ret"])
    peak = -np.inf
    off_streak = 0
    exit_t = clock_i
    exit_ret = clock_ret
    fired = False
    for t, r in enumerate(rets):
        if not np.isfinite(r):
            continue
        if r > peak:
            peak = r
        if state[t] < 0.5:
            off_streak += 1
        else:
            off_streak = 0
        if t < min_hold_s:
            continue
        giveback_frac = (peak - r) / peak if peak > 1e-6 else 0.0
        if peak >= 0.05 and giveback_frac >= 0.50 and off_streak >= 3:
            exit_t = t
            exit_ret = float(r)
            fired = True
            break
    return {
        "rule_ret": exit_ret,
        "rule_exit_t": int(exit_t),
        "rule_early_exit": bool(fired),
        "rule_lift_vs_clock": float(exit_ret - clock_ret),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [Path(x.strip()) for x in args.panel_cache_dirs.split(",") if x.strip()]
    train_months = [m.strip() for m in args.train_months.split(",") if m.strip()]
    val_month = args.val_month.strip()

    trades = pd.read_parquet(args.trades)
    trades = trades[trades["month"].isin(["2026-04", "2026-05", "2026-06"])].copy()
    # attach fail buckets if present
    scored = Path("factor_lab/results/0dte_state_gate_no_trade_gate_h1/apr_jun_confirm_scored.parquet")
    if scored.exists():
        sc = pd.read_parquet(scored)[["timestamp", "ticker", "fail_bucket"]]
        sc["timestamp"] = pd.to_datetime(sc["timestamp"])
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades = trades.merge(sc, on=["timestamp", "ticker"], how="left")

    july = pd.read_parquet(args.july_trades).copy()
    july["month"] = "2026-07"
    ja = Path("factor_lab/results/0dte_state_gate_july_w1_failure_attribution/trade_attribution.parquet")
    if ja.exists():
        j = pd.read_parquet(ja)[["timestamp", "ticker", "fail_bucket"]]
        j["timestamp"] = pd.to_datetime(j["timestamp"])
        july["timestamp"] = pd.to_datetime(july["timestamp"])
        july = july.merge(j, on=["timestamp", "ticker"], how="left")

    samples, metas = collect_dataset(
        pd.concat([trades, july], ignore_index=True),
        cache_dirs,
        commission=args.commission_per_contract,
        sample_every_s=args.sample_every_s,
        min_hold_s=args.min_hold_s,
        label_margin=args.label_margin,
    )
    if samples.empty:
        raise SystemExit("no path samples built")
    samples.to_parquet(out_dir / "path_samples.parquet", index=False)

    train = samples[samples["month"].isin(train_months)].copy()
    val = samples[samples["month"].eq(val_month)].copy()
    print(
        f"[exit-model] samples train={len(train)} val={len(val)} "
        f"pos_rate_train={train['y_exit_beats_clock'].mean():.3f}",
        flush=True,
    )
    model, med, prior = fit_model(train)
    if model is not None:
        train_p = predict_proba(model, train, med, prior)
        print(f"[exit-model] train AUC={roc_auc_score(train['y_exit_beats_clock'], train_p):.3f}", flush=True)
        if not val.empty and val["y_exit_beats_clock"].nunique() > 1:
            val_p = predict_proba(model, val, med, prior)
            print(f"[exit-model] val AUC={roc_auc_score(val['y_exit_beats_clock'], val_p):.3f}", flush=True)

    val_metas = [m for m in metas if m["month"] == val_month]
    thr, thr_meta = choose_threshold(
        val_metas,
        model,
        med,
        prior,
        position_frac=args.position_frac,
        min_hold_s=args.min_hold_s,
        sample_every_s=args.sample_every_s,
        default_threshold=args.default_threshold,
    )
    print(f"[exit-model] chosen threshold={thr}", flush=True)

    def eval_metas(ms: list[dict], label: str) -> tuple[pd.DataFrame, dict]:
        sims = []
        for m in ms:
            s = simulate_policy_on_meta(
                m,
                model,
                med,
                prior,
                threshold=thr if thr <= 1.0 else 9.0,  # never fire if abstain
                min_hold_s=args.min_hold_s,
                sample_every_s=args.sample_every_s,
            )
            rb = rule_baseline_exit(m, min_hold_s=args.min_hold_s)
            s.update(rb)
            if thr > 1.0:
                s["model_ret"] = s["clock_ret"]
                s["early_exit"] = False
                s["exit_t"] = int(m["hold_s"])
                s["lift_vs_clock"] = 0.0
            sims.append(s)
        df = pd.DataFrame(sims)
        out = {
            "clock": account_metrics(df["clock_ret"], label=f"{label}_clock", position_frac=args.position_frac),
            "model": account_metrics(df["model_ret"], label=f"{label}_model", position_frac=args.position_frac),
            "rule_giveback_stateoff": account_metrics(
                df["rule_ret"], label=f"{label}_rule", position_frac=args.position_frac
            ),
            "early_exit_rate": float(df["early_exit"].mean()) if len(df) else 0.0,
            "avg_lift_vs_clock": float(df["lift_vs_clock"].mean()) if len(df) else 0.0,
        }
        if "fail_bucket" in df.columns:
            for b in ["C_mfe_but_exit_fail", "F_winner"]:
                sub = df[df["fail_bucket"].eq(b)]
                if sub.empty:
                    continue
                out[f"{b}_clock"] = account_metrics(sub["clock_ret"], label=b, position_frac=args.position_frac)
                out[f"{b}_model"] = account_metrics(sub["model_ret"], label=b, position_frac=args.position_frac)
                out[f"{b}_rule"] = account_metrics(sub["rule_ret"], label=b, position_frac=args.position_frac)
        return df, out

    apr_jun_metas = [m for m in metas if m["month"] in ["2026-04", "2026-05", "2026-06"]]
    july_metas = [m for m in metas if m["month"] == "2026-07"]
    aj_df, aj_sum = eval_metas(apr_jun_metas, "apr_jun")
    jul_df, jul_sum = eval_metas(july_metas, "july")
    aj_df.to_parquet(out_dir / "apr_jun_exit_replay.parquet", index=False)
    jul_df.to_parquet(out_dir / "july_exit_replay.parquet", index=False)

    # coefficients for interpretability
    coef = {}
    if model is not None:
        names = FEATURE_COLS
        vals = model.named_steps["logit"].coef_.ravel()
        coef = {k: float(v) for k, v in sorted(zip(names, vals), key=lambda kv: -abs(kv[1]))}

    decision = {
        "promote_to_default": False,
        "threshold": thr,
        "reason": "",
    }
    # promote only if Jul improves vs clock and Apr-Jun not materially worse
    jul_lift = jul_sum["model"]["total_return_position"] - jul_sum["clock"]["total_return_position"]
    aj_lift = aj_sum["model"]["total_return_position"] - aj_sum["clock"]["total_return_position"]
    if thr <= 1.0 and jul_lift > 0 and aj_lift >= -0.30:
        decision["promote_to_default"] = False  # still shadow until more OOS months
        decision["reason"] = (
            f"model helps Jul (acct lift {jul_lift:.3f}) with limited Apr-Jun damage "
            f"({aj_lift:.3f}); keep as shadow until more OOS months"
        )
    elif thr > 1.0:
        decision["reason"] = "validation preferred clock; model not promoted"
    else:
        decision["reason"] = (
            f"not promoted: jul_lift={jul_lift:.3f}, apr_jun_lift={aj_lift:.3f}"
        )

    summary = {
        "experiment_type": "feature-based exit model vs state clock",
        "config": vars(args),
        "fit": {
            "train_months": train_months,
            "n_path_samples": int(len(train)),
            "positive_rate": float(train["y_exit_beats_clock"].mean()),
            "prior": prior,
            "coefficients": coef,
        },
        "validation": thr_meta,
        "apr_jun": aj_sum,
        "july": jul_sum,
        "decision": decision,
        "files": {
            "path_samples": str(out_dir / "path_samples.parquet"),
            "apr_jun_replay": str(out_dir / "apr_jun_exit_replay.parquet"),
            "july_replay": str(out_dir / "july_exit_replay.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "threshold": thr,
        "apr_jun": {"clock": aj_sum["clock"], "model": aj_sum["model"], "rule": aj_sum["rule_giveback_stateoff"]},
        "july": {"clock": jul_sum["clock"], "model": jul_sum["model"], "rule": jul_sum["rule_giveback_stateoff"]},
        "july_C_model": jul_sum.get("C_mfe_but_exit_fail_model"),
        "july_C_clock": jul_sum.get("C_mfe_but_exit_fail_clock"),
        "decision": decision,
        "top_coef": dict(list(coef.items())[:8]),
    }, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
