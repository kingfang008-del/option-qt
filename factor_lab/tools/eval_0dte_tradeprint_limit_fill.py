#!/usr/bin/env python3
"""Conservative limit-fill check for 0DTE trade-print factor gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_tradeprint_factors import load_factor_dataset
from factor_lab.tools.eval_0dte_tradeprint_factor_gates import add_composite_scores, gate_mask, thresholds


GATES = [
    "score_hot_quote95",
    "score_hot_quote_tight95",
    "score_hot_quote_imb95",
    "notional90_quote90_tight",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = add_composite_scores(df).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    for col in ["bid", "ask", "mid", "spread_pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def simulate_limit_fill(
    quotes: pd.DataFrame,
    selected_ids: set[int],
    *,
    improve: float,
    fill_window_s: int,
    horizon_s: int,
    max_exit_lag_s: int,
    commission: float,
) -> pd.DataFrame:
    trades = []
    if not selected_ids:
        return pd.DataFrame()
    selected_ids_arr = np.array(sorted(selected_ids), dtype=np.int64)
    for _, g in quotes[quotes["row_id"].isin(selected_ids_arr)].groupby("ticker", sort=False):
        # This group only has selected rows, so use the full quote group for path lookup.
        ticker = str(g["ticker"].iloc[0])
        full = quotes[quotes["ticker"].eq(ticker)].sort_values("timestamp").reset_index(drop=True)
        row_pos = {int(rid): pos for pos, rid in enumerate(full["row_id"].to_numpy(dtype=np.int64))}
        ts_series = pd.to_datetime(full["timestamp"])
        ts = ts_series.to_numpy()
        ts_ns = ts_series.astype("int64").to_numpy()
        ask = full["ask"].to_numpy(dtype=float)
        bid = full["bid"].to_numpy(dtype=float)
        mid = full["mid"].to_numpy(dtype=float)
        spread = full["spread_pct"].to_numpy(dtype=float)
        meta = full[["row_id", "timestamp", "date_str", "ticker", "side", "gate_score"]].copy()

        for rid in g["row_id"].to_numpy(dtype=np.int64):
            pos = row_pos.get(int(rid))
            if pos is None:
                continue
            signal_ts = pd.Timestamp(ts[pos])
            entry_ask = ask[pos]
            entry_mid = mid[pos]
            entry_bid = bid[pos]
            entry_spread = spread[pos]
            if not np.isfinite(entry_ask) or not np.isfinite(entry_mid) or entry_ask <= 0:
                continue
            limit_price = entry_ask - improve * max(entry_ask - entry_mid, 0.0)
            fill_pos = None
            for j in range(pos + 1, len(full)):
                dt = (pd.Timestamp(ts[j]) - signal_ts).total_seconds()
                if dt > fill_window_s:
                    break
                if dt <= 0:
                    continue
                if np.isfinite(ask[j]) and ask[j] <= limit_price:
                    fill_pos = j
                    break
            if fill_pos is None:
                continue
            fill_ts = pd.Timestamp(ts[fill_pos])
            fill_bid = bid[fill_pos]
            fill_ask = ask[fill_pos]
            fill_mid = mid[fill_pos]
            fill_spread = spread[fill_pos]
            target_exit_ts = fill_ts + pd.Timedelta(seconds=horizon_s)
            exit_pos = int(np.searchsorted(ts_ns, target_exit_ts.value, side="left"))
            if exit_pos >= len(full):
                continue
            exit_lag = (pd.Timestamp(ts[exit_pos]) - target_exit_ts).total_seconds()
            if exit_lag < 0 or exit_lag > max_exit_lag_s:
                continue
            exit_bid = bid[exit_pos]
            exit_mid = mid[exit_pos]
            if not np.isfinite(exit_bid) or exit_bid <= 0 or not np.isfinite(exit_mid) or exit_mid <= 0:
                continue
            post1_ts = fill_ts + pd.Timedelta(seconds=1)
            post1_pos = int(np.searchsorted(ts_ns, post1_ts.value, side="left"))
            post1_bid = np.nan
            post1_mid = np.nan
            post1_lag = np.nan
            if post1_pos < len(full):
                post1_lag = (pd.Timestamp(ts[post1_pos]) - post1_ts).total_seconds()
                if 0 <= post1_lag <= max_exit_lag_s:
                    post1_bid = bid[post1_pos]
                    post1_mid = mid[post1_pos]
            cost = 2.0 * commission / (limit_price * 100.0)
            signal_row = meta.iloc[pos]
            trades.append(
                {
                    "row_id": int(rid),
                    "timestamp": signal_ts,
                    "date_str": str(signal_row["date_str"]),
                    "ticker": str(signal_row["ticker"]),
                    "side": str(signal_row["side"]),
                    "gate_score": float(signal_row["gate_score"]),
                    "limit_price": float(limit_price),
                    "signal_bid": float(entry_bid),
                    "signal_ask": float(entry_ask),
                    "signal_mid": float(entry_mid),
                    "signal_spread_pct": float(entry_spread),
                    "fill_ts": fill_ts,
                    "fill_wait_s": float((fill_ts - signal_ts).total_seconds()),
                    "fill_bid": float(fill_bid),
                    "fill_ask": float(fill_ask),
                    "fill_mid": float(fill_mid),
                    "fill_spread_pct": float(fill_spread),
                    "post1_bid": float(post1_bid),
                    "post1_mid": float(post1_mid),
                    "post1_lag_s": float(post1_lag),
                    "fill_bid_ge_signal_bid": bool(np.isfinite(fill_bid) and np.isfinite(entry_bid) and fill_bid >= entry_bid),
                    "fill_mid_ge_signal_mid": bool(np.isfinite(fill_mid) and fill_mid >= entry_mid),
                    "fill_spread_not_wider": bool(np.isfinite(fill_spread) and np.isfinite(entry_spread) and fill_spread <= entry_spread),
                    "post1_bid_ge_fill_bid": bool(np.isfinite(post1_bid) and np.isfinite(fill_bid) and post1_bid >= fill_bid),
                    "post1_mid_ge_fill_mid": bool(np.isfinite(post1_mid) and np.isfinite(fill_mid) and post1_mid >= fill_mid),
                    "exit_ts": pd.Timestamp(ts[exit_pos]),
                    "exit_lag_s": float(exit_lag),
                    "ret_exit_bid": float(exit_bid / limit_price - 1.0 - cost),
                    "ret_exit_mid": float(exit_mid / limit_price - 1.0 - cost),
                }
            )
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, *, signals: int, dates: int) -> dict:
    out = {
        "signals": int(signals),
        "fills": int(len(trades)),
        "fill_rate": float(len(trades) / signals) if signals else 0.0,
        "dates": int(dates),
    }
    if trades.empty:
        return out
    for col in ["ret_exit_bid", "ret_exit_mid", "fill_wait_s"]:
        s = pd.to_numeric(trades[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        out[f"{col}_mean"] = float(s.mean())
        out[f"{col}_p50"] = float(s.quantile(0.50))
        out[f"{col}_p90"] = float(s.quantile(0.90))
        if col.startswith("ret_"):
            out[f"{col}_pos_rate"] = float((s > 0).mean())
    return out


def adverse_filter_variants(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"all": trades}
    variants = {"all": trades}
    variants["fill_mid_ge_signal_mid"] = trades[trades["fill_mid_ge_signal_mid"]]
    variants["fill_bid_ge_signal_bid"] = trades[trades["fill_bid_ge_signal_bid"]]
    variants["spread_not_wider"] = trades[trades["fill_spread_not_wider"]]
    variants["post1_mid_confirm"] = trades[trades["post1_mid_ge_fill_mid"]]
    variants["post1_bid_confirm"] = trades[trades["post1_bid_ge_fill_bid"]]
    variants["fill_mid_and_spread_ok"] = trades[trades["fill_mid_ge_signal_mid"] & trades["fill_spread_not_wider"]]
    variants["post1_mid_and_spread_ok"] = trades[trades["post1_mid_ge_fill_mid"] & trades["fill_spread_not_wider"]]
    variants["strict_quality"] = trades[
        trades["fill_mid_ge_signal_mid"]
        & trades["fill_spread_not_wider"]
        & trades["post1_mid_ge_fill_mid"]
    ]
    return variants


def daily_topk_ids(selected: pd.DataFrame, k: int, cooldown_s: int) -> set[int]:
    ids: set[int] = set()
    for _, g in selected.sort_values(["date_str", "gate_score"], ascending=[True, False]).groupby("date_str"):
        last_ts = None
        chosen = 0
        for row in g.sort_values("gate_score", ascending=False).itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "timestamp"))
            if last_ts is not None and abs((ts - last_ts).total_seconds()) <= cooldown_s:
                continue
            ids.add(int(getattr(row, "row_id")))
            last_ts = ts
            chosen += 1
            if chosen >= k:
                break
    return ids


def evaluate(
    test: pd.DataFrame,
    th: dict,
    *,
    improvements: tuple[float, ...],
    fill_windows: tuple[int, ...],
    horizon_s: int,
    max_exit_lag_s: int,
    commission: float,
    cooldown_s: int,
    skip_topk: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = test.copy()
    test["gate_score"] = pd.to_numeric(test["score_hot_quote_tight"], errors="coerce").fillna(0.0)
    rows = []
    topk_rows = []
    for side in ["PUT", "CALL"]:
        for gate in GATES:
            mask = gate_mask(test, side, th, gate)
            selected = test[mask].copy()
            if selected.empty:
                continue
            selected_ids = set(selected["row_id"].astype(int).tolist())
            for improve in improvements:
                for win in fill_windows:
                    trades = simulate_limit_fill(
                        test,
                        selected_ids,
                        improve=improve,
                        fill_window_s=win,
                        horizon_s=horizon_s,
                        max_exit_lag_s=max_exit_lag_s,
                        commission=commission,
                    )
                    for filter_name, filtered in adverse_filter_variants(trades).items():
                        rows.append(
                            {
                                "side": side,
                                "gate": gate,
                                "entry_improve": improve,
                                "fill_window_s": win,
                                "adverse_filter": filter_name,
                                **summarize(filtered, signals=len(selected_ids), dates=selected["date_str"].nunique()),
                            }
                        )
                    if skip_topk:
                        continue
                    for k in (1, 2, 3, 5):
                        ids = daily_topk_ids(selected, k, cooldown_s)
                        topk_trades = simulate_limit_fill(
                            test,
                            ids,
                            improve=improve,
                            fill_window_s=win,
                            horizon_s=horizon_s,
                            max_exit_lag_s=max_exit_lag_s,
                            commission=commission,
                        )
                        for filter_name, filtered in adverse_filter_variants(topk_trades).items():
                            topk_rows.append(
                                {
                                    "side": side,
                                    "gate": gate,
                                    "topk_per_day": k,
                                    "entry_improve": improve,
                                    "fill_window_s": win,
                                    "adverse_filter": filter_name,
                                    **summarize(filtered, signals=len(ids), dates=selected["date_str"].nunique()),
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
    p.add_argument("--horizon-s", type=int, default=30)
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--improvements", default="0.25,0.5,0.75,1.0")
    p.add_argument("--fill-windows", default="1,2,3")
    p.add_argument("--max-exit-lag-s", type=int, default=2)
    p.add_argument("--cooldown-s", type=int, default=30)
    p.add_argument("--skip-topk", action="store_true")
    p.add_argument("--output-dir", default="factor_lab/results/0dte_tradeprint_limit_fill")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    improvements = tuple(float(x) for x in args.improvements.split(",") if x.strip())
    fill_windows = tuple(int(x) for x in args.fill_windows.split(",") if x.strip())
    horizons = (args.horizon_s,)

    print("[limit-fill] loading train", flush=True)
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
    print("[limit-fill] loading test", flush=True)
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
    test = prepare_data(test)
    th = thresholds(train, (0.25, 0.50, 0.75, 0.90, 0.95))

    print(f"[limit-fill] train_rows={len(train)} test_rows={len(test)}", flush=True)
    gate_summary, topk_summary = evaluate(
        test,
        th,
        improvements=improvements,
        fill_windows=fill_windows,
        horizon_s=args.horizon_s,
        max_exit_lag_s=args.max_exit_lag_s,
        commission=args.commission_per_contract,
        cooldown_s=args.cooldown_s,
        skip_topk=bool(args.skip_topk),
    )
    gate_summary.to_csv(out_dir / "limit_fill_gate_summary.csv", index=False)
    topk_summary.to_csv(out_dir / "limit_fill_daily_topk_summary.csv", index=False)
    top = gate_summary[gate_summary["fills"] >= 100].sort_values("ret_exit_bid_mean", ascending=False).head(20)
    if topk_summary.empty:
        topk = topk_summary
    else:
        topk = topk_summary[topk_summary["fills"] >= 5].sort_values("ret_exit_bid_mean", ascending=False).head(20)
    summary = {
        "config": vars(args),
        "rows": {"train": int(len(train)), "test": int(len(test))},
        "top_gate_by_filled_bid_return": top.to_dict("records"),
        "top_daily_topk_by_filled_bid_return": topk.to_dict("records"),
        "files": {
            "gate_summary": str(out_dir / "limit_fill_gate_summary.csv"),
            "daily_topk_summary": str(out_dir / "limit_fill_daily_topk_summary.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
