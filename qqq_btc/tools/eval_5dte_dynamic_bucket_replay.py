#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate 5DTE ladder replay with bucket locked at entry time."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_btc.common.exit_rails import PositionState, check_exit
from qqq_btc.qqq import config as qcfg


CALL_BUCKETS = (4, 5, 6, 7)
PUT_BUCKETS = (0, 1, 2, 3)


@dataclass
class OpenPos:
    entry_ts: object
    entry_bar: int
    session_entry_bar: int | None
    leg: str
    bucket_id: int
    entry_price: float
    signal_edge: float
    state: PositionState


@dataclass
class PendingEntry:
    due_bar: int
    leg: str
    bucket_id: int
    edge: float


def _normalize_ts(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s)
    if out.dt.tz is None:
        return out.dt.tz_localize("America/New_York")
    return out.dt.tz_convert("America/New_York")


def _quote_cols(bucket_id: int) -> tuple[str, str]:
    return f"b{bucket_id}_bid", f"b{bucket_id}_ask"


def _mid(row: pd.Series, bucket_id: int) -> float:
    b_col, a_col = _quote_cols(bucket_id)
    b = float(row.get(b_col, np.nan))
    a = float(row.get(a_col, np.nan))
    if b > 0 and a >= b:
        return (b + a) / 2.0
    return float("nan")


def _spread_pct(row: pd.Series, bucket_id: int) -> float:
    b_col, a_col = _quote_cols(bucket_id)
    b = float(row.get(b_col, np.nan))
    a = float(row.get(a_col, np.nan))
    if b > 0 and a >= b:
        mid = (b + a) / 2.0
        return (a - b) / mid if mid > 0 else float("nan")
    return float("nan")


def _entry_fill(row: pd.Series, bucket_id: int) -> float:
    b_col, a_col = _quote_cols(bucket_id)
    return qcfg.FILL_MODEL.entry_fill(row.get(b_col, np.nan), row.get(a_col, np.nan))


def _exit_fill(row: pd.Series, bucket_id: int) -> float:
    b_col, a_col = _quote_cols(bucket_id)
    return qcfg.FILL_MODEL.exit_fill(row.get(b_col, np.nan), row.get(a_col, np.nan))


def attach_all_bucket_quotes(
    df: pd.DataFrame,
    option_root: Path,
    symbol: str,
    tolerance: str = "5min",
) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = _normalize_ts(out["timestamp"])
    tol = pd.Timedelta(tolerance)
    dates = out["timestamp"].dt.strftime("%Y-%m-%d").unique()

    for bucket_id in PUT_BUCKETS + CALL_BUCKETS:
        parts = []
        for day in dates:
            fp = option_root / symbol / f"{symbol}_{day}.parquet"
            if not fp.exists():
                continue
            opt = pd.read_parquet(fp, columns=["timestamp", "bucket_id", "bid", "ask"])
            opt = opt[opt["bucket_id"].astype(int) == int(bucket_id)]
            if opt.empty:
                continue
            opt["timestamp"] = _normalize_ts(opt["timestamp"])
            opt = opt.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            opt = opt.rename(columns={"bid": f"b{bucket_id}_bid", "ask": f"b{bucket_id}_ask"})
            parts.append(opt[["timestamp", f"b{bucket_id}_bid", f"b{bucket_id}_ask"]])
        if not parts:
            continue
        quotes = pd.concat(parts, ignore_index=True).sort_values("timestamp")
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            quotes,
            on="timestamp",
            direction="backward",
            tolerance=tol,
        )
    return out.sort_values("timestamp").reset_index(drop=True)


def _selected_bucket(row: pd.Series, mode: str) -> int:
    if mode == "model_bucket":
        bid = int(row.get("best_bucket_id", -1))
        return bid if 0 <= bid <= 7 else -1
    if mode == "call_fixed6":
        return 6
    if mode == "put_fixed2":
        return 2
    raise ValueError(f"unknown mode={mode}")


def _selected_edge(row: pd.Series, bucket_id: int, edge_mode: str) -> float:
    if edge_mode == "net":
        return float(row.get("net_edge", np.nan))
    if edge_mode == "leg":
        if bucket_id in CALL_BUCKETS:
            return float(row.get("call_net_edge", np.nan))
        if bucket_id in PUT_BUCKETS:
            return float(row.get("put_net_edge", np.nan))
    if edge_mode == "side_prob":
        if bucket_id in CALL_BUCKETS:
            return float(row.get("best_side_call_prob", np.nan) - row.get("best_side_none_prob", 0.0))
        if bucket_id in PUT_BUCKETS:
            return float(row.get("best_side_put_prob", np.nan) - row.get("best_side_none_prob", 0.0))
    return float("nan")


def _summary(trades: list[dict], equity: list[float], position_frac: float) -> dict:
    if not trades:
        return {"trades": 0, "total_net_return": 0.0, "position_frac": position_frac}
    tr = pd.DataFrame(trades)
    rets = tr["net_return"].to_numpy(dtype=float)
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
    losses = rets[rets < 0]
    wins = rets[rets > 0]
    return {
        "trades": int(len(tr)),
        "position_frac": float(position_frac),
        "total_net_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
        "avg_net_return": float(np.mean(rets)),
        "sum_net_return": float(np.sum(rets)),
        "full_size_compound": float(np.prod(1.0 + rets) - 1.0),
        "hit_rate": float(np.mean(rets > 0)),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "max_drawdown_mtm": float(((eq - peak) / peak).min()) if len(eq) else 0.0,
        "avg_bars_held": float(tr["bars_held"].mean()),
        "worst_trade": float(np.min(rets)),
        "exit_reasons": tr["exit_reason"].value_counts().to_dict(),
        "trades_by_leg": tr["leg"].value_counts().to_dict(),
        "trades_by_bucket": {str(k): int(v) for k, v in tr["bucket_id"].value_counts().to_dict().items()},
    }


def run_dynamic_replay(
    df: pd.DataFrame,
    *,
    mode: str,
    edge_mode: str,
    entry_threshold: float,
    entry_quantile: float | None,
    entry_quantile_window: int,
    entry_quantile_min_obs: int,
    max_spread_pct: float,
    long_only: bool,
    position_frac: float,
    max_trades_per_day: int,
    cooldown_bars: int,
) -> tuple[dict, pd.DataFrame]:
    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = _normalize_ts(work["timestamp"])
    if "session_bar" not in work.columns:
        work["session_bar"] = (work["timestamp"].dt.hour * 60 + work["timestamp"].dt.minute) - (9 * 60 + 30)
    work["_day"] = work["timestamp"].dt.strftime("%Y-%m-%d")

    trades: list[dict] = []
    equity_curve: list[float] = []
    edge_buf: list[float] = []
    pos: OpenPos | None = None
    pending: PendingEntry | None = None
    equity = 1.0
    cooldown_until = -1
    trades_today = 0
    day_key = None

    for bar, row in work.iterrows():
        cur_day = str(row["_day"])
        if cur_day != day_key:
            day_key = cur_day
            trades_today = 0
            cooldown_until = -1
            pending = None

        session_bar = int(row["session_bar"]) if np.isfinite(row["session_bar"]) else None

        if pos is not None:
            mtm = _mid(row, pos.bucket_id)
            if np.isfinite(mtm) and mtm > 0:
                reason = check_exit(qcfg.EXIT_RAILS, pos.state, float(mtm), bar, session_bar_index=session_bar)
                if reason is not None:
                    exit_px = _exit_fill(row, pos.bucket_id)
                    if not (np.isfinite(exit_px) and exit_px > 0):
                        exit_px = mtm
                        reason = f"{reason}|NO_QUOTE"
                    net_ret = float(exit_px) / pos.entry_price - 1.0
                    net_ret -= qcfg.FILL_MODEL.commission_return_drag(pos.entry_price)
                    trades.append(
                        {
                            "entry_ts": pos.entry_ts,
                            "exit_ts": row["timestamp"],
                            "leg": pos.leg,
                            "bucket_id": pos.bucket_id,
                            "entry_price": pos.entry_price,
                            "exit_price": float(exit_px),
                            "net_return": float(net_ret),
                            "exit_reason": reason,
                            "bars_held": int(bar - pos.entry_bar),
                            "signal_edge": pos.signal_edge,
                        }
                    )
                    equity *= 1.0 + position_frac * net_ret
                    equity_curve.append(equity)
                    pos = None
                    pending = None
                    cooldown_until = bar + cooldown_bars
                    trades_today += 1
            continue

        if pending is not None and bar >= pending.due_bar:
            sp = _spread_pct(row, pending.bucket_id)
            entry_px = _entry_fill(row, pending.bucket_id)
            if np.isfinite(entry_px) and entry_px > 0 and np.isfinite(sp) and sp <= max_spread_pct:
                pos = OpenPos(
                    entry_ts=row["timestamp"],
                    entry_bar=bar,
                    session_entry_bar=session_bar,
                    leg=pending.leg,
                    bucket_id=pending.bucket_id,
                    entry_price=float(entry_px),
                    signal_edge=float(pending.edge),
                    state=PositionState(entry_price=float(entry_px), entry_bar=bar),
                )
            pending = None
            continue

        if session_bar is None or session_bar < 15 or session_bar > 300:
            continue
        if bar < cooldown_until or trades_today >= max_trades_per_day:
            continue

        bucket_id = _selected_bucket(row, mode)
        if bucket_id not in PUT_BUCKETS + CALL_BUCKETS:
            continue
        if long_only and bucket_id in PUT_BUCKETS:
            continue
        edge = _selected_edge(row, bucket_id, edge_mode)
        if not (np.isfinite(edge) and edge > 0):
            continue
        edge_buf.append(float(edge))
        th = float(entry_threshold)
        if entry_quantile is not None and len(edge_buf) >= entry_quantile_min_obs:
            hist = np.asarray(edge_buf[-entry_quantile_window:], dtype=float)
            hist = hist[np.isfinite(hist)]
            if len(hist) >= entry_quantile_min_obs:
                th = max(th, float(np.quantile(hist, entry_quantile)))
        if edge < th:
            continue
        sp = _spread_pct(row, bucket_id)
        if not (np.isfinite(sp) and sp <= max_spread_pct):
            continue
        leg = "CALL" if bucket_id in CALL_BUCKETS else "PUT"
        pending = PendingEntry(
            due_bar=bar + int(qcfg.REPLAY.entry_delay_bars),
            leg=leg,
            bucket_id=bucket_id,
            edge=float(edge),
        )

    summary = _summary(trades, equity_curve, position_frac)
    summary.update(
        {
            "mode": mode,
            "edge_mode": edge_mode,
            "entry_threshold": entry_threshold,
            "entry_quantile": entry_quantile,
            "max_spread_pct": max_spread_pct,
            "long_only": long_only,
        }
    )
    return summary, pd.DataFrame(trades)


def main() -> None:
    ap = argparse.ArgumentParser(description="5DTE dynamic bucket replay")
    ap.add_argument("--infer", default="/tmp/qqq_btc_test_eval_v4_dte5_ladder_conservative_b6p2/test_infer.parquet")
    ap.add_argument("--option-1m-root", default="/mnt/s990/data/raw_1m/dte5_options")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--output-dir", default="/tmp/qqq_btc_test_eval_v4_dte5_dynamic_bucket")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(Path(args.infer).expanduser())
    enriched = attach_all_bucket_quotes(base, Path(args.option_1m_root).expanduser(), args.symbol)
    enriched_path = out_dir / "test_infer_all_buckets.parquet"
    enriched.to_parquet(enriched_path, index=False)

    grid = []
    best = None
    best_trades = pd.DataFrame()
    for mode in ("model_bucket", "call_fixed6"):
        for edge_mode in ("net", "leg", "side_prob"):
            for th in (0.02, 0.03, 0.05, 0.07):
                for q in (None, 0.8, 0.9, 0.95):
                    summary, trades = run_dynamic_replay(
                        enriched,
                        mode=mode,
                        edge_mode=edge_mode,
                        entry_threshold=th,
                        entry_quantile=q,
                        entry_quantile_window=1500,
                        entry_quantile_min_obs=300,
                        max_spread_pct=0.10,
                        long_only=True,
                        position_frac=0.25,
                        max_trades_per_day=4,
                        cooldown_bars=10,
                    )
                    grid.append(summary)
                    if best is None or summary.get("total_net_return", -999) > best.get("total_net_return", -999):
                        best, best_trades = summary, trades

    grid_df = pd.DataFrame(grid).sort_values("total_net_return", ascending=False)
    grid_df.to_csv(out_dir / "dynamic_bucket_grid.csv", index=False)
    if best is None:
        best = {"trades": 0}
    (out_dir / "dynamic_bucket_best_summary.json").write_text(json.dumps(best, indent=2, default=str))
    best_trades.to_parquet(out_dir / "dynamic_bucket_best_trades.parquet", index=False)
    print(grid_df.head(20).to_string(index=False))
    print(f"best -> {out_dir / 'dynamic_bucket_best_summary.json'}")


if __name__ == "__main__":
    main()
