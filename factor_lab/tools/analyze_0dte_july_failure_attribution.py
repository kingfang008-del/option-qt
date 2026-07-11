#!/usr/bin/env python3
"""Failure attribution for frozen curated State Gate trades (July W1 OOS).

Classifies each trade into mutually prioritized buckets:
  A_direction_wrong
  B_direction_ok_option_dead
  C_mfe_but_exit_fail
  D_spread_execution
  E_should_no_trade
  F_winner / G_other
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_state_gate_mfe_exit import exec_path_returns
from factor_lab.tools.run_0dte_minimal_five_layer_loop import load_stock_state_features


def load_stock_close_series(stock_root: Path, start: str, end: str, symbol: str = "QQQ") -> pd.DataFrame:
    """Raw 1s close path (not just state features)."""
    frames = []
    for fp in sorted(Path(stock_root).glob(f"{symbol}_*.parquet")):
        d = fp.stem.split("_", 1)[-1]
        if d < start or d > end:
            continue
        df = pd.read_parquet(fp)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        df["date_str"] = d
        frames.append(df[["timestamp", "date_str", "close"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp", "date_str", "close"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def stock_return_over(
    stock: pd.DataFrame,
    ts: pd.Timestamp,
    hold_s: int,
) -> dict[str, float]:
    if stock.empty:
        return {"stock_ret_hold": np.nan, "stock_mfe": np.nan, "stock_mae": np.nan}
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    day = stock[stock["date_str"] == ts.strftime("%Y-%m-%d")]
    if day.empty:
        day = stock
    tns = pd.to_datetime(day["timestamp"]).astype("int64").to_numpy()
    pos = int(np.searchsorted(tns, ts.value, side="left"))
    if pos >= len(day):
        return {"stock_ret_hold": np.nan, "stock_mfe": np.nan, "stock_mae": np.nan}
    end = min(len(day) - 1, pos + int(hold_s))
    px = pd.to_numeric(day["close"].iloc[pos : end + 1], errors="coerce").to_numpy(dtype=float)
    if len(px) < 2 or not np.isfinite(px[0]) or px[0] <= 0:
        return {"stock_ret_hold": np.nan, "stock_mfe": np.nan, "stock_mae": np.nan}
    rets = px / px[0] - 1.0
    return {
        "stock_ret_hold": float(rets[-1]),
        "stock_mfe": float(np.nanmax(rets)),
        "stock_mae": float(np.nanmin(rets)),
    }


def option_path_stats(
    panel: pd.DataFrame,
    trade: pd.Series,
    *,
    commission: float,
) -> dict[str, float]:
    ticker = str(trade["ticker"])
    path = panel[panel["ticker"].astype(str) == ticker].sort_values("timestamp").reset_index(drop=True)
    if path.empty:
        return {}
    ts = pd.Timestamp(trade["timestamp"])
    tns = pd.to_datetime(path["timestamp"]).astype("int64").to_numpy()
    pos = int(np.searchsorted(tns, ts.value, side="left"))
    if pos >= len(path):
        return {}
    if abs(int(tns[pos]) - ts.value) > 1_500_000_000:
        exact = np.where(tns == ts.value)[0]
        if len(exact) == 0:
            return {}
        pos = int(exact[0])
    hold_s = int(trade.get("hold_s", 45))
    end = min(len(path) - 1, pos + hold_s)
    seg = path.iloc[pos : end + 1]
    entry_ask = float(pd.to_numeric(seg["ask"].iloc[0], errors="coerce"))
    entry_bid = float(pd.to_numeric(seg["bid"].iloc[0], errors="coerce"))
    entry_mid = float(pd.to_numeric(seg["mid"].iloc[0], errors="coerce")) if "mid" in seg.columns else np.nan
    if not np.isfinite(entry_ask) or entry_ask <= 0:
        return {}
    bids = pd.to_numeric(seg["bid"], errors="coerce").to_numpy(dtype=float)
    mids = pd.to_numeric(seg["mid"], errors="coerce").to_numpy(dtype=float) if "mid" in seg.columns else np.full(len(seg), np.nan)
    asks = pd.to_numeric(seg["ask"], errors="coerce").to_numpy(dtype=float)
    exec_rets = exec_path_returns(bids, entry_ask, commission)
    cost = 2.0 * commission / (entry_ask * 100.0)
    mid_rets = mids / entry_mid - 1.0 if np.isfinite(entry_mid) and entry_mid > 0 else np.full(len(mids), np.nan)
    valid = np.isfinite(exec_rets)
    if not valid.any():
        return {}
    mfe = float(np.nanmax(exec_rets))
    mae = float(np.nanmin(exec_rets))
    mfe_t = int(np.nanargmax(np.where(valid, exec_rets, -np.inf)))
    final = float(exec_rets[min(hold_s, len(exec_rets) - 1)])
    final_mid = float(mid_rets[min(hold_s, len(mid_rets) - 1)]) if np.isfinite(mid_rets).any() else np.nan
    mfe_mid = float(np.nanmax(mid_rets)) if np.isfinite(mid_rets).any() else np.nan
    spread0 = float((asks[0] - bids[0]) / entry_mid) if np.isfinite(entry_mid) and entry_mid > 0 else np.nan
    spread_end = float((asks[-1] - bids[-1]) / mids[-1]) if np.isfinite(mids[-1]) and mids[-1] > 0 else np.nan
    half_spread_cost = float((entry_ask - entry_bid) / entry_ask) if entry_ask > 0 else np.nan
    profit_hits = np.where(valid & (exec_rets > 0))[0]
    ttp = int(profit_hits[0]) if len(profit_hits) else -1
    return {
        "entry_bid": entry_bid,
        "entry_mid": entry_mid,
        "half_spread_cost": half_spread_cost,
        "spread_pct_entry": spread0,
        "spread_pct_exit": spread_end,
        "mfe": mfe,
        "mae": mae,
        "mfe_t": mfe_t,
        "time_to_profit": ttp,
        "final_exec": final,
        "final_mid": final_mid,
        "mfe_mid": mfe_mid,
        "commission_cost": cost,
        "giveback_from_mfe": float(mfe - final) if np.isfinite(mfe) and np.isfinite(final) else np.nan,
    }


def classify_row(r: pd.Series) -> tuple[str, str]:
    """Return (bucket, reason). Priority: winner → A → C → B → D → E → G."""
    side = str(r["side"]).upper()
    final = float(r["path_exec_ret"])
    stock = float(r.get("stock_ret_hold", np.nan))
    mfe = float(r.get("mfe", np.nan))
    mae = float(r.get("mae", np.nan))
    final_mid = float(r.get("final_mid", np.nan))
    half_spread = float(r.get("half_spread_cost", np.nan))
    eps_stock = 5e-5  # ~0.5bp over hold; tiny moves count as flat

    if final > 0:
        return "F_winner", "final exec return positive"

    # Favorable stock move for long option side
    if side == "CALL":
        stock_fav = np.isfinite(stock) and stock > eps_stock
        stock_against = np.isfinite(stock) and stock < -eps_stock
    else:
        stock_fav = np.isfinite(stock) and stock < -eps_stock
        stock_against = np.isfinite(stock) and stock > eps_stock

    # C: had meaningful MFE but finished red
    if np.isfinite(mfe) and mfe >= 0.03 and final <= 0:
        return "C_mfe_but_exit_fail", f"mfe={mfe:.2%} at t={int(r.get('mfe_t', -1))} but final={final:.2%}"

    # A: underlying moved against
    if stock_against:
        return "A_direction_wrong", f"stock_ret_hold={stock:.4%} against {side}"

    # B: underlying helped (or flat) but option lost
    if stock_fav and final <= 0:
        return "B_direction_ok_option_dead", f"stock_ret_hold={stock:.4%} favored {side} but option={final:.2%}"

    # D: mid path ok / less bad, or spread dominates
    mid_better = np.isfinite(final_mid) and final_mid > 0 and final <= 0
    spread_dominates = np.isfinite(half_spread) and half_spread >= 0.02 and abs(final) <= half_spread + float(r.get("commission_cost", 0) or 0)
    if mid_better:
        return "D_spread_execution", f"mid_final={final_mid:.2%} >0 but exec={final:.2%}"
    if spread_dominates and (not np.isfinite(mfe) or mfe < 0.03):
        return "D_spread_execution", f"half_spread={half_spread:.2%} dominates loss {final:.2%}"

    # E: never tradeable — no profit print and deep MAE
    if (int(r.get("time_to_profit", -1)) < 0) and np.isfinite(mae) and mae <= -0.05 and (not np.isfinite(mfe) or mfe < 0.01):
        return "E_should_no_trade", f"never green; mae={mae:.2%} mfe={mfe:.2%}"

    if np.isfinite(stock) and abs(stock) <= eps_stock and final <= 0:
        return "B_direction_ok_option_dead", f"stock flat ({stock:.4%}) but option={final:.2%}"

    return "G_other", f"final={final:.2%} stock={stock:.4%} mfe={mfe:.2%}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trades",
        default="factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/trades_all.parquet",
    )
    p.add_argument("--panel-cache", default="factor_lab/results/0dte_state_gate_jul_w1_cache/score_dataset_2026-07.parquet")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_state_gate_july_w1_failure_attribution")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_parquet(args.trades).sort_values(["date_str", "timestamp"]).reset_index(drop=True)
    panel = pd.read_parquet(args.panel_cache)
    start, end = str(trades["date_str"].min()), str(trades["date_str"].max())
    stock = load_stock_close_series(Path(args.stock_root), start, end)

    rows = []
    for _, tr in trades.iterrows():
        opt = option_path_stats(panel, tr, commission=args.commission_per_contract)
        stk = stock_return_over(stock, tr["timestamp"], int(tr.get("hold_s", 45)))
        rec = {
            "date_str": tr["date_str"],
            "timestamp": tr["timestamp"],
            "side": tr["side"],
            "ticker": tr["ticker"],
            "active_state": tr["active_state"],
            "hold_s": int(tr["hold_s"]),
            "entry_ask": float(tr["entry_ask"]),
            "exit_bid": float(tr["exit_bid"]),
            "path_exec_ret": float(tr["path_exec_ret"]),
            "spread_pct": float(tr.get("spread_pct", np.nan)),
            "tree_edge_score": float(tr.get("tree_edge_score", np.nan)),
            **opt,
            **stk,
        }
        # signed stock for option holder
        if str(tr["side"]).upper() == "CALL":
            rec["signed_stock_ret"] = rec.get("stock_ret_hold", np.nan)
        else:
            rec["signed_stock_ret"] = (
                -rec["stock_ret_hold"] if np.isfinite(rec.get("stock_ret_hold", np.nan)) else np.nan
            )
        bucket, reason = classify_row(pd.Series(rec))
        rec["fail_bucket"] = bucket
        rec["fail_reason"] = reason
        rows.append(rec)

    diag = pd.DataFrame(rows)
    diag.to_parquet(out_dir / "trade_attribution.parquet", index=False)
    diag.to_csv(out_dir / "trade_attribution.csv", index=False)

    counts = diag["fail_bucket"].value_counts().to_dict()
    by_bucket = {}
    for b, g in diag.groupby("fail_bucket"):
        by_bucket[b] = {
            "n": int(len(g)),
            "avg_return": float(g["path_exec_ret"].mean()),
            "avg_mfe": float(pd.to_numeric(g["mfe"], errors="coerce").mean()),
            "avg_mae": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
            "avg_stock_ret_hold": float(pd.to_numeric(g["stock_ret_hold"], errors="coerce").mean()),
            "avg_signed_stock_ret": float(pd.to_numeric(g["signed_stock_ret"], errors="coerce").mean()),
            "states": g["active_state"].value_counts().to_dict(),
            "sides": g["side"].value_counts().to_dict(),
        }

    # actionable implications
    n = len(diag)
    n_loss = int((diag["path_exec_ret"] <= 0).sum())
    implications = []
    a = counts.get("A_direction_wrong", 0)
    b = counts.get("B_direction_ok_option_dead", 0)
    c = counts.get("C_mfe_but_exit_fail", 0)
    d = counts.get("D_spread_execution", 0)
    e = counts.get("E_should_no_trade", 0)
    if a / max(n_loss, 1) >= 0.4:
        implications.append("方向错误占比高 → State/Rule 本身在 7 月初失效，优先 No-Trade / 降权，而非改 exit")
    if c / max(n_loss, 1) >= 0.3:
        implications.append("MFE 未兑现占比高 → 保留信号，优先 Exit / trailing，而非删规则")
    if b / max(n_loss, 1) >= 0.3:
        implications.append("正股方向对但期权不涨 → 检查 DTE/IV/theta/合约选择，考虑 1DTE 或更近 ATM")
    if d / max(n_loss, 1) >= 0.2:
        implications.append("点差/执行吞噬明显 → 收紧 spread 风控与报价质量过滤")
    if e / max(n_loss, 1) >= 0.2:
        implications.append("大量本不该交易 → No-Trade 标签可直接从 never-green + deep MAE 起步")
    if not implications:
        implications.append("失败类型分散 → 先按桶分别处理，避免单一改法")

    summary = {
        "n_trades": n,
        "n_loss": n_loss,
        "bucket_counts": counts,
        "by_bucket": by_bucket,
        "implications": implications,
        "files": {
            "csv": str(out_dir / "trade_attribution.csv"),
            "parquet": str(out_dir / "trade_attribution.parquet"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
