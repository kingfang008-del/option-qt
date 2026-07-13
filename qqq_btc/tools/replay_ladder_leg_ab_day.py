#!/usr/bin/env python3
"""Jul1 快速 A/B：同一信号序列下 primary 腿 vs value_score 选腿。

对齐 1dte_ladder_upgrade_architecture.md Phase 2：
  - L1 信号冻结（不换特征约）
  - L2 仅在同侧 ladder 池内换执行腿
  - value_score 禁止使用未来收益（非 oracle）

数据：
  - 信号：FT56 honest Jul1 signals CSV
  - 全链分钟：new_option_data_s3（Polygon minute_aggs）
  - Primary / 8-ladder：jul openwin / w1_8contract map
  - 现货：spnq_train 1m

用法：
  python qqq_btc/tools/replay_ladder_leg_ab_day.py --date 2026-07-01
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
OCC_RE = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")


def _ny(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="coerce")
    return s.dt.tz_convert(NY)


def parse_occ(ticker: str) -> tuple[str, str, float] | None:
    m = OCC_RE.match(str(ticker))
    if not m:
        return None
    yymmdd, cp, strike_raw = m.group(2), m.group(3), m.group(4)
    exp = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    return exp, cp, float(int(strike_raw) / 1000.0)


def load_stock_1m(path: Path, date: str) -> pd.Series:
    df = pd.read_parquet(path)
    df["timestamp"] = _ny(df["timestamp"])
    day = df[df["timestamp"].dt.date == pd.Timestamp(date).date()].copy()
    return day.set_index("timestamp")["close"].sort_index()


def load_fullchain(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    parsed = df["ticker"].map(parse_occ)
    df["expiration"] = [p[0] if p else None for p in parsed]
    df["cp"] = [p[1] if p else None for p in parsed]
    df["strike"] = [p[2] if p else np.nan for p in parsed]
    return df.dropna(subset=["expiration", "cp", "strike"]).sort_values(["ticker", "timestamp"])


def build_open_ladder(
    full: pd.DataFrame,
    spot: float,
    *,
    expiration: str,
    band_pct: float,
    lock_ts: pd.Timestamp,
) -> pd.DataFrame:
    """开盘锁池：指定到期 + spot±band，取 lock 时刻有报价的合约。"""
    lo, hi = spot * (1.0 - band_pct), spot * (1.0 + band_pct)
    cand = full[(full["expiration"] == expiration) & (full["strike"] >= lo) & (full["strike"] <= hi)].copy()
    # 09:30–lock 内至少一根 bar
    early = cand[cand["timestamp"] <= lock_ts]
    tickers = sorted(early["ticker"].unique())
    rows = []
    for t in tickers:
        p = parse_occ(t)
        if not p:
            continue
        exp, cp, strike = p
        rows.append(
            {
                "ticker": t,
                "expiration": exp,
                "side": "PUT" if cp == "P" else "CALL",
                "strike": strike,
                "abs_moneyness": abs(np.log(strike / spot)),
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "strike"]).reset_index(drop=True)


def synthetic_quote(row: pd.Series) -> tuple[float, float, float]:
    """分钟 bar 无 bid/ask：用 HL 估点差，close 为 mid。"""
    mid = float(row["close"])
    hi = float(row.get("high", mid) or mid)
    lo = float(row.get("low", mid) or mid)
    hl = max(hi - lo, 0.0)
    # 保守：半宽至少 0.5% mid 或 0.25*HL
    half = max(0.005 * mid, 0.25 * hl, 0.01)
    bid = max(mid - half, 0.01)
    ask = mid + half
    return bid, ask, mid


def fill_buy(bid: float, ask: float, frac: float = 0.775) -> float:
    return bid + frac * (ask - bid)


def fill_sell(bid: float, ask: float, frac: float = 0.775) -> float:
    return ask - frac * (ask - bid)


def path_asof(contract_bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    g = contract_bars[contract_bars["timestamp"] <= ts]
    if g.empty:
        return None
    return g.iloc[-1]


def value_score_row(
    row: pd.Series,
    *,
    spot: float,
    side: str,
    primary_strike: float,
) -> float:
    """入场时 ex-ante score（无未来收益）。"""
    bid, ask, mid = synthetic_quote(row)
    if mid <= 0 or ask < bid:
        return -np.inf
    spread_pct = (ask - bid) / mid
    if spread_pct > 0.15:
        return -np.inf
    premium_pct = mid / max(spot, 1e-6)
    strike = float(row["strike"])
    # 偏好靠近 primary / ATM 一带，惩罚过远行权价与贵权利金、宽点差
    strike_pen = min(abs(strike - primary_strike) / max(spot, 1.0), 0.05) / 0.05
    # PUT 略偏好略虚值到 ATM；CALL 同理用 moneyness 绝对值
    moneyness = np.log(strike / spot)
    if side == "PUT":
        delta_proxy = 1.0 - min(abs(moneyness - 0.0) / 0.02, 1.0)  # near ATM
    else:
        delta_proxy = 1.0 - min(abs(moneyness - 0.0) / 0.02, 1.0)
    vol = float(row.get("volume", 0.0) or 0.0)
    liq = min(np.log1p(vol) / np.log1p(200.0), 1.0)
    score = (
        1.2 * delta_proxy
        + 0.4 * liq
        - 3.0 * spread_pct
        - 0.8 * premium_pct
        - 1.0 * strike_pen
    )
    return float(score)


def replay_leg(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    hard_stop: float = -0.28,
    early_stop_bars: int = 15,
    early_stop_roi: float = -0.12,
    time_stop_bars: int = 30,
    max_hold_bars: int = 45,
    fill_frac: float = 0.775,
) -> dict | None:
    """分钟路径护栏（简化版 1DTE rails）。"""
    entry_row = path_asof(bars, entry_ts)
    if entry_row is None:
        return None
    e_bid, e_ask, _ = synthetic_quote(entry_row)
    entry_px = fill_buy(e_bid, e_ask, fill_frac)
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    # 从 entry 下一分钟起走路径
    fut = bars[bars["timestamp"] > entry_ts].sort_values("timestamp").head(max_hold_bars)
    if fut.empty:
        return None

    max_roi = 0.0
    exit_reason = "MAX_HOLD"
    exit_px = entry_px
    exit_ts = entry_ts
    hold = 0
    for i, (_, r) in enumerate(fut.iterrows(), start=1):
        hold = i
        b, a, _ = synthetic_quote(r)
        mark = fill_sell(b, a, fill_frac)
        if not np.isfinite(mark) or mark <= 0:
            continue
        roi = mark / entry_px - 1.0
        max_roi = max(max_roi, roi)
        exit_px, exit_ts = mark, r["timestamp"]
        if roi <= hard_stop:
            exit_reason = "HARD_STOP"
            break
        if i >= early_stop_bars and roi <= early_stop_roi:
            exit_reason = "EARLY_STOP"
            break
        if i >= time_stop_bars:
            exit_reason = "TIME_STOP"
            break
        if i >= max_hold_bars:
            exit_reason = "MAX_HOLD"
            break

    net = exit_px / entry_px - 1.0
    # 往返佣金拖累（约）
    net -= 2.0 * 0.65 / (entry_px * 100.0)
    return {
        "entry_px": float(entry_px),
        "exit_px": float(exit_px),
        "exit_ts": str(exit_ts),
        "hold_bars": int(hold),
        "exit_reason": exit_reason,
        "net_return": float(net),
        "max_roi": float(max_roi),
        "entry_spread_pct": float((e_ask - e_bid) / max(0.5 * (e_bid + e_ask), 1e-6)),
    }


def pick_value_leg(
    pool: pd.DataFrame,
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    side: str,
    entry_ts: pd.Timestamp,
    spot: float,
    primary_strike: float,
    max_abs_strike_diff: float,
) -> tuple[str | None, float]:
    side_pool = pool[pool["side"] == side].copy()
    side_pool = side_pool[np.abs(side_pool["strike"] - primary_strike) <= max_abs_strike_diff]
    best_t, best_s = None, -np.inf
    for t, strike in side_pool[["ticker", "strike"]].itertuples(index=False):
        bars = bars_by_ticker.get(t)
        if bars is None:
            continue
        row = path_asof(bars, entry_ts)
        if row is None:
            continue
        row = row.copy()
        row["strike"] = strike
        s = value_score_row(row, spot=spot, side=side, primary_strike=primary_strike)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, float(best_s) if best_t else -np.inf


def load_signals(path: Path, date: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df[df["timestamp"].dt.date == pd.Timestamp(date).date()].copy()
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-07-01")
    ap.add_argument(
        "--signals",
        default="qqq_btc/results/july_w1_ft56_honest_3gate_smoke7/signals_2026-07-01.csv",
    )
    ap.add_argument(
        "--fullchain",
        default="/home/kingfang007/data/new_option_data_s3/QQQ/QQQ_2026-07-01.parquet",
    )
    ap.add_argument(
        "--primary-map",
        default=str(Path.home() / "train_data/locked_targets_map_1dte_jul2026_openwin.parquet"),
    )
    ap.add_argument(
        "--stock",
        default=str(Path.home() / "train_data/spnq_train/QQQ/2026-07.parquet"),
    )
    ap.add_argument("--band-pct", type=float, default=0.02)
    ap.add_argument("--max-strike-diff", type=float, default=3.0)
    ap.add_argument("--edge-thresh", type=float, default=0.03)
    ap.add_argument(
        "--out-dir",
        default="qqq_btc/results/july_w1_ladder_leg_ab_20260701",
    )
    args = ap.parse_args()
    date = args.date
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stock = load_stock_1m(Path(args.stock), date)
    lock_ts = pd.Timestamp(f"{date} 09:40:00", tz=NY)
    if lock_ts not in stock.index:
        # nearest
        lock_ts = stock.index[stock.index.get_indexer([lock_ts], method="nearest")[0]]
    spot_lock = float(stock.loc[lock_ts])

    primary_map = pd.read_parquet(args.primary_map)
    primary_day = primary_map[primary_map["date_str"].astype(str) == date].copy()
    if primary_day.empty:
        raise SystemExit(f"no primary contracts for {date} in {args.primary_map}")
    primary_put = primary_day.loc[primary_day["bucket_id"] == 0, "contract_symbol"].iloc[0]
    primary_call = primary_day.loc[primary_day["bucket_id"] == 2, "contract_symbol"].iloc[0]
    p_put = parse_occ(primary_put)
    p_call = parse_occ(primary_call)
    assert p_put and p_call
    expiration = p_put[0]

    full = load_fullchain(Path(args.fullchain))
    ladder = build_open_ladder(
        full, spot_lock, expiration=expiration, band_pct=args.band_pct, lock_ts=lock_ts
    )
    # 确保 primary 在池内
    for t, side in [(primary_put, "PUT"), (primary_call, "CALL")]:
        if t not in set(ladder["ticker"]):
            pp = parse_occ(t)
            if pp:
                ladder = pd.concat(
                    [
                        ladder,
                        pd.DataFrame(
                            [
                                {
                                    "ticker": t,
                                    "expiration": pp[0],
                                    "side": side,
                                    "strike": pp[2],
                                    "abs_moneyness": abs(np.log(pp[2] / spot_lock)),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    tickers = sorted(set(ladder["ticker"]))
    bars_by = {t: full[full["ticker"] == t].copy() for t in tickers}

    sig = load_signals(Path(args.signals), date)
    live_enters = sig[sig["kind"] == "ENTER"].copy()

    # 诊断加厚：edge 过阈的分钟（仍用模型 edge，不换特征）
    edge_rows = sig[
        ((sig["leg"] == "PUT") & (sig["put_edge"] >= args.edge_thresh))
        | ((sig["leg"] == "CALL") & (sig["call_edge"] >= args.edge_thresh))
        | (
            (sig["kind"] == "BLOCK")
            & (
                (sig["put_edge"] >= args.edge_thresh)
                | (sig["call_edge"] >= args.edge_thresh)
            )
        )
    ].copy()
    # 对 BLOCK：取更强一侧
    def _infer_leg(r):
        if pd.notna(r.get("leg")) and str(r["leg"]) in {"PUT", "CALL"}:
            return str(r["leg"])
        pe = float(r.get("put_edge") or -np.inf)
        ce = float(r.get("call_edge") or -np.inf)
        return "PUT" if pe >= ce else "CALL"

    candidates = []
    for _, r in live_enters.iterrows():
        candidates.append(
            {
                "source": "live_enter",
                "timestamp": r["timestamp"],
                "leg": str(r["leg"]),
                "edge": float(r.get("edge") or r.get("put_edge") or r.get("call_edge") or np.nan),
            }
        )
    # denser: minutes where either edge >= thresh, pick stronger side, de-dupe by 15min
    dense = []
    for _, r in sig.iterrows():
        pe = float(r["put_edge"]) if pd.notna(r.get("put_edge")) else -np.inf
        ce = float(r["call_edge"]) if pd.notna(r.get("call_edge")) else -np.inf
        if max(pe, ce) < args.edge_thresh:
            continue
        leg = "PUT" if pe >= ce else "CALL"
        dense.append(
            {
                "source": "edge_dense",
                "timestamp": r["timestamp"],
                "leg": leg,
                "edge": float(max(pe, ce)),
            }
        )
    dense_df = pd.DataFrame(dense)
    if not dense_df.empty:
        dense_df = dense_df.sort_values("timestamp")
        kept = []
        last_ts = None
        for _, r in dense_df.iterrows():
            if last_ts is not None and (r["timestamp"] - last_ts).total_seconds() < 15 * 60:
                continue
            kept.append(r)
            last_ts = r["timestamp"]
        candidates.extend(kept)

    trade_rows = []
    for c in candidates:
        side = c["leg"]
        entry_ts = pd.Timestamp(c["timestamp"]).tz_convert(NY)
        # spot at entry
        sp_idx = stock.index[stock.index.get_indexer([entry_ts], method="ffill")[0]]
        spot = float(stock.loc[sp_idx])
        primary_ticker = primary_put if side == "PUT" else primary_call
        primary_strike = parse_occ(primary_ticker)[2]

        vs_ticker, vs_score = pick_value_leg(
            ladder,
            bars_by,
            side=side,
            entry_ts=entry_ts,
            spot=spot,
            primary_strike=primary_strike,
            max_abs_strike_diff=args.max_strike_diff,
        )
        if vs_ticker is None:
            vs_ticker = primary_ticker
            vs_score = float("nan")

        for mode, ticker in [("primary", primary_ticker), ("value_score", vs_ticker)]:
            bars = bars_by.get(ticker)
            if bars is None:
                # 尝试从 full 拉（primary 可能刚补进 ladder）
                bars = full[full["ticker"] == ticker].copy()
                bars_by[ticker] = bars
            rep = replay_leg(bars, entry_ts) if not bars.empty else None
            trade_rows.append(
                {
                    "source": c["source"],
                    "entry_ts": str(entry_ts),
                    "side": side,
                    "signal_edge": c["edge"],
                    "mode": mode,
                    "ticker": ticker,
                    "strike": parse_occ(ticker)[2] if parse_occ(ticker) else np.nan,
                    "vs_score": vs_score if mode == "value_score" else np.nan,
                    "delta_strike_vs_primary": (
                        abs(parse_occ(ticker)[2] - primary_strike) if parse_occ(ticker) else np.nan
                    ),
                    **(rep or {"net_return": np.nan, "exit_reason": "NO_PATH"}),
                }
            )

    trades = pd.DataFrame(trade_rows)
    trades.to_csv(out_dir / "trades_ab.csv", index=False)
    ladder.to_csv(out_dir / "ladder_pool.csv", index=False)

    def summarize(src: str) -> dict:
        sub = trades[trades["source"] == src]
        if sub.empty:
            return {"source": src, "n_signals": 0}
        piv = sub.pivot_table(
            index=["entry_ts", "side"], columns="mode", values="net_return", aggfunc="first"
        )
        if "primary" not in piv.columns or "value_score" not in piv.columns:
            return {"source": src, "n_signals": int(len(piv)), "note": "missing mode"}
        piv = piv.dropna()
        diff = piv["value_score"] - piv["primary"]
        same = (
            sub[sub["mode"] == "value_score"]["ticker"].values
            == sub[sub["mode"] == "primary"]["ticker"].values
        )
        # align by entry
        vs = sub[sub["mode"] == "value_score"].set_index(["entry_ts", "side"])
        pr = sub[sub["mode"] == "primary"].set_index(["entry_ts", "side"])
        common = vs.index.intersection(pr.index)
        n_same_ticker = int((vs.loc[common, "ticker"].values == pr.loc[common, "ticker"].values).sum())
        return {
            "source": src,
            "n_signals": int(len(piv)),
            "primary_mean_net": float(piv["primary"].mean()),
            "value_score_mean_net": float(piv["value_score"].mean()),
            "uplift_mean": float(diff.mean()),
            "uplift_median": float(diff.median()),
            "pct_vs_better": float((diff > 1e-9).mean()),
            "pct_vs_worse": float((diff < -1e-9).mean()),
            "n_same_ticker": n_same_ticker,
            "n_switched": int(len(common) - n_same_ticker),
            "primary_sum": float(piv["primary"].sum()),
            "value_score_sum": float(piv["value_score"].sum()),
        }

    summary = {
        "date": date,
        "spot_at_lock": spot_lock,
        "lock_ts": str(lock_ts),
        "expiration": expiration,
        "primary_put": primary_put,
        "primary_call": primary_call,
        "ladder_n": int(len(ladder)),
        "ladder_puts": int((ladder["side"] == "PUT").sum()),
        "ladder_calls": int((ladder["side"] == "CALL").sum()),
        "band_pct": args.band_pct,
        "max_strike_diff": args.max_strike_diff,
        "live_enter": summarize("live_enter"),
        "edge_dense": summarize("edge_dense"),
        "note": (
            "分钟全链无真实 bid/ask，点差由 HL 合成；"
            "live_enter=FT56 真实开仓时点；edge_dense=同日 edge≥阈且≥15min 间隔的加厚诊断"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir / 'trades_ab.csv'}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
