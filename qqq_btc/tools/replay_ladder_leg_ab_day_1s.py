#!/usr/bin/env python3
"""Jul1 秒级 A/B：同一信号下 primary vs value_score（独立脚本，不影响分钟版）。

数据：
  - 信号：FT56 honest Jul1 signals（冻结）
  - 全链 1s：new_option_data_s3_trades（Polygon trades → 1s OHLCV）
  - Primary：locked_targets_map_1dte_jul2026_openwin
  - 现货：优先 raw_1s stocks，否则 spnq 1m

护栏按「分钟轨 × 60」换算到秒（early=15min, time=30min, max=45min）。

用法：
  python qqq_btc/tools/replay_ladder_leg_ab_day_1s.py --date 2026-07-01
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

# 分钟轨 → 秒
HARD_STOP = -0.28
EARLY_STOP_SEC = 15 * 60
EARLY_STOP_ROI = -0.12
TIME_STOP_SEC = 30 * 60
MAX_HOLD_SEC = 45 * 60
FILL_FRAC = 0.775
# 无真实 bid/ask：用 1s HL 估半宽，并设下限（比分钟版更紧）
MIN_HALF_SPREAD_PCT = 0.0025


def parse_occ(ticker: str) -> tuple[str, str, float] | None:
    m = OCC_RE.match(str(ticker))
    if not m:
        return None
    yymmdd, cp, strike_raw = m.group(2), m.group(3), m.group(4)
    exp = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    return exp, cp, float(int(strike_raw) / 1000.0)


def _ny_ts(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.tz_convert(NY)


def load_spot(date: str, stock_1s: Path | None, stock_1m: Path) -> pd.Series:
    if stock_1s is not None and stock_1s.exists():
        df = pd.read_parquet(stock_1s)
        ts_col = "timestamp" if "timestamp" in df.columns else "ts"
        df["timestamp"] = _ny_ts(df[ts_col])
        px = "close" if "close" in df.columns else "price"
        day = df[df["timestamp"].dt.date == pd.Timestamp(date).date()].copy()
        return day.set_index("timestamp")[px].sort_index().astype(float)
    df = pd.read_parquet(stock_1m)
    df["timestamp"] = _ny_ts(df["timestamp"])
    day = df[df["timestamp"].dt.date == pd.Timestamp(date).date()].copy()
    return day.set_index("timestamp")["close"].sort_index().astype(float)


def load_trades_1s(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename = {"c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
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
    lo, hi = spot * (1.0 - band_pct), spot * (1.0 + band_pct)
    cand = full[
        (full["expiration"] == expiration) & (full["strike"] >= lo) & (full["strike"] <= hi)
    ]
    early = cand[cand["timestamp"] <= lock_ts]
    rows = []
    for t in sorted(early["ticker"].unique()):
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
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "strike"]).reset_index(drop=True)


def synthetic_ba(row: pd.Series) -> tuple[float, float, float]:
    mid = float(row["close"])
    hi = float(row.get("high", mid) or mid)
    lo = float(row.get("low", mid) or mid)
    half = max(MIN_HALF_SPREAD_PCT * mid, 0.25 * max(hi - lo, 0.0), 0.01)
    return max(mid - half, 0.01), mid + half, mid


def fill_buy(bid: float, ask: float, frac: float = FILL_FRAC) -> float:
    return bid + frac * (ask - bid)


def fill_sell(bid: float, ask: float, frac: float = FILL_FRAC) -> float:
    return ask - frac * (ask - bid)


def asof_row(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    g = bars[bars["timestamp"] <= ts]
    if g.empty:
        return None
    return g.iloc[-1]


def value_score(
    row: pd.Series,
    *,
    spot: float,
    side: str,
    primary_strike: float,
) -> float:
    bid, ask, mid = synthetic_ba(row)
    if mid <= 0 or ask < bid:
        return -np.inf
    spread_pct = (ask - bid) / mid
    if spread_pct > 0.15:
        return -np.inf
    premium_pct = mid / max(spot, 1e-6)
    strike = float(row["strike"])
    strike_pen = min(abs(strike - primary_strike) / max(spot, 1.0), 0.05) / 0.05
    moneyness = abs(np.log(strike / spot))
    atm_pref = 1.0 - min(moneyness / 0.02, 1.0)
    vol = float(row.get("volume", 0.0) or 0.0)
    liq = min(np.log1p(vol) / np.log1p(50.0), 1.0)
    return float(
        1.2 * atm_pref + 0.4 * liq - 3.0 * spread_pct - 0.8 * premium_pct - 1.0 * strike_pen
    )


def replay_1s(bars: pd.DataFrame, entry_ts: pd.Timestamp) -> dict | None:
    entry = asof_row(bars, entry_ts)
    if entry is None:
        return None
    e_bid, e_ask, _ = synthetic_ba(entry)
    entry_px = fill_buy(e_bid, e_ask)
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None

    fut = bars[bars["timestamp"] > entry_ts].sort_values("timestamp")
    if fut.empty:
        return None
    # 截断到 max hold 窗口
    end_ts = entry_ts + pd.Timedelta(seconds=MAX_HOLD_SEC)
    fut = fut[fut["timestamp"] <= end_ts]
    if fut.empty:
        return None

    max_roi = 0.0
    exit_reason = "MAX_HOLD"
    exit_px = entry_px
    exit_ts = entry_ts
    hold_sec = 0
    for _, r in fut.iterrows():
        hold_sec = int((r["timestamp"] - entry_ts).total_seconds())
        b, a, _ = synthetic_ba(r)
        mark = fill_sell(b, a)
        if not np.isfinite(mark) or mark <= 0:
            continue
        roi = mark / entry_px - 1.0
        max_roi = max(max_roi, roi)
        exit_px, exit_ts = mark, r["timestamp"]
        if roi <= HARD_STOP:
            exit_reason = "HARD_STOP"
            break
        if hold_sec >= EARLY_STOP_SEC and roi <= EARLY_STOP_ROI:
            exit_reason = "EARLY_STOP"
            break
        if hold_sec >= TIME_STOP_SEC:
            exit_reason = "TIME_STOP"
            break
        if hold_sec >= MAX_HOLD_SEC:
            exit_reason = "MAX_HOLD"
            break

    net = exit_px / entry_px - 1.0
    net -= 2.0 * 0.65 / (entry_px * 100.0)
    return {
        "entry_px": float(entry_px),
        "exit_px": float(exit_px),
        "exit_ts": str(exit_ts),
        "hold_sec": int(hold_sec),
        "exit_reason": exit_reason,
        "net_return": float(net),
        "max_roi": float(max_roi),
        "entry_spread_pct": float((e_ask - e_bid) / max(0.5 * (e_bid + e_ask), 1e-6)),
    }


def pick_value_leg(
    pool: pd.DataFrame,
    bars_by: dict[str, pd.DataFrame],
    *,
    side: str,
    entry_ts: pd.Timestamp,
    spot: float,
    primary_strike: float,
    max_abs_strike_diff: float,
) -> tuple[str | None, float]:
    side_pool = pool[pool["side"] == side]
    side_pool = side_pool[np.abs(side_pool["strike"] - primary_strike) <= max_abs_strike_diff]
    best_t, best_s = None, -np.inf
    for t, strike in side_pool[["ticker", "strike"]].itertuples(index=False):
        bars = bars_by.get(t)
        if bars is None or bars.empty:
            continue
        row = asof_row(bars, entry_ts)
        if row is None:
            continue
        row = row.copy()
        row["strike"] = strike
        s = value_score(row, spot=spot, side=side, primary_strike=primary_strike)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, float(best_s) if best_t else -np.inf


def summarize(trades: pd.DataFrame, src: str) -> dict:
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
    vs = sub[sub["mode"] == "value_score"].set_index(["entry_ts", "side"])
    pr = sub[sub["mode"] == "primary"].set_index(["entry_ts", "side"])
    common = vs.index.intersection(pr.index)
    n_same = int((vs.loc[common, "ticker"].to_numpy() == pr.loc[common, "ticker"].to_numpy()).sum())
    return {
        "source": src,
        "n_signals": int(len(piv)),
        "primary_mean_net": float(piv["primary"].mean()),
        "value_score_mean_net": float(piv["value_score"].mean()),
        "uplift_mean": float(diff.mean()),
        "uplift_median": float(diff.median()),
        "pct_vs_better": float((diff > 1e-9).mean()),
        "pct_vs_worse": float((diff < -1e-9).mean()),
        "n_same_ticker": n_same,
        "n_switched": int(len(common) - n_same),
        "primary_sum": float(piv["primary"].sum()),
        "value_score_sum": float(piv["value_score"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="1s trades ladder leg A/B (independent of minute script)")
    ap.add_argument("--date", default="2026-07-01")
    ap.add_argument(
        "--signals",
        default="qqq_btc/results/july_w1_ft56_honest_3gate_smoke7/signals_2026-07-01.csv",
    )
    ap.add_argument(
        "--trades-1s",
        default="/home/kingfang007/data/new_option_data_s3_trades/QQQ/QQQ_2026-07-01.parquet",
    )
    ap.add_argument(
        "--primary-map",
        default=str(Path.home() / "train_data/locked_targets_map_1dte_jul2026_openwin.parquet"),
    )
    ap.add_argument(
        "--stock-1s",
        default="/mnt/s990/data/raw_1s/stocks/QQQ/QQQ_2026-07-01.parquet",
    )
    ap.add_argument(
        "--stock-1m",
        default=str(Path.home() / "train_data/spnq_train/QQQ/2026-07.parquet"),
    )
    ap.add_argument("--band-pct", type=float, default=0.02)
    ap.add_argument("--max-strike-diff", type=float, default=3.0)
    ap.add_argument("--edge-thresh", type=float, default=0.03)
    ap.add_argument(
        "--out-dir",
        default="qqq_btc/results/july_w1_ladder_leg_ab_1s_20260701",
    )
    args = ap.parse_args()
    date = args.date
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stock = load_spot(date, Path(args.stock_1s), Path(args.stock_1m))
    lock_ts = pd.Timestamp(f"{date} 09:40:00", tz=NY)
    lock_idx = stock.index[stock.index.get_indexer([lock_ts], method="nearest")[0]]
    spot_lock = float(stock.loc[lock_idx])

    primary_map = pd.read_parquet(args.primary_map)
    primary_day = primary_map[primary_map["date_str"].astype(str) == date]
    if primary_day.empty:
        raise SystemExit(f"no primary for {date}")
    primary_put = primary_day.loc[primary_day["bucket_id"] == 0, "contract_symbol"].iloc[0]
    primary_call = primary_day.loc[primary_day["bucket_id"] == 2, "contract_symbol"].iloc[0]
    expiration = parse_occ(primary_put)[0]

    full = load_trades_1s(Path(args.trades_1s))
    ladder = build_open_ladder(
        full, spot_lock, expiration=expiration, band_pct=args.band_pct, lock_ts=lock_ts
    )
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
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    tickers = sorted(set(ladder["ticker"]))
    bars_by = {t: full[full["ticker"] == t].copy() for t in tickers}

    sig = pd.read_csv(args.signals)
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True).dt.tz_convert(NY)
    sig = sig[sig["timestamp"].dt.date == pd.Timestamp(date).date()]

    candidates: list[dict] = []
    for _, r in sig[sig["kind"] == "ENTER"].iterrows():
        candidates.append(
            {
                "source": "live_enter",
                "timestamp": r["timestamp"],
                "leg": str(r["leg"]),
                "edge": float(r.get("edge") or np.nan),
            }
        )

    dense = []
    for _, r in sig.iterrows():
        pe = float(r["put_edge"]) if pd.notna(r.get("put_edge")) else -np.inf
        ce = float(r["call_edge"]) if pd.notna(r.get("call_edge")) else -np.inf
        if max(pe, ce) < args.edge_thresh:
            continue
        dense.append(
            {
                "source": "edge_dense",
                "timestamp": r["timestamp"],
                "leg": "PUT" if pe >= ce else "CALL",
                "edge": float(max(pe, ce)),
            }
        )
    if dense:
        dense_df = pd.DataFrame(dense).sort_values("timestamp")
        last = None
        for _, r in dense_df.iterrows():
            if last is not None and (r["timestamp"] - last).total_seconds() < 15 * 60:
                continue
            candidates.append(r.to_dict())
            last = r["timestamp"]

    rows = []
    for c in candidates:
        side = c["leg"]
        entry_ts = pd.Timestamp(c["timestamp"]).tz_convert(NY)
        sp_i = stock.index[stock.index.get_indexer([entry_ts], method="ffill")[0]]
        spot = float(stock.loc[sp_i])
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
            if bars is None or bars.empty:
                bars = full[full["ticker"] == ticker].copy()
                bars_by[ticker] = bars
            rep = replay_1s(bars, entry_ts) if not bars.empty else None
            rows.append(
                {
                    "source": c["source"],
                    "entry_ts": str(entry_ts),
                    "side": side,
                    "signal_edge": c["edge"],
                    "mode": mode,
                    "ticker": ticker,
                    "strike": parse_occ(ticker)[2] if parse_occ(ticker) else np.nan,
                    "vs_score": vs_score if mode == "value_score" else np.nan,
                    "delta_strike_vs_primary": abs(parse_occ(ticker)[2] - primary_strike)
                    if parse_occ(ticker)
                    else np.nan,
                    **(rep or {"net_return": np.nan, "exit_reason": "NO_PATH"}),
                }
            )

    trades = pd.DataFrame(rows)
    trades.to_csv(out_dir / "trades_ab_1s.csv", index=False)
    ladder.to_csv(out_dir / "ladder_pool_1s.csv", index=False)

    summary = {
        "date": date,
        "data": "trades_1s",
        "trades_path": str(args.trades_1s),
        "spot_at_lock": spot_lock,
        "expiration": expiration,
        "primary_put": primary_put,
        "primary_call": primary_call,
        "ladder_n": int(len(ladder)),
        "ladder_puts": int((ladder["side"] == "PUT").sum()),
        "ladder_calls": int((ladder["side"] == "CALL").sum()),
        "rails_sec": {
            "hard_stop": HARD_STOP,
            "early_stop_sec": EARLY_STOP_SEC,
            "early_stop_roi": EARLY_STOP_ROI,
            "time_stop_sec": TIME_STOP_SEC,
            "max_hold_sec": MAX_HOLD_SEC,
        },
        "live_enter": summarize(trades, "live_enter"),
        "edge_dense": summarize(trades, "edge_dense"),
        "note": "独立于分钟版 replay_ladder_leg_ab_day.py；成交用 trades→1s，点差仍由 1s HL 合成",
    }
    (out_dir / "summary_1s.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir / 'trades_ab_1s.csv'}")
    print(f"wrote {out_dir / 'summary_1s.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
