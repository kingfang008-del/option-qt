#!/usr/bin/env python3
"""Validate premarket / after-hours / open-handoff edges on SPX-universe 15s bars.

Data: /mnt/s990/data/all_data/stocks_15s_parquet/{SYMBOL}/{YYYY-MM}.parquet
Sessions (America/New_York):
  premarket 04:00–09:30
  RTH       09:30–16:00
  after     16:00–20:00

Tests (equal-weight, causal day signals; no lookahead within day for entry at open):
  1) gap_continue: sign(premarket gap) hold open→30m / 90m / EOD
  2) gap_fade:     opposite of gap, same holds
  3) pre_catch:    if |gap| already large by 09:00, enter at 09:00 in gap dir, exit 09:35 / 10:00
  4) ah_continue:  sign(AH ret) → next-day open→30m (uses prior AH only)

Liquidity filter: top-N by average RTH dollar volume in the window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NY = "America/New_York"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--top-n", type=int, default=100, help="top liquid names by RTH $volume")
    p.add_argument("--gap-thr", type=float, default=0.005, help="min |premarket gap| to trade")
    p.add_argument("--pre09-thr", type=float, default=0.008, help="|ret by 09:00| to attempt pre catch")
    p.add_argument("--max-symbols-scan", type=int, default=0, help="0=all folders for liquidity rank")
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_premarket_ah_validation",
    )
    return p.parse_args()


def month_range(start: str, end: str) -> list[str]:
    idx = pd.period_range(start=start, end=end, freq="M")
    return [str(p) for p in idx]


def list_symbols(root: Path, limit: int = 0) -> list[str]:
    syms = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if limit > 0:
        syms = syms[:limit]
    return syms


def load_symbol_months(root: Path, symbol: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        fp = root / symbol / f"{m}.parquet"
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        try:
            df = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            print(f"[warn] skip {fp}: {e}", flush=True)
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df.assign(timestamp=ts).sort_values("timestamp")
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["tod"] = df["timestamp"].dt.time
    return df


def session_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    t = df["tod"]
    return {
        "pre": (t >= pd.Timestamp("04:00").time()) & (t < pd.Timestamp("09:30").time()),
        "rth": (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time()),
        "ah": (t >= pd.Timestamp("16:00").time()) & (t < pd.Timestamp("20:00").time()),
    }


def px_at_or_before(day: pd.DataFrame, hhmm: str, col: str = "close") -> float:
    target = pd.Timestamp(hhmm).time()
    sub = day[day["tod"] <= target]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1][col])


def px_at_or_after(day: pd.DataFrame, hhmm: str, col: str = "open") -> float:
    target = pd.Timestamp(hhmm).time()
    sub = day[day["tod"] >= target]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0][col])


def build_day_panel(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    masks = session_masks(df)
    rows = []
    for date_str, day in df.groupby("date_str", sort=True):
        day = day.sort_values("timestamp")
        rth = day.loc[masks["rth"].loc[day.index]]
        pre = day.loc[masks["pre"].loc[day.index]]
        ah = day.loc[masks["ah"].loc[day.index]]
        if rth.empty or len(rth) < 20:
            continue
        rth_open = float(rth.iloc[0]["open"])
        rth_close = float(rth.iloc[-1]["close"])
        pre_last = float(pre.iloc[-1]["close"]) if not pre.empty else float("nan")
        pre_0900 = px_at_or_before(day, "09:00")
        open_0935 = px_at_or_before(day, "09:35")
        open_1000 = px_at_or_before(day, "10:00")
        open_1100 = px_at_or_before(day, "11:00")
        ah_last = float(ah.iloc[-1]["close"]) if not ah.empty else float("nan")
        rth_dollar = float((rth["close"] * rth["volume"]).sum())
        rows.append(
            {
                "symbol": symbol,
                "date_str": date_str,
                "rth_open": rth_open,
                "rth_close": rth_close,
                "pre_last": pre_last,
                "pre_0900": pre_0900,
                "px_0935": open_0935,
                "px_1000": open_1000,
                "px_1100": open_1100,
                "ah_last": ah_last,
                "rth_dollar": rth_dollar,
                "has_pre": int(not pre.empty),
                "has_ah": int(not ah.empty),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("date_str")
    out["prev_rth_close"] = out["rth_close"].shift(1)
    out["gap"] = out["rth_open"] / out["prev_rth_close"] - 1.0
    out["pre_to_open"] = out["rth_open"] / out["pre_last"] - 1.0
    out["ret_0930_0935"] = out["px_0935"] / out["rth_open"] - 1.0
    out["ret_0930_1000"] = out["px_1000"] / out["rth_open"] - 1.0
    out["ret_0930_1100"] = out["px_1100"] / out["rth_open"] - 1.0
    out["ret_rth"] = out["rth_close"] / out["rth_open"] - 1.0
    out["ret_ah"] = out["ah_last"] / out["rth_close"] - 1.0
    out["ret_by_0900"] = out["pre_0900"] / out["prev_rth_close"] - 1.0
    out["ret_0900_0935"] = out["px_0935"] / out["pre_0900"] - 1.0
    out["ret_0900_1000"] = out["px_1000"] / out["pre_0900"] - 1.0
    # next open continuation from AH
    out["next_open"] = out["rth_open"].shift(-1)
    out["next_1000"] = out["px_1000"].shift(-1)
    out["ah_to_next_open"] = out["next_open"] / out["ah_last"] - 1.0
    out["next_open_30m"] = out["next_1000"] / out["next_open"] - 1.0
    return out


def summarize_trades(rets: pd.Series, label: str) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {"label": label, "n": 0}
    # equal-weight day-symbol trades compounded as portfolio of independent bets at 100% is wrong;
    # report trade stats + simple cumulative of mean daily cross-section
    return {
        "label": label,
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "p05": float(r.quantile(0.05)),
        "p95": float(r.quantile(0.95)),
        "sum_ret": float(r.sum()),
    }


def daily_cs_account(trades: pd.DataFrame, ret_col: str, position_frac: float = 0.1) -> dict:
    """Each day equal-weight active names; compound day returns with position_frac on the basket."""
    if trades.empty:
        return {"n_days": 0, "account_ret": 0.0, "max_dd": 0.0, "avg_day": float("nan")}
    day = trades.groupby("date_str")[ret_col].mean().sort_index()
    eq = np.cumprod(1.0 + position_frac * day.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n_days": int(len(day)),
        "avg_day": float(day.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_day_rate": float((day > 0).mean()),
    }


def rank_liquidity(root: Path, symbols: list[str], months: list[str], top_n: int) -> pd.DataFrame:
    # use last 3 months of window for ranking speed
    rank_months = months[-3:] if len(months) >= 3 else months
    rows = []
    for i, sym in enumerate(symbols):
        df = load_symbol_months(root, sym, rank_months)
        if df.empty:
            continue
        m = session_masks(df)
        rth = df.loc[m["rth"]]
        if rth.empty:
            continue
        rows.append({"symbol": sym, "rth_dollar": float((rth["close"] * rth["volume"]).sum())})
        if (i + 1) % 50 == 0:
            print(f"[liq] ranked {i+1}/{len(symbols)}", flush=True)
    out = pd.DataFrame(rows).sort_values("rth_dollar", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(root, args.max_symbols_scan)
    print(f"[stock] symbols={len(symbols)} months={months[0]}..{months[-1]}", flush=True)

    liq = rank_liquidity(root, symbols, months, args.top_n)
    liq.to_csv(out / "liquidity_topn.csv", index=False)
    top_syms = liq["symbol"].tolist()
    print(f"[stock] top{args.top_n}: {top_syms[:10]}...", flush=True)

    panels = []
    for i, sym in enumerate(top_syms):
        df = load_symbol_months(root, sym, months)
        panel = build_day_panel(df, sym)
        if not panel.empty:
            panels.append(panel)
        if (i + 1) % 10 == 0:
            print(f"[stock] panels {i+1}/{len(top_syms)}", flush=True)
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    panel.to_parquet(out / "day_panel.parquet", index=False)
    print(f"[stock] day_panel rows={len(panel)} symbols={panel['symbol'].nunique()}", flush=True)

    # Strategies
    gap = panel.dropna(subset=["gap", "ret_0930_1000", "ret_rth"]).copy()
    gap = gap[gap["gap"].abs() >= args.gap_thr]
    gap["dir"] = np.sign(gap["gap"])
    gap["cont_30m"] = gap["dir"] * gap["ret_0930_1000"]
    gap["cont_90m"] = gap["dir"] * gap["ret_0930_1100"]
    gap["cont_eod"] = gap["dir"] * gap["ret_rth"]
    gap["fade_30m"] = -gap["dir"] * gap["ret_0930_1000"]
    gap["fade_90m"] = -gap["dir"] * gap["ret_0930_1100"]
    gap["fade_eod"] = -gap["dir"] * gap["ret_rth"]

    pre = panel.dropna(subset=["ret_by_0900", "ret_0900_1000"]).copy()
    pre = pre[pre["ret_by_0900"].abs() >= args.pre09_thr]
    pre["dir"] = np.sign(pre["ret_by_0900"])
    pre["catch_0935"] = pre["dir"] * pre["ret_0900_0935"]
    pre["catch_1000"] = pre["dir"] * pre["ret_0900_1000"]

    ah = panel.dropna(subset=["ret_ah", "next_open_30m"]).copy()
    ah = ah[ah["ret_ah"].abs() >= args.gap_thr]
    ah["dir"] = np.sign(ah["ret_ah"])
    ah["ah_cont_next30"] = ah["dir"] * ah["next_open_30m"]
    ah["ah_fade_next30"] = -ah["dir"] * ah["next_open_30m"]

    trade_stats = [
        summarize_trades(gap["cont_30m"], "gap_continue_open_to_1000"),
        summarize_trades(gap["cont_90m"], "gap_continue_open_to_1100"),
        summarize_trades(gap["cont_eod"], "gap_continue_open_to_eod"),
        summarize_trades(gap["fade_30m"], "gap_fade_open_to_1000"),
        summarize_trades(gap["fade_90m"], "gap_fade_open_to_1100"),
        summarize_trades(gap["fade_eod"], "gap_fade_open_to_eod"),
        summarize_trades(pre["catch_0935"], "pre09_catch_to_0935"),
        summarize_trades(pre["catch_1000"], "pre09_catch_to_1000"),
        summarize_trades(ah["ah_cont_next30"], "ah_continue_next_open_30m"),
        summarize_trades(ah["ah_fade_next30"], "ah_fade_next_open_30m"),
    ]
    acct_stats = {
        "gap_continue_30m": daily_cs_account(gap.assign(ret=gap["cont_30m"]), "ret"),
        "gap_fade_30m": daily_cs_account(gap.assign(ret=gap["fade_30m"]), "ret"),
        "gap_continue_eod": daily_cs_account(gap.assign(ret=gap["cont_eod"]), "ret"),
        "gap_fade_eod": daily_cs_account(gap.assign(ret=gap["fade_eod"]), "ret"),
        "pre09_catch_1000": daily_cs_account(pre.assign(ret=pre["catch_1000"]), "ret"),
        "ah_continue_next30": daily_cs_account(ah.assign(ret=ah["ah_cont_next30"]), "ret"),
        "ah_fade_next30": daily_cs_account(ah.assign(ret=ah["ah_fade_next30"]), "ret"),
    }

    # split by gap size tercile for continue vs fade
    gap2 = gap.copy()
    gap2["gap_abs_bin"] = pd.qcut(gap2["gap"].abs(), 3, labels=["small", "mid", "large"])
    by_size = (
        gap2.groupby("gap_abs_bin", observed=True)
        .agg(
            n=("cont_30m", "size"),
            cont_30m=("cont_30m", "mean"),
            fade_30m=("fade_30m", "mean"),
            cont_eod=("cont_eod", "mean"),
            fade_eod=("fade_eod", "mean"),
            cont_wr=("cont_30m", lambda s: (s > 0).mean()),
            fade_wr=("fade_30m", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    # coverage of extended hours
    cov = {
        "n_symbol_days": int(len(panel)),
        "pre_coverage": float(panel["has_pre"].mean()) if len(panel) else 0.0,
        "ah_coverage": float(panel["has_ah"].mean()) if len(panel) else 0.0,
        "mean_abs_gap": float(panel["gap"].abs().mean()) if len(panel) else float("nan"),
        "frac_gap_ge_thr": float((panel["gap"].abs() >= args.gap_thr).mean()) if len(panel) else 0.0,
    }

    summary = {
        "experiment": "stock_premarket_ah_validation",
        "config": vars(args),
        "liquidity_top": top_syms[:20],
        "coverage": cov,
        "trade_stats": trade_stats,
        "account_10pct_basket": acct_stats,
        "by_gap_size": by_size.to_dict(orient="records"),
        "interpretation_hints": [
            "If fade_30m >> continue_30m: open handoff is mean-reverting after premarket move",
            "If pre09_catch positive: can capture late-premarket continuation into open",
            "If both weak: pattern exists descriptively but not easily tradable at open",
        ],
        "files": {
            "panel": str(out / "day_panel.parquet"),
            "liquidity": str(out / "liquidity_topn.csv"),
            "summary": str(out / "summary.json"),
        },
    }
    pd.DataFrame(trade_stats).to_csv(out / "trade_stats.csv", index=False)
    by_size.to_csv(out / "by_gap_size.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"coverage": cov, "trade_stats": trade_stats, "account": acct_stats, "by_gap_size": summary["by_gap_size"]}, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
