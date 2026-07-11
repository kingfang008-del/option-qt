#!/usr/bin/env python3
"""SPX-universe premarket vs after-hours ignition: sliding-window VP / acceleration.

Universe: all symbols under stocks_15s_parquet (~SPY500).
Bars: resample 15s -> 1min for 5/10/15-minute sliding windows.

Build (entry): unusual volume-price + acceleration vs same clock-time history.
Flat (exit): 09:15 ET (15 minutes before RTH open) — for AH signals = next session 09:15.

Also OLS: fwd_to_0915 ~ features, to see if lift is regressable.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

NY = "America/New_York"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--max-symbols", type=int, default=0, help="0=all")
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--z-vol", type=float, default=2.0)
    p.add_argument("--z-vp", type=float, default=2.0)
    p.add_argument("--min-ret", type=float, default=0.002, help="min |window ret| to arm")
    p.add_argument("--accel-mult", type=float, default=1.0, help="|accel| >= accel_mult * |prior_half_ret|")
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_ext_hours_ignition_sw",
    )
    return p.parse_args()


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start=start, end=end, freq="M")]


def list_symbols(root: Path, limit: int = 0) -> list[str]:
    syms = sorted(p.name for p in root.iterdir() if p.is_dir())
    return syms[:limit] if limit > 0 else syms


def load_1min(root: Path, symbol: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        fp = root / symbol / f"{m}.parquet"
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        try:
            raw = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        if raw.empty:
            continue
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    ohlc = df.resample("1min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    ohlc = ohlc.dropna(subset=["close"])
    ohlc = ohlc.reset_index()
    ohlc["date_str"] = ohlc["timestamp"].dt.strftime("%Y-%m-%d")
    ohlc["tod"] = ohlc["timestamp"].dt.time
    ohlc["mod"] = ohlc["timestamp"].dt.hour * 60 + ohlc["timestamp"].dt.minute
    return ohlc


def add_window_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 15)) -> pd.DataFrame:
    """Sliding windows in minutes on 1-min bars. Groupby date avoids overnight bleed."""
    x = df.sort_values(["date_str", "timestamp"]).copy()
    dvol = x["close"].astype(float) * x["volume"].astype(float).fillna(0.0)
    g = x.groupby("date_str", sort=False)
    bar_abs = g["close"].pct_change(1).abs()
    for w in windows:
        x[f"ret_{w}"] = g["close"].pct_change(w)
        x[f"vol_{w}"] = g["volume"].transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).sum())
        x[f"dvol_{w}"] = dvol.groupby(x["date_str"], sort=False).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 2)).sum()
        )
        x[f"vp_{w}"] = x[f"dvol_{w}"] * x[f"ret_{w}"].abs()  # 量价强度：成交额 × |收益|
        half = max(1, w // 2)
        half_ret = g["close"].pct_change(half)
        x[f"accel_{w}"] = half_ret - half_ret.groupby(x["date_str"], sort=False).shift(half)
        abs_sum = bar_abs.groupby(x["date_str"], sort=False).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 2)).sum()
        )
        x[f"eff_{w}"] = x[f"ret_{w}"].abs() / abs_sum.clip(lower=1e-8)
    return x


def add_causal_tod_z(df: pd.DataFrame, cols: list[str], lookback: int) -> pd.DataFrame:
    """Same minute-of-day causal mean/std over prior days (1 row/mod/day)."""
    x = df.sort_values("timestamp").copy()
    for col in cols:
        g = x.groupby("mod", sort=False)[col]
        base = g.transform(lambda s: s.shift(1).rolling(lookback, min_periods=5).mean())
        std = g.transform(lambda s: s.shift(1).rolling(lookback, min_periods=5).std())
        x[f"z_{col}"] = (x[col] - base) / std.replace(0, np.nan)
    return x


def session_mask(tod: pd.Series, session: str) -> pd.Series:
    if session == "pre":
        return (tod >= pd.Timestamp("04:15").time()) & (tod <= pd.Timestamp("09:00").time())
    # AH entry: after close until late; exit next 09:15 so allow until 19:45
    return (tod >= pd.Timestamp("16:15").time()) & (tod <= pd.Timestamp("19:45").time())


def first_signal_rows(day: pd.DataFrame, args: argparse.Namespace, windows=(5, 10, 15)) -> dict[str, pd.Series] | None:
    """Return first firing row per window + combined any."""
    if day.empty:
        return None
    fires = {}
    any_mask = pd.Series(False, index=day.index)
    for w in windows:
        z_vol = day[f"z_vol_{w}"]
        z_vp = day[f"z_vp_{w}"]
        ret = day[f"ret_{w}"]
        accel = day[f"accel_{w}"]
        # 异常：同向拉升 + (量z 或 量价强度z) + 加速度同向
        unusual = (z_vol >= args.z_vol) | (z_vp >= args.z_vp)
        arm = (
            unusual.fillna(False)
            & (ret.abs() >= args.min_ret)
            & (np.sign(accel.fillna(0.0)) == np.sign(ret.fillna(0.0)))
            & (ret.fillna(0.0) != 0)
        )
        fires[w] = arm
        any_mask = any_mask | arm.fillna(False)
    if not any_mask.any():
        return None
    out = {"any": day.loc[any_mask].iloc[0]}
    for w, m in fires.items():
        if m.fillna(False).any():
            out[f"w{w}"] = day.loc[m.fillna(False)].iloc[0]
    return out


def px_at(day_full: pd.DataFrame, date_str: str, hhmm: str, prefer: str = "close") -> float:
    t = pd.Timestamp(hhmm).time()
    sub = day_full[(day_full["date_str"] == date_str) & (day_full["tod"] <= t)]
    if sub.empty:
        sub = day_full[(day_full["date_str"] == date_str) & (day_full["tod"] >= t)]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0]["open" if prefer == "open" else "close"])
    return float(sub.iloc[-1]["close"])


def next_trading_date(dates: list[str], cur: str) -> str | None:
    try:
        i = dates.index(cur)
    except ValueError:
        return None
    return dates[i + 1] if i + 1 < len(dates) else None


def process_symbol(df_1m: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    if df_1m.empty:
        return pd.DataFrame()
    feat_cols = []
    for w in (5, 10, 15):
        feat_cols += [f"vol_{w}", f"vp_{w}", f"ret_{w}", f"accel_{w}"]

    # split sessions then feature+z within each stream (mod unique enough across sessions)
    trades = []
    dates = sorted(df_1m["date_str"].unique())

    for session in ("pre", "ah"):
        if session == "pre":
            mask = (df_1m["tod"] >= pd.Timestamp("04:00").time()) & (df_1m["tod"] < pd.Timestamp("09:30").time())
        else:
            mask = (df_1m["tod"] >= pd.Timestamp("16:00").time()) & (df_1m["tod"] < pd.Timestamp("20:00").time())
        sess = df_1m.loc[mask].copy()
        if sess.empty:
            continue
        sess = add_window_features(sess, (5, 10, 15))
        z_cols = [f"vol_{w}" for w in (5, 10, 15)] + [f"vp_{w}" for w in (5, 10, 15)]
        sess = add_causal_tod_z(sess, z_cols, args.lookback_days)

        for date_str, day in sess.groupby("date_str", sort=True):
            day = day.loc[session_mask(day["tod"], session)]
            if len(day) < 20:
                continue
            hits = first_signal_rows(day, args)
            if not hits:
                continue

            # exit price at 09:15 (same day for pre; next day for ah)
            if session == "pre":
                exit_date = date_str
            else:
                exit_date = next_trading_date(dates, date_str)
                if exit_date is None:
                    continue
            exit_px = px_at(df_1m, exit_date, "09:15")
            if not np.isfinite(exit_px):
                continue

            for key, row in hits.items():
                entry_px = float(row["close"])
                if entry_px <= 0 or not np.isfinite(entry_px):
                    continue
                # direction from strongest available window ret
                direction = 0.0
                for w in (15, 10, 5):
                    r = row.get(f"ret_{w}", np.nan)
                    if pd.notna(r) and abs(r) > 1e-8:
                        direction = float(np.sign(r))
                        break
                if direction == 0:
                    continue
                fwd = direction * (exit_px / entry_px - 1.0)
                # also mark-to-open for reference
                open_px = px_at(df_1m, exit_date, "09:30", prefer="open")
                fwd_open = direction * (open_px / entry_px - 1.0) if np.isfinite(open_px) else float("nan")

                rec = {
                    "symbol": symbol,
                    "session": session,
                    "signal": key,
                    "date_str": date_str,
                    "exit_date": exit_date,
                    "entry_ts": str(row["timestamp"]),
                    "entry_mod": int(row["mod"]),
                    "direction": direction,
                    "fwd_to_0915": fwd,
                    "fwd_to_0930": fwd_open,
                    "hold_hours": (pd.Timestamp(f"{exit_date} 09:15", tz=NY) - row["timestamp"]).total_seconds() / 3600.0,
                }
                for w in (5, 10, 15):
                    for c in (f"ret_{w}", f"accel_{w}", f"z_vol_{w}", f"z_vp_{w}", f"eff_{w}"):
                        val = row.get(c, np.nan)
                        rec[c] = float(val) if pd.notna(val) else float("nan")
                trades.append(rec)
    return pd.DataFrame(trades)


def summarize(rets: pd.Series, label: str) -> dict:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    if r.empty:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(len(r)),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "p05": float(r.quantile(0.05)),
        "p95": float(r.quantile(0.95)),
    }


def daily_account(trades: pd.DataFrame, ret_col: str = "fwd_to_0915", frac: float = 0.1) -> dict:
    if trades.empty:
        return {"n_days": 0}
    # use exit_date for AH overnight alignment
    key = "exit_date" if "exit_date" in trades.columns else "date_str"
    day = trades.groupby(key)[ret_col].mean().sort_index()
    eq = np.cumprod(1.0 + frac * day.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[:-1]
    return {
        "n_days": int(len(day)),
        "avg_day": float(day.mean()),
        "account_ret": float(eq[-1] - 1.0),
        "max_dd": float((eq / peaks - 1.0).min()),
        "win_day_rate": float((day > 0).mean()),
    }


def ols_fit(df: pd.DataFrame, ycol: str, xcols: list[str]) -> dict:
    sub = df[[ycol] + xcols].dropna()
    if len(sub) < 50:
        return {"n": int(len(sub)), "r2": float("nan")}
    y = sub[ycol].to_numpy()
    X = sub[xcols].to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "n": int(len(sub)),
            "r2": float(r2),
            "intercept": float(beta[0]),
            "coefs": {c: float(b) for c, b in zip(xcols, beta[1:])},
        }
    except Exception as e:
        return {"n": int(len(sub)), "r2": float("nan"), "error": str(e)}


def _worker(payload: tuple) -> pd.DataFrame:
    root_s, sym, months, args_dict = payload
    ns = argparse.Namespace(**args_dict)
    df = load_1min(Path(root_s), sym, months)
    return process_symbol(df, sym, ns)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    months = month_range(args.start_month, args.end_month)
    symbols = list_symbols(root, args.max_symbols)
    print(f"[sw] symbols={len(symbols)} months={months[0]}..{months[-1]} exit=09:15", flush=True)

    args_dict = vars(args)
    payloads = [(str(root), sym, months, args_dict) for sym in symbols]

    all_tr = []
    # parallel by symbol
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, p): p[1] for p in payloads}
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                tr = fut.result()
                if tr is not None and not tr.empty:
                    all_tr.append(tr)
            except Exception as e:
                print(f"[warn] {sym}: {e}", flush=True)
            if done % 20 == 0 or done == 1:
                n = sum(len(x) for x in all_tr)
                print(f"[sw] {done}/{len(symbols)} last={sym} cum_trades={n}", flush=True)

    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
    trades.to_parquet(out / "trades.parquet", index=False)

    stats = []
    accts = {}
    regs = {}
    for session in ("pre", "ah"):
        for sig in ("any", "w5", "w10", "w15"):
            g = trades[(trades["session"] == session) & (trades["signal"] == sig)]
            stats.append(summarize(g["fwd_to_0915"], f"{session}/{sig}/to_0915"))
            stats.append(summarize(g["fwd_to_0930"], f"{session}/{sig}/to_0930"))
            accts[f"{session}/{sig}"] = daily_account(g)
            # regression: can features explain fwd?
            xcols = ["z_vol_10", "z_vp_10", "accel_10", "ret_10", "eff_10"]
            regs[f"{session}/{sig}"] = ols_fit(g, "fwd_to_0915", xcols)

    # compare pre vs ah on any
    compare = {
        "pre_any": summarize(trades.query("session=='pre' and signal=='any'")["fwd_to_0915"], "pre"),
        "ah_any": summarize(trades.query("session=='ah' and signal=='any'")["fwd_to_0915"], "ah"),
        "pre_account": accts.get("pre/any", {}),
        "ah_account": accts.get("ah/any", {}),
        "pre_ols": regs.get("pre/any", {}),
        "ah_ols": regs.get("ah/any", {}),
    }

    counts = {}
    if len(trades):
        for (a, b), v in trades.groupby(["session", "signal"]).size().items():
            counts[f"{a}/{b}"] = int(v)

    summary = {
        "experiment": "stock_ext_hours_ignition_sliding_window",
        "config": vars(args),
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["symbol"].nunique()) if len(trades) else 0,
        "counts": counts,
        "trade_stats": stats,
        "accounts_10pct": accts,
        "ols_fwd_to_0915": regs,
        "compare_pre_vs_ah": compare,
        "notes": [
            "Entry: first 5/10/15m window with high z_vol & z_vp, directional ret, same-sign accel",
            "Exit: 09:15 ET (15m before open); AH holds overnight to next 09:15",
            "OLS tests whether ignition features linearly explain hold-to-0915 return",
        ],
    }
    pd.DataFrame(stats).to_csv(out / "trade_stats.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"compare_pre_vs_ah": compare, "counts": summary["counts"]}, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
