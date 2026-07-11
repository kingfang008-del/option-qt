#!/usr/bin/env python3
"""Premarket ignition validation from 04:00 ET (not 09:00 chase).

Detect unusual premarket activity on 15s bars using:
  - volume z vs same time-of-day baseline (causal, prior days)
  - volume-price intensity (dollar volume × |ret| / baselines)
  - price acceleration (recent short ret vs prior short ret)
  - smooth rally path efficiency

Entry: first ignition in [04:15, 09:15], direction = sign(recent ret).
Forward holds: +15m / +30m / +60m / to 09:30 / to 10:00 / to EOD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
PRE_START = pd.Timestamp("04:00").time()
PRE_END = pd.Timestamp("09:30").time()
SIG_START = pd.Timestamp("04:15").time()
SIG_END = pd.Timestamp("09:15").time()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/s990/data/all_data/stocks_15s_parquet")
    p.add_argument("--start-month", default="2024-01")
    p.add_argument("--end-month", default="2025-06")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument(
        "--liquidity-csv",
        default="qqq_btc/results/stock_premarket_ah_validation/liquidity_topn.csv",
        help="reuse prior liquidity ranking if present",
    )
    p.add_argument("--lookback-days", type=int, default=20, help="TOD baseline lookback")
    p.add_argument("--vol-z", type=float, default=2.5)
    p.add_argument("--ret-z", type=float, default=1.5)
    p.add_argument("--accel-thr", type=float, default=0.002, help="1m accel abs threshold")
    p.add_argument("--smooth-ret", type=float, default=0.004, help="5m net ret for smooth")
    p.add_argument("--smooth-eff", type=float, default=0.55)
    p.add_argument("--vp-z", type=float, default=2.0, help="volume-price intensity z")
    p.add_argument(
        "--output-dir",
        default="qqq_btc/results/stock_premarket_ignition_validation",
    )
    return p.parse_args()


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start=start, end=end, freq="M")]


def load_symbol_months(root: Path, symbol: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        fp = root / symbol / f"{m}.parquet"
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        try:
            df = pd.read_parquet(
                fp,
                columns=["timestamp", "open", "high", "low", "close", "volume", "transactions"],
            )
        except Exception:
            try:
                df = pd.read_parquet(
                    fp, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["transactions"] = np.nan
            except Exception as e:
                print(f"[warn] skip {fp}: {e}", flush=True)
                continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df.assign(timestamp=ts).sort_values("timestamp").reset_index(drop=True)
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["tod"] = df["timestamp"].dt.time
    # minute-of-day key for TOD baseline (04:00 -> 240)
    df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


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


def daily_cs_account(trades: pd.DataFrame, ret_col: str, position_frac: float = 0.1) -> dict:
    if trades.empty:
        return {"n_days": 0, "account_ret": 0.0, "max_dd": 0.0}
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


def px_at_or_after(day: pd.DataFrame, hhmm: str) -> float:
    t = pd.Timestamp(hhmm).time()
    sub = day[day["tod"] >= t]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["open"] if "open" in sub.columns else sub.iloc[0]["close"])


def px_at_or_before(day: pd.DataFrame, hhmm: str) -> float:
    t = pd.Timestamp(hhmm).time()
    sub = day[day["tod"] <= t]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["close"])


def build_features(pre: pd.DataFrame) -> pd.DataFrame:
    """Intraday premarket features on 15s bars."""
    x = pre.copy()
    c = x["close"].astype(float)
    v = x["volume"].astype(float).fillna(0.0)
    dvol = c * v
    # 1m=4 bars, 5m=20 bars
    x["ret_1m"] = c.pct_change(4)
    x["ret_5m"] = c.pct_change(20)
    x["ret_bar"] = c.pct_change(1)
    prior_1m = c.pct_change(4).shift(4)
    x["accel_1m"] = x["ret_1m"] - prior_1m
    x["vol_1m"] = v.rolling(4, min_periods=2).sum()
    x["vol_5m"] = v.rolling(20, min_periods=5).sum()
    x["dvol_1m"] = dvol.rolling(4, min_periods=2).sum()
    x["dvol_5m"] = dvol.rolling(20, min_periods=5).sum()
    # volume-price: dollar intensity of the move (high = heavy participation)
    x["vp_1m"] = x["dvol_1m"] * x["ret_1m"].abs().clip(lower=1e-6)
    # path efficiency over 5m
    abs_sum = x["ret_bar"].abs().rolling(20, min_periods=5).sum()
    x["path_eff_5m"] = x["ret_5m"].abs() / abs_sum.clip(lower=1e-8)
    return x


def causal_tod_stats(hist: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per mod: mean/std from historical premarket bars."""
    if hist.empty:
        return pd.DataFrame()
    g = hist.groupby("mod")[cols]
    out = g.agg(["mean", "std", "median"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def zscore(series: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    s = std.replace(0, np.nan)
    return (series - mean) / s


def detect_day_signals(
    day_feat: pd.DataFrame,
    tod_stats: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict]:
    if day_feat.empty or tod_stats.empty:
        return []
    x = day_feat.merge(tod_stats, left_on="mod", right_index=True, how="left")
    x["z_vol_1m"] = zscore(x["vol_1m"], x["vol_1m_mean"], x["vol_1m_std"])
    x["z_ret_1m"] = zscore(x["ret_1m"], x["ret_1m_mean"], x["ret_1m_std"])
    x["z_vp_1m"] = zscore(x["vp_1m"], x["vp_1m_mean"], x["vp_1m_std"])
    x["z_dvol_5m"] = zscore(x["dvol_5m"], x["dvol_5m_mean"], x["dvol_5m_std"])

    # signal windows
    in_win = (x["tod"] >= SIG_START) & (x["tod"] <= SIG_END)
    x = x.loc[in_win].copy()
    if x.empty:
        return []

    vol_spike = (x["z_vol_1m"] >= args.vol_z) & (x["ret_1m"].abs() >= 0.001)
    vp_ignite = (x["z_vp_1m"] >= args.vp_z) & (x["z_vol_1m"] >= 1.5) & (x["ret_1m"].abs() >= 0.0015)
    accel = (x["accel_1m"].abs() >= args.accel_thr) & (x["z_vol_1m"] >= 1.5)
    smooth = (
        (x["ret_5m"].abs() >= args.smooth_ret)
        & (x["path_eff_5m"] >= args.smooth_eff)
        & (x["z_dvol_5m"] >= 1.0)
    )
    any_sig = vol_spike | vp_ignite | accel | smooth
    if not any_sig.any():
        return []

    # first hit per family + first any
    families = {
        "vol_spike": vol_spike,
        "vp_ratio": vp_ignite,
        "accel": accel,
        "smooth_rally": smooth,
        "any": any_sig,
    }
    out = []
    for name, mask in families.items():
        hit = x.loc[mask]
        if hit.empty:
            continue
        row = hit.iloc[0]
        direction = float(np.sign(row["ret_5m"] if abs(row.get("ret_5m", 0) or 0) > 1e-6 else row["ret_1m"]))
        if direction == 0:
            continue
        out.append(
            {
                "signal": name,
                "entry_ts": row["timestamp"],
                "entry_px": float(row["close"]),
                "direction": direction,
                "mod": int(row["mod"]),
                "z_vol_1m": float(row["z_vol_1m"]) if pd.notna(row["z_vol_1m"]) else float("nan"),
                "z_vp_1m": float(row["z_vp_1m"]) if pd.notna(row["z_vp_1m"]) else float("nan"),
                "z_ret_1m": float(row["z_ret_1m"]) if pd.notna(row["z_ret_1m"]) else float("nan"),
                "accel_1m": float(row["accel_1m"]) if pd.notna(row["accel_1m"]) else float("nan"),
                "ret_5m": float(row["ret_5m"]) if pd.notna(row["ret_5m"]) else float("nan"),
                "path_eff_5m": float(row["path_eff_5m"]) if pd.notna(row["path_eff_5m"]) else float("nan"),
                "flags": {
                    "vol_spike": bool(vol_spike.loc[row.name]),
                    "vp_ratio": bool(vp_ignite.loc[row.name]),
                    "accel": bool(accel.loc[row.name]),
                    "smooth_rally": bool(smooth.loc[row.name]),
                },
            }
        )
    return out


def forward_rets(full_day: pd.DataFrame, entry_ts, entry_px: float, direction: float) -> dict:
    after = full_day[full_day["timestamp"] > entry_ts]
    if after.empty or not np.isfinite(entry_px) or entry_px <= 0:
        return {}

    def ret_at(ts_delta_min: int | None = None, hhmm: str | None = None) -> float:
        if ts_delta_min is not None:
            target = entry_ts + pd.Timedelta(minutes=ts_delta_min)
            sub = after[after["timestamp"] <= target]
            if sub.empty:
                return float("nan")
            px = float(sub.iloc[-1]["close"])
        else:
            assert hhmm is not None
            # prefer at-or-after for session landmarks
            t = pd.Timestamp(hhmm).time()
            sub = full_day[full_day["tod"] >= t]
            if sub.empty:
                return float("nan")
            # only if landmark is after entry
            if sub.iloc[0]["timestamp"] <= entry_ts:
                # already past; use first bar after entry at/after time next... skip
                sub2 = after[after["tod"] >= t]
                if sub2.empty:
                    return float("nan")
                px = float(sub2.iloc[0]["close"])
            else:
                px = float(sub.iloc[0]["open"])
        return direction * (px / entry_px - 1.0)

    rth = full_day[(full_day["tod"] >= PRE_END) & (full_day["tod"] < pd.Timestamp("16:00").time())]
    eod = float(rth.iloc[-1]["close"]) if not rth.empty else float("nan")
    out = {
        "fwd_15m": ret_at(15),
        "fwd_30m": ret_at(30),
        "fwd_60m": ret_at(60),
        "fwd_0930": ret_at(hhmm="09:30"),
        "fwd_1000": ret_at(hhmm="10:00"),
        "fwd_eod": direction * (eod / entry_px - 1.0) if np.isfinite(eod) else float("nan"),
    }
    return out


def process_symbol(df: pd.DataFrame, symbol: str, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pre_mask = (df["tod"] >= PRE_START) & (df["tod"] < PRE_END)
    pre_all = build_features(df.loc[pre_mask].copy())
    dates = sorted(df["date_str"].unique())
    # keep rolling hist of premarket feature rows
    hist_parts: list[pd.DataFrame] = []
    trades = []
    feat_cols = ["vol_1m", "ret_1m", "vp_1m", "dvol_5m"]

    for date_str in dates:
        day = df[df["date_str"] == date_str]
        pre = pre_all[pre_all["date_str"] == date_str]
        if len(pre) < 40:
            continue
        # causal baseline from prior lookback days only
        if hist_parts:
            hist = pd.concat(hist_parts[-args.lookback_days :], ignore_index=True)
            tod_stats = causal_tod_stats(hist, feat_cols)
        else:
            tod_stats = pd.DataFrame()

        sigs = detect_day_signals(pre, tod_stats, args)
        for s in sigs:
            fr = forward_rets(day, s["entry_ts"], s["entry_px"], s["direction"])
            if not fr:
                continue
            hour = int(s["entry_ts"].hour)
            bucket = "04-06" if hour < 6 else ("06-08" if hour < 8 else "08-09")
            trades.append(
                {
                    "symbol": symbol,
                    "date_str": date_str,
                    "signal": s["signal"],
                    "entry_ts": str(s["entry_ts"]),
                    "entry_hour_bucket": bucket,
                    "direction": s["direction"],
                    "z_vol_1m": s["z_vol_1m"],
                    "z_vp_1m": s["z_vp_1m"],
                    "z_ret_1m": s["z_ret_1m"],
                    "accel_1m": s["accel_1m"],
                    "ret_5m_at_entry": s["ret_5m"],
                    "path_eff_5m": s["path_eff_5m"],
                    **fr,
                    **{f"flag_{k}": int(v) for k, v in s["flags"].items()},
                }
            )

        # update hist after day (no leakage)
        hist_parts.append(pre[feat_cols + ["mod", "date_str"]].dropna(subset=["vol_1m"]))
        if len(hist_parts) > args.lookback_days + 5:
            hist_parts = hist_parts[-args.lookback_days :]

    return pd.DataFrame(trades)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    months = month_range(args.start_month, args.end_month)

    liq_path = Path(args.liquidity_csv)
    if liq_path.exists():
        liq = pd.read_csv(liq_path).head(args.top_n)
        symbols = liq["symbol"].tolist()
        print(f"[ignition] reuse liquidity top{args.top_n} from {liq_path}", flush=True)
    else:
        raise SystemExit(f"missing liquidity csv {liq_path}; run validate_stock_premarket_ah.py first")

    all_trades = []
    for i, sym in enumerate(symbols):
        df = load_symbol_months(root, sym, months)
        tr = process_symbol(df, sym, args)
        if not tr.empty:
            all_trades.append(tr)
        print(f"[ignition] {i+1}/{len(symbols)} {sym} trades={len(tr)}", flush=True)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_parquet(out / "ignition_trades.parquet", index=False)

    # stats by signal family
    holds = ["fwd_15m", "fwd_30m", "fwd_60m", "fwd_0930", "fwd_1000", "fwd_eod"]
    trade_stats = []
    acct = {}
    for sig, g in trades.groupby("signal"):
        for h in holds:
            trade_stats.append(summarize(g[h], f"{sig}__{h}"))
        for h in ["fwd_30m", "fwd_0930", "fwd_1000"]:
            acct[f"{sig}__{h}"] = daily_cs_account(g.dropna(subset=[h]), h)

    by_bucket = (
        trades[trades["signal"] == "any"]
        .groupby("entry_hour_bucket")[holds]
        .agg(["count", "mean"])
        .reset_index()
        if not trades.empty
        else pd.DataFrame()
    )
    # flatten multiindex cols for csv
    if not by_bucket.empty:
        by_bucket.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c for c in by_bucket.columns
        ]

    # quality: require multiple flags
    multi = trades[(trades["signal"] == "any")].copy()
    if not multi.empty:
        multi["n_flags"] = (
            multi["flag_vol_spike"]
            + multi["flag_vp_ratio"]
            + multi["flag_accel"]
            + multi["flag_smooth_rally"]
        )
        for nflag in [1, 2, 3]:
            sub = multi[multi["n_flags"] >= nflag]
            for h in ["fwd_30m", "fwd_0930", "fwd_1000"]:
                trade_stats.append(summarize(sub[h], f"any_flags>={nflag}__{h}"))

    summary = {
        "experiment": "stock_premarket_ignition_validation",
        "config": vars(args),
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["symbol"].nunique()) if len(trades) else 0,
        "signal_counts": trades["signal"].value_counts().to_dict() if len(trades) else {},
        "trade_stats": trade_stats,
        "account_10pct_basket": acct,
        "by_hour_bucket_any": by_bucket.to_dict(orient="records") if not by_bucket.empty else [],
        "interpretation_hints": [
            "Entry from 04:15 when unusual vol/vp/accel/smooth fires — not 09:00 chase",
            "fwd_15m/30m tests whether ignition itself is catchable inside premarket",
            "fwd_0930/1000 tests handoff into RTH after early ignition",
        ],
    }
    pd.DataFrame(trade_stats).to_csv(out / "trade_stats.csv", index=False)
    if not by_bucket.empty:
        by_bucket.to_csv(out / "by_hour_bucket.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # compact print
    focus = [s for s in trade_stats if s.get("n", 0) > 0 and ("__fwd_30m" in s["label"] or "__fwd_0930" in s["label"] or "__fwd_1000" in s["label"])]
    print(json.dumps({"n_trades": summary["n_trades"], "signal_counts": summary["signal_counts"], "focus": focus, "account": acct}, indent=2, default=str))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
