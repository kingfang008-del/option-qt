#!/usr/bin/env python3
"""Batch factor analysis for 0DTE option trade-print/order-flow signals.

This script intentionally avoids fitting a predictive model.  It asks a simpler
question first: which order-flow and quote-response factors lead future option
price moves, and does that signal survive strict bid/ask execution costs?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NY = "America/New_York"
ROLL_WINDOWS = (1, 3, 5, 10, 30, 60)
HORIZONS = (5, 10, 30)
CORE_FACTORS = [
    "flow_imbalance_1s",
    "flow_imbalance_3s",
    "flow_imbalance_5s",
    "flow_imbalance_10s",
    "flow_toxicity_1s",
    "flow_toxicity_5s",
    "flow_toxicity_10s",
    "buy_ratio",
    "signed_buy_ratio_side",
    "net_buy_sum_3s",
    "net_buy_sum_5s",
    "net_buy_sum_10s",
    "net_buy_accel_3s",
    "net_buy_accel_5s",
    "net_buy_accel_10s",
    "trade_notional",
    "trade_notional_log1p",
    "trade_notional_sum_5s",
    "trade_notional_sum_10s",
    "trade_notional_sum_60s",
    "notional_accel_3s",
    "notional_accel_5s",
    "notional_accel_10s",
    "quote_imbalance",
    "quote_event_intensity",
    "quote_events_sum_5s",
    "quote_events_sum_10s",
    "quote_event_accel_5s",
    "mid_up_minus_down_sum_5s",
    "spread_tighten_minus_widen_sum_5s",
    "spread_compress_3s",
    "mid_ret_past_1s",
    "mid_ret_past_3s",
    "mid_ret_past_5s",
    "flow_confirm_5s",
    "toxic_momentum_5s",
    "spread_pct",
    "tod_frac",
]


BASE_NUMERIC_COLS = [
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "quote_imbalance",
    "quote_events",
    "bid_up_events",
    "bid_down_events",
    "ask_up_events",
    "ask_down_events",
    "mid_up_events",
    "mid_down_events",
    "spread_tighten_events",
    "spread_widen_events",
    "trade_count",
    "trade_volume",
    "trade_notional",
    "buy_volume",
    "sell_volume",
    "unknown_volume",
    "last_trade_price",
    "net_buy_volume",
    "buy_ratio",
    "bucket_id",
    "strike",
]


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def normalize_day(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    for col in BASE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    if "mid" not in raw.columns or df["mid"].isna().all():
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    if "spread_pct" not in raw.columns or df["spread_pct"].isna().all():
        df["spread_pct"] = safe_div(df["ask"] - df["bid"], df["mid"])
    df["ticker"] = df["ticker"].astype(str)
    side = df.get("side", pd.Series("", index=df.index)).astype(str).str.upper()
    # Infer CALL/PUT from OCC ticker when side missing (common in raw_1s sniper dumps).
    missing = ~side.isin(["CALL", "PUT"])
    if missing.any():
        inferred = df.loc[missing, "ticker"].str.extract(r"[0-9]([CP])[0-9]{8}$", expand=False)
        side.loc[missing] = inferred.map({"C": "CALL", "P": "PUT"}).fillna("")
    df["side"] = side
    df["side_code"] = df["side"].map({"PUT": -1.0, "CALL": 1.0}).fillna(0.0)
    df["tod_frac"] = (
        df["timestamp"].dt.hour * 3600
        + df["timestamp"].dt.minute * 60
        + df["timestamp"].dt.second
        - (9 * 3600 + 30 * 60)
    ) / (6.5 * 3600)
    return df


def add_contract_factors(day: pd.DataFrame, horizons: tuple[int, ...], commission: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, g0 in day.sort_values(["ticker", "timestamp"]).groupby("ticker", sort=False):
        g = g0.drop_duplicates("timestamp", keep="last").copy().reset_index(drop=True)
        buy = g["buy_volume"].fillna(0.0)
        sell = g["sell_volume"].fillna(0.0)
        net = g["net_buy_volume"].fillna(0.0)
        traded = buy + sell

        g["flow_imbalance_1s"] = safe_div(buy - sell, traded)
        g["flow_toxicity_1s"] = safe_div((buy - sell).abs(), traded)
        g["signed_buy_ratio_side"] = np.where(g["side_code"] > 0, g["buy_ratio"], 1.0 - g["buy_ratio"])
        g["trade_notional_log1p"] = np.log1p(g["trade_notional"].clip(lower=0.0))
        g["spread_tighten_minus_widen"] = g["spread_tighten_events"] - g["spread_widen_events"]
        g["mid_up_minus_down"] = g["mid_up_events"] - g["mid_down_events"]
        g["quote_event_intensity"] = np.log1p(g["quote_events"].clip(lower=0.0))

        for w in ROLL_WINDOWS:
            buy_w = buy.rolling(w, min_periods=1).sum()
            sell_w = sell.rolling(w, min_periods=1).sum()
            g[f"net_buy_sum_{w}s"] = net.rolling(w, min_periods=1).sum()
            g[f"trade_volume_sum_{w}s"] = g["trade_volume"].rolling(w, min_periods=1).sum()
            g[f"trade_notional_sum_{w}s"] = g["trade_notional"].rolling(w, min_periods=1).sum()
            g[f"flow_imbalance_{w}s"] = safe_div(buy_w - sell_w, buy_w + sell_w)
            g[f"flow_toxicity_{w}s"] = safe_div((buy_w - sell_w).abs(), buy_w + sell_w)
            g[f"quote_events_sum_{w}s"] = g["quote_events"].rolling(w, min_periods=1).sum()
            g[f"mid_up_minus_down_sum_{w}s"] = g["mid_up_minus_down"].rolling(w, min_periods=1).sum()
            g[f"spread_tighten_minus_widen_sum_{w}s"] = g["spread_tighten_minus_widen"].rolling(w, min_periods=1).sum()

        for w in (3, 5, 10):
            g[f"net_buy_accel_{w}s"] = g[f"net_buy_sum_{w}s"] - g[f"net_buy_sum_{w}s"].shift(w)
            g[f"notional_accel_{w}s"] = g[f"trade_notional_sum_{w}s"] - g[f"trade_notional_sum_{w}s"].shift(w)
            g[f"quote_event_accel_{w}s"] = g[f"quote_events_sum_{w}s"] - g[f"quote_events_sum_{w}s"].shift(w)

        for w in (1, 3, 5, 10):
            g[f"mid_ret_past_{w}s"] = g["mid"] / g["mid"].shift(w) - 1.0
            g[f"bid_ret_past_{w}s"] = g["bid"] / g["bid"].shift(w) - 1.0
        g["spread_chg_3s"] = g["spread_pct"] - g["spread_pct"].shift(3)
        g["spread_compress_3s"] = -g["spread_chg_3s"]
        g["flow_confirm_5s"] = g["flow_imbalance_5s"] * g["mid_ret_past_5s"]
        g["toxic_momentum_5s"] = g["flow_toxicity_5s"] * g["mid_ret_past_5s"].abs()

        for h in horizons:
            future_mid = g["mid"].shift(-h)
            future_bid = g["bid"].shift(-h)
            g[f"target_mid_ret_{h}s"] = future_mid / g["mid"] - 1.0
            cost_frac = 2.0 * commission / (g["ask"] * 100.0).replace(0, np.nan)
            g[f"target_exec_ret_{h}s"] = future_bid / g["ask"] - 1.0 - cost_frac
            g[f"target_mid_burst_{h}s"] = (g[f"target_mid_ret_{h}s"] >= 0.05).astype(float)
            g[f"target_exec_burst_{h}s"] = (g[f"target_exec_ret_{h}s"] >= 0.05).astype(float)
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def add_dynamic_universe(df: pd.DataFrame, top_n: int, lookback_s: int, per_side: bool) -> pd.DataFrame:
    if top_n <= 0:
        return df.copy()
    work = df.sort_values(["ticker", "timestamp"]).copy()
    work["rolling_notional"] = (
        work.groupby("ticker")["trade_notional"].rolling(lookback_s, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    work["rolling_trades"] = (
        work.groupby("ticker")["trade_count"].rolling(lookback_s, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    # Sniper raw->micro often has zero trade prints. Fall back to quote-liquidity score.
    if float(work["rolling_trades"].fillna(0).sum()) <= 0:
        depth = pd.to_numeric(work.get("bid_size"), errors="coerce").fillna(0) + pd.to_numeric(
            work.get("ask_size"), errors="coerce"
        ).fillna(0)
        mid = pd.to_numeric(work.get("mid"), errors="coerce").fillna(0)
        spread = pd.to_numeric(work.get("spread_pct"), errors="coerce").replace(0, np.nan)
        work["rolling_notional"] = mid * depth / (1.0 + spread.fillna(1.0))
        work["rolling_trades"] = depth.clip(lower=0) + 1.0
    work = work[work["rolling_trades"] > 0].copy()
    if work.empty:
        return work
    keys = ["timestamp", "side"] if per_side else ["timestamp"]
    work["universe_rank"] = work.groupby(keys)["rolling_notional"].rank(method="first", ascending=False)
    return work[work["universe_rank"] <= top_n].copy()


def factor_columns(df: pd.DataFrame, *, expanded: bool) -> list[str]:
    if not expanded:
        return [c for c in CORE_FACTORS if c in df.columns]
    prefixes = (
        "flow_",
        "net_buy_",
        "trade_volume_sum_",
        "trade_notional_sum_",
        "notional_accel_",
        "quote_event",
        "mid_up_minus_down",
        "spread_tighten",
        "spread_compress",
        "mid_ret_past_",
        "bid_ret_past_",
        "toxic_momentum",
    )
    cols = [
        "buy_ratio",
        "signed_buy_ratio_side",
        "trade_count",
        "trade_volume",
        "trade_notional",
        "trade_notional_log1p",
        "quote_imbalance",
        "quote_events",
        "spread_pct",
        "tod_frac",
    ]
    cols.extend(c for c in df.columns if c.startswith(prefixes))
    return sorted({c for c in cols if c in df.columns})


def target_columns(horizons: tuple[int, ...], *, include_bursts: bool) -> list[str]:
    out: list[str] = []
    for h in horizons:
        out.extend([f"target_mid_ret_{h}s", f"target_exec_ret_{h}s"])
        if include_bursts:
            out.extend([f"target_mid_burst_{h}s", f"target_exec_burst_{h}s"])
    return out


def load_factor_dataset(
    micro_root: Path,
    start: str,
    end: str,
    horizons: tuple[int, ...],
    *,
    top_n: int,
    lookback_s: int,
    per_side: bool,
    commission: float,
    max_spread_pct: float,
    min_ask: float,
    symbol: str = "QQQ",
) -> pd.DataFrame:
    sym = str(symbol).upper()
    files = sorted((micro_root / f"contract_1s/{sym}").glob(f"{sym}_*.parquet"))
    prefix = f"{sym}_"
    files = [p for p in files if start <= p.stem.replace(prefix, "") <= end]
    frames: list[pd.DataFrame] = []
    for fp in files:
        raw = pd.read_parquet(fp)
        if raw.empty:
            continue
        day = add_contract_factors(normalize_day(raw), horizons, commission)
        day = add_dynamic_universe(day, top_n=top_n, lookback_s=lookback_s, per_side=per_side)
        tradable = (
            day["side"].isin(["CALL", "PUT"])
            & (day["ask"] >= min_ask)
            & (day["bid"] > 0)
            & (day["spread_pct"] <= max_spread_pct)
            & day["bucket_id"].notna()
        )
        day = day[tradable].copy()
        if day.empty:
            continue
        day["date_str"] = fp.stem.replace(prefix, "")
        day["underlying"] = sym
        frames.append(day)
    if not frames:
        raise SystemExit(f"no factor dataset for {sym} {start}..{end} under {micro_root}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    sample = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 100 or sample["x"].nunique() < 3 or sample["y"].nunique() < 3:
        return float("nan")
    return float(sample["x"].rank().corr(sample["y"].rank()))


def summarize_ic(
    df: pd.DataFrame,
    factors: list[str],
    targets: list[str],
    group_name: str,
    *,
    daily_ic: bool,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        for factor in factors:
            sample = df[[factor, target, "date_str"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 500:
                continue
            if daily_ic:
                daily = sample.groupby("date_str", group_keys=False).apply(
                    lambda g: spearman_ic(g[factor], g[target])
                )
                daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
                daily_mean = float(daily.mean()) if len(daily) else float("nan")
                daily_std = float(daily.std(ddof=0)) if len(daily) else float("nan")
                daily_ir = float(daily_mean / daily_std) if daily_std > 0 else float("nan")
                positive_days = int((daily > 0).sum()) if len(daily) else 0
                days = int(len(daily))
            else:
                daily_mean = float("nan")
                daily_std = float("nan")
                daily_ir = float("nan")
                positive_days = 0
                days = int(sample["date_str"].nunique())
            rows.append(
                {
                    "group": group_name,
                    "target": target,
                    "factor": factor,
                    "n": int(len(sample)),
                    "ic": spearman_ic(sample[factor], sample[target]),
                    "daily_ic_mean": daily_mean,
                    "daily_ic_std": daily_std,
                    "daily_ic_ir": daily_ir,
                    "positive_days": positive_days,
                    "days": days,
                }
            )
    return pd.DataFrame(rows)


def quantile_table(df: pd.DataFrame, factors: list[str], targets: list[str], group_name: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        for factor in factors:
            sample = df[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sample) < 1000 or sample[factor].nunique() < 10:
                continue
            q10 = sample[factor].quantile(0.10)
            q90 = sample[factor].quantile(0.90)
            low = sample[sample[factor] <= q10][target]
            high = sample[sample[factor] >= q90][target]
            rows.append(
                {
                    "group": group_name,
                    "target": target,
                    "factor": factor,
                    "n": int(len(sample)),
                    "low_n": int(len(low)),
                    "high_n": int(len(high)),
                    "low_mean": float(low.mean()),
                    "high_mean": float(high.mean()),
                    "high_minus_low": float(high.mean() - low.mean()),
                    "low_pos_rate": float((low > 0).mean()),
                    "high_pos_rate": float((high > 0).mean()),
                    "high_p95": float(high.quantile(0.95)),
                    "high_p99": float(high.quantile(0.99)),
                }
            )
    return pd.DataFrame(rows)


def run_grouped_analysis(
    df: pd.DataFrame,
    factors: list[str],
    targets: list[str],
    *,
    out_dir: Path,
    daily_ic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: list[tuple[str, pd.Series]] = [
        ("all", pd.Series(True, index=df.index)),
        ("CALL", df["side"] == "CALL"),
        ("PUT", df["side"] == "PUT"),
    ]
    if "universe_rank" in df.columns:
        groups.append(("rank1", df["universe_rank"] <= 1))
    groups.extend(
        [
            ("tight_spread", df["spread_pct"] <= df["spread_pct"].quantile(0.25)),
            ("high_notional", df["trade_notional_sum_60s"] >= df["trade_notional_sum_60s"].quantile(0.75)),
        ]
    )
    ic_frames = []
    q_frames = []
    for name, mask in groups:
        sub = df[mask].copy()
        if len(sub) < 5000:
            continue
        print(f"[factor-analysis] group={name} rows={len(sub)}", flush=True)
        ic_frames.append(summarize_ic(sub, factors, targets, name, daily_ic=daily_ic))
        q_frames.append(quantile_table(sub, factors, targets, name))
        pd.concat(ic_frames, ignore_index=True).to_csv(out_dir / "factor_ic_partial.csv", index=False)
        pd.concat(q_frames, ignore_index=True).to_csv(out_dir / "quantile_returns_partial.csv", index=False)
    return pd.concat(ic_frames, ignore_index=True), pd.concat(q_frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--start", default="2026-04-13")
    p.add_argument("--end", default="2026-06-30")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--lookback-s", type=int, default=60)
    p.add_argument("--per-side", action="store_true")
    p.add_argument("--horizons", default="5,10,30")
    p.add_argument("--expanded-factors", action="store_true")
    p.add_argument("--include-bursts", action="store_true")
    p.add_argument("--daily-ic", action="store_true")
    p.add_argument("--commission-per-contract", type=float, default=0.65)
    p.add_argument("--max-spread-pct", type=float, default=0.05)
    p.add_argument("--min-ask", type=float, default=0.20)
    p.add_argument("--output-dir", default="factor_lab/results/0dte_tradeprint_factor_analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[factor-analysis] loading {args.start}..{args.end} top_n={args.top_n} per_side={bool(args.per_side)}",
        flush=True,
    )
    df = load_factor_dataset(
        Path(args.micro_root),
        args.start,
        args.end,
        horizons,
        top_n=args.top_n,
        lookback_s=args.lookback_s,
        per_side=args.per_side,
        commission=args.commission_per_contract,
        max_spread_pct=args.max_spread_pct,
        min_ask=args.min_ask,
    )
    print(f"[factor-analysis] loaded rows={len(df)} dates={df['date_str'].nunique()}", flush=True)
    factors = factor_columns(df, expanded=args.expanded_factors)
    targets = target_columns(horizons, include_bursts=args.include_bursts)
    print(f"[factor-analysis] factors={len(factors)} targets={len(targets)}", flush=True)
    ic, quantiles = run_grouped_analysis(df, factors, targets, out_dir=out_dir, daily_ic=bool(args.daily_ic))
    ic_sort_col = "daily_ic_mean" if args.daily_ic else "ic"
    ic = ic.sort_values(["target", "group", ic_sort_col], ascending=[True, True, False])
    quantiles = quantiles.sort_values(["target", "group", "high_minus_low"], ascending=[True, True, False])

    ic.to_csv(out_dir / "factor_ic.csv", index=False)
    quantiles.to_csv(out_dir / "quantile_returns.csv", index=False)
    summary = {
        "config": {
            "micro_root": args.micro_root,
            "start": args.start,
            "end": args.end,
            "top_n": args.top_n,
            "lookback_s": args.lookback_s,
            "per_side": bool(args.per_side),
            "horizons": list(horizons),
            "expanded_factors": bool(args.expanded_factors),
            "include_bursts": bool(args.include_bursts),
            "commission_per_contract": args.commission_per_contract,
            "max_spread_pct": args.max_spread_pct,
            "min_ask": args.min_ask,
        },
        "rows": int(len(df)),
        "dates": int(df["date_str"].nunique()),
        "side_counts": df["side"].value_counts().to_dict(),
        "targets": {
            t: {
                "mean": float(pd.to_numeric(df[t], errors="coerce").mean()),
                "positive_rate": float((pd.to_numeric(df[t], errors="coerce") > 0).mean()),
                "p95": float(pd.to_numeric(df[t], errors="coerce").quantile(0.95)),
                "p99": float(pd.to_numeric(df[t], errors="coerce").quantile(0.99)),
            }
            for t in targets
        },
        "top_ic_exec_10s": ic[ic["target"].eq("target_exec_ret_10s")].head(20).to_dict("records"),
        "top_spread_exec_10s": quantiles[quantiles["target"].eq("target_exec_ret_10s")].head(20).to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
