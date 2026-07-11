#!/usr/bin/env python3
"""Institutional fingerprints F1/F2/F3 on curated 0DTE trades (causal, pre-entry).

F1 Inventory/Replenish — sweep then spread/size recover
F2 Gamma-day — realized range so far vs option premium proxy
F3 Opening-flow — session minute + flow persistence/acceleration

Protocol: features use only data with timestamp <= entry_ts.
Jul must not be used to tune thresholds; report only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_lab.tools.analyze_0dte_july_failure_attribution import classify_row


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
        "--jul-trades",
        default=(
            "factor_lab/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/"
            "trades_all.parquet"
        ),
    )
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--stock-root", default="/mnt/s990/data/raw_1s/stocks/QQQ")
    p.add_argument("--lookback-s", type=int, default=30)
    p.add_argument(
        "--output-dir",
        default="factor_lab/results/0dte_institutional_fingerprints",
    )
    return p.parse_args()


def load_trades(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date_str" not in df.columns:
        df["date_str"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["split"] = split
    df["path_exec_ret"] = pd.to_numeric(df["path_exec_ret"], errors="coerce")
    return df.dropna(subset=["path_exec_ret"])


def load_day_micro(micro_root: Path, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_path = micro_root / "contract_1s" / "QQQ" / f"QQQ_{date_str}.parquet"
    f_path = micro_root / "features_1s" / "QQQ" / f"QQQ_{date_str}.parquet"
    contract = pd.read_parquet(c_path) if c_path.exists() else pd.DataFrame()
    feats = pd.read_parquet(f_path) if f_path.exists() else pd.DataFrame()
    if not contract.empty:
        contract = contract.copy()
        contract["timestamp"] = pd.to_datetime(contract["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
    if not feats.empty:
        feats = feats.copy()
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
    return contract, feats


def load_stock_day(stock_root: Path, date_str: str) -> pd.DataFrame:
    fp = stock_root / f"QQQ_{date_str}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("America/New_York")
    raw = raw.sort_values("timestamp")
    # RTH
    t = raw["timestamp"].dt.time
    raw = raw[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    return raw


def f1_inventory(contract: pd.DataFrame, entry_ts: pd.Timestamp, lookback_s: int) -> dict:
    if contract.empty:
        return {"f1_replenish": np.nan, "f1_quote_thinning": np.nan, "f1_size_asym": np.nan}
    w0 = entry_ts - pd.Timedelta(seconds=lookback_s)
    # near ATM: first put+call buckets
    sub = contract[
        (contract["timestamp"] > w0)
        & (contract["timestamp"] <= entry_ts)
        & (contract["bucket_id"].isin([0, 1, 4, 5]))
    ].copy()
    if sub.empty:
        return {"f1_replenish": np.nan, "f1_quote_thinning": np.nan, "f1_size_asym": np.nan}
    # aggregate per second across near-ATM
    g = (
        sub.groupby("timestamp", as_index=False)
        .agg(
            spread_pct=("spread_pct", "mean"),
            bid_size=("bid_size", "sum"),
            ask_size=("ask_size", "sum"),
            net_buy=("net_buy_volume", "sum"),
            trade_volume=("trade_volume", "sum"),
        )
        .sort_values("timestamp")
    )
    if len(g) < 5:
        return {"f1_replenish": np.nan, "f1_quote_thinning": np.nan, "f1_size_asym": np.nan}

    day_med_spread = float(pd.to_numeric(contract["spread_pct"], errors="coerce").median())
    thin = float(g["spread_pct"].mean() / day_med_spread) if day_med_spread > 0 else np.nan
    size_sum = (g["bid_size"] + g["ask_size"]).replace(0, np.nan)
    asym = float(((g["bid_size"] - g["ask_size"]).abs() / size_sum).mean())

    # sweep: high |net_buy| seconds; replenish if next 3s spread drops
    abs_nb = g["net_buy"].abs()
    thr = abs_nb.quantile(0.8)
    sweeps = abs_nb >= thr
    replenish_hits = 0
    replenish_n = 0
    spreads = g["spread_pct"].to_numpy()
    for i in np.where(sweeps.to_numpy())[0]:
        if i + 3 >= len(g):
            continue
        replenish_n += 1
        if spreads[i + 3] < spreads[i] * 0.98:
            replenish_hits += 1
    repl = replenish_hits / replenish_n if replenish_n else np.nan
    return {"f1_replenish": repl, "f1_quote_thinning": thin, "f1_size_asym": asym}


def f2_gamma_day(stock: pd.DataFrame, contract: pd.DataFrame, entry_ts: pd.Timestamp) -> dict:
    if stock.empty:
        return {"f2_range_so_far": np.nan, "f2_premium_proxy": np.nan, "f2_rv_vs_prem": np.nan}
    so_far = stock[stock["timestamp"] <= entry_ts]
    if so_far.empty:
        return {"f2_range_so_far": np.nan, "f2_premium_proxy": np.nan, "f2_rv_vs_prem": np.nan}
    open_px = float(so_far.iloc[0]["open"] if pd.notna(so_far.iloc[0]["open"]) else so_far.iloc[0]["close"])
    hi = float(so_far["high"].max())
    lo = float(so_far["low"].min())
    rng = (hi - lo) / open_px if open_px > 0 else np.nan
    spot = float(so_far.iloc[-1]["close"])
    prem = np.nan
    if not contract.empty:
        near = contract[
            (contract["timestamp"] <= entry_ts)
            & (contract["timestamp"] > entry_ts - pd.Timedelta(seconds=5))
            & (contract["bucket_id"].isin([0, 4]))
        ]
        if near.empty:
            near = contract[
                (contract["timestamp"] <= entry_ts) & (contract["bucket_id"].isin([0, 4]))
            ].tail(20)
        if not near.empty and spot > 0:
            prem = float(pd.to_numeric(near["mid"], errors="coerce").mean() / spot)
    rv_vs = rng / (prem + 1e-6) if np.isfinite(rng) and np.isfinite(prem) else np.nan
    return {"f2_range_so_far": rng, "f2_premium_proxy": prem, "f2_rv_vs_prem": rv_vs}


def f3_opening_flow(feats: pd.DataFrame, entry_ts: pd.Timestamp, lookback_s: int) -> dict:
    open_ts = entry_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
    # handle DST: entry_ts already tz-aware NY
    if entry_ts.tzinfo is not None:
        open_ts = pd.Timestamp(f"{entry_ts.date()} 09:30:00", tz=entry_ts.tz)
    session_min = float((entry_ts - open_ts).total_seconds() / 60.0)
    if feats.empty:
        return {
            "f3_session_minute": session_min,
            "f3_flow_persist": np.nan,
            "f3_flow_accel": np.nan,
        }
    w0 = entry_ts - pd.Timedelta(seconds=lookback_s)
    sub = feats[(feats["timestamp"] > w0) & (feats["timestamp"] <= entry_ts)].sort_values("timestamp")
    if sub.empty or "cp_net_buy_diff" not in sub.columns:
        return {
            "f3_session_minute": session_min,
            "f3_flow_persist": np.nan,
            "f3_flow_accel": np.nan,
        }
    nb = pd.to_numeric(sub["cp_net_buy_diff"], errors="coerce").fillna(0.0)
    if nb.abs().sum() == 0:
        persist = np.nan
    else:
        # persistence vs dominant sign in window
        dom = np.sign(nb.sum()) or 1.0
        persist = float((np.sign(nb) == dom).mean())
    # accel: last 10s sum vs previous 10s
    if len(sub) >= 20:
        last = nb.iloc[-10:].sum()
        prev = nb.iloc[-20:-10].sum()
        accel = float(last - prev)
    else:
        accel = np.nan
    return {
        "f3_session_minute": session_min,
        "f3_flow_persist": persist,
        "f3_flow_accel": accel,
    }


def attach_fingerprints(trades: pd.DataFrame, micro_root: Path, stock_root: Path, lookback_s: int) -> pd.DataFrame:
    rows = []
    for date_str, day_tr in trades.groupby("date_str"):
        contract, feats = load_day_micro(micro_root, date_str)
        stock = load_stock_day(stock_root, date_str)
        for _, tr in day_tr.iterrows():
            entry = tr["timestamp"]
            rec = tr.to_dict()
            rec.update(f1_inventory(contract, entry, lookback_s))
            rec.update(f2_gamma_day(stock, contract, entry))
            rec.update(f3_opening_flow(feats, entry, lookback_s))
            bucket, reason = classify_row(pd.Series(rec))
            rec["fail_bucket"] = bucket
            rec["fail_reason"] = reason
            rows.append(rec)
        print(f"[fp] {date_str} n={len(day_tr)}", flush=True)
    return pd.DataFrame(rows)


def tercile_table(df: pd.DataFrame, col: str, ret_col: str = "path_exec_ret") -> list[dict]:
    s = pd.to_numeric(df[col], errors="coerce")
    valid = df[s.notna()].copy()
    if valid.empty:
        return []
    try:
        valid["_bin"] = pd.qcut(pd.to_numeric(valid[col], errors="coerce"), 3, labels=["low", "mid", "high"])
    except ValueError:
        return []
    out = []
    for b, g in valid.groupby("_bin", observed=True):
        out.append(
            {
                "feature": col,
                "tercile": str(b),
                "n": int(len(g)),
                "avg_ret": float(g[ret_col].mean()),
                "win_rate": float((g[ret_col] > 0).mean()),
                "share_B_C_D": float(g["fail_bucket"].isin(["B_direction_ok_option_dead", "C_mfe_but_exit_fail", "D_spread_execution"]).mean()),
                "share_A": float((g["fail_bucket"] == "A_direction_wrong").mean()),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jan = load_trades(Path(args.trades), "jan_jun")
    jul = load_trades(Path(args.jul_trades), "jul_oos")
    all_tr = pd.concat([jan, jul], ignore_index=True)
    print(f"[fp] trades={len(all_tr)} days={all_tr['date_str'].nunique()}", flush=True)

    scored = attach_fingerprints(all_tr, Path(args.micro_root), Path(args.stock_root), args.lookback_s)
    scored.to_parquet(out / "trades_with_fingerprints.parquet", index=False)

    diag_rows = []
    for split, g in scored.groupby("split"):
        for col in ["f1_replenish", "f1_quote_thinning", "f2_rv_vs_prem", "f3_flow_persist", "f3_session_minute"]:
            for row in tercile_table(g, col):
                row["split"] = split
                diag_rows.append(row)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out / "fingerprint_terciles.csv", index=False)

    # Jan-Jun separation summary for promotion heuristic
    jj = scored[scored["split"] == "jan_jun"]
    summary = {
        "experiment": "institutional_fingerprints_f1_f2_f3",
        "n_trades": int(len(scored)),
        "lookback_s": args.lookback_s,
        "definitions": {
            "F1": "inventory replenish after sweep; quote thinning; size asymmetry",
            "F2": "realized range so far / ATM mid premium proxy",
            "F3": "session minute + cp flow persistence/accel",
        },
        "jan_jun_terciles": diag[diag["split"] == "jan_jun"].to_dict(orient="records") if not diag.empty else [],
        "jul_oos_terciles": diag[diag["split"] == "jul_oos"].to_dict(orient="records") if not diag.empty else [],
        "next": [
            "If F1 high tercile has higher B/C/D share → wire into Tradeable",
            "If F2 low tercile has worse 0DTE avg → route those days to 1DTE when micro ready",
            "If F3 low persist + late session hurts → add to confirm/timing",
            "Do not tune on Jul",
        ],
        "files": {
            "trades": str(out / "trades_with_fingerprints.parquet"),
            "terciles": str(out / "fingerprint_terciles.csv"),
            "summary": str(out / "summary.json"),
        },
    }
    # quick headline: F1 replenish high vs low avg ret on jan_jun
    for col in ["f1_replenish", "f2_rv_vs_prem", "f3_flow_persist"]:
        sub = diag[(diag["split"] == "jan_jun") & (diag["feature"] == col)]
        if len(sub) >= 2:
            lo = sub[sub["tercile"] == "low"]
            hi = sub[sub["tercile"] == "high"]
            if not lo.empty and not hi.empty:
                summary[f"headline_{col}"] = {
                    "low_avg_ret": float(lo.iloc[0]["avg_ret"]),
                    "high_avg_ret": float(hi.iloc[0]["avg_ret"]),
                    "low_share_BCD": float(lo.iloc[0]["share_B_C_D"]),
                    "high_share_BCD": float(hi.iloc[0]["share_B_C_D"]),
                }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k.startswith("headline_") or k in ("n_trades", "next")}, indent=2))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
