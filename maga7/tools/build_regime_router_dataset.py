#!/usr/bin/env python3
"""Build causal morning features for regime Router (label = oracle day_type).

Features as of ``--asof`` (default 10:30 ET): QQQ microstructure, overnight /
prior-day path, VIXY z, Mag7 breadth. No look-ahead past asof.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.regime import _vixy_z_series
from maga7.common.replay import month_list
from maga7.common.signals import (
    attach_mf_features,
    build_all_first_rule_a_signals,
    load_stock_month_files,
)

FEATURE_COLS = [
    # QQQ session-to-asof
    "qqq_gap_1030",
    "qqq_from_prev_1030",
    "qqq_mf10_1030",
    "qqq_above_open",
    "qqq_bounce_lod",
    "qqq_range",
    "qqq_vol_z",
    "qqq_open_vs_prev",
    "qqq_low_open_reclaim",
    # prior path
    "qqq_prev_day_ret",
    "qqq_prev_day_range",
    "qqq_ret_2d",
    "qqq_ret_3d",
    "qqq_prev_abs_ret",
    # VIXY
    "vixy_z_1030",
    "vixy_gap_1030",
    "vixy_from_prev_1030",
    # Mag7 breadth / pressure
    "n_rule_a",
    "n_rule_a_up",
    "n_rule_a_dn",
    "frac_rule_a_dn",
    "mean_abs_from_prev",
    "mean_vol_z",
    "max_abs_from_prev",
    "breadth_up",
    "breadth_dn",
    "breadth_dn_frac",
    "mag7_mean_open_vs_prev",
    "mag7_frac_above_open",
    "mag7_mean_bounce_lod",
]


def _bar_upto(df: pd.DataFrame, date: str, asof: pd.Timestamp) -> pd.DataFrame:
    day = df[df["date"].astype(str) == str(date)].copy()
    if day.empty:
        return day
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    day["_ts"] = ts
    return day[day["_ts"] <= asof].sort_values("_ts")


def _day_ohlc(df: pd.DataFrame, date: str) -> dict[str, float] | None:
    day = df[df["date"].astype(str) == str(date)].sort_values("timestamp")
    if day.empty:
        return None
    o = float(day.iloc[0]["open"] if "open" in day.columns else day.iloc[0]["close"])
    c = float(day.iloc[-1]["close"])
    lo = float(pd.to_numeric(day["low"] if "low" in day.columns else day["close"], errors="coerce").min())
    hi = float(pd.to_numeric(day["high"] if "high" in day.columns else day["close"], errors="coerce").max())
    if not np.isfinite(o) or o <= 0:
        return None
    return {"open": o, "close": c, "low": lo, "high": hi, "ret": c / o - 1.0, "range": (hi - lo) / o}


def _qqq_feats(qqq: pd.DataFrame, date: str, asof: pd.Timestamp, prev_dates: list[str]) -> dict | None:
    upto = _bar_upto(qqq, date, asof)
    if upto.empty:
        return None
    open_px = float(upto.iloc[0]["open"] if "open" in upto.columns else upto.iloc[0]["close"])
    px = float(upto.iloc[-1]["close"])
    lo = float(pd.to_numeric(upto["low"] if "low" in upto.columns else upto["close"], errors="coerce").min())
    hi = float(pd.to_numeric(upto["high"] if "high" in upto.columns else upto["close"], errors="coerce").max())
    last = upto.iloc[-1]
    fp = float(last["from_prev"]) if "from_prev" in last.index and pd.notna(last["from_prev"]) else 0.0
    mf = float(last["mf10"]) if "mf10" in last.index and pd.notna(last["mf10"]) else 0.0
    vz = float(last["vol_z"]) if "vol_z" in last.index and pd.notna(last["vol_z"]) else 0.0

    prev_close = np.nan
    if prev_dates:
        prev = _day_ohlc(qqq, prev_dates[-1])
        if prev:
            prev_close = float(prev["close"])
    open_vs_prev = (open_px / prev_close - 1.0) if np.isfinite(prev_close) and prev_close > 0 else 0.0
    # low-open reclaim: opened below prev close, but by asof back above today's open
    low_open_reclaim = 1.0 if (open_vs_prev < 0 and px > open_px) else 0.0

    prev_day_ret = prev_day_range = 0.0
    if prev_dates:
        p1 = _day_ohlc(qqq, prev_dates[-1])
        if p1:
            prev_day_ret = float(p1["ret"])
            prev_day_range = float(p1["range"])
    closes = []
    for d in prev_dates[-3:]:
        p = _day_ohlc(qqq, d)
        if p:
            closes.append(float(p["close"]))
    ret_2d = ret_3d = 0.0
    if len(closes) >= 2 and closes[-2] > 0:
        ret_2d = closes[-1] / closes[-2] - 1.0
    if len(closes) >= 3 and closes[-3] > 0:
        ret_3d = closes[-1] / closes[-3] - 1.0

    return {
        "qqq_gap_1030": px / open_px - 1.0 if open_px else 0.0,
        "qqq_from_prev_1030": fp,
        "qqq_mf10_1030": mf,
        "qqq_above_open": 1.0 if px > open_px else 0.0,
        "qqq_bounce_lod": (px / lo - 1.0) if lo > 0 else 0.0,
        "qqq_range": (hi - lo) / open_px if open_px else 0.0,
        "qqq_vol_z": vz,
        "qqq_open_vs_prev": float(open_vs_prev),
        "qqq_low_open_reclaim": float(low_open_reclaim),
        "qqq_prev_day_ret": float(prev_day_ret),
        "qqq_prev_day_range": float(prev_day_range),
        "qqq_ret_2d": float(ret_2d),
        "qqq_ret_3d": float(ret_3d),
        "qqq_prev_abs_ret": float(abs(prev_day_ret)),
    }


def _vixy_feats(vixy: pd.DataFrame | None, date: str, asof: pd.Timestamp, prev_dates: list[str]) -> dict:
    out = {"vixy_z_1030": 0.0, "vixy_gap_1030": 0.0, "vixy_from_prev_1030": 0.0}
    if vixy is None or vixy.empty:
        return out
    upto = _bar_upto(vixy, date, asof)
    if upto.empty:
        return out
    open_px = float(upto.iloc[0]["open"] if "open" in upto.columns else upto.iloc[0]["close"])
    px = float(upto.iloc[-1]["close"])
    last = upto.iloc[-1]
    vz = float(last["vixy_z"]) if "vixy_z" in last.index and pd.notna(last["vixy_z"]) else 0.0
    fp = float(last["from_prev"]) if "from_prev" in last.index and pd.notna(last["from_prev"]) else 0.0
    out["vixy_z_1030"] = vz
    out["vixy_gap_1030"] = px / open_px - 1.0 if open_px else 0.0
    out["vixy_from_prev_1030"] = fp
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--labels", default="maga7/results/regime_router/day_type_labels.csv")
    ap.add_argument("--start-date", default="2025-07-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--asof", default="10:30")
    ap.add_argument("--out", default="maga7/results/regime_router/router_dataset_v2.parquet")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    label_map = {str(r.date): str(r.day_type) for r in lab.itertuples(index=False)}
    expert_set = {"rebound_trap_dn", "dn_toxic", "up_toxic"}

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof["symbols"])
    months = month_list(args.start_date, args.end_date)
    # load a bit of history before start for prior-day features
    hist_start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    months_hist = month_list(hist_start, args.end_date)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in list(dict.fromkeys(symbols + ["QQQ", "VIXY"])):
        raw = load_stock_month_files(paths["stock_root"], sym, months_hist)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= hist_start) & (raw["date"] <= args.end_date)]
        feat = attach_mf_features(
            raw,
            mf_window=int(prof["signal"].get("mf_window", 10)),
            vol_ma_window=int(prof["signal"].get("vol_ma_window", 20)),
        )
        if sym == "VIXY":
            feat = feat.sort_values("timestamp").copy()
            feat["vixy_z"] = _vixy_z_series(feat["close"]).to_numpy()
        stock_by[sym] = feat

    qqq = stock_by.get("QQQ")
    vixy = stock_by.get("VIXY")
    if qqq is None or qqq.empty:
        raise SystemExit("QQQ missing")

    trade_stock = {s: stock_by[s] for s in symbols if s in stock_by}
    sigs = build_all_first_rule_a_signals(trade_stock, prof["signal"])
    if not sigs.empty:
        sigs = sigs[(sigs["date"] >= args.start_date) & (sigs["date"] <= args.end_date)].copy()
        sigs["date"] = sigs["date"].astype(str)

    all_dates = sorted(qqq["date"].astype(str).unique())
    dates = [d for d in all_dates if args.start_date <= d <= args.end_date]

    rows = []
    for date in dates:
        asof = pd.Timestamp(f"{date} {args.asof}", tz="America/New_York")
        prev_dates = [d for d in all_dates if d < date][-3:]
        qf = _qqq_feats(qqq, date, asof, prev_dates)
        if qf is None:
            continue
        vf = _vixy_feats(vixy, date, asof, prev_dates)

        day_sigs = sigs[sigs["date"] == date] if not sigs.empty else pd.DataFrame()
        if not day_sigs.empty:
            st = pd.to_datetime(day_sigs["sig_ts"])
            if getattr(st.dt, "tz", None) is None:
                st = st.dt.tz_localize("America/New_York")
            else:
                st = st.dt.tz_convert("America/New_York")
            day_sigs = day_sigs.loc[st <= asof]
        n_a = len(day_sigs)
        n_up = int((day_sigs["dir"] == "UP").sum()) if n_a else 0
        n_dn = int((day_sigs["dir"] == "DN").sum()) if n_a else 0
        fps: list[float] = []
        vzs: list[float] = []
        if n_a:
            for r in day_sigs.itertuples(index=False):
                sdf = stock_by.get(r.symbol)
                if sdf is None:
                    continue
                upto = _bar_upto(sdf, date, asof)
                if upto.empty:
                    continue
                last = upto.iloc[-1]
                if "from_prev" in last.index and pd.notna(last["from_prev"]):
                    fps.append(abs(float(last["from_prev"])))
                if "vol_z" in last.index and pd.notna(last["vol_z"]):
                    vzs.append(float(last["vol_z"]))

        b_up = b_dn = 0
        open_vs_prev_list: list[float] = []
        above_open_list: list[float] = []
        bounce_list: list[float] = []
        for sym, sdf in trade_stock.items():
            upto = _bar_upto(sdf, date, asof)
            if upto.empty:
                continue
            open_px = float(upto.iloc[0]["open"] if "open" in upto.columns else upto.iloc[0]["close"])
            px = float(upto.iloc[-1]["close"])
            lo = float(pd.to_numeric(upto["low"] if "low" in upto.columns else upto["close"], errors="coerce").min())
            if "mf10" in upto.columns:
                mf = float(upto.iloc[-1]["mf10"])
                if np.isfinite(mf):
                    if mf > 0:
                        b_up += 1
                    elif mf < 0:
                        b_dn += 1
            if open_px > 0:
                above_open_list.append(1.0 if px > open_px else 0.0)
                if lo > 0:
                    bounce_list.append(px / lo - 1.0)
            if prev_dates:
                prev = _day_ohlc(sdf, prev_dates[-1])
                if prev and prev["close"] > 0:
                    open_vs_prev_list.append(open_px / float(prev["close"]) - 1.0)

        y_type = label_map.get(date, "baseline")
        if y_type not in expert_set:
            y_type = "baseline"
        y_need = 0 if y_type == "baseline" else 1
        y_rebound = 1 if y_type == "rebound_trap_dn" else 0

        n_breadth = b_up + b_dn
        row = {
            "date": date,
            "y_type": y_type,
            "y_need_expert": y_need,
            "y_rebound": y_rebound,
            "n_rule_a": float(n_a),
            "n_rule_a_up": float(n_up),
            "n_rule_a_dn": float(n_dn),
            "frac_rule_a_dn": float(n_dn / n_a) if n_a else 0.0,
            "mean_abs_from_prev": float(np.mean(fps)) if fps else 0.0,
            "mean_vol_z": float(np.mean(vzs)) if vzs else 0.0,
            "max_abs_from_prev": float(np.max(fps)) if fps else 0.0,
            "breadth_up": float(b_up),
            "breadth_dn": float(b_dn),
            "breadth_dn_frac": float(b_dn / n_breadth) if n_breadth else 0.0,
            "mag7_mean_open_vs_prev": float(np.mean(open_vs_prev_list)) if open_vs_prev_list else 0.0,
            "mag7_frac_above_open": float(np.mean(above_open_list)) if above_open_list else 0.0,
            "mag7_mean_bounce_lod": float(np.mean(bounce_list)) if bounce_list else 0.0,
            **qf,
            **vf,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "n_rows": int(len(df)),
        "feature_cols": FEATURE_COLS,
        "y_type_counts": df["y_type"].value_counts().to_dict(),
        "y_need_rate": float(df["y_need_expert"].mean()),
        "y_rebound_rate": float(df["y_rebound"].mean()),
        "n_rebound": int(df["y_rebound"].sum()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "asof": args.asof,
        "labels": args.labels,
        "version": "v2_enriched",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
