#!/usr/bin/env python3
"""Grid-search causal ladder target: maximize 1s scalp PnL (not day_lock match).

Frozen entry schedule from an existing trades CSV (sig_ts/spot). For each
(alpha, H) or speed-horizon, pick open-ladder strike nearest to projected
spot and simulate_trade on 1s quotes.

  target = spot + alpha * (spot - lock_spot)     # alpha mode
  target = spot + speed_$per_min * H_minutes     # speed mode

Usage:
  python -m maga7.tools.run_ladder_project_grid \\
    --trades maga7/results/open_ladder_ab_1s_otm5_jan_jul/day_lock/trades.csv \\
    --ladder-profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json
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
from maga7.common.fills import FillSpec
from maga7.common.open_lock import direction_ladder_buckets, strike_from_occ
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files


def _load_lock_spot_index(lock_path: Path) -> dict[tuple[str, str], float]:
    mp = pd.read_parquet(lock_path)
    out: dict[tuple[str, str], float] = {}
    for (sym, date), g in mp.groupby(["symbol", "date_str"]):
        if "lock_spot" in g.columns and g["lock_spot"].notna().any():
            out[(str(sym), str(date))] = float(g["lock_spot"].dropna().iloc[0])
    return out


def _ladder_strikes(
    mp: pd.DataFrame,
    *,
    symbol: str,
    date: str,
    direction: str,
    otm_rungs: int,
) -> list[tuple[str, float, int]]:
    """Return [(ticker, strike, dte), ...] preferring dte order 0,1,2 but keeping all."""
    g = mp[(mp["symbol"] == symbol) & (mp["date_str"].astype(str) == str(date))]
    if g.empty:
        return []
    bids = direction_ladder_buckets(direction, otm_rungs=otm_rungs)
    rows: list[tuple[str, float, int]] = []
    for dte in [0, 1, 2]:
        gd = g[g["front_dte"] == dte]
        for b in bids:
            sub = gd[gd["bucket_id"] == b]
            if sub.empty:
                continue
            t = str(sub.iloc[0]["contract_symbol"]).replace("O:", "")
            k = float(sub.iloc[0]["strike"]) if "strike" in sub.columns else strike_from_occ(t)
            rows.append((t, k, int(dte)))
    return rows


def _pick_nearest(cands: list[tuple[str, float, int]], target: float, prefer_dte: int = 0) -> tuple[str, float, int] | None:
    if not cands:
        return None
    # Prefer prefer_dte if it has candidates within same distance band, else global nearest
    pref = [c for c in cands if c[2] == prefer_dte]
    pool = pref if pref else cands
    return min(pool, key=lambda x: abs(x[1] - target))


def _speed_usd_per_min(stock_day: pd.DataFrame, sig_ts, window: int = 10) -> float:
    ts = to_ny(sig_ts)
    hist = stock_day[stock_day["timestamp"] <= ts]
    if len(hist) < 3:
        return 0.0
    w = hist.tail(window)
    dt = max((w["timestamp"].iloc[-1] - w["timestamp"].iloc[0]).total_seconds() / 60.0, 1e-6)
    return (float(w["close"].iloc[-1]) - float(w["close"].iloc[0])) / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        default=str(ROOT / "maga7/results/open_ladder_ab_1s_otm5_jan_jul/day_lock/trades.csv"),
        help="frozen entry schedule (needs date,symbol,dir/direction,sig_ts,sig_spot)",
    )
    ap.add_argument(
        "--ladder-profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_v1.json"),
    )
    ap.add_argument("--tag", default="ladder_project_grid_1s_otm5")
    ap.add_argument("--hold-minutes", type=int, default=30, help="trade hold for TP/SL clock (fill sim)")
    args = ap.parse_args()

    profile = load_profile(args.ladder_profile)
    trade = profile.get("trade") or {}
    otm_rungs = int(trade.get("ladder_otm_rungs") or (profile.get("lock") or {}).get("otm_rungs") or 5)
    quote_root = Path(profile["_paths"]["quote_1s_root"])
    lock_path = Path(profile["_paths"]["open_locked_map"])
    mp = pd.read_parquet(lock_path)
    mp["date_str"] = mp["date_str"].astype(str)
    lock_spot_idx = _load_lock_spot_index(lock_path)

    fill = FillSpec(
        entry_frac=float(profile["fill"].get("entry_frac", 0.8)),
        exit_frac=float(profile["fill"].get("exit_frac", 0.8)),
    )
    tp = float(trade.get("tp_mult", 1.6))
    sl = float(trade.get("sl_mult", 0.4))

    tr = pd.read_csv(args.trades)
    if "direction" not in tr.columns:
        tr["direction"] = tr["dir"]
    tr["date"] = tr["date"].astype(str)
    tr["sig_ts"] = pd.to_datetime(tr["sig_ts"])
    # unique entries by key
    keys = ["date", "symbol", "direction", "n_in_day"]
    tr = tr.drop_duplicates(keys, keep="first")

    # stock for speed
    months = month_list(profile["date_range"]["start"], profile["date_range"]["end"])
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(profile["_paths"]["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= profile["date_range"]["start"]) & (raw["date"] <= profile["date_range"]["end"])]
        stock_by[sym] = attach_mf_features(raw, mf_window=10, vol_ma_window=20)

    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def get_q(sym: str, date: str):
        k = (sym, date)
        if k not in quote_cache:
            quote_cache[k] = load_quotes(quote_root, sym, date)
        return quote_cache[k]

    # Precompute per-row features
    feats = []
    for r in tr.itertuples():
        sym, date = str(r.symbol), str(r.date)
        spot = float(r.sig_spot)
        lock_spot = lock_spot_idx.get((sym, date), spot)
        sdf = stock_by.get(sym)
        speed = 0.0
        if sdf is not None:
            day = sdf[sdf["date"] == date].copy()
            if not day.empty:
                day["timestamp"] = pd.to_datetime(day["timestamp"])
                if day["timestamp"].dt.tz is None:
                    day["timestamp"] = day["timestamp"].dt.tz_localize("America/New_York")
                else:
                    day["timestamp"] = day["timestamp"].dt.tz_convert("America/New_York")
                speed = _speed_usd_per_min(day, r.sig_ts, window=10)
        cands = _ladder_strikes(mp, symbol=sym, date=date, direction=str(r.direction), otm_rungs=otm_rungs)
        feats.append(
            {
                "date": date,
                "symbol": sym,
                "direction": str(r.direction),
                "n_in_day": int(r.n_in_day),
                "sig_ts": r.sig_ts,
                "spot": spot,
                "lock_spot": lock_spot,
                "speed": speed,
                "day_ticker": str(getattr(r, "ticker", "")).replace("O:", ""),
                "cands": cands,
            }
        )

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    horizons = [0, 5, 10, 15, 20, 30]  # 0 == nearest/speed unused

    def eval_rule(name: str, target_fn) -> dict:
        rets = []
        n_miss = 0
        n_same_day = 0
        for f in feats:
            if not f["cands"]:
                n_miss += 1
                continue
            tgt = float(target_fn(f))
            pick = _pick_nearest(f["cands"], tgt, prefer_dte=0)
            if pick is None:
                n_miss += 1
                continue
            ticker, _k, _dte = pick
            if ticker == f["day_ticker"]:
                n_same_day += 1
            path = path_for_ticker(get_q(f["symbol"], f["date"]), ticker)
            sim = simulate_trade(
                path,
                f["sig_ts"],
                fill=fill,
                tp_mult=tp,
                sl_mult=sl,
                hold_minutes=int(args.hold_minutes),
            )
            if sim is None:
                n_miss += 1
                continue
            rets.append(sim.ret)
        arr = np.asarray(rets, dtype=float) if rets else np.asarray([])
        return {
            "rule": name,
            "n": int(len(arr)),
            "n_miss": int(n_miss),
            "mean_ret": float(arr.mean()) if len(arr) else None,
            "sum_ret": float(arr.sum()) if len(arr) else None,
            "win_rate": float((arr > 0).mean()) if len(arr) else None,
            "p50": float(np.median(arr)) if len(arr) else None,
            "same_day_ticker_rate": float(n_same_day / max(len(arr) + n_miss, 1)),
        }

    rows = []
    # baselines
    rows.append(eval_rule("nearest_spot", lambda f: f["spot"]))
    rows.append(
        eval_rule(
            "day_lock_ticker",
            # force day ticker if on ladder else nearest
            lambda f: next((k for t, k, _ in f["cands"] if t == f["day_ticker"]), f["spot"]),
        )
    )
    # For day_lock_ticker we need special pick - fix properly
    def eval_day_lock_ticker() -> dict:
        rets = []
        n_miss = 0
        n_hit = 0
        for f in feats:
            ticker = f["day_ticker"]
            on = any(t == ticker for t, _, _ in f["cands"])
            if on:
                n_hit += 1
            else:
                pick = _pick_nearest(f["cands"], f["spot"], prefer_dte=0) if f["cands"] else None
                if pick is None:
                    n_miss += 1
                    continue
                ticker = pick[0]
            path = path_for_ticker(get_q(f["symbol"], f["date"]), ticker)
            sim = simulate_trade(
                path,
                f["sig_ts"],
                fill=fill,
                tp_mult=tp,
                sl_mult=sl,
                hold_minutes=int(args.hold_minutes),
            )
            if sim is None:
                n_miss += 1
                continue
            rets.append(sim.ret)
        arr = np.asarray(rets, dtype=float) if rets else np.asarray([])
        return {
            "rule": "day_lock_ticker",
            "n": int(len(arr)),
            "n_miss": int(n_miss),
            "mean_ret": float(arr.mean()) if len(arr) else None,
            "sum_ret": float(arr.sum()) if len(arr) else None,
            "win_rate": float((arr > 0).mean()) if len(arr) else None,
            "p50": float(np.median(arr)) if len(arr) else None,
            "same_day_ticker_rate": float(n_hit / max(len(feats), 1)),
        }

    rows = [
        eval_rule("nearest_spot", lambda f: f["spot"]),
        eval_day_lock_ticker(),
    ]
    for a in alphas:
        rows.append(
            eval_rule(
                f"alpha_{a}",
                lambda f, a=a: f["spot"] + a * (f["spot"] - f["lock_spot"]),
            )
        )
    for h in horizons:
        if h == 0:
            continue
        rows.append(
            eval_rule(
                f"speed_H{h}",
                lambda f, h=h: f["spot"] + f["speed"] * h,
            )
        )

    # combo: mild alpha + short speed
    for a in [0.25, 0.5]:
        for h in [5, 10]:
            rows.append(
                eval_rule(
                    f"alpha_{a}_speed_H{h}",
                    lambda f, a=a, h=h: f["spot"] + a * (f["spot"] - f["lock_spot"]) + f["speed"] * h,
                )
            )

    df = pd.DataFrame(rows).sort_values("mean_ret", ascending=False)
    out_dir = Path(profile["_paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "grid.csv", index=False)
    best = df.iloc[0].to_dict()
    near = df[df["rule"] == "nearest_spot"].iloc[0].to_dict()
    day = df[df["rule"] == "day_lock_ticker"].iloc[0].to_dict()
    summary = {
        "trades_schedule": str(args.trades),
        "n_schedule": int(len(feats)),
        "quote_root": str(quote_root),
        "otm_rungs": otm_rungs,
        "best": best,
        "nearest_spot": near,
        "day_lock_ticker": day,
        "beats_nearest": bool(best["rule"] != "nearest_spot" and (best.get("mean_ret") or -1) > (near.get("mean_ret") or 0)),
        "beats_day_lock": bool((best.get("mean_ret") or -1) > (day.get("mean_ret") or 0)),
        "top10": df.head(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("\nfull grid (mean_ret desc):")
    print(df.to_string(index=False))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
