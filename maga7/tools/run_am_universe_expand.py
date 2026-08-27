#!/usr/bin/env python3
"""AM pulse universe expand: Mag7 vs +MU vs +MU+AVGO (mixed seats vs overlay).

Frozen cell: FO≥0.8% DN, TP15/SL20, sp≤15%, lag≤5s.
Sessions: AM 09:30–10:25 + EXT 10:25–11:30 (independent seat pools).

Modes:
  mix_u8      — Mag7+GOOGL only, max_concurrent=2 @20%
  mix_u9      — +MU into same 2 seats
  mix_u10     — +MU+AVGO into same 2 seats
  overlay_mu  — u8 book + independent MU-only pool (1 seat @10%)
  overlay_x2  — u8 book + independent MU+AVGO pool (1 seat @10%)

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_universe_expand \\
    --tag research_am_universe_expand_20260728 --stock-from-1s
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.am_pulse_scout import parse_am_pulse_scout, scan_day
from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import load_symbol_1s_bars, session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path
from maga7.tools.scan_session_horizon_foresight import _spot_at_arr, _stock_arrays

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
MAG7 = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
LOCK_DEFAULT = (
    "/home/kingfang007/train_data/"
    "locked_targets_map_maga7_googl_mu_avgo_open_ladder_atm5otm_research.parquet"
)
SESSIONS = (
    ("AM", "09:30", "10:25"),
    ("EXT", "10:25", "11:30"),
)
WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("apr", "2026-04-01", "2026-04-30"),
    ("may_jul", "2026-05-01", "2026-07-09"),
)


def _spot_from_1m(day: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if day is None or day.empty:
        return None
    t = to_ny(ts)
    sub = day[pd.to_datetime(day["timestamp"]) <= t]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def _equity(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        return {"compound": 0.0, "maxdd": 0.0}
    eq = peak = 1.0
    maxdd = 0.0
    for _, r in df.sort_values(["date", "entry_ts"]).iterrows():
        eq *= 1.0 + float(r["pnl_frac"])
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
    return {"compound": float(eq - 1.0), "maxdd": float(maxdd)}


def _stack_lanes(
    trades: pd.DataFrame,
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float = 10.0,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    parts: list[dict] = []
    for lane in ("AM", "EXT"):
        sub = trades[trades["lane"] == lane]
        by_d: dict[str, list] = {}
        for _, r in sub.iterrows():
            by_d.setdefault(str(r["date"]), []).append(r.to_dict())
        for _, rs in sorted(by_d.items()):
            parts.extend(
                _portfolio_day(
                    sorted(rs, key=lambda x: (str(x["entry_ts"]), str(x["symbol"]))),
                    position_frac=position_frac,
                    max_concurrent=max_concurrent,
                    cooldown_minutes=cooldown_minutes,
                )
            )
    return pd.DataFrame(parts)


def _summarize(df: pd.DataFrame, label: str) -> dict[str, Any]:
    if df is None or df.empty:
        out: dict[str, Any] = {
            "mode": label,
            "n": 0,
            "win": None,
            "add": 0.0,
            "mean": None,
            "compound": 0.0,
            "maxdd": 0.0,
            "n_loss": 0,
            "loss_pnl": 0.0,
        }
    else:
        eq = _equity(df)
        out = {
            "mode": label,
            "n": int(len(df)),
            "win": float((df["ret"] > 0).mean()),
            "add": float(df["pnl_frac"].sum()),
            "mean": float(df["ret"].mean()),
            "compound": eq["compound"],
            "maxdd": eq["maxdd"],
            "n_loss": int((df["ret"] <= 0).sum()),
            "loss_pnl": float(df.loc[df["ret"] <= 0, "pnl_frac"].sum()),
        }
        for sym, g in df.groupby("symbol"):
            out[f"n_{sym}"] = int(len(g))
            out[f"add_{sym}"] = float(g["pnl_frac"].sum())
    for w, a, b in WINDOWS:
        if df is None or df.empty:
            out[f"{w}_n"] = 0
            out[f"{w}_add"] = 0.0
            out[f"{w}_win"] = None
            continue
        sub = df[(df["date"].astype(str) >= a) & (df["date"].astype(str) <= b)]
        out[f"{w}_n"] = int(len(sub))
        out[f"{w}_add"] = float(sub["pnl_frac"].sum()) if len(sub) else 0.0
        out[f"{w}_win"] = float((sub["ret"] > 0).mean()) if len(sub) else None
        out[f"{w}_maxdd"] = _equity(sub)["maxdd"] if len(sub) else 0.0
    return out


def _scan_fills(
    *,
    symbols: list[str],
    dates: list[str],
    stock_by_sym: dict[str, pd.DataFrame],
    lock: dict,
    quote_root: Path,
    stock_1s: Path,
    otm: int,
    fill: FillSpec,
    thr: float,
    tp: float,
    sl: float,
    max_sp: float,
    max_lag: float,
    max_hold_sec: int,
    min_mid: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skip = {"no_stock": 0, "no_lock": 0, "no_quote": 0, "no_ticker": 0, "no_fill": 0}
    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"  [day] {date} ({di+1}/{len(dates)}) fills={len(rows)}", flush=True)
        for sym in symbols:
            sdf = stock_by_sym.get(sym)
            if sdf is None:
                skip["no_stock"] += 1
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                skip["no_stock"] += 1
                continue
            by_dte = lock.get((sym, date))
            if not by_dte:
                skip["no_lock"] += 1
                continue
            try:
                qday = _prep_path(load_quotes(quote_root, sym, date))
            except Exception:
                qday = None
            if qday is None or qday.empty:
                skip["no_quote"] += 1
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                ts_ns, px = _stock_arrays(day1s)

            for lane, w0, w1 in SESSIONS:
                cfg = parse_am_pulse_scout(
                    {
                        "enabled": True,
                        "window_start": w0,
                        "window_end": w1,
                        "min_fav_from_open": thr,
                        "lookback_bars": 2,
                        "min_lookback_ret": 0.99,
                        "dirs": ["DN"],
                        "max_alerts_per_symbol": 1,
                    }
                )
                for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                    if a.arm != "FO" or a.dir != "DN":
                        continue
                    arm_ts = to_ny(pd.Timestamp(a.ts))
                    spot = None
                    if ts_ns is not None and px is not None:
                        spot = _spot_at_arr(ts_ns, px, arm_ts)
                    if spot is None:
                        spot = _spot_from_1m(day1m, arm_ts)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction=a.dir,
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=[0, 1, 2],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        skip["no_ticker"] += 1
                        continue
                    path = _prep_path(path_for_ticker(qday, ticker))
                    if path is None or path.empty:
                        skip["no_ticker"] += 1
                        continue
                    probe = entry_quote_row(
                        path,
                        arm_ts,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=min_mid,
                    )
                    if probe is None:
                        skip["no_fill"] += 1
                        continue
                    sim = simulate_quote_tpsl(
                        path,
                        arm_ts,
                        tp=tp,
                        sl=sl,
                        max_hold_sec=max_hold_sec,
                        fill=fill,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=min_mid,
                    )
                    if sim is None or not np.isfinite(sim["ret"]):
                        skip["no_fill"] += 1
                        continue
                    rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": "DN",
                            "lane": lane,
                            "entry_ts": sim["entry_ts"],
                            "exit_ts": sim["exit_ts"],
                            "ticker": ticker,
                            "dte": dte,
                            "ret": float(sim["ret"]),
                            "exit_reason": sim["reason"],
                            "hold_sec": float(sim["hold_sec"]),
                            "mfe": float(sim["mfe"]),
                            "mae": float(sim["mae"]),
                        }
                    )
    print(f"  fills={len(rows)} skip={skip}", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_universe_expand_20260728")
    ap.add_argument("--open-locked-map", default=LOCK_DEFAULT)
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-09")
    ap.add_argument("--thr", type=float, default=0.008)
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.20)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--sleeve-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--stock-from-1s", action="store_true")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    symbols = MAG7 + ["MU", "AVGO"]
    lock = load_multidte_lock_index(Path(args.open_locked_map).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    fill = FillSpec(0.75, 0.75)
    quote_root = Path(paths["quote_1s_root"])
    stock_root = Path(paths["stock_root"])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()

    dates = [d for d in session_dates(args.start, args.end) if args.start <= d <= args.end]
    months = month_list(args.start, args.end)
    print(
        f"AM universe expand {args.start}..{args.end} symbols={symbols} "
        f"lock={Path(args.open_locked_map).name}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    if args.stock_from_1s:
        for sym in symbols:
            sdf = load_symbol_1s_bars(stock_1s, sym, dates, bar_seconds=60)
            if sdf is not None and not sdf.empty:
                stock_by_sym[sym] = sdf
                print(f"  stock1s {sym}: days={sdf['date'].nunique()}", flush=True)
    else:
        for sym in symbols:
            sdf = load_stock_month_files(stock_root, sym, months)
            if sdf is not None and not sdf.empty:
                stock_by_sym[sym] = sdf

    raw = _scan_fills(
        symbols=symbols,
        dates=dates,
        stock_by_sym=stock_by_sym,
        lock=lock,
        quote_root=quote_root,
        stock_1s=stock_1s,
        otm=otm,
        fill=fill,
        thr=args.thr,
        tp=args.tp,
        sl=args.sl,
        max_sp=args.max_spread_pct,
        max_lag=args.max_lag_sec,
        max_hold_sec=args.max_hold_sec,
        min_mid=0.05,
    )
    raw.to_csv(out / "trades_raw_all_symbols.csv", index=False)

    u8 = raw[raw["symbol"].isin(MAG7)].copy()
    u9 = raw[raw["symbol"].isin(MAG7 + ["MU"])].copy()
    u10 = raw.copy()
    mu_only = raw[raw["symbol"] == "MU"].copy()
    x2_only = raw[raw["symbol"].isin(["MU", "AVGO"])].copy()

    pf = float(args.position_frac)
    mc = int(args.max_concurrent)
    sf = float(args.sleeve_frac)

    books: dict[str, pd.DataFrame] = {
        "mix_u8": _stack_lanes(u8, position_frac=pf, max_concurrent=mc),
        "mix_u9": _stack_lanes(u9, position_frac=pf, max_concurrent=mc),
        "mix_u10": _stack_lanes(u10, position_frac=pf, max_concurrent=mc),
    }
    # Overlay: Mag7 main book + independent small sleeve (does not share seats).
    base_book = books["mix_u8"]
    mu_sleeve = _stack_lanes(mu_only, position_frac=sf, max_concurrent=1)
    x2_sleeve = _stack_lanes(x2_only, position_frac=sf, max_concurrent=1)
    if len(base_book) and len(mu_sleeve):
        books["overlay_mu"] = pd.concat([base_book, mu_sleeve], ignore_index=True)
    else:
        books["overlay_mu"] = base_book.copy() if len(base_book) else mu_sleeve
    if len(base_book) and len(x2_sleeve):
        books["overlay_x2"] = pd.concat([base_book, x2_sleeve], ignore_index=True)
    else:
        books["overlay_x2"] = base_book.copy() if len(base_book) else x2_sleeve

    # Crowding audit: Mag7 trades dropped when mixing
    def _keys(df: pd.DataFrame) -> set[str]:
        if df is None or df.empty:
            return set()
        return {
            f"{r.date}|{r.symbol}|{r.lane}|{r.entry_ts}"
            for r in df.itertuples()
        }

    k8 = _keys(books["mix_u8"])
    crowd = {
        "mix_u9_displaces_mag7": sorted(k8 - _keys(books["mix_u9"])),
        "mix_u10_displaces_mag7": sorted(k8 - _keys(books["mix_u10"])),
        "n_mu_raw": int(len(mu_only)),
        "n_avgo_raw": int(len(raw[raw.symbol == "AVGO"])) if len(raw) else 0,
        "mu_raw_mean": float(mu_only["ret"].mean()) if len(mu_only) else None,
        "avgo_raw_mean": float(raw.loc[raw.symbol == "AVGO", "ret"].mean())
        if len(raw) and (raw.symbol == "AVGO").any()
        else None,
        "mu_raw_win": float((mu_only["ret"] > 0).mean()) if len(mu_only) else None,
        "avgo_raw_win": float((raw.loc[raw.symbol == "AVGO", "ret"] > 0).mean())
        if len(raw) and (raw.symbol == "AVGO").any()
        else None,
    }

    score_rows = [_summarize(df, name) for name, df in books.items()]
    # retain vs mix_u8
    base_add = float(score_rows[0]["add"]) if score_rows else 0.0
    for r in score_rows:
        r["retain_vs_u8"] = (
            float(r["add"]) / base_add if abs(base_add) > 1e-12 else float("nan")
        )
        for w, _, _ in WINDOWS:
            b = float(score_rows[0].get(f"{w}_add") or 0.0)
            v = float(r.get(f"{w}_add") or 0.0)
            r[f"{w}_retain"] = (v / b) if abs(b) > 1e-12 else float("nan")

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    for name, df in books.items():
        if len(df):
            df.to_csv(out / f"book_{name}.csv", index=False)
    if len(mu_only):
        mu_only.to_csv(out / "trades_raw_mu.csv", index=False)
    if len(raw) and (raw.symbol == "AVGO").any():
        raw[raw.symbol == "AVGO"].to_csv(out / "trades_raw_avgo.csv", index=False)

    # Verdict: promote only if dual jan_mar+may_jul retain>=1.0 and add up, no big displace
    def _verdict(r: dict[str, Any]) -> str:
        if r["mode"] == "mix_u8":
            return "BASE"
        jm = float(r.get("jan_mar_retain") or 0.0)
        mj = float(r.get("may_jul_retain") or 0.0)
        if r["mode"].startswith("overlay"):
            # overlay should not hurt base; sleeve add > 0 both windows preferred
            if jm >= 1.0 and mj >= 1.0 and float(r["add"]) > base_add:
                return "PASS_OVERLAY"
            if float(r["add"]) > base_add and min(jm, mj) >= 0.98:
                return "WEAK_OVERLAY"
            return "FAIL"
        # mixed: need both windows not collapse; prefer lift
        if jm >= 0.95 and mj >= 0.95 and float(r["add"]) >= base_add * 1.05:
            return "PASS_MIX"
        if jm >= 0.90 and mj >= 0.90 and float(r["add"]) >= base_add:
            return "WEAK_MIX"
        return "FAIL"

    for r in score_rows:
        r["verdict"] = _verdict(r)

    summary = {
        "tag": args.tag,
        "range": f"{args.start}..{args.end}",
        "cell": f"FO_t{args.thr}_tp{args.tp}_sl{args.sl}_sp{args.max_spread_pct}_lag{args.max_lag_sec}",
        "position_frac": pf,
        "sleeve_frac": sf,
        "max_concurrent": mc,
        "crowd": crowd,
        "scoreboard": score_rows,
        "note": (
            "AM-only universe expand (NOT CORE). "
            "mix_* share 2 seats; overlay_* keeps Mag7 seats + small independent sleeve."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    score2 = pd.DataFrame(score_rows)
    score2.to_csv(out / "scoreboard.csv", index=False)

    cols = [
        "mode",
        "verdict",
        "n",
        "win",
        "add",
        "retain_vs_u8",
        "jan_mar_add",
        "apr_add",
        "may_jul_add",
        "jan_mar_retain",
        "may_jul_retain",
        "maxdd",
        "n_MU",
        "add_MU",
        "n_AVGO",
        "add_AVGO",
    ]
    show = score2[[c for c in cols if c in score2.columns]]
    print("\n=== AM universe expand ===", flush=True)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(
        f"\ncrowd displace u9={len(crowd['mix_u9_displaces_mag7'])} "
        f"u10={len(crowd['mix_u10_displaces_mag7'])} "
        f"MU raw n={crowd['n_mu_raw']} mean={crowd['mu_raw_mean']} "
        f"AVGO raw n={crowd['n_avgo_raw']} mean={crowd['avgo_raw_mean']}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
