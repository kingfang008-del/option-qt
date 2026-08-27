#!/usr/bin/env python3
"""QQQ 0DTE open_cont (09:45) small-swing: clock vs path/Greeks exits.

Entry = open continuation at 09:45 (same family as research_qqq_0dte_morn_struct).
Exit variants:
  - clock_h180 / clock_h300  (pure horizon + TP/SL rails)
  - trail15                  (mtm trail, max 300s)
  - greeks_toxic / greeks_winner_safe  (path_greeks early cut on h300 clock)

Uses BS IV/Δ from quote mid + stock (no stress_test_1s_greeks dependency).
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.fills import FillSpec
from maga7.common.path_greeks_exit import PathGreeksState, cfg_from_preset
from maga7.common.replay import simulate_trade, to_ny
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import (
    BUCKET_ATM,
    _discover_option_dates,
    _load_atm_path,
)
from maga7.tools.scan_morning_sec_edge import _bdates, _morning_slice, _prior_close

NY = "America/New_York"
DEFAULT_OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _cp_from_ticker(ticker: str, direction: str) -> str:
    t = str(ticker or "").upper().replace(" ", "")
    if "C" in t[6:]:
        # OCC: ROOT + YYMMDD + C/P + strike
        for i, ch in enumerate(t):
            if ch in ("C", "P") and i >= 6:
                return "c" if ch == "C" else "p"
    return "c" if direction == "UP" else "p"


def _clock_sim(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    direction: str,
    hold_sec: int,
    fill: FillSpec,
    tp_mult: float,
    sl_mult: float,
) -> Any | None:
    return simulate_trade(
        path,
        entry_ts,
        fill=fill,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        hold_minutes=max(1, int(np.ceil(hold_sec / 60.0))),
        direction=direction,
        exit_mode=None,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(hold_sec)),
        trade_toxic={"enabled": False},
        stock_bar_delay_seconds=0,
    )


def _trail_sim(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    direction: str,
    max_hold_sec: int,
    fill: FillSpec,
    tp_mult: float,
    sl_mult: float,
) -> Any | None:
    return simulate_trade(
        path,
        entry_ts,
        fill=fill,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        hold_minutes=max(1, int(np.ceil(max_hold_sec / 60.0))),
        direction=direction,
        exit_mode="mtm_trail",
        trail_activate=0.15,
        trail_dd=0.08,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(max_hold_sec)),
        trade_toxic={"enabled": False},
        stock_bar_delay_seconds=0,
    )


def _greeks_early(
    path: pd.DataFrame,
    stock: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    direction: str,
    hold_sec: int,
    fill: FillSpec,
    tp_mult: float,
    sl_mult: float,
    ticker: str,
    strike: float,
    date: str,
    preset: str,
) -> tuple[Any | None, str | None]:
    """Run clock sim, then walk path with path_greeks; if L2 fires earlier, refill exit."""
    base = _clock_sim(
        path,
        entry_ts,
        direction=direction,
        hold_sec=hold_sec,
        fill=fill,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
    )
    if base is None:
        return None, None
    cfg, naive = cfg_from_preset(preset)
    if not cfg.enabled:
        return base, None

    cp = _cp_from_ticker(ticker, direction)
    expiry = pd.Timestamp(f"{date} 16:00", tz=NY)
    after = path[path["timestamp"] >= entry_ts].copy()
    if after.empty:
        return base, None
    # align stock (series already tz-aware from cache)
    st_day = stock.sort_values("timestamp")
    s_idx = pd.DatetimeIndex(pd.to_datetime(st_day["timestamp"]))
    if s_idx.tz is None:
        s_idx = s_idx.tz_localize(NY, ambiguous="infer")
    else:
        s_idx = s_idx.tz_convert(NY)
    s_px = st_day["close"].astype(float).to_numpy()

    entry_row = after.iloc[0]
    entry_mid = 0.5 * (float(entry_row["bid"]) + float(entry_row["ask"]))
    if not np.isfinite(entry_mid) or entry_mid <= 0:
        entry_mid = float(base.entry)

    st = PathGreeksState(
        entry_px=float(entry_mid),
        K=float(strike),
        cp=cp,
        expiry_ts=float(expiry.timestamp()),
        cfg=cfg,
        naive_half_peak=naive,
        entry_ts=float(pd.Timestamp(entry_ts).timestamp()),
    )
    base_exit = to_ny(base.exit_ts)
    # stride ~2s to keep BS IV loop tractable on 1s paths
    rows = after.iloc[::2]
    for _, row in rows.iterrows():
        ts = to_ny(row["timestamp"])
        if ts > base_exit:
            break
        bid, ask = float(row["bid"]), float(row["ask"])
        mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else (bid or ask)
        if not np.isfinite(mid) or mid <= 0:
            continue
        j = int(s_idx.searchsorted(ts, side="right")) - 1
        if j < 0:
            continue
        S = float(s_px[j])
        reason, _met = st.on_tick(ts=float(ts.timestamp()), mid=float(mid), S=S)
        if reason:
            sell = fill.sell(bid, ask)
            base_reason = str(base.reason)
            if base_reason in {"TP", "SL"} and to_ny(base.exit_ts) <= ts:
                return base, None
            from types import SimpleNamespace

            ret = float(sell / base.entry - 1.0) if base.entry > 0 else 0.0
            return (
                SimpleNamespace(
                    entry=base.entry,
                    exit=float(sell),
                    ret=ret,
                    reason=reason,
                    entry_ts=base.entry_ts,
                    exit_ts=ts,
                ),
                reason,
            )
    return base, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK))
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--clock", default="09:45")
    ap.add_argument("--from-open-mins", default="0,0.002,0.003")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--tp-mult", type=float, default=1.6)
    ap.add_argument("--sl-mult", type=float, default=0.45)
    ap.add_argument(
        "--results-dir",
        default="/mnt/s990/data/maga7/results",
    )
    ap.add_argument("--tag", default="qqq_open_cont_path_exit_feb_jun_v1")
    args = ap.parse_args()

    opt_root = Path(args.opt_root).expanduser()
    stock_1s = Path(args.stock_1s_root).expanduser()
    out = Path(args.results_dir).expanduser() / args.tag
    out.mkdir(parents=True, exist_ok=True)

    dates = [
        d
        for d in _discover_option_dates(opt_root, args.start_date, args.end_date)
        if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    if not dates:
        raise SystemExit("no overlapping days")
    all_bd = _bdates(dates[0], dates[-1])
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    fo_mins = _parse_floats(args.from_open_mins)
    clock = str(args.clock)

    exits = [
        ("clock_h180", "clock", 180),
        ("clock_h300", "clock", 300),
        ("trail15_h300", "trail", 300),
        ("greeks_toxic_h300", "greeks_toxic", 300),
        ("greeks_winner_h300", "greeks_winner_safe", 300),
    ]

    print(f"dates={len(dates)} {dates[0]}..{dates[-1]}", flush=True)
    day_cache: dict[str, dict] = {}
    stock_cache: dict[str, pd.DataFrame] = {}
    for date in dates:
        day = load_stock_1s_day(stock_1s, "QQQ", date)
        buf = _morning_slice(day, start="09:30", end="16:00")
        if buf.empty:
            continue
        stock_cache[date] = buf
        ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
        if ts.tz is None:
            ts = ts.tz_localize(NY, ambiguous="infer")
        else:
            ts = ts.tz_convert(NY)
        close = buf["close"].astype(float).to_numpy()
        open_px = float(close[0])
        prev = _prior_close(stock_1s, "QQQ", date, all_bd)
        day_cache[date] = {"ts": ts, "close": close, "open": open_px, "prev": prev}
    print(f"stock days={len(day_cache)}", flush=True)

    path_cache: dict[tuple[str, str], tuple] = {}

    def get_path(date: str, direction: str):
        key = (date, direction)
        if key not in path_cache:
            path_cache[key] = _load_atm_path(opt_root, date, direction)
        return path_cache[key]

    score_rows: list[dict] = []
    all_trades: list[dict] = []

    for fo_min in fo_mins:
        # collect entries once
        entries: list[dict] = []
        for date, d in day_cache.items():
            ts, close, open_px = d["ts"], d["close"], d["open"]
            t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
            i = int(ts.searchsorted(t0, side="left"))
            if i >= len(close) - 1:
                continue
            from_open = float((close[i] - open_px) / open_px) if open_px else 0.0
            if abs(from_open) < float(fo_min):
                continue
            direction = "UP" if from_open > 0 else "DN"
            path, ticker, strike = get_path(date, direction)
            if path is None or path.empty or strike is None:
                continue
            entry_ts = to_ny(ts[i])
            after = path[path["timestamp"] >= entry_ts]
            if after.empty:
                continue
            lag = (to_ny(after.iloc[0]["timestamp"]) - entry_ts).total_seconds()
            if lag > 5:
                continue
            entries.append(
                {
                    "date": date,
                    "direction": direction,
                    "from_open": from_open,
                    "entry_ts": entry_ts,
                    "path": path,
                    "ticker": ticker,
                    "strike": float(strike),
                    "lag": lag,
                }
            )
        print(f"fo>={fo_min}: entries={len(entries)}", flush=True)

        for exit_name, exit_kind, hold_sec in exits:
            variant = f"open_cont_{clock.replace(':', '')}_fo{str(fo_min).replace('.', 'p')}__{exit_name}"
            raw_trades: list[dict] = []
            n_l2 = 0
            for e in entries:
                if exit_kind == "clock":
                    sim = _clock_sim(
                        e["path"],
                        e["entry_ts"],
                        direction=e["direction"],
                        hold_sec=hold_sec,
                        fill=fill,
                        tp_mult=float(args.tp_mult),
                        sl_mult=float(args.sl_mult),
                    )
                    l2 = None
                elif exit_kind == "trail":
                    sim = _trail_sim(
                        e["path"],
                        e["entry_ts"],
                        direction=e["direction"],
                        max_hold_sec=hold_sec,
                        fill=fill,
                        tp_mult=float(args.tp_mult),
                        sl_mult=float(args.sl_mult),
                    )
                    l2 = None
                else:
                    preset = "toxic_only" if exit_kind == "greeks_toxic" else "winner_safe"
                    sim, l2 = _greeks_early(
                        e["path"],
                        stock_cache[e["date"]],
                        e["entry_ts"],
                        direction=e["direction"],
                        hold_sec=hold_sec,
                        fill=fill,
                        tp_mult=float(args.tp_mult),
                        sl_mult=float(args.sl_mult),
                        ticker=e["ticker"],
                        strike=e["strike"],
                        date=e["date"],
                        preset=preset,
                    )
                    if l2:
                        n_l2 += 1
                if sim is None:
                    continue
                reason = str(sim.reason)
                if reason == "DISPLACE":
                    reason = f"H{hold_sec}"
                held = (to_ny(sim.exit_ts) - to_ny(sim.entry_ts)).total_seconds()
                raw_trades.append(
                    {
                        "variant": variant,
                        "date": e["date"],
                        "month": e["date"][:7],
                        "symbol": "QQQ",
                        "dir": e["direction"],
                        "from_open": e["from_open"],
                        "exit_name": exit_name,
                        "horizon_sec": hold_sec,
                        "from_open_min": fo_min,
                        "entry_ts": e["entry_ts"],
                        "exit_ts": sim.exit_ts,
                        "ticker": e["ticker"],
                        "strike": e["strike"],
                        "ret": float(sim.ret),
                        "reason": reason,
                        "l2_reason": l2,
                        "held_sec": float(held),
                        "entry": float(sim.entry),
                        "exit": float(sim.exit),
                        "entry_lag_sec": e["lag"],
                    }
                )

            by_day: dict[str, list[dict]] = {}
            for tr in raw_trades:
                by_day.setdefault(str(tr["date"]), []).append(tr)
            sized: list[dict] = []
            for _, rows in sorted(by_day.items()):
                sized.extend(
                    _portfolio_day(
                        rows,
                        position_frac=float(args.position_frac),
                        max_concurrent=1,
                        cooldown_minutes=0,
                    )
                )
            trdf = pd.DataFrame(sized)
            stats = _equity_stats(trdf)
            reasons = (
                {str(k): int(v) for k, v in trdf["reason"].value_counts().items()}
                if not trdf.empty and "reason" in trdf.columns
                else {}
            )
            row = {
                "variant": variant,
                "clock": clock,
                "from_open_min": fo_min,
                "exit_name": exit_name,
                "horizon_sec": hold_sec,
                "n_entries": int(len(entries)),
                "n_fills": int(len(raw_trades)),
                "n_l2_cuts": int(n_l2),
                "held_sec_p50": float(trdf["held_sec"].median()) if not trdf.empty else None,
                "reasons": reasons,
                **stats,
            }
            score_rows.append(row)
            all_trades.extend(sized)
            print(
                f"  {variant}: ret={stats.get('total_ret', 0):+.3f} "
                f"exp={stats.get('exp', 0):+.3f} dd={stats.get('maxdd', 0):.3f} "
                f"n={stats.get('n_trades', 0)} l2={n_l2} "
                f"held_p50={row['held_sec_p50']}",
                flush=True,
            )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    # json-friendly
    slim = []
    for r in score_rows:
        slim.append({k: v for k, v in r.items()})
    (out / "scoreboard.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    if all_trades:
        pd.DataFrame(all_trades).to_csv(out / "trades.csv", index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
