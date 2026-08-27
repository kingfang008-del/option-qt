#!/usr/bin/env python3
"""Option-fill for PM fade events (prefer 1DTE+, short hold).

Example:
  python -m maga7.tools.run_pm_fade_option_fill \\
    --events-tag research_pm_fade_apr_jul \\
    --tag research_pm_fade_fill_apr_jul \\
    --ext-min 0.008
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
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract, resolve_otm_rungs
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day, _spot_at

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ext-min", type=float, default=0.008)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--prefer-dte", type=int, default=1)
    ap.add_argument("--allowed-dte", default="1,2")
    ap.add_argument("--hold-minutes", type=int, default=15)
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    results_dir = Path(paths["results_dir"])
    ev_path = results_dir / args.events_tag / "events.parquet"
    if ev_path.is_file():
        events = pd.read_parquet(ev_path)
    else:
        events = pd.read_csv(results_dir / args.events_tag / "events.csv")
    events = events[np.isclose(events["ext_min"].astype(float), float(args.ext_min))].copy()
    if events.empty:
        raise SystemExit(f"no events at ext_min={args.ext_min}")

    allowed_dte = [int(x) for x in args.allowed_dte.split(",") if x.strip()]
    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)
    fill = FillSpec(
        entry_frac=float(fill_cfg.get("entry_frac", 0.75)),
        exit_frac=float(fill_cfg.get("exit_frac", 0.75)),
    )
    toxic_cfg = (trade.get("trade_toxic") or {}) if args.toxic else {"enabled": False}
    dates = sorted(events["date"].astype(str).unique())
    start, end = dates[0], dates[-1]
    months = month_list(start, end)
    symbols = sorted(events["symbol"].astype(str).unique())
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
        if not raw.empty:
            stock_by[sym] = attach_mf_features(raw)

    quote_cache: dict[tuple[str, str], Any] = {}
    raw_trades: list[dict] = []
    n_opt = n_miss = 0
    H = int(args.hold_minutes)
    for _, row in events.iterrows():
        sym = str(row["symbol"])
        date = str(row["date"])
        direction = str(row["dir"])
        sig_ts = to_ny(row["ts"])
        sdf = stock_by.get(sym)
        spot = _spot_at(sdf, sig_ts)
        by_dte = multi_idx.get((sym, date))
        ticker, dte, src = resolve_open_lock_contract(
            by_dte,
            direction=direction,
            moneyness="ATM",
            spot=spot,
            prefer_dte=int(args.prefer_dte),
            allowed_dte=allowed_dte,
            clear_otm_thresh=float(trade.get("clear_otm_ban_0dte_pct", 0.01) or 0.01),
            ladder=True,
            otm_rungs=otm_rungs,
        )
        if not ticker or int(dte or -1) == 0:
            n_miss += 1
            continue
        qkey = (sym, date)
        if qkey not in quote_cache:
            quote_cache[qkey] = load_quotes(paths["quote_1s_root"], sym, date)
        path = path_for_ticker(quote_cache[qkey], ticker)
        if path is None or path.empty:
            n_miss += 1
            continue
        after = path[path["timestamp"] >= sig_ts]
        if after.empty:
            n_miss += 1
            continue
        entry_ts = to_ny(after.iloc[0]["timestamp"])
        force_exit = entry_ts + pd.Timedelta(minutes=H)
        # hard flatten by 15:45
        flat_by = pd.Timestamp(f"{date} 15:45:00", tz="America/New_York")
        if force_exit > flat_by:
            force_exit = flat_by
        stock_day = sdf[sdf["date"].astype(str) == date] if sdf is not None else None
        stock_1s = load_stock_1s_day(paths["stock_1s_root"], sym, date)
        sim = simulate_trade(
            path,
            entry_ts,
            fill=fill,
            tp_mult=float(trade.get("tp_mult", 1.6)),
            sl_mult=float(trade.get("sl_mult", 0.45)),
            hold_minutes=max(1, H),
            direction=direction,
            stock_day=stock_day,
            exit_mode=None,
            force_exit_ts=force_exit,
            stock_bar_delay_seconds=0,
            trade_toxic=toxic_cfg,
            stock_1s=stock_1s if stock_1s is not None and not stock_1s.empty else None,
        )
        if sim is None:
            n_miss += 1
            continue
        n_opt += 1
        raw_trades.append(
            {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "sig_ts": str(sig_ts),
                "entry_ts": sim.entry_ts,
                "exit_ts": sim.exit_ts,
                "ticker": ticker,
                "dte": dte,
                "lock_source": src,
                "entry": float(sim.entry),
                "exit": float(sim.exit),
                "ret": float(sim.ret),
                "reason": str(sim.reason),
                "ext_from_anchor": float(row["ext_from_anchor"]),
                "clock": row.get("clock"),
                "sleeve": "PM",
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
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=int(args.cooldown_minutes),
            )
        )
    trades_df = pd.DataFrame(sized)
    stats = _equity_stats(trades_df)
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if len(trades_df):
        trades_df.to_csv(out / "trades.csv", index=False)
    summary = {
        "events_tag": args.events_tag,
        "ext_min": float(args.ext_min),
        "prefer_dte": int(args.prefer_dte),
        "allowed_dte": allowed_dte,
        "hold_minutes": H,
        "position_frac": float(args.position_frac),
        "n_signals": int(len(events)),
        "n_opt_fills": int(n_opt),
        "n_miss": int(n_miss),
        **stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
