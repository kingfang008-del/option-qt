#!/usr/bin/env python3
"""Offline Mag7 replay from second-level stock + second-level option quotes.

Stock fact source: paths.stock_1s_root (/mnt/s990/data/raw_1s/stocks)
Option fact source: paths.quote_1s_root
Signal path: Mag7Scanner on_stock_second (1s → completed 1m)
Fill path: Mag7OmsDryRunner + QuoteSimSession disk 1s quotes
mf_flip / regime stock frames: rebuilt from the same 1s stock root
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.provenance import code_fingerprint
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    coverage_report,
    regime_gate_from_1s,
    session_dates,
)
from maga7.live.oms_dry import Mag7OmsDryRunner
from maga7.live.scanner import Mag7Scanner, write_signal_audit

PEER3 = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json"
)
NY = "America/New_York"


def _drive_day(
    scanner: Mag7Scanner,
    stock_1s_root: Path,
    date: str,
    symbols: list[str],
) -> dict[str, int]:
    frames = []
    missing = 0
    for sym in symbols:
        raw = load_stock_1s_day(stock_1s_root, sym, date)
        if raw.empty:
            missing += 1
            continue
        raw = raw.copy()
        raw["symbol"] = sym
        frames.append(raw)
    if not frames:
        return {"ticks": 0, "missing_symbols": missing}
    all_ticks = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    for row in all_ticks.itertuples(index=False):
        scanner.on_stock_second(
            row.symbol,
            {
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            },
        )
    return {"ticks": int(len(all_ticks)), "missing_symbols": missing}


def _build_daily_report(trades: pd.DataFrame, equity0: float = 100.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_trades",
                "n_wins",
                "n_losses",
                "day_pnl",
                "day_ret",
                "equity_end",
                "gross_exposure",
                "avg_size_frac",
                "max_size_frac",
                "symbols",
                "reasons",
            ]
        )
    rows = []
    eq = equity0
    for date, day in trades.groupby("date", sort=True):
        day = day.copy()
        day_pnl = float(day["pnl_equity"].sum()) if "pnl_equity" in day.columns else 0.0
        # Reconstruct path-dependent equity if pnl_equity present; else compound by size*ret.
        if "pnl_equity" not in day.columns:
            day_pnl = 0.0
            local_eq = eq
            for r in day.itertuples(index=False):
                pnl = local_eq * float(r.qty_frac) * float(r.ret)
                local_eq += pnl
                day_pnl += pnl
            eq = local_eq
        else:
            eq = eq + day_pnl
        day_ret = day_pnl / (eq - day_pnl) if abs(eq - day_pnl) > 1e-12 else float("nan")
        rows.append(
            {
                "date": str(date),
                "n_trades": int(len(day)),
                "n_wins": int((day["ret"] > 0).sum()),
                "n_losses": int((day["ret"] <= 0).sum()),
                "day_pnl": day_pnl,
                "day_ret": day_ret,
                "equity_end": eq,
                "gross_exposure": float(day["qty_frac"].sum()),
                "avg_size_frac": float(day["qty_frac"].mean()),
                "max_size_frac": float(day["qty_frac"].max()),
                "symbols": ",".join(sorted(day["symbol"].astype(str).unique())),
                "reasons": ",".join(
                    f"{k}:{int(v)}" for k, v in day["reason"].value_counts().items()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 offline replay from stock 1s + option 1s")
    p.add_argument("--profile", default=str(PEER3))
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--scheme", default="single")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    profile = load_profile(args.profile)
    start = args.start_date or profile["date_range"]["start"]
    end = args.end_date or profile["date_range"]["end"]
    profile["date_range"]["start"] = start
    profile["date_range"]["end"] = end
    dates = session_dates(start, end)

    tag = args.tag or f"replay_stock1s_opt1s_{args.scheme}_{start}_{end}"
    out_dir = Path(profile["_paths"]["results_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"building 1s→1m stock frames for {start}..{end} ({len(dates)} sessions)", flush=True)
    stock_by = build_stock_by_from_1s(profile, dates=dates, include_refs=True)
    coverage = coverage_report(
        stock_by,
        dates=dates,
        symbols=list(profile["symbols"]) + ["QQQ", "VIXY"],
    )
    (out_dir / "stock_1s_coverage.json").write_text(
        json.dumps(coverage, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(coverage, indent=2), flush=True)

    runner = Mag7OmsDryRunner(profile)
    # Replace mf_flip stock cache with 1s-derived bars for Mag7 names.
    for sym in profile["symbols"]:
        if sym in stock_by:
            runner.session.stock_by[sym] = stock_by[sym]

    scanner = Mag7Scanner.from_profile(profile, scheme=args.scheme)
    scanner.regime_gate = regime_gate_from_1s(profile, stock_by)
    scanner.stock_by = stock_by  # peer_align parity with offline feature_ts asof
    runner.scanner = scanner
    scanner.on_signal = runner.process_one
    scanner.is_symbol_active = lambda sym: (
        runner.session.open_until.get(sym) is not None
    )

    stock_1s_root = Path(profile["_paths"]["stock_1s_root"])
    day_stats = []
    for i, date in enumerate(dates, 1):
        before = len(runner.trades)
        eq_before = runner.eq
        stats = _drive_day(scanner, stock_1s_root, date, list(profile["symbols"]))
        n_new = len(runner.trades) - before
        day_stats.append(
            {
                "date": date,
                "ticks": stats["ticks"],
                "missing_symbols": stats["missing_symbols"],
                "n_new_trades": n_new,
                "equity_end": runner.eq,
                "day_pnl": runner.eq - eq_before,
            }
        )
        if i % 5 == 0 or i == len(dates) or n_new:
            print(
                f"[{i}/{len(dates)}] {date} ticks={stats['ticks']} "
                f"new_trades={n_new} equity={runner.eq:.4f}",
                flush=True,
            )

    scanner.flush_seconds()
    summary = runner.finalize_summary()
    summary.update(
        {
            "mode": "REPLAY_STOCK1S_OPT1S",
            "ingest": "stock_1s",
            "quote_source": "option_1s",
            "scheme": args.scheme,
            "start": start,
            "end": end,
            "profile": profile.get("profile_id") or profile.get("profile"),
            "strategy_fingerprint": code_fingerprint(profile["_profile_path"]),
            "stock_1s_root": str(stock_1s_root),
            "quote_1s_root": str(profile["_paths"]["quote_1s_root"]),
            "bar_availability_delay_seconds": int(
                (profile.get("trade") or {}).get("bar_availability_delay_seconds", 0) or 0
            ),
            "compared_to_headline_3375": {
                "headline_total_ret": 33.750961292339156,
                "headline_maxdd": -0.22104930562551695,
                "headline_n_trades": 247,
            },
        }
    )
    runner.summary = summary
    runner.write(out_dir)
    write_signal_audit(scanner.signals, out_dir / "signals.jsonl")

    trades = pd.DataFrame([t.__dict__ for t in runner.trades])
    if not trades.empty:
        trades["date"] = trades["date"].astype(str)
        # ensure chronological trade list
        trades = trades.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)
        trades.to_csv(out_dir / "trades.csv", index=False)

    daily = _build_daily_report(trades, equity0=100.0)
    # Prefer runner equity path if available
    if runner.daily_rows:
        eq_map = {str(r["date"]): float(r["equity"]) for r in runner.daily_rows}
        if not daily.empty:
            daily["equity_end"] = daily["date"].map(eq_map).fillna(daily["equity_end"])
            daily["day_pnl"] = daily["equity_end"].diff()
            if len(daily):
                daily.loc[daily.index[0], "day_pnl"] = (
                    float(daily.loc[daily.index[0], "equity_end"]) - 100.0
                )
                daily["day_ret"] = daily["day_pnl"] / (
                    daily["equity_end"] - daily["day_pnl"]
                )
    daily.to_csv(out_dir / "daily_detail.csv", index=False)
    pd.DataFrame(day_stats).to_csv(out_dir / "day_ingest.csv", index=False)

    # Compact trade blotter for reading
    if not trades.empty:
        blotter = trades[
            [
                c
                for c in [
                    "date",
                    "symbol",
                    "direction",
                    "contract",
                    "rank",
                    "qty_frac",
                    "entry_ts",
                    "exit_ts",
                    "entry",
                    "exit",
                    "ret",
                    "pnl_equity",
                    "reason",
                ]
                if c in trades.columns
            ]
        ].copy()
        blotter.to_csv(out_dir / "trade_blotter.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"→ {out_dir}")
    if not daily.empty:
        print("\nDaily P&L (first/last 5):")
        show = pd.concat([daily.head(5), daily.tail(5)]).drop_duplicates()
        print(
            show[
                [
                    "date",
                    "n_trades",
                    "n_wins",
                    "n_losses",
                    "day_ret",
                    "equity_end",
                    "gross_exposure",
                    "avg_size_frac",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
