#!/usr/bin/env python3
"""Export signal-time ATM contract map for step2 1s quote download.

Walks the same TopK / m5_circuit / regime path as stable replay, resolves
signal_atm (with day_lock fallback), and writes a step1-compatible parquet:

  date_str, symbol, contract_symbol, bucket_id, front_dte, tag, source

Usage:
  python -m maga7.tools.export_signal_atm_lock_map \\
    --profile maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json \\
    --scheme m5_circuit \\
    --output ~/train_data/locked_targets_map_maga7_signal_atm_jan_jul.parquet
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
from maga7.common.contract_select import DayIvChainCache, lock_policy_from_profile, resolve_contract
from maga7.common.reentry import resolve_only_win_reenter
from maga7.common.regime import Mag7RegimeGate
from maga7.common.replay import BUCKET_MAP, load_lock_index, month_list, to_ny
from maga7.common.signals import (
    all_rule_a_times,
    attach_mf_features,
    build_topk_signals,
    load_stock_month_files,
)


def _spot_at(sdf: pd.DataFrame, date: str, ts) -> float | None:
    ts = to_ny(ts)
    day = sdf[sdf["date"] == date]
    if day.empty:
        return None
    bar = day[day["timestamp"] <= ts].tail(1)
    if bar.empty:
        return None
    px = float(bar.iloc[0]["close"])
    return px if np.isfinite(px) and px > 0 else None


def collect_picks(profile: dict, *, scheme: str) -> pd.DataFrame:
    paths = profile["_paths"]
    sig_cfg = profile["signal"]
    trade = profile["trade"]
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    symbols = list(profile["symbols"])
    money = str(trade.get("moneyness", "ATM"))
    prefer_dte, allowed_dte = lock_policy_from_profile(profile)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
        stock_by[sym] = attach_mf_features(
            raw,
            mf_window=int(sig_cfg.get("mf_window", 10)),
            vol_ma_window=int(sig_cfg.get("vol_ma_window", 20)),
        )

    top2 = build_topk_signals(stock_by, sig_cfg)
    lock_idx = load_lock_index(paths["locked_map"])
    chain_cache = DayIvChainCache(paths["day_iv_root"])
    regime_gate = Mag7RegimeGate.from_profile(profile, months=months)

    only_win = resolve_only_win_reenter(trade)
    use_reentry = scheme.startswith("m5")
    use_circuit = "circuit" in scheme
    cooldown = int(trade.get("cooldown_minutes", 5))
    max_n = int(trade.get("max_entries_per_symbol", 5))
    circuit = trade.get("day_circuit", None) if use_circuit else None
    # circuit needs equity; for map export we only need event enumeration —
    # ignore circuit halt so we still download contracts that would have traded
    # before a halt (conservative: slightly more contracts).
    _ = circuit

    rows: list[dict] = []
    for date, day_sigs in top2.groupby("date", sort=True):
        syms = list(day_sigs.sort_values("sig_ts")["symbol"].unique())

        if not use_reentry:
            events = [(to_ny(r.sig_ts), r.symbol, r.dir) for r in day_sigs.itertuples(index=False)]
        else:
            events = []
            for r in day_sigs.itertuples(index=False):
                for ts in all_rule_a_times(
                    stock_by[r.symbol][stock_by[r.symbol]["date"] == date],
                    r.dir,
                    window_start=str(sig_cfg.get("window_start", "10:30")),
                    window_end=str(sig_cfg.get("window_end", "14:00")),
                    streak_min=int(sig_cfg.get("streak_min", 8)),
                    from_prev_abs=float(sig_cfg.get("from_prev_abs", 0.02)),
                    vol_z_min=float(sig_cfg.get("vol_z_min", 1.0)),
                ):
                    events.append((to_ny(ts), r.symbol, r.dir))
            events.sort(key=lambda x: x[0])

        last_exit = {s: None for s in syms}
        last_win = {s: True for s in syms}
        n_done = {s: 0 for s in syms}
        open_until = {s: None for s in syms}

        for ts, sym, direction in events:
            if use_reentry:
                if n_done[sym] >= max_n:
                    continue
                if open_until[sym] is not None and ts < open_until[sym]:
                    continue
                if last_exit[sym] is not None and ts < last_exit[sym] + pd.Timedelta(minutes=cooldown):
                    continue
                if only_win and n_done[sym] > 0 and not last_win[sym]:
                    continue
            else:
                if n_done[sym] >= 1:
                    continue

            if regime_gate is not None:
                dec = regime_gate.check(direction, ts)
                if not dec.allow:
                    continue

            buckets = lock_idx.get((sym, date))
            day_ticker = buckets.get(BUCKET_MAP[(direction, money)]) if buckets else None
            spot = _spot_at(stock_by[sym], date, ts)
            pick = resolve_contract(
                mode="signal_atm",
                chain=chain_cache.get(sym, date),
                date=str(date),
                direction=direction,
                sig_ts=ts,
                spot=spot,
                day_lock_ticker=day_ticker,
                prefer_dte=prefer_dte,
                allowed_dte=allowed_dte,
                fallback_day_lock=True,
            )
            if pick is None:
                continue

            # Count event as "taken" for only_win / max_n sequencing even without fills,
            # so map matches live selection order; assume win to keep collecting later entries
            # that only_win would allow after a win. (Export is supersets of filled trades.)
            n_done[sym] += 1
            last_exit[sym] = ts + pd.Timedelta(minutes=int(trade.get("hold_minutes", 30)))
            open_until[sym] = last_exit[sym]
            last_win[sym] = True

            rows.append(
                {
                    "date_str": str(date),
                    "symbol": sym,
                    "contract_symbol": f"O:{pick.ticker}" if not str(pick.ticker).startswith("O:") else pick.ticker,
                    "front_dte": int(pick.dte) if pick.dte >= 0 else -1,
                    "source": pick.source,
                    "dir": direction,
                    "sig_ts": str(ts),
                    "sig_spot": spot,
                    "strike": pick.strike if np.isfinite(pick.strike) else None,
                    "day_lock_ticker": day_ticker,
                    "n_in_day": n_done[sym],
                }
            )
            # Also keep day-lock ATM so fallback / same-ticker days have 1s quotes.
            if day_ticker:
                rows.append(
                    {
                        "date_str": str(date),
                        "symbol": sym,
                        "contract_symbol": (
                            f"O:{day_ticker}" if not str(day_ticker).startswith("O:") else day_ticker
                        ),
                        "front_dte": -1,
                        "source": "day_lock_companion",
                        "dir": direction,
                        "sig_ts": str(ts),
                        "sig_spot": spot,
                        "strike": None,
                        "day_lock_ticker": day_ticker,
                        "n_in_day": n_done[sym],
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    # Unique contracts per (symbol, date); assign dense bucket_id for step2.
    uniq = (
        raw.sort_values(["date_str", "symbol", "contract_symbol", "source"])
        .drop_duplicates(["date_str", "symbol", "contract_symbol"], keep="first")
        .copy()
    )
    uniq["bucket_id"] = uniq.groupby(["date_str", "symbol"]).cumcount()
    uniq["tag"] = uniq["source"]
    uniq["dte_mode"] = "trading"
    return uniq.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Mag7 signal_atm lock map for step2")
    ap.add_argument(
        "--profile",
        default=str(ROOT / "maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_stable_v1.json"),
    )
    ap.add_argument("--scheme", default="m5_circuit", choices=["single", "m5", "m5_circuit"])
    ap.add_argument(
        "--output",
        default=str(Path.home() / "train_data/locked_targets_map_maga7_signal_atm_jan_jul.parquet"),
    )
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    args = ap.parse_args()

    profile = load_profile(args.profile)
    if args.start_date:
        profile["date_range"]["start"] = args.start_date
    if args.end_date:
        profile["date_range"]["end"] = args.end_date

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = collect_picks(profile, scheme=args.scheme)
    if df.empty:
        raise SystemExit("no contracts collected")
    df.to_parquet(out, index=False)

    meta = {
        "n_rows": int(len(df)),
        "n_day_symbol": int(df.groupby(["date_str", "symbol"]).ngroups),
        "n_dates": int(df["date_str"].nunique()),
        "source_counts": df["source"].value_counts().to_dict(),
        "output": str(out),
        "scheme": args.scheme,
        "start": profile["date_range"]["start"],
        "end": profile["date_range"]["end"],
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
