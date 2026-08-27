#!/usr/bin/env python3
"""Validate morning sec-MF on *existing dense ATM* 1s quotes only.

Prior morning option fills mixed in ATM paths that start at 10:00 (download
window). Those inflate hold times (~25m) and fake trend-exit labels. This
runner keeps only ATM contracts with:

  - >= ``min_morn_quotes`` prints in 09:30–10:00
  - first quote within ``max_entry_lag_sec`` of the signal
  - enough quotes covering the hold window

Research only. Independent of the 10:30 freeze book.
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
from maga7.common.sec_mf import attach_sec_mf_features
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.tools.run_morning_sec_option_fill import (
    DEFAULT_CANDIDATES,
    _equity_stats,
    _filter_events,
    _portfolio_day,
    _spot_at,
)
from maga7.tools.run_morning_sec_trend_exit import _resolve_trend_exit
from maga7.tools.scan_morning_sec_edge import _bdates, _prior_close

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _path_ts(path: pd.DataFrame) -> pd.DatetimeIndex:
    ts = pd.to_datetime(path["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    return pd.DatetimeIndex(ts)


def atm_morning_ok(
    path: pd.DataFrame | None,
    entry_ts,
    *,
    hold_sec: int,
    min_morn_quotes: int = 100,
    max_entry_lag_sec: float = 5.0,
    min_hold_quotes: int = 10,
) -> tuple[bool, dict[str, Any]]:
    """Return (ok, diagnostics) for dense morning ATM coverage."""
    diag: dict[str, Any] = {
        "morn_n": 0,
        "entry_lag_sec": None,
        "hold_quotes": 0,
        "first_quote": None,
        "why": "empty",
    }
    if path is None or path.empty:
        return False, diag
    ts = _path_ts(path)
    t = ts.time
    morn = int(((t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("10:00").time())).sum())
    diag["morn_n"] = morn
    diag["first_quote"] = str(ts[0])
    entry = to_ny(entry_ts)
    i = int(ts.searchsorted(entry, side="left"))
    if i >= len(ts):
        diag["why"] = "no_after"
        return False, diag
    lag = float((ts[i] - entry).total_seconds())
    diag["entry_lag_sec"] = lag
    hold_n = int(((ts >= entry) & (ts <= entry + pd.Timedelta(seconds=int(hold_sec)))).sum())
    diag["hold_quotes"] = hold_n
    if morn < int(min_morn_quotes):
        diag["why"] = "thin_morn"
        return False, diag
    if lag > float(max_entry_lag_sec):
        diag["why"] = "entry_lag"
        return False, diag
    if hold_n < int(min_hold_quotes):
        diag["why"] = "thin_hold"
        return False, diag
    diag["why"] = "OK"
    return True, diag


def _ungated_events(events: pd.DataFrame, *, mf_window_sec: int, streak_min: int, horizon_sec: int) -> pd.DataFrame:
    sub = events[
        (events["mf_window_sec"] == int(mf_window_sec))
        & (events["streak_min"] == int(streak_min))
        & (events["horizon_sec"] == int(horizon_sec))
    ].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("ts").drop_duplicates(
        ["date", "symbol", "dir", "mf_window_sec", "streak_min"], keep="first"
    )
    return sub.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--events-tag", default="research_morn_sec_edge_feb_jul")
    ap.add_argument("--tag", default="research_morn_sec_atm_dense_feb_jul")
    ap.add_argument(
        "--candidates",
        default="ungated_w100_s20_h180,w100_s20_h180_fp005_vz1_p0,w100_s20_h180_fp01_vz1_p0",
        help="comma names; ungated_* = rising edges without from_prev/vol/peer gates",
    )
    ap.add_argument("--min-morn-quotes", type=int, default=100)
    ap.add_argument("--max-entry-lag-sec", type=float, default=5.0)
    ap.add_argument("--exits", default="h180,h300,mf_flip_min30_max300,mf_flip_min60_max600")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    results_dir = Path(paths["results_dir"])
    events_path = results_dir / args.events_tag / "events.parquet"
    events = pd.read_parquet(events_path) if events_path.is_file() else pd.read_csv(events_path.with_suffix(".csv"))

    cand_by = {c["name"]: c for c in DEFAULT_CANDIDATES}
    want = [x.strip() for x in args.candidates.split(",") if x.strip()]
    specs: list[dict[str, Any]] = []
    for name in want:
        if name.startswith("ungated_"):
            # e.g. ungated_w100_s20_h180
            parts = name.replace("ungated_", "").split("_")
            # w100 s20 h180
            try:
                w = int(parts[0][1:])
                s = int(parts[1][1:])
                h = int(parts[2][1:])
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"bad ungated name {name}: {exc}") from exc
            specs.append(
                {
                    "name": name,
                    "mf_window_sec": w,
                    "streak_min": s,
                    "horizon_sec": h,
                    "from_prev_min": 0.0,
                    "vol_z_min": -1e9,
                    "peer_min": 0,
                    "ungated": True,
                }
            )
        elif name in cand_by:
            specs.append({**cand_by[name], "ungated": False})
        else:
            raise SystemExit(f"unknown candidate {name}")

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    stock_root = Path(paths["stock_root"]).expanduser()
    stock_1s_root = Path(paths["stock_1s_root"]).expanduser()
    otm_rungs = resolve_otm_rungs(prof, default=3)
    prefer_dte = int((prof.get("lock") or {}).get("prefer_dte", 0))
    allowed_dte = list((prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2])
    clear_otm = float(trade.get("clear_otm_ban_0dte_pct", 0.01) or 0.01)
    fill = FillSpec(
        entry_frac=float(fill_cfg.get("entry_frac", 0.75)),
        exit_frac=float(fill_cfg.get("exit_frac", 0.75)),
    )
    tp = float(trade.get("tp_mult", 1.6))
    sl = float(trade.get("sl_mult", 0.45))
    toxic = (trade.get("trade_toxic") or {}) if args.toxic else {"enabled": False}

    # preload 1m stock once for all specs
    start = str(events["date"].min())
    end = str(events["date"].max())
    months = month_list(start, end)
    symbols = sorted(events["symbol"].astype(str).unique())
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(stock_root, sym, months)
        if not raw.empty:
            stock_by[sym] = attach_mf_features(raw)
    dates_all = _bdates(start, end)

    out_root = results_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    feat_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    score_rows: list[dict[str, Any]] = []

    exit_specs: list[dict[str, Any]] = []
    for tok in [x.strip() for x in args.exits.split(",") if x.strip()]:
        if tok.startswith("h") and tok[1:].isdigit():
            exit_specs.append({"name": tok, "kind": "horizon", "horizon_sec": int(tok[1:])})
        elif tok.startswith("mf_flip_min"):
            # mf_flip_min30_max300
            body = tok.replace("mf_flip_", "")
            parts = body.split("_")
            min_h = int(parts[0].replace("min", ""))
            max_h = int(parts[1].replace("max", ""))
            exit_specs.append(
                {"name": tok, "kind": "mf_flip", "min_hold_sec": min_h, "max_hold_sec": max_h}
            )
        else:
            raise SystemExit(f"unknown exit {tok}")

    def get_quotes(sym: str, date: str) -> pd.DataFrame | None:
        key = (sym, date)
        if key not in quote_cache:
            quote_cache[key] = load_quotes(quote_root, sym, date)
        return quote_cache[key]

    def get_feat(sym: str, date: str, W: int) -> pd.DataFrame:
        key = (sym, date, int(W))
        if key in feat_cache:
            return feat_cache[key]
        day = load_stock_1s_day(stock_1s_root, sym, date)
        if day.empty:
            feat_cache[key] = pd.DataFrame()
            return feat_cache[key]
        ts = pd.to_datetime(day["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        day = day.copy()
        day["timestamp"] = ts
        t = day["timestamp"].dt.time
        buf = day[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("11:30").time())]
        prev = _prior_close(stock_1s_root, sym, date, dates_all)
        feat_cache[key] = attach_sec_mf_features(
            buf, mf_window_sec=int(W), vol_ma_sec=max(300, int(W) * 2), prev_close=prev
        )
        return feat_cache[key]

    for spec in specs:
        if spec.get("ungated"):
            sigs = _ungated_events(
                events,
                mf_window_sec=int(spec["mf_window_sec"]),
                streak_min=int(spec["streak_min"]),
                horizon_sec=int(spec["horizon_sec"]),
            )
        else:
            sigs = _filter_events(events, spec)
        print(f"\n=== candidate {spec['name']} signals={len(sigs)} ===", flush=True)
        W = int(spec["mf_window_sec"])

        for ex in exit_specs:
            hold_probe = int(ex.get("horizon_sec") or ex.get("max_hold_sec") or 300)
            variant = f"{spec['name']}__{ex['name']}"
            print(f"--- {variant} ---", flush=True)
            raw_trades: list[dict] = []
            skip: dict[str, int] = {}
            hold_secs: list[float] = []
            entry_lags: list[float] = []
            exit_reasons: dict[str, int] = {}
            stock_fwds: list[float] = []

            for _, row in sigs.iterrows():
                sym = str(row["symbol"])
                date = str(row["date"])
                direction = str(row["dir"])
                entry_ts = to_ny(row["ts"])
                sdf = stock_by.get(sym)
                spot = _spot_at(sdf, entry_ts)
                ticker, dte, src = resolve_open_lock_contract(
                    multi_idx.get((sym, date)),
                    direction=direction,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=prefer_dte,
                    allowed_dte=allowed_dte,
                    clear_otm_thresh=clear_otm,
                    ladder=True,
                    otm_rungs=otm_rungs,
                )
                if not ticker:
                    skip["no_lock"] = skip.get("no_lock", 0) + 1
                    continue
                path = path_for_ticker(get_quotes(sym, date), ticker)
                ok, diag = atm_morning_ok(
                    path,
                    entry_ts,
                    hold_sec=hold_probe,
                    min_morn_quotes=int(args.min_morn_quotes),
                    max_entry_lag_sec=float(args.max_entry_lag_sec),
                )
                if not ok:
                    why = str(diag.get("why") or "reject")
                    skip[why] = skip.get(why, 0) + 1
                    continue

                if ex["kind"] == "horizon":
                    exit_ts = entry_ts + pd.Timedelta(seconds=int(ex["horizon_sec"]))
                    exit_why = f"H{int(ex['horizon_sec'])}"
                    hold_minutes = max(1, int(np.ceil(int(ex["horizon_sec"]) / 60.0)))
                else:
                    feat = get_feat(sym, date, W)
                    exit_ts, exit_why = _resolve_trend_exit(
                        feat,
                        entry_ts=entry_ts,
                        direction=direction,
                        mode="mf_flip",
                        min_hold_sec=int(ex["min_hold_sec"]),
                        max_hold_sec=int(ex["max_hold_sec"]),
                    )
                    if exit_ts is None:
                        skip["no_exit"] = skip.get("no_exit", 0) + 1
                        continue
                    hold_minutes = max(1, int(np.ceil(int(ex["max_hold_sec"]) / 60.0)))

                stock_day = sdf[sdf["date"].astype(str) == date] if sdf is not None else None
                stock_1s = load_stock_1s_day(stock_1s_root, sym, date)
                sim = simulate_trade(
                    path,
                    entry_ts,
                    fill=fill,
                    tp_mult=tp,
                    sl_mult=sl,
                    hold_minutes=hold_minutes,
                    direction=direction,
                    stock_day=stock_day,
                    exit_mode=None,
                    force_exit_ts=exit_ts,
                    trade_toxic=toxic,
                    stock_bar_delay_seconds=0,
                    stock_1s=stock_1s if not stock_1s.empty else None,
                )
                if sim is None:
                    skip["sim_none"] = skip.get("sim_none", 0) + 1
                    continue

                # reject if fill clock drifted (should be rare after atm_morning_ok)
                fill_lag = (to_ny(sim.entry_ts) - entry_ts).total_seconds()
                if fill_lag > float(args.max_entry_lag_sec) + 1.0:
                    skip["fill_lag"] = skip.get("fill_lag", 0) + 1
                    continue

                reason = str(sim.reason)
                if reason == "DISPLACE":
                    reason = exit_why
                held = (to_ny(sim.exit_ts) - to_ny(sim.entry_ts)).total_seconds()
                hold_secs.append(held)
                entry_lags.append(float(diag["entry_lag_sec"] or 0.0))
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                if "fwd_ret_signed" in row and pd.notna(row["fwd_ret_signed"]):
                    stock_fwds.append(float(row["fwd_ret_signed"]))

                raw_trades.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": direction,
                        "entry_ts": entry_ts,
                        "sim_entry_ts": sim.entry_ts,
                        "exit_ts": sim.exit_ts,
                        "force_exit_ts": exit_ts,
                        "ticker": ticker,
                        "dte": dte,
                        "lock_source": src,
                        "ret": float(sim.ret),
                        "reason": reason,
                        "held_sec": held,
                        "entry_lag_sec": diag["entry_lag_sec"],
                        "morn_n": diag["morn_n"],
                        "entry": float(sim.entry),
                        "exit": float(sim.exit),
                        "stock_fwd": float(row["fwd_ret_signed"])
                        if "fwd_ret_signed" in row and pd.notna(row["fwd_ret_signed"])
                        else np.nan,
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
            trdf = pd.DataFrame(sized)
            stats = _equity_stats(trdf)
            row_out: dict[str, Any] = {
                "variant": variant,
                "candidate": spec["name"],
                "exit": ex["name"],
                "n_signals": int(len(sigs)),
                "n_atm_dense": int(len(raw_trades)),
                "n_skip": int(sum(skip.values())),
                "skip": skip,
                "held_sec_p50": float(np.median(hold_secs)) if hold_secs else None,
                "held_sec_mean": float(np.mean(hold_secs)) if hold_secs else None,
                "entry_lag_p50": float(np.median(entry_lags)) if entry_lags else None,
                "stock_fwd_mean": float(np.mean(stock_fwds)) if stock_fwds else None,
                "stock_fwd_win": float(np.mean([1.0 if x > 0 else 0.0 for x in stock_fwds])) if stock_fwds else None,
                "exit_reasons": exit_reasons,
                "min_morn_quotes": int(args.min_morn_quotes),
                "max_entry_lag_sec": float(args.max_entry_lag_sec),
                **stats,
            }
            sub = out_root / variant
            sub.mkdir(parents=True, exist_ok=True)
            if not trdf.empty:
                trdf.to_csv(sub / "trades.csv", index=False)
            pd.DataFrame(raw_trades).to_csv(sub / "trades_raw.csv", index=False)
            (sub / "summary.json").write_text(json.dumps(row_out, indent=2, default=str), encoding="utf-8")
            score_rows.append(row_out)
            print(
                json.dumps(
                    {
                        k: row_out[k]
                        for k in (
                            "variant",
                            "n_atm_dense",
                            "trade_win",
                            "exp",
                            "total_ret",
                            "maxdd",
                            "held_sec_p50",
                            "entry_lag_p50",
                            "stock_fwd_mean",
                            "skip",
                            "exit_reasons",
                        )
                    },
                    indent=2,
                    default=str,
                ),
                flush=True,
            )

    board = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("skip", "exit_reasons")}
            | {
                "skip": json.dumps(r.get("skip") or {}),
                "exit_reasons": json.dumps(r.get("exit_reasons") or {}),
            }
            for r in score_rows
        ]
    )
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2, default=str), encoding="utf-8")
    print("\n=== ATM-dense morning scoreboard ===")
    cols = [
        "variant",
        "n_atm_dense",
        "trade_win",
        "exp",
        "total_ret",
        "maxdd",
        "held_sec_p50",
        "entry_lag_p50",
        "stock_fwd_mean",
        "day_win",
    ]
    show = board[[c for c in cols if c in board.columns]].sort_values("total_ret", ascending=False)
    print(show.to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
