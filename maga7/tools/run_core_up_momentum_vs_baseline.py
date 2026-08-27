#!/usr/bin/env python3
"""Causal CORE UP momentum (foresight champ) vs research_baseline on a window.

Champion rule (from ``research_session_horizon_foresight_jul10_23`` distill):
  session CORE 10:30–11:30, dir=UP only, |1s ret_60| >= thr (default 0.002),
  clock hold H (default 300s), open_ladder ATM on option trade last ± slip.

Pricing matches foresight scan (trades). Optional trade-toxic early cut.
Baseline arm = offline ``run_offline_replay`` on the same dates (quote 1s).

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_up_momentum_vs_baseline \\
    --start 2026-07-10 --end 2026-07-23 \\
    --tag research_core_up_vs_baseline_jul10_23
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import run_offline_replay, to_ny
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    regime_gate_from_1s,
    session_dates,
)
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day
from maga7.tools.scan_session_horizon_foresight import (
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SESS_START = "10:30"
SESS_END = "11:30"


def _toxic_clock(
    ts_ns: np.ndarray,
    last: np.ndarray,
    entry_ts: pd.Timestamp,
    *,
    hold_sec: int,
    slip: float,
    cut_ret: float,
    mfe_bypass: float,
) -> dict[str, Any] | None:
    """Clock hold with optional adverse toxic cut on trade-last path."""
    t0 = int(to_ny(entry_ts).value)
    i0 = int(np.searchsorted(ts_ns, t0, side="left"))
    if i0 >= len(ts_ns):
        return None
    lag = (int(ts_ns[i0]) - t0) / 1e9
    if lag > 5:
        return None
    entry = float(last[i0]) * (1.0 + float(slip))
    if not np.isfinite(entry) or entry <= 0:
        return None
    sell_mult = 1.0 - float(slip)
    end_ns = int(ts_ns[i0]) + int(hold_sec) * 1_000_000_000
    i_end = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
    if i_end < i0:
        return None
    mfe = -1.0
    exit_i = i_end
    reason = f"clock_H{hold_sec}"
    for k in range(i0 + 1, i_end + 1):
        px = float(last[k])
        if not np.isfinite(px) or px <= 0:
            continue
        ret = px * sell_mult / entry - 1.0
        if ret > mfe:
            mfe = ret
        if mfe < float(mfe_bypass) and ret <= -abs(float(cut_ret)):
            exit_i = k
            reason = "trade_toxic"
            break
    px_x = float(last[exit_i])
    ret = px_x * sell_mult / entry - 1.0
    hold = (int(ts_ns[exit_i]) - int(ts_ns[i0])) / 1e9
    return {
        "ret": float(ret),
        "reason": reason,
        "hold_sec": float(hold),
        "mfe": float(mfe if mfe > -1 else ret),
        "entry_lag_sec": float(lag),
    }


def _scan_champion(
    *,
    symbols: list[str],
    dates: list[str],
    stock_1s_root: Path,
    trades_root: Path,
    multi_idx: dict,
    otm_rungs: int,
    thr: float,
    lookback_sec: int,
    stride_sec: int,
    hold_sec: int,
    slip: float,
    toxic: bool,
    toxic_cut: float,
    toxic_mfe_bypass: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stride = pd.Timedelta(seconds=int(stride_sec))
    lb = max(int(lookback_sec), int(stride_sec))
    for date in dates:
        for sym in symbols:
            day = load_stock_1s_day(stock_1s_root, sym, date)
            if day is None or day.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            trade_paths = _paths_by_ticker(tday)
            if not trade_paths:
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = multi_idx.get((sym, date))
            if not by_dte:
                continue
            t_start = pd.Timestamp(f"{date} {SESS_START}:00", tz=NY)
            t_end = pd.Timestamp(f"{date} {SESS_END}:00", tz=NY)
            t = t_start + pd.Timedelta(seconds=lb)
            while t < t_end:
                direction, sr = _stock_dir_arr(
                    ts_ns, px, t, lookback_sec, float(thr)
                )
                if direction != "UP":
                    t += stride
                    continue
                spot = _spot_at_arr(ts_ns, px, t)
                ticker, dte, _src = resolve_open_lock_contract(
                    by_dte,
                    direction="UP",
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=[0, 1, 2],
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm_rungs,
                )
                if not ticker:
                    t += stride
                    continue
                key = str(ticker).replace("O:", "")
                arr = trade_paths.get(key)
                if arr is None:
                    t += stride
                    continue
                pts, plast = arr
                if toxic:
                    sim = _toxic_clock(
                        pts,
                        plast,
                        t,
                        hold_sec=hold_sec,
                        slip=slip,
                        cut_ret=toxic_cut,
                        mfe_bypass=toxic_mfe_bypass,
                    )
                else:
                    frs = _fwd_trade_rets_arr(
                        pts, plast, t, [hold_sec], slip=slip
                    )
                    if not frs:
                        t += stride
                        continue
                    fr = frs[0]
                    sim = {
                        "ret": float(fr["clock_ret"]),
                        "reason": f"clock_H{hold_sec}",
                        "hold_sec": float(hold_sec),
                        "mfe": float(fr["mfe"]),
                        "entry_lag_sec": float(fr["entry_lag_sec"]),
                    }
                if sim is None or not np.isfinite(sim["ret"]):
                    t += stride
                    continue
                et = to_ny(t)
                xt = et + pd.Timedelta(seconds=float(sim["hold_sec"]))
                rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": "UP",
                        "sig_ts": str(et),
                        "entry_ts": str(et),
                        "exit_ts": str(xt),
                        "ticker": ticker,
                        "dte": dte,
                        "ret": float(sim["ret"]),
                        "reason": sim["reason"],
                        "hold_sec": float(sim["hold_sec"]),
                        "mfe": float(sim.get("mfe") or np.nan),
                        "stock_ret_lb": float(sr),
                        "session": "CORE_1030_1130",
                        "sleeve": "CORE",
                    }
                )
                t += stride
    return rows


def _honest(trades_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {
            "sum_pnl_frac_additive": 0.0,
            "day_compound_ret": 0.0,
            "day_compound_maxdd": 0.0,
            "day_win": None,
            "trades_per_day": 0.0,
            "trade_mean": None,
            "n_toxic": 0,
        }
    pnl = trades_df["pnl_frac"].astype(float)
    day = trades_df.groupby(trades_df["date"].astype(str))["pnl_frac"].sum()
    day_eq = 1.0
    peak = 1.0
    day_mdd = 0.0
    for x in day.sort_index():
        day_eq *= 1.0 + float(x)
        peak = max(peak, day_eq)
        day_mdd = min(day_mdd, day_eq / peak - 1.0)
    reasons = trades_df.get("reason")
    n_tox = int((reasons == "trade_toxic").sum()) if reasons is not None else 0
    return {
        "sum_pnl_frac_additive": float(pnl.sum()),
        "day_compound_ret": float(day_eq - 1.0),
        "day_compound_maxdd": float(day_mdd),
        "day_win": float((day > 0).mean()),
        "trades_per_day": float(len(trades_df) / max(int(trades_df["date"].nunique()), 1)),
        "trade_mean": float(trades_df["ret"].astype(float).mean()),
        "n_toxic": n_tox,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start", default="2026-07-10")
    ap.add_argument("--end", default="2026-07-23")
    ap.add_argument("--tag", default="research_core_up_vs_baseline_jul10_23")
    ap.add_argument("--thr", type=float, default=0.002)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--stride-sec", type=int, default=60)
    ap.add_argument("--hold-sec", type=int, default=300)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--max-per-symbol-day", type=int, default=1)
    ap.add_argument("--toxic", action="store_true", default=True)
    ap.add_argument("--no-toxic", action="store_true")
    ap.add_argument("--toxic-cut", type=float, default=0.25)
    ap.add_argument("--toxic-mfe-bypass", type=float, default=0.05)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args(argv)

    toxic = bool(args.toxic) and not bool(args.no_toxic)
    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    dates = session_dates(args.start, args.end)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    stock_1s_root = Path(
        paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks"
    ).expanduser()
    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    print(
        f"champion scan CORE UP thr={args.thr} H={args.hold_sec} toxic={toxic} "
        f"{args.start}..{args.end} days={len(dates)}",
        flush=True,
    )
    raw = _scan_champion(
        symbols=symbols,
        dates=dates,
        stock_1s_root=stock_1s_root,
        trades_root=Path(args.trades_root),
        multi_idx=multi_idx,
        otm_rungs=otm_rungs,
        thr=float(args.thr),
        lookback_sec=int(args.lookback_sec),
        stride_sec=int(args.stride_sec),
        hold_sec=int(args.hold_sec),
        slip=float(args.slip),
        toxic=toxic,
        toxic_cut=float(args.toxic_cut),
        toxic_mfe_bypass=float(args.toxic_mfe_bypass),
    )
    print(f"raw signals {len(raw)}", flush=True)

    # causal first + per-symbol day cap, then portfolio concurrent
    by_date: dict[str, list[dict]] = {}
    for tr in raw:
        by_date.setdefault(str(tr["date"]), []).append(tr)
    picked: list[dict] = []
    max_sym = int(args.max_per_symbol_day)
    for date in sorted(by_date):
        rows = sorted(by_date[date], key=lambda r: (str(r["entry_ts"]), str(r["symbol"])))
        for tr in rows:
            if max_sym > 0:
                sym_n = sum(
                    1 for x in picked if x["date"] == date and x["symbol"] == tr["symbol"]
                )
                if sym_n >= max_sym:
                    continue
            picked.append(tr)

    sized: list[dict] = []
    by_day: dict[str, list[dict]] = {}
    for tr in picked:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    for _, rows in sorted(by_day.items()):
        sized.extend(
            _portfolio_day(
                rows,
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
        )
    champ_df = pd.DataFrame(sized)
    if len(champ_df) and "pnl_frac" not in champ_df.columns:
        champ_df["pnl_frac"] = champ_df["ret"].astype(float) * champ_df["size"].astype(float)
    champ_stats = {**_equity_stats(champ_df), **_honest(champ_df)}
    if len(champ_df):
        champ_df.to_parquet(out / "champion_trades.parquet", index=False)
        champ_df.to_csv(out / "champion_trades.csv", index=False)

    baseline_summary: dict[str, Any] | None = None
    if not args.skip_baseline:
        print(f"baseline replay {args.start}..{args.end}", flush=True)
        cfg = load_profile(args.profile)
        cfg.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
        cfg["date_range"]["start"] = args.start
        cfg["date_range"]["end"] = args.end
        stock_by = build_stock_by_from_1s(cfg, dates=dates, include_refs=True)
        regime_gate = regime_gate_from_1s(cfg, stock_by)
        result = run_offline_replay(
            cfg, scheme="single", stock_by=stock_by, regime_gate=regime_gate
        )
        baseline_summary = result["summary"]
        btrades = result.get("trades")
        if btrades is not None and len(btrades):
            btrades.to_parquet(out / "baseline_trades.parquet", index=False)
            btrades.to_csv(out / "baseline_trades.csv", index=False)
        (out / "baseline_summary.json").write_text(
            json.dumps(baseline_summary, indent=2, default=str), encoding="utf-8"
        )

    compare = {
        "window": {"start": args.start, "end": args.end, "n_sessions": len(dates)},
        "champion": {
            "rule": (
                f"CORE {SESS_START}-{SESS_END} UP-only "
                f"|1s_ret_{args.lookback_sec}|>={args.thr} clock_H{args.hold_sec}"
            ),
            "pricing": "option_trades_last_slip",
            "toxic": toxic,
            "toxic_cut": float(args.toxic_cut) if toxic else None,
            "position_frac": float(args.position_frac),
            "max_concurrent": int(args.max_concurrent),
            "max_per_symbol_day": int(args.max_per_symbol_day),
            "n_raw_signals": int(len(raw)),
            "n_picked": int(len(picked)),
            **champ_stats,
        },
        "baseline": {
            "profile_id": prof.get("profile_id"),
            "n_trades": None if baseline_summary is None else baseline_summary.get("n_trades"),
            "total_ret": None if baseline_summary is None else baseline_summary.get("total_ret"),
            "maxdd": None if baseline_summary is None else baseline_summary.get("maxdd"),
            "trade_win": None if baseline_summary is None else baseline_summary.get("trade_win"),
            "n_regime_block": None
            if baseline_summary is None
            else baseline_summary.get("n_regime_block"),
            "n_stock_path_confirm_block": None
            if baseline_summary is None
            else baseline_summary.get("n_stock_path_confirm_block"),
            "end_equity": None if baseline_summary is None else baseline_summary.get("end_equity"),
        },
        "note": (
            "Champion uses trades±slip clock (foresight-aligned). "
            "Baseline uses quote-1s offline replay. Compare additive/day_compound "
            "on champion vs total_ret on baseline; not identical fill engines."
        ),
    }
    (out / "compare.json").write_text(json.dumps(compare, indent=2, default=str), encoding="utf-8")
    (out / "champion_summary.json").write_text(
        json.dumps(compare["champion"], indent=2, default=str), encoding="utf-8"
    )

    print("\n=== COMPARE ===", flush=True)
    print(json.dumps(compare, indent=2, default=str), flush=True)
    if len(champ_df):
        cols = [
            c
            for c in [
                "date",
                "symbol",
                "entry_ts",
                "ret",
                "reason",
                "size",
                "pnl_frac",
                "stock_ret_lb",
            ]
            if c in champ_df.columns
        ]
        print("\nchampion trades:", flush=True)
        print(champ_df[cols].to_string(index=False), flush=True)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
