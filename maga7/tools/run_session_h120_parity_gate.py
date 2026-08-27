#!/usr/bin/env python3
"""Pre-promote parity gate: stream vs batch-offline for session H120 sleeves.

Supports reinforce gates and pricing=trades|quotes. Compares trade keys / ret / size.

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_parity_gate \\
    --session MID_1230_1330 --start-date 2026-04-01 --end-date 2026-07-22 \\
    --reinforce mf_p2_fo035 --pricing trades \\
    --tag parity_gate_mid_mf_p2_fo_trades_apr_jul
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

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.session_entry_reinforce import SessionReinforceConfig, evaluate_reinforce
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import (
    SESSIONS,
    _bdates,
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)
from maga7.tools.run_session_h120_stream_parity import compare

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")


def _rein_cfg(name: str) -> SessionReinforceConfig:
    n = str(name or "none").lower()
    if n in {"", "none", "off", "base"}:
        return SessionReinforceConfig()
    if n == "mf":
        return SessionReinforceConfig(require_mf=True)
    if n in {"mf_p2_fo035", "mf_p2_fo"}:
        return SessionReinforceConfig(require_mf=True, peer_min=2, from_open_max=0.035)
    raise SystemExit(f"unknown reinforce={name}")


def _collect_raw(
    *,
    symbols: list[str],
    stock_by: dict[str, pd.DataFrame],
    multi_idx: dict,
    otm_rungs: list[int],
    dates: list[str],
    session: str,
    session_bounds: tuple[str, str],
    trades_root: Path,
    quote_root: Path | None,
    pricing: str,
    fill_frac: float,
    h: int,
    lookback_sec: int,
    stride_sec: int,
    min_abs: float,
    prefer_dte: int,
    allowed_dte: list[int],
    slip: float,
    rein: SessionReinforceConfig,
    need_mf: bool,
) -> list[dict[str, Any]]:
    """All causal candidates (no portfolio yet), sorted by (date, entry_ts, symbol)."""
    s0, s1 = session_bounds
    lb = max(int(lookback_sec), int(stride_sec))
    stride = pd.Timedelta(seconds=int(stride_sec))
    fill = FillSpec(entry_frac=float(fill_frac), exit_frac=float(fill_frac))
    raw: list[dict[str, Any]] = []

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[cand] {date} ({di+1}/{len(dates)}) n={len(raw)}", flush=True)
        day_stock: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        day_trade_paths: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        day_quotes: dict[str, Any] = {}
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            day = sdf[sdf["date"].astype(str) == date]
            if day.empty:
                continue
            day_stock[sym] = _stock_arrays(day)
            if pricing == "trades":
                tday = load_option_trades(trades_root, sym, date)
                if tday is None or tday.empty:
                    continue
                paths = _paths_by_ticker(tday)
                if paths:
                    day_trade_paths[sym] = paths
            else:
                q = load_quotes(quote_root, sym, date) if quote_root else None
                if q is not None and not q.empty:
                    day_quotes[sym] = q

        t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY)
        t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
        t = t_start + pd.Timedelta(seconds=lb)
        while t < t_end:
            cands: list[dict[str, Any]] = []
            for sym in symbols:
                arr = day_stock.get(sym)
                if arr is None:
                    continue
                ts_ns, px = arr
                direction, sr = _stock_dir_arr(ts_ns, px, t, lookback_sec, min_abs)
                if direction is None:
                    continue
                if need_mf:
                    ok, _meta = evaluate_reinforce(
                        stock_by=stock_by,
                        symbol=sym,
                        date=date,
                        entry_ts=t,
                        direction=direction,
                        cfg=rein,
                        peer_symbols=symbols,
                    )
                    if not ok:
                        continue
                spot = _spot_at_arr(ts_ns, px, t)
                ticker, dte, _src = resolve_open_lock_contract(
                    multi_idx.get((sym, date)),
                    direction=direction,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=prefer_dte,
                    allowed_dte=allowed_dte,
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm_rungs,
                )
                if not ticker:
                    continue
                ret = None
                reason = f"clock_H{h}"
                xt = t + pd.Timedelta(seconds=h)
                if pricing == "trades":
                    key = str(ticker).replace("O:", "")
                    path = (day_trade_paths.get(sym) or {}).get(key)
                    if path is None:
                        continue
                    frs = _fwd_trade_rets_arr(path[0], path[1], t, [h], slip=float(slip))
                    if not frs:
                        continue
                    ret = float(frs[0]["clock_ret"])
                else:
                    qpath = path_for_ticker(day_quotes.get(sym), ticker)
                    if qpath is None or qpath.empty:
                        continue
                    sim = simulate_trade(
                        qpath,
                        t,
                        fill=fill,
                        tp_mult=100.0,
                        sl_mult=0.01,
                        hold_minutes=max(1, int(np.ceil(h / 60))),
                        direction=direction,
                        force_exit_ts=xt,
                        trade_toxic={"enabled": False},
                    )
                    if sim is None:
                        continue
                    ret = float(sim.ret)
                    reason = str(sim.reason)
                    xt = to_ny(sim.exit_ts)
                if ret is None or not np.isfinite(ret):
                    continue
                cands.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": direction,
                        "entry_ts": str(t),
                        "exit_ts": str(xt),
                        "sig_ts": str(t),
                        "ticker": ticker,
                        "dte": dte,
                        "ret": ret,
                        "reason": reason,
                        "session": session,
                        "stock_ret_lb": sr,
                    }
                )
            raw.extend(sorted(cands, key=lambda r: str(r["symbol"])))
            t += stride
    return raw


def _size_book(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> pd.DataFrame:
    by_day: dict[str, list[dict]] = {}
    for tr in raw:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    sized: list[dict] = []
    for d in sorted(by_day):
        sized.extend(
            _portfolio_day(
                by_day[d],
                position_frac=position_frac,
                max_concurrent=max_concurrent,
                cooldown_minutes=cooldown_minutes,
            )
        )
    df = pd.DataFrame(sized)
    if len(df) and "pnl_frac" not in df.columns and "size" in df.columns:
        df["pnl_frac"] = df["ret"].astype(float) * df["size"].astype(float)
    if len(df) and "size_frac" not in df.columns and "size" in df.columns:
        df["size_frac"] = df["size"]
    return df


def _stream_from_raw(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> pd.DataFrame:
    """Online seat/cooldown walk — same accept order as offline day sort."""
    # Group by date; within day process in (entry_ts, symbol) order online.
    by_day: dict[str, list[dict]] = {}
    for tr in raw:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    out: list[dict] = []
    for d in sorted(by_day):
        rows = sorted(by_day[d], key=lambda r: (str(r["entry_ts"]), str(r["symbol"])))
        open_pos: list[tuple[pd.Timestamp, str]] = []
        last_exit: dict[str, pd.Timestamp] = {}
        for tr in rows:
            et = to_ny(tr["entry_ts"])
            xt = to_ny(tr["exit_ts"])
            sym = str(tr["symbol"])
            open_pos = [(x, s) for x, s in open_pos if x > et]
            if any(s == sym for _, s in open_pos):
                continue
            if sym in last_exit and (et - last_exit[sym]).total_seconds() < float(
                cooldown_minutes
            ) * 60:
                continue
            if len(open_pos) >= int(max_concurrent):
                continue
            n_active = len(open_pos) + 1
            size = float(position_frac) / float(n_active)
            row = dict(tr)
            row["size"] = size
            row["size_frac"] = size
            row["pnl_frac"] = float(tr["ret"]) * size
            row["source"] = "stream"
            out.append(row)
            open_pos.append((xt, sym))
            last_exit[sym] = xt
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--session", required=True, choices=[s[0] for s in SESSIONS])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reinforce", default="none")
    ap.add_argument("--pricing", choices=["trades", "quotes"], default="trades")
    ap.add_argument("--fill-frac", type=float, default=0.8)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stride-sec", type=int, default=120)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--min-abs-stock-ret", type=float, default=0.0005)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=2.0)
    ap.add_argument("--prefer-dte", type=int, default=0)
    ap.add_argument("--allowed-dte", default="0,1,2")
    args = ap.parse_args(argv)

    if args.pricing == "quotes" and str(args.session).startswith("AM"):
        print("AM quotes usually missing; refuse quotes pricing for AM", flush=True)
        return 2

    rein = _rein_cfg(args.reinforce)
    need_mf = bool(
        rein.require_mf
        or rein.streak_min > 0
        or rein.peer_min > 0
        or rein.vol_z_min > 0
        or rein.from_open_max > 0
    )
    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    symbols = list(prof.get("symbols") or [])
    allowed_dte = [int(x) for x in args.allowed_dte.split(",") if x.strip()]
    session, s0, s1 = next(s for s in SESSIONS if s[0] == args.session)
    dates = _bdates(args.start_date, args.end_date)
    months = month_list(args.start_date, args.end_date)

    print(f"loading stock ({'mf' if need_mf else 'raw'}) {args.start_date}..{args.end_date}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
        if raw.empty:
            continue
        sdf = attach_mf_features(raw) if need_mf else raw
        sdf = sdf[(sdf["date"] >= args.start_date) & (sdf["date"] <= args.end_date)].copy()
        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
        if sdf["timestamp"].dt.tz is None:
            sdf["timestamp"] = sdf["timestamp"].dt.tz_localize(NY)
        else:
            sdf["timestamp"] = sdf["timestamp"].dt.tz_convert(NY)
        stock_by[sym] = sdf

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    raw = _collect_raw(
        symbols=symbols,
        stock_by=stock_by,
        multi_idx=multi_idx,
        otm_rungs=otm_rungs,
        dates=dates,
        session=session,
        session_bounds=(s0, s1),
        trades_root=Path(args.trades_root),
        quote_root=Path(paths["quote_1s_root"]),
        pricing=args.pricing,
        fill_frac=float(args.fill_frac),
        h=int(args.horizon_sec),
        lookback_sec=int(args.lookback_sec),
        stride_sec=int(args.stride_sec),
        min_abs=float(args.min_abs_stock_ret),
        prefer_dte=int(args.prefer_dte),
        allowed_dte=allowed_dte,
        slip=float(args.slip),
        rein=rein,
        need_mf=need_mf,
    )
    # typo guard - I had a silly ternary, fix is just float(args.fill_frac)
    off = _size_book(
        raw,
        position_frac=float(args.position_frac),
        max_concurrent=int(args.max_concurrent),
        cooldown_minutes=float(args.cooldown_minutes),
    )
    st = _stream_from_raw(
        raw,
        position_frac=float(args.position_frac),
        max_concurrent=int(args.max_concurrent),
        cooldown_minutes=float(args.cooldown_minutes),
    )
    cmp = compare(off, st)
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if len(off):
        off.to_csv(out / "offline_trades.csv", index=False)
    if len(st):
        st.to_csv(out / "stream_trades.csv", index=False)
    summary = {
        "session": session,
        "start": args.start_date,
        "end": args.end_date,
        "reinforce": args.reinforce,
        "pricing": args.pricing,
        "fill_frac": float(args.fill_frac),
        "n_raw_signals": int(len(raw)),
        "compare": cmp,
        "ok": cmp["ok"],
    }
    (out / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(cmp, indent=2), flush=True)
    print(f"ok={cmp['ok']} wrote {out}", flush=True)
    return 0 if cmp["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
