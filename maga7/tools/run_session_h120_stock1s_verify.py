#!/usr/bin/env python3
"""Re-verify session H120 signals on causal stock 1s (no left-label leak).

Baseline foresight used ``stock_root`` 1m left-labeled bars and
``searchsorted(..., side='right')-1`` close → can read the *open* minute's
close up to ~59s early.

This tool:
  1) recomputes 60s momentum from ``/mnt/s990/data/raw_1s/stocks`` closes
     (last print ≤ t vs last print ≤ t−60s);
  2) keeps an event only if 1s direction matches the foresight ``dir``;
  3) optional reinforce on 1m bars aggregated from 1s, using **completed**
     bars only (bar_ts + 1m ≤ asof).

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_stock1s_verify \\
    --events-tag research_session_horizon_foresight_apr_jul \\
    --session AM_0930_1000 --reinforce MF \\
    --tag research_session_am_mf_stock1s_apr_jul
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
from maga7.common.replay import to_ny
from maga7.common.session_entry_reinforce import (
    SessionReinforceConfig,
    cfg_to_dict,
    evaluate_reinforce,
    parse_reinforce,
)
from maga7.common.signals import attach_mf_features
from maga7.common.stock_1s import load_symbol_1s_bars, shift_completed_1m
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day
from maga7.tools.run_session_h120_reinforce_ablation import VARIANTS, _fill, _honest

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")


def _ts_ns(ts: pd.Timestamp) -> int:
    return int(to_ny(ts).value)


def _dir_1s(
    ts_ns: np.ndarray,
    px: np.ndarray,
    t: pd.Timestamp,
    lookback_sec: int,
    min_abs: float,
) -> tuple[str | None, float]:
    if len(ts_ns) < 2:
        return None, float("nan")
    t_ns = _ts_ns(t)
    t0_ns = t_ns - int(lookback_sec) * 1_000_000_000
    i1 = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    i0 = int(np.searchsorted(ts_ns, t0_ns, side="right") - 1)
    if i1 < 0 or i0 < 0:
        return None, float("nan")
    a, b = float(px[i0]), float(px[i1])
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return None, float("nan")
    # require lookback anchor actually near t0 (not a stale print hours earlier)
    if abs(int(ts_ns[i0]) - t0_ns) > 5_000_000_000:  # >5s slack
        return None, float("nan")
    if abs(int(ts_ns[i1]) - t_ns) > 5_000_000_000:
        return None, float("nan")
    sr = b / a - 1.0
    if abs(sr) < float(min_abs):
        return None, sr
    return ("UP" if b > a else "DN"), sr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--min-abs-stock-ret", type=float, default=0.0005)
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK_1S))
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=2.0)
    ap.add_argument(
        "--reinforce",
        default="none",
        help="none | variant name in reinforce ablation (MF, MF_P2_FO035, …) | json",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(args.stock_1s_root)

    ev_p = results_dir / args.events_tag / "events.parquet"
    events = pd.read_parquet(ev_p) if ev_p.is_file() else pd.read_csv(
        results_dir / args.events_tag / "events.csv"
    )
    h = int(args.horizon_sec)
    sub = events[
        (events["session"].astype(str) == args.session)
        & (events["horizon_sec"].astype(int) == h)
    ].copy()
    if sub.empty:
        raise SystemExit(f"no events for {args.session} H={h}")

    sig = (
        sub.drop_duplicates(subset=["date", "symbol", "entry_ts", "dir"])
        .sort_values(["date", "entry_ts", "symbol"])
        .reset_index(drop=True)
    )
    dates = sorted(sig["date"].astype(str).unique())
    print(
        f"verify 1s stock={stock_1s} session={args.session} signals={len(sig)} "
        f"dates={dates[0]}..{dates[-1]}",
        flush=True,
    )

    # cache 1s arrays
    arr_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for sym in symbols:
        for date in dates:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw.empty:
                continue
            ts = pd.to_datetime(raw["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(NY)
            else:
                ts = ts.dt.tz_convert(NY)
            ts_ns = ts.astype("int64").to_numpy()
            px = raw["close"].astype(float).to_numpy()
            order = np.argsort(ts_ns)
            arr_cache[(sym, date)] = (ts_ns[order], px[order])
        print(f"  1s {sym} days={sum(1 for d in dates if (sym, d) in arr_cache)}", flush=True)

    # reinforce stock frames from 1s → 1m completed
    rein_name = str(args.reinforce).strip()
    cfg: SessionReinforceConfig | None = None
    if rein_name and rein_name.lower() not in ("none", "base", ""):
        if rein_name in VARIANTS:
            cfg = VARIANTS[rein_name]
        elif rein_name.startswith("{"):
            cfg = parse_reinforce(json.loads(rein_name))
        else:
            raise SystemExit(f"unknown reinforce {rein_name}; known={list(VARIANTS)}")

    stock_by: dict[str, pd.DataFrame] = {}
    if cfg is not None:
        print("building completed 1m+mf from 1s …", flush=True)
        for sym in symbols:
            bars = load_symbol_1s_bars(stock_1s, sym, dates)
            if bars.empty:
                continue
            feat = attach_mf_features(bars)
            stock_by[sym] = shift_completed_1m(feat)
            print(f"  1m-complete {sym} n={len(stock_by[sym])}", flush=True)

    n_keep = n_flip = n_drop = n_miss_1s = 0
    kept: list[dict[str, Any]] = []
    for r in sig.itertuples():
        date = str(r.date)
        sym = str(r.symbol)
        et = to_ny(r.entry_ts)
        old_dir = str(r.dir).upper()
        key = (sym, date)
        if key not in arr_cache:
            n_miss_1s += 1
            n_drop += 1
            continue
        ts_ns, px = arr_cache[key]
        new_dir, sr = _dir_1s(
            ts_ns, px, et, int(args.lookback_sec), float(args.min_abs_stock_ret)
        )
        if new_dir is None:
            n_drop += 1
            continue
        if new_dir != old_dir:
            n_flip += 1
            continue
        if cfg is not None:
            ok, meta = evaluate_reinforce(
                stock_by=stock_by,
                symbol=sym,
                date=date,
                entry_ts=et,
                direction=old_dir,
                cfg=cfg,
                peer_symbols=symbols,
            )
            if not ok:
                n_drop += 1
                continue
        else:
            meta = {}
        n_keep += 1
        ret = float(r.clock_ret)
        if not np.isfinite(ret):
            continue
        kept.append(
            {
                "date": date,
                "symbol": sym,
                "dir": old_dir,
                "entry_ts": str(et),
                "exit_ts": str(et + pd.Timedelta(seconds=h)),
                "sig_ts": str(et),
                "ticker": str(getattr(r, "ticker", "") or ""),
                "ret": ret,
                "reason": f"clock_H{h}",
                "session": args.session,
                "stock_ret_lb_1s": sr,
                "variant": rein_name or "BASE",
                **{f"rein_{k}": v for k, v in meta.items() if k in ("mf10", "reason")},
            }
        )

    trades = _fill(
        kept,
        position_frac=float(args.position_frac),
        max_concurrent=int(args.max_concurrent),
        cooldown_minutes=float(args.cooldown_minutes),
    )
    if len(trades) and "pnl_frac" not in trades.columns and "size" in trades.columns:
        trades["pnl_frac"] = trades["ret"].astype(float) * trades["size"].astype(float)

    eq = _equity_stats(trades)
    hon = _honest(trades)
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if len(trades):
        trades.to_csv(out / "trades.csv", index=False)
    summary = {
        "session": args.session,
        "events_tag": args.events_tag,
        "stock_1s_root": str(stock_1s),
        "horizon_sec": h,
        "lookback_sec": int(args.lookback_sec),
        "reinforce": rein_name,
        "reinforce_cfg": cfg_to_dict(cfg) if cfg is not None else None,
        "n_signals_in": int(len(sig)),
        "n_keep_dir_match": int(n_keep),
        "n_flip": int(n_flip),
        "n_drop": int(n_drop),
        "n_miss_1s": int(n_miss_1s),
        "keep_rate": float(n_keep / max(len(sig), 1)),
        "n_trades": int(len(trades)),
        "trade_mean": hon.get("trade_mean"),
        "trade_win": eq.get("trade_win"),
        **hon,
        "note": (
            "Direction from causal 1s closes; reinforce mf on 1s→1m with +1m "
            "availability shift (no left-label leak)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
