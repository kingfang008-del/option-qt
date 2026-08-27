#!/usr/bin/env python3
"""Quote FillSpec dual accept for CORE DN sync certainty morph champions.

Companion to ``scan_am_certainty_morph_tpsl`` (CORE-only DN sync trades PASS).

Default cells = top dual-pass from
``research_certainty_morph_core_dn_sync_dual_n7``.

Windows: may_jul09 / jul10_23. Session: CORE 10:30–11:30. Dir: DN.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_certainty_morph_quote_dual \\
    --tag research_certainty_morph_core_dn_sync_quote_dual
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_certainty_morph_tpsl import _stock_signed
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path, _stats, _ok
from maga7.tools.scan_session_horizon_foresight import (
    _spot_at_arr,
    _stock_arrays,
    _stock_dir_arr,
)

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SESS_START, SESS_END = "10:30", "11:30"
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)

DEFAULT_CELLS = (
    {"name": "sync_t0.003_ss30_so30_tp0.2_sl0.15", "thr": 0.003, "ss": 30, "so": 30, "tp": 0.20, "sl": 0.15},
    {"name": "sync_t0.003_ss30_so30_tp0.25_sl0.15", "thr": 0.003, "ss": 30, "so": 30, "tp": 0.25, "sl": 0.15},
    {"name": "sync_t0.003_ss60_so30_tp0.2_sl0.15", "thr": 0.003, "ss": 60, "so": 30, "tp": 0.20, "sl": 0.15},
    {"name": "sync_t0.003_ss60_so30_tp0.25_sl0.15", "thr": 0.003, "ss": 60, "so": 30, "tp": 0.25, "sl": 0.15},
    {"name": "ctrl_nosync_t0.003_tp0.2_sl0.15", "thr": 0.003, "ss": 0, "so": 0, "tp": 0.20, "sl": 0.15},
)


def _quote_opt_ret(
    path: pd.DataFrame,
    t: pd.Timestamp,
    lookback_sec: int,
    *,
    fill: FillSpec,
    max_lag_sec: float,
    max_spread_pct: float,
    min_mid: float,
) -> float | None:
    """Option ret over lookback using FillSpec buy@t-lb → sell@t."""
    t1 = to_ny(t)
    t0 = t1 - pd.Timedelta(seconds=int(lookback_sec))
    ent0 = entry_quote_row(
        path, t0, max_lag_sec=max_lag_sec, max_spread_pct=max_spread_pct, min_mid=min_mid
    )
    if ent0 is None:
        return None
    # mark at t1: last quote ≤ t1 with usable book
    upto = path[(path["timestamp"] >= ent0["entry_ts"]) & (path["timestamp"] <= t1)]
    if upto.empty:
        return None
    r1 = upto.iloc[-1]
    bid, ask = float(r1["bid"]), float(r1["ask"])
    if not (np.isfinite(bid) and np.isfinite(ask) and ask > bid > 0):
        return None
    entry_px = fill.buy(ent0["bid"], ent0["ask"])
    exit_px = fill.sell(bid, ask)
    if entry_px <= 0:
        return None
    return float(exit_px / entry_px - 1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_certainty_morph_core_dn_sync_quote_dual")
    ap.add_argument("--max-spreads", default="0.08,0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=5.0)
    ap.add_argument("--min-n", type=int, default=7)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--stride-sec", type=int, default=60)
    args = ap.parse_args(argv)

    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = session_dates(start_all, end_all)
    thr_need = sorted({float(c["thr"]) for c in DEFAULT_CELLS})
    print(
        f"CORE DN sync QUOTE dual {start_all}..{end_all} cells={len(DEFAULT_CELLS)} "
        f"sp={spreads} lag={lags}",
        flush=True,
    )

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            qday = _prep_path(load_quotes(quote_root, sym, date))
            if qday is None or qday.empty:
                continue
            ts_ns, px = _stock_arrays(day)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            t0 = pd.Timestamp(f"{date} {SESS_START}:00", tz=NY) + pd.Timedelta(
                seconds=int(args.lookback_sec)
            )
            t1 = pd.Timestamp(f"{date} {SESS_END}:00", tz=NY)
            fired: set[float] = set()
            t = t0
            stride = pd.Timedelta(seconds=int(args.stride_sec))
            while t < t1:
                for thr in thr_need:
                    direction, sr = _stock_dir_arr(
                        ts_ns, px, t, int(args.lookback_sec), float(thr)
                    )
                    if direction != "DN" or float(thr) in fired:
                        continue
                    spot = _spot_at_arr(ts_ns, px, t)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction="DN",
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=[0, 1, 2],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker:
                        continue
                    path = _prep_path(path_for_ticker(qday, ticker))
                    if path is None or path.empty:
                        continue
                    probe = entry_quote_row(
                        path,
                        t,
                        max_lag_sec=max(lags),
                        max_spread_pct=max(spreads),
                        min_mid=float(args.min_mid),
                    )
                    if probe is None:
                        continue
                    fired.add(float(thr))
                    arms.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "thr": float(thr),
                            "arm_ts": to_ny(t),
                            "stock_ret_lb": float(sr),
                            "ticker": ticker,
                            "dte": dte,
                            "path": path,
                            "sts": ts_ns,
                            "spx": px,
                            "probe_spread": float(probe["spread_pct"]),
                            "probe_lag": float(probe["lag_sec"]),
                        }
                    )
                t += stride

    print(f"arms_resolvable={len(arms)}", flush=True)

    def window_of(date: str) -> str | None:
        for wname, a, b in WINDOWS:
            if a <= date <= b:
                return wname
        return None

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in DEFAULT_CELLS:
        thr, ss, so = float(cell["thr"]), int(cell["ss"]), int(cell["so"])
        tp, sl = float(cell["tp"]), float(cell["sl"])
        for max_sp in spreads:
            for max_lag in lags:
                win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
                n_sig = n_block = n_fill = 0
                for arm in arms:
                    if float(arm["thr"]) != thr:
                        continue
                    wname = window_of(str(arm["date"]))
                    if wname is None:
                        continue
                    n_sig += 1
                    if float(arm["probe_spread"]) > max_sp or float(arm["probe_lag"]) > max_lag:
                        n_block += 1
                        continue
                    entry_ts = arm["arm_ts"]
                    if ss > 0:
                        sret = _stock_signed(arm["sts"], arm["spx"], entry_ts, ss, "DN")
                        if sret is None or sret < 0:
                            n_block += 1
                            continue
                    if so > 0:
                        oret = _quote_opt_ret(
                            arm["path"],
                            entry_ts,
                            so,
                            fill=fill,
                            max_lag_sec=max_lag,
                            max_spread_pct=max_sp,
                            min_mid=float(args.min_mid),
                        )
                        if oret is None or oret <= 0:
                            n_block += 1
                            continue
                    sim = simulate_quote_tpsl(
                        arm["path"],
                        entry_ts,
                        tp=tp,
                        sl=sl,
                        max_hold_sec=int(args.max_hold_sec),
                        fill=fill,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=float(args.min_mid),
                    )
                    if sim is None or not np.isfinite(sim["ret"]):
                        n_block += 1
                        continue
                    n_fill += 1
                    win_raw[wname].append(
                        {
                            "date": arm["date"],
                            "symbol": arm["symbol"],
                            "dir": "DN",
                            "entry_ts": str(sim["entry_ts"]),
                            "exit_ts": str(sim["exit_ts"]),
                            "ticker": arm["ticker"],
                            "dte": arm["dte"],
                            "ret": sim["ret"],
                            "exit_reason": sim["reason"],
                            "hold_sec": sim["hold_sec"],
                            "cell": cell["name"],
                            "window": wname,
                        }
                    )

                win_stats: dict[str, dict] = {}
                sized_all: list[dict] = []
                for wname, _, _ in WINDOWS:
                    raw = win_raw[wname]
                    by_d: dict[str, list] = {}
                    for r in raw:
                        by_d.setdefault(str(r["date"]), []).append(r)
                    sized: list[dict] = []
                    for _, rs in sorted(by_d.items()):
                        sized.extend(
                            _portfolio_day(
                                sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                                position_frac=float(args.position_frac),
                                max_concurrent=int(args.max_concurrent),
                                cooldown_minutes=float(args.cooldown_minutes),
                            )
                        )
                    win_stats[wname] = _stats(sized)
                    sized_all.extend(sized)

                both = True
                for wname, _, _ in WINDOWS:
                    mn = int(args.min_n)
                    if wname == "jul10_23":
                        mn = min(mn, 7)
                    if not _ok(
                        win_stats[wname],
                        min_n=mn,
                        min_day_win=float(args.min_day_win),
                    ):
                        both = False
                        break

                row: dict[str, Any] = {
                    "cell": cell["name"],
                    "thr": thr,
                    "ss": ss,
                    "so": so,
                    "tp": tp,
                    "sl": sl,
                    "max_spread_pct": max_sp,
                    "max_lag_sec": max_lag,
                    "dual_pass": both,
                    "n_sig": n_sig,
                    "n_block": n_block,
                    "n_fill": n_fill,
                    "fill_frac": float(n_fill / n_sig) if n_sig else None,
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    key = f"{cell['name']}_sp{max_sp}_lag{max_lag}"
                    dual_pass.append(row)
                    trade_dump[key] = pd.DataFrame(sized_all)
                    print(f"  *** QUOTE DUAL PASS {key}", flush=True)

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass = sorted(
        dual_pass,
        key=lambda r: (
            float(r.get("may_jul09_add") or 0) + float(r.get("jul10_23_add") or 0)
        ),
        reverse=True,
    )
    for i, p in enumerate(dual_pass[:10]):
        key = f"{p['cell']}_sp{p['max_spread_pct']}_lag{p['max_lag_sec']}"
        if key in trade_dump and len(trade_dump[key]):
            trade_dump[key].to_csv(out / f"trades_dual{i:02d}_{key}.csv", index=False)

    summary = {
        "session": "CORE_1030_1130",
        "dir": "DN",
        "morph": "sync",
        "book": "quote_fill_tpsl",
        "complements": "research_certainty_morph_core_dn_sync_dual_n7",
        "windows": [list(w) for w in WINDOWS],
        "n_arms": int(len(arms)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "PASS" if dual_pass else "REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "decision": (
            "CORE_DN_SYNC_QUOTE_PASS"
            if dual_pass
            else "CORE_DN_SYNC_TRADES_PASS_QUOTE_REJECT"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:30], indent=2, default=str), encoding="utf-8"
    )

    print("\n=== QUOTE", summary["verdict"], summary["decision"], flush=True)
    if dual_pass:
        print(json.dumps(dual_pass[0], indent=2, default=str), flush=True)
    elif not score.empty:
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        near = score.sort_values("_sum", ascending=False).head(10)
        cols = [
            c
            for c in [
                "cell",
                "max_spread_pct",
                "max_lag_sec",
                "fill_frac",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_win",
                "may_jul09_day_win",
                "may_jul09_add",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_win",
                "jul10_23_day_win",
                "jul10_23_add",
            ]
            if c in near.columns
        ]
        print(near[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
