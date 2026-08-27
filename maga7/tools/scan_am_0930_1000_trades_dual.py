#!/usr/bin/env python3
"""AM sleeve only: 09:30–10:00 × trade-last TP/SL × dual-window accept.

Converges research to the open half-hour. Pricing book is
``/mnt/s990/new_option_data_s3_trades`` (last ± slip). Stock edges from
launch-slope events, hard-cut to signal time ∈ [09:30, 10:00).

Dual PASS (both Jan–Mar and May–Jul):
  mean>0, add>0, day_win≥0.55, n≥15, frac_max_hold≤0.50

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_trades_dual \\
    --tag research_am_0930_1000_trades_dual
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
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.tools.scan_launch_slope_tpsl import DEFAULT_CELLS, _filter_events, _port
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SIGNAL_START = "09:30"
SIGNAL_END = "10:00"

WINDOWS = {
    "jan_mar": {
        "events_tag": "research_launch_slope_jan_mar_am",
        "start": "2026-01-01",
        "end": "2026-03-31",
    },
    "may_jul": {
        "events_tag": "research_launch_slope_may_jul",
        "start": "2026-05-01",
        "end": "2026-07-22",
    },
}


def _load_events(results_dir: Path, tag: str) -> pd.DataFrame:
    p = results_dir / tag / "events.parquet"
    if p.is_file():
        ev = pd.read_parquet(p)
    else:
        c = results_dir / tag / "events.csv"
        if not c.is_file():
            raise SystemExit(f"missing events: {results_dir / tag}")
        ev = pd.read_csv(c)
    ev = ev.drop_duplicates(
        ["date", "symbol", "dir", "ts", "session", "slope_sec", "abs_ret_min"]
    ).reset_index(drop=True)
    return ev


def _cut_am(events: pd.DataFrame) -> pd.DataFrame:
    """Keep signals with NY clock in [09:30, 10:00)."""
    ts = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert(NY)
    t = ts.dt.time
    t0 = pd.Timestamp(SIGNAL_START).time()
    t1 = pd.Timestamp(SIGNAL_END).time()
    mask = (t >= t0) & (t < t1)
    out = events.loc[mask].copy()
    out["ts"] = ts.loc[mask].astype(str)
    return out.reset_index(drop=True)


def _date_in_window(date: str, start: str, end: str) -> bool:
    return start <= str(date) <= end


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--tag", default="research_am_0930_1000_trades_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.08,0.10,0.15,0.25")
    ap.add_argument(
        "--cells",
        default="",
        help="comma open cell names; empty=all open_* defaults",
    )
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    trades_root = Path(args.trades_root)
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]

    cells = [c for c in DEFAULT_CELLS if str(c["session"]).startswith("open_")]
    if args.cells.strip():
        want = {x.strip() for x in args.cells.split(",") if x.strip()}
        cells = [c for c in cells if c["name"] in want]
        if not cells:
            raise SystemExit(f"no open cells matched {want}")

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    # Load + AM-cut events per window
    win_events: dict[str, pd.DataFrame] = {}
    for wname, wcfg in WINDOWS.items():
        ev = _load_events(results_dir, wcfg["events_tag"])
        ev = ev[ev["date"].astype(str).map(lambda d: _date_in_window(d, wcfg["start"], wcfg["end"]))]
        ev = _cut_am(ev)
        win_events[wname] = ev
        print(
            f"[{wname}] events_tag={wcfg['events_tag']} after AM cut={len(ev)} "
            f"dates={ev['date'].nunique() if len(ev) else 0}",
            flush=True,
        )

    # Resolve fills once across all windows
    union_meta: dict[tuple[str, str, str, str], float | None] = {}
    cell_sigs: dict[str, dict[str, pd.DataFrame]] = {c["name"]: {} for c in cells}
    for wname, ev in win_events.items():
        for cell in cells:
            sigs = _filter_events(ev, cell)
            cell_sigs[cell["name"]][wname] = sigs
            for _, r in sigs.iterrows():
                k = (str(r["date"]), str(r["symbol"]), str(r["dir"]), str(r["ts"]))
                px = float(r["entry_px"]) if "entry_px" in r and pd.notna(r["entry_px"]) else None
                union_meta[k] = px

    print(
        f"AM {SIGNAL_START}-{SIGNAL_END} cells={len(cells)} unique_sigs={len(union_meta)} "
        f"tp={tps} sl={sls} pricing=trades@{trades_root}",
        flush=True,
    )

    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fills: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    n_miss = 0
    for i, ((date, sym, direction, ts_s), spot) in enumerate(sorted(union_meta.items())):
        if i % 200 == 0:
            print(f"[resolve] {i}/{len(union_meta)} fills={len(fills)} miss={n_miss}", flush=True)
        qkey = (sym, date)
        if qkey not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[qkey] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        trade_paths = path_cache[qkey]
        if not trade_paths:
            n_miss += 1
            continue
        entry_ts = to_ny(ts_s)
        by_dte = multi_idx.get((sym, date))
        ticker, dte, _ = resolve_open_lock_contract(
            by_dte,
            direction=direction,
            moneyness="ATM",
            spot=spot,
            prefer_dte=0,
            allowed_dte=[0, 1, 2],
            clear_otm_thresh=0.01,
            ladder=True,
            otm_rungs=otm_rungs,
        )
        if not ticker:
            n_miss += 1
            continue
        key = str(ticker).replace("O:", "")
        path = trade_paths.get(key)
        if path is None:
            n_miss += 1
            continue
        pts, plast = path
        fills[(date, sym, direction, ts_s)] = {
            "date": date,
            "symbol": sym,
            "dir": direction,
            "entry_ts": entry_ts,
            "ticker": ticker,
            "dte": dte,
            "pts": pts,
            "plast": plast,
        }

    print(f"resolved fills={len(fills)} miss={n_miss}; scoring dual grid…", flush=True)

    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    dual_pass_list: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in cells:
        for tp in tps:
            for sl in sls:
                win_stats: dict[str, dict[str, Any]] = {}
                win_raw: dict[str, list[dict[str, Any]]] = {}
                for wname in WINDOWS:
                    sigs = cell_sigs[cell["name"]].get(wname)
                    if sigs is None or sigs.empty:
                        win_stats[wname] = _port([])
                        win_raw[wname] = []
                        continue
                    raw: list[dict[str, Any]] = []
                    for _, r in sigs.iterrows():
                        k = (str(r["date"]), str(r["symbol"]), str(r["dir"]), str(r["ts"]))
                        f = fills.get(k)
                        if f is None:
                            continue
                        sim = simulate_trade_tpsl(
                            f["pts"],
                            f["plast"],
                            f["entry_ts"],
                            tp=tp,
                            sl=sl,
                            max_hold_sec=int(args.max_hold_sec),
                            slip=float(args.slip),
                        )
                        if sim is None or not np.isfinite(sim["ret"]):
                            continue
                        et = f["entry_ts"]
                        raw.append(
                            {
                                "date": f["date"],
                                "symbol": f["symbol"],
                                "dir": f["dir"],
                                "entry_ts": str(et),
                                "exit_ts": str(et + pd.Timedelta(seconds=sim["hold_sec"])),
                                "ticker": f["ticker"],
                                "dte": f["dte"],
                                "ret": sim["ret"],
                                "exit_reason": sim["reason"],
                                "hold_sec": sim["hold_sec"],
                                "window": wname,
                            }
                        )
                    st = _port(raw)
                    win_stats[wname] = st
                    win_raw[wname] = raw
                    print(
                        f"[{cell['name']} tp={tp} sl={sl} {wname}] "
                        f"n={st['n']} mean={st['mean']} add={st['add']:+.3f} "
                        f"day_win={st['day_win']} mh%={st.get('frac_max_hold')}",
                        flush=True,
                    )

                def _ok(st: dict[str, Any]) -> bool:
                    mean = st.get("mean")
                    day_win = st.get("day_win")
                    mh = st.get("frac_max_hold")
                    add = st.get("add")
                    if mean is None or day_win is None or mh is None or add is None:
                        return False
                    return bool(
                        int(st.get("n") or 0) >= int(args.min_n)
                        and float(mean) > 0
                        and float(add) > 0
                        and float(day_win) >= 0.55
                        and float(mh) <= 0.50
                    )

                both = _ok(win_stats["jan_mar"]) and _ok(win_stats["may_jul"])
                row: dict[str, Any] = {
                    "cell": cell["name"],
                    "session": f"AM_{SIGNAL_START.replace(':', '')}_{SIGNAL_END.replace(':', '')}",
                    "signal_start": SIGNAL_START,
                    "signal_end": SIGNAL_END,
                    "tp": tp,
                    "sl": sl,
                    "max_hold_sec": int(args.max_hold_sec),
                    "dual_pass": both,
                }
                for wname in WINDOWS:
                    st = win_stats[wname]
                    for k, v in st.items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    key = f"{cell['name']}|tp{tp}|sl{sl}"
                    dual_pass_list.append(row)
                    all_raw = win_raw["jan_mar"] + win_raw["may_jul"]
                    trade_dump[key] = pd.DataFrame(all_raw)
                    print(f"  *** DUAL PASS {key}", flush=True)

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass_list = sorted(
        dual_pass_list,
        key=lambda r: (
            float(r.get("may_jul_add") or 0) + float(r.get("jan_mar_add") or 0)
        ),
        reverse=True,
    )
    for i, p in enumerate(dual_pass_list[:12]):
        key = f"{p['cell']}|tp{p['tp']}|sl{p['sl']}"
        if key in trade_dump:
            trade_dump[key].to_csv(
                out / f"trades_dual{i}_{p['cell']}_tp{p['tp']}_sl{p['sl']}.csv",
                index=False,
            )

    verdict = "PASS" if dual_pass_list else "REJECT"
    summary = {
        "session": f"AM_{SIGNAL_START}_{SIGNAL_END}",
        "signal_start": SIGNAL_START,
        "signal_end": SIGNAL_END,
        "pricing": "new_option_data_s3_trades",
        "slip": float(args.slip),
        "exit": "tp_sl_first_passage_trade_last",
        "max_hold_sec_safety": int(args.max_hold_sec),
        "windows": WINDOWS,
        "n_cells": len(cells),
        "n_unique_sigs": int(len(union_meta)),
        "n_resolved_fills": int(len(fills)),
        "n_miss": int(n_miss),
        "dual_pass_n": int(len(dual_pass_list)),
        "verdict": verdict,
        "dual_pass": dual_pass_list[:30],
        "note": (
            "Research accept on trade-last book for AM open half-hour only. "
            "Not a quote-executable promote; quote FillSpec still required before live."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass_list[:30], indent=2, default=str), encoding="utf-8"
    )
    print(f"\n=== dual PASS ({len(dual_pass_list)}) verdict={verdict} ===", flush=True)
    print(json.dumps(dual_pass_list[:10], indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
