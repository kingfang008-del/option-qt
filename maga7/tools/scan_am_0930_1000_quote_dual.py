#!/usr/bin/env python3
"""AM 09:30–10:00 × quote FillSpec TP/SL × dual-window accept.

Companion to ``scan_am_0930_1000_trades_dual``. Same AM hard-cut and launch
cells; pricing is quote FillSpec (executable). Default cells = trades dual
champions.

Dual PASS (both Jan–Mar and May–Jul):
  mean>0, add>0, day_win≥0.55, n≥15, frac_max_hold≤0.50

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_quote_dual \\
    --tag research_am_0930_1000_quote_dual
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
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, path_for_ticker, to_ny
from maga7.tools.scan_am_0930_1000_trades_dual import (
    SIGNAL_END,
    SIGNAL_START,
    WINDOWS,
    _cut_am,
    _date_in_window,
    _load_events,
)
from maga7.tools.scan_launch_slope_tpsl import DEFAULT_CELLS, _filter_events, _port

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Trades dual-pass open cells (AM research champions).
DEFAULT_CELLS_NAMES = (
    "open_s3_r002_p2,"
    "open_s3_r002_p3,"
    "open_s3_r002_fp003_p2,"
    "open_s3_r002_fp003_p2_mf1"
)


def _ok(st: dict[str, Any], *, min_n: int) -> bool:
    mean = st.get("mean")
    day_win = st.get("day_win")
    mh = st.get("frac_max_hold")
    add = st.get("add")
    if mean is None or day_win is None or mh is None or add is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= 0.55
        and float(mh) <= 0.50
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--tag", default="research_am_0930_1000_quote_dual")
    ap.add_argument("--cells", default=DEFAULT_CELLS_NAMES)
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.15,0.25")
    ap.add_argument("--max-spreads", default="0.08,0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--min-n", type=int, default=15)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    quote_root = Path(paths["quote_1s_root"])
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    want = {x.strip() for x in args.cells.split(",") if x.strip()}
    cells = [c for c in DEFAULT_CELLS if c["name"] in want]
    if not cells:
        raise SystemExit(f"no cells matched {want}")

    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    max_sp_cache = max(spreads)
    max_lag_cache = max(lags)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    win_events: dict[str, pd.DataFrame] = {}
    for wname, wcfg in WINDOWS.items():
        ev = _load_events(results_dir, wcfg["events_tag"])
        ev = ev[
            ev["date"].astype(str).map(lambda d: _date_in_window(d, wcfg["start"], wcfg["end"]))
        ]
        ev = _cut_am(ev)
        win_events[wname] = ev
        print(
            f"[{wname}] after AM cut={len(ev)} dates={ev['date'].nunique() if len(ev) else 0}",
            flush=True,
        )

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
        f"AM {SIGNAL_START}-{SIGNAL_END} quote dual cells={len(cells)} "
        f"unique_sigs={len(union_meta)} sp={spreads} lag={lags} tp={tps} sl={sls}",
        flush=True,
    )

    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    resolved: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    n_miss = 0
    for i, ((date, sym, direction, ts_s), spot) in enumerate(sorted(union_meta.items())):
        if i % 100 == 0:
            print(
                f"[resolve] {i}/{len(union_meta)} ok={len(resolved)} miss={n_miss}",
                flush=True,
            )
        qkey = (sym, date)
        if qkey not in quote_cache:
            quote_cache[qkey] = load_quotes(quote_root, sym, date)
        qday = quote_cache[qkey]
        if qday is None or qday.empty:
            n_miss += 1
            continue
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
        path = path_for_ticker(qday, ticker)
        if path is None or path.empty:
            n_miss += 1
            continue
        probe = entry_quote_row(
            path,
            to_ny(ts_s),
            max_lag_sec=max_lag_cache,
            max_spread_pct=max_sp_cache,
            min_mid=float(args.min_mid),
        )
        if probe is None:
            n_miss += 1
            continue
        resolved[(date, sym, direction, ts_s)] = {
            "date": date,
            "symbol": sym,
            "dir": direction,
            "sig_ts": to_ny(ts_s),
            "ticker": ticker,
            "dte": dte,
            "path": path,
            "entry_spread_pct": probe["spread_pct"],
            "entry_lag_sec": probe["lag_sec"],
            "entry_mid": probe["mid"],
        }

    print(f"resolved_wide={len(resolved)} miss={n_miss}; scoring…", flush=True)

    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    dual_pass_list: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in cells:
        for max_sp in spreads:
            for max_lag in lags:
                for tp in tps:
                    for sl in sls:
                        win_stats: dict[str, dict[str, Any]] = {}
                        win_raw: dict[str, list[dict[str, Any]]] = {}
                        for wname in WINDOWS:
                            sigs = cell_sigs[cell["name"]].get(wname)
                            raw: list[dict[str, Any]] = []
                            if sigs is not None and not sigs.empty:
                                for _, r in sigs.iterrows():
                                    k = (
                                        str(r["date"]),
                                        str(r["symbol"]),
                                        str(r["dir"]),
                                        str(r["ts"]),
                                    )
                                    f = resolved.get(k)
                                    if f is None:
                                        continue
                                    if float(f["entry_spread_pct"]) > max_sp:
                                        continue
                                    if float(f["entry_lag_sec"]) > max_lag:
                                        continue
                                    sim = simulate_quote_tpsl(
                                        f["path"],
                                        f["sig_ts"],
                                        tp=tp,
                                        sl=sl,
                                        max_hold_sec=int(args.max_hold_sec),
                                        fill=fill,
                                        max_lag_sec=max_lag,
                                        max_spread_pct=max_sp,
                                        min_mid=float(args.min_mid),
                                    )
                                    if sim is None or not np.isfinite(sim["ret"]):
                                        continue
                                    raw.append(
                                        {
                                            "date": f["date"],
                                            "symbol": f["symbol"],
                                            "dir": f["dir"],
                                            "entry_ts": str(sim["entry_ts"]),
                                            "exit_ts": str(sim["exit_ts"]),
                                            "ticker": f["ticker"],
                                            "dte": f["dte"],
                                            "ret": sim["ret"],
                                            "exit_reason": sim["reason"],
                                            "hold_sec": sim["hold_sec"],
                                            "entry_spread_pct": sim["entry_spread_pct"],
                                            "entry_lag_sec": sim["entry_lag_sec"],
                                            "window": wname,
                                        }
                                    )
                            st = _port(raw)
                            win_stats[wname] = st
                            win_raw[wname] = raw
                            if st.get("n", 0) >= 10:
                                print(
                                    f"[{cell['name']} sp≤{max_sp} lag≤{max_lag} "
                                    f"tp{tp}/sl{sl} {wname}] n={st['n']} "
                                    f"mean={st['mean']} add={st['add']:+.3f} "
                                    f"day_win={st['day_win']}",
                                    flush=True,
                                )

                        both = _ok(win_stats["jan_mar"], min_n=int(args.min_n)) and _ok(
                            win_stats["may_jul"], min_n=int(args.min_n)
                        )
                        row: dict[str, Any] = {
                            "cell": cell["name"],
                            "session": f"AM_{SIGNAL_START}_{SIGNAL_END}",
                            "max_spread_pct": max_sp,
                            "max_lag_sec": max_lag,
                            "min_mid": float(args.min_mid),
                            "tp": tp,
                            "sl": sl,
                            "max_hold_sec": int(args.max_hold_sec),
                            "dual_pass": both,
                        }
                        for wname in WINDOWS:
                            for k, v in win_stats[wname].items():
                                row[f"{wname}_{k}"] = v
                        score_rows.append(row)
                        if both:
                            key = (
                                f"{cell['name']}|sp{max_sp}|lag{max_lag}|tp{tp}|sl{sl}"
                            )
                            dual_pass_list.append(row)
                            trade_dump[key] = pd.DataFrame(
                                win_raw["jan_mar"] + win_raw["may_jul"]
                            )
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
        key = f"{p['cell']}|sp{p['max_spread_pct']}|lag{p['max_lag_sec']}|tp{p['tp']}|sl{p['sl']}"
        if key in trade_dump:
            trade_dump[key].to_csv(
                out
                / (
                    f"trades_dual{i}_{p['cell']}_sp{p['max_spread_pct']}"
                    f"_tp{p['tp']}_sl{p['sl']}.csv"
                ),
                index=False,
            )

    # Champion trades cell spotlight
    champ = score[
        (score["cell"] == "open_s3_r002_p2")
        & (score["tp"] == 0.15)
        & (score["sl"] == 0.25)
    ].copy()
    champ_path = out / "champion_open_s3_r002_p2_tp15_sl25.csv"
    if len(champ):
        champ.to_csv(champ_path, index=False)

    verdict = "PASS" if dual_pass_list else "REJECT"
    summary = {
        "session": f"AM_{SIGNAL_START}_{SIGNAL_END}",
        "pricing": "quote_FillSpec",
        "entry_frac": float(args.entry_frac),
        "exit_frac": float(args.exit_frac),
        "min_mid": float(args.min_mid),
        "windows": WINDOWS,
        "cells": [c["name"] for c in cells],
        "n_unique_sigs": int(len(union_meta)),
        "n_resolved_wide": int(len(resolved)),
        "n_miss": int(n_miss),
        "dual_pass_n": int(len(dual_pass_list)),
        "verdict": verdict,
        "dual_pass": dual_pass_list[:30],
        "champion_trades_cell": "open_s3_r002_p2 tp15/sl25",
        "champion_quote_rows": int(len(champ)),
        "note": (
            "AM-only quote executable accept on trades dual champions. "
            "PASS here is required before any live/dry profile freeze."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass_list[:30], indent=2, default=str), encoding="utf-8"
    )
    print(f"\n=== dual PASS ({len(dual_pass_list)}) verdict={verdict} ===", flush=True)
    print(json.dumps(dual_pass_list[:10], indent=2, default=str), flush=True)
    if len(champ):
        cols = [
            c
            for c in [
                "max_spread_pct",
                "max_lag_sec",
                "jan_mar_n",
                "jan_mar_mean",
                "jan_mar_day_win",
                "may_jul_n",
                "may_jul_mean",
                "may_jul_day_win",
                "dual_pass",
            ]
            if c in champ.columns
        ]
        print("\n=== champion open_s3_r002_p2 tp15/sl25 ===", flush=True)
        print(champ[cols].to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
