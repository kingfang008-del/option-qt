#!/usr/bin/env python3
"""Launch-slope entries + option trade-last TP/SL exits (no clock primary).

Reuses stock 1s launch edges from ``research_launch_slope_*`` events, fills on
``new_option_data_s3_trades`` last ± slip, exits on first +tp / −sl.
``max_hold_sec`` is safety flatten only.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_launch_slope_tpsl \\
    --events-tag research_launch_slope_may_jul \\
    --tag research_launch_slope_tpsl_may_jul
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
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Entry cells (no clock H). Includes prior option-fill winners + nearby filters.
DEFAULT_CELLS: list[dict[str, Any]] = [
    {
        "name": "open_s3_r002_p3",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 0,
    },
    {
        "name": "open_s3_r002_p2",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "open_s3_r002_fp003_p2",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "open_s3_r002_fp003_p2_mf1",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 1,
    },
    {
        "name": "open_s3_r003_fp003_p2",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.003,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "open_s5_r002_fp005_mf1",
        "session": "open_0930_1030",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 0,
        "mf_confirm": 1,
    },
    {
        "name": "open_s5_r002_p3",
        "session": "open_0930_1030",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 0,
    },
    {
        "name": "open_s10_r003_fp005_p3_mf1",
        "session": "open_0930_1030",
        "slope_sec": 10,
        "abs_ret_min": 0.003,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 1,
    },
    {
        "name": "mid_s5_r002_fp005",
        "session": "mid_1030_1100",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 0,
        "mf_confirm": 0,
    },
    {
        "name": "mid_s5_r002_fp005_vz1",
        "session": "mid_1030_1100",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 0,
        "mf_confirm": 0,
    },
    {
        "name": "mid_s5_r002_p2",
        "session": "mid_1030_1100",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "mid_s3_r002_p3",
        "session": "mid_1030_1100",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 0,
    },
    {
        "name": "mid_s10_r002_fp005_p2",
        "session": "mid_1030_1100",
        "slope_sec": 10,
        "abs_ret_min": 0.002,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
]


def _port(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "red_days": 0}
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(by[d], position_frac=0.10, max_concurrent=2, cooldown_minutes=2.0)
        )
    if not sized:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None, "red_days": 0}
    t = pd.DataFrame(sized)
    t["pnl_frac"] = t["ret"].astype(float) * t["size"].astype(float)
    day = t.groupby("date")["pnl_frac"].sum()
    reasons = pd.Series([r.get("exit_reason") for r in sized])
    return {
        "n": int(len(t)),
        "mean": float(t["ret"].mean()),
        "win": float((t["ret"] > 0).mean()),
        "add": float(t["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "red_days": int((day < 0).sum()),
        "tpd": float(len(t) / max(t["date"].nunique(), 1)),
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons == "sl").mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "hold_p50": float(pd.Series([r.get("hold_sec") for r in sized]).median()),
    }


def _filter_events(events: pd.DataFrame, cell: dict[str, Any]) -> pd.DataFrame:
    sess = str(cell["session"])
    slope = int(cell["slope_sec"])
    thr = float(cell["abs_ret_min"])
    fp = float(cell["from_prev_min"])
    vz = float(cell["vol_z_min"])
    peer = int(cell["peer_min"])
    mfc = int(cell.get("mf_confirm", 0) or 0)
    sub = events[
        (events["session"] == sess)
        & (events["slope_sec"] == slope)
        & (np.isclose(events["abs_ret_min"].astype(float), thr))
    ].copy()
    if sub.empty:
        return sub
    up_ok = (sub["dir"] == "UP") & (sub["from_prev"] >= fp)
    dn_ok = (sub["dir"] == "DN") & (sub["from_prev"] <= -fp)
    sub = sub[up_ok | dn_ok]
    sub = sub[(sub["vol_z"].isna()) | (sub["vol_z"] >= vz)]
    sub = sub[sub["peer_n"] >= peer]
    if mfc:
        if "mf_ok" in sub.columns:
            sub = sub[sub["mf_ok"].astype(bool)]
        else:
            return sub.iloc[0:0]
    return sub.sort_values("ts").drop_duplicates(
        ["date", "symbol", "dir", "session", "slope_sec", "abs_ret_min"], keep="first"
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", default="research_launch_slope_may_jul")
    ap.add_argument("--tag", default="research_launch_slope_tpsl_may_jul")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--sls", default="0.05,0.08,0.10,0.15,0.25")
    ap.add_argument("--cells", default="", help="comma cell names; empty=all defaults")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    events_path = results_dir / args.events_tag / "events.parquet"
    if events_path.is_file():
        events = pd.read_parquet(events_path)
    else:
        csv_path = results_dir / args.events_tag / "events.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing events: {events_path}")
        events = pd.read_csv(csv_path)

    # One row per stock edge (events are duplicated across stock horizons).
    events = events.drop_duplicates(
        ["date", "symbol", "dir", "ts", "session", "slope_sec", "abs_ret_min"]
    ).reset_index(drop=True)

    cells = list(DEFAULT_CELLS)
    if args.cells.strip():
        want = {x.strip() for x in args.cells.split(",") if x.strip()}
        cells = [c for c in DEFAULT_CELLS if c["name"] in want]
        if not cells:
            raise SystemExit(f"no cells matched {want}")

    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]
    trades_root = Path(args.trades_root)

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    # Resolve fillable signals once (union of all cells).
    union_meta: dict[tuple[str, str, str, str], float | None] = {}
    cell_sigs: dict[str, pd.DataFrame] = {}
    for cell in cells:
        sigs = _filter_events(events, cell)
        cell_sigs[cell["name"]] = sigs
        for _, r in sigs.iterrows():
            k = (str(r["date"]), str(r["symbol"]), str(r["dir"]), str(r["ts"]))
            px = float(r["entry_px"]) if "entry_px" in r and pd.notna(r["entry_px"]) else None
            union_meta[k] = px
    print(
        f"launch TP/SL cells={len(cells)} unique_sigs={len(union_meta)} "
        f"tp={tps} sl={sls} max_hold={args.max_hold_sec}s",
        flush=True,
    )

    # Cache trade paths by (sym, date).
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fills: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    n_miss = 0
    for i, ((date, sym, direction, ts_s), spot) in enumerate(sorted(union_meta.items())):
        if i % 200 == 0:
            print(f"[resolve] {i}/{len(union_meta)} fills={len(fills)} miss={n_miss}", flush=True)
        qkey = (sym, date)
        if qkey not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[qkey] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
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

    print(f"resolved fills={len(fills)} miss={n_miss}; scoring grid…", flush=True)

    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    best_trades: dict[str, pd.DataFrame] = {}

    for cell in cells:
        sigs = cell_sigs[cell["name"]]
        for tp in tps:
            for sl in sls:
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
                            "mfe": sim["mfe"],
                            "mae": sim["mae"],
                        }
                    )
                st = _port(raw)
                row = {
                    "cell": cell["name"],
                    "session": cell["session"],
                    "slope_sec": cell["slope_sec"],
                    "abs_ret_min": cell["abs_ret_min"],
                    "from_prev_min": cell["from_prev_min"],
                    "vol_z_min": cell["vol_z_min"],
                    "peer_min": cell["peer_min"],
                    "mf_confirm": cell["mf_confirm"],
                    "n_signals": int(len(sigs)),
                    "tp": tp,
                    "sl": sl,
                    "max_hold_sec": int(args.max_hold_sec),
                    **st,
                }
                score_rows.append(row)
                key = f"{cell['name']}|tp{tp}|sl{sl}"
                if st.get("n", 0) > 0 and st.get("mean") is not None and st["mean"] > 0:
                    best_trades[key] = pd.DataFrame(raw)
                print(
                    f"[{cell['name']} tp={tp} sl={sl}] n={st['n']} mean={st['mean']} "
                    f"add={st['add']:+.3f} day_win={st['day_win']} "
                    f"tp%={st.get('frac_tp')} sl%={st.get('frac_sl')} mh%={st.get('frac_max_hold')}",
                    flush=True,
                )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    picks: list[dict[str, Any]] = []
    if len(score):
        ok = score[
            (score["mean"].fillna(-1) > 0)
            & (score["add"].fillna(0) > 0)
            & (score["day_win"].fillna(0) >= 0.55)
            & (score["n"].fillna(0) >= 20)
            & (score["frac_max_hold"].fillna(1) <= 0.50)
        ].sort_values(["session", "add"], ascending=[True, False])
        picks = ok.to_dict(orient="records")
        for i, p in enumerate(picks[:12]):
            key = f"{p['cell']}|tp{p['tp']}|sl{p['sl']}"
            if key in best_trades:
                best_trades[key].to_csv(
                    out / f"trades_pick{i}_{p['cell']}_tp{p['tp']}_sl{p['sl']}.csv",
                    index=False,
                )

    dates = sorted(events["date"].astype(str).unique())
    summary = {
        "start": dates[0] if dates else None,
        "end": dates[-1] if dates else None,
        "events_tag": args.events_tag,
        "exit": "tp_sl_first_passage_trade_last",
        "max_hold_sec_safety": int(args.max_hold_sec),
        "slip": float(args.slip),
        "tps": tps,
        "sls": sls,
        "n_cells": len(cells),
        "n_unique_sigs": int(len(union_meta)),
        "n_resolved_fills": int(len(fills)),
        "n_picks": int(len(picks)),
        "picks": picks[:30],
        "top_by_add": (
            score.sort_values("add", ascending=False).head(20).to_dict(orient="records")
            if len(score)
            else []
        ),
        "note": (
            "Launch-slope 1s stock edges; option PnL on trade-last ± slip with TP/SL. "
            "No fixed clock hold as primary exit. Prior quote/horizon +19% cells are "
            "re-scored here under trade TP/SL."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "picks.json").write_text(json.dumps(picks[:30], indent=2, default=str), encoding="utf-8")
    print(f"\n=== picks ({len(picks)}) ===", flush=True)
    print(json.dumps(picks[:15], indent=2, default=str), flush=True)
    if len(score):
        print("\n=== top by add (any sign) ===", flush=True)
        cols = [
            "cell",
            "tp",
            "sl",
            "n",
            "mean",
            "win",
            "add",
            "day_win",
            "frac_tp",
            "frac_sl",
            "frac_max_hold",
            "hold_p50",
        ]
        print(score.sort_values("add", ascending=False)[cols].head(25).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
