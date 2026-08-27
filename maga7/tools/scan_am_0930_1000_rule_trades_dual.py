#!/usr/bin/env python3
"""AM sleeve rule search: 09:30–10:00 × trades TP/SL × dual-window.

Complements the 10:30 research_baseline. Signals from causal stock 1s in
``[09:30, 10:00)``; pricing ``/mnt/s990/new_option_data_s3_trades`` (last±slip);
exit = first-passage TP/SL (``max_hold_sec`` safety only).

Rule families:
  - ``fo_HHMM``: Mag7(+GOOGL) from_open at clock + peer breadth
  - ``launch_sN``: first local launch impulse (slope N sec) + peer
  - ``qqq_fo_HHMM``: QQQ-only from_open clock (open_cont family)

Dual PASS (both Jan–Mar and May–Jul):
  mean>0, add>0, day_win≥0.55, n≥15, frac_max_hold≤0.50

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_rule_trades_dual \\
    --tag research_am_0930_1000_rule_trades_dual
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
from maga7.common.launch_slope import attach_launch_slope_features, launch_edges_multi
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
MAG7 = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
SIGNAL_START = "09:30"
SIGNAL_END = "10:00"

WINDOWS = (
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("may_jul", "2026-05-01", "2026-07-22"),
)


def _port(
    rows: list[dict[str, Any]],
    *,
    position_frac: float = 0.10,
    max_concurrent: int = 1,
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "frac_tp": None,
            "frac_sl": None,
            "frac_max_hold": None,
            "hold_p50": None,
        }
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r["date"]), []).append(r)
    sized: list[dict] = []
    for d in sorted(by):
        sized.extend(
            _portfolio_day(
                by[d],
                position_frac=float(position_frac),
                max_concurrent=int(max_concurrent),
                cooldown_minutes=0.0,
            )
        )
    if not sized:
        return {
            "n": 0,
            "mean": None,
            "win": None,
            "add": 0.0,
            "day_win": None,
            "frac_tp": None,
            "frac_sl": None,
            "frac_max_hold": None,
            "hold_p50": None,
        }
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
        "worst_day": float(day.min()),
        "frac_tp": float((reasons == "tp").mean()) if len(reasons) else None,
        "frac_sl": float((reasons == "sl").mean()) if len(reasons) else None,
        "frac_max_hold": float((reasons == "max_hold").mean()) if len(reasons) else None,
        "hold_p50": float(pd.Series([r.get("hold_sec") for r in sized]).median()),
    }


def _ok(st: dict[str, Any], *, min_n: int) -> bool:
    mean, day_win, mh, add = (
        st.get("mean"),
        st.get("day_win"),
        st.get("frac_max_hold"),
        st.get("add"),
    )
    if mean is None or day_win is None or mh is None or add is None:
        return False
    return bool(
        int(st.get("n") or 0) >= min_n
        and float(mean) > 0
        and float(add) > 0
        and float(day_win) >= 0.55
        and float(mh) <= 0.50
    )


def _am_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    t = out["timestamp"].dt.time
    lo = pd.Timestamp(SIGNAL_START).time()
    hi = pd.Timestamp(SIGNAL_END).time()
    return out[(t >= lo) & (t < hi)].sort_values("timestamp").reset_index(drop=True)


def _px_at(ts: pd.Series, close: np.ndarray, asof: pd.Timestamp) -> float | None:
    if len(ts) == 0:
        return None
    i = int(ts.searchsorted(asof, side="right") - 1)
    if i < 0:
        return None
    px = float(close[i])
    return px if np.isfinite(px) and px > 0 else None


def _rule_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for clock in ("09:35", "09:40", "09:45"):
        for fo in (0.002, 0.003, 0.005):
            for peer in (0, 2, 3):
                cells.append(
                    {
                        "name": f"fo_{clock.replace(':', '')}_r{int(fo*1000):03d}_p{peer}",
                        "family": "from_open",
                        "clock": clock,
                        "fo_min": fo,
                        "peer_min": peer,
                        "universe": "mag7",
                    }
                )
    for slope in (3, 5):
        for ret in (0.002, 0.003):
            for peer in (2, 3):
                cells.append(
                    {
                        "name": f"launch_s{slope}_r{int(ret*1000):03d}_p{peer}",
                        "family": "launch",
                        "slope_sec": slope,
                        "abs_ret_min": ret,
                        "peer_min": peer,
                        "universe": "mag7",
                    }
                )
    for clock in ("09:40", "09:45"):
        for fo in (0.002, 0.003):
            cells.append(
                {
                    "name": f"qqq_fo_{clock.replace(':', '')}_r{int(fo*1000):03d}",
                    "family": "qqq_from_open",
                    "clock": clock,
                    "fo_min": fo,
                    "peer_min": 0,
                    "universe": "qqq",
                }
            )
    return cells


def _build_day_signals(
    date: str,
    stock_1s: Path,
    cells: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return cell_name -> list of signal dicts for one date."""
    out: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in cells}
    am_by: dict[str, pd.DataFrame] = {}
    for sym in MAG7 + ["QQQ"]:
        raw = load_stock_1s_day(stock_1s, sym, date)
        am = _am_slice(raw)
        if not am.empty:
            am_by[sym] = am
    if not am_by:
        return out

    # --- from_open clocks (Mag7 + QQQ) ---
    clocks = sorted(
        {
            c["clock"]
            for c in cells
            if c["family"] in ("from_open", "qqq_from_open") and "clock" in c
        }
    )
    fo_cache: dict[str, dict[str, tuple[str, float, float, pd.Timestamp]]] = {}
    # clock -> sym -> (dir, fo, spot, ts)
    for clock in clocks:
        asof = pd.Timestamp(f"{date} {clock}:00", tz=NY)
        fo_cache[clock] = {}
        for sym, am in am_by.items():
            ts = am["timestamp"]
            close = am["close"].to_numpy(dtype=float)
            open_px = float(close[0])
            if not np.isfinite(open_px) or open_px <= 0:
                continue
            px = _px_at(ts, close, asof)
            if px is None:
                continue
            fo = px / open_px - 1.0
            if not np.isfinite(fo) or abs(fo) < 1e-12:
                continue
            d = "UP" if fo > 0 else "DN"
            fo_cache[clock][sym] = (d, float(fo), float(px), asof)

    for cell in cells:
        if cell["family"] == "from_open":
            clock = cell["clock"]
            fo_min = float(cell["fo_min"])
            peer_min = int(cell["peer_min"])
            snap = fo_cache.get(clock) or {}
            for sym in MAG7:
                if sym not in snap:
                    continue
                d, fo, spot, asof = snap[sym]
                if abs(fo) < fo_min:
                    continue
                peer = sum(
                    1
                    for s2 in MAG7
                    if s2 != sym
                    and s2 in snap
                    and snap[s2][0] == d
                    and abs(snap[s2][1]) >= fo_min * 0.5
                )
                # include self in peer_align_min style (≥ peer_min same-dir names)
                peer_n = peer + 1
                if peer_n < peer_min:
                    continue
                out[cell["name"]].append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": d,
                        "ts": asof,
                        "spot": spot,
                        "fo": fo,
                        "peer_n": peer_n,
                    }
                )
        elif cell["family"] == "qqq_from_open":
            clock = cell["clock"]
            fo_min = float(cell["fo_min"])
            snap = fo_cache.get(clock) or {}
            if "QQQ" not in snap:
                continue
            d, fo, spot, asof = snap["QQQ"]
            if abs(fo) < fo_min:
                continue
            out[cell["name"]].append(
                {
                    "date": date,
                    "symbol": "QQQ",
                    "dir": d,
                    "ts": asof,
                    "spot": spot,
                    "fo": fo,
                    "peer_n": 1,
                }
            )

    # --- launch first-touch ---
    launch_cells = [c for c in cells if c["family"] == "launch"]
    slopes = sorted({int(c["slope_sec"]) for c in launch_cells})
    thrs = sorted({float(c["abs_ret_min"]) for c in launch_cells})
    # (slope, dir, thr) -> list of edge dicts
    edges_map: dict[tuple[int, str, float], list[dict[str, Any]]] = {}
    for slope in slopes:
        for sym in MAG7:
            am = am_by.get(sym)
            if am is None or am.empty:
                continue
            feat = attach_launch_slope_features(am, slope_sec=slope, mf_window_sec=None)
            if feat.empty:
                continue
            ts = feat["timestamp"]
            close = feat["close"].to_numpy(dtype=float)
            ret = feat["ret_k"].to_numpy(dtype=float)
            multi = launch_edges_multi(
                feat, abs_ret_mins=thrs, require_local_peak=True
            )
            for (d, thr), idxs in multi.items():
                key = (slope, d, float(thr))
                bucket = edges_map.setdefault(key, [])
                for i in idxs:
                    i = int(i)
                    bucket.append(
                        {
                            "symbol": sym,
                            "dir": d,
                            "ts": to_ny(ts.iloc[i]),
                            "spot": float(close[i]),
                            "ret_k": float(ret[i]),
                        }
                    )

    for cell in launch_cells:
        slope = int(cell["slope_sec"])
        ret_min = float(cell["abs_ret_min"])
        peer_min = int(cell["peer_min"])
        cands = sorted(
            edges_map.get((slope, "UP", ret_min), [])
            + edges_map.get((slope, "DN", ret_min), []),
            key=lambda x: (x["ts"], x["symbol"]),
        )
        seen: set[str] = set()
        for e in cands:
            sym = e["symbol"]
            if sym in seen:
                continue
            d = e["dir"]
            t0 = e["ts"]
            peer = 1 + sum(
                1
                for o in cands
                if o["symbol"] != sym
                and o["dir"] == d
                and abs((o["ts"] - t0).total_seconds()) <= 30
            )
            if peer < peer_min:
                continue
            seen.add(sym)
            out[cell["name"]].append(
                {
                    "date": date,
                    "symbol": sym,
                    "dir": d,
                    "ts": t0,
                    "spot": e["spot"],
                    "fo": float(e["ret_k"]),
                    "peer_n": peer,
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--tag", default="research_am_0930_1000_rule_trades_dual")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--tps", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--sls", default="0.10,0.15,0.25")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="portfolio seats per day (1=sparse; 2=denser AM sleeve)",
    )
    ap.add_argument(
        "--position-frac",
        type=float,
        default=0.10,
        help="size per seat (default 0.10; with concurrent=2 consider 0.10 or 0.05)",
    )
    ap.add_argument(
        "--families",
        default="from_open,launch,qqq_from_open",
        help="comma families to include",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    stock_1s = Path(paths["stock_1s_root"])
    trades_root = Path(args.trades_root)
    results_dir = Path(paths["results_dir"])
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)

    want_fam = {x.strip() for x in args.families.split(",") if x.strip()}
    cells = [c for c in _rule_cells() if c["family"] in want_fam]
    tps = [float(x) for x in args.tps.split(",") if x.strip()]
    sls = [float(x) for x in args.sls.split(",") if x.strip()]

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    otm_rungs = resolve_otm_rungs(prof, default=3)

    print(
        f"AM {SIGNAL_START}-{SIGNAL_END} cells={len(cells)} "
        f"tp={tps} sl={sls} trades={trades_root}",
        flush=True,
    )

    # Build signals per window
    win_sigs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    union_keys: dict[tuple[str, str, str, str], float] = {}
    for wname, w0, w1 in WINDOWS:
        dates = session_dates(w0, w1)
        by_cell: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in cells}
        print(f"[{wname}] building 1s signals days={len(dates)} …", flush=True)
        for i, date in enumerate(dates, 1):
            day = _build_day_signals(date, stock_1s, cells)
            for name, sigs in day.items():
                by_cell[name].extend(sigs)
                for s in sigs:
                    k = (str(s["date"]), str(s["symbol"]), str(s["dir"]), str(s["ts"]))
                    union_keys[k] = float(s["spot"])
            if i % 20 == 0 or i == len(dates):
                ntot = sum(len(v) for v in by_cell.values())
                print(f"  [{wname} {i}/{len(dates)}] sigs={ntot}", flush=True)
        win_sigs[wname] = by_cell
        print(
            f"[{wname}] total_sigs={sum(len(v) for v in by_cell.values())} "
            f"by_cell_top="
            + ", ".join(
                f"{n}:{len(by_cell[n])}"
                for n in sorted(by_cell, key=lambda x: -len(by_cell[x]))[:5]
            ),
            flush=True,
        )

    # Resolve option trade paths once
    print(f"resolving fills unique={len(union_keys)} …", flush=True)
    path_cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fills: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    n_miss = 0
    for i, ((date, sym, direction, ts_s), spot) in enumerate(sorted(union_keys.items())):
        if i % 300 == 0:
            print(f"  [resolve] {i}/{len(union_keys)} fills={len(fills)} miss={n_miss}", flush=True)
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
    print(f"resolved fills={len(fills)} miss={n_miss}; scoring…", flush=True)

    score_rows: list[dict[str, Any]] = []
    dual_pass_list: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in cells:
        for tp in tps:
            for sl in sls:
                win_stats: dict[str, dict[str, Any]] = {}
                win_raw: dict[str, list[dict[str, Any]]] = {}
                for wname, _, _ in WINDOWS:
                    sigs = win_sigs[wname].get(cell["name"]) or []
                    raw: list[dict[str, Any]] = []
                    for s in sigs:
                        k = (
                            str(s["date"]),
                            str(s["symbol"]),
                            str(s["dir"]),
                            str(s["ts"]),
                        )
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
                                "exit_ts": str(
                                    et + pd.Timedelta(seconds=sim["hold_sec"])
                                ),
                                "ticker": f["ticker"],
                                "dte": f["dte"],
                                "ret": sim["ret"],
                                "exit_reason": sim["reason"],
                                "hold_sec": sim["hold_sec"],
                                "window": wname,
                                "cell": cell["name"],
                            }
                        )
                    st = _port(
                        raw,
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                    )
                    win_stats[wname] = st
                    win_raw[wname] = raw

                both = all(
                    _ok(win_stats[w], min_n=int(args.min_n)) for w, _, _ in WINDOWS
                )
                row: dict[str, Any] = {
                    "cell": cell["name"],
                    "family": cell["family"],
                    "session": "AM_0930_1000",
                    "tp": tp,
                    "sl": sl,
                    "max_hold_sec": int(args.max_hold_sec),
                    "max_concurrent": int(args.max_concurrent),
                    "position_frac": float(args.position_frac),
                    "dual_pass": both,
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    key = f"{cell['name']}|tp{tp}|sl{sl}"
                    dual_pass_list.append(row)
                    trade_dump[key] = pd.DataFrame(
                        win_raw["jan_mar"] + win_raw["may_jul"]
                    )
                    print(
                        f"  *** DUAL PASS {key} "
                        f"JM add={row['jan_mar_add']:+.3f} "
                        f"MJ add={row['may_jul_add']:+.3f}",
                        flush=True,
                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass_list = sorted(
        dual_pass_list,
        key=lambda r: (
            float(r.get("may_jul_add") or 0) + float(r.get("jan_mar_add") or 0)
        ),
        reverse=True,
    )
    for i, p in enumerate(dual_pass_list[:15]):
        key = f"{p['cell']}|tp{p['tp']}|sl{p['sl']}"
        if key in trade_dump:
            trade_dump[key].to_csv(
                out / f"trades_dual{i:02d}_{p['cell']}_tp{p['tp']}_sl{p['sl']}.csv",
                index=False,
            )

    verdict = "PASS" if dual_pass_list else "REJECT"
    champ = dual_pass_list[0] if dual_pass_list else None
    summary = {
        "session": "AM_0930_1000",
        "signal_start": SIGNAL_START,
        "signal_end": SIGNAL_END,
        "complements": "research_baseline window_start=10:30",
        "pricing": str(trades_root),
        "slip": float(args.slip),
        "exit": "tp_sl_first_passage_trade_last",
        "max_hold_sec_safety": int(args.max_hold_sec),
        "max_concurrent": int(args.max_concurrent),
        "position_frac": float(args.position_frac),
        "stock_source": str(stock_1s),
        "windows": [list(w) for w in WINDOWS],
        "n_cells": len(cells),
        "n_unique_sigs": int(len(union_keys)),
        "n_resolved_fills": int(len(fills)),
        "n_miss": int(n_miss),
        "dual_pass_n": int(len(dual_pass_list)),
        "verdict": verdict,
        "champion": champ,
        "dual_pass": dual_pass_list[:30],
        "note": (
            "AM-only sleeve on trades book. Quote FillSpec still required before "
            "live; prior Mag7 launch AM was trades-PASS / quote-REJECT."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass_list[:30], indent=2, default=str), encoding="utf-8"
    )

    print(f"\n=== dual PASS ({len(dual_pass_list)}) verdict={verdict} ===", flush=True)
    if champ:
        print(json.dumps(champ, indent=2, default=str), flush=True)
    if len(score):
        cols = [
            c
            for c in [
                "cell",
                "family",
                "tp",
                "sl",
                "dual_pass",
                "jan_mar_n",
                "jan_mar_mean",
                "jan_mar_day_win",
                "jan_mar_add",
                "may_jul_n",
                "may_jul_mean",
                "may_jul_day_win",
                "may_jul_add",
            ]
            if c in score.columns
        ]
        top = score.sort_values(
            ["dual_pass", "may_jul_add", "jan_mar_add"],
            ascending=[False, False, False],
        )
        print(top[cols].head(20).to_string(index=False), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
