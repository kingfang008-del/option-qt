#!/usr/bin/env python3
"""Quote FillSpec dual for AM pulse sleeve (trades PASS champions first).

Independent sleeve — not Mag7 Rule-A. Signal window 09:30–10:25.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pulse_quote_dual \\
    --tag research_am_pulse_quote_dual \\
    --champions-json /mnt/s990/data/maga7/results/research_am_pulse_trades_dual/champions.json
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

from maga7.common.am_pulse_scout import parse_am_pulse_scout, scan_day
from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import entry_quote_row, simulate_quote_tpsl
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _prep_path, _stats
from maga7.tools.scan_session_horizon_foresight import _spot_at_arr, _stock_arrays

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
SESSION = "AM_0930_1025"
SIGNAL_END = "10:25"

# Fallback if no champions.json (FO open_cont-like + impulse-like)
DEFAULT_CELLS = (
    {"name": "pulse_FO_t0.01_tp0.1_sl0.25", "arm": "FO", "thr": 0.01, "lookback_bars": 2, "tp": 0.10, "sl": 0.25},
    {"name": "pulse_FO_t0.01_tp0.2_sl0.2", "arm": "FO", "thr": 0.01, "lookback_bars": 2, "tp": 0.20, "sl": 0.20},
    {"name": "pulse_LB_t0.008_lb2_tp0.2_sl0.2", "arm": "LB", "thr": 0.008, "lookback_bars": 2, "tp": 0.20, "sl": 0.20},
)


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _spot_from_1m(day: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if day is None or day.empty:
        return None
    t = to_ny(ts)
    sub = day[pd.to_datetime(day["timestamp"]) <= t]
    if sub.empty:
        return None
    px = float(sub.iloc[-1]["close"])
    return px if px > 0 else None


def _load_cells(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return [dict(c) for c in DEFAULT_CELLS]
    p = Path(path)
    if not p.exists():
        print(f"champions missing {p}; using DEFAULT_CELLS", flush=True)
        return [dict(c) for c in DEFAULT_CELLS]
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not raw:
        return [dict(c) for c in DEFAULT_CELLS]
    return list(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_pulse_quote_dual")
    ap.add_argument("--champions-json", default="")
    ap.add_argument("--dirs", default="DN")
    ap.add_argument("--max-spreads", default="0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    args = ap.parse_args(argv)

    dirs = {x.strip().upper() for x in args.dirs.split(",") if x.strip()}
    cells = _load_cells(args.champions_json or None)
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_root = Path(paths["stock_root"])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in WINDOWS)
    end_all = max(w[2] for w in WINDOWS)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    months = month_list(start_all, end_all)
    print(
        f"am_pulse QUOTE dual {start_all}..{end_all} cells={len(cells)} "
        f"sp={spreads} lag={lags} dirs={sorted(dirs)}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_stock_month_files(stock_root, sym, months)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf

    # Unique (arm, thr, lookback) probes
    probes = {(c["arm"], float(c["thr"]), int(c.get("lookback_bars", 2))) for c in cells}

    arms: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) arms={len(arms)}", flush=True)
        for sym in symbols:
            sdf = stock_by_sym.get(sym)
            if sdf is None:
                continue
            day1m = sdf[sdf["date"].astype(str) == date]
            if day1m.empty:
                continue
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            qday = _prep_path(load_quotes(quote_root, sym, date))
            if qday is None or qday.empty:
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            ts_ns = px = None
            if day1s is not None and not day1s.empty:
                ts_ns, px = _stock_arrays(day1s)

            for arm_name, thr, lb_bars in sorted(probes):
                if arm_name == "FO":
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": "09:30",
                            "window_end": SIGNAL_END,
                            "min_fav_from_open": thr,
                            "lookback_bars": lb_bars,
                            "min_lookback_ret": 0.99,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                else:
                    cfg = parse_am_pulse_scout(
                        {
                            "enabled": True,
                            "window_start": "09:30",
                            "window_end": SIGNAL_END,
                            "min_fav_from_open": 0.99,
                            "lookback_bars": lb_bars,
                            "min_lookback_ret": thr,
                            "dirs": sorted(dirs),
                            "max_alerts_per_symbol": 1,
                        }
                    )
                for a in scan_day(day1m, date=date, symbol=sym, cfg=cfg):
                    if a.arm != arm_name or a.dir not in dirs:
                        continue
                    arm_ts = to_ny(pd.Timestamp(a.ts))
                    spot = None
                    if ts_ns is not None and px is not None:
                        spot = _spot_at_arr(ts_ns, px, arm_ts)
                    if spot is None:
                        spot = _spot_from_1m(day1m, arm_ts)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction=a.dir,
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
                        arm_ts,
                        max_lag_sec=max(lags),
                        max_spread_pct=max(spreads),
                        min_mid=float(args.min_mid),
                    )
                    if probe is None:
                        continue
                    arms.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "dir": a.dir,
                            "arm": arm_name,
                            "thr": float(thr),
                            "lookback_bars": int(lb_bars),
                            "session": SESSION,
                            "arm_ts": arm_ts,
                            "ticker": ticker,
                            "dte": dte,
                            "path": path,
                            "probe_spread": float(probe["spread_pct"]),
                            "probe_lag": float(probe["lag_sec"]),
                        }
                    )

    print(f"arms_resolvable={len(arms)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    dual_pass: list[dict[str, Any]] = []
    trade_dump: dict[str, pd.DataFrame] = {}

    for cell in cells:
        arm_n, thr = str(cell["arm"]), float(cell["thr"])
        lb = int(cell.get("lookback_bars", 2))
        tp, sl = float(cell["tp"]), float(cell["sl"])
        for max_sp in spreads:
            for max_lag in lags:
                name = f"{cell['name']}_sp{max_sp}_lag{max_lag}"
                win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
                n_sig = n_block = n_fill = 0
                for arm in arms:
                    if str(arm["arm"]) != arm_n or float(arm["thr"]) != thr:
                        continue
                    if int(arm["lookback_bars"]) != lb:
                        continue
                    wname = _window_of(str(arm["date"]))
                    if wname is None:
                        continue
                    n_sig += 1
                    if float(arm["probe_spread"]) > max_sp or float(arm["probe_lag"]) > max_lag:
                        n_block += 1
                        continue
                    sim = simulate_quote_tpsl(
                        arm["path"],
                        arm["arm_ts"],
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
                            "dir": arm["dir"],
                            "session": arm["session"],
                            "entry_ts": str(sim["entry_ts"]),
                            "exit_ts": str(sim["exit_ts"]),
                            "ticker": arm["ticker"],
                            "ret": sim["ret"],
                            "exit_reason": sim["reason"],
                            "hold_sec": sim["hold_sec"],
                            "cell": name,
                            "event_source": "am_pulse_sleeve",
                            "window": wname,
                        }
                    )

                win_stats: dict[str, Any] = {}
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
                    st = _stats(sized)
                    # quote gate: frac_max_hold ≤ 0.50
                    if st.get("frac_max_hold") is not None and float(st["frac_max_hold"]) > 0.50:
                        st["quote_hold_fail"] = True
                    win_stats[wname] = st
                    sized_all.extend(sized)

                both = True
                for wname, _, _ in WINDOWS:
                    mn = int(args.min_n)
                    if wname == "jul10_23":
                        mn = min(mn, 6)
                    st = win_stats[wname]
                    if st.get("quote_hold_fail"):
                        both = False
                        break
                    if not _ok(st, min_n=mn, min_day_win=float(args.min_day_win)):
                        both = False
                        break

                row = {
                    "name": name,
                    "base": cell["name"],
                    "arm": arm_n,
                    "thr": thr,
                    "lookback_bars": lb,
                    "tp": tp,
                    "sl": sl,
                    "max_spread_pct": max_sp,
                    "max_lag_sec": max_lag,
                    "dual_pass": both,
                    "n_sig": n_sig,
                    "n_block": n_block,
                    "n_fill": n_fill,
                }
                for wname, _, _ in WINDOWS:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                if both:
                    dual_pass.append(row)
                    trade_dump[name] = pd.DataFrame(sized_all)
                    print(
                        f"  *** QUOTE DUAL PASS {name} "
                        f"MJ09 n={row.get('may_jul09_n')} mean={row.get('may_jul09_mean')} "
                        f"J10 n={row.get('jul10_23_n')} mean={row.get('jul10_23_mean')}",
                        flush=True,
                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    dual_pass = sorted(
        dual_pass,
        key=lambda r: float(r.get("may_jul09_add") or 0) + float(r.get("jul10_23_add") or 0),
        reverse=True,
    )
    for i, p in enumerate(dual_pass[:10]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)

    summary = {
        "expert_kind": "am_pulse_sleeve",
        "pricing": "quote_FillSpec",
        "session": SESSION,
        "dirs": sorted(dirs),
        "n_arms": int(len(arms)),
        "n_rows": int(len(score_rows)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "QUOTE_PASS" if dual_pass else "QUOTE_REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "isolation": "independent sleeve; not Mag7 Rule-A",
        "windows": [list(w) for w in WINDOWS],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:40], indent=2, default=str), encoding="utf-8"
    )
    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        c = dual_pass[0]
        print(
            f"champion {c['name']}: "
            f"MJ09 n={c.get('may_jul09_n')} mean={c.get('may_jul09_mean')} | "
            f"J10 n={c.get('jul10_23_n')} mean={c.get('jul10_23_mean')}",
            flush=True,
        )
    elif not score.empty:
        score["_sum"] = score["may_jul09_add"].fillna(0) + score["jul10_23_add"].fillna(0)
        cols = [
            c
            for c in [
                "name",
                "may_jul09_n",
                "may_jul09_mean",
                "may_jul09_day_win",
                "jul10_23_n",
                "jul10_23_mean",
                "jul10_23_day_win",
                "n_fill",
            ]
            if c in score.columns
        ]
        print(score.sort_values("_sum", ascending=False)[cols].head(12).to_string(index=False))
    print(f"wrote {out}", flush=True)
    return 0 if dual_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
