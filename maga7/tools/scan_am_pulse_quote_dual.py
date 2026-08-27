#!/usr/bin/env python3
"""Quote FillSpec dual for AM pulse sleeve (trades PASS champions first).

Independent sleeve — not Mag7 Rule-A. Signal window is configurable.

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

from maga7.common.am_pulse_scout import (
    am_pulse_decision_ts,
    load_am_pulse_lane_cfg,
    parse_am_pulse_scout,
    scan_day,
)
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
from maga7.common.stock_1s import load_symbol_1s_bars, session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _ok, _prep_path, _stats
from maga7.tools.scan_session_horizon_foresight import _spot_at_arr, _stock_arrays

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)
# Fallback if no champions.json (FO open_cont-like + impulse-like)
DEFAULT_CELLS = (
    {"name": "pulse_FO_t0.01_tp0.1_sl0.25", "arm": "FO", "thr": 0.01, "lookback_bars": 2, "tp": 0.10, "sl": 0.25},
    {"name": "pulse_FO_t0.01_tp0.2_sl0.2", "arm": "FO", "thr": 0.01, "lookback_bars": 2, "tp": 0.20, "sl": 0.20},
    {"name": "pulse_LB_t0.008_lb2_tp0.2_sl0.2", "arm": "LB", "thr": 0.008, "lookback_bars": 2, "tp": 0.20, "sl": 0.20},
)


def _parse_windows(spec: str | None) -> tuple[tuple[str, str, str], ...]:
    """Parse ``name:start:end,...`` or fall back to DEFAULT_WINDOWS."""
    if not spec or not str(spec).strip():
        return DEFAULT_WINDOWS
    out: list[tuple[str, str, str]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise SystemExit(f"bad --eval-windows chunk {part!r}; want name:YYYY-MM-DD:YYYY-MM-DD")
        out.append((bits[0].strip(), bits[1].strip(), bits[2].strip()))
    if not out:
        return DEFAULT_WINDOWS
    return tuple(out)


def _window_of(date: str, windows: tuple[tuple[str, str, str], ...]) -> str | None:
    for name, a, b in windows:
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
    ap.add_argument("--lane", choices=("am_pulse", "am_pulse_extension"), default="am_pulse")
    ap.add_argument("--tag", default="research_am_pulse_quote_dual")
    ap.add_argument("--champions-json", default="")
    ap.add_argument("--window-start", default="", help="Empty = profile lane value")
    ap.add_argument("--window-end", default="", help="Empty = profile lane value")
    ap.add_argument("--flatten-before", default="", help="Empty = profile lane value")
    ap.add_argument("--session-tag", default="", help="Empty = derived from profile lane")
    ap.add_argument("--dirs", default="", help="Empty = profile lane directions")
    ap.add_argument("--allowed-dte", default="", help="Empty = profile lane/lock values")
    ap.add_argument("--max-fav-from-open", type=float, default=None)
    ap.add_argument("--max-spreads", default="0.10,0.15")
    ap.add_argument("--max-lags", default="2,3")
    ap.add_argument("--min-mid", type=float, default=None, help="Empty = profile lane value")
    ap.add_argument("--max-hold-sec", type=int, default=0, help="0 = profile lane value")
    ap.add_argument(
        "--bar-delay-sec",
        type=int,
        default=60,
        help="decision_ts = feature_ts + delay (left-labeled 1m availability)",
    )
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--min-day-win", type=float, default=0.55)
    ap.add_argument(
        "--eval-windows",
        default="",
        help="Dual/eval calendar splits name:start:end,... (default may_jul09+jul10_23)",
    )
    ap.add_argument(
        "--open-locked-map",
        default="",
        help="Override paths.open_locked_map (e.g. 2025h2 lock parquet)",
    )
    ap.add_argument(
        "--stock-from-1s",
        action="store_true",
        help="Build 1m bars from stock_1s_root (needed when spnq_train lacks months, e.g. 2025H2)",
    )
    ap.add_argument(
        "--write-all-trades",
        action="store_true",
        help="Also dump trades CSV for cells that fail dual_pass",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    lane_cfg = load_am_pulse_lane_cfg(prof, args.lane)
    windows = _parse_windows(args.eval_windows or None)
    window_start = str(args.window_start or lane_cfg.get("window_start") or "09:30")
    window_end = str(args.window_end or lane_cfg.get("window_end") or "10:30")
    flatten_before = str(
        args.flatten_before or lane_cfg.get("flatten_before") or ""
    ).strip()
    session = str(
        args.session_tag
        or ("AM_EXT_1030_1130" if args.lane == "am_pulse_extension" else "AM_0930_1030")
    )
    dirs_spec = args.dirs or ",".join(lane_cfg.get("dirs") or ["DN", "UP"])
    dirs = {x.strip().upper() for x in dirs_spec.split(",") if x.strip()}
    max_fo = (
        float(args.max_fav_from_open)
        if args.max_fav_from_open is not None
        else float(lane_cfg.get("max_fav_from_open", 0.0) or 0.0)
    )
    min_mid = (
        float(args.min_mid)
        if args.min_mid is not None
        else float(lane_cfg.get("min_mid", 0.05) or 0.05)
    )
    max_hold_sec = (
        int(args.max_hold_sec)
        if int(args.max_hold_sec) > 0
        else int(lane_cfg.get("max_hold_sec", 900) or 900)
    )
    bar_delay_sec = max(0, int(args.bar_delay_sec))
    prefer_dte = int(lane_cfg.get("prefer_dte", 0) or 0)
    allowed_raw: Any = args.allowed_dte or lane_cfg.get("allowed_dte")
    if not allowed_raw:
        allowed_raw = (prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2]
    if isinstance(allowed_raw, str):
        allowed_dte = [int(x.strip()) for x in allowed_raw.split(",") if x.strip()]
    else:
        allowed_dte = [int(x) for x in allowed_raw]
    cells = _load_cells(args.champions_json or None)
    spreads = [float(x) for x in args.max_spreads.split(",") if x.strip()]
    lags = [float(x) for x in args.max_lags.split(",") if x.strip()]
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_root = Path(paths["stock_root"])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    quote_root = Path(paths["quote_1s_root"])
    lock_path = Path(args.open_locked_map).expanduser() if args.open_locked_map else Path(paths["open_locked_map"]).expanduser()
    lock = load_multidte_lock_index(lock_path)
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    start_all = min(w[1] for w in windows)
    end_all = max(w[2] for w in windows)
    dates = [d for d in session_dates(start_all, end_all) if start_all <= d <= end_all]
    months = month_list(start_all, end_all)
    print(
        f"am_pulse QUOTE dual {window_start}..{window_end} "
        f"{start_all}..{end_all} cells={len(cells)} "
        f"sp={spreads} lag={lags} dirs={sorted(dirs)} lock={lock_path.name}",
        flush=True,
    )

    stock_by_sym: dict[str, pd.DataFrame] = {}
    if bool(args.stock_from_1s):
        print(f"stock source=1s→1m root={stock_1s} days={len(dates)}", flush=True)
        for sym in symbols:
            sdf = load_symbol_1s_bars(stock_1s, sym, dates, bar_seconds=60)
            if sdf is not None and not sdf.empty:
                stock_by_sym[sym] = sdf
                print(f"  {sym}: bars={len(sdf)} days={sdf['date'].nunique()}", flush=True)
    else:
        for sym in symbols:
            sdf = load_stock_month_files(stock_root, sym, months)
            if sdf is not None and not sdf.empty:
                stock_by_sym[sym] = sdf
    if not stock_by_sym:
        print("WARNING: no stock frames loaded — arms will be 0", flush=True)

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
                            "window_start": window_start,
                            "window_end": window_end,
                            "min_fav_from_open": thr,
                            "max_fav_from_open": max_fo,
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
                            "window_start": window_start,
                            "window_end": window_end,
                            "min_fav_from_open": 0.99,
                            "max_fav_from_open": max_fo,
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
                    decision_ts = am_pulse_decision_ts(
                        arm_ts, delay_seconds=bar_delay_sec
                    )
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
                        prefer_dte=prefer_dte,
                        allowed_dte=allowed_dte,
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
                        decision_ts,
                        max_lag_sec=max(lags),
                        max_spread_pct=max(spreads),
                        min_mid=min_mid,
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
                            "session": session,
                            "arm_ts": arm_ts,
                            "decision_ts": decision_ts,
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
                win_raw: dict[str, list] = {w[0]: [] for w in windows}
                n_sig = n_block = n_fill = 0
                for arm in arms:
                    if str(arm["arm"]) != arm_n or float(arm["thr"]) != thr:
                        continue
                    if int(arm["lookback_bars"]) != lb:
                        continue
                    wname = _window_of(str(arm["date"]), windows)
                    if wname is None:
                        continue
                    n_sig += 1
                    if float(arm["probe_spread"]) > max_sp or float(arm["probe_lag"]) > max_lag:
                        n_block += 1
                        continue
                    hold_sec = max_hold_sec
                    if flatten_before:
                        flat_ts = pd.Timestamp(
                            f"{arm['date']} {flatten_before}", tz=NY
                        )
                        hold_sec = min(
                            hold_sec,
                            max(
                                1,
                                int(
                                    (
                                        flat_ts - to_ny(arm["decision_ts"])
                                    ).total_seconds()
                                ),
                            ),
                        )
                    sim = simulate_quote_tpsl(
                        arm["path"],
                        arm["decision_ts"],
                        tp=tp,
                        sl=sl,
                        max_hold_sec=hold_sec,
                        fill=fill,
                        max_lag_sec=max_lag,
                        max_spread_pct=max_sp,
                        min_mid=min_mid,
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
                            "event_source": (
                                "am_pulse_extension_sleeve"
                                if args.lane == "am_pulse_extension"
                                else "am_pulse_sleeve"
                            ),
                            "window": wname,
                        }
                    )

                win_stats: dict[str, Any] = {}
                sized_all: list[dict] = []
                for wname, _, _ in windows:
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
                for wi, (wname, _, _) in enumerate(windows):
                    mn = int(args.min_n)
                    # Second split may be short (legacy jul10_23 used min_n=6).
                    if wi == 1:
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
                for wname, _, _ in windows:
                    for k, v in win_stats[wname].items():
                        row[f"{wname}_{k}"] = v
                score_rows.append(row)
                trade_dump[name] = pd.DataFrame(sized_all)
                if both:
                    dual_pass.append(row)
                    w0, w1 = windows[0][0], windows[1][0] if len(windows) > 1 else windows[0][0]
                    print(
                        f"  *** QUOTE DUAL PASS {name} "
                        f"{w0} n={row.get(f'{w0}_n')} mean={row.get(f'{w0}_mean')} "
                        f"{w1} n={row.get(f'{w1}_n')} mean={row.get(f'{w1}_mean')}",
                        flush=True,
                    )

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    w_names = [w[0] for w in windows]

    def _add_sum(r: dict[str, Any]) -> float:
        return float(sum(float(r.get(f"{w}_add") or 0.0) for w in w_names))

    dual_pass = sorted(dual_pass, key=_add_sum, reverse=True)
    for i, p in enumerate(dual_pass[:10]):
        name = p["name"]
        if name in trade_dump and len(trade_dump[name]):
            trade_dump[name].to_csv(out / f"trades_dual{i:02d}_{name}.csv", index=False)
    if bool(args.write_all_trades):
        for name, tdf in trade_dump.items():
            if tdf is None or len(tdf) == 0:
                continue
            safe = str(name).replace("/", "_")
            tdf.to_csv(out / f"trades_all_{safe}.csv", index=False)

    summary = {
        "expert_kind": "am_pulse_sleeve",
        "pricing": "quote_FillSpec",
        "session": session,
        "window": [window_start, window_end],
        "dirs": sorted(dirs),
        "bar_delay_sec": int(bar_delay_sec),
        "entry_anchor": "decision_ts=feature_ts+bar_delay_sec",
        "n_arms": int(len(arms)),
        "n_rows": int(len(score_rows)),
        "dual_pass_n": int(len(dual_pass)),
        "verdict": "QUOTE_PASS" if dual_pass else "QUOTE_REJECT",
        "champion": dual_pass[0] if dual_pass else None,
        "isolation": "independent sleeve; not Mag7 Rule-A",
        "windows": [list(w) for w in windows],
        "open_locked_map": str(lock_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "dual_pass.json").write_text(
        json.dumps(dual_pass[:40], indent=2, default=str), encoding="utf-8"
    )
    print("\n=== verdict", summary["verdict"], "dual_pass_n=", len(dual_pass), flush=True)
    if dual_pass:
        c = dual_pass[0]
        bits = [
            f"{w} n={c.get(f'{w}_n')} mean={c.get(f'{w}_mean')}" for w in w_names
        ]
        print(f"champion {c['name']}: " + " | ".join(bits), flush=True)
    elif not score.empty:
        score["_sum"] = 0.0
        for w in w_names:
            col = f"{w}_add"
            if col in score.columns:
                score["_sum"] = score["_sum"] + score[col].fillna(0)
        cols = ["name", "n_fill"]
        for w in w_names:
            for suf in ("n", "mean", "day_win"):
                c = f"{w}_{suf}"
                if c in score.columns:
                    cols.append(c)
        print(score.sort_values("_sum", ascending=False)[cols].head(12).to_string(index=False))
    print(f"wrote {out}", flush=True)
    return 0 if dual_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
