#!/usr/bin/env python3
"""New signal source: stock-1s **buyer impact** → ATM trade-last scalp.

Replaces FO/VWAP pocket densify. Uses rare causal impact gates from
``scan_buyer_impact_1s`` (volr / short-ret∩vol / impact_score), rising-edge
+ cooldown, Mag7 ATM open-lock, ``simulate_trade_tpsl``.

Objective: higher trade count than vd_soft keep, dual-window econ > 0,
prefer high win / mean option ret ~3–8%.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_impact_scalp_tpsl \\
    --tag research_am_impact_scalp_tpsl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

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
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of
from maga7.tools.scan_am_pocket_risk_optimize import _equity_stats
from maga7.tools.scan_buyer_impact_1s import _dir_from_ret, _impact_row
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker, _spot_at_arr

DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
BDAYS = 60
WINDOWS = ("may_jul09", "jul10_23")


GateFn = Callable[[dict[str, float]], bool]


def _gates() -> list[tuple[str, GateFn]]:
    def volr2(imp: dict[str, float]) -> bool:
        v = imp.get("volume_ratio_60", float("nan"))
        return bool(np.isfinite(v) and v >= 2.0)

    def volr15(imp: dict[str, float]) -> bool:
        v = imp.get("volume_ratio_60", float("nan"))
        return bool(np.isfinite(v) and v >= 1.5)

    def ret30_20(imp: dict[str, float]) -> bool:
        a = imp.get("abs_ret30", float("nan"))
        return bool(np.isfinite(a) and a >= 0.002)

    def ret30_volr(imp: dict[str, float]) -> bool:
        return ret30_20(imp) and volr15(imp)

    def ret15_volr(imp: dict[str, float]) -> bool:
        a = imp.get("abs_ret15", float("nan"))
        v = imp.get("volume_ratio_60", float("nan"))
        return bool(np.isfinite(a) and a >= 0.0015 and np.isfinite(v) and v >= 1.5)

    def impact_hi(imp: dict[str, float]) -> bool:
        s = imp.get("impact_score", float("nan"))
        return bool(np.isfinite(s) and s >= 3.5)

    def impact_med(imp: dict[str, float]) -> bool:
        s = imp.get("impact_score", float("nan"))
        return bool(np.isfinite(s) and s >= 2.5)

    def volz_ret(imp: dict[str, float]) -> bool:
        z = imp.get("vol_z", float("nan"))
        a = imp.get("abs_ret30", float("nan"))
        return bool(np.isfinite(z) and z >= 2.0 and np.isfinite(a) and a >= 0.0015)

    def burst(imp: dict[str, float]) -> bool:
        """Union of strongest rare priors."""
        return volr2(imp) or ret30_volr(imp) or impact_hi(imp)

    return [
        ("volr2", volr2),
        ("volr15", volr15),
        ("ret30_volr", ret30_volr),
        ("ret15_volr", ret15_volr),
        ("impact_hi", impact_hi),
        ("impact_med", impact_med),
        ("volz_ret", volz_ret),
        ("burst", burst),
    ]


def _exit_grid() -> list[dict[str, Any]]:
    out = []
    for tp in (0.05, 0.08, 0.10, 0.15):
        for sl in (0.08, 0.10, 0.15, 0.20):
            for h in (60, 90, 120, 180, 240):
                out.append({"name": f"tp{tp:g}_sl{sl:g}_h{h}", "tp": tp, "sl": sl, "max_hold": h})
    return out


def collect_events(
    *,
    dates: list[str],
    symbols: list[str],
    stock_1s: Path,
    lock: dict,
    otm: list,
    trades_root: Path,
    window_start: str,
    window_end: str,
    stride_sec: int,
    cooldown_sec: int,
    gate_name: str,
    gate_fn: GateFn,
) -> list[dict[str, Any]]:
    """Rising-edge impact events with per-symbol cooldown."""
    events: list[dict[str, Any]] = []
    path_cache: dict[tuple[str, str], dict] = {}

    def paths_for(date: str, sym: str):
        key = (date, sym)
        if key not in path_cache:
            tday = load_option_trades(trades_root, sym, date)
            path_cache[key] = (
                _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
            )
        return path_cache[key]

    for di, date in enumerate(dates):
        if di % 10 == 0:
            print(f"  [{gate_name}] day {date} ({di+1}/{len(dates)}) events={len(events)}", flush=True)
        w = _window_of(date)
        if w is None:
            continue
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            sarr = prepare_day_arrays(raw)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            tpaths = paths_for(date, sym)
            if not tpaths:
                continue
            ts_ns = sarr["ts_ns"]
            t0 = to_ny(pd.Timestamp(f"{date} {window_start}", tz=NY))
            t1 = to_ny(pd.Timestamp(f"{date} {window_end}", tz=NY))
            i0 = int(np.searchsorted(ts_ns, int(t0.value), side="left"))
            i1 = int(np.searchsorted(ts_ns, int(t1.value), side="right") - 1)
            if i1 <= i0:
                continue
            stride = max(1, int(stride_sec))
            prev_on = False
            last_fire_ns = -10**18
            for i in range(i0, i1 + 1, stride):
                t = pd.Timestamp(int(ts_ns[i]), tz="UTC").tz_convert(NY)
                feat = features_at(sarr, t)
                if feat is None:
                    prev_on = False
                    continue
                imp = _impact_row(feat)
                on = bool(gate_fn(imp))
                rising = on and not prev_on
                prev_on = on
                if not rising:
                    continue
                if int(ts_ns[i]) - last_fire_ns < int(cooldown_sec) * 1_000_000_000:
                    continue
                direction = _dir_from_ret(imp["ret15"]) or _dir_from_ret(imp["ret30"])
                if direction is None:
                    continue
                spot = float(feat.get("px") or np.nan)
                if not np.isfinite(spot):
                    sv = _spot_at_arr(sarr["ts_ns"], sarr["close"], t)
                    spot = float(sv) if sv is not None else float("nan")
                if not np.isfinite(spot):
                    continue
                ticker, dte, _ = resolve_open_lock_contract(
                    by_dte,
                    direction=direction,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=(0, 1, 2),
                    clear_otm_thresh=0.01,
                    ladder=True,
                    otm_rungs=otm,
                )
                if not ticker:
                    continue
                key = str(ticker).replace("O:", "")
                path = tpaths.get(key)
                if path is None:
                    continue
                last_fire_ns = int(ts_ns[i])
                events.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "dir": direction,
                        "entry_ts": t,
                        "ticker": ticker,
                        "dte": dte,
                        "window": w,
                        "gate": gate_name,
                        "impact_score": imp.get("impact_score"),
                        "volume_ratio_60": imp.get("volume_ratio_60"),
                        "abs_ret30": imp.get("abs_ret30"),
                        "pts": path[0],
                        "plast": path[1],
                    }
                )
    return events


def _score_book(
    events: list[dict[str, Any]],
    *,
    exit_cfg: dict[str, Any],
    slip: float,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> dict[str, Any]:
    win_raw: dict[str, list] = {w: [] for w in WINDOWS}
    for e in events:
        sim = simulate_trade_tpsl(
            e["pts"],
            e["plast"],
            e["entry_ts"],
            tp=float(exit_cfg["tp"]),
            sl=float(exit_cfg["sl"]),
            max_hold_sec=int(exit_cfg["max_hold"]),
            slip=slip,
        )
        if sim is None or not np.isfinite(sim.get("ret", np.nan)):
            continue
        hold = float(sim["hold_sec"])
        et = e["entry_ts"]
        win_raw[e["window"]].append(
            {
                "date": e["date"],
                "symbol": e["symbol"],
                "dir": e["dir"],
                "entry_ts": et,
                "exit_ts": et + pd.Timedelta(seconds=hold),
                "ticker": e["ticker"],
                "ret": float(sim["ret"]),
                "exit_reason": str(sim["reason"]),
                "hold_sec": hold,
            }
        )
    sized_all = []
    win_stats = {}
    for w in WINDOWS:
        by_d: dict[str, list] = {}
        for t in win_raw[w]:
            by_d.setdefault(str(t["date"]), []).append(t)
        sized = []
        for _, rs in sorted(by_d.items()):
            sized.extend(
                _portfolio_day(
                    sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                    position_frac=position_frac,
                    max_concurrent=max_concurrent,
                    cooldown_minutes=cooldown_minutes,
                )
            )
        ste = _equity_stats(pd.DataFrame(sized)) if sized else {"n": 0, "compound": 0.0}
        win_stats[w] = ste
        sized_all.extend(sized)
    if not sized_all:
        return {"n": 0, "econ_dual": False}
    rr = np.array([t["ret"] for t in sized_all], dtype=float)
    disc = float(win_stats["may_jul09"].get("compound") or 0)
    blind = float(win_stats["jul10_23"].get("compound") or 0)
    n_d = int(win_stats["may_jul09"].get("n") or 0)
    n_b = int(win_stats["jul10_23"].get("n") or 0)
    n = len(sized_all)
    return {
        "n": n,
        "tpd": n / float(BDAYS),
        "active_days": len({t["date"] for t in sized_all}),
        "trade_win": float((rr > 0).mean()),
        "mean_ret": float(rr.mean()),
        "med_ret": float(np.median(rr)),
        "disc_compound": disc,
        "blind_compound": blind,
        "disc_n": n_d,
        "blind_n": n_b,
        "disc_maxdd": float(win_stats["may_jul09"].get("maxdd") or 0),
        "econ_dual": bool(n_d >= 15 and n_b >= 5 and disc > 0 and blind > 0),
        "frac_tp": float(np.mean([t["exit_reason"] == "tp" for t in sized_all])),
        "hold_p50": float(np.median([t["hold_sec"] for t in sized_all])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_am_impact_scalp_tpsl")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:30")
    ap.add_argument("--stride-sec", type=int, default=10)
    ap.add_argument("--cooldowns", default="60,120,180")
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--port-cooldown-min", type=float, default=1.0)
    ap.add_argument(
        "--gates",
        default="volr2,ret30_volr,impact_hi,burst,ret15_volr,volz_ret",
    )
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    symbols = list(prof.get("symbols") or [])
    dates = [d for d in session_dates(args.start_date, args.end_date) if args.start_date <= d <= args.end_date]
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]

    gmap = dict(_gates())
    want_g = [x.strip() for x in str(args.gates).split(",") if x.strip() and x.strip() in gmap]
    cds = [int(x) for x in str(args.cooldowns).split(",") if x.strip()]
    exits = _exit_grid()
    # trim exits for speed: focus scalp + a few wider
    exits = [
        e
        for e in exits
        if e["tp"] in (0.05, 0.08, 0.10, 0.15)
        and e["sl"] in (0.08, 0.10, 0.15)
        and e["max_hold"] in (60, 90, 120, 180)
    ]
    print(
        f"impact scalp days={len(dates)} syms={len(symbols)} gates={want_g} "
        f"cd={cds} exits={len(exits)}",
        flush=True,
    )

    # Collect events once per (gate, cooldown)
    event_books: dict[tuple[str, int], list[dict]] = {}
    for gname in want_g:
        for cd in cds:
            ev = collect_events(
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                lock=lock,
                otm=otm,
                trades_root=trades_root,
                window_start=args.window_start,
                window_end=args.window_end,
                stride_sec=int(args.stride_sec),
                cooldown_sec=cd,
                gate_name=gname,
                gate_fn=gmap[gname],
            )
            event_books[(gname, cd)] = ev
            print(f"events {gname} cd{cd}: n={len(ev)} (~{len(ev)/BDAYS:.1f}/bday)", flush=True)
            # persist lightweight event meta
            meta = [
                {
                    k: v
                    for k, v in e.items()
                    if k not in ("pts", "plast")
                }
                for e in ev
            ]
            pd.DataFrame(meta).to_csv(out / f"events_{gname}_cd{cd}.csv", index=False)

    rows = []
    for (gname, cd), ev in event_books.items():
        if len(ev) < 20:
            continue
        for ex in exits:
            st = _score_book(
                ev,
                exit_cfg=ex,
                slip=float(args.slip),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.port_cooldown_min),
            )
            if st.get("n", 0) == 0:
                continue
            row = {
                "signal": "buyer_impact",
                "gate": gname,
                "event_cd": cd,
                "exit": ex["name"],
                "n_events": len(ev),
                **st,
            }
            rows.append(row)
        # progress best so far for this book
        sub = [r for r in rows if r["gate"] == gname and r["event_cd"] == cd]
        if sub:
            best = max(sub, key=lambda r: (r["econ_dual"], r.get("mean_ret") or -9, r.get("trade_win") or 0))
            print(
                f"  scored {gname}/cd{cd}: best_mean={best['mean_ret']:+.3f} "
                f"win={best['trade_win']:.2f} tpd={best['tpd']:.1f} econ={best['econ_dual']} "
                f"exit={best['exit']}",
                flush=True,
            )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    ok = sb[sb.econ_dual == True].copy()  # noqa: E712
    ok = ok.sort_values(
        ["mean_ret", "trade_win", "tpd"],
        ascending=[False, False, False],
    )
    ok.to_csv(out / "ranked.csv", index=False)

    # also soft: positive both windows even if n small
    soft = sb[(sb.disc_compound > 0) & (sb.blind_compound > 0)].copy()
    soft = soft.sort_values(["mean_ret", "tpd"], ascending=[False, False])

    dense = ok[(ok.tpd >= 5) & (ok.trade_win >= 0.55) & (ok.mean_ret >= 0.03)]
    promote = "NONE"
    best = ok.iloc[0].to_dict() if len(ok) else (soft.iloc[0].to_dict() if len(soft) else None)
    if len(dense):
        promote = f"IMPACT_{dense.iloc[0]['gate']}_cd{dense.iloc[0]['event_cd']}__{dense.iloc[0]['exit']}"
    elif len(ok):
        promote = f"IMPACT_ECON_{ok.iloc[0]['gate']}_cd{ok.iloc[0]['event_cd']}__{ok.iloc[0]['exit']}"
    elif len(soft):
        promote = f"IMPACT_SOFT_{soft.iloc[0]['gate']}_cd{soft.iloc[0]['event_cd']}__{soft.iloc[0]['exit']}"

    summary = {
        "protocol": "am_impact_scalp_tpsl",
        "signal_source": "buyer_impact_1s_rare_gates",
        "replaces": "FO/VWAP foresight pocket densify",
        "portfolio": {
            "position_frac": float(args.position_frac),
            "max_concurrent": int(args.max_concurrent),
            "cooldown_minutes": float(args.port_cooldown_min),
        },
        "n_combos": int(len(sb)),
        "n_econ_dual": int(len(ok)),
        "n_soft_pos": int(len(soft)),
        "n_dense_target": int(len(dense)),
        "promote": promote,
        "best": best,
        "top15_econ": ok.head(15).to_dict(orient="records"),
        "top10_soft": soft.head(10).to_dict(orient="records"),
        "event_counts": {
            f"{g}_cd{cd}": len(ev) for (g, cd), ev in event_books.items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    cols = [
        "gate", "event_cd", "exit", "tpd", "trade_win", "mean_ret",
        "disc_compound", "blind_compound", "n", "econ_dual", "frac_tp", "hold_p50",
    ]
    print("\n=== ECON DUAL TOP ===", flush=True)
    print(ok[cols].head(15).to_string(index=False) if len(ok) else "(none)", flush=True)
    print("\n=== SOFT POS TOP ===", flush=True)
    print(soft[cols].head(10).to_string(index=False) if len(soft) else "(none)", flush=True)
    print(json.dumps({"promote": promote, "n_econ": int(len(ok)), "n_soft": int(len(soft))}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
