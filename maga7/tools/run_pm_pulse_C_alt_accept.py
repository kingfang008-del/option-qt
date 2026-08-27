#!/usr/bin/env python3
"""PM window C alternative probes (not day-open FO).

After FO@0.8% 12:30+ failed quote accept, try expressions that do not
rely on stale morning extension:

  LB2 / LB5 / LB10  — AmPulseScout lookback impulse (window-local)
  MOM5 / MOM15     — first |ret|≥thr over N minutes inside window (stride 5m)

Primary clock: **14:00–15:30** (after CORE 10:30–14:00).
Also scan 12:30–14:00 as overlap research-only.

Contract: **1DTE ATM** only. Exit: quote TP15/SL20 FillSpec 0.75 / lag5 / sp15.
Portfolio: c2 @ 20%, cooldown 10m.

Example:
  PYTHONPATH=. python -m maga7.tools.run_pm_pulse_C_alt_accept \\
    --tag research_pm_pulse_C_alt_20260728
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

from maga7.common.am_pulse_scout import AmPulseScoutConfig, scan_day
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import simulate_quote_tpsl
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import load_stock_month_files
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
TP = 0.15
SL = 0.20
MAX_HOLD = 900
POS = 0.20
MAX_CONCURRENT = 2
COOLDOWN_MIN = 10


def _hhmm_min(hhmm: str) -> int:
    p = str(hhmm).split(":")
    return int(p[0]) * 60 + int(p[1])


def _in_win(ts: pd.Timestamp, ws: str, we: str) -> bool:
    t = to_ny(ts)
    hm = t.hour * 60 + t.minute
    return _hhmm_min(ws) <= hm < _hhmm_min(we)


def _stats_book(tr: pd.DataFrame) -> dict[str, Any]:
    if tr is None or tr.empty:
        return {
            "n": 0,
            "win": None,
            "add": 0.0,
            "mean": None,
            "mult": 1.0,
            "maxdd": 0.0,
            "day_win": None,
            "n_days": 0,
            "n_up": 0,
            "n_dn": 0,
        }
    d = tr.groupby("date")["pnl_frac"].sum().sort_index()
    eq = (1.0 + d).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    up = tr[tr["dir"] == "UP"]
    dn = tr[tr["dir"] == "DN"]
    return {
        "n": int(len(tr)),
        "win": float((tr["ret"] > 0).mean()),
        "add": float(d.sum()),
        "mean": float(tr["ret"].mean()),
        "mult": float(eq.iloc[-1]),
        "maxdd": float(dd.min()) if len(dd) else 0.0,
        "day_win": float((d > 0).mean()) if len(d) else None,
        "n_days": int(len(d)),
        "n_up": int(len(up)),
        "n_dn": int(len(dn)),
    }


def _verdict(may: dict[str, Any], feb: dict[str, Any]) -> str:
    if not may["n"] or may["mean"] is None:
        return "FAIL"
    if may["mean"] <= 0:
        return "FAIL"
    if feb["n"] >= 8 and feb["mean"] is not None and feb["mean"] <= 0:
        return "FAIL"
    if may["day_win"] is None or may["day_win"] < 0.55:
        return "FAIL"
    if may["n"] < 15:
        return "THIN"
    if may["maxdd"] < -0.25:
        return "FAIL"
    if may["mean"] >= 0.08 and (feb["mean"] is None or feb["mean"] >= 0):
        return "PASS"
    return "WEAK"


def _mom_alerts(
    day1m: pd.DataFrame,
    *,
    date: str,
    symbol: str,
    window_start: str,
    window_end: str,
    lookback_bars: int,
    thr: float,
) -> list[dict[str, Any]]:
    """First same-day MOM hit per symbol: |close/close[-lb]-1| ≥ thr inside window."""
    if day1m is None or day1m.empty:
        return []
    day = day1m.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    closes = day["close"].astype(float).to_numpy()
    out: list[dict[str, Any]] = []
    lb = int(lookback_bars)
    # Stride every bar; take first hit only.
    for i in range(lb, len(day)):
        t = ts.iloc[i]
        if not _in_win(t, window_start, window_end):
            continue
        p0 = float(closes[i - lb])
        px = float(closes[i])
        if p0 <= 0 or px <= 0:
            continue
        ret = px / p0 - 1.0
        if abs(ret) + 1e-12 < float(thr):
            continue
        d = "UP" if ret >= 0 else "DN"
        out.append(
            {
                "date": date,
                "symbol": symbol,
                "dir": d,
                "ts": t,
                "px": px,
                "arm": f"MOM{lb}",
                "lookback_ret": float(ret),
                "fav_from_open": np.nan,
            }
        )
        break
    return out


def _lb_alerts(
    day1m: pd.DataFrame,
    *,
    date: str,
    symbol: str,
    window_start: str,
    window_end: str,
    lookback_bars: int,
    thr: float,
) -> list[dict[str, Any]]:
    cfg = AmPulseScoutConfig(
        enabled=True,
        window_start=window_start,
        window_end=window_end,
        min_fav_from_open=0.99,  # FO off
        lookback_bars=int(lookback_bars),
        min_lookback_ret=float(thr),
        dirs=("DN", "UP"),
        max_alerts_per_symbol=1,
        rth_open_only=True,
    )
    out: list[dict[str, Any]] = []
    for a in scan_day(day1m, date=date, symbol=symbol, cfg=cfg):
        if a.arm != "LB":
            continue
        out.append(
            {
                "date": date,
                "symbol": symbol,
                "dir": a.dir,
                "ts": to_ny(pd.Timestamp(a.ts)),
                "px": float(a.px),
                "arm": f"LB{lookback_bars}",
                "lookback_ret": float(a.lookback_ret) if a.lookback_ret is not None else np.nan,
                "fav_from_open": float(a.fav_from_open),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="research_pm_pulse_C_alt_20260728")
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-23")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    stock_root = Path(paths["stock_root"])
    quote_root = Path(paths["quote_1s_root"])
    lock_path = Path(paths["open_locked_map"])
    out_dir = Path(paths["results_dir"]) / str(args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [str(s).upper() for s in (prof.get("symbols") or [])]
    months = month_list(args.start_date, args.end_date)
    dates = session_dates(args.start_date, args.end_date)
    print(f"[init] dates={len(dates)} symbols={len(symbols)}", flush=True)

    lock = load_multidte_lock_index(lock_path)
    otm = resolve_otm_rungs(prof)
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)

    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        stock_by[sym] = load_stock_month_files(stock_root, sym, months)
        print(f"  stock {sym}: {len(stock_by[sym])}", flush=True)

    windows = [
        ("POST_1400_1530", "14:00", "15:30"),  # primary — after CORE
        ("POST_1400_1500", "14:00", "15:00"),
        ("MID_1230_1400", "12:30", "14:00"),  # overlap CORE — research only
    ]
    # (arm_name, kind, lookback_bars, thr)
    arms = [
        ("LB2_t008", "LB", 2, 0.008),
        ("LB5_t008", "LB", 5, 0.008),
        ("LB10_t008", "LB", 10, 0.008),
        ("LB5_t006", "LB", 5, 0.006),
        ("LB5_t010", "LB", 5, 0.010),
        ("MOM5_t005", "MOM", 5, 0.005),
        ("MOM5_t008", "MOM", 5, 0.008),
        ("MOM15_t008", "MOM", 15, 0.008),
        ("MOM15_t010", "MOM", 15, 0.010),
    ]

    books: dict[tuple[str, str], list[dict[str, Any]]] = {
        (w[0], a[0]): [] for w in windows for a in arms
    }

    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)})", flush=True)
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None or sdf.empty:
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

            for wname, ws, we in windows:
                for aname, kind, lb, thr in arms:
                    if kind == "LB":
                        alerts = _lb_alerts(
                            day1m,
                            date=date,
                            symbol=sym,
                            window_start=ws,
                            window_end=we,
                            lookback_bars=lb,
                            thr=thr,
                        )
                    else:
                        alerts = _mom_alerts(
                            day1m,
                            date=date,
                            symbol=sym,
                            window_start=ws,
                            window_end=we,
                            lookback_bars=lb,
                            thr=thr,
                        )
                    for a in alerts:
                        arm_ts = to_ny(a["ts"])
                        ticker, dte, _ = resolve_open_lock_contract(
                            by_dte,
                            direction=a["dir"],
                            moneyness="ATM",
                            spot=float(a["px"]),
                            prefer_dte=1,
                            allowed_dte=[1],
                            clear_otm_thresh=0.01,
                            ladder=True,
                            otm_rungs=otm,
                        )
                        if not ticker or int(dte) != 1:
                            continue
                        path = _prep_path(path_for_ticker(qday, ticker))
                        if path is None or path.empty:
                            continue
                        sim = simulate_quote_tpsl(
                            path,
                            arm_ts,
                            tp=TP,
                            sl=SL,
                            max_hold_sec=MAX_HOLD,
                            fill=fill,
                            max_lag_sec=5.0,
                            max_spread_pct=0.15,
                            min_mid=0.05,
                        )
                        if sim is None:
                            continue
                        books[(wname, aname)].append(
                            {
                                "date": date,
                                "symbol": sym,
                                "dir": a["dir"],
                                "entry_ts": str(sim.get("entry_ts") or arm_ts),
                                "exit_ts": str(sim.get("exit_ts") or ""),
                                "ticker": ticker,
                                "dte": 1,
                                "ret": float(sim["ret"]),
                                "exit_reason": str(sim.get("reason") or ""),
                                "hold_sec": float(sim.get("hold_sec") or 0.0),
                                "arm": aname,
                                "window": wname,
                                "lookback_ret": a.get("lookback_ret"),
                                "entry_mid": float(sim.get("entry_mid") or np.nan),
                            }
                        )

    score_rows: list[dict[str, Any]] = []
    for (wname, aname), rows in books.items():
        tag = f"{wname}__{aname}"
        raw = pd.DataFrame(rows)
        if raw.empty:
            score_rows.append(
                {"name": tag, "window": wname, "arm": aname, "verdict": "EMPTY", "may_n": 0}
            )
            continue
        raw = raw.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)
        port_parts: list[pd.DataFrame] = []
        for _, g in raw.groupby("date", sort=True):
            sized = _portfolio_day(
                g.to_dict(orient="records"),
                position_frac=POS,
                max_concurrent=MAX_CONCURRENT,
                cooldown_minutes=COOLDOWN_MIN,
            )
            if sized:
                port_parts.append(pd.DataFrame(sized))
        book = (
            pd.concat(port_parts, ignore_index=True)
            if port_parts
            else raw.assign(size=POS, pnl_frac=raw["ret"].astype(float) * POS)
        )
        raw.to_csv(out_dir / f"raw_{tag}.csv", index=False)
        book.to_csv(out_dir / f"book_{tag}.csv", index=False)

        may = _stats_book(book[(book.date >= "2026-05-01") & (book.date <= "2026-07-23")])
        feb = _stats_book(book[(book.date >= "2026-02-01") & (book.date <= "2026-03-31")])
        verd = _verdict(may, feb)
        score_rows.append(
            {
                "name": tag,
                "window": wname,
                "arm": aname,
                "verdict": verd,
                **{f"may_{k}": v for k, v in may.items()},
                **{f"feb_{k}": v for k, v in feb.items()},
            }
        )
        print(
            f"[{tag}] may n={may['n']} mean={may['mean']} mult={may['mult']:.2f} "
            f"maxdd={may['maxdd']:.2%} → {verd}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    order = {"PASS": 0, "WEAK": 1, "THIN": 2, "FAIL": 3, "EMPTY": 4}
    sb["_o"] = sb["verdict"].map(order)
    sb = sb.sort_values(["_o", "may_mult", "may_mean"], ascending=[True, False, False]).drop(
        columns=["_o"]
    )
    sb.to_csv(out_dir / "scoreboard.csv", index=False)

    promote = sb.loc[sb.verdict == "PASS", "name"].tolist()
    weak = sb.loc[sb.verdict == "WEAK", "name"].tolist()
    post = sb[sb.window.str.startswith("POST_")]
    summary = {
        "tag": args.tag,
        "note": (
            "C-alt after FO day-open failed. 1DTE-only quote TP15/SL20. "
            "Primary = 14:00+ (post CORE). MID_1230_1400 overlaps CORE (research only). "
            "pm_fade prior accept was negative on strong window — included as context only."
        ),
        "dte": "1 only",
        "tp": TP,
        "sl": SL,
        "promote": promote,
        "weak": weak,
        "best_post_core": post.head(8).to_dict(orient="records"),
        "pm_fade_ref": {
            "tag": "pm_fade_accept_v1",
            "strong_ext012_total_ret": -0.048,
            "status": "FAIL_strong",
        },
        "scoreboard": sb.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "promote": promote, "weak": weak[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
