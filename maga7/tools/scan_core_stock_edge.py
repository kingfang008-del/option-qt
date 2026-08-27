#!/usr/bin/env python3
"""Scan Mag7 **stock-only** edge on raw 1s (no options).

Question: given CORE-like Rule-A + peer3 + regime entries, is there a *stable*
underlying PnL pocket (vs option leverage)?

Two layers:
  1) foresight — signed stock return at fixed horizons (+15/30/60/120m)
  2) book     — simple TP/SL/TIME on 1s path, cost_bps round-trip, dual windows

Example:
  PYTHONPATH=. python -m maga7.tools.scan_core_stock_edge \\
    --tag research_core_stock_edge_jan_jul
"""
from __future__ import annotations

import argparse
import copy
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
from maga7.common.regime import Mag7RegimeGate
from maga7.common.replay import month_list
from maga7.common.signals import build_all_first_rule_a_signals, count_peer_align
from maga7.common.stock_1s import build_stock_by_from_1s, session_dates

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")
NY = "America/New_York"

WINDOWS = (
    ("strong_apr_jul", "2026-04-01", "2026-07-24"),
    ("weak_jan_mar", "2026-01-02", "2026-03-31"),
    ("week_0720_24", "2026-07-20", "2026-07-24"),
    ("all_jan_jul", "2026-01-02", "2026-07-24"),
)

HORIZONS_MIN = (15, 30, 60, 120)


def _to_ny(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def _px_after(day: pd.DataFrame, entry_ts: pd.Timestamp, minutes: float) -> float | None:
    target = entry_ts + pd.Timedelta(minutes=float(minutes))
    after = day[day["timestamp"] >= target]
    if after.empty:
        # last available after entry
        sub = day[day["timestamp"] >= entry_ts]
        if sub.empty:
            return None
        px = float(sub.iloc[-1]["close"])
        return px if px > 0 else None
    px = float(after.iloc[0]["close"])
    return px if px > 0 else None


def _sim_tpsl_1s(
    day: pd.DataFrame,
    *,
    entry_ts: pd.Timestamp,
    direction: str,
    tp: float,
    sl: float,
    max_hold_min: float,
    cost_bps: float,
    stride: int = 5,
) -> dict[str, Any] | None:
    """First-touch TP/SL on 1s close path (stride for speed)."""
    entry_ts = _to_ny(entry_ts)
    after = day[day["timestamp"] >= entry_ts]
    if after.empty:
        return None
    entry_px = float(after.iloc[0]["close"])
    if entry_px <= 0:
        return None
    closes = after["close"].astype(float).to_numpy()
    ts = after["timestamp"]
    d = str(direction).upper()
    exit_i = len(closes) - 1
    reason = "TIME_EOD"
    for i in range(stride, len(closes), stride):
        t = _to_ny(ts.iloc[i])
        held = (t - entry_ts).total_seconds() / 60.0
        px = float(closes[i])
        if px <= 0:
            continue
        signed = (px / entry_px - 1.0) if d == "UP" else (1.0 - px / entry_px)
        if signed >= float(tp):
            exit_i, reason = i, "TP"
            break
        if signed <= -float(sl):
            exit_i, reason = i, "SL"
            break
        if held >= float(max_hold_min):
            exit_i, reason = i, "TIME"
            break
        if t.hour * 60 + t.minute >= 15 * 60 + 55:
            exit_i, reason = i, "EOD"
            break
    exit_px = float(closes[exit_i])
    exit_ts = _to_ny(ts.iloc[exit_i])
    raw = exit_px / entry_px - 1.0
    signed = raw if d == "UP" else -raw
    cost = 2.0 * (float(cost_bps) / 1e4)
    return {
        "entry_px": entry_px,
        "exit_px": exit_px,
        "exit_ts": exit_ts,
        "exit_reason": reason,
        "raw_stock_ret": float(raw),
        "ret": float(signed - cost),
        "hold_min": float((exit_ts - entry_ts).total_seconds() / 60.0),
    }


def _compound(trades: pd.DataFrame, *, frac: float = 0.2) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "total_ret": 0.0,
            "maxdd": 0.0,
            "trade_win": None,
            "avg_ret": None,
            "med_ret": None,
        }
    t = trades.copy()
    t["ret"] = pd.to_numeric(t["ret"], errors="coerce").fillna(0.0)
    t["date"] = t["date"].astype(str)
    daily = []
    for date, g in t.groupby("date", sort=True):
        n = max(len(g), 1)
        day_ret = float((g["ret"] * (frac / n)).sum())
        daily.append({"date": date, "day_ret": day_ret})
    ddf = pd.DataFrame(daily)
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for r in ddf.itertuples():
        eq *= 1.0 + float(r.day_ret)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
    return {
        "n_trades": int(len(t)),
        "n_days": int(len(ddf)),
        "total_ret": float(eq - 1.0),
        "maxdd": float(maxdd),
        "trade_win": float((t["ret"] > 0).mean()),
        "avg_ret": float(t["ret"].mean()),
        "med_ret": float(t["ret"].median()),
    }


def _collect_candidates(
    *,
    stock_by: dict[str, pd.DataFrame],
    symbols: list[str],
    sig_cfg: dict[str, Any],
    regime_gate: Mag7RegimeGate | None,
    dates: list[str],
    top_k: int,
) -> pd.DataFrame:
    trade_stock = {
        s: df[(df["date"].astype(str) >= dates[0]) & (df["date"].astype(str) <= dates[-1])].copy()
        for s, df in stock_by.items()
        if s in symbols and df is not None and not df.empty
    }
    all_first = build_all_first_rule_a_signals(trade_stock, sig_cfg)
    if all_first.empty:
        return all_first
    peer_min = int(sig_cfg.get("peer_align_min") or 0)
    peer_mode = str(sig_cfg.get("peer_align_mode") or "mf10")
    streak_min = int(sig_cfg.get("streak_min") or 8)
    rows: list[dict[str, Any]] = []
    for date, g in all_first.groupby("date", sort=True):
        d = str(date)
        if d < dates[0] or d > dates[-1]:
            continue
        day_rows = g.sort_values("sig_ts")
        kept = 0
        for r in day_rows.itertuples(index=False):
            if kept >= int(top_k):
                break
            ts = _to_ny(r.sig_ts)
            direction = str(r.dir).upper()
            if regime_gate is not None:
                dec = regime_gate.check(direction, ts)
                if not dec.allow:
                    continue
            peer_n = 0
            if peer_min > 0:
                peers = [s for s in symbols if s != str(r.symbol).upper()]
                peer_n = count_peer_align(
                    trade_stock,
                    date=d,
                    asof_ts=ts,
                    direction=direction,
                    peer_symbols=peers,
                    mode=peer_mode,
                    streak_min=streak_min,
                )
                if peer_n < peer_min:
                    continue
            rows.append(
                {
                    "date": d,
                    "symbol": str(r.symbol).upper(),
                    "dir": direction,
                    "sig_ts": ts,
                    "from_prev": float(getattr(r, "from_prev", float("nan"))),
                    "peer_n": int(peer_n),
                    "spot": float(getattr(r, "spot", float("nan"))),
                }
            )
            kept += 1
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--stock-1s", default=str(STOCK_1S))
    ap.add_argument("--tag", default="research_core_stock_edge_jan_jul")
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-24")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=2.0, help="one-way; RT=2x on ret")
    ap.add_argument("--size-frac", type=float, default=0.2)
    ap.add_argument("--tp", default="0.005,0.008,0.012")
    ap.add_argument("--sl", default="0.005,0.008,0.012")
    ap.add_argument("--hold", default="30,60,120")
    ap.add_argument(
        "--windows",
        default="strong_apr_jul,weak_jan_mar,week_0720_24,all_jan_jul",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    # force stock_1s root
    prof = copy.deepcopy(prof)
    prof["_paths"]["stock_1s_root"] = str(Path(args.stock_1s))
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    symbols = [str(s).upper() for s in (prof.get("symbols") or [])]
    sig_cfg = dict(prof.get("signal") or {})
    dates = session_dates(args.start_date, args.end_date)
    print(f"building 1s→1m stock_by {args.start_date}..{args.end_date} n={len(dates)}", flush=True)
    prof["date_range"] = {"start": args.start_date, "end": args.end_date}
    stock_by = build_stock_by_from_1s(prof, dates=dates, include_refs=True)

    # regime from 1m stock_root (QQQ/VIXY)
    reg = dict(prof.get("regime") or {})
    reg["enabled"] = True
    prof_reg = copy.deepcopy(prof)
    prof_reg["regime"] = reg
    gate = Mag7RegimeGate.from_profile(prof_reg, months=month_list(args.start_date, args.end_date))
    print("collecting CORE-like stock candidates…", flush=True)
    cands = _collect_candidates(
        stock_by=stock_by,
        symbols=symbols,
        sig_cfg=sig_cfg,
        regime_gate=gate,
        dates=dates,
        top_k=int(args.top_k),
    )
    cands.to_csv(out / "candidates.csv", index=False)
    print(f"candidates n={len(cands)} days={cands['date'].nunique() if len(cands) else 0}", flush=True)

    # cache 1s days
    day_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_day(sym: str, date: str) -> pd.DataFrame | None:
        key = (sym, date)
        if key in day_cache:
            return day_cache[key]
        raw = load_stock_1s_day(Path(args.stock_1s), sym, date)
        if raw is None or getattr(raw, "empty", True):
            day_cache[key] = pd.DataFrame()
            return None
        d = raw.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        if getattr(d["timestamp"].dt, "tz", None) is None:
            d["timestamp"] = d["timestamp"].dt.tz_localize(NY)
        else:
            d["timestamp"] = d["timestamp"].dt.tz_convert(NY)
        d = d.sort_values("timestamp")
        day_cache[key] = d
        return d

    # --- foresight ---
    foresight_rows: list[dict[str, Any]] = []
    for r in cands.itertuples(index=False):
        day = get_day(str(r.symbol), str(r.date))
        if day is None or day.empty:
            continue
        entry_ts = _to_ny(r.sig_ts)
        after = day[day["timestamp"] >= entry_ts]
        if after.empty:
            continue
        entry_px = float(after.iloc[0]["close"])
        if entry_px <= 0:
            continue
        row: dict[str, Any] = {
            "date": str(r.date),
            "symbol": str(r.symbol),
            "dir": str(r.dir),
            "sig_ts": str(entry_ts),
            "entry_px": entry_px,
            "peer_n": int(r.peer_n),
        }
        for h in HORIZONS_MIN:
            px = _px_after(day, entry_ts, h)
            if px is None:
                row[f"ret_{h}m"] = None
                continue
            raw = px / entry_px - 1.0
            signed = raw if str(r.dir) == "UP" else -raw
            row[f"ret_{h}m"] = float(signed)
        foresight_rows.append(row)
    foresight = pd.DataFrame(foresight_rows)
    foresight.to_csv(out / "foresight_returns.csv", index=False)

    want = {x.strip() for x in str(args.windows).split(",") if x.strip()}
    wins = [w for w in WINDOWS if w[0] in want]

    foresight_score: list[dict[str, Any]] = []
    for wname, w0, w1 in wins:
        sub = foresight[(foresight.date >= w0) & (foresight.date <= w1)] if len(foresight) else foresight
        row: dict[str, Any] = {"window": wname, "n": int(len(sub))}
        for h in HORIZONS_MIN:
            col = f"ret_{h}m"
            if sub.empty or col not in sub.columns:
                row[f"win_{h}m"] = None
                row[f"avg_{h}m"] = None
                row[f"med_{h}m"] = None
                continue
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"win_{h}m"] = float((s > 0).mean()) if len(s) else None
            row[f"avg_{h}m"] = float(s.mean()) if len(s) else None
            row[f"med_{h}m"] = float(s.median()) if len(s) else None
        foresight_score.append(row)
    fs = pd.DataFrame(foresight_score)
    fs.to_csv(out / "foresight_scoreboard.csv", index=False)
    print("=== foresight (signed stock, no cost) ===", flush=True)
    print(fs.to_string(index=False), flush=True)

    # --- TP/SL books ---
    tps = [float(x) for x in str(args.tp).split(",") if x.strip()]
    sls = [float(x) for x in str(args.sl).split(",") if x.strip()]
    holds = [float(x) for x in str(args.hold).split(",") if x.strip()]
    book_rows: list[dict[str, Any]] = []
    best_trades: dict[str, pd.DataFrame] = {}

    for tp in tps:
        for sl in sls:
            for hold in holds:
                trades: list[dict[str, Any]] = []
                for r in cands.itertuples(index=False):
                    day = get_day(str(r.symbol), str(r.date))
                    if day is None or day.empty:
                        continue
                    sim = _sim_tpsl_1s(
                        day,
                        entry_ts=_to_ny(r.sig_ts),
                        direction=str(r.dir),
                        tp=tp,
                        sl=sl,
                        max_hold_min=hold,
                        cost_bps=float(args.cost_bps),
                    )
                    if sim is None:
                        continue
                    trades.append(
                        {
                            "date": str(r.date),
                            "symbol": str(r.symbol),
                            "dir": str(r.dir),
                            "sig_ts": str(r.sig_ts),
                            "tp": tp,
                            "sl": sl,
                            "max_hold": hold,
                            **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in sim.items()},
                        }
                    )
                tdf = pd.DataFrame(trades)
                key = f"tp{tp:g}_sl{sl:g}_h{hold:g}"
                if not tdf.empty:
                    tdf.to_csv(out / f"trades_{key}.csv", index=False)
                for wname, w0, w1 in wins:
                    sub = tdf[(tdf.date >= w0) & (tdf.date <= w1)] if len(tdf) else tdf
                    m = _compound(sub, frac=float(args.size_frac))
                    book_rows.append(
                        {
                            "variant": key,
                            "window": wname,
                            "tp": tp,
                            "sl": sl,
                            "hold": hold,
                            **m,
                        }
                    )
                # stash all-window for report pick
                if "all_jan_jul" in want:
                    sub_all = (
                        tdf[(tdf.date >= "2026-01-02") & (tdf.date <= "2026-07-24")]
                        if len(tdf)
                        else tdf
                    )
                    best_trades[key] = sub_all

    books = pd.DataFrame(book_rows)
    books.to_csv(out / "book_scoreboard.csv", index=False)

    # pick by strong keep-like: maximize weak total_ret subject to strong>0 and week not worse than 0 by much
    verdict: dict[str, Any] = {
        "protocol": "core_stock_edge",
        "stock_1s": str(args.stock_1s),
        "n_candidates": int(len(cands)),
        "cost_bps_one_way": float(args.cost_bps),
        "size_frac": float(args.size_frac),
        "top_k": int(args.top_k),
        "foresight": foresight_score,
    }

    def _cell(variant: str, window: str) -> dict[str, Any] | None:
        sub = books[(books.variant == variant) & (books.window == window)]
        if sub.empty:
            return None
        return sub.iloc[0].to_dict()

    ranked: list[dict[str, Any]] = []
    for variant in sorted(books.variant.unique()) if len(books) else []:
        s = _cell(variant, "strong_apr_jul")
        w = _cell(variant, "weak_jan_mar")
        wk = _cell(variant, "week_0720_24")
        a = _cell(variant, "all_jan_jul")
        if not s or not w:
            continue
        strong_ok = float(s["total_ret"]) > 0 and float(s.get("trade_win") or 0) >= 0.52
        weak_ok = float(w["total_ret"]) > 0 and float(w.get("trade_win") or 0) >= 0.50
        week_ok = wk is None or float(wk["total_ret"]) >= -0.01
        stable = bool(strong_ok and weak_ok and week_ok)
        ranked.append(
            {
                "variant": variant,
                "strong_ret": float(s["total_ret"]),
                "strong_win": s.get("trade_win"),
                "strong_dd": s.get("maxdd"),
                "weak_ret": float(w["total_ret"]),
                "weak_win": w.get("trade_win"),
                "weak_dd": w.get("maxdd"),
                "week_ret": None if wk is None else float(wk["total_ret"]),
                "all_ret": None if a is None else float(a["total_ret"]),
                "all_n": None if a is None else int(a["n_trades"]),
                "stable": stable,
            }
        )
    ranked.sort(
        key=lambda x: (
            int(x["stable"]),
            float(x["weak_ret"]) + float(x["strong_ret"]),
            float(x.get("all_ret") or -999),
        ),
        reverse=True,
    )
    verdict["book_ranked"] = ranked[:12]
    stable_n = sum(1 for x in ranked if x["stable"])
    best = ranked[0] if ranked else None
    verdict["n_stable_variants"] = int(stable_n)
    verdict["best"] = best
    # Heuristic promote
    if best and best["stable"] and float(best["strong_ret"]) >= 0.02 and float(best["weak_ret"]) >= 0.01:
        verdict["promote"] = "STOCK_EDGE_RESEARCH"
    elif best and best["stable"]:
        verdict["promote"] = "STOCK_EDGE_WEAK"
    else:
        verdict["promote"] = "NONE"

    # foresight stability note
    f_all = next((x for x in foresight_score if x["window"] == "all_jan_jul"), None)
    f_s = next((x for x in foresight_score if x["window"] == "strong_apr_jul"), None)
    f_w = next((x for x in foresight_score if x["window"] == "weak_jan_mar"), None)
    verdict["foresight_verdict"] = {
        "all_30m_avg": None if not f_all else f_all.get("avg_30m"),
        "all_30m_win": None if not f_all else f_all.get("win_30m"),
        "strong_30m_avg": None if not f_s else f_s.get("avg_30m"),
        "weak_30m_avg": None if not f_w else f_w.get("avg_30m"),
        "edge_present": bool(
            f_all
            and f_all.get("avg_30m") is not None
            and f_all["avg_30m"] > 0
            and f_all.get("win_30m") is not None
            and f_all["win_30m"] >= 0.52
            and f_s
            and f_s.get("avg_30m") is not None
            and f_s["avg_30m"] > 0
            and f_w
            and f_w.get("avg_30m") is not None
            and f_w["avg_30m"] > 0
        ),
    }

    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str))
    print("=== top books ===", flush=True)
    print(pd.DataFrame(ranked[:8]).to_string(index=False) if ranked else "(none)", flush=True)
    print(json.dumps({k: verdict[k] for k in ("promote", "n_stable_variants", "best", "foresight_verdict")}, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
