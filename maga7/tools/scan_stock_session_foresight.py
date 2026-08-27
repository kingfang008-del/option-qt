#!/usr/bin/env python3
"""Session foresight scan on Mag7 **stocks** (pre + RTH + AH), no CORE baseline.

Uses ``/mnt/s990/data/raw_1s/stocks`` full extended tape (≈04:00–20:00 ET).
At each clock, measure signed forward returns under simple policies:

  LONG  — always long
  MOM   — with lookback sign
  FADE  — against lookback sign

Goal: find which (session × clock × policy × lookback) pockets have dual-window
positive foresight — research ranking only (lookahead labels for discovery).

Example:
  PYTHONPATH=. python -m maga7.tools.scan_stock_session_foresight \\
    --tag research_stock_session_foresight_jan_jul
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

from maga7.common.bar_agg import aggregate_1s_to_1m, load_stock_1s_day
from maga7.common.stock_1s import session_dates

STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")
RESULTS = Path("/mnt/s990/data/maga7/results")
NY = "America/New_York"

SYMS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]

# Entry clocks (ET HH:MM) across extended session
CLOCKS = [
    # premarket
    "04:00", "05:00", "06:00", "07:00", "08:00", "08:30", "09:00",
    # open / AM
    "09:30", "09:45", "10:00", "10:30", "11:00", "11:30",
    # midday / PM
    "12:00", "13:00", "14:00", "14:30", "15:00", "15:30",
    # after hours
    "16:00", "16:30", "17:00", "18:00", "19:00",
]

HORIZONS = (15, 30, 60, 120)
LOOKBACKS = (5, 15, 30, 60)

WINDOWS = (
    ("strong_apr_jul", "2026-04-01", "2026-07-24"),
    ("weak_jan_mar", "2026-01-02", "2026-03-31"),
    ("week_0720_24", "2026-07-20", "2026-07-24"),
    ("all_jan_jul", "2026-01-02", "2026-07-24"),
)


def _hm(tod: str) -> int:
    h, m = tod.split(":")
    return int(h) * 60 + int(m)


def _session_bucket(tod: str) -> str:
    m = _hm(tod)
    if m < 9 * 60 + 30:
        return "PRE"
    if m < 11 * 60 + 30:
        return "AM"
    if m < 14 * 60:
        return "MID"
    if m < 16 * 60:
        return "PM"
    return "AH"


def _to_ny_series(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts)
    if getattr(out.dt, "tz", None) is None:
        return out.dt.tz_localize(NY)
    return out.dt.tz_convert(NY)


def _load_1m_day(root: Path, sym: str, date: str) -> pd.DataFrame:
    raw = load_stock_1s_day(root, sym, date)
    if raw is None or raw.empty:
        return pd.DataFrame()
    # full extended hours
    bars = aggregate_1s_to_1m(raw, symbol=sym, rth_only=False)
    if bars.empty:
        return bars
    bars = bars.copy()
    bars["timestamp"] = _to_ny_series(bars["timestamp"])
    bars["hm"] = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    return bars.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)


def _px_at(bars: pd.DataFrame, hm: int) -> tuple[pd.Timestamp | None, float | None]:
    sub = bars[bars["hm"] <= hm]
    if sub.empty:
        return None, None
    row = sub.iloc[-1]
    px = float(row["close"])
    if px <= 0:
        return None, None
    return pd.Timestamp(row["timestamp"]), px


def _px_forward(bars: pd.DataFrame, entry_ts: pd.Timestamp, minutes: int) -> float | None:
    target = entry_ts + pd.Timedelta(minutes=int(minutes))
    after = bars[bars["timestamp"] >= target]
    if after.empty:
        # clamp to last print of the file (AH end)
        after = bars[bars["timestamp"] >= entry_ts]
        if after.empty:
            return None
        px = float(after.iloc[-1]["close"])
        return px if px > 0 else None
    px = float(after.iloc[0]["close"])
    return px if px > 0 else None


def _lookback_ret(bars: pd.DataFrame, entry_ts: pd.Timestamp, minutes: int) -> float | None:
    start = entry_ts - pd.Timedelta(minutes=int(minutes))
    pre = bars[(bars["timestamp"] <= entry_ts) & (bars["timestamp"] >= start)]
    if len(pre) < 2:
        # fall back: last bar before entry vs earlier
        upto = bars[bars["timestamp"] <= entry_ts]
        if len(upto) < 2:
            return None
        a = float(upto.iloc[-min(len(upto), minutes + 1)]["close"])
        b = float(upto.iloc[-1]["close"])
        if a <= 0 or b <= 0:
            return None
        return b / a - 1.0
    a = float(pre.iloc[0]["close"])
    b = float(pre.iloc[-1]["close"])
    if a <= 0 or b <= 0:
        return None
    return b / a - 1.0


def scan_day(bars: pd.DataFrame, *, symbol: str, date: str, prev_close: float | None) -> list[dict[str, Any]]:
    if bars.empty:
        return []
    # session open = first print of the day file (often 04:00)
    sess_open = float(bars.iloc[0]["close"])
    rth = bars[bars["hm"] >= 9 * 60 + 30]
    rth_open = float(rth.iloc[0]["close"]) if not rth.empty else sess_open
    rows: list[dict[str, Any]] = []
    for tod in CLOCKS:
        hm = _hm(tod)
        entry_ts, entry_px = _px_at(bars, hm)
        if entry_ts is None or entry_px is None:
            continue
        # skip if clock is before first print
        if entry_ts.hour * 60 + entry_ts.minute < hm - 5 and hm > bars["hm"].iloc[0]:
            # allow stale up to 5m; if first bar much later than clock, still use first available after clock?
            pass
        # require a bar at/after clock-30 so we are not using stale overnight wrongly for late clocks
        if bars["hm"].iloc[0] > hm + 30:
            continue

        from_sess = entry_px / sess_open - 1.0 if sess_open > 0 else float("nan")
        from_rth = entry_px / rth_open - 1.0 if rth_open > 0 else float("nan")
        from_prev = entry_px / prev_close - 1.0 if prev_close and prev_close > 0 else float("nan")

        fwd: dict[str, float | None] = {}
        for h in HORIZONS:
            px = _px_forward(bars, entry_ts, h)
            fwd[f"raw_{h}m"] = (px / entry_px - 1.0) if px is not None else None

        lb_rets: dict[int, float | None] = {}
        for lb in LOOKBACKS:
            lb_rets[lb] = _lookback_ret(bars, entry_ts, lb)

        base = {
            "date": date,
            "symbol": symbol,
            "tod": tod,
            "session": _session_bucket(tod),
            "entry_ts": str(entry_ts),
            "entry_px": entry_px,
            "from_sess_open": from_sess,
            "from_rth_open": from_rth,
            "from_prev": from_prev,
            **{f"lb_{lb}m": lb_rets[lb] for lb in LOOKBACKS},
            **fwd,
        }
        rows.append(base)
    return rows


def _policy_signed(raw: float, *, policy: str, lb_ret: float | None) -> float | None:
    if raw is None or not np.isfinite(raw):
        return None
    if policy == "LONG":
        return float(raw)
    if lb_ret is None or not np.isfinite(lb_ret) or abs(lb_ret) < 1e-12:
        return None
    sign = 1.0 if lb_ret > 0 else -1.0
    if policy == "MOM":
        return float(sign * raw)
    if policy == "FADE":
        return float(-sign * raw)
    return None


def score_pocket(
    df: pd.DataFrame,
    *,
    session: str | None,
    tod: str | None,
    policy: str,
    lb: int | None,
    horizon: int,
    w0: str,
    w1: str,
    min_n: int,
) -> dict[str, Any] | None:
    sub = df[(df.date >= w0) & (df.date <= w1)]
    if session:
        sub = sub[sub.session == session]
    if tod:
        sub = sub[sub.tod == tod]
    if sub.empty:
        return None
    raw_col = f"raw_{horizon}m"
    if raw_col not in sub.columns:
        return None
    signed: list[float] = []
    for r in sub.itertuples(index=False):
        raw = getattr(r, raw_col)
        lb_ret = None if lb is None else getattr(r, f"lb_{lb}m", None)
        s = _policy_signed(raw, policy=policy, lb_ret=lb_ret)
        if s is not None and np.isfinite(s):
            signed.append(float(s))
    if len(signed) < min_n:
        return None
    arr = np.asarray(signed, dtype=float)
    return {
        "session": session or "ALL",
        "tod": tod or "ALL",
        "policy": policy,
        "lookback": lb,
        "horizon": horizon,
        "n": int(len(arr)),
        "win": float((arr > 0).mean()),
        "avg": float(arr.mean()),
        "med": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-1s", default=str(STOCK_1S))
    ap.add_argument("--tag", default="research_stock_session_foresight_jan_jul")
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-24")
    ap.add_argument("--symbols", default=",".join(SYMS))
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args(argv)

    root = Path(args.stock_1s)
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    dates = session_dates(args.start_date, args.end_date)
    print(f"foresight scan {args.start_date}..{args.end_date} syms={symbols} days={len(dates)}", flush=True)

    # prev close cache: last RTH-ish close from prior available day file
    prev_close: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        for sym in symbols:
            bars = _load_1m_day(root, sym, date)
            if bars.empty:
                continue
            day_rows = scan_day(bars, symbol=sym, date=date, prev_close=prev_close.get(sym))
            rows.extend(day_rows)
            # update prev close = last bar of day
            prev_close[sym] = float(bars.iloc[-1]["close"])
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i+1}/{len(dates)}] {date} rows={len(rows)}", flush=True)

    panel = pd.DataFrame(rows)
    panel.to_csv(out / "panel_clock_returns.csv", index=False)
    print(f"panel n={len(panel)}", flush=True)

    # Rank pockets: session×policy×lb×horizon and tod×policy×lb×horizon
    pocket_rows: list[dict[str, Any]] = []
    policies_lb = [("LONG", None)] + [(p, lb) for p in ("MOM", "FADE") for lb in LOOKBACKS]

    for wname, w0, w1 in WINDOWS:
        for session in ("PRE", "AM", "MID", "PM", "AH", None):
            for policy, lb in policies_lb:
                for h in HORIZONS:
                    sc = score_pocket(
                        panel,
                        session=session,
                        tod=None,
                        policy=policy,
                        lb=lb,
                        horizon=h,
                        w0=w0,
                        w1=w1,
                        min_n=int(args.min_n),
                    )
                    if sc:
                        pocket_rows.append({"window": wname, "grain": "session", **sc})
        for tod in CLOCKS:
            for policy, lb in policies_lb:
                for h in HORIZONS:
                    sc = score_pocket(
                        panel,
                        session=None,
                        tod=tod,
                        policy=policy,
                        lb=lb,
                        horizon=h,
                        w0=w0,
                        w1=w1,
                        min_n=max(20, int(args.min_n) // 2),
                    )
                    if sc:
                        pocket_rows.append({"window": wname, "grain": "tod", **sc})

    pockets = pd.DataFrame(pocket_rows)
    pockets.to_csv(out / "pockets.csv", index=False)

    # Dual-window stable: strong avg>0, weak avg>0, win both >=0.52, prefer higher min(avg)
    strong = pockets[(pockets.window == "strong_apr_jul") & (pockets.grain == "session")]
    weak = pockets[(pockets.window == "weak_jan_mar") & (pockets.grain == "session")]
    keys = ["session", "policy", "lookback", "horizon"]
    merged = strong.merge(
        weak,
        on=keys,
        suffixes=("_s", "_w"),
    )
    if not merged.empty:
        merged["stable"] = (
            (merged["avg_s"] > 0)
            & (merged["avg_w"] > 0)
            & (merged["win_s"] >= 0.52)
            & (merged["win_w"] >= 0.52)
        )
        merged["score"] = merged[["avg_s", "avg_w"]].min(axis=1)
        merged = merged.sort_values(["stable", "score"], ascending=[False, False])
        merged.to_csv(out / "session_dual_rank.csv", index=False)
        top_stable = merged[merged.stable].head(15)
    else:
        top_stable = pd.DataFrame()

    # TOD dual rank (finer)
    strong_t = pockets[(pockets.window == "strong_apr_jul") & (pockets.grain == "tod")]
    weak_t = pockets[(pockets.window == "weak_jan_mar") & (pockets.grain == "tod")]
    merged_t = strong_t.merge(weak_t, on=["tod", "policy", "lookback", "horizon", "session"], suffixes=("_s", "_w"))
    if not merged_t.empty:
        # session comes from score as ALL for tod grain — drop
        merged_t["stable"] = (
            (merged_t["avg_s"] > 0)
            & (merged_t["avg_w"] > 0)
            & (merged_t["win_s"] >= 0.52)
            & (merged_t["win_w"] >= 0.52)
        )
        merged_t["score"] = merged_t[["avg_s", "avg_w"]].min(axis=1)
        merged_t = merged_t.sort_values(["stable", "score"], ascending=[False, False])
        merged_t.to_csv(out / "tod_dual_rank.csv", index=False)
        top_tod = merged_t[merged_t.stable].head(20)
    else:
        top_tod = pd.DataFrame()

    # Week snapshot for top session pockets
    week = pockets[(pockets.window == "week_0720_24") & (pockets.grain == "session")]

    summary: dict[str, Any] = {
        "protocol": "stock_session_foresight",
        "stock_1s": str(root),
        "symbols": symbols,
        "date_range": [args.start_date, args.end_date],
        "n_panel": int(len(panel)),
        "n_stable_session": int(top_stable.shape[0]) if len(top_stable) else 0,
        "n_stable_tod": int(top_tod.shape[0]) if len(top_tod) else 0,
        "top_session": top_stable.head(10).to_dict(orient="records") if len(top_stable) else [],
        "top_tod": top_tod.head(15).to_dict(orient="records") if len(top_tod) else [],
    }
    if len(top_stable):
        summary["promote"] = "FORESIGHT_POCKET"
    elif len(top_tod):
        summary["promote"] = "FORESIGHT_TOD_ONLY"
    else:
        summary["promote"] = "NONE"

    # Also dump simple session×LONG baseline for readability
    base_long = []
    for wname, w0, w1 in WINDOWS:
        for session in ("PRE", "AM", "MID", "PM", "AH"):
            for h in (30, 60):
                sc = score_pocket(
                    panel, session=session, tod=None, policy="LONG", lb=None, horizon=h,
                    w0=w0, w1=w1, min_n=20,
                )
                if sc:
                    base_long.append({"window": wname, **sc})
    pd.DataFrame(base_long).to_csv(out / "baseline_long_by_session.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("=== top stable session pockets ===", flush=True)
    if len(top_stable):
        cols = [
            "session", "policy", "lookback", "horizon",
            "avg_s", "win_s", "n_s", "avg_w", "win_w", "n_w", "score",
        ]
        print(top_stable[cols].head(12).to_string(index=False), flush=True)
    else:
        print("(none)", flush=True)
    print("=== top stable tod pockets ===", flush=True)
    if len(top_tod):
        cols = [
            "tod", "policy", "lookback", "horizon",
            "avg_s", "win_s", "n_s", "avg_w", "win_w", "n_w", "score",
        ]
        print(top_tod[cols].head(15).to_string(index=False), flush=True)
    else:
        print("(none)", flush=True)
    print(json.dumps({"promote": summary["promote"], "n_stable_session": summary["n_stable_session"], "n_stable_tod": summary["n_stable_tod"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
