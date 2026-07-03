#!/usr/bin/env python3
"""
Replay QQQ minimal stack without TFT — rule-based alpha + Polygon 1m stock/option.

No slow model required. Alpha proxy = momentum (snap + 5m ROC), mimicking what TFT
would roughly signal on recovery legs.

Usage:
  cd New_Pro/baseline_qqq
  source config/minimal_stack.env
  python tools/replay_qqq_no_model.py --date 2026-05-28
  python tools/replay_qqq_no_model.py --date 2026-05-28 --profile multi_band
  python tools/replay_qqq_no_model.py --cache reports/replay_cache/2026-05-28.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bidirectional_regime import resolve_day_type
from tools.exec_path_analyzer import MinuteBar, print_report, run_replay

NY = pytz.timezone("America/New_York")
CACHE_DIR = ROOT / "reports" / "replay_cache"
DEFAULT_OPT_TICKER = "O:QQQ260528C00730000"  # 0DTE ATM ~730 call; overridden per date


def _polygon_key() -> str:
    return (
        os.environ.get("POLYGON_API_KEY", "").strip()
        or os.environ.get("POLYGON_KEY", "").strip()
        or "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
    )


def _fetch_json(url: str, *, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(0.35 * (attempt + 1))
            with urllib.request.urlopen(url, timeout=45) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    return {}


def _fetch_minute_aggs(ticker: str, date: str, api_key: str) -> List[dict]:
    enc_ticker = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{enc_ticker}/range/1/minute/"
        f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    data = _fetch_json(url)
    if data.get("status") != "OK":
        raise RuntimeError(f"Polygon error for {ticker} {date}: {data}")
    return list(data.get("results") or [])


def _rth_minutes(stock_rows: List[dict], opt_rows: List[dict]) -> List[int]:
    stock_ts = {r["t"] for r in stock_rows}
    opt_ts = {r["t"] for r in opt_rows}
    keys = sorted(stock_ts & opt_ts)
    out = []
    for k in keys:
        ts = datetime.fromtimestamp(k / 1000, NY)
        if (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and ts.hour < 16:
            out.append(k)
    return out


def _roc(cur: float, prev: float) -> float:
    if prev <= 0:
        return 0.0
    return (cur - prev) / prev


def proxy_alpha(*, snap_roc: float, roc_5m: float, day_roc: float, opt_mid: float) -> float:
    """
    Rule-based edge proxy (no TFT).
    Direction follows snap first (human-like), magnitude from momentum.
    """
    mag = abs(snap_roc) * 14.0 + abs(roc_5m) * 8.0
    mag = max(mag, 0.012)

    # 跌日 V 底：snap 转正 → 强制 call 正 alpha
    if snap_roc >= 0.0008 and day_roc < 0.01:
        return min(0.06, max(0.020, mag))
    if snap_roc <= -0.0008 and day_roc > -0.01:
        return -min(0.06, max(0.020, mag))

    raw = snap_roc * 14.0 + roc_5m * 8.0
    if opt_mid < 0.85 and snap_roc > 0.001:
        raw = max(raw, 0.022)
    return max(-0.06, min(0.06, raw))


def load_or_fetch_day(
    date: str,
    *,
    opt_ticker: str,
    cache_path: Optional[Path] = None,
) -> dict:
    cache_path = cache_path or (CACHE_DIR / f"{date}.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    api_key = _polygon_key()
    stock_rows = _fetch_minute_aggs("QQQ", date, api_key)
    opt_rows = _fetch_minute_aggs(opt_ticker, date, api_key)
    payload = {
        "date": date,
        "opt_ticker": opt_ticker,
        "stock": stock_rows,
        "option": opt_rows,
        "fetched_at": datetime.now(NY).isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_minute_bars(payload: dict) -> Tuple[List[MinuteBar], dict]:
    stock_by_ts = {r["t"]: r for r in payload["stock"]}
    opt_by_ts = {r["t"]: r for r in payload["option"]}
    keys = _rth_minutes(payload["stock"], payload["option"])
    if not keys:
        raise RuntimeError("No overlapping RTH minutes between stock and option")

    session_open = stock_by_ts[keys[0]]["o"]
    bars: List[MinuteBar] = []
    prev_stock = session_open
    prev_5m_stock = session_open
    last_5m_ts = keys[0]

    for k in keys:
        s = stock_by_ts[k]
        o = opt_by_ts[k]
        ts = datetime.fromtimestamp(k / 1000, NY)
        if ts.hour == 9 and ts.minute < 45:
            prev_stock = s["c"]
            continue

        stock_c = float(s["c"])
        opt_c = float(o["c"])
        opt_l = float(o["l"])
        opt_h = float(o["h"])
        spread_pct = (opt_h - opt_l) / opt_c if opt_c > 0.01 else 0.08
        spread_pct = min(max(spread_pct, 0.04), 0.10)  # minute OHLC 高估 spread，cap 10%
        half = opt_c * spread_pct / 2.0
        bid = max(0.01, opt_c - half)
        ask = opt_c + half

        snap = _roc(stock_c, prev_stock)
        if k - last_5m_ts >= 5 * 60_000:
            prev_5m_stock = stock_by_ts[last_5m_ts]["c"]
            last_5m_ts = k
        roc_5m = _roc(stock_c, prev_5m_stock)
        day_roc = _roc(stock_c, session_open)
        alpha = proxy_alpha(
            snap_roc=snap, roc_5m=roc_5m, day_roc=day_roc, opt_mid=opt_c,
        )
        idx = day_roc * 0.85
        vol_z = 2.2 if abs(snap) > 0.001 else 1.6
        macd = 0.008 + snap * 4.0

        bars.append(
            MinuteBar(
                time_str=ts.strftime("%H:%M"),
                opt_mid=opt_c,
                opt_bid=bid,
                opt_ask=ask,
                stock_price=stock_c,
                stock_roc_5m=roc_5m,
                snap_roc=snap,
                alpha=alpha,
                vol_z=vol_z,
                spy_roc=idx * 0.95,
                qqq_roc=idx,
                spread_feat=min(spread_pct, 0.14),
                iv_momentum=0.12,
                is_volatile=abs(snap) > 0.0015,
                is_ready=True,
                macd_hist=macd,
                qqq_day_roc=day_roc,
            )
        )
        prev_stock = stock_c

    meta = {
        "date": payload["date"],
        "opt_ticker": payload.get("opt_ticker"),
        "session_open": session_open,
        "n_bars": len(bars),
        "call_min": min(b.opt_mid for b in bars),
        "call_max": max(b.opt_mid for b in bars),
        "stock_day_roc": _roc(bars[-1].stock_price, session_open) if bars else 0.0,
        "day_type": resolve_day_type({
            "qqq_day_roc": _roc(bars[-1].stock_price, session_open) if bars else 0.0,
            "stock_roc": bars[-1].stock_roc_5m if bars else 0.0,
            "snap_roc": bars[-1].snap_roc if bars else 0.0,
            "alpha": bars[-1].alpha if bars else 0.0,
        }).value if bars else "unknown",
    }
    return bars, meta


def print_preflight(meta: dict, bars: List[MinuteBar]) -> None:
    print("=" * 72)
    print("QQQ No-Model Replay (rule alpha, no TFT)")
    print(f"Date: {meta['date']} | option: {meta.get('opt_ticker')}")
    print(f"Bars: {meta['n_bars']} | day_type≈{meta['day_type']}")
    print(
        f"Stock open→close ROC: {meta['stock_day_roc']:.2%} | "
        f"Call range: ${meta['call_min']:.2f} – ${meta['call_max']:.2f}"
    )
    print("Alpha proxy: snap*14 + roc5m*8 (+ recovery boost on cheap option + snap)")
    print("=" * 72)
    for b in bars:
        if b.time_str in {"09:49", "09:50", "09:51", "10:12", "10:13", "11:00"}:
            print(
                f"  {b.time_str}  call=${b.opt_mid:.2f}  qqq={b.stock_price:.2f}  "
                f"snap={b.snap_roc:.4f}  a={b.alpha:.3f}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay minimal stack without TFT")
    parser.add_argument("--date", default="2026-05-28", help="Trading date YYYY-MM-DD")
    parser.add_argument("--profile", default=os.environ.get("EXEC_PROFILE", "auto_hybrid"))
    parser.add_argument("--opt-ticker", default="", help="Polygon option ticker (O:...)")
    parser.add_argument("--cache", default="", help="Use specific cache json path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args(argv)

    date = args.date
    opt_ticker = args.opt_ticker.strip() or DEFAULT_OPT_TICKER
    cache_path = Path(args.cache) if args.cache else CACHE_DIR / f"{date}.json"

    payload = load_or_fetch_day(date, opt_ticker=opt_ticker, cache_path=cache_path)
    if args.fetch_only:
        print(f"Cached → {cache_path}")
        return 0

    bars, meta = build_minute_bars(payload)
    print_preflight(meta, bars)

    result = run_replay(
        bars,
        exec_profile=args.profile,
        day=date,
    )
    result["meta"] = meta
    result["alpha_mode"] = "rule_proxy_no_tft"

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print_report(result)
        print("-" * 72)
        entries = [
            e for row in result["timeline"]
            for e in row["events"]
            if e.get("type") == "ENTRY"
        ]
        rejects = [
            e for row in result["timeline"]
            for e in row["events"]
            if e.get("type") == "ENTRY_REJECT"
        ]
        print(f"Entries: {len(entries)} | Rejects: {len(rejects)}")
        if entries:
            for e in entries:
                print(f"  ENTRY {e.get('time')} dir={e.get('dir')} reason={e.get('reason')}")
        else:
            print("  (no entry — check alpha proxy vs gates; try EXEC_PROFILE=multi_band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
