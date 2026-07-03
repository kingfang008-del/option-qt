#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实行情门控验证 —— Polygon 1m 股票+期权 → 对比 FAST_GATE 开/关 + entry_bridge。

用法:
  cd New_Pro/baseline_qqq
  python tools/validate_gates_realday.py --symbol AAPL --date 2026-07-02
  python tools/validate_gates_realday.py --symbol AAPL --date 2026-07-02 --csv /path/bars.parquet

无 POLYGON_API_KEY 时可只传 --csv(需列: timestamp, stock_close, opt_bid, opt_ask, opt_mid)。
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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz

_BASELINE = Path(__file__).resolve().parents[1]
_REPO = _BASELINE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

import baseline_paths  # noqa: E402,F401

NY = pytz.timezone("America/New_York")


def _polygon_key() -> str:
    return (
        os.environ.get("POLYGON_API_KEY", "").strip()
        or os.environ.get("POLYGON_KEY", "").strip()
        or "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"  # 与 replay_qqq_no_model 同源 fallback
    )


def _fetch_json(url: str, *, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            time.sleep(0.35 * (attempt + 1))
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 403) and attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    return {}


def _fetch_minute_aggs(ticker: str, date: str, api_key: str) -> List[dict]:
    enc = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/minute/"
        f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    data = _fetch_json(url)
    if data.get("status") != "OK":
        raise RuntimeError(f"Polygon {ticker} {date}: {data.get('status')} {data.get('error', '')}")
    return list(data.get("results") or [])


def _occ_call(symbol: str, yymmdd: str, strike: float) -> str:
    strike_int = int(round(strike * 1000))
    return f"O:{symbol}{yymmdd}C{strike_int:08d}"


def _pick_atm_strike(open_px: float) -> float:
    step = 2.5 if open_px < 200 else 5.0
    return round(open_px / step) * step


def _rth_overlap(stock_rows: List[dict], opt_rows: List[dict]) -> List[int]:
    s_ts = {r["t"] for r in stock_rows}
    o_ts = {r["t"] for r in opt_rows}
    keys = sorted(s_ts & o_ts)
    out = []
    for k in keys:
        ts = datetime.fromtimestamp(k / 1000, NY)
        if (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and ts.hour < 16:
            out.append(k)
    return out


def _roc(cur: float, prev: float) -> float:
    return (cur - prev) / prev if prev > 0 else 0.0


def proxy_edge(snap_roc: float, roc_5m: float) -> Tuple[float, float]:
    """动量 proxy 代替 TFT net_edge / q10(保守: q10 = edge - 0.003)。"""
    mag = abs(snap_roc) * 12.0 + abs(roc_5m) * 6.0
    direction = 1.0 if snap_roc >= 0 else -1.0
    edge = direction * min(0.08, max(0.0, mag))
    q10 = edge - 0.003
    return edge, q10


@dataclass
class MinuteRow:
    ts_ms: int
    time: datetime
    stock_close: float
    opt_bid: float
    opt_ask: float
    opt_mid: float
    spread_pct: float
    vw_spread: float
    snap_roc: float
    roc_5m: float
    net_edge: float
    net_edge_q10: float


def load_polygon_day(symbol: str, date: str, api_key: str) -> Tuple[List[MinuteRow], str]:
    yymmdd = date.replace("-", "")[2:]  # 2026-07-02 -> 260702
    stock_ticker = symbol.upper()
    stock_rows = _fetch_minute_aggs(stock_ticker, date, api_key)
    if not stock_rows:
        raise RuntimeError(f"无股票分钟线: {stock_ticker} {date}")

    open_px = float(stock_rows[0].get("o") or stock_rows[0].get("c") or 0)
    strike = _pick_atm_strike(open_px)
    opt_ticker = _occ_call(stock_ticker, yymmdd, strike)
    try:
        opt_rows = _fetch_minute_aggs(opt_ticker, date, api_key)
    except Exception:
        # 0DTE 不可用则试最近周五 weekly(简化: 同 yymmdd)
        opt_rows = _fetch_minute_aggs(opt_ticker, date, api_key)

    keys = _rth_overlap(stock_rows, opt_rows)
    stock_by_t = {r["t"]: r for r in stock_rows}
    opt_by_t = {r["t"]: r for r in opt_rows}

    rows: List[MinuteRow] = []
    prev_close = None
    close_5m_ago = None
    for i, k in enumerate(keys):
        sr = stock_by_t[k]
        orow = opt_by_t[k]
        ts = datetime.fromtimestamp(k / 1000, NY)
        sc = float(sr.get("c") or 0)
        bid = float(orow.get("l") or orow.get("c") or 0)  # minute low ~ bid proxy
        ask = float(orow.get("h") or orow.get("c") or 0)
        mid = float(orow.get("c") or (bid + ask) / 2)
        if bid <= 0 or ask <= 0 or mid <= 0:
            bid, ask = mid * 0.98, mid * 1.02
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        vw_spread = spread_pct * 1.05  # FCS vw 常略高于 ATM 点差

        snap = _roc(sc, prev_close) if prev_close else 0.0
        if i >= 5:
            close_5m_ago = float(stock_by_t[keys[i - 5]].get("c") or sc)
        roc5 = _roc(sc, close_5m_ago) if close_5m_ago else 0.0
        edge, q10 = proxy_edge(snap, roc5)

        rows.append(
            MinuteRow(
                ts_ms=k,
                time=ts,
                stock_close=sc,
                opt_bid=bid,
                opt_ask=ask,
                opt_mid=mid,
                spread_pct=spread_pct,
                vw_spread=vw_spread,
                snap_roc=snap,
                roc_5m=roc5,
                net_edge=edge,
                net_edge_q10=q10,
            )
        )
        prev_close = sc

    return rows, opt_ticker


def load_csv(path: Path) -> List[MinuteRow]:
    import pandas as pd

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    rows = []
    for _, r in df.iterrows():
        mid = float(r.get("opt_mid", 0) or 0)
        bid = float(r.get("opt_bid", mid * 0.98) or 0)
        ask = float(r.get("opt_ask", mid * 1.02) or 0)
        spread_pct = (ask - bid) / mid if mid > 0 else 0.0
        rows.append(
            MinuteRow(
                ts_ms=int(r["timestamp"].timestamp() * 1000),
                time=r["timestamp"].to_pydatetime(),
                stock_close=float(r.get("stock_close", 0) or 0),
                opt_bid=bid,
                opt_ask=ask,
                opt_mid=mid,
                spread_pct=spread_pct,
                vw_spread=float(r.get("options_vw_spread", spread_pct) or spread_pct),
                snap_roc=float(r.get("snap_roc", 0) or 0),
                roc_5m=float(r.get("roc_5m", 0) or 0),
                net_edge=float(r.get("net_edge", 0) or 0),
                net_edge_q10=float(r.get("net_edge_q10", 0) or 0),
            )
        )
    return rows


def _build_ctx(row: MinuteRow, *, fast_gate_on: bool) -> dict:
    return {
        "symbol": "AAPL",
        "is_ready": True,
        "is_banned": False,
        "position": 0,
        "cooldown_until": 0.0,
        "curr_ts": row.ts_ms / 1000.0,
        "time": row.time,
        "net_edge_raw": row.net_edge,
        "alpha_z": row.net_edge,
        "net_edge_q10": row.net_edge_q10,
        "call_edge": row.net_edge,
        "put_edge": 0.0,
        "vol_z": 0.5,
        "options_vw_spread": row.vw_spread,
        "options_iv_momentum": 0.0,
        "bid": row.opt_bid,
        "ask": row.opt_ask,
        "curr_price": row.opt_mid,
        "spread_divergence": 0.0,
        "spy_roc": row.snap_roc,
        "qqq_roc": row.snap_roc,
        "snap_roc": row.snap_roc,
        "stock_roc": row.roc_5m,
        "_fast_gate_on": fast_gate_on,
    }


def _run_day(rows: List[MinuteRow], *, fast_gate_on: bool, use_entry_bridge: bool) -> Counter:
    from strategy.config0 import StrategyConfig
    from strategy.core_v0 import StrategyCoreV0

    cfg = StrategyConfig()
    cfg.FAST_GATE_ENABLED = bool(fast_gate_on)

    if use_entry_bridge:
        os.environ["QQQ_BTC_LIVE"] = "1"
        from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

        apply_strategy_entry_patch(StrategyCoreV0)

    core = StrategyCoreV0(cfg)
    blocks = Counter()
    entries = 0
    for row in rows:
        ctx = _build_ctx(row, fast_gate_on=fast_gate_on)
        sig = core.decide_entry(ctx)
        if sig:
            entries += 1
        else:
            trace = core.get_last_gate_trace()
            blk = next((g.get("gate") for g in reversed(trace) if g.get("status") == "block"), "unknown")
            blocks[blk or "unknown"] += 1
    blocks["_entries"] = entries
    blocks["_minutes"] = len(rows)
    return blocks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="真实日门控对比(FAST_GATE 开/关)")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--date", default="2026-07-02", help="NY 交易日 YYYY-MM-DD")
    parser.add_argument("--csv", default="", help="本地 parquet/csv 替代 Polygon")
    args = parser.parse_args(argv)

    if args.csv:
        rows = load_csv(Path(args.csv))
        opt_ticker = "csv"
    else:
        key = _polygon_key()
        if not key:
            print("需要 POLYGON_API_KEY 或 --csv", file=sys.stderr)
            return 1
        print(f"拉取 Polygon {args.symbol} {args.date} ...")
        rows, opt_ticker = load_polygon_day(args.symbol.upper(), args.date, key)

    if not rows:
        print("无 RTH 重叠分钟", file=sys.stderr)
        return 1

    spreads = [r.spread_pct for r in rows]
    print(f"\n=== {args.symbol} {args.date} | opt={opt_ticker} | bars={len(rows)} ===")
    print(f"期权点差 bid/ask: min={min(spreads):.3f} med={sorted(spreads)[len(spreads)//2]:.3f} max={max(spreads):.3f}")
    print(f"vw_spread>FAST_GATE(12%): {sum(1 for r in rows if r.vw_spread > 0.12)} bars")
    print(f"spread>replay(6%):      {sum(1 for r in rows if r.spread_pct > 0.06)} bars")

    legacy = _run_day(rows, fast_gate_on=True, use_entry_bridge=True)
    converged = _run_day(rows, fast_gate_on=False, use_entry_bridge=True)

    print("\n--- 入场次数 (entry_bridge + q10 proxy) ---")
    print(f"  FAST_GATE=1 (legacy): {legacy['_entries']} / {legacy['_minutes']}")
    print(f"  FAST_GATE=0 (收敛后): {converged['_entries']} / {converged['_minutes']}")
    delta = converged["_entries"] - legacy["_entries"]
    print(f"  增量入场: {delta:+d}")

    def _top(c: Counter, n=5):
        items = [(k, v) for k, v in c.items() if not str(k).startswith("_")]
        return sorted(items, key=lambda x: -x[1])[:n]

    print("\n--- Top 拦截门 (FAST_GATE=1) ---")
    for g, n in _top(legacy):
        print(f"  {g:28s} {n:4d}")

    print("\n--- Top 拦截门 (FAST_GATE=0) ---")
    for g, n in _top(converged):
        print(f"  {g:28s} {n:4d}")

    only_fast = legacy.get("E6b.fast_spread", 0)
    e9_delta = converged.get("E9.qqq_btc_entry", 0) - legacy.get("E9.qqq_btc_entry", 0)
    if only_fast > 0 and legacy["_entries"] == converged["_entries"]:
        print(
            f"\n✓  FAST_GATE 在 {only_fast} 分钟上先于 choose_entry 拦截,"
            f"但收敛后这些分钟改由 E9 拦截(+{e9_delta} 次);净入场次数不变 → 重复门控已消除"
        )
    elif delta > 0:
        print(f"\n✓  收敛后多放行 {delta} 次入场(FAST_GATE 曾误挡)")
    elif delta < 0:
        print(f"\n⚠️  收敛后入场减少 {delta}(异常,需排查)")
    else:
        print("\n✓  本日 FAST_GATE 开/关入场次数相同")

    print(f"\n配置: COOLDOWN_MINUTES={__import__('config').COOLDOWN_MINUTES} FAST_GATE_ENABLED(bootstrap)={os.environ.get('FAST_GATE_ENABLED', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
