#!/usr/bin/env python3
"""
Lightweight WebSocket bridge for dashboard live UI.

The Streamlit page still owns initial rendering and order actions. This bridge
pushes minute bars, momentum leaders, underlying option quotes, and chart-symbol
position hints (derived from intraday helpers) so the browser updates without
rerun-based polling. Full-portfolio metrics / holdings live under **Trade Log**
and optional ``GET /api/intraday`` (below), not duplicated in each WS frame.

Run (default ``ws://127.0.0.1:8765/ws``)::

    python production/baseline/DAO/dashboard_ws_bridge.py

REST (CORS ``*``, optional debugging or external tooling)::

    GET /api/intraday?symbol=SPY

In ``dashboard_monitor_ultimate.py``, enable sidebar **Realtime WS UI** and set
``DASHBOARD_KLINE_WS_URL`` if the bridge is not on localhost (optional; default
appends ``?symbol=`` to the base URL).
"""

import argparse
import asyncio
import json
import math
import sys
import time
import warnings
from pathlib import Path

from aiohttp import web
import numpy as np
import pandas as pd
import psycopg2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PG_DB_URL  # noqa: E402
from dashboard_intraday_snap import augment_intraday_for_chart, build_intraday_snapshot  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)


_LEADER_CACHE = {
    "expires_at": 0.0,
    "payload": {"long": [], "short": [], "stats": {}},
}


def _clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _safe_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    std = float(s.std(ddof=0))
    if std < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return ((s - float(s.mean())) / std).fillna(0.0)


def _rows_for_ws(df: pd.DataFrame, max_rows: int = 5) -> list[dict]:
    if df.empty:
        return []
    cols = ["symbol", "momentum_score", "ret_5m", "alpha", "vol_z"]
    rows = []
    for item in df.head(max_rows).to_dict("records"):
        rows.append({col: _clean_value(item.get(col)) for col in cols})
    return rows


def fetch_latest_bar(symbol: str) -> dict | None:
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return None
    sql = """
        SELECT ts, open, high, low, close, volume
        FROM market_bars_1m
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT 1
    """
    with psycopg2.connect(PG_DB_URL) as conn:
        df = pd.read_sql_query(sql, conn, params=(symbol,))
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "time": int(row["ts"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0.0) or 0.0),
    }


def fetch_momentum_leaders(max_symbols: int = 5, ttl_sec: float = 2.5) -> dict:
    now = time.time()
    if now < float(_LEADER_CACHE["expires_at"]):
        return _LEADER_CACHE["payload"]

    start_ts = int(now) - 3 * 24 * 3600
    with psycopg2.connect(PG_DB_URL) as conn:
        sql_px = """
            WITH recent AS (
                SELECT
                    symbol,
                    ts,
                    close,
                    volume,
                    LAG(close, 5) OVER (PARTITION BY symbol ORDER BY ts) AS close_5m_ago,
                    LAG(close, 15) OVER (PARTITION BY symbol ORDER BY ts) AS close_15m_ago,
                    AVG(volume) OVER (
                        PARTITION BY symbol
                        ORDER BY ts
                        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                    ) AS vol_avg_20
                FROM market_bars_1m
                WHERE ts >= %s
            ),
            latest AS (
                SELECT DISTINCT ON (symbol)
                    symbol, ts, close, volume, close_5m_ago, close_15m_ago, vol_avg_20
                FROM recent
                ORDER BY symbol, ts DESC
            )
            SELECT *
            FROM latest
            WHERE close > 0
        """
        df_px = pd.read_sql_query(sql_px, conn, params=(start_ts,))

        sql_alpha = """
            SELECT DISTINCT ON (symbol)
                symbol, ts AS alpha_ts, alpha, iv, vol_z
            FROM alpha_logs
            WHERE ts >= %s
            ORDER BY symbol, ts DESC
        """
        df_alpha = pd.read_sql_query(sql_alpha, conn, params=(start_ts,))

    if df_px.empty:
        payload = {"long": [], "short": [], "stats": {}}
        _LEADER_CACHE.update(expires_at=now + ttl_sec, payload=payload)
        return payload

    df = df_px.merge(df_alpha, on="symbol", how="left")
    df["ret_5m"] = np.where(df["close_5m_ago"] > 0, df["close"] / df["close_5m_ago"] - 1.0, np.nan)
    df["ret_15m"] = np.where(df["close_15m_ago"] > 0, df["close"] / df["close_15m_ago"] - 1.0, np.nan)
    df["vol_impulse"] = np.where(df["vol_avg_20"] > 0, df["volume"] / df["vol_avg_20"] - 1.0, np.nan)
    df["momentum_score"] = (
        0.40 * _safe_zscore(df["ret_5m"])
        + 0.25 * _safe_zscore(df["ret_15m"])
        + 0.20 * _safe_zscore(df.get("alpha", 0.0))
        + 0.10 * _safe_zscore(df.get("vol_z", 0.0))
        + 0.05 * _safe_zscore(df["vol_impulse"])
    )
    quality = (
        pd.to_numeric(df["close"], errors="coerce").fillna(0) > 0
    ) & (
        pd.to_numeric(df["volume"], errors="coerce").fillna(0) >= 0
    )
    df = df[quality].copy()
    if df.empty:
        payload = {"long": [], "short": [], "stats": {}}
        _LEADER_CACHE.update(expires_at=now + ttl_sec, payload=payload)
        return payload

    long_df = df.sort_values("momentum_score", ascending=False).head(max_symbols).copy()
    short_df = df.sort_values("momentum_score", ascending=True).head(max_symbols).copy()
    breadth = float((pd.to_numeric(df["ret_5m"], errors="coerce") > 0).mean())
    median_ret_5m = float(pd.to_numeric(df["ret_5m"], errors="coerce").median())
    top3_strength = float(long_df["momentum_score"].head(3).mean()) if not long_df.empty else 0.0
    if breadth >= 0.60 and median_ret_5m > 0:
        regime = "RISK-ON"
    elif breadth <= 0.40 and median_ret_5m < 0:
        regime = "RISK-OFF"
    else:
        regime = "CHOP"

    payload = {
        "long": _rows_for_ws(long_df, max_symbols),
        "short": _rows_for_ws(short_df, max_symbols),
        "stats": {
            "regime": regime,
            "breadth_up_ratio": breadth,
            "median_ret_5m": median_ret_5m,
            "top3_strength": top3_strength,
            "sample_size": int(len(df)),
            "latest_ts": int(pd.to_numeric(df["ts"], errors="coerce").max()),
        },
    }
    _LEADER_CACHE.update(expires_at=now + ttl_sec, payload=payload)
    return payload


def build_intraday_public_payload(chart_symbol: str) -> dict:
    """Portfolio + chart-symbol quotes/position, JSON-safe values (no Redis/PG internals)."""
    chart_symbol = str(chart_symbol or "").strip().upper()
    base_snap = build_intraday_snapshot()
    snap = augment_intraday_for_chart(base_snap, chart_symbol)
    intra = {k: v for k, v in snap.items() if k != "positions_by_symbol"}
    return json.loads(json.dumps(intra, allow_nan=False))


def build_snapshot(symbol: str, max_leaders: int) -> dict:
    sym_upper = str(symbol or "").strip().upper()
    base = {
        "type": "snapshot",
        "symbol": sym_upper,
        "bar": fetch_latest_bar(sym_upper),
        "leaders": fetch_momentum_leaders(max_symbols=max_leaders),
    }
    if not sym_upper:
        return base

    intra = build_intraday_public_payload(sym_upper)
    base["quotes"] = intra.get("quotes") or {}
    base["position"] = intra.get("chart_position")

    try:
        return json.loads(json.dumps(base, allow_nan=False))
    except (TypeError, ValueError):
        base.pop("quotes", None)
        base.pop("position", None)
        return base


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)

    symbol = str(request.query.get("symbol", "") or "").strip().upper()
    interval = max(float(request.app["interval"]), 0.25)
    max_leaders = int(request.app["max_leaders"])
    if not symbol:
        await ws.send_str(json.dumps({"type": "error", "message": "missing symbol"}))
        await ws.close()
        return ws

    while not ws.closed:
        try:
            payload = await asyncio.to_thread(build_snapshot, symbol, max_leaders)
            await ws.send_str(json.dumps(payload, allow_nan=False))
        except (ConnectionResetError, asyncio.CancelledError):
            break
        except Exception as exc:
            if ws.closed:
                break
            try:
                await ws.send_str(json.dumps({"type": "error", "message": str(exc)}))
            except (ConnectionResetError, asyncio.CancelledError):
                break
        await asyncio.sleep(interval)
    return ws


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "dashboard_ws_bridge"})


def _cors_intraday(resp: web.Response) -> web.Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


async def intraday_options_handler(_request: web.Request) -> web.Response:
    return _cors_intraday(web.Response(status=204))


async def intraday_api_handler(request: web.Request) -> web.Response:
    sym_upper = str(request.query.get("symbol", "") or "").strip().upper()
    if not sym_upper:
        return _cors_intraday(web.json_response({"error": "missing symbol"}, status=400))
    try:
        payload = {"type": "intraday_public", "intraday": build_intraday_public_payload(sym_upper)}
        return _cors_intraday(web.json_response(payload))
    except Exception as exc:
        return _cors_intraday(
            web.json_response({"intraday": {"error": str(exc)}}, status=500)
        )


def make_app(interval: float, max_leaders: int) -> web.Application:
    app = web.Application()
    app["interval"] = interval
    app["max_leaders"] = max_leaders
    app.router.add_get("/", health_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_options("/api/intraday", intraday_options_handler)
    app.router.add_get("/api/intraday", intraday_api_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard WebSocket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-leaders", type=int, default=5)
    args = parser.parse_args()
    web.run_app(make_app(args.interval, args.max_leaders), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
