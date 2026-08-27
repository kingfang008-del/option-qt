"""TradingView-style K-line for Mag7 Dash (reuse production futu_kline component)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parents[2]
FUTU_KLINE_PATH = (
    REPO / "production" / "baseline" / "DAO" / "components" / "futu_kline"
)

_FUTU_KLINE = None
if FUTU_KLINE_PATH.is_dir():
    _FUTU_KLINE = components.declare_component(
        "maga7_futu_kline",
        path=str(FUTU_KLINE_PATH),
    )


def _clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    return value


def _json_sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return _clean_json_value(obj)


def bars_for_tv(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert OHLCV frame → Lightweight Charts candlestick rows."""
    if df is None or df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            rows.append(
                {
                    "time": int(float(row["ts"])),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return rows


def lock_markers(lock_ts: float | None, *, text: str = "LOCK") -> list[dict[str, Any]]:
    if lock_ts is None:
        return []
    try:
        t = int(float(lock_ts))
        t = t - (t % 60)
    except Exception:
        return []
    return [
        {
            "time": t,
            "position": "aboveBar",
            "color": "#bb3e03",
            "shape": "arrowDown",
            "text": text,
        }
    ]


def render_tv_kline(
    symbol: str,
    df_candle: pd.DataFrame,
    *,
    chart_date: str = "",
    color_mode: str = "us",
    theme_mode: str = "light",
    markers: list[dict[str, Any]] | None = None,
    key: str | None = None,
) -> Any:
    """Render production ``futu_kline`` (TradingView Lightweight Charts) read-only."""
    if _FUTU_KLINE is None:
        st.error(
            f"未找到 TradingView 组件目录：`{FUTU_KLINE_PATH}`。"
            "请确认 production/baseline/DAO/components/futu_kline 存在。"
        )
        return None
    if df_candle is None or df_candle.empty:
        st.info(f"{symbol}: 暂无 1m OHLCV（等 tape/scanner 积累）")
        return None

    if color_mode == "cn":
        colors = {"up": "#EF553B", "down": "#00CC96"}
    else:
        colors = {"up": "#00CC96", "down": "#EF553B"}
    theme = (
        {
            "background": "#f7f9fc",
            "panel": "#ffffff",
            "text": "#1f2937",
            "muted": "#667085",
            "grid": "rgba(31,41,55,0.10)",
            "border": "rgba(31,41,55,0.14)",
        }
        if theme_mode == "light"
        else {
            "background": "#0f141d",
            "panel": "#0f141d",
            "text": "#c9d2e3",
            "muted": "rgba(232,238,248,0.74)",
            "grid": "rgba(255,255,255,0.06)",
            "border": "rgba(255,255,255,0.10)",
        }
    )
    return _FUTU_KLINE(
        symbol=str(symbol or "").upper(),
        bars=_json_sanitize(bars_for_tv(df_candle)),
        quotes={},
        position=None,
        readOnly=True,
        chartDate=str(chart_date or ""),
        colors=colors,
        theme=theme,
        websocketUrl="",
        embedLiveLeaders=False,
        initialLeaders=None,
        defaultMoneyness="ATM",
        ivMarkers=_json_sanitize(markers or []),
        orderSession={},
        key=key or f"maga7_tv_{symbol}_{chart_date or 'live'}_{theme_mode}",
        default=None,
    )
