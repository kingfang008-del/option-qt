#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QQQ_BTC focused dashboard.

This page is intentionally read-only. It borrows the useful ideas from the
legacy New_Pro dashboard (Redis stream probes, live OMS projection, topology
view, shadow fill audit), but keeps the qqq_btc-specific contract visible:

* one symbol: QQQ
* one fill model: qqq_btc.qqq.config.FILL_MODEL
* replay/live parity gates: G0 -> G3
* three live processes: FCS, Signal, OMS
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional visual dependency.
    go = None

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.live.fill_audit_writer import default_audit_path
from qqq_btc.qqq import config as qcfg
from qqq_btc.tools.parity_audit import audit_exit_reasons, audit_fill

try:
    import redis
except Exception:  # pragma: no cover - dashboard can render without redis pkg.
    redis = None


STREAM_FUSED_MARKET = "fused_market_stream"
STREAM_INFERENCE = "unified_inference_stream"
STREAM_ORCH_SIGNAL = "orch_trade_signals"
STREAM_TRADE_LOG = "trade_log_stream"
HASH_OPTION_SNAPSHOT = "live_option_snapshot"
OMS_LIVE_POSITIONS = "oms:live_positions"
OMS_LEDGER = "meta:oms_ledger"

DEFAULT_REDIS = {
    "host": os.environ.get("REDIS_HOST", "localhost"),
    "port": int(os.environ.get("REDIS_PORT", "6379")),
    "db": int(os.environ.get("REDIS_DB", "0")),
}
DEFAULT_SYMBOLS = [
    s.strip().upper()
    for s in os.environ.get("QQQ_BTC_DASH_SYMBOLS", "QQQ,NVDA").split(",")
    if s.strip()
][:5]


@dataclass
class StreamProbe:
    name: str
    length: int = 0
    last_id: str = ""
    last_age_sec: float | None = None
    groups: list[dict[str, Any]] | None = None
    latest_payload: dict[str, Any] | None = None
    error: str = ""


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _jsonish_load(value: Any) -> Any:
    value = _decode(value)
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        try:
            return pickle.loads(value)
        except Exception:
            return value
    s = value.strip()
    if not s:
        return s
    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _coerce_payload(fields: Mapping[Any, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        key = str(_decode(k))
        val = _jsonish_load(v)
        out[key] = val
    for key in ("payload", "data", "json", "message", "frame"):
        val = out.get(key)
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                return json.loads(val)
            except Exception:
                pass
    return out


def _stream_id_age(stream_id: str) -> float | None:
    try:
        ms = int(str(stream_id).split("-")[0])
        return max(0.0, time.time() - ms / 1000.0)
    except Exception:
        return None


def _age_label(age: float | None) -> str:
    if age is None:
        return "-"
    if age < 60:
        return f"{age:.1f}s"
    if age < 3600:
        return f"{age / 60:.1f}m"
    return f"{age / 3600:.1f}h"


def _status_from_age(age: float | None, warn: float = 15.0, crit: float = 90.0) -> str:
    if age is None:
        return "off"
    if age > crit:
        return "crit"
    if age > warn:
        return "warn"
    return "ok"


def _status_color(status: str) -> str:
    return {
        "ok": "#16a34a",
        "warn": "#d97706",
        "crit": "#dc2626",
        "off": "#6b7280",
        "info": "#2563eb",
        "pending": "#7c3aed",
    }.get(status, "#6b7280")


@st.cache_resource(show_spinner=False)
def redis_client(host: str, port: int, db: int):
    if redis is None:
        return None
    return redis.Redis(host=host, port=port, db=db, decode_responses=False)


def probe_stream(r, name: str, count: int = 1) -> StreamProbe:
    probe = StreamProbe(name=name)
    if r is None:
        probe.error = "redis package/client unavailable"
        return probe
    try:
        info = r.xinfo_stream(name)
        probe.length = int(info.get(b"length", info.get("length", 0)) or 0)
        last = info.get(b"last-generated-id", info.get("last-generated-id", b""))
        probe.last_id = str(_decode(last) or "")
        probe.last_age_sec = _stream_id_age(probe.last_id)
    except Exception as exc:
        probe.error = str(exc)
        return probe
    try:
        groups = []
        for g in r.xinfo_groups(name):
            groups.append({str(_decode(k)): _decode(v) for k, v in g.items()})
        probe.groups = groups
    except Exception:
        probe.groups = []
    try:
        rows = r.xrevrange(name, count=count)
        if rows:
            probe.latest_payload = _coerce_payload(rows[0][1])
    except Exception:
        probe.latest_payload = None
    return probe


def read_hash_json(r, key: str) -> dict[str, Any]:
    if r is None:
        return {}
    try:
        raw = r.hgetall(key) or {}
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        out[str(_decode(k))] = _jsonish_load(v)
    return out


def _latest_alpha_item(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload.get("items"), list) and payload["items"]:
        item = payload["items"][0]
        return item if isinstance(item, dict) else {}
    for key in ("alpha", "net_edge", "call_edge", "put_edge"):
        if key in payload:
            return payload
    return {}


def latest_alpha_items(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    out: dict[str, dict[str, Any]] = {}
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", "") or "").upper()
            if sym:
                out[sym] = item
    if not out:
        item = _latest_alpha_item(payload)
        sym = str(item.get("symbol", qcfg.SYMBOL) or qcfg.SYMBOL).upper()
        if item:
            out[sym] = item
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_edge(value: Any) -> str:
    return f"{_safe_float(value):+.4f}"


def _fmt_pct(value: Any) -> str:
    v = _safe_float(value)
    return f"{v:+.1%}" if abs(v) < 5 else f"{v:+.2f}"


def _edge_status(edge: float, q10: float, has_position: bool) -> str:
    if has_position:
        return "ok"
    if edge >= qcfg.REPLAY.entry_threshold and q10 > 0:
        return "ok"
    if abs(edge) >= qcfg.REPLAY.entry_threshold * 0.6:
        return "warn"
    return "off"


def _flatten_position_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            return json.loads(payload)
        except Exception:
            return {"raw": payload}
    return {"raw": payload}


def fetch_positions(r) -> pd.DataFrame:
    raw = read_hash_json(r, OMS_LIVE_POSITIONS)
    rows = []
    for sym, payload in raw.items():
        if sym == "____SYSTEM_CASH____":
            continue
        item = _flatten_position_payload(payload)
        rows.append(
            {
                "symbol": sym,
                "qty": item.get("qty", item.get("position", item.get("contracts", ""))),
                "avg_px": item.get("avg_px", item.get("entry_price", item.get("entry_px", ""))),
                "mark": item.get("mark", item.get("current_price", item.get("price", ""))),
                "roi": item.get("roi", item.get("unrealized_roi", "")),
                "reason": item.get("reason", item.get("entry_reason", "")),
                "updated_at": item.get("updated_at", item.get("ts", "")),
            }
        )
    return pd.DataFrame(rows)


def read_audit_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()


def symbol_audit_stats(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    if df.empty or "symbol" not in df.columns:
        return {"rows": 0}
    s = symbol.upper()
    mask = df["symbol"].astype(str).str.upper().str.contains(s, regex=False, na=False)
    sub = df[mask].copy()
    if sub.empty:
        return {"rows": 0}
    if "fill_spread_frac" in sub.columns:
        frac = pd.to_numeric(sub["fill_spread_frac"], errors="coerce").dropna()
    else:
        frac = pd.Series(dtype=float)
    if "action" in sub.columns:
        closes = sub[sub["action"].astype(str).str.upper() == "CLOSE"]
    else:
        closes = pd.DataFrame()
    latest = sub.tail(1).to_dict("records")[0]
    return {
        "rows": int(len(sub)),
        "median_frac": float(frac.median()) if len(frac) else None,
        "last_action": latest.get("action", "-"),
        "last_delta": latest.get("delta_frac", ""),
        "n_close": int(len(closes)),
        "last_exit": latest.get("exit_reason", ""),
    }


def position_for_symbol(positions: pd.DataFrame, symbol: str) -> dict[str, Any]:
    if positions.empty or "symbol" not in positions.columns:
        return {}
    s = symbol.upper()
    mask = positions["symbol"].astype(str).str.upper().str.contains(s, regex=False, na=False)
    sub = positions[mask]
    if sub.empty:
        return {}
    return sub.tail(1).to_dict("records")[0]
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_fill_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"error": f"missing {path}"}
    try:
        return audit_fill(path, target_frac=qcfg.FILL_MODEL.entry_frac)
    except Exception as exc:
        return {"error": str(exc)}


def build_exit_report(path: Path, replay_path: Path | None) -> dict[str, Any]:
    if not path.exists():
        return {"error": f"missing {path}"}
    try:
        return audit_exit_reasons(path, replay_path if replay_path and replay_path.exists() else None)
    except Exception as exc:
        return {"error": str(exc)}


def metric_card(label: str, value: Any, status: str = "info", help_text: str = "") -> None:
    color = _status_color(status)
    st.markdown(
        f"""
        <div class="qbd-card">
          <div class="qbd-card-label">{label}</div>
          <div class="qbd-card-value" style="color:{color};">{value}</div>
          <div class="qbd-card-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_live_topology(probes: dict[str, StreamProbe], qqq_live: bool):
    if go is None:
        return None
    nodes = {
        "IBKR": (0.0, 1.0, "IBKR\n行情/订单"),
        "FCS": (1.4, 1.0, "FCS\nfeature"),
        "SE": (2.8, 1.0, "Signal\nqqq_btc"),
        "OMS": (4.2, 1.0, "OMS\nlegacy host"),
        "Audit": (5.6, 1.0, "Shadow\nfill/exits"),
        "Common": (2.8, 0.15, "common\nfill/replay/rails"),
    }
    edges = [
        ("IBKR", "FCS", STREAM_FUSED_MARKET),
        ("FCS", "SE", STREAM_INFERENCE),
        ("SE", "OMS", STREAM_ORCH_SIGNAL),
        ("OMS", "Audit", "fill_audit.csv"),
        ("Common", "SE", "same state"),
        ("Common", "OMS", "0.775 limit"),
    ]
    fig = go.Figure()
    for src, dst, label in edges:
        x0, y0, _ = nodes[src]
        x1, y1, _ = nodes[dst]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#94a3b8", width=2, dash="dot" if src == "Common" else "solid"),
                hovertext=label,
                hoverinfo="text",
                showlegend=False,
            )
        )
    statuses = {
        "IBKR": _status_from_age(probes[STREAM_FUSED_MARKET].last_age_sec),
        "FCS": _status_from_age(probes[STREAM_INFERENCE].last_age_sec),
        "SE": _status_from_age(probes[STREAM_ORCH_SIGNAL].last_age_sec),
        "OMS": _status_from_age(probes[STREAM_TRADE_LOG].last_age_sec, warn=60, crit=300),
        "Audit": "ok" if qqq_live else "pending",
        "Common": "ok",
    }
    fig.add_trace(
        go.Scatter(
            x=[nodes[k][0] for k in nodes],
            y=[nodes[k][1] for k in nodes],
            mode="markers+text",
            marker=dict(
                size=58,
                color=[_status_color(statuses[k]) for k in nodes],
                line=dict(color="white", width=2),
            ),
            text=[nodes[k][2] for k in nodes],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=[f"{k}: {statuses[k]}" for k in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=24, b=8),
        xaxis=dict(visible=False, range=[-0.4, 6.0]),
        yaxis=dict(visible=False, range=[-0.2, 1.35]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.1rem; max-width: 1500px; }
        .qbd-title { font-size: 1.55rem; font-weight: 700; margin-bottom: .15rem; }
        .qbd-sub { color: #64748b; font-size: .9rem; margin-bottom: .8rem; }
        .qbd-section-title { font-size: 1.05rem; font-weight: 700; margin: .9rem 0 .45rem; }
        .qbd-card {
          border: 1px solid #e2e8f0; border-radius: 8px; padding: .75rem .85rem;
          background: #ffffff; min-height: 88px;
        }
        .qbd-card-label { color: #64748b; font-size: .78rem; text-transform: uppercase; }
        .qbd-card-value { font-size: 1.35rem; font-weight: 700; line-height: 1.55rem; margin-top: .2rem; }
        .qbd-card-help { color: #64748b; font-size: .76rem; margin-top: .25rem; }
        .qbd-pill {
          display: inline-block; padding: .16rem .45rem; border-radius: 999px;
          font-size: .75rem; font-weight: 600; margin-right: .3rem; border: 1px solid #e2e8f0;
        }
        .qbd-symbol-card {
          border: 1px solid #dbe3ee; border-radius: 8px; padding: .75rem;
          background: #ffffff; min-height: 230px;
        }
        .qbd-symbol-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.45rem; }
        .qbd-symbol-name { font-size:1.2rem; font-weight:800; color:#0f172a; }
        .qbd-status-dot { width:.65rem; height:.65rem; border-radius:999px; display:inline-block; margin-right:.35rem; }
        .qbd-small { color:#64748b; font-size:.75rem; line-height:1.1rem; }
        .qbd-edge { font-size:1.5rem; line-height:1.8rem; font-weight:800; }
        .qbd-kv { display:grid; grid-template-columns: 1fr 1fr; gap:.35rem .6rem; margin-top:.55rem; }
        .qbd-kv div { border-top:1px solid #eef2f7; padding-top:.32rem; min-width:0; }
        .qbd-kv span { display:block; color:#64748b; font-size:.7rem; }
        .qbd-kv strong { display:block; color:#0f172a; font-size:.84rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .qbd-mini-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:.55rem; margin:.65rem 0 .4rem; }
        .qbd-code {
          background:#0f172a; color:#e2e8f0; padding:.8rem; border-radius:8px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          font-size:.82rem; white-space:pre-wrap;
        }
        div[data-testid="stMetric"] { border: 1px solid #e2e8f0; border-radius: 8px; padding: .45rem .6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[dict[str, Any], Path, Path | None, bool, int, list[str]]:
    st.sidebar.header("QQQ_BTC Dash")
    raw_symbols = st.sidebar.text_input("Symbols", value=",".join(DEFAULT_SYMBOLS))
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()][:5]
    if not symbols:
        symbols = [qcfg.SYMBOL]
    host = st.sidebar.text_input("Redis host", value=DEFAULT_REDIS["host"])
    port = st.sidebar.number_input("Redis port", value=DEFAULT_REDIS["port"], min_value=1, max_value=65535)
    db = st.sidebar.number_input("Redis DB", value=DEFAULT_REDIS["db"], min_value=0, max_value=15)
    audit_path = Path(
        st.sidebar.text_input("fill_audit.csv", value=str(default_audit_path()))
    ).expanduser()
    replay_raw = st.sidebar.text_input("replay trades CSV", value="")
    replay_path = Path(replay_raw).expanduser() if replay_raw.strip() else None
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    refresh_sec = st.sidebar.slider("Refresh seconds", min_value=2, max_value=60, value=5)
    st.sidebar.caption(f"RUN_MODE={os.environ.get('RUN_MODE', 'REALTIME_DRY')}")
    st.sidebar.caption(f"QQQ_BTC_LIVE={os.environ.get('QQQ_BTC_LIVE', '0')}")
    if auto_refresh:
        st.sidebar.caption(f"Next refresh in ~{refresh_sec}s")
    return (
        {"host": host, "port": int(port), "db": int(db)},
        audit_path,
        replay_path,
        auto_refresh,
        int(refresh_sec),
        symbols,
    )


def render_header(redis_cfg: dict[str, Any], qqq_live: bool, symbols: list[str]) -> None:
    st.markdown('<div class="qbd-title">Quant Live Board</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qbd-sub">最多 5 个标的一屏观察: 每个标的独立看 edge、持仓、fill 审计和链路新鲜度；'
        "QQQ_BTC 的 fill_model / replay_session / live OMS 仍共用同一套假设。</div>",
        unsafe_allow_html=True,
    )
    live_status = "ok" if qqq_live else "pending"
    st.markdown(
        f"""
        <span class="qbd-pill" style="color:{_status_color(live_status)};">QQQ_BTC_LIVE={int(qqq_live)}</span>
        <span class="qbd-pill">Redis {redis_cfg["host"]}:{redis_cfg["port"]}/{redis_cfg["db"]}</span>
        <span class="qbd-pill">Symbols {",".join(symbols)}</span>
        <span class="qbd-pill">fill_frac={qcfg.FILL_MODEL.entry_frac:.3f}</span>
        <span class="qbd-pill">entry_threshold={qcfg.REPLAY.entry_threshold:.3f}</span>
        """,
        unsafe_allow_html=True,
    )


def render_global_strip(probes: dict[str, StreamProbe], audit_df: pd.DataFrame) -> None:
    st.markdown('<div class="qbd-section-title">Global Chain</div>', unsafe_allow_html=True)
    rows = [
        ("Market -> FCS", probes[STREAM_FUSED_MARKET], STREAM_FUSED_MARKET),
        ("FCS -> Signal", probes[STREAM_INFERENCE], STREAM_INFERENCE),
        ("Signal -> OMS", probes[STREAM_ORCH_SIGNAL], STREAM_ORCH_SIGNAL),
        ("Trade Log", probes[STREAM_TRADE_LOG], STREAM_TRADE_LOG),
    ]
    html = ['<div class="qbd-mini-grid">']
    for label, probe, stream in rows:
        status = _status_from_age(probe.last_age_sec, warn=60 if stream == STREAM_TRADE_LOG else 15, crit=300 if stream == STREAM_TRADE_LOG else 90)
        html.append(
            f'<div class="qbd-card">'
            f'<div class="qbd-card-label">{label}</div>'
            f'<div class="qbd-card-value" style="color:{_status_color(status)};">{_age_label(probe.last_age_sec)}</div>'
            f'<div class="qbd-card-help">{probe.length} events · {stream}</div>'
            f"</div>"
        )
    html.append(
        f'<div class="qbd-card">'
        f'<div class="qbd-card-label">Shadow Audit</div>'
        f'<div class="qbd-card-value">{len(audit_df)}</div>'
        f'<div class="qbd-card-help">fill_audit rows</div>'
        f"</div>"
    )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_symbol_board(
    symbols: list[str],
    alpha_by_symbol: dict[str, dict[str, Any]],
    positions: pd.DataFrame,
    audit_df: pd.DataFrame,
    probes: dict[str, StreamProbe],
) -> None:
    st.markdown('<div class="qbd-section-title">Symbols</div>', unsafe_allow_html=True)
    cols = st.columns(len(symbols))
    signal_age = probes[STREAM_ORCH_SIGNAL].last_age_sec
    if signal_age is None or signal_age > 90:
        signal_age = probes[STREAM_INFERENCE].last_age_sec
    for col, symbol in zip(cols, symbols):
        item = alpha_by_symbol.get(symbol, {})
        pos = position_for_symbol(positions, symbol)
        aud = symbol_audit_stats(audit_df, symbol)
        edge = _safe_float(item.get("alpha", item.get("net_edge", item.get("call_edge", 0.0))))
        call_edge = _safe_float(item.get("call_edge", item.get("call_net_edge", edge)))
        put_edge = _safe_float(item.get("put_edge", item.get("put_net_edge", 0.0)))
        q10 = _safe_float(item.get("net_edge_q10", item.get("edge_q10", 0.0)))
        leg = str(item.get("chosen_leg", "-") or "-")
        has_position = bool(pos)
        status = _edge_status(edge, q10, has_position)
        status_label = "POSITION" if has_position else ("READY" if status == "ok" else ("WATCH" if status == "warn" else "IDLE"))
        roi = pos.get("roi", "-") if pos else "-"
        qty = pos.get("qty", "-") if pos else "-"
        mark = pos.get("mark", "-") if pos else "-"
        median_frac = aud.get("median_frac")
        median_label = f"{median_frac:.3f}" if isinstance(median_frac, float) else "-"
        last_action = aud.get("last_action", "-") or "-"
        last_exit = aud.get("last_exit", "") or "-"
        opt_data = item.get("opt_data") if isinstance(item.get("opt_data"), dict) else {}
        spread = opt_data.get("spread_pct", "")
        spread_label = f"{_safe_float(spread):.2%}" if spread not in ("", None) else "-"
        with col:
            st.markdown(
                f"""
                <div class="qbd-symbol-card">
                  <div class="qbd-symbol-head">
                    <div class="qbd-symbol-name">{symbol}</div>
                    <div class="qbd-small"><span class="qbd-status-dot" style="background:{_status_color(status)};"></span>{status_label}</div>
                  </div>
                  <div class="qbd-small">net edge</div>
                  <div class="qbd-edge" style="color:{_status_color(status)};">{_fmt_edge(edge)}</div>
                  <div class="qbd-small">q10 {_fmt_edge(q10)} · leg {leg} · signal {_age_label(signal_age)}</div>
                  <div class="qbd-kv">
                    <div><span>CALL</span><strong>{_fmt_edge(call_edge)}</strong></div>
                    <div><span>PUT</span><strong>{_fmt_edge(put_edge)}</strong></div>
                    <div><span>Position</span><strong>{qty}</strong></div>
                    <div><span>ROI</span><strong>{_fmt_pct(roi) if roi != "-" else "-"}</strong></div>
                    <div><span>Mark</span><strong>{mark}</strong></div>
                    <div><span>Spread</span><strong>{spread_label}</strong></div>
                    <div><span>Fill median</span><strong>{median_label}</strong></div>
                    <div><span>Last audit</span><strong>{last_action}</strong></div>
                  </div>
                  <div class="qbd-small" style="margin-top:.45rem;">{last_exit}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_detail_sections(
    symbols: list[str],
    probes: dict[str, StreamProbe],
    positions: pd.DataFrame,
    ledger: dict[str, Any],
    audit_path: Path,
    replay_path: Path | None,
) -> None:
    left, right = st.columns([1.15, 1.0])
    with left:
        st.markdown('<div class="qbd-section-title">Live Topology</div>', unsafe_allow_html=True)
        topo = draw_live_topology(probes, os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in {"1", "true", "yes", "on"})
        if topo is not None:
            st.plotly_chart(topo, use_container_width=True)
        else:
            st.info("Plotly is not installed in this Python env; topology is summarized above.")
    with right:
        st.markdown('<div class="qbd-section-title">Positions</div>', unsafe_allow_html=True)
        if positions.empty:
            st.info("No live position projection in oms:live_positions.")
        else:
            st.dataframe(positions, use_container_width=True, hide_index=True)
        if ledger:
            with st.expander("OMS ledger", expanded=False):
                st.json(ledger, expanded=False)

    lower_left, lower_right = st.columns([1.2, 1.0])
    with lower_left:
        st.markdown('<div class="qbd-section-title">Streams</div>', unsafe_allow_html=True)
        rows = []
        for name, probe in probes.items():
            rows.append(
                {
                    "stream": name,
                    "length": probe.length,
                    "last_id": probe.last_id,
                    "age": _age_label(probe.last_age_sec),
                    "status": _status_from_age(probe.last_age_sec),
                    "groups": len(probe.groups or []),
                    "error": probe.error,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with lower_right:
        st.markdown('<div class="qbd-section-title">Contract</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    ("symbols", ",".join(symbols)),
                    ("fill_frac", qcfg.FILL_MODEL.entry_frac),
                    ("entry_threshold", qcfg.REPLAY.entry_threshold),
                    ("max_spread_pct", qcfg.REPLAY.max_spread_pct),
                    ("max_trades_per_day", qcfg.REPLAY.max_trades_per_day),
                    ("hard_stop_roi", qcfg.EXIT_RAILS.hard_stop_roi),
                    ("disaster_stop_roi", qcfg.EXIT_RAILS.disaster_stop_roi),
                ],
                columns=["parameter", "value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Shadow audit detail", expanded=False):
        render_audit_tab(audit_path, replay_path)
    with st.expander("Runbook", expanded=False):
        render_ops_tab(audit_path)


def render_live_tab(
    probes: dict[str, StreamProbe],
    alpha_item: dict[str, Any],
    positions: pd.DataFrame,
    ledger: dict[str, Any],
    qqq_live: bool,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Market -> FCS", _age_label(probes[STREAM_FUSED_MARKET].last_age_sec), _status_from_age(probes[STREAM_FUSED_MARKET].last_age_sec), STREAM_FUSED_MARKET)
    with c2:
        metric_card("FCS -> Signal", _age_label(probes[STREAM_INFERENCE].last_age_sec), _status_from_age(probes[STREAM_INFERENCE].last_age_sec), STREAM_INFERENCE)
    with c3:
        metric_card("Signal -> OMS", _age_label(probes[STREAM_ORCH_SIGNAL].last_age_sec), _status_from_age(probes[STREAM_ORCH_SIGNAL].last_age_sec), STREAM_ORCH_SIGNAL)
    with c4:
        metric_card("Trade log", _age_label(probes[STREAM_TRADE_LOG].last_age_sec), _status_from_age(probes[STREAM_TRADE_LOG].last_age_sec, warn=60, crit=300), STREAM_TRADE_LOG)

    topo = draw_live_topology(probes, qqq_live)
    if topo is not None:
        st.plotly_chart(topo, use_container_width=True)
    else:
        st.info("Plotly is not installed in this Python env; topology graph is shown as stream cards above.")

    left, right = st.columns([1.1, 1.0])
    with left:
        st.subheader("Latest Edge")
        edge_cols = st.columns(5)
        edge_cols[0].metric("net_edge", f"{float(alpha_item.get('alpha', alpha_item.get('net_edge', 0)) or 0):.4f}")
        edge_cols[1].metric("call_edge", f"{float(alpha_item.get('call_edge', alpha_item.get('call_net_edge', 0)) or 0):.4f}")
        edge_cols[2].metric("put_edge", f"{float(alpha_item.get('put_edge', alpha_item.get('put_net_edge', 0)) or 0):.4f}")
        edge_cols[3].metric("q10", f"{float(alpha_item.get('net_edge_q10', 0) or 0):.4f}")
        edge_cols[4].metric("leg", str(alpha_item.get("chosen_leg", "-")))
        opt_data = alpha_item.get("opt_data") if isinstance(alpha_item.get("opt_data"), dict) else {}
        if opt_data:
            st.dataframe(pd.DataFrame([opt_data]), use_container_width=True, hide_index=True)
    with right:
        st.subheader("OMS Projection")
        if not positions.empty:
            st.dataframe(positions, use_container_width=True, hide_index=True)
        else:
            st.info("No live position projection in oms:live_positions.")
        if ledger:
            st.caption("meta:oms_ledger")
            st.json(ledger, expanded=False)


def render_contract_tab() -> None:
    st.subheader("Single Source Contract")
    cols = st.columns(4)
    cols[0].metric("Symbol", qcfg.SYMBOL)
    cols[1].metric("Fill frac", f"{qcfg.FILL_MODEL.entry_frac:.3f}")
    cols[2].metric("Commission", f"${qcfg.FILL_MODEL.commission_per_contract:.2f}")
    cols[3].metric("Max positions", qcfg.MAX_POSITIONS)

    replay = qcfg.REPLAY
    rails = qcfg.EXIT_RAILS
    cfg_rows = [
        ("entry_threshold", replay.entry_threshold),
        ("max_spread_pct", replay.max_spread_pct),
        ("cooldown_bars", replay.cooldown_bars),
        ("max_trades_per_day", replay.max_trades_per_day),
        ("daily_loss_stop", replay.daily_loss_stop),
        ("session_entry_end_bar", replay.session_entry_end_bar),
        ("hard_stop_roi", rails.hard_stop_roi),
        ("soft_stop_roi", rails.soft_stop_roi),
        ("trailing_trigger_roi", rails.trailing_trigger_roi),
        ("disaster_stop_roi", rails.disaster_stop_roi),
        ("eod_close_bar_index", rails.eod_close_bar_index),
    ]
    st.dataframe(
        pd.DataFrame(cfg_rows, columns=["parameter", "value"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("阈值表来自 qqq_btc.qqq.config, dashboard 不维护副本。")


def render_audit_tab(audit_path: Path, replay_path: Path | None) -> None:
    st.subheader("Shadow Audit")
    df = read_audit_csv(audit_path)
    fill_report = build_fill_report(audit_path)
    exit_report = build_exit_report(audit_path, replay_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Median fill frac", f"{fill_report.get('median', 0):.3f}" if "median" in fill_report else "-")
    c3.metric("Target", f"{qcfg.FILL_MODEL.entry_frac:.3f}")
    c4.metric("Exit closes", exit_report.get("n_close", "-"))

    left, right = st.columns(2)
    with left:
        st.caption("Fill parity")
        st.json(fill_report, expanded=True)
    with right:
        st.caption("Exit distribution")
        st.json(exit_report, expanded=True)

    if not df.empty:
        st.caption(str(audit_path))
        keep = [c for c in ["ts", "symbol", "action", "side", "fill_px", "bid", "ask", "fill_spread_frac", "delta_frac", "reason", "exit_reason", "mode"] if c in df.columns]
        st.dataframe(df[keep].tail(100).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info(f"No audit CSV found at {audit_path}.")


def render_streams_tab(probes: dict[str, StreamProbe]) -> None:
    rows = []
    for name, probe in probes.items():
        rows.append(
            {
                "stream": name,
                "length": probe.length,
                "last_id": probe.last_id,
                "age": _age_label(probe.last_age_sec),
                "status": _status_from_age(probe.last_age_sec),
                "groups": len(probe.groups or []),
                "error": probe.error,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect latest payload", list(probes.keys()))
    probe = probes[selected]
    col_a, col_b = st.columns([1.0, 1.0])
    with col_a:
        st.caption("Latest payload")
        st.json(probe.latest_payload or {}, expanded=False)
    with col_b:
        st.caption("Consumer groups")
        st.json(probe.groups or [], expanded=False)


def render_ops_tab(audit_path: Path) -> None:
    st.subheader("Runbook")
    st.markdown("Signal / OMS 启动仍走 qqq_btc 工具入口，dashboard 只观察状态。")
    st.markdown(
        f"""
<div class="qbd-code">cd New_Pro/baseline_qqq
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_signal_qqq.py \\
  --checkpoint ~/quant_project/checkpoints_qqq_net_edge_v2/best.pth

QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_exec_qqq.py

python qqq_btc/tools/parity_audit.py fill \\
  --audit-log {audit_path}

streamlit run qqq_btc/dashboard/qqq_btc_dash.py --server.port 8502</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("旧 New_Pro dashboard 可继续用于 legacy 诊断；本页面聚焦 QQQ_BTC 的 G0-G3。")


def main() -> None:
    st.set_page_config(page_title="QQQ_BTC Dash", layout="wide")
    render_css()
    redis_cfg, audit_path, replay_path, auto_refresh, refresh_sec, symbols = render_sidebar()
    qqq_live = os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}
    render_header(redis_cfg, qqq_live, symbols)

    r = redis_client(**redis_cfg)
    stream_names = [STREAM_FUSED_MARKET, STREAM_INFERENCE, STREAM_ORCH_SIGNAL, STREAM_TRADE_LOG]
    probes = {name: probe_stream(r, name) for name in stream_names}
    alpha_by_symbol = latest_alpha_items(probes[STREAM_ORCH_SIGNAL].latest_payload)
    if not alpha_by_symbol:
        alpha_by_symbol = latest_alpha_items(probes[STREAM_INFERENCE].latest_payload)
    positions = fetch_positions(r)
    ledger = read_hash_json(r, OMS_LEDGER)
    audit_df = read_audit_csv(audit_path)

    render_symbol_board(symbols, alpha_by_symbol, positions, audit_df, probes)
    render_global_strip(probes, audit_df)
    render_detail_sections(symbols, probes, positions, ledger, audit_path, replay_path)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
