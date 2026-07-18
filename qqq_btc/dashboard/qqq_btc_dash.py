#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QQQ_BTC focused dashboard.

This page is intentionally read-only. It borrows the useful ideas from the
legacy New_Pro dashboard (Redis stream probes, live OMS projection, topology
view, shadow fill audit), but keeps the qqq_btc-specific contract visible:

* primary symbol: QQQ (sidebar can show up to 5 symbols)
* one fill model: qqq_btc.qqq.config.FILL_MODEL (1DTE family)
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
from datetime import datetime
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

from qqq_btc.dashboard.backfill_board import (
    DEFAULT_EXP as BACKFILL_DEFAULT_EXP,
    DEFAULT_FEATURES_ROOT,
    DEFAULT_FEAT_HISTORY,
    DEFAULT_FROZEN_NORM,
    api_key_status,
    build_pipeline_cmd,
    data_directory_rows,
    discover_backfill_days,
    fix_stock_bar_labels,
    load_lock_report,
    load_pipeline_summary,
    load_warmup_report,
    read_job_state,
    resolve_python,
    run_warmup_check_job,
    scan_stock_bar_labels,
    start_backfill_job,
    stop_backfill_job,
    suggested_commands as backfill_suggested_commands,
    tail_log as backfill_tail_log,
)
from qqq_btc.dashboard.offline_board import (
    OFFLINE_ROOT,
    build_replay_cmd,
    daily_rows_from_summary,
    diagnostic_bundle,
    discover_offline_runs,
    feature_norm_hint,
    headline_rows,
    read_job_state as read_offline_job_state,
    recipe_offline_options,
    start_offline_replay_job,
    stop_offline_replay_job,
    tail_log as offline_tail_log,
)
from qqq_btc.dashboard.live_board import (
    DEFAULT_LIVE_FROZEN,
    build_live_export_cmd,
    default_paths as live_default_paths,
    live_norm_policy,
    read_job_state as read_live_job_state,
    start_deploy_job,
    start_refresh_live_frozen,
    stop_job as stop_live_job,
    suggested_live_upto_date,
    tail_log as live_tail_log,
)
from qqq_btc.dashboard.stream_parity_jobs import (
    DEFAULT_FROZEN_OUT,
    _prev_month,
    build_export_frozen_cmd,
    build_stream_parity_cmd,
    default_paths as stream_default_paths,
    frozen_npz_meta,
    read_job_state as read_stream_job_state,
    resolve_features_bundle,
    start_export_frozen_job,
    start_stream_parity_job,
    stop_job as stop_stream_job,
    tail_log as stream_tail_log,
)
from qqq_btc.dashboard.parity_board import (
    annotate_runs,
    daily_acct_rows,
    discover_stream_runs,
    gate_pass_labels,
    list_strategy_profiles,
    load_catalog,
    offline_headline,
    recipe_card_state,
    tail_text,
    trades_frame_rows,
)
from qqq_btc.dashboard.maga7_board import render_maga7_board
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
    try:
        return pd.read_csv(path)
    except Exception:
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


def render_sidebar() -> tuple[str, dict[str, Any], Path, Path | None, bool, int, list[str]]:
    st.sidebar.header("QQQ_BTC Dash")
    board = st.sidebar.radio(
        "Board",
        options=["Live", "Download", "Offline Replay", "Stream Parity", "Mag7"],
        index=0,
        help=(
            "Live=实时链路；Download=锁约/quote/day_iv（不含归一化）；"
            "Offline Replay=离线日收益与诊断；Stream Parity=三闸门流式对拍；"
            "Mag7=多标的 Rule-A Top2 offline/parity/scanner（非 TFT）"
        ),
    )
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
    auto_default = board == "Live"
    auto_refresh = st.sidebar.toggle("Auto refresh", value=auto_default)
    refresh_sec = st.sidebar.slider("Refresh seconds", min_value=2, max_value=60, value=5)
    st.sidebar.caption(f"RUN_MODE={os.environ.get('RUN_MODE', 'REALTIME_DRY')}")
    st.sidebar.caption(f"QQQ_BTC_LIVE={os.environ.get('QQQ_BTC_LIVE', '0')}")
    if auto_refresh:
        st.sidebar.caption(f"Next refresh in ~{refresh_sec}s")
    return (
        board,
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
        '<div class="qbd-sub">对拍通过后的实盘路径：'
        "<b>刷新 frozen（过去 raw）→ Warmup → FCS 推理 → Signal/OMS</b>。"
        " 实盘无离线金标，但归一化仍用与对拍相同的 frozen 提取。</div>",
        unsafe_allow_html=True,
    )
    live_status = "ok" if qqq_live else "pending"
    st.markdown(
        f"""
        <span class="qbd-pill" style="color:{_status_color(live_status)};">QQQ_BTC_LIVE={int(qqq_live)}</span>
        <span class="qbd-pill">Redis {redis_cfg["host"]}:{redis_cfg["port"]}/{redis_cfg["db"]}</span>
        <span class="qbd-pill">Symbols {",".join(symbols)}</span>
        <span class="qbd-pill">profile={getattr(qcfg, "PROFILE", "qqq")}</span>
        <span class="qbd-pill">fill_frac={qcfg.FILL_MODEL.entry_frac:.3f}</span>
        <span class="qbd-pill">entry_threshold={qcfg.REPLAY.entry_threshold:.3f}</span>
        """,
        unsafe_allow_html=True,
    )


def render_live_deploy_panel() -> None:
    """Live：frozen 刷新 + shadow/dry/live 启动。"""
    st.markdown(
        '<div class="qbd-section-title">Deploy prep（对拍通过 → 实盘）</div>',
        unsafe_allow_html=True,
    )
    policy = live_norm_policy()
    with st.expander("实盘归一化要不要像对拍一样提取 frozen？", expanded=True):
        st.markdown(f"- {policy['same_as_parity']}")
        st.markdown(f"- {policy['no_gold']}")
        st.markdown(f"- {policy['refresh']}")
        st.markdown(f"- {policy['warmup']}")

    paths = live_default_paths()
    profiles = list_strategy_profiles()
    profile_labels = [
        f"{p['profile_id']} · {Path(p['path']).name}" for p in profiles
    ] or ["(no profiles)"]
    # Prefer production / ft56 production if present
    default_idx = 0
    for i, p in enumerate(profiles):
        if "production" in p["profile_id"] or "vx_live" in p["profile_id"]:
            default_idx = i
            break

    c1, c2 = st.columns(2)
    with c1:
        profile_pick = st.selectbox(
            "Strategy profile（与 Offline/Stream 共用）",
            profile_labels,
            index=min(default_idx, max(0, len(profile_labels) - 1)),
            key="live_profile",
        )
        profile_row = profiles[profile_labels.index(profile_pick)] if profiles else None
    with c2:
        python_live = st.text_input("Python", value=resolve_python(), key="live_py")

    frozen_raw = st.text_input(
        "Frozen 源 raw（quote_features_raw）",
        value=paths["frozen_raw_root"],
        key="live_frozen_raw",
    )
    frozen_out = st.text_input(
        "Live frozen .npz（FCS_FROZEN_NORM_PATH）",
        value=paths["frozen_live"],
        key="live_frozen_out",
        help="部署脚本默认读此路径；也可指到对拍导出的 frozen_norm_dash_stream.npz",
    )
    cut_mode = st.radio(
        "冻结截止方式",
        options=["upto_date（日更，推荐实盘）", "upto_month（月冻，对齐月度对拍）"],
        index=0,
        key="live_cut_mode",
        horizontal=True,
    )
    if cut_mode.startswith("upto_date"):
        upto_date = st.text_input(
            "upto-date（不含当日）",
            value=suggested_live_upto_date(),
            key="live_upto_date",
        )
        upto_month = None
    else:
        upto_month = st.text_input(
            "upto-month",
            value=_prev_month(datetime.now().strftime("%Y-%m")),
            key="live_upto_month",
        )
        upto_date = None

    meta = frozen_npz_meta(frozen_out)
    if meta.get("exists"):
        st.success(
            f"frozen ready · dims={meta.get('dims')} frames={meta.get('frames')} "
            f"upto={meta.get('upto_date') or meta.get('upto_month') or '-'} "
            f"mtime={meta.get('mtime')}"
        )
        st.caption(f"source={meta.get('source_dir')}")
    else:
        st.warning(f"缺少 frozen：`{frozen_out}` — 开盘前请先刷新")

    export_cmd = build_live_export_cmd(
        features_raw_root=frozen_raw,
        output=frozen_out,
        upto_date=upto_date,
        upto_month=upto_month,
        python_bin=python_live,
    )
    st.code(" ".join(export_cmd), language="bash")

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("① 刷新 frozen", type="primary", use_container_width=True, key="live_btn_frozen"):
            try:
                state = start_refresh_live_frozen(
                    features_raw_root=frozen_raw,
                    output=frozen_out,
                    upto_date=upto_date,
                    upto_month=upto_month,
                    python_bin=python_live,
                )
                st.success(f"refresh pid={state.get('pid')}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with b2:
        if st.button("② 依赖检查", use_container_width=True, key="live_btn_check"):
            try:
                state = start_deploy_job(
                    mode="check",
                    frozen_norm=frozen_out,
                    strategy_profile=profile_row["abs_path"] if profile_row else None,
                    python_bin=python_live,
                )
                st.success(f"check pid={state.get('pid')}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with b3:
        if st.button("③ Shadow 启动", use_container_width=True, key="live_btn_shadow"):
            if not Path(frozen_out).expanduser().is_file():
                st.error("请先刷新 frozen")
            else:
                try:
                    state = start_deploy_job(
                        mode="shadow",
                        frozen_norm=frozen_out,
                        strategy_profile=profile_row["abs_path"] if profile_row else None,
                        python_bin=python_live,
                    )
                    st.success(f"shadow pid={state.get('pid')}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with b4:
        if st.button("④ Dry 启动", use_container_width=True, key="live_btn_dry"):
            if not Path(frozen_out).expanduser().is_file():
                st.error("请先刷新 frozen")
            else:
                try:
                    state = start_deploy_job(
                        mode="dry",
                        frozen_norm=frozen_out,
                        strategy_profile=profile_row["abs_path"] if profile_row else None,
                        python_bin=python_live,
                    )
                    st.success(f"dry pid={state.get('pid')}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with b5:
        if st.button("⏹ 停止栈", use_container_width=True, key="live_btn_stop"):
            stop_live_job()
            st.rerun()

    with st.expander("⚠ LIVE_TRADE=1 真单（危险）", expanded=False):
        st.warning("仅在对拍通过且确认账户/风控后使用。")
        if st.button("⑤ 真盘启动 LIVE_TRADE=1", key="live_btn_real"):
            if not Path(frozen_out).expanduser().is_file():
                st.error("请先刷新 frozen")
            else:
                try:
                    state = start_deploy_job(
                        mode="live",
                        frozen_norm=frozen_out,
                        strategy_profile=profile_row["abs_path"] if profile_row else None,
                        python_bin=python_live,
                        live_trade=True,
                    )
                    st.success(f"LIVE pid={state.get('pid')}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    job = read_live_job_state()
    jstatus = str(job.get("status") or "idle")
    jtone = (
        "ok"
        if jstatus == "done"
        else ("warn" if jstatus == "running" else ("crit" if jstatus in {"failed", "stopped"} else "pending"))
    )
    st.markdown(
        f"""
        {_status_pill(f"job={jstatus}", jtone)}
        <span class="qbd-pill">kind={job.get('kind', '-')}</span>
        <span class="qbd-pill">pid={job.get('pid', '-')}</span>
        <span class="qbd-pill">frozen={Path(str(job.get('frozen_norm') or frozen_out)).name}</span>
        """,
        unsafe_allow_html=True,
    )
    log = live_tail_log(job.get("log_file"))
    if log:
        with st.expander("Live job log", expanded=jstatus == "running"):
            st.code(log, language="text")

    st.caption(
        f"脚本：`{paths['shadow_script']}` / `{paths['deploy_script']}` · "
        "Shadow=不下单；Dry=REALTIME_DRY；Live=真单。"
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
        selected = st.selectbox("Inspect latest payload", list(probes.keys()))
        probe = probes[selected]
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Latest payload")
            st.json(probe.latest_payload or {}, expanded=False)
        with col_b:
            st.caption("Consumer groups")
            st.json(probe.groups or [], expanded=False)
    with lower_right:
        st.markdown('<div class="qbd-section-title">Contract</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                [
                    ("symbols", ",".join(symbols)),
                    ("profile", getattr(qcfg, "PROFILE", "")),
                    ("fill_frac", qcfg.FILL_MODEL.entry_frac),
                    ("entry_threshold", qcfg.REPLAY.entry_threshold),
                    ("max_spread_pct", qcfg.REPLAY.max_spread_pct),
                    ("max_trades_per_day", qcfg.REPLAY.max_trades_per_day),
                    ("session_entry_end_bar", qcfg.REPLAY.session_entry_end_bar),
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
        keep = [
            c
            for c in [
                "ts",
                "symbol",
                "action",
                "side",
                "fill_px",
                "bid",
                "ask",
                "fill_spread_frac",
                "delta_frac",
                "reason",
                "exit_reason",
                "mode",
                "leg",
            ]
            if c in df.columns
        ]
        st.dataframe(df[keep].tail(100).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info(f"No audit CSV found at {audit_path}.")


def render_ops_tab(audit_path: Path) -> None:
    st.subheader("Runbook")
    st.markdown("Signal / OMS 启动仍走 qqq_btc 工具入口，dashboard 只观察状态。")
    st.markdown(
        f"""
<div class="qbd-code"># from repo root
python qqq_btc/tools/run_dashboard_qqq.py

QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_signal_qqq.py \\
  --checkpoint ~/quant_project/checkpoints_qqq_net_edge_v2/best.pth

QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_exec_qqq.py

python qqq_btc/tools/parity_audit.py fill --audit-log {audit_path}
python qqq_btc/tools/parity_audit.py exits --audit-log {audit_path}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("旧 New_Pro dashboard 可继续用于 legacy 诊断；本页面聚焦 QQQ_BTC 的 G0-G3。")


def _fmt_pct_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}%"


def _status_pill(label: str, status: str) -> str:
    return (
        f'<span class="qbd-pill" style="color:{_status_color(status)};">'
        f"{label}</span>"
    )


def render_parity_board() -> None:
    st.markdown('<div class="qbd-title">Stream Parity</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qbd-sub">流式三闸门对拍（最后一步）。流程：'
        "<b>① 从离线 quote_features_raw 导出 frozen</b> → "
        "<b>② 触发 stream 脚本</b> → ③ 看 Gate1/2/3 结果。"
        " 与 Offline Replay 共用 <code>strategy_profiles</code>。</div>",
        unsafe_allow_html=True,
    )

    recipes = load_catalog()
    paths = stream_default_paths()
    recipe_runnable = [r for r in recipes if r.stream_script or r.stream_cmd]
    recipe_labels = [
        f"{r.recipe_id} · {r.title}" for r in recipe_runnable
    ] or ["(no stream recipe)"]

    # ---- ① Export frozen + ② Trigger ----
    st.markdown(
        '<div class="qbd-section-title">① 导出 frozen（从离线 raw）</div>',
        unsafe_allow_html=True,
    )
    pr1, pr2 = st.columns(2)
    with pr1:
        run_recipe_label = st.selectbox(
            "Stream recipe",
            recipe_labels,
            index=0,
            key="stream_recipe_pick",
        )
        run_recipe = next(
            (r for r in recipe_runnable if run_recipe_label.startswith(r.recipe_id + " ·")),
            recipe_runnable[0] if recipe_runnable else None,
        )
    with pr2:
        python_bin_sp = st.text_input("Python", value=resolve_python(), key="stream_py")

    features_root_sp = st.text_input(
        "Gate 金标 features root（Offline 生成的 raw/test）",
        value=paths["features_root"],
        key="stream_feat_root",
    )
    frozen_raw_root = st.text_input(
        "Frozen 导出源（quote_features_raw，需含 upto-month）",
        value=paths["frozen_raw_root"],
        key="stream_frozen_raw_root",
        help="通常用长历史 ~/train_data/quote_features_raw；与 Gate 金标可不同",
    )
    parity_month = st.text_input(
        "Parity month（Gate 金标月）",
        value="2026-07",
        key="stream_parity_month",
    )
    upto_month = st.text_input(
        "Frozen upto-month（统计截止月，通常=对拍月的前一月）",
        value=_prev_month(parity_month.strip() or "2026-07"),
        key="stream_upto_month",
    )
    frozen_out = st.text_input(
        "Frozen .npz 输出路径",
        value=str(DEFAULT_FROZEN_OUT),
        key="stream_frozen_out",
    )
    bundle = resolve_features_bundle(
        features_root_sp, month=parity_month.strip() or "2026-07"
    )
    st.caption(
        f"Gate raw: `{bundle['offline_raw']}` ({'OK' if bundle['raw_exists'] else 'MISSING'}) · "
        f"Gate norm: `{bundle['offline_norm']}` ({'OK' if bundle['norm_exists'] else 'MISSING'})"
    )
    meta = frozen_npz_meta(frozen_out)
    if meta.get("exists"):
        st.success(
            f"已有 frozen：dims={meta.get('dims')} frames={meta.get('frames')} "
            f"upto={meta.get('upto_month') or meta.get('upto_date') or '-'} "
            f"mtime={meta.get('mtime')}"
        )
        st.caption(f"source={meta.get('source_dir')}")
    else:
        st.warning(f"尚未导出 frozen：`{frozen_out}`")

    slow_cfg = None
    if run_recipe and run_recipe.strategy_profile:
        prof = json.loads(run_recipe.strategy_profile.read_text(encoding="utf-8"))
        slow_cfg = (prof.get("features") or {}).get("slow_feature_config")

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("▶ 导出 frozen 参数", type="primary", use_container_width=True, key="btn_export_frozen"):
            try:
                state = start_export_frozen_job(
                    features_raw_root=frozen_raw_root,
                    output=frozen_out,
                    upto_month=upto_month.strip(),
                    slow_config=slow_cfg,
                    python_bin=python_bin_sp,
                )
                st.success(f"export started pid={state.get('pid')}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with ex2:
        if st.button("⏹ 停止任务", use_container_width=True, key="btn_stop_stream"):
            stop_stream_job()
            st.rerun()
    with ex3:
        if st.button("刷新状态", use_container_width=True, key="btn_refresh_stream"):
            st.rerun()

    export_cmd = build_export_frozen_cmd(
        features_raw_root=frozen_raw_root,
        output=frozen_out,
        upto_month=upto_month.strip(),
        slow_config=slow_cfg,
        python_bin=python_bin_sp,
    )
    st.code(" ".join(export_cmd), language="bash")

    st.markdown(
        '<div class="qbd-section-title">② 触发流式对拍</div>',
        unsafe_allow_html=True,
    )
    days_default = "2026-07-01 2026-07-02 2026-07-06 2026-07-07 2026-07-08 2026-07-09"
    days_sp = st.text_input("DAYS（空格分隔）", value=days_default, key="stream_days")
    out_name = st.text_input(
        "Result dir name（under qqq_btc/results/）",
        value=f"{(run_recipe.recipe_id if run_recipe else 'stream')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        key="stream_out_name",
    )
    out_dir = _REPO / "qqq_btc" / "results" / out_name.strip()

    tr1, tr2 = st.columns(2)
    with tr1:
        if st.button("▶ 启动 Stream Parity", type="primary", use_container_width=True, key="btn_start_stream"):
            if not run_recipe:
                st.error("无可用 stream recipe")
            elif not Path(frozen_out).expanduser().is_file():
                st.error("请先导出 frozen .npz")
            elif not bundle["raw_exists"]:
                st.error(f"缺少离线 raw 金标: {bundle['offline_raw']}")
            else:
                try:
                    state = start_stream_parity_job(
                        recipe=run_recipe,
                        frozen_norm=frozen_out,
                        features_root=features_root_sp,
                        days=days_sp,
                        out_dir=out_dir,
                        python_bin=python_bin_sp,
                    )
                    st.success(f"stream started pid={state.get('pid')} → {out_dir}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with tr2:
        if run_recipe:
            try:
                scmd, senv = build_stream_parity_cmd(
                    recipe=run_recipe,
                    frozen_norm=frozen_out,
                    features_root=features_root_sp,
                    days=days_sp,
                    out_dir=out_dir,
                    python_bin=python_bin_sp,
                )
                st.caption("等价启动（含 FROZEN_NORM / HONEST_FEAT_ROOT / OFFLINE_*）")
                st.code(
                    " ".join(f'{k}="{v}"' for k, v in senv.items() if k in {
                        "FROZEN_NORM", "HONEST_FEAT_ROOT", "OFFLINE_RAW", "OFFLINE_NORM",
                        "QQQ_BTC_STRATEGY_PROFILE", "DAYS", "HONEST_OUT_DIR",
                    })
                    + "\n"
                    + " ".join(scmd),
                    language="bash",
                )
            except Exception as exc:
                st.warning(str(exc))

    job = read_stream_job_state()
    jstatus = str(job.get("status") or "idle")
    jtone = (
        "ok"
        if jstatus == "done"
        else ("warn" if jstatus == "running" else ("crit" if jstatus in {"failed", "stopped"} else "pending"))
    )
    st.markdown(
        f"""
        {_status_pill(f"job={jstatus}", jtone)}
        <span class="qbd-pill">kind={job.get('kind', '-')}</span>
        <span class="qbd-pill">pid={job.get('pid', '-')}</span>
        """,
        unsafe_allow_html=True,
    )
    slog = stream_tail_log(job.get("log_file"))
    if slog:
        with st.expander("Job log", expanded=jstatus == "running"):
            st.code(slog, language="text")

    # ---- ③ Results ----
    st.markdown(
        '<div class="qbd-section-title">③ 对拍结果</div>',
        unsafe_allow_html=True,
    )
    runs = annotate_runs(discover_stream_runs(limit=100), recipes)
    cards = [recipe_card_state(r, runs) for r in recipes]

    cols = st.columns(min(4, max(1, len(cards))))
    for col, card in zip(cols, cards):
        status_raw = str(card.get("parity_status") or "UNKNOWN")
        status = (
            "ok"
            if status_raw == "PASS"
            else ("warn" if status_raw in {"UNGATED", "NO_RUN"} else "crit")
        )
        with col:
            st.markdown(
                f"""
                <div class="qbd-card">
                  <div class="qbd-card-label">{card["recipe_id"]}</div>
                  <div class="qbd-card-value" style="font-size:1.0rem;">{card["title"]}</div>
                  <div class="qbd-card-help">
                    {_status_pill(status_raw, status)}
                    acct {_fmt_pct_num(card.get("acct25_pct"))}
                    · Δ {_fmt_pct_num(card.get("delta_pp"))} vs offline
                    <br/>latest: {card.get("latest_run") or "-"}
                    <br/>profile: {card.get("profile_id") or "-"}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    recipe_options = ["(all discovered runs)"] + [f"{r.recipe_id} · {r.title}" for r in recipes]
    selected_recipe_label = st.selectbox("Filter by recipe", recipe_options, index=0)
    selected_recipe_id = ""
    if selected_recipe_label != "(all discovered runs)":
        selected_recipe_id = selected_recipe_label.split(" · ", 1)[0]

    filtered = [r for r in runs if not selected_recipe_id or r.recipe_id == selected_recipe_id]
    if not filtered:
        st.warning("没有匹配的流式对拍结果目录（需含 manifest.json 或 stream_summary_paired.json）。")
        return

    run_labels = []
    for run in filtered:
        acct = f"{run.acct25 * 100:+.2f}%" if run.acct25 is not None else "-"
        run_labels.append(
            f"{run.name} | {run.parity_status} | {acct} | profile={run.profile_id or '-'}"
        )
    chosen_label = st.selectbox("Stream run", run_labels, index=0)
    run = filtered[run_labels.index(chosen_label)]
    recipe = next((r for r in recipes if r.recipe_id == run.recipe_id), None)
    if recipe is None and selected_recipe_id:
        recipe = next((r for r in recipes if r.recipe_id == selected_recipe_id), None)

    gates = gate_pass_labels(run.gates)
    gate_status = "ok" if all(v == "PASS" for v in gates.values()) and gates else (
        "warn" if run.parity_status == "UNGATED" else "crit" if gates else "pending"
    )
    offline = offline_headline(recipe.offline_result) if recipe else {}
    baseline_acct = None
    if recipe and recipe.baseline_acct25_pct is not None:
        baseline_acct = float(recipe.baseline_acct25_pct)
    elif offline.get("acct25_pct") is not None:
        baseline_acct = float(offline["acct25_pct"])
    delta_pp = None
    if run.acct25 is not None and baseline_acct is not None:
        delta_pp = run.acct25 * 100.0 - baseline_acct

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("parity_status", run.parity_status)
    m2.metric("acct@25%", _fmt_pct_num(run.acct25 * 100 if run.acct25 is not None else None))
    m3.metric("trades", run.trades if run.trades is not None else "-")
    m4.metric("Δ vs offline", _fmt_pct_num(delta_pp))
    m5.metric("gates", " / ".join(f"{k}:{v}" for k, v in gates.items()) or "-")

    st.markdown(
        f"""
        {_status_pill(f"gates={gate_status}", gate_status)}
        <span class="qbd-pill">dir={run.path}</span>
        <span class="qbd-pill">profile={run.profile_id or '-'}</span>
        <span class="qbd-pill">sha={(run.profile_sha[:12] + '…') if run.profile_sha else '-'}</span>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_logic, tab_trades, tab_artifacts, tab_profiles = st.tabs(
        ["Overview", "Logic / Config", "Trades", "Artifacts / Logs", "Profiles"]
    )

    with tab_overview:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.markdown('<div class="qbd-section-title">Run identity</div>', unsafe_allow_html=True)
            identity_rows = [
                ("result_dir", str(run.path)),
                ("recipe_id", run.recipe_id or "-"),
                ("strategy_profile_id", run.profile_id or "-"),
                ("strategy_profile_sha256", run.profile_sha or "-"),
                ("checkpoint", run.summary.get("checkpoint") or run.manifest.get("checkpoint") or "-"),
                ("put_gate", run.summary.get("put_gate") or run.manifest.get("put_gate") or "-"),
                ("tick_exits", run.manifest.get("tick_exits") or "-"),
                ("rule_profile_selector", run.manifest.get("rule_profile_selector") or "-"),
                ("days", ", ".join(run.summary.get("days") or [])),
                ("git", json.dumps(run.manifest.get("git") or {}, ensure_ascii=False)),
            ]
            st.dataframe(
                pd.DataFrame(identity_rows, columns=["field", "value"]),
                use_container_width=True,
                hide_index=True,
            )
            daily = daily_acct_rows(run.summary)
            if daily:
                st.markdown('<div class="qbd-section-title">Daily acct@25%</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(daily), use_container_width=True, hide_index=True)
        with right:
            st.markdown('<div class="qbd-section-title">Baseline / Commands</div>', unsafe_allow_html=True)
            if recipe:
                st.write(
                    {
                        "title": recipe.title,
                        "notes": recipe.notes,
                        "baseline_acct25_pct": baseline_acct,
                        "baseline_trades": recipe.baseline_trades or offline.get("trades"),
                        "offline_result": str(recipe.offline_result) if recipe.offline_result else "",
                        "offline_headline": offline,
                        "docs": list(recipe.docs),
                    }
                )
                if recipe.stream_cmd:
                    st.caption("Stream command")
                    st.code(recipe.stream_cmd, language="bash")
                if recipe.offline_cmd:
                    st.caption("Offline command")
                    st.code(recipe.offline_cmd, language="bash")
            else:
                st.info("该 run 未匹配 catalog recipe；仍可从 manifest / summary 查看。")
            with st.expander("manifest.json", expanded=False):
                st.json(run.manifest or {}, expanded=False)
            with st.expander("gates_status.json", expanded=False):
                st.json(run.gates or {}, expanded=False)

    with tab_logic:
        st.markdown(
            '<div class="qbd-section-title">Active logic file</div>',
            unsafe_allow_html=True,
        )
        profile_path = (
            run.manifest.get("strategy_profile")
            or (str(recipe.strategy_profile) if recipe and recipe.strategy_profile else "")
        )
        st.code(str(profile_path) or "(missing)", language="text")
        exec_cfg = (run.resolved.get("execution") if run.resolved else None) or {}
        selector_cfg = (run.resolved.get("selector") if run.resolved else None) or {}
        features_cfg = (run.resolved.get("features") if run.resolved else None) or {}
        model_cfg = (run.resolved.get("model") if run.resolved else None) or {}
        st.dataframe(
            pd.DataFrame(
                [
                    ("selector.mode", selector_cfg.get("mode")),
                    ("execution.put_gate_mode", exec_cfg.get("put_gate_mode")),
                    ("execution.tick_exits", exec_cfg.get("tick_exits")),
                    ("execution.fill_spread_frac", exec_cfg.get("fill_spread_frac")),
                    ("execution.live_label_shift_sec", exec_cfg.get("live_label_shift_sec")),
                    ("features.slow_feature_config", features_cfg.get("slow_feature_config")),
                    ("features.frozen_norm", features_cfg.get("frozen_norm")),
                    ("model.checkpoint", model_cfg.get("checkpoint")),
                ],
                columns=["key", "value"],
            ),
            use_container_width=True,
            hide_index=True,
        )
        replay_cfg = run.resolved.get("resolved_replay_config") if run.resolved else None
        with st.expander("strategy_profile.resolved.json", expanded=False):
            st.json(run.resolved or {}, expanded=False)
        if isinstance(replay_cfg, dict):
            with st.expander("resolved_replay_config", expanded=False):
                st.json(replay_cfg, expanded=False)
        if recipe and recipe.stream_script:
            st.caption("Stream script path")
            st.code(str(recipe.stream_script), language="text")

    with tab_trades:
        trade_rows = trades_frame_rows(run.summary)
        if trade_rows:
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.info("summary 中无 trades_detail。")
        fill_files = sorted(run.path.glob("fill_audit_*.csv"))
        if fill_files:
            fill_choice = st.selectbox("fill_audit file", [p.name for p in fill_files])
            fill_path = run.path / fill_choice
            try:
                fill_df = pd.read_csv(fill_path)
                st.dataframe(fill_df, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"无法读取 {fill_path}: {exc}")

    with tab_artifacts:
        st.caption(f"{len(run.artifacts)} files in {run.path}")
        st.dataframe(
            pd.DataFrame({"artifact": run.artifacts}),
            use_container_width=True,
            hide_index=True,
            height=260,
        )
        log_candidates = [
            name
            for name in run.artifacts
            if name.endswith(".log") or name in {"summary.txt", "feat_parity_gate1_raw.txt", "feat_parity_gate2_norm.txt"}
        ]
        if log_candidates:
            log_name = st.selectbox("Preview log / text", log_candidates, index=0)
            st.code(tail_text(run.path / log_name), language="text")
        summary_json = run.path / "stream_summary_paired.json"
        if summary_json.is_file():
            with st.expander("stream_summary_paired.json", expanded=False):
                st.json(run.summary or {}, expanded=False)

    with tab_profiles:
        profiles = list_strategy_profiles()
        st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)
        st.caption("所有 strategy profile 来自 qqq_btc/CONFIG/strategy_profiles/")


def render_offline_replay_board() -> None:
    st.markdown('<div class="qbd-title">Offline Replay</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qbd-sub">离线 live-aligned 回放：日收益、诊断、与流式 <b>共用</b> '
        "<code>strategy_profiles</code>。默认配方 <b>Jul1–13 = W1 + Jul13</b> "
        "（01–10 对齐 W1 +65%；13 用 old_lock）。"
        "股价 July 已修回右标签；纯 old_lock 期权≠W1 勿直接比。"
        "W1 冻结对拍请选 catalog 里的 V0 W1。</div>",
        unsafe_allow_html=True,
    )

    # ---- 特征 / 归一化（从 Download 移到这里） ----
    st.markdown(
        '<div class="qbd-section-title">① 特征 + 归一化（前置）</div>',
        unsafe_allow_html=True,
    )
    hint = feature_norm_hint()
    st.caption(hint.get("note", ""))
    feat_exp_raw = st.text_input(
        "Quote exp（1s quote 根）",
        value=str(BACKFILL_DEFAULT_EXP),
        key="offline_feat_exp",
    )
    feat_exp = Path(feat_exp_raw).expanduser()
    features_root_raw = st.text_input(
        "Features root（归一化输出根）",
        value=str(DEFAULT_FEATURES_ROOT),
        key="offline_features_root",
        help="写入 $ROOT/quote_features_raw 与 $ROOT/quote_features_test",
    )
    features_root = Path(features_root_raw).expanduser()
    st.caption(
        f"raw（未归一化，本轮生成）→ `{features_root / 'quote_features_raw'}` · "
        f"test（归一化后；rolling 时会多借前 1–2 月作窗口）→ `{features_root / 'quote_features_test'}`"
    )
    feat_hist_raw = st.text_input(
        "Rolling history root（借前月 raw）",
        value=str(DEFAULT_FEAT_HISTORY),
        key="offline_feat_hist",
    )
    frozen_norm_raw = st.text_input(
        "Frozen norm .npz（仅 frozen 模式）",
        value=str(DEFAULT_FROZEN_NORM),
        key="offline_frozen_norm",
    )
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        feat_start = st.text_input("Feat start", value="2026-07-01", key="offline_feat_start")
    with fc2:
        feat_end = st.text_input("Feat end", value="2026-07-13", key="offline_feat_end")
    with fc3:
        feat_symbols = st.text_input("Feat symbols", value="QQQ", key="offline_feat_sym")
    with fc4:
        feat_force = st.toggle("Force overwrite features", value=False, key="offline_feat_force")
    fn1, fn2, fn3 = st.columns(3)
    with fn1:
        norm_mode = st.selectbox(
            "归一化模式",
            options=["rolling", "frozen", "none"],
            index=0,
            key="offline_norm_mode",
            help=(
                "rolling=经典离线 window=2000（训练默认）；"
                "frozen=流式/deploy 同款 frozen_norm_qqq_daily.npz；"
                "none=只写 quote_features_raw"
            ),
        )
    with fn2:
        strict_warmup = st.toggle("Strict warmup", value=True, key="offline_strict_wu")
    with fn3:
        python_bin = st.text_input("Python", value=resolve_python(), key="offline_py")
    if norm_mode == "rolling":
        st.caption("rolling：尽量借 Rolling history root 做跨月 buffer。")
    elif norm_mode == "frozen":
        st.caption("frozen：用上方 Frozen norm .npz；与流式 Gate2 对齐。")

    fb1, fb2, fb3 = st.columns(3)
    with fb1:
        if st.button("▶ 生成特征+归一化", type="primary", use_container_width=True, key="btn_feat_norm"):
            try:
                state = start_backfill_job(
                    start_date=feat_start.strip(),
                    end_date=feat_end.strip(),
                    exp=feat_exp,
                    mode="full",
                    force=bool(feat_force),
                    symbols=feat_symbols.strip() or "QQQ",
                    python_bin=python_bin,
                    strict_warmup=bool(strict_warmup),
                    warmup_trading_days=10,
                    vix_history_months=7,
                    norm_mode=str(norm_mode),
                    features_root=features_root,
                    feat_history_root=Path(feat_hist_raw).expanduser(),
                    frozen_norm=Path(frozen_norm_raw).expanduser(),
                )
                st.success(f"features+norm started pid={state.get('pid')}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with fb2:
        if st.button("⏹ 停止特征任务", use_container_width=True, key="btn_stop_feat"):
            stop_backfill_job(feat_exp)
            st.rerun()
    with fb3:
        if st.button("刷新特征日志", use_container_width=True, key="btn_refresh_feat"):
            st.rerun()

    feat_job = read_job_state(feat_exp)
    feat_status = str(feat_job.get("status") or "idle")
    feat_tone = (
        "ok"
        if feat_status == "done"
        else (
            "warn"
            if feat_status == "running"
            else ("crit" if feat_status in {"failed", "stopped"} else "pending")
        )
    )
    st.markdown(
        f"""
        {_status_pill(f"feat_job={feat_status}", feat_tone)}
        <span class="qbd-pill">mode={feat_job.get('mode', '-')}</span>
        <span class="qbd-pill">norm={norm_mode}</span>
        <span class="qbd-pill">pid={feat_job.get('pid', '-')}</span>
        """,
        unsafe_allow_html=True,
    )
    feat_cmd = build_pipeline_cmd(
        start_date=feat_start.strip(),
        end_date=feat_end.strip(),
        exp=feat_exp,
        mode="full",
        force=bool(feat_force),
        symbols=feat_symbols.strip() or "QQQ",
        python_bin=python_bin,
        strict_warmup=bool(strict_warmup),
        norm_mode=str(norm_mode),
        features_root=features_root,
        feat_history_root=Path(feat_hist_raw).expanduser(),
        frozen_norm=Path(frozen_norm_raw).expanduser(),
    )
    st.code(" ".join(feat_cmd), language="bash")
    feat_log = backfill_tail_log(feat_job.get("log_file"))
    if feat_log:
        with st.expander("特征/归一化日志", expanded=feat_status == "running"):
            st.code(feat_log, language="text")
    with st.expander("归一化口径说明", expanded=False):
        st.write(hint)
        st.write(
            {
                "quote_exp": str(feat_exp),
                "features_root": str(features_root),
                "quote_features_raw": str(features_root / "quote_features_raw"),
                "quote_features_test (norm out)": str(features_root / "quote_features_test"),
                "feat_history_root": feat_hist_raw,
                "frozen_norm": frozen_norm_raw,
            }
        )

    # ---- Offline replay ----
    st.markdown(
        '<div class="qbd-section-title">② Offline Replay</div>',
        unsafe_allow_html=True,
    )

    recipes = recipe_offline_options()
    profiles = list_strategy_profiles()
    profile_labels = [
        f"{p['profile_id']} · {Path(p['path']).name}" for p in profiles
    ] or ["(no profiles)"]

    c1, c2, c3 = st.columns(3)
    with c1:
        recipe_labels = ["(manual profile)"] + [
            f"{r['recipe_id']} · {r['title']}" for r in recipes
        ]
        recipe_pick = st.selectbox("Catalog recipe", recipe_labels, index=1 if len(recipe_labels) > 1 else 0)
    selected_recipe = None
    if recipe_pick != "(manual profile)":
        rid = recipe_pick.split(" · ", 1)[0]
        selected_recipe = next((r for r in recipes if r["recipe_id"] == rid), None)
    default_profile_idx = 0
    if selected_recipe and selected_recipe.get("profile"):
        for i, p in enumerate(profiles):
            if p["path"] == selected_recipe["profile"] or p["abs_path"].endswith(
                selected_recipe["profile"]
            ):
                default_profile_idx = i
                break
    with c2:
        profile_pick = st.selectbox("Strategy profile", profile_labels, index=default_profile_idx)
        profile_row = profiles[profile_labels.index(profile_pick)] if profiles else None
    with c3:
        months = st.text_input(
            "Months",
            value="2026-07" if selected_recipe else "2026-06,2026-07",
        )
    out_name = st.text_input(
        "Out name (under offline_live_aligned/)",
        value=(
            Path(selected_recipe["offline_result"]).name
            if selected_recipe and selected_recipe.get("offline_result")
            else (profile_row["profile_id"] + "_offline" if profile_row else "offline_run")
        ),
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("▶ 启动 Offline Replay", type="primary", use_container_width=True):
            if not profile_row:
                st.error("无 strategy profile")
            else:
                try:
                    state = start_offline_replay_job(
                        strategy_profile=profile_row["abs_path"],
                        months=months.strip(),
                        out_name=out_name.strip(),
                        python_bin=python_bin,
                    )
                    st.success(f"started pid={state.get('pid')}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with r2:
        if st.button("⏹ 停止 Replay", use_container_width=True):
            stop_offline_replay_job()
            st.rerun()
    with r3:
        if st.button("刷新结果列表", use_container_width=True):
            st.rerun()

    job = read_offline_job_state()
    status = str(job.get("status") or "idle")
    tone = (
        "ok"
        if status == "done"
        else ("warn" if status == "running" else ("crit" if status in {"failed", "stopped"} else "pending"))
    )
    st.markdown(
        f"""
        {_status_pill(f"job={status}", tone)}
        <span class="qbd-pill">out={job.get('out_name', '-')}</span>
        <span class="qbd-pill">pid={job.get('pid', '-')}</span>
        """,
        unsafe_allow_html=True,
    )
    if profile_row:
        cmd = build_replay_cmd(
            strategy_profile=profile_row["abs_path"],
            months=months.strip(),
            out_name=out_name.strip(),
            python_bin=python_bin,
        )
        st.code(" ".join(cmd), language="bash")
    if selected_recipe and selected_recipe.get("offline_cmd"):
        st.caption("Catalog offline_cmd")
        st.code(selected_recipe["offline_cmd"], language="bash")

    log_text = offline_tail_log(job.get("log_file"))
    if log_text:
        with st.expander("Job log", expanded=status == "running"):
            st.code(log_text, language="text")

    runs = discover_offline_runs(limit=50)
    if not runs:
        st.warning(f"尚无离线结果：{OFFLINE_ROOT}")
        return

    run_labels = []
    for run in runs:
        hl = headline_rows(run.summary)
        acct = hl[0].get("acct25_pct") if hl else None
        acct_s = f"{acct:+.2f}%" if isinstance(acct, (int, float)) else "-"
        run_labels.append(
            f"{run.name} | {acct_s} | profile={run.profile_id or '-'} | {_fmt_age(run.mtime)}"
        )
    chosen = st.selectbox("Offline run", run_labels, index=0)
    run = runs[run_labels.index(chosen)]
    summary = run.summary
    headlines = headline_rows(summary)
    months_avail = [h["month"] for h in headlines] or list((summary.get("months") or {}).keys())
    month = st.selectbox("Month", months_avail, index=0) if months_avail else ""

    if headlines:
        cols = st.columns(min(4, len(headlines)))
        for col, row in zip(cols, headlines):
            with col:
                st.metric(
                    row.get("month", "?"),
                    _fmt_pct_num(row.get("acct25_pct")),
                    help=f"trades={row.get('trades')} mdd={row.get('mdd_pct')}",
                )

    diag = diagnostic_bundle(summary, month=month) if month else {}
    if diag:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("acct@25%", _fmt_pct_num(diag.get("acct25_pct")))
        m2.metric("trades", diag.get("trades") if diag.get("trades") is not None else "-")
        m3.metric("hit%", _fmt_pct_num(diag.get("hit_pct")))
        m4.metric("mdd%", _fmt_pct_num(diag.get("mdd_pct")))
        m5.metric("Δ vs baseline", _fmt_pct_num(diag.get("delta_vs_baseline_pp")))
        st.caption(
            f"legs={diag.get('legs')} · win_days={diag.get('n_win_days')} · "
            f"loss_days={diag.get('n_loss_days')} · early4_min_cum={diag.get('early4_min_cum')}"
        )

    daily = daily_rows_from_summary(summary, month=month or None)
    tab_daily, tab_diag, tab_logic = st.tabs(["Daily PnL", "Diagnostics", "Logic / Provenance"])
    with tab_daily:
        if not daily:
            st.info("summary 中无 regime.daily")
        else:
            df = pd.DataFrame(daily)
            if go is not None and "cum_acct25_pct" in df.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["cum_acct25_pct"],
                        mode="lines+markers",
                        name="cum acct@25%",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=df["date"],
                        y=df["day_acct25_pct"],
                        name="day acct@25%",
                        opacity=0.45,
                        yaxis="y2",
                    )
                )
                fig.update_layout(
                    height=360,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(title="cum %"),
                    yaxis2=dict(title="day %", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_diag:
        if not diag:
            st.info("无诊断数据")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Best days**")
                st.dataframe(pd.DataFrame(diag.get("best_days") or []), use_container_width=True, hide_index=True)
                st.markdown("**Worst days**")
                st.dataframe(pd.DataFrame(diag.get("worst_days") or []), use_container_width=True, hide_index=True)
            with right:
                st.markdown("**Segments**")
                st.json(diag.get("segments") or [], expanded=False)
                st.markdown("**Replay gates (offline cfg)**")
                st.json(diag.get("gates") or {}, expanded=False)
                st.write(
                    {
                        "baseline_acct25_pct": diag.get("baseline_acct25_pct"),
                        "baseline_trades": diag.get("baseline_trades"),
                        "profile_day_counts": diag.get("profile_day_counts"),
                        "path": str(run.path),
                        "profile_id": run.profile_id,
                        "profile_sha": (run.profile_sha[:12] + "…") if run.profile_sha else "",
                    }
                )

    with tab_logic:
        st.caption("调规则只改 strategy profile；Offline 与 Stream 共用同一文件。")
        st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)
        with st.expander("manifest.json", expanded=False):
            st.json(run.manifest or {}, expanded=False)
        with st.expander("summary.json (truncated)", expanded=False):
            st.json(
                {
                    "headline": summary.get("headline"),
                    "gates": summary.get("gates"),
                    "provenance": summary.get("provenance"),
                },
                expanded=False,
            )


def _fmt_age(mtime: float) -> str:
    age = max(0.0, time.time() - float(mtime))
    if age < 3600:
        return f"{int(age // 60)}m ago"
    if age < 86400:
        return f"{age / 3600:.1f}h ago"
    return f"{age / 86400:.1f}d ago"


def render_backfill_board() -> None:
    st.markdown('<div class="qbd-title">Download</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qbd-sub">开盘价锁 4 合约 → quote → 1m/day_iv。'
        "<b>不含特征/归一化</b>——下载后到 Offline Replay / Stream Parity 前再生成连续完整特征并归一化。"
        " 脚本：<code>run_backfill_open_lock_pipeline.py</code>。"
        " <b>W1 对拍约定</b>：Massive raw=左标签；"
        "<code>spnq_train_resampled</code> 进特征前必须右标签（首根 09:31）。</div>",
        unsafe_allow_html=True,
    )

    exp_raw = st.text_input("Download exp root", value=str(BACKFILL_DEFAULT_EXP))
    exp = Path(exp_raw).expanduser()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_date = st.text_input("Start date", value="2026-07-01")
    with c2:
        end_date = st.text_input("End date", value="2026-07-13")
    with c3:
        symbols = st.text_input("Symbols", value="QQQ")
    with c4:
        force = st.toggle("Force overwrite", value=False)

    st.markdown('<div class="qbd-section-title">Data directories</div>', unsafe_allow_html=True)
    dir_rows = data_directory_rows(exp, symbols=symbols.strip() or "QQQ")
    st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)
    missing_dirs = [r["path"] for r in dir_rows if r["exists"] == "no" and r["role"] in {
        "exp_root", "stock_root", "stock_1min (warmup)", "anchor_config"
    }]
    if missing_dirs:
        st.caption("关键路径尚不存在：" + " · ".join(f"`{p}`" for p in missing_dirs[:4]))
    else:
        st.caption(
            f"quote exp=`{exp}`（文件在 QQQ/）；"
            "缺数/预热比对股价 `~/train_data/spnq_train_resampled`。"
        )

    # ---- W1 bar-label gate (auto) ----
    st.markdown(
        '<div class="qbd-section-title">W1 股价标签检测（resampled）</div>',
        unsafe_allow_html=True,
    )
    label_scan = scan_stock_bar_labels(
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        symbols=symbols.strip() or "QQQ",
    )
    st.session_state["backfill_bar_label_scan"] = label_scan
    bad_label_files = label_scan.get("bad_files") or []
    label_rows = [
        {
            "symbol": f.get("symbol"),
            "ym": f.get("ym"),
            "res": f.get("res"),
            "label": f.get("label"),
            "first": f.get("first_hhmm"),
            "ok_w1": f.get("ok_for_w1"),
            "note": f.get("note"),
        }
        for f in (label_scan.get("files") or [])
        if f.get("exists")
    ]
    if label_rows:
        st.dataframe(pd.DataFrame(label_rows), use_container_width=True, hide_index=True)
    if label_scan.get("ok"):
        st.success(
            f"W1 右标签 OK：区间 {start_date}→{end_date} 内 "
            f"QQQ/VIXY 1min/5min 首根均为 09:31。"
        )
    else:
        st.error(
            f"发现 {label_scan.get('n_bad', 0)} 个左标签/异常文件"
            "（Massive raw 是左标签；resampled 进特征前必须改成右标签，否则 Offline/对拍会偏 1 分钟）。"
        )
        lb1, lb2, lb3 = st.columns(3)
        with lb1:
            if st.button("↻ 重新检测标签", use_container_width=True, key="btn_relabel_scan"):
                st.rerun()
        with lb2:
            if st.button(
                "🛠 一键纠正 → 右标签",
                type="primary",
                use_container_width=True,
                key="btn_fix_right_label",
            ):
                with st.spinner("正在把左标签 +1min 改为 W1 右标签（自动备份）…"):
                    fix_rep = fix_stock_bar_labels(
                        start_date=start_date.strip(),
                        end_date=end_date.strip(),
                        symbols=symbols.strip() or "QQQ",
                        dry_run=False,
                    )
                st.session_state["backfill_bar_label_fix"] = fix_rep
                if fix_rep.get("after_ok"):
                    st.success(
                        f"已纠正 {fix_rep.get('n_changed')}/{fix_rep.get('n_attempted')} 个文件"
                    )
                else:
                    st.warning(
                        f"已尝试 {fix_rep.get('n_attempted')} 个；"
                        f"changed={fix_rep.get('n_changed')} after_ok={fix_rep.get('after_ok')}"
                    )
                st.rerun()
        with lb3:
            st.caption("备份：`*.bak_left_label_YYYYMMDD_HHMMSS`")
        with st.expander("待纠正文件", expanded=True):
            st.json(bad_label_files, expanded=False)
    fix_flash = st.session_state.get("backfill_bar_label_fix")
    if fix_flash and label_scan.get("ok"):
        st.caption(
            f"最近纠正：changed={fix_flash.get('n_changed')} "
            f"at {fix_flash.get('scan_after', {}).get('months')}"
        )
    with st.expander("标签约定说明", expanded=False):
        st.write(label_scan.get("convention") or {})
        st.code(
            "python -m qqq_btc.common.bar_label_convention "
            f"--start {start_date.strip()} --end {end_date.strip()} --scan\n"
            "python -m qqq_btc.common.bar_label_convention "
            f"--start {start_date.strip()} --end {end_date.strip()} --fix",
            language="bash",
        )

    key_info = api_key_status()
    api_key_input = st.text_input(
        "Massive/Polygon API key（可空，优先用环境变量）",
        value="",
        type="password",
        help=key_info["hint"],
    )
    if key_info["ok"]:
        st.caption(f"环境已有 key：{key_info['hint']}")
    else:
        st.warning(key_info["hint"])

    with st.expander("高级选项", expanded=False):
        python_bin = st.text_input("Python", value=resolve_python())
        max_workers = st.number_input("Max workers", min_value=1, max_value=64, value=16)
        warmup_days = st.number_input("Warmup trading days", min_value=1, max_value=60, value=10)
        vix_months = st.number_input("VIXY history months", min_value=1, max_value=24, value=7)

    st.markdown('<div class="qbd-section-title">Coverage / 缺数检查</div>', unsafe_allow_html=True)
    wu1, wu2 = st.columns([1, 3])
    with wu1:
        if st.button("检查缺数 / 预热", use_container_width=True):
            with st.spinner("检查 QQQ/VIXY 历史连续性…"):
                rep = run_warmup_check_job(
                    start_date=start_date.strip(),
                    end_date=end_date.strip(),
                    exp=exp,
                    symbols=symbols.strip() or "QQQ",
                    python_bin=python_bin,
                    warmup_trading_days=int(warmup_days),
                    vix_history_months=int(vix_months),
                )
            st.session_state["backfill_warmup_flash"] = rep
            st.rerun()
    warmup = load_warmup_report(exp) or st.session_state.get("backfill_warmup_flash") or {}
    if warmup:
        cov = warmup.get("coverage_vs_today") or {}
        if cov:
            asof = cov.get("asof", "?")
            union_miss = cov.get("union_missing_days") or []
            if cov.get("ok"):
                st.success(f"相对当前（asof={asof}）：近 45 日日历内 1min 无缺日。")
            else:
                st.warning(
                    f"相对当前（asof={asof}）缺 {len(union_miss)} 个交易日："
                    f"`{', '.join(union_miss[-12:])}`"
                    + (" …" if len(union_miss) > 12 else "")
                )
            for row in cov.get("symbols") or []:
                st.caption(
                    f"{row.get('symbol')}: latest={row.get('latest_present')} "
                    f"lag={row.get('lag_trading_days')}d missing={row.get('missing_count')}"
                )
        if warmup.get("ok"):
            st.success(
                f"目标区间预热 OK：前 {warmup.get('warmup_trading_days')} 交易日连续；"
                f"VIXY 历史 {warmup.get('vix_history_months')} 月齐全。"
            )
        else:
            st.error("目标区间预热有缺口 —— 后续做特征前请先补齐分钟数据。")
            for b in warmup.get("blockers") or []:
                st.markdown(f"- `{b}`")
        for w in warmup.get("warnings") or []:
            st.warning(w)
        with st.expander("warmup_report.json", expanded=not bool(warmup.get("ok"))):
            st.json(warmup, expanded=False)
    else:
        st.info("建议先点「检查缺数 / 预热」：相对今天缺哪几天 + 所选区间窗口是否连续。")

    st.markdown('<div class="qbd-section-title">One-click download</div>', unsafe_allow_html=True)
    if not label_scan.get("ok"):
        st.warning("股价标签未通过 W1 右标签检测：建议先点上方「一键纠正」，再下载/做特征。")
    b1, b2, b3 = st.columns(3)
    run_mode = None
    with b1:
        if st.button("① 仅锁约", use_container_width=True):
            run_mode = "lock_only"
    with b2:
        if st.button("② 下载+day_iv", use_container_width=True, type="primary"):
            run_mode = "download"
    with b3:
        if st.button("⏹ 停止任务", use_container_width=True):
            stop_backfill_job(exp)
            st.rerun()

    if run_mode:
        try:
            state = start_backfill_job(
                start_date=start_date.strip(),
                end_date=end_date.strip(),
                exp=exp,
                mode=run_mode,
                force=force,
                symbols=symbols.strip() or "QQQ",
                python_bin=python_bin,
                api_key=api_key_input or None,
                max_workers=int(max_workers),
                strict_warmup=False,
                warmup_trading_days=int(warmup_days),
                vix_history_months=int(vix_months),
                norm_mode="none",
            )
            st.success(f"已启动 {run_mode} pid={state.get('pid')}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    job = read_job_state(exp)
    status = str(job.get("status") or "idle")
    status_tone = (
        "ok" if status == "done" else ("warn" if status == "running" else ("crit" if status in {"failed", "stopped"} else "pending"))
    )
    st.markdown(
        f"""
        {_status_pill(f"job={status}", status_tone)}
        <span class="qbd-pill">mode={job.get('mode', '-')}</span>
        <span class="qbd-pill">pid={job.get('pid', '-')}</span>
        <span class="qbd-pill">{job.get('start_date', '-')} → {job.get('end_date', '-')}</span>
        """,
        unsafe_allow_html=True,
    )
    if job.get("cmd"):
        st.code(" ".join(str(x) for x in job["cmd"]), language="bash")

    log_text = backfill_tail_log(job.get("log_file"))
    if log_text:
        st.markdown('<div class="qbd-section-title">Live log</div>', unsafe_allow_html=True)
        st.code(log_text, language="text")
        if status == "running":
            st.caption("任务运行中：可开 Auto refresh，或点下方刷新。")
            if st.button("刷新日志"):
                st.rerun()

    cmds = backfill_suggested_commands(
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        exp=exp,
        features=False,
    )
    with st.expander("等价 CLI", expanded=False):
        st.caption(cmds.get("note", "") + " · 本页不跑 features/norm")
        st.code(cmds["lock_only"], language="bash")
        st.code(cmds["full_pipeline"], language="bash")

    summary = load_pipeline_summary(exp)
    lock_report = load_lock_report(exp)
    days = discover_backfill_days(exp)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("exp exists", "yes" if exp.is_dir() else "no")
    m2.metric("by_date days", len(days))
    m3.metric("lock rows", lock_report.get("n_rows", "-"))
    m4.metric("lock days", lock_report.get("n_days", len(summary.get("days") or [])))

    left, right = st.columns([1.1, 1.0])
    with left:
        st.markdown('<div class="qbd-section-title">by_date manifests</div>', unsafe_allow_html=True)
        if not days:
            st.info(f"尚无 {exp}/by_date/* ；点「下载+day_iv」。")
        else:
            rows = [
                {
                    "date": d.date,
                    "n_contracts": d.n_contracts,
                    "stock_open": d.stock_open,
                    "raw_1s": bool(d.manifest.get("raw_1s")),
                    "options_1m": bool(d.manifest.get("options_1m")),
                    "day_iv": bool(d.manifest.get("day_iv")),
                    "contracts": ", ".join(d.manifest.get("contracts") or []),
                }
                for d in days
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            labels = [d.date for d in days]
            pick = st.selectbox("Inspect day", labels, index=0)
            day = next(d for d in days if d.date == pick)
            st.json(day.manifest, expanded=True)
            lock_p = day.path / "lock_map.parquet"
            if lock_p.is_file():
                st.caption(str(lock_p))
                st.dataframe(pd.read_parquet(lock_p), use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="qbd-section-title">Logic / Config</div>', unsafe_allow_html=True)
        st.write(
            {
                "pipeline_order": "Download → Offline Replay → Stream Parity",
                "w1_bar_label": {
                    "massive_raw_spnq_train": "left (09:30) — keep",
                    "spnq_train_resampled": "right (09:31) — required before features",
                    "live_bridge": "FCS alpha_label_ts + 60s",
                    "tool": "python -m qqq_btc.common.bar_label_convention",
                },
                "anchor_config": "preprocess/CONFIG/anchor_qqq_1dte_4bucket.json",
                "lock_script": "preprocess/download/step1_lock_4bucket_from_open.py",
                "pipeline": "preprocess/download/run_backfill_open_lock_pipeline.py",
                "quote_sniper": "preprocess/download/step2_polygon_second_sniper_v1.py",
                "buckets": "0 PUT ATM | 1 PUT OTM | 2 CALL ATM | 3 CALL OTM",
                "this_page": {
                    "仅锁约": "lock_only",
                    "下载+day_iv": "lock → quote → 1m → day_iv → by_date",
                    "一键纠正标签": "resampled left→right (+1min, backup)",
                },
                "not_here": "feature_merge / rolling_norm / frozen_norm → Offline 或 Stream 前置",
            }
        )
        if lock_report:
            with st.expander("lock_report.json", expanded=False):
                st.json(lock_report, expanded=False)
        if summary:
            with st.expander("pipeline_summary.json", expanded=False):
                st.json(summary, expanded=False)
        map_path = exp / "locked_targets_map_open_4bucket.parquet"
        if map_path.is_file():
            st.caption(str(map_path))
            st.dataframe(pd.read_parquet(map_path).head(20), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="QQQ_BTC Dash", layout="wide")
    render_css()
    board, redis_cfg, audit_path, replay_path, auto_refresh, refresh_sec, symbols = render_sidebar()

    if board == "Stream Parity":
        render_parity_board()
        if auto_refresh:
            time.sleep(refresh_sec)
            st.rerun()
        return

    if board == "Download":
        render_backfill_board()
        if auto_refresh:
            time.sleep(refresh_sec)
            st.rerun()
        return

    if board == "Offline Replay":
        render_offline_replay_board()
        if auto_refresh:
            time.sleep(refresh_sec)
            st.rerun()
        return

    if board == "Mag7":
        render_maga7_board()
        if auto_refresh:
            time.sleep(refresh_sec)
            st.rerun()
        return

    qqq_live = os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}
    render_header(redis_cfg, qqq_live, symbols)
    render_live_deploy_panel()

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
