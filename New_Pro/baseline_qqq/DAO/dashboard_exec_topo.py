"""
实时执行拓扑：IBKR 双向 lock → FCS 快门控 → 慢 TFT → Strategy → OMS。

Dashboard 只读 Redis / PG / 推理流，不修改交易状态。
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None


def _dec(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _dec_hash(h: Optional[dict]) -> Dict[str, str]:
    if not h:
        return {}
    out: Dict[str, str] = {}
    for k, v in h.items():
        out[_dec(k)] = _dec(v)
    return out


def _lag_status(lag_sec: Optional[float], *, warn: float = 3.0, crit: float = 15.0) -> str:
    if lag_sec is None:
        return "off"
    if lag_sec > crit:
        return "crit"
    if lag_sec > warn:
        return "warn"
    return "ok"


def _status_color(status: str, is_light: bool) -> str:
    palette = {
        "ok": ("#00CC96", "#059669"),
        "warn": ("#FECB52", "#D97706"),
        "crit": ("#EF553B", "#DC2626"),
        "off": ("#6B7280", "#9CA3AF"),
        "active": ("#636EFA", "#4F46E5"),
        "idle": ("#AB63FA", "#7C3AED"),
    }
    pair = palette.get(status, palette["off"])
    return pair[0] if not is_light else pair[1]


def _fetch_lock_tags(symbol: str, pg_url: str) -> Dict[str, bool]:
    out = {"CALL_ATM": False, "PUT_ATM": False}
    if not pg_url or psycopg2 is None:
        return out
    try:
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()
        for tag in out:
            cur.execute(
                """
                SELECT 1 FROM contract_locks
                WHERE symbol=%s AND tag=%s
                ORDER BY date DESC LIMIT 1
                """,
                (symbol.upper(), tag),
            )
            out[tag] = cur.fetchone() is not None
        conn.close()
    except Exception:
        pass
    return out


def _alpha_from_batch(batch: Optional[dict], symbol: str) -> Tuple[float, float]:
    if not batch or "symbols" not in batch:
        return 0.0, 0.0
    try:
        syms = batch.get("symbols") or []
        if symbol not in syms:
            return 0.0, 0.0
        i = syms.index(symbol)
        alpha = float((batch.get("alphas") or [0.0] * len(syms))[i] or 0.0)
        vol_raw = batch.get("vol_z")
        if isinstance(vol_raw, dict):
            vol_z = float(vol_raw.get(symbol, 0.0) or 0.0)
        else:
            vol_z = float((vol_raw or [0.0] * len(syms))[i] or 0.0)
        return alpha, vol_z
    except Exception:
        return 0.0, 0.0


def _parse_oms_row(raw_val: Any) -> dict:
    if raw_val is None:
        return {}
    if isinstance(raw_val, (bytes, bytearray)):
        try:
            raw_val = pickle.loads(raw_val)
        except Exception:
            try:
                raw_val = json.loads(raw_val.decode("utf-8", errors="ignore"))
            except Exception:
                return {}
    if isinstance(raw_val, str):
        try:
            raw_val = json.loads(raw_val)
        except Exception:
            return {}
    return raw_val if isinstance(raw_val, dict) else {}


@dataclass
class ExecTopoState:
    symbol: str
    updated_at: float = 0.0
    # IBKR
    call_locked: bool = False
    put_locked: bool = False
    src_lag: Optional[float] = None
    stream_status: str = "off"
    eng_lag: Optional[float] = None
    fast_spread: float = 0.0
    fast_iv_mom: float = 0.0
    fast_gate_ok: bool = False
    # Slow TFT
    alpha: float = 0.0
    vol_z: float = 0.0
    side_hint: str = "FLAT"  # CALL | PUT | FLAT
    eng_status: str = "off"
    # Strategy
    session: str = "unknown"
    entry_result: str = "-"
    entry_block: str = ""
    exit_result: str = "-"
    # OMS
    position: int = 0
    exec_profile: str = ""
    exec_band: str = ""
    oms_status: str = "flat"
    band_legs_today: int = 0
    # path highlight
    active_leg: str = "none"  # call | put | none
    node_notes: Dict[str, str] = field(default_factory=dict)


def build_exec_topo_state(
    r,
    symbol: str,
    *,
    pg_url: str = "",
    eng_batch: Optional[dict] = None,
    src_lag: Optional[float] = None,
    eng_lag: Optional[float] = None,
    fast_spread_max: float = 0.12,
    fast_iv_max: float = 0.50,
    exec_profile_env: str = "",
) -> ExecTopoState:
    sym = str(symbol or "QQQ").upper()
    st = ExecTopoState(symbol=sym, updated_at=time.time())

    locks = _fetch_lock_tags(sym, pg_url)
    st.call_locked = locks.get("CALL_ATM", False)
    st.put_locked = locks.get("PUT_ATM", False)
    st.src_lag = src_lag
    st.stream_status = _lag_status(src_lag)

    st.eng_lag = eng_lag
    st.eng_status = _lag_status(eng_lag)
    st.alpha, st.vol_z = _alpha_from_batch(eng_batch, sym)
    if st.alpha > 0.0001:
        st.side_hint = "CALL"
    elif st.alpha < -0.0001:
        st.side_hint = "PUT"
    else:
        st.side_hint = "FLAT"

    gg = _dec_hash(r.hgetall("meta:global_gates") if r else {})
    st.session = gg.get("session", "unknown")

    # gate trace (latest entry — hash overwritten per publish)
    gt = _dec_hash(r.hgetall(f"meta:gate_trace:{sym}") if r else {})
    kind = gt.get("kind", "")
    result = gt.get("result", "-")
    if kind == "exit":
        st.exit_result = result
    else:
        st.entry_result = result
        st.entry_block = gt.get("last_block", "")

    # OMS position
    try:
        raw_map = r.hgetall("oms:live_positions") or {}
        pos = {}
        for raw_k, raw_v in raw_map.items():
            k = _dec(raw_k)
            if k == "____SYSTEM_CASH____":
                continue
            if k.split("|", 1)[0].strip().upper() == sym:
                pos = _parse_oms_row(raw_v)
                break
        st.position = int(pos.get("position", 0) or 0)
        st.exec_profile = str(pos.get("exec_profile", "") or exec_profile_env or "")
        st.exec_band = str(pos.get("exec_band", "") or "")
        st.band_legs_today = int(pos.get("band_legs_today", 0) or 0)
        if st.position > 0:
            st.oms_status = "call"
            st.active_leg = "call"
        elif st.position < 0:
            st.oms_status = "put"
            st.active_leg = "put"
        else:
            st.oms_status = "flat"
            st.active_leg = st.side_hint.lower() if st.side_hint != "FLAT" else "none"
    except Exception:
        st.exec_profile = exec_profile_env or ""

    if st.active_leg == "none" and st.side_hint != "FLAT":
        st.active_leg = st.side_hint.lower()

    # fast gate from latest inference batch if present
    if eng_batch:
        spreads = eng_batch.get("options_vw_spread") or {}
        iv_moms = eng_batch.get("options_iv_momentum") or {}
        if isinstance(spreads, dict):
            st.fast_spread = float(spreads.get(sym, 0.0) or 0.0)
        if isinstance(iv_moms, dict):
            st.fast_iv_mom = abs(float(iv_moms.get(sym, 0.0) or 0.0))
        st.fast_gate_ok = (
            st.fast_spread <= fast_spread_max > 0
            and st.fast_iv_mom <= fast_iv_max
        )

    st.node_notes = {
        "ibkr": f"CALL {'✓' if st.call_locked else '✗'} | PUT {'✓' if st.put_locked else '✗'}",
        "call": "locked" if st.call_locked else "missing",
        "put": "locked" if st.put_locked else "missing",
        "fcs": f"spread {st.fast_spread:.1%} ivΔ {st.fast_iv_mom:.2f}",
        "slow": f"α={st.alpha:+.3f} → {st.side_hint}",
        "strategy": st.entry_result if st.entry_result != "-" else st.session,
        "oms": (
            f"{st.oms_status.upper()} band={st.exec_band or '-'}"
            if st.position != 0
            else f"flat prof={st.exec_profile or '-'}"
        ),
    }
    return st


def draw_exec_topo_figure(state: ExecTopoState, *, is_light: bool = True) -> go.Figure:
    """Plotly 双向执行拓扑（实时状态着色）。"""
    # 主链节点
    main_nodes = {
        "ibkr": (0.0, 1.0, "IBKR\n双向 Lock"),
        "fcs": (1.35, 1.0, "FCS\n快通道门控"),
        "slow": (2.7, 1.0, "慢 TFT\nnet_edge"),
        "strategy": (4.05, 1.0, "Strategy\nV0 门禁"),
        "oms": (5.4, 1.0, "OMS\n执行"),
    }
    sub_nodes = {
        "call": (0.0, 1.55, "CALL_ATM"),
        "put": (0.0, 0.45, "PUT_ATM"),
    }

    def _node_status(name: str) -> str:
        if name == "ibkr":
            if state.call_locked and state.put_locked:
                return "ok"
            if state.call_locked or state.put_locked:
                return "warn"
            return "crit"
        if name == "call":
            return "active" if state.active_leg == "call" else ("ok" if state.call_locked else "off")
        if name == "put":
            return "active" if state.active_leg == "put" else ("ok" if state.put_locked else "off")
        if name == "fcs":
            return "ok" if state.fast_gate_ok else ("warn" if state.stream_status == "ok" else state.stream_status)
        if name == "slow":
            return state.eng_status if state.eng_status != "off" else "warn"
        if name == "strategy":
            if str(state.entry_result).startswith("REJECT"):
                return "warn"
            if state.entry_result == "BUY":
                return "ok"
            return "idle"
        if name == "oms":
            return "active" if state.position != 0 else "idle"
        return "off"

    fig = go.Figure()
    edge_color = "#9CA3AF" if is_light else "#555555"
    active_color = "#636EFA" if not is_light else "#4F46E5"

    def _edge(x0, y0, x1, y1, *, highlight: bool = False):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=4 if highlight else 2,
                    color=active_color if highlight else edge_color,
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # IBKR → CALL / PUT
    _edge(0.05, 1.0, 0.05, 1.55, highlight=state.active_leg == "call")
    _edge(0.05, 1.0, 0.05, 0.45, highlight=state.active_leg == "put")
    _edge(0.12, 1.55, 0.55, 1.08, highlight=state.active_leg == "call")
    _edge(0.12, 0.45, 0.55, 0.92, highlight=state.active_leg == "put")

    chain = ["ibkr", "fcs", "slow", "strategy", "oms"]
    for u, v in zip(chain, chain[1:]):
        _edge(main_nodes[u][0], main_nodes[u][1], main_nodes[v][0], main_nodes[v][1], highlight=False)

    def _add_nodes(node_dict: dict, size: int = 46):
        xs, ys, texts, colors, hovers = [], [], [], [], []
        for key, (x, y, label) in node_dict.items():
            xs.append(x)
            ys.append(y)
            note = state.node_notes.get(key, "")
            texts.append(label)
            colors.append(_status_color(_node_status(key), is_light))
            hovers.append(f"{label.replace(chr(10), ' ')}<br>{note}")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=size, color=colors, line=dict(width=2, color="white")),
                text=texts,
                textposition="middle center",
                textfont=dict(size=10, color="white" if not is_light else "#111827"),
                hovertext=hovers,
                hoverinfo="text",
                showlegend=False,
            )
        )

    _add_nodes(sub_nodes, size=36)
    _add_nodes(main_nodes, size=52)

    # Phase 标注（架构层，非运行时开关）
    annotations = [
        dict(x=0.0, y=1.78, text="① 数据", showarrow=False, font=dict(size=10, color="#888")),
        dict(x=1.35, y=1.78, text="快 gate", showarrow=False, font=dict(size=10, color="#888")),
        dict(x=2.7, y=1.78, text="③ 定边", showarrow=False, font=dict(size=10, color="#888")),
        dict(x=4.05, y=1.78, text="② 规则", showarrow=False, font=dict(size=10, color="#888")),
        dict(x=5.4, y=1.78, text="④ 打法", showarrow=False, font=dict(size=10, color="#888")),
    ]
    subtitle = (
        f"{state.symbol} | session={state.session} | "
        f"α={state.alpha:+.3f} side={state.side_hint} | "
        f"pos={state.position} profile={state.exec_profile or '-'} band={state.exec_band or '-'}"
    )
    if state.entry_block:
        subtitle += f" | block={state.entry_block}"

    fig.update_layout(
        title=dict(
            text=f"🗺️ 执行拓扑（双向 CALL/PUT）<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=14 if is_light else 15),
        ),
        annotations=annotations,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.35, 5.85]),
        yaxis=dict(visible=False, range=[0.15, 1.95]),
        height=300,
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_exec_topo_panel(
    r,
    symbol: str,
    *,
    is_light: bool = True,
    eng_batch: Optional[dict] = None,
    src_lag: Optional[float] = None,
    eng_lag: Optional[float] = None,
    pg_url: str = "",
    exec_profile: str = "",
) -> ExecTopoState:
    """Streamlit 侧：构建状态并返回 figure 由调用方 st.plotly_chart。"""
    state = build_exec_topo_state(
        r,
        symbol,
        pg_url=pg_url,
        eng_batch=eng_batch,
        src_lag=src_lag,
        eng_lag=eng_lag,
        exec_profile_env=exec_profile,
    )
    return state
