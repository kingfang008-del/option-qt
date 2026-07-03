"""
qqq_btc / V8 系统全景 —— 离线管线 + common 层 + 双引擎实盘拓扑。

Dashboard 只读 Redis / PG / Stream，不修改交易状态。
复用 dashboard_exec_topo 的实时执行链，并扩展为 ARCHITECTURE.md 完整流程图。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from dashboard_exec_topo import ExecTopoState, draw_exec_topo_figure


def _status_color(status: str, is_light: bool) -> str:
    palette = {
        "ok": ("#00CC96", "#059669"),
        "warn": ("#FECB52", "#D97706"),
        "crit": ("#EF553B", "#DC2626"),
        "off": ("#6B7280", "#9CA3AF"),
        "idle": ("#AB63FA", "#7C3AED"),
        "active": ("#636EFA", "#4F46E5"),
        "info": ("#19D3F3", "#0891B2"),
    }
    pair = palette.get(status, palette["off"])
    return pair[0] if not is_light else pair[1]


def _lag_status(lag: Optional[float], *, warn: float = 3.0, crit: float = 15.0) -> str:
    if lag is None or lag >= 900:
        return "off"
    if lag > crit:
        return "crit"
    if lag > warn:
        return "warn"
    return "ok"


@dataclass
class GateSnapshot:
    """G0→G3 验收门 —— 在线可推断部分 + 静态说明。"""
    g0: str = "pending"
    g1: str = "pending"
    g2: str = "pending"
    g3: str = "pending"
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemFlowContext:
    symbol: str
    run_mode: str
    is_light: bool = True
    # live probes (from SystemStatus / Redis)
    src_lag: float = 999.0
    raw_lag: float = 999.0
    eng_lag: float = 999.0
    ibkr_state: str = "unknown"
    ibkr_connected: bool = False
    session: str = "unknown"
    alpha: float = 0.0
    net_edge: float = 0.0
    position: int = 0
    entry_result: str = "-"
    exit_result: str = "-"
    stream_rows: List[dict] = field(default_factory=list)
    exec_topo: Optional[ExecTopoState] = None
    qqq_btc_live_hint: bool = False


def infer_gate_snapshot(ctx: SystemFlowContext) -> GateSnapshot:
    """从在线健康度粗推断验收门(非严格 CI，仅供运维扫一眼)。"""
    snap = GateSnapshot()
    data_ok = ctx.src_lag < 15 and ctx.raw_lag < 15
    fcs_ok = ctx.eng_lag < 15
    snap.g0 = "ok" if data_ok and fcs_ok else ("warn" if data_ok or fcs_ok else "crit")
    snap.notes["G0"] = "数据/FCS 延迟 <15s"

    orch_lag = None
    oms_lag = None
    for row in ctx.stream_rows:
        stream = str(row.get("stream", ""))
        lag = row.get("lag")
        try:
            lag_i = int(lag) if lag is not None else 0
        except Exception:
            lag_i = 0
        if "inference" in stream:
            orch_lag = lag_i
        if "orch_trade" in stream:
            oms_lag = lag_i
    se_ok = ctx.eng_lag < 10 and (orch_lag is None or orch_lag < 50)
    snap.g1 = "ok" if se_ok and ctx.ibkr_connected else ("warn" if se_ok else "crit")
    snap.notes["G1"] = "IBKR + FCS→SE 推理链"

    has_pos_signal = ctx.entry_result not in ("-", "") or ctx.position != 0
    snap.g2 = "warn" if has_pos_signal else "pending"
    snap.notes["G2"] = "strict replay 需离线跑 run_replay"

    snap.g3 = "pending"
    snap.notes["G3"] = "影子 parity_audit fill/exits"
    if ctx.qqq_btc_live_hint:
        snap.g3 = "warn"
        snap.notes["G3"] = "QQQ_BTC_LIVE 已启用，待 2 周对账"
    return snap


def draw_offline_pipeline_figure(ctx: SystemFlowContext) -> go.Figure:
    """① 数据 → ② common → ③ 模型 → ④ 回放(静态架构 + 少量 live 着色)。"""
    nodes = {
        "s1": (0.0, 1.0, "step1\n选约锁定"),
        "s2": (1.1, 1.0, "step2\nsniper quote"),
        "merge": (2.2, 1.0, "feature\nmerge"),
        "label": (3.3, 1.0, "label_pipeline\n★ fill 0.775"),
        "lmdb": (4.4, 1.0, "norm →\nLMDB"),
        "train": (5.5, 1.0, "train\nTFT v2"),
        "infer": (6.6, 1.0, "run_inference\nedge parquet"),
        "l1": (7.7, 1.25, "L1\nstrict replay"),
        "l2": (7.7, 0.75, "L2\nevent+tick"),
    }
    common_nodes = {
        "fill": (2.2, 0.15, "fill_model\n0.775"),
        "entry": (3.3, 0.15, "entry_decision"),
        "exit": (4.4, 0.15, "exit_rails"),
        "session": (5.5, 0.15, "replay_session\n★ 状态机"),
    }

    def _node_status(key: str) -> str:
        if key in ("label", "fill"):
            return "active"
        if key == "infer":
            return _lag_status(ctx.eng_lag if ctx.eng_lag < 900 else None)
        if key == "l1":
            return "info"
        if key == "l2":
            return "info"
        if key in ("s1", "s2", "merge", "lmdb", "train"):
            return "idle"
        if key in ("entry", "exit", "session"):
            return "ok" if ctx.eng_lag < 15 else "warn"
        return "off"

    fig = go.Figure()
    edge_c = "#9CA3AF" if ctx.is_light else "#555555"
    chain = ["s1", "s2", "merge", "label", "lmdb", "train", "infer"]
    for u, v in zip(chain, chain[1:]):
        x0, y0, _ = nodes[u]
        x1, y1, _ = nodes[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=2, color=edge_c), hoverinfo="none", showlegend=False,
            )
        )
    for u, v in [("infer", "l1"), ("infer", "l2")]:
        x0, y0, _ = nodes[u]
        x1, y1, _ = nodes[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=2, color=edge_c, dash="dot"),
                hoverinfo="none", showlegend=False,
            )
        )
    cchain = ["fill", "entry", "exit", "session"]
    for u, v in zip(cchain, cchain[1:]):
        x0, y0, _ = common_nodes[u]
        x1, y1, _ = common_nodes[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=3, color="#636EFA"), hoverinfo="none", showlegend=False,
            )
        )
    lx, ly = nodes["label"][0], nodes["label"][1]
    fx, fy = common_nodes["fill"][0], common_nodes["fill"][1]
    fig.add_trace(
        go.Scatter(
            x=[lx, fx, None], y=[ly - 0.35, fy + 0.35, None],
            mode="lines", line=dict(width=2, color="#636EFA", dash="dash"),
            hoverinfo="none", showlegend=False,
        )
    )

    def _add_block(node_dict: dict, size: int = 44):
        xs, ys, texts, colors, hovers = [], [], [], [], []
        for key, (x, y, label) in node_dict.items():
            xs.append(x)
            ys.append(y)
            texts.append(label)
            colors.append(_status_color(_node_status(key), ctx.is_light))
            hovers.append(label.replace("\n", " "))
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(size=size, color=colors, line=dict(width=2, color="white")),
                text=texts, textposition="middle center",
                textfont=dict(size=9, color="white" if not ctx.is_light else "#111827"),
                hovertext=hovers, hoverinfo="text", showlegend=False,
            )
        )

    _add_block(nodes, 48)
    _add_block(common_nodes, 42)

    fig.add_annotation(x=3.3, y=1.55, text="① DATA preprocess/", showarrow=False, font=dict(size=10, color="#888"))
    fig.add_annotation(x=4.4, y=-0.25, text="② COMMON 单一真相源", showarrow=False, font=dict(size=10, color="#888"))
    fig.add_annotation(x=6.6, y=1.55, text="③ MODEL", showarrow=False, font=dict(size=10, color="#888"))
    fig.add_annotation(x=7.7, y=1.55, text="④ REPLAY G2", showarrow=False, font=dict(size=10, color="#888"))

    fig.update_layout(
        title=dict(
            text=f"📦 离线管线 & 回放验收 · {ctx.symbol}<br>"
                 f"<sup>fill_model 贯穿 标签→replay→实盘 OMS | eng_lag={ctx.eng_lag:.1f}s</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.4, 8.4]),
        yaxis=dict(visible=False, range=[-0.35, 1.75]),
        height=280,
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_live_triprocess_figure(ctx: SystemFlowContext) -> go.Figure:
    """⑤ 实盘三进程 + Redis 流(实时着色)。"""
    nodes = {
        "ibkr": (0.0, 1.0, "IBKR\ntick/1m"),
        "fcs": (1.5, 1.0, "FCS\nfeature_compute"),
        "signal": (3.2, 1.0, "Signal\nrun_live_signal"),
        "oms": (4.9, 1.0, "OMS\nrun_live_exec"),
        "ibkr_out": (6.4, 1.0, "IBKR\n订单/fill"),
    }
    streams = {
        "fused": (0.75, 0.35, "fused_market\n1s tick"),
        "infer": (2.35, 0.35, "unified_inference\n1m features"),
        "alpha": (4.05, 0.35, "orch_trade_signals\nALPHA_FRAME"),
    }

    ibkr_st = "ok" if ctx.ibkr_connected and ctx.src_lag < 10 else (
        "warn" if ctx.ibkr_connected else "crit"
    )
    fcs_st = _lag_status(ctx.raw_lag if ctx.raw_lag < 900 else None)
    sig_st = _lag_status(ctx.eng_lag if ctx.eng_lag < 900 else None)
    oms_st = "active" if ctx.position != 0 else (
        "ok" if ctx.entry_result == "BUY" else "idle"
    )

    status_map = {"ibkr": ibkr_st, "fcs": fcs_st, "signal": sig_st, "oms": oms_st, "ibkr_out": ibkr_st}

    fig = go.Figure()
    edge_c = "#9CA3AF" if ctx.is_light else "#555555"
    hl = "#636EFA" if ctx.is_light else "#818CF8"

    main_chain = ["ibkr", "fcs", "signal", "oms", "ibkr_out"]
    for u, v in zip(main_chain, main_chain[1:]):
        x0, y0, _ = nodes[u]
        x1, y1, _ = nodes[v]
        highlight = u in ("signal", "oms") and ctx.position != 0
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=4 if highlight else 2, color=hl if highlight else edge_c),
                hoverinfo="none", showlegend=False,
            )
        )

    def _stream_edge(x0, y0, x1, y1):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=1.5, color="#19D3F3", dash="dot"),
                hoverinfo="none", showlegend=False,
            )
        )

    _stream_edge(nodes["ibkr"][0], nodes["ibkr"][1] - 0.3, streams["fused"][0], streams["fused"][1] + 0.2)
    _stream_edge(streams["fused"][0], streams["fused"][1] + 0.2, nodes["fcs"][0], nodes["fcs"][1] - 0.3)
    _stream_edge(nodes["fcs"][0], nodes["fcs"][1] - 0.3, streams["infer"][0], streams["infer"][1] + 0.2)
    _stream_edge(streams["infer"][0], streams["infer"][1] + 0.2, nodes["signal"][0], nodes["signal"][1] - 0.3)
    _stream_edge(nodes["signal"][0], nodes["signal"][1] - 0.3, streams["alpha"][0], streams["alpha"][1] + 0.2)
    _stream_edge(streams["alpha"][0], streams["alpha"][1] + 0.2, nodes["oms"][0], nodes["oms"][1] - 0.3)
    _stream_edge(streams["fused"][0], streams["fused"][1] + 0.2, nodes["oms"][0], nodes["oms"][1] - 0.45)

    xs, ys, texts, colors, hovers, sizes = [], [], [], [], [], []
    for key, (x, y, label) in {**nodes, **streams}.items():
        xs.append(x)
        ys.append(y)
        texts.append(label)
        st_key = key if key in status_map else "info"
        colors.append(_status_color(status_map.get(st_key, "info"), ctx.is_light))
        sizes.append(52 if key in nodes else 36)
        note = ""
        if key == "signal" and ctx.exec_topo:
            note = f"α={ctx.exec_topo.alpha:+.3f} {ctx.exec_topo.side_hint}"
        elif key == "oms":
            note = f"pos={ctx.position} entry={ctx.entry_result}"
        elif key == "infer":
            note = f"lag={ctx.eng_lag:.1f}s"
        hovers.append(f"{label.replace(chr(10), ' ')}<br>{note}")

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
            text=texts, textposition="middle center",
            textfont=dict(size=9, color="white" if not ctx.is_light else "#111827"),
            hovertext=hovers, hoverinfo="text", showlegend=False,
        )
    )

    patch_note = "qqq_btc 薄层" if ctx.qqq_btc_live_hint else "legacy SE/OMS"
    fig.update_layout(
        title=dict(
            text=f"⚡ 实盘三进程 · {ctx.run_mode}<br>"
                 f"<sup>{patch_note} | tick=disaster_only | 分钟=exit_rails | session={ctx.session}</sup>",
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 7.0]),
        yaxis=dict(visible=False, range=[0.0, 1.45]),
        height=300,
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_decision_flow_figure(ctx: SystemFlowContext) -> go.Figure:
    """入场/出场决策链(与 replay_session 同语义)。"""
    steps = [
        (0, "FCS 1m bar", _lag_status(ctx.eng_lag if ctx.eng_lag < 900 else None)),
        (1, "net_edge\n推理", _lag_status(ctx.eng_lag if ctx.eng_lag < 900 else None)),
        (2, "entry_decision\n阈值/点差", "ok" if ctx.entry_result == "BUY" else "idle"),
        (3, "pending\nentry_delay", "warn" if ctx.entry_result not in ("-", "REJECT") else "off"),
        (4, "fill 0.775\nOMS 限价", "active" if ctx.position != 0 else "off"),
        (5, "持仓 rails\n分钟 check_exit", "active" if ctx.position != 0 else "off"),
        (6, "tick\ndisaster_stop", "info"),
        (7, "平仓/审计", "ok" if ctx.exit_result.startswith("SELL") else "off"),
    ]
    fig = go.Figure()
    edge_c = "#9CA3AF" if ctx.is_light else "#555555"
    xs = [s[0] for s in steps]
    for i in range(len(steps) - 1):
        fig.add_trace(
            go.Scatter(
                x=[xs[i], xs[i + 1], None], y=[0, 0, None],
                mode="lines", line=dict(width=3, color=edge_c), hoverinfo="none", showlegend=False,
            )
        )
    labels = [s[1] for s in steps]
    colors = [_status_color(s[2], ctx.is_light) for s in steps]
    fig.add_trace(
        go.Scatter(
            x=xs, y=[0] * len(steps), mode="markers+text",
            marker=dict(size=40, color=colors, line=dict(width=2, color="white")),
            text=labels, textposition="top center",
            textfont=dict(size=8, color="#374151" if ctx.is_light else "#E5E7EB"),
            hovertext=labels, hoverinfo="text", showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(text="🔀 决策链(replay = live 同口径)", x=0.02, font=dict(size=13)),
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 7.5]),
        yaxis=dict(visible=False, range=[-0.8, 0.6]),
        height=180,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _gate_badge(status: str) -> str:
    return {"ok": "🟢", "warn": "🟡", "crit": "🔴", "pending": "⚪", "info": "🔵"}.get(status, "⚪")


def render_system_flow_header(
    ctx: SystemFlowContext,
    *,
    show_charts: bool = True,
) -> None:
    """页顶紧凑摘要(替代旧双拓扑占屏)。"""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Symbol", ctx.symbol)
    c2.metric("Run Mode", ctx.run_mode)
    c3.metric("Session", ctx.session)
    edge_disp = ctx.net_edge if ctx.net_edge else (ctx.exec_topo.alpha if ctx.exec_topo else 0.0)
    c4.metric("net_edge / α", f"{edge_disp:+.4f}")
    c5.metric("持仓", ctx.position, delta=ctx.exec_topo.side_hint if ctx.exec_topo else None)
    lag_max = max(ctx.src_lag, ctx.raw_lag, ctx.eng_lag)
    c6.metric("Max Lag", f"{lag_max:.1f}s", delta="ok" if lag_max < 5 else "warn")

    if show_charts and ctx.exec_topo is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(draw_live_triprocess_figure(ctx), use_container_width=True)
        with col_b:
            st.plotly_chart(draw_exec_topo_figure(ctx.exec_topo, is_light=ctx.is_light), use_container_width=True)


def build_flow_context(
    *,
    symbol: str,
    run_mode: str,
    status: Any,
    exec_topo: ExecTopoState,
    is_light: bool,
    ibkr_conn: Optional[dict] = None,
    stream_rows: Optional[List[dict]] = None,
    r=None,
) -> SystemFlowContext:
    """从 dashboard 已有 SystemStatus / ExecTopoState 组装上下文。"""
    qqq_hint = os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")
    if r is not None and not qqq_hint:
        try:
            raw = r.get("meta:qqq_btc_live")
            if raw:
                qqq_hint = str(raw.decode() if isinstance(raw, bytes) else raw).lower() in ("1", "true", "yes")
        except Exception:
            pass

    alpha = float(exec_topo.alpha or 0.0)
    ctx = SystemFlowContext(
        symbol=str(symbol).upper(),
        run_mode=str(run_mode),
        is_light=is_light,
        src_lag=float(getattr(status, "src_lag", 999.0) or 999.0),
        raw_lag=float(getattr(status, "raw_lag", 999.0) or 999.0),
        eng_lag=float(getattr(status, "eng_lag", 999.0) or 999.0),
        ibkr_state=str((ibkr_conn or {}).get("state", "unknown")),
        ibkr_connected=bool((ibkr_conn or {}).get("connected", False)),
        session=str(exec_topo.session or "unknown"),
        alpha=alpha,
        net_edge=alpha,
        position=int(exec_topo.position or 0),
        entry_result=str(exec_topo.entry_result or "-"),
        exit_result=str(exec_topo.exit_result or "-"),
        stream_rows=list(stream_rows or []),
        exec_topo=exec_topo,
        qqq_btc_live_hint=qqq_hint,
    )
    return ctx


def render_system_flow_tab(ctx: SystemFlowContext) -> None:
    """系统全景 Tab 主内容。"""
    st.header("🗺️ 系统全景 — qqq_btc 端到端流程")
    st.caption(
        "可视化 ARCHITECTURE.md 全链路: 离线 DATA → COMMON 单一真相源 → MODEL → REPLAY 验收 → 实盘三进程。"
        "本页只读 Redis/PG/Stream，不会修改任何交易状态。"
    )

    gates = infer_gate_snapshot(ctx)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("G0 数据/FCS", _gate_badge(gates.g0), gates.notes.get("G0", ""))
    g2.metric("G1 实盘链路", _gate_badge(gates.g1), gates.notes.get("G1", ""))
    g3.metric("G2 strict replay", _gate_badge(gates.g2), gates.notes.get("G2", ""))
    g4.metric("G3 parity 影子", _gate_badge(gates.g3), gates.notes.get("G3", ""))

    if ctx.qqq_btc_live_hint:
        st.info("QQQ_BTC_LIVE 已启用: Signal/OMS 走 qqq_btc 薄层(0.775 限价、exit_rails、fill 审计)。")
    else:
        st.caption("当前 Dashboard 进程未检测到 QQQ_BTC_LIVE;拓扑展示 legacy + 目标形态对照。")

    st.plotly_chart(draw_offline_pipeline_figure(ctx), use_container_width=True)

    col_live, col_dec = st.columns([3, 2])
    with col_live:
        st.plotly_chart(draw_live_triprocess_figure(ctx), use_container_width=True)
    with col_dec:
        st.plotly_chart(draw_decision_flow_figure(ctx), use_container_width=True)

    if ctx.exec_topo is not None:
        st.plotly_chart(draw_exec_topo_figure(ctx.exec_topo, is_light=ctx.is_light), use_container_width=True)

    st.subheader("🔗 Redis Stream 消费")
    if not ctx.stream_rows:
        st.info("暂无 Stream 健康数据 — 见「Stream Health」Tab 或等待缓存刷新。")
    else:
        import pandas as pd

        color_map = {"ok": "🟢", "warn": "🟡", "err": "🔴", "gray": "⚪"}
        rows_view = []
        for row in ctx.stream_rows:
            lag = row.get("lag")
            try:
                lag_i = int(lag) if lag is not None else 0
            except Exception:
                lag_i = 0
            if not row.get("stream_exists"):
                level, msg = "gray", "stream 不存在"
            elif not row.get("group_exists"):
                level, msg = "warn", "组未创建"
            elif lag_i >= 500:
                level, msg = "err", f"lag={lag_i}"
            elif lag_i >= 50:
                level, msg = "warn", f"lag={lag_i}"
            else:
                level, msg = "ok", "healthy"
            rows_view.append({
                "状态": f"{color_map.get(level, '⚪')} {msg}",
                "Stream": row.get("stream", ""),
                "Group": row.get("group", ""),
                "lag": lag if lag is not None else "-",
                "pending": row.get("pending") if row.get("pending") is not None else "-",
            })
        st.dataframe(pd.DataFrame(rows_view), use_container_width=True, hide_index=True)

    with st.expander("📋 架构不变量 & 启动命令", expanded=False):
        st.markdown(
            """
**不变量**
- 标签、strict replay、实盘 OMS 共用 `fill_model`(0.775) + `replay_session`
- 分钟退出: `exit_rails.check_exit`; tick 仅 `check_disaster_stop`
- 绝对 net_edge，无截面 z-score

**启动(qqq_btc 双引擎)**
```bash
QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_signal_qqq.py --checkpoint <best.pth>
QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_exec_qqq.py
```

**验收**
```bash
python qqq_btc/tools/run_replay.py          # G2 strict
python qqq_btc/tools/parity_audit.py fill   # G3 对账
```
            """
        )

    st.caption(f"刷新于 {time.strftime('%H:%M:%S')} · 文档: qqq_btc/ARCHITECTURE.md")
