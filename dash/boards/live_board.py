"""Live board — Shadow/Paper/Live sessions + positions + sliding windows."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from boards.common import metric, redis_client, render_contract_banner
from sources import (
    ALIGNMENT_GAPS,
    feed_health_frame,
    fetch_connector_status,
    fetch_locks_payload,
    fetch_oms_meta,
    gate_reject_frame,
    live_ops_overview,
    live_session_frames,
    load_exit_health,
    load_order_events,
    load_prevention,
    load_tape_parity,
    load_watchdog_hunt,
    locks_frame,
    locks_summary_frame,
    mf_series_for_symbol,
    ohlcv_1m_for_symbol,
    process_snapshot,
    reconcile_compare_frame,
    redis_snapshot,
    resolve_live_trace_bundle,
    stream_probe,
    subscription_frame,
    tape_inventory,
    trade_spreads_from_events,
)
from boards.tv_chart import lock_markers, render_tv_kline

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None


def _request_auto_refresh(enabled: bool) -> None:
    """Defer sleep/rerun until all Live tabs have rendered.

    Streamlit executes every ``with tab:`` block; an early ``st.rerun()``
    aborts later tabs and leaves them blank.
    """
    if enabled:
        st.session_state["_maga7_live_auto_refresh"] = True


def _flush_auto_refresh(*, seconds: float = 5.0) -> None:
    if st.session_state.pop("_maga7_live_auto_refresh", False):
        time.sleep(float(seconds))
        st.rerun()


def render_live_board(
    *,
    host: str,
    port: int,
    db: int,
    profile: dict,
    live_sessions,
) -> None:
    st.markdown("### ④ Live / 实盘")
    st.caption(
        "与对拍同一套决策栈；数据源换成 IBKR，Paper/Live 换成真限价。"
        "本页只读：持仓、Feed、门禁/对账、session 证据、启停命令提示。"
    )
    render_contract_banner()
    # Reset each run; tabs may re-arm via checkbox.
    st.session_state["_maga7_live_auto_refresh"] = False

    (
        tab_topo,
        tab_subs,
        tab_pos,
        tab_feed,
        tab_gates,
        tab_sess,
        tab_redis,
        tab_news,
        tab_ops,
    ) = st.tabs(
        [
            "数据流拓扑",
            "订阅 / 锁约",
            "持仓 / 滑动窗口",
            "Feed 健康",
            "门禁 / 对账",
            "Live sessions",
            "Redis 链路",
            "事件/新闻审核",
            "启停命令",
        ]
    )

    with tab_topo:
        _render_topology(host, port, db, profile)
    with tab_subs:
        _render_subscriptions_locks(host, port, db, profile)
    with tab_pos:
        _render_positions(host, port, db, profile)
    with tab_feed:
        _render_feed_health(host, port, db, profile)
    with tab_gates:
        _render_gates_reconcile(host, port, db, profile)
    with tab_sess:
        _render_sessions(live_sessions)
    with tab_redis:
        _render_redis(host, port, db)
    with tab_news:
        from boards.event_news_board import render_event_news_board

        render_event_news_board()
    with tab_ops:
        _render_ops(profile)

    _flush_auto_refresh()


def _render_subscriptions_locks(
    host: str, port: int, db: int, profile: dict
) -> None:
    """Subscribed underlyings (spot chart) + open-ladder lock results / trigger time."""
    st.caption(
        "已订阅个股 TradingView K 线（复用 production futu_kline / Lightweight Charts）"
        " + open ladder 锁约结果。"
        "K 线数据：`tape/{pre|rth|post}` 秒级聚合 1m + scanner 1m；"
        "锁约来自 `locks.json`（`lock_ts` / `lock_spot`）。"
    )
    auto = st.checkbox("自动刷新 (5s)", value=False, key="live_subs_auto")
    client = redis_client(host, port, db)
    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            st.warning(f"Redis 不可用，将尝试磁盘：{exc}")
            client = None

    probe = resolve_live_trace_bundle(client, prefer_disk=client is None)
    ids = probe.get("session_ids") or (
        [] if not probe.get("session_id") else [probe["session_id"]]
    )
    if not ids:
        st.info("未发现 Mag7 live session。")
        return
    default_i = ids.index(probe["session_id"]) if probe.get("session_id") in ids else 0
    session_id = st.selectbox("Session", ids, index=default_i, key="live_subs_session")
    bundle = resolve_live_trace_bundle(client, session_id=session_id, prefer_disk=False)
    session_dir = Path(bundle["session_dir"]) if bundle.get("session_dir") else None
    connector = fetch_connector_status(client, session_id=session_id) if client else {}
    if not connector and session_dir is not None:
        connector = ((bundle.get("manifest") or {}).get("connector") or {})
    overview = live_ops_overview(
        client, session_id=session_id, session_dir=session_dir, profile=profile
    )
    locks_payload = fetch_locks_payload(session_dir)
    sub_df = subscription_frame(
        connector, profile=profile, locks_payload=locks_payload
    )
    lock_sum = locks_summary_frame(locks_payload)
    lock_df = locks_frame(locks_payload)

    lock_status = (
        locks_payload.get("status")
        or connector.get("lock_status")
        or overview.get("connector", {}).get("lock_status")
        or "-"
    )
    trigger_ts = None
    if not lock_sum.empty and "lock_ts" in lock_sum.columns:
        nums = pd.to_numeric(lock_sum["lock_ts"], errors="coerce").dropna()
        if not nums.empty:
            trigger_ts = float(nums.min())
    if trigger_ts is None and locks_payload.get("updated_at") is not None:
        try:
            trigger_ts = float(locks_payload["updated_at"])
        except Exception:
            trigger_ts = None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lock", str(lock_status))
    m2.metric(
        "锁定触发",
        time.strftime("%H:%M:%S", time.localtime(trigger_ts)) if trigger_ts else "未锁",
    )
    m3.metric("订阅标的", int(sub_df["subscribed"].sum()) if not sub_df.empty else 0)
    m4.metric("锁约合约数", int(len(lock_df)) if not lock_df.empty else 0)
    m5.metric("时段", overview.get("session_phase") or "-")

    st.markdown("**已订阅个股**")
    if sub_df.empty:
        st.info("尚无 stock_feed；确认 connector 已启动并订阅。")
    else:
        show = sub_df.copy()
        if "spot" in show.columns:
            show["spot"] = show["spot"].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) and x not in ("", None) else ""
            )
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("**个股 TradingView K 线**")
    chart_syms = []
    if not sub_df.empty:
        chart_syms = [
            str(s) for s in sub_df.loc[sub_df["subscribed"].astype(bool), "symbol"].tolist()
        ]
    if not chart_syms:
        chart_syms = list(((profile.get("profile") or {}).get("symbols") or []))
    if chart_syms:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sym = st.selectbox("图表标的", chart_syms, key="live_subs_tv_sym")
        with c2:
            color_mode = st.selectbox(
                "涨跌色",
                options=["us", "cn"],
                format_func=lambda x: "美式(绿涨)" if x == "us" else "中式(红涨)",
                key="live_subs_tv_color",
            )
        with c3:
            theme_mode = st.selectbox(
                "主题",
                options=["light", "dark"],
                key="live_subs_tv_theme",
            )
        trade_date = ""
        if session_dir is not None:
            trade_date = session_dir.parent.name
        if not trade_date:
            trade_date = time.strftime("%Y-%m-%d")
        candle = ohlcv_1m_for_symbol(
            sym,
            scanner_state=bundle.get("scanner") or {},
            session_dir=session_dir,
            phases=("pre", "rth", "post"),
        )
        sym_lock_ts = trigger_ts
        if not lock_sum.empty and "symbol" in lock_sum.columns:
            hit = lock_sum[lock_sum["symbol"].astype(str) == str(sym)]
            if not hit.empty and pd.notna(hit.iloc[0].get("lock_ts")):
                try:
                    sym_lock_ts = float(hit.iloc[0]["lock_ts"])
                except Exception:
                    pass
        if candle.empty:
            st.caption("暂无 tape / scanner 1m 棒，等行情积累后再看 K 线。")
        else:
            st.caption(
                f"{sym} · {len(candle)} 根 1m · "
                f"{time.strftime('%H:%M', time.localtime(float(candle.iloc[0]['ts'])))}"
                f" → {time.strftime('%H:%M', time.localtime(float(candle.iloc[-1]['ts'])))}"
                + (
                    f" · lock={time.strftime('%H:%M:%S', time.localtime(sym_lock_ts))}"
                    if sym_lock_ts
                    else ""
                )
            )
            render_tv_kline(
                sym,
                candle,
                chart_date=trade_date,
                color_mode=color_mode,
                theme_mode=theme_mode,
                markers=lock_markers(sym_lock_ts, text="LOCK"),
                key=f"live_subs_tv_{session_id}_{sym}_{trade_date}",
            )
            with st.expander("OHLCV 尾部（调试）"):
                show_c = candle.tail(40).copy()
                show_c["time"] = show_c["ts"].map(
                    lambda x: time.strftime("%H:%M:%S", time.localtime(float(x)))
                )
                st.dataframe(
                    show_c[["time", "open", "high", "low", "close", "volume"]],
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("**合约锁定结果（按标的）**")
    if lock_sum.empty:
        st.info(
            "尚未写入 locks.json（盘前 STARTING / 未到 09:30 锁约属预期）。"
            "锁定后会显示触发时间与 lock_spot。"
        )
    else:
        st.dataframe(lock_sum, use_container_width=True, hide_index=True)

    st.markdown("**锁定合约明细**")
    if lock_df.empty:
        st.caption("无锁约明细")
    else:
        show_l = lock_df.copy()
        for col in ("strike", "lock_spot"):
            if col in show_l.columns:
                show_l[col] = show_l[col].map(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) else ""
                )
        cols = [
            c
            for c in (
                "symbol",
                "right",
                "strike",
                "dte",
                "rung",
                "localSymbol",
                "lock_spot",
                "lock_time",
                "expiry",
                "conId",
            )
            if c in show_l.columns
        ]
        st.dataframe(show_l[cols], use_container_width=True, hide_index=True)

    errs = (locks_payload or {}).get("errors") or {}
    if errs:
        st.warning(f"锁约错误: {errs}")

    _render_tape_panel(session_dir, phase=str(overview.get("session_phase") or ""))

    with st.expander("locks.json raw"):
        st.json(locks_payload if locks_payload else {})

    _request_auto_refresh(auto)


def _health_color(health: str) -> str:
    text = str(health or "")
    if "🔴" in text:
        return "#9b2226"
    if "🟠" in text:
        return "#bb3e03"
    if "🟢" in text:
        return "#2d6a4f"
    return "#6c757d"


def _topology_flow_html(topology: list[dict]) -> str:
    """Horizontal node flow for Mag7 live data path."""
    if not topology:
        return "<p>暂无拓扑数据</p>"
    chunks: list[str] = []
    for i, node in enumerate(topology):
        color = _health_color(str(node.get("health") or ""))
        age = node.get("age_sec")
        age_txt = f"{age}s" if age is not None else "-"
        detail = str(node.get("detail") or "").replace("<", "&lt;").replace(">", "&gt;")
        if len(detail) > 72:
            detail = detail[:69] + "…"
        chunks.append(
            (
                f'<div style="min-width:120px;max-width:170px;padding:10px 12px;'
                f'border-radius:8px;background:{color};color:#fff;text-align:center;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.18);">'
                f'<div style="font-weight:700;font-size:14px;">{node.get("node")}</div>'
                f'<div style="margin-top:4px;font-size:12px;">{node.get("health")}</div>'
                f'<div style="margin-top:2px;opacity:.9;font-size:11px;">age {age_txt}</div>'
                f'<div style="margin-top:6px;opacity:.85;font-size:10px;line-height:1.25;'
                f'word-break:break-all;">{detail}</div>'
                f"</div>"
            )
        )
        if i < len(topology) - 1:
            chunks.append(
                '<div style="align-self:center;padding:0 6px;color:#adb5bd;'
                'font-size:20px;font-weight:700;">→</div>'
            )
    return (
        '<div style="display:flex;flex-wrap:wrap;align-items:stretch;gap:4px;'
        f'padding:8px 0;">{"".join(chunks)}</div>'
    )


def _render_topology(host: str, port: int, db: int, profile: dict) -> None:
    """At-a-glance Mag7 data-flow health (IBKR → Fused → Scanner → OMS → Disk)."""
    st.caption(
        "新架构数据流：IB Gateway → Connector → Redis Fused → Scanner（含 stock_by）"
        " → OMS → Session 磁盘。盘前/盘后走 `tape/pre|post` + `*:pre|*:post` 校验流"
        "（允许 partial）；盘中权威流仍要求全标的齐套。本页只读。"
    )
    auto = st.checkbox("自动刷新 (5s)", value=False, key="live_topo_auto")
    client = redis_client(host, port, db)
    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            st.warning(f"Redis 不可用，将尝试磁盘：{exc}")
            client = None

    probe = resolve_live_trace_bundle(client, prefer_disk=client is None)
    ids = probe.get("session_ids") or (
        [] if not probe.get("session_id") else [probe["session_id"]]
    )
    if not ids:
        st.info("未发现 Mag7 live session。先 `./start_maga7_live_session.sh start dry`。")
        return
    default_i = ids.index(probe["session_id"]) if probe.get("session_id") in ids else 0
    session_id = st.selectbox("Session", ids, index=default_i, key="live_topo_session")
    bundle = resolve_live_trace_bundle(client, session_id=session_id, prefer_disk=False)
    session_dir = Path(bundle["session_dir"]) if bundle.get("session_dir") else None
    overview = live_ops_overview(
        client, session_id=session_id, session_dir=session_dir, profile=profile
    )
    topology = overview.get("topology") or []

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("整体", overview.get("topology_overall") or "-")
    m2.metric("时段", overview.get("session_phase") or "-")
    m3.metric("环境", overview.get("env_label") or "-")
    m4.metric("Arm", overview.get("arm_label") or "-")
    m5.metric(
        "stock_by",
        f"{overview.get('stock_by_syms') or 0} sym / {overview.get('stock_by_bars') or 0} bars",
    )
    m6.metric("QQQ bars", overview.get("qqq_bars") if overview.get("qqq_bars") is not None else 0)

    phase = str(overview.get("session_phase") or "")
    overall = str(overview.get("topology_overall") or "")
    if phase in {"PRE", "POST"}:
        st.info(
            f"当前 {phase}：只验证 IBKR→校验流→`tape/{phase.lower()}`；"
            "Scanner/OMS 显示 Idle 属预期，不计入整体健康。"
        )
    red_nodes = [
        str(n.get("node") or "")
        for n in topology
        if "🔴" in str(n.get("health") or "")
    ]
    warn_nodes = [
        str(n.get("node") or "")
        for n in topology
        if "🟠" in str(n.get("health") or "")
    ]
    if "🔴" in overall:
        focus = " / ".join(red_nodes) if red_nodes else "IBKR / Redis Fused / Disk(tape)"
        st.error(f"数据流不健康：先查 {focus} 节点。")
    elif "🟠" in overall:
        focus = " / ".join(warn_nodes) if warn_nodes else "部分节点"
        st.warning(
            f"数据流降级（{focus}）：有节点滞后或 Warm（盘前可能因稀疏成交尚未齐套）。"
        )
    elif "🟢" in overall:
        st.success("数据流健康。" + (f"（{phase} 校验模式）" if phase in {"PRE", "POST"} else ""))

    _render_tape_parity_card(session_dir)
    _render_watchdog_hunt_card(session_dir, session_id=session_id, client=client)
    _render_prevention_card(session_dir)
    _render_exit_arms_card(session_dir, session_id=session_id, client=client)

    st.markdown("**链路**")
    st.markdown(_topology_flow_html(topology), unsafe_allow_html=True)

    st.markdown("**节点明细**")
    st.dataframe(
        pd.DataFrame(topology),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**进程探针**")
        procs = process_snapshot()
        st.dataframe(pd.DataFrame(procs), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**关键键**")
        stream_key = None
        connector = overview.get("connector") or {}
        if isinstance(connector, dict):
            stream_key = connector.get("stream")
        if not stream_key and session_id:
            stream_key = f"fused_market_stream:maga7:{session_id}"
        active_stream = overview.get("stream_key") or stream_key
        st.code(
            "\n".join(
                [
                    f"session={session_id}",
                    f"phase={overview.get('session_phase') or '-'}",
                    f"active_stream={active_stream or '-'}",
                    f"auth_stream={stream_key or '-'}",
                    f"feed_health=maga7:feed_health:{session_id}",
                    f"scanner_state=maga7:scanner_state:{session_id}",
                    f"oms_meta=maga7:oms_meta:{session_id}",
                    f"tape={session_dir / 'tape' if session_dir else '-'}",
                    f"disk={session_dir or '-'}",
                ]
            ),
            language="text",
        )
        if active_stream and client is not None:
            st.json(stream_probe(client, str(active_stream)))

    _render_tape_panel(session_dir, phase=str(overview.get("session_phase") or ""))

    _request_auto_refresh(auto)


def _render_tape_parity_card(session_dir: Path | None) -> None:
    """Compact card for the latest intraday tape↔Scanner parity run."""
    st.markdown("**最近对拍**（tape → shadow Scanner vs live signals，约每 10 分钟）")
    report = load_tape_parity(session_dir)
    if report.get("_missing"):
        st.info(
            "尚无 `tape_parity.json`。"
            " 启动：`./maga7/SHELL/maga7_system.sh watch` 或 `parity`。"
        )
        return

    ok = bool(report.get("ok"))
    stage = str(report.get("stage") or "-")
    age = report.get("_age_sec")
    age_txt = f"{age:.0f}s" if isinstance(age, (int, float)) else "-"
    note = str(report.get("note") or "")
    issues = report.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]

    if ok:
        st.success(f"对拍通过 · stage=`{stage}` · 报告年龄 {age_txt}")
    else:
        st.error(f"对拍失败 · stage=`{stage}` · 报告年龄 {age_txt}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ok", "✅" if ok else "❌")
    c2.metric("stage", stage)
    c3.metric("matched", report.get("matched") if report.get("matched") is not None else "-")
    c4.metric(
        "live / replay",
        f"{report.get('live_signals', '-')} / {report.get('replay_signals', '-')}",
    )
    c5.metric("报告年龄", age_txt)
    health = report.get("health") if isinstance(report.get("health"), dict) else {}
    c6.metric(
        "tape/rth",
        (
            f"{health.get('tape_rth_fresh_sec'):.0f}s"
            if isinstance(health.get("tape_rth_fresh_sec"), (int, float))
            else "-"
        ),
    )

    if note:
        st.caption(f"note: {note}")
    if issues:
        # Keep compact: show first few issues inline.
        shown = [str(i) for i in issues[:5]]
        st.caption("issues: " + " · ".join(shown) + (" …" if len(issues) > 5 else ""))

    only_live = report.get("only_live") or []
    only_replay = report.get("only_replay") or []
    if only_live or only_replay:
        with st.expander(
            f"信号差分 only_live={len(only_live)} only_replay={len(only_replay)}",
            expanded=False,
        ):
            left, right = st.columns(2)
            with left:
                st.markdown("only_live")
                st.dataframe(
                    pd.DataFrame(only_live, columns=["minute", "symbol", "dir", "contract"])
                    if only_live and isinstance(only_live[0], (list, tuple))
                    else pd.DataFrame({"row": [str(x) for x in only_live]}),
                    use_container_width=True,
                    hide_index=True,
                )
            with right:
                st.markdown("only_replay")
                st.dataframe(
                    pd.DataFrame(
                        only_replay, columns=["minute", "symbol", "dir", "contract"]
                    )
                    if only_replay and isinstance(only_replay[0], (list, tuple))
                    else pd.DataFrame({"row": [str(x) for x in only_replay]}),
                    use_container_width=True,
                    hide_index=True,
                )

    st.caption(f"`{report.get('path')}` · log: `logs/maga7/tape_parity.log`")


def _render_watchdog_hunt_card(
    session_dir: Path | None,
    *,
    session_id: str | None = None,
    client=None,
) -> None:
    """P3：当日 Watchdog 状态 + Hunt 候选/发出计数（可与基线单区分）。"""
    st.markdown("**Watchdog / Hunt**（Halt · Degrade · Hunt 槽）")
    report = load_watchdog_hunt(session_dir, client=client, session_id=session_id)
    if report.get("_missing"):
        st.info(
            "尚无 Watchdog 快照。live session 写入 `scanner_state.watchdog` / "
            "`oms_meta.watchdog` 后可见（需含 Hunt 注入的进程）。"
        )
        return
    state = str(report.get("state") or "off").upper()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("state", state)
    c2.metric("hunt_armed", "YES" if report.get("hunt_armed") else "no")
    c3.metric(
        "hunt cand/emit",
        f"{report.get('n_hunt_candidates', 0)}/{report.get('n_hunt_emitted', 0)}",
        delta=f"pending={report.get('pending_hunts', 0)}",
    )
    c4.metric(
        "budget/mutex skip",
        f"{report.get('n_hunt_budget_skip', 0)}/{report.get('n_hunt_mutex_skip', 0)}",
    )
    c5.metric("day_halt", "YES" if report.get("day_halt") else "no")
    syms = report.get("day_hunt_symbols") or []
    st.caption(
        f"reason=`{report.get('reason') or '-'}` · route=`{report.get('route') or '-'}` · "
        f"hunt_syms={syms or '-'} · src=`{report.get('source') or '-'}`"
    )
    cands = report.get("candidates") or []
    if cands:
        with st.expander(f"Hunt 候选 ({len(cands)})", expanded=False):
            st.dataframe(pd.DataFrame(cands), use_container_width=True, hide_index=True)


def _render_prevention_card(session_dir: Path | None) -> None:
    """早盘预测性 prevention（非连亏熔断）。"""
    st.markdown("**早盘 Prevention**（因果特征 → expert；不是连亏熔断）")
    report = load_prevention(session_dir)
    if report.get("_missing"):
        st.info(
            "尚无 `prevention.json`。Watchdog 评估后由 live engine 随 "
            "`scanner_state` 写入；resume 后可见。"
        )
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("enabled", "ON" if report.get("enabled") else "OFF")
    c2.metric("state", str(report.get("state") or report.get("watchdog_state") or "-"))
    c3.metric("expert", str(report.get("expert") or "-"))
    c4.metric(
        "prefer_risk_off",
        "yes" if report.get("prefer_risk_off") else "no",
    )
    st.caption(
        f"rule=`{report.get('rule') or '-'}` · reason=`{report.get('reason') or report.get('watchdog_reason') or '-'}` · "
        f"route=`{report.get('route_tag') or report.get('watchdog_route') or '-'}` · "
        f"`{report.get('source') or '-'}`"
    )


def _render_exit_arms_card(
    session_dir: Path | None,
    *,
    session_id: str | None = None,
    client=None,
) -> None:
    """出口臂开关 + 当日触发计数 + 建议（不自动关闸）。"""
    st.markdown("**出口臂 / 出场健康度**（可开关层；建议仅告警、不自动 disable）")
    report = load_exit_health(session_dir)
    oms_meta = fetch_oms_meta(client, session_id=session_id, session_dir=session_dir)
    arms = {}
    health = {}
    if isinstance(oms_meta.get("exit_arms"), dict) and oms_meta.get("exit_arms"):
        arms = oms_meta["exit_arms"]
        health = oms_meta.get("exit_health") or {}
        src = str(oms_meta.get("source") or "oms_meta")
    elif not report.get("_missing"):
        arms = report.get("exit_arms") or {}
        health = report.get("exit_health") or {}
        src = str(report.get("source") or "exit_health.json")
    else:
        st.info(
            "尚无出口臂快照。重启/resume live session 后由 OMS `publish_state` 写入 "
            "`exit_health.json` / `oms_meta`。"
        )
        return

    tox = arms.get("trade_toxic") or {}
    hwd = arms.get("hold_watchdog") or {}
    sltp = arms.get("sl_tp") or {}
    circ = arms.get("day_circuit") or {}

    def _on(v: bool) -> str:
        return "ON" if v else "OFF"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "trade_toxic",
        _on(bool(tox.get("enabled"))),
        delta=f"n={tox.get('n_triggers', 0)} cut={tox.get('cut_ret', '-')}",
    )
    c2.metric(
        "hold_watchdog",
        _on(bool(hwd.get("enabled"))),
        delta=f"n={hwd.get('n_triggers', 0)} thr={hwd.get('qqq_adverse_from_entry', '-')}",
    )
    c3.metric(
        "SL / TP",
        f"{sltp.get('n_sl', 0)} / {sltp.get('n_tp', 0)}",
        delta=f"sl={sltp.get('sl_mult', '-')} tp={sltp.get('tp_mult', '-')}",
    )
    c4.metric(
        "day_circuit",
        _on(bool(circ.get("enabled"))),
        delta=f"n={circ.get('n_triggers', 0)} thr={circ.get('threshold')}",
    )
    c5.metric(
        "closes",
        health.get("n_closes") if health.get("n_closes") is not None else "-",
        delta=f"early={health.get('n_early_cut', 0)}",
    )

    suggestions = health.get("suggestions") or []
    if suggestions:
        for tip in suggestions:
            st.warning(str(tip))
    else:
        st.caption(
            f"source=`{src}` · auto_disable=false · "
            f"by_reason={health.get('closes_by_reason') or {}}"
        )
    path = report.get("path") if isinstance(report, dict) else None
    if path:
        st.caption(f"`{path}`")


def _render_tape_panel(session_dir: Path | None, *, phase: str = "") -> None:
    """Show where PRE/RTH/POST seconds are written (disk + sample rows)."""
    st.markdown("**盘前/盘中/盘后落盘（tape）**")
    inv = tape_inventory(session_dir)
    root = inv.get("root")
    if not root:
        st.info("当前 session 无磁盘目录，尚无 tape。")
        return
    st.code(
        "\n".join(
            [
                f"root={root}",
                "pre  → tape/pre/{SYM}_{date}.jsonl   + Redis …:pre",
                "rth  → tape/rth/{SYM}_{date}.jsonl   + 权威 fused stream",
                "post → tape/post/{SYM}_{date}.jsonl  + Redis …:post",
            ]
        ),
        language="text",
    )
    files = inv.get("files")
    if files is None or getattr(files, "empty", True):
        st.warning(
            f"`{root}` 下还没有 jsonl。"
            "确认 live session 已用含 tape 写入的新代码启动，且 IB 有盘前 tick。"
        )
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("tape 文件", inv.get("n_files") or 0)
    m2.metric("总行数", inv.get("n_lines") or 0)
    m3.metric("phases", ",".join(inv.get("phases") or []) or "-")
    show = files.copy()
    if "path" in show.columns:
        # keep path in expander; table shows relative-ish name
        show = show.drop(columns=["path"])
    st.dataframe(show, use_container_width=True, hide_index=True)
    samples = inv.get("samples") or {}
    if samples:
        # Prefer current phase sample
        prefer = str(phase or "").lower()
        keys = list(samples)
        if prefer:
            keys = sorted(keys, key=lambda k: (0 if k.startswith(prefer + "/") else 1, k))
        pick = keys[0]
        with st.expander(f"样例行（{pick}）"):
            st.json(samples[pick])


def _render_gates_reconcile(host: str, port: int, db: int, profile: dict) -> None:
    """ENTRY reject reasons + Broker↔OMS reconcile (read-only)."""
    st.caption(
        "今日拦截原因（ENTRY_REJECT / ENTRY_WAIT）与 Broker↔OMS 合约对账。"
        "数据来自 `maga7:order_events` / `maga7:oms_meta` / `oms_state.json`。"
    )
    auto = st.checkbox("自动刷新 (5s)", value=False, key="live_gates_auto")
    client = redis_client(host, port, db)
    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            st.warning(f"Redis 不可用，将尝试磁盘：{exc}")
            client = None

    probe = resolve_live_trace_bundle(client, prefer_disk=client is None)
    ids = probe.get("session_ids") or ([] if not probe.get("session_id") else [probe["session_id"]])
    if not ids:
        st.info("未发现 Mag7 live session。")
        return
    default_i = ids.index(probe["session_id"]) if probe.get("session_id") in ids else 0
    session_id = st.selectbox("Session", ids, index=default_i, key="live_gates_session")
    bundle = resolve_live_trace_bundle(client, session_id=session_id, prefer_disk=False)
    session_dir = Path(bundle["session_dir"]) if bundle.get("session_dir") else None
    overview = live_ops_overview(
        client, session_id=session_id, session_dir=session_dir, profile=profile
    )
    oms_meta = fetch_oms_meta(client, session_id=session_id, session_dir=session_dir)
    events = load_order_events(
        client, session_id=session_id, session_dir=session_dir, limit=3000
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("环境", overview.get("env_label") or "-")
    m2.metric("Arm", overview.get("arm_label") or "-")
    m3.metric("Day halted", "YES" if overview.get("day_halted") else "NO")
    reco = overview.get("reconcile_ok")
    m4.metric(
        "Reconcile",
        "OK" if reco is True else ("FAIL" if reco is False else "-"),
    )

    st.markdown("**Topology**")
    st.dataframe(
        pd.DataFrame(overview.get("topology") or []),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**拦截 / 等待原因 TOP**")
        reject_df = gate_reject_frame(events)
        if reject_df.empty:
            st.info("暂无 ENTRY_REJECT / ENTRY_WAIT 事件")
        else:
            show = reject_df.copy()
            if "last_ts" in show.columns:
                show["last_ts"] = show["last_ts"].map(
                    lambda x: time.strftime("%H:%M:%S", time.localtime(x))
                    if pd.notna(x) and x
                    else ""
                )
            st.dataframe(show, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Broker ↔ OMS 对账**")
        last = (oms_meta.get("last_reconcile") or {}) if oms_meta else {}
        if not last and events:
            for row in reversed(events):
                if str(row.get("kind") or "") == "RECONCILE":
                    last = {
                        "ok": row.get("ok"),
                        "broker": row.get("broker") or {},
                        "internal": row.get("internal") or {},
                        "ts": row.get("ts"),
                    }
                    break
        reco_df = reconcile_compare_frame(last)
        if last.get("ts"):
            try:
                st.caption(
                    f"last_reconcile={time.strftime('%H:%M:%S', time.localtime(float(last['ts'])))} | "
                    f"ok={last.get('ok')}"
                )
            except Exception:
                st.caption(f"ok={last.get('ok')}")
        if reco_df.empty:
            st.info(
                "尚无对账快照（Shadow 恒为 OK；Paper/Live 需 OMS reconcile 跑过一轮；"
                "旧进程需重启才会写 `maga7:oms_meta`）。"
            )
        else:
            bad = int(reco_df["status"].astype(str).str.contains("🔴").sum())
            if bad:
                st.error(f"{bad} 条合约不一致")
            st.dataframe(reco_df, use_container_width=True, hide_index=True)

    with st.expander("最近 order events（尾部 80）"):
        if not events:
            st.caption("无事件")
        else:
            st.dataframe(pd.DataFrame(events).tail(80), use_container_width=True, hide_index=True)

    _request_auto_refresh(auto)


def _render_feed_health(host: str, port: int, db: int, profile: dict) -> None:
    """Per-symbol subscription / lag panel (production-monitor style, Mag7 Redis)."""
    st.caption(
        "各标的股票 tick / 期权报价 / scanner 1m 新鲜度。"
        "阈值对齐 risk：stock≈2s、option≈5s；中断会显示 Stale / Warmup。"
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        auto = st.checkbox("自动刷新 (5s)", value=False, key="live_feed_auto")
    with c2:
        st.caption("数据来自 `maga7:feed_health:{session}` + connector status。")

    client = redis_client(host, port, db)
    if client is None:
        st.error("redis package 未安装")
        return
    try:
        client.ping()
    except Exception as exc:
        st.error(f"Redis unavailable: {exc}")
        return

    probe = resolve_live_trace_bundle(client, prefer_disk=False)
    ids = probe.get("session_ids") or ([] if not probe.get("session_id") else [probe["session_id"]])
    if not ids:
        st.info("未发现 Mag7 live session。先启动 Shadow/Paper connector。")
        return
    default_i = ids.index(probe["session_id"]) if probe.get("session_id") in ids else 0
    session_id = st.selectbox("Session", ids, index=default_i, key="live_feed_session")
    bundle = resolve_live_trace_bundle(client, session_id=session_id, prefer_disk=False)
    connector = fetch_connector_status(client, session_id=session_id)
    scanner = bundle.get("scanner") or {}

    stream_key = None
    if isinstance(connector, dict):
        stream_key = connector.get("stream") or (
            (connector.get("feed_health") or {}).get("stream")
        )
    if not stream_key:
        stream_key = f"fused_market_stream:maga7:{session_id}"
    stream_info = stream_probe(client, str(stream_key))
    stream_age = stream_info.get("age_sec")

    df = feed_health_frame(
        connector,
        scanner,
        profile=profile,
        stream_age_sec=stream_age,
    )

    state = str(connector.get("state") or "-")
    data_mode = str(connector.get("data_mode") or "-")
    connected = connector.get("connected")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Connector", state)
    m2.metric("Connected", "YES" if connected else "NO")
    m3.metric("Data mode", data_mode)
    m4.metric(
        "Stream age",
        f"{stream_age:.1f}s" if stream_age is not None else "-",
    )
    stale_n = 0
    if not df.empty and "overall" in df.columns:
        stale_n = int(df["overall"].astype(str).str.contains("Stale|DELAYED|DISCONNECTED").sum())
    m5.metric("Stale symbols", stale_n)

    if data_mode == "DELAYED_BLOCKED":
        st.error("DELAYED_BLOCKED：IBKR 延迟行情已拦截，禁止当 LIVE 交易。")
    if connected is False or state in {"DISCONNECTED", "HEARTBEAT_FAIL", "CONNECT_FAILED"}:
        st.error(f"Connector 异常：state={state}")
    if not df.empty and stale_n:
        st.warning(f"{stale_n} 个标的行情过期，请检查订阅/网络/IB Gateway。")

    if df.empty:
        st.info(
            "尚无 feed_health 快照。请确认 connector 已启动；"
            "旧进程需重启后才会按秒写入 `maga7:feed_health:*`。"
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Connector raw status"):
        st.json(connector if connector else {})

    _request_auto_refresh(auto)


def _render_positions(host: str, port: int, db: int, profile: dict) -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        auto = st.checkbox("自动刷新 (5s)", value=False, key="live_trace_auto")
    with c2:
        prefer_disk = st.checkbox("优先磁盘快照", value=False, key="live_trace_disk")
    with c3:
        st.caption("盘中看 Redis(DB0)；盘后勾选磁盘 session 快照。")

    client = redis_client(host, port, db)
    if client is not None:
        try:
            client.ping()
        except Exception as exc:
            st.warning(f"Redis 不可用，将尝试磁盘 session：{exc}")
            client = None

    probe = resolve_live_trace_bundle(client, prefer_disk=prefer_disk)
    ids = probe.get("session_ids") or ([] if not probe.get("session_id") else [probe["session_id"]])
    if not ids:
        st.info("未发现 Mag7 live session。先 `./start_maga7_live_session.sh start shadow`。")
        return

    default_i = ids.index(probe["session_id"]) if probe.get("session_id") in ids else 0
    session_id = st.selectbox("Session", ids, index=default_i, key="live_trace_session")
    bundle = resolve_live_trace_bundle(client, session_id=session_id, prefer_disk=prefer_disk)
    oms = bundle.get("oms") or {}
    meta = oms.get("meta") or {}
    pos_df = bundle.get("positions_df")
    mf_df = bundle.get("mf_df")
    fires_df = bundle.get("fires_df")
    scanner = bundle.get("scanner") or {}

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("同时持仓", bundle.get("concurrent", 0))
    m2.metric("Day halted", "YES" if meta.get("day_halted") else "NO")
    eq = meta.get("equity")
    m3.metric("Equity", f"{float(eq):,.0f}" if eq is not None else "-")
    af = meta.get("available_funds")
    m4.metric("Avail", f"{float(af):,.0f}" if af is not None else "-")
    rp = meta.get("realized_pnl")
    m5.metric("Realized PnL", f"{float(rp):+,.0f}" if rp is not None else "-")
    reco = meta.get("reconcile_ok")
    m6.metric(
        "Reconcile",
        "OK" if reco is True else ("FAIL" if reco is False else (meta.get("mode") or "-")),
    )
    st.caption(
        f"source_pos={meta.get('source') or '-'} | "
        f"source_scan={scanner.get('_source') or '-'} | "
        f"dir={bundle.get('session_dir') or '-'}"
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Active positions（同时交易）**")
        if pos_df is None or pos_df.empty:
            st.info("当前无持仓")
        else:
            show = pos_df.copy()
            if "mtm_ret" in show.columns:
                show["mtm_ret"] = show["mtm_ret"].map(
                    lambda x: f"{x:+.1%}" if pd.notna(x) else ""
                )
            st.dataframe(show, use_container_width=True, hide_index=True)
        intents = oms.get("intents") or {}
        if intents:
            st.markdown("**Pending orders**")
            st.dataframe(
                pd.DataFrame(list(intents.values())),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("**滑动窗口截面（与对拍同一套 mf 状态）**")
        if mf_df is None or mf_df.empty:
            st.info("无 scanner 窗口状态")
        else:
            show_mf = mf_df.copy()
            for col in ("mf10", "mf_fast", "cum"):
                if col in show_mf.columns:
                    show_mf[col] = show_mf[col].map(
                        lambda x: f"{float(x):,.0f}" if pd.notna(x) else ""
                    )
            st.dataframe(show_mf, use_container_width=True, hide_index=True)

    st.markdown("**单标的滑动窗口轨迹**")
    symbols = []
    if mf_df is not None and not mf_df.empty and "symbol" in mf_df.columns:
        symbols = list(mf_df["symbol"].astype(str))
    if pos_df is not None and not pos_df.empty and "symbol" in pos_df.columns:
        for s in pos_df["symbol"].astype(str):
            if s not in symbols:
                symbols.append(s)
    if not symbols:
        symbols = list((profile.get("profile") or {}).get("symbols") or [])
    if symbols:
        sym = st.selectbox("Symbol", symbols, key="live_trace_sym")
        sig_cfg = (profile.get("profile") or {}).get("signal") or {}
        mf_w = int(sig_cfg.get("mf_window", 10))
        series = mf_series_for_symbol(scanner, sym, mf_window=mf_w, mf_fast_n=3)
        a, b, c, d = st.columns(4)
        if not series.empty:
            last = series.iloc[-1]
            a.metric(
                "mf10",
                f"{float(last['mf10']):,.0f}" if pd.notna(last.get("mf10")) else "-",
            )
            b.metric(
                "mf_fast(3)",
                f"{float(last['mf_fast']):,.0f}" if pd.notna(last.get("mf_fast")) else "-",
            )
            c.metric(
                "close",
                f"{float(last['close']):.2f}" if pd.notna(last.get("close")) else "-",
            )
            row = (
                mf_df[mf_df["symbol"].astype(str) == str(sym)]
                if mf_df is not None and not mf_df.empty
                else pd.DataFrame()
            )
            if not row.empty:
                d.metric(
                    "streak U/D",
                    f"{int(row.iloc[0].get('streak_up') or 0)} / {int(row.iloc[0].get('streak_dn') or 0)}",
                )
            if px is not None and "timestamp" in series.columns:
                plot_df = series.dropna(subset=["timestamp"]).copy()
                value_cols = [
                    col for col in ("mf10", "mf_fast") if col in plot_df.columns
                ]
                for col in value_cols:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
                plot_long = (
                    plot_df.melt(
                        id_vars=["timestamp"],
                        value_vars=value_cols,
                        var_name="metric",
                        value_name="value",
                    ).dropna(subset=["value"])
                    if value_cols
                    else pd.DataFrame()
                )
                if not plot_long.empty:
                    st.plotly_chart(
                        px.line(
                            plot_long,
                            x="timestamp",
                            y="value",
                            color="metric",
                            title=f"{sym} sliding MF (same as parity)",
                        ),
                        use_container_width=True,
                    )
        else:
            st.caption("该标的暂无 bars")

    if fires_df is not None and not fires_df.empty:
        st.markdown("**当日 fires / signals**")
        st.dataframe(fires_df, use_container_width=True, hide_index=True)

    session_dir = bundle.get("session_dir")
    if session_dir:
        _render_trade_spreads(Path(session_dir))
        events_path = Path(session_dir) / "order_events.jsonl"
        if events_path.is_file():
            rows = []
            try:
                with events_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except OSError:
                rows = []
            if rows:
                st.markdown("**Order / fill events**")
                st.dataframe(
                    pd.DataFrame(rows).tail(100),
                    use_container_width=True,
                    hide_index=True,
                )

    _request_auto_refresh(auto)


def _render_sessions(sessions) -> None:
    if not sessions:
        st.info("尚无 live_sessions/manifest.json")
        return
    rows = []
    for session in sessions:
        manifest = session.manifest
        connector = manifest.get("connector") or {}
        engine = manifest.get("engine_metrics") or {}
        oms = manifest.get("oms") or {}
        rows.append(
            {
                "session": session.name,
                "mode": session.mode,
                "state": manifest.get("state"),
                "data": connector.get("data_mode"),
                "lock": connector.get("lock_status"),
                "frames": engine.get("frames"),
                "rejected": engine.get("rejected"),
                "reconcile": oms.get("reconcile_ok"),
                "positions": oms.get("positions"),
                "G4 evidence": session.base_ok,
                "broker lifecycle": session.broker_lifecycle_ok,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    options = {session.name: session for session in sessions}
    selected = options[st.selectbox("Live session", list(options), key="live_session")]
    frames = live_session_frames(selected)
    a, b, c = st.tabs(["交易点差（开/平仓）", "Order / fill events", "Open ladder locks"])
    with a:
        spreads = frames.get("trade_spreads")
        if spreads is None or spreads.empty:
            st.info("尚无 OPEN/CLOSE 点差记录（成交后写入 trade_spreads.csv）")
        else:
            st.dataframe(
                _format_spread_df(spreads),
                use_container_width=True,
                hide_index=True,
            )
    with b:
        st.dataframe(frames["events"], use_container_width=True, hide_index=True)
    with c:
        st.dataframe(frames["locks"], use_container_width=True, hide_index=True)
    with st.expander("Session manifest"):
        st.json(selected.manifest)


def _render_redis(host: str, port: int, db: int) -> None:
    st.dataframe(
        pd.DataFrame(process_snapshot()),
        use_container_width=True,
        hide_index=True,
    )
    client = redis_client(host, port, db)
    if client is None:
        st.error("redis package 未安装")
        return
    try:
        client.ping()
    except Exception as exc:
        st.error(f"Redis unavailable: {exc}")
        return
    snapshot = redis_snapshot(client)
    st.dataframe(pd.DataFrame(snapshot["streams"]), use_container_width=True, hide_index=True)
    with st.expander("Live hashes / status"):
        st.dataframe(
            pd.DataFrame(snapshot["live_hashes"]),
            use_container_width=True,
            hide_index=True,
        )


def _format_spread_df(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    if "spread_pct" in show.columns:
        show["spread_pct"] = show["spread_pct"].map(
            lambda x: f"{float(x):.2%}" if pd.notna(x) and x != "" else ""
        )
    if "fill_spread_frac" in show.columns:
        show["fill_spread_frac"] = show["fill_spread_frac"].map(
            lambda x: f"{float(x):.2f}" if pd.notna(x) and x != "" else ""
        )
    prefer = [
        "ts",
        "action",
        "symbol",
        "contract",
        "side",
        "fill_px",
        "bid",
        "ask",
        "spread",
        "spread_pct",
        "fill_spread_frac",
        "reason",
        "ret",
        "mode",
    ]
    cols = [c for c in prefer if c in show.columns] + [
        c for c in show.columns if c not in prefer
    ]
    return show[cols]


def _render_trade_spreads(session_dir: Path) -> None:
    csv_path = session_dir / "trade_spreads.csv"
    spreads = None
    if csv_path.is_file():
        try:
            spreads = pd.read_csv(csv_path)
        except Exception:
            spreads = None
    if spreads is None or spreads.empty:
        events_path = session_dir / "order_events.jsonl"
        rows = []
        if events_path.is_file():
            try:
                with events_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except OSError:
                rows = []
        spreads = trade_spreads_from_events(rows)
    st.markdown("**交易点差（开仓 / 平仓）**")
    st.caption("每笔 OPEN/CLOSE 记录当时 bid/ask、绝对点差、spread_pct、成交落在点差的位置 fill_spread_frac。")
    if spreads is None or spreads.empty:
        st.info("尚无成交点差记录")
        return
    st.dataframe(_format_spread_df(spreads), use_container_width=True, hide_index=True)


def _render_ops(profile: dict) -> None:
    st.markdown("**启停（在 SHELL 执行；Dashboard 不代发）**")
    st.code(
        f"""cd maga7/SHELL
# 盘前加固 / 一天对拍（先过再开实盘）
./run_day_stream_check.sh 2026-05-28

# G4 Shadow（真实行情，不发单）— 与对拍同 profile
./start_maga7_live_session.sh start shadow
# profile 默认：
# {profile.get('path')}

./start_maga7_live_session.sh status
./start_maga7_live_session.sh stop

# G5 Paper（需账户）
MAG7_ACCOUNT=DUxxxxxx ./start_maga7_live_session.sh start paper --account DUxxxxxx
""",
        language="bash",
    )
    st.markdown("**上线缺口**")
    st.dataframe(pd.DataFrame(ALIGNMENT_GAPS), use_container_width=True, hide_index=True)
