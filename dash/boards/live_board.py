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
    live_session_frames,
    mf_series_for_symbol,
    process_snapshot,
    redis_snapshot,
    resolve_live_trace_bundle,
    trade_spreads_from_events,
)

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


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
        "本页只读：持仓、滑动窗、session 证据、启停命令提示。"
    )
    render_contract_banner()

    tab_pos, tab_sess, tab_redis, tab_ops = st.tabs(
        ["持仓 / 滑动窗口", "Live sessions", "Redis 链路", "启停命令"]
    )

    with tab_pos:
        _render_positions(host, port, db, profile)
    with tab_sess:
        _render_sessions(live_sessions)
    with tab_redis:
        _render_redis(host, port, db)
    with tab_ops:
        _render_ops(profile)


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

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("同时持仓", bundle.get("concurrent", 0))
    m2.metric("Day halted", "YES" if meta.get("day_halted") else "NO")
    eq = meta.get("equity")
    m3.metric("Equity", f"{float(eq):,.0f}" if eq is not None else "-")
    rp = meta.get("realized_pnl")
    m4.metric("Realized PnL", f"{float(rp):+,.0f}" if rp is not None else "-")
    m5.metric("Mode", meta.get("mode") or "-")
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
                if not plot_df.empty:
                    st.plotly_chart(
                        px.line(
                            plot_df,
                            x="timestamp",
                            y=["mf10", "mf_fast"],
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

    if auto:
        time.sleep(5)
        st.rerun()


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
