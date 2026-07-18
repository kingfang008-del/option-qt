#!/usr/bin/env python3
"""Mag7 control plane — qqq_btc-style boards: Data / Offline / Parity / Live."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

DASH_DIR = Path(__file__).resolve().parent
REPO = DASH_DIR.parent
for path in (str(DASH_DIR), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from boards.data_board import render_data_board  # noqa: E402
from boards.live_board import render_live_board  # noqa: E402
from boards.offline_board import render_offline_board  # noqa: E402
from boards.parity_board import render_parity_board  # noqa: E402
from sources import (  # noqa: E402
    PROD_PROFILE,
    discover_live_sessions,
    discover_maga7_runs,
    pipeline_gates,
    profile_snapshot,
)

st.set_page_config(
    page_title="Mag7 Control Plane",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data(ttl=5)
def _runs():
    return discover_maga7_runs(limit=150)


@st.cache_data(ttl=5)
def _live_sessions():
    return discover_live_sessions(limit=150)


@st.cache_data(ttl=30)
def _profile(path: str):
    return profile_snapshot(Path(path))


st.markdown("## Mag7 Control Plane")
st.caption(
    "对齐 qqq_btc dashboard 分层：**补数据 → Offline → 对拍 → 实盘**。"
    "对拍与实盘共用同一 profile / Scanner / 退出逻辑，只换数据源与成交方式。"
)

with st.sidebar:
    st.header("导航")
    board = st.radio(
        "Board",
        options=["Download", "Offline Replay", "Stream Parity", "Live"],
        index=0,
        help=(
            "Download=补数据；Offline=离线金标；"
            "Stream Parity=流式/S5 对拍（模拟数据与成交）；"
            "Live=Shadow/Paper/Live 持仓与 session"
        ),
    )
    st.divider()
    st.header("连接")
    profile_path = st.text_input("Mag7 profile", str(PROD_PROFILE))
    redis_host = st.text_input("Redis host", os.environ.get("REDIS_HOST", "127.0.0.1"))
    redis_port = st.number_input("Redis port", value=int(os.environ.get("REDIS_PORT", "6379")))
    redis_db = st.number_input(
        "Redis DB",
        value=int(os.environ.get("REDIS_DB", "0")),
        help="Live 默认 0；S5 研究常用 1",
    )
    if st.button("刷新"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.info("安全边界")
    st.caption(
        "Download 可启停补数任务并写日志；"
        "不写 Redis、不代启停 Live、不发单。"
    )

    # Mini gate strip
    runs_preview = _runs()
    live_preview = _live_sessions()
    prof_preview = _profile(profile_path)
    gates = pipeline_gates(runs_preview, prof_preview, live_preview)
    passed = sum(1 for g in gates if g["status"] == "PASS")
    st.metric("Gates PASS", f"{passed}/{len(gates)}")

runs = _runs()
live_sessions = _live_sessions()
profile = _profile(profile_path)

if board == "Download":
    render_data_board(profile)
elif board == "Offline Replay":
    render_offline_board(runs, profile)
elif board == "Stream Parity":
    render_parity_board(runs, profile)
else:
    render_live_board(
        host=redis_host,
        port=int(redis_port),
        db=int(redis_db),
        profile=profile,
        live_sessions=live_sessions,
    )
