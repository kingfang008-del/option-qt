"""Offline Replay board — Mag7 research / gold-path results."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from boards.common import fmt_time, metric, render_contract_banner, run_options
from sources import REPO, run_frames

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


def render_offline_board(runs, profile: dict) -> None:
    st.markdown("### ② Offline Replay")
    st.caption(
        "离线金标：磁盘 1s → Rule-A/TopK → 模型成交。"
        "与对拍/实盘共用同一 profile；这里只看结果与复跑命令。"
    )
    render_contract_banner()

    offline = [r for r in runs if r.stage == "offline"]
    options = run_options(offline)
    if not options:
        st.info("未发现 offline summary（`maga7/results/**/summary.json` 且非 S5/parity）")
    else:
        selected = st.selectbox("Offline run", list(options), key="offline_run")
        run = options[selected]
        st.code(str(run.path.relative_to(REPO)), language=None)
        s = run.summary or {}
        a, b, c, d = st.columns(4)
        a.metric("total_ret", metric(s.get("total_ret")))
        b.metric("maxdd", metric(s.get("maxdd")))
        c.metric("n_trades", metric(s.get("n_trades")))
        d.metric("mtime", fmt_time(run.mtime))
        frames = run_frames(run)
        tabs = st.tabs(["Trades", "Daily", "Summary JSON"])
        with tabs[0]:
            st.dataframe(frames["trades"], use_container_width=True, hide_index=True)
        with tabs[1]:
            df = frames["daily"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            if px is not None and not df.empty:
                y = next((c for c in ("equity", "ret", "day_ret") if c in df.columns), None)
                x = next((c for c in ("date", "ts") if c in df.columns), None)
                if x and y:
                    st.plotly_chart(px.line(df, x=x, y=y), use_container_width=True)
        with tabs[2]:
            st.json(s)

    cfg = profile.get("profile") or {}
    scheme = cfg.get("recommended_scheme") or "single"
    st.markdown("**复跑命令**")
    st.code(
        f"""export PYTHONPATH=$PWD
python -m maga7.tools.run_replay_offline \\
  --profile {profile.get('path')} \\
  --scheme {scheme} \\
  --start-date 2026-05-01 --end-date 2026-07-16
""",
        language="bash",
    )
    st.caption("调规则只改 strategy profile；Offline / Parity / Live 读同一文件。")
