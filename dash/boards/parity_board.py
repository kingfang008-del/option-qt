"""Stream Parity board — same logic as live; historical data + simulated fills."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from boards.common import render_contract_banner, run_options
from sources import REPO, latest_by_stage, run_frames


def _show_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    if "spread_pct" in show.columns:
        show["spread_pct"] = show["spread_pct"].map(
            lambda x: f"{float(x):.2%}" if pd.notna(x) and str(x) != "" else ""
        )
    prefer = [
        "action",
        "ts",
        "symbol",
        "dir",
        "contract",
        "px",
        "bid",
        "ask",
        "spread",
        "spread_pct",
        "reason",
        "ret",
    ]
    cols = [c for c in prefer if c in show.columns] + [
        c for c in show.columns if c not in prefer
    ]
    return show[cols]


def render_parity_board(runs, profile: dict) -> None:
    st.markdown("### ③ Stream Parity / 一天流式对拍")
    st.caption(
        "与实盘**同一套** Scanner → 选约 → 退出状态机；"
        "差别只有：数据=历史 1s 流式打入，成交=模型 fill（模拟）。"
    )
    render_contract_banner()

    latest = latest_by_stage(runs)
    p_run = latest.get("stream_parity")
    s5 = latest.get("redis_replay")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "G2 Stream parity",
            "PASS" if p_run and p_run.ok else "—",
            help="offline ↔ stream/scanner 规则对拍",
        )
        if p_run:
            st.caption(p_run.name)
    with c2:
        st.metric(
            "G3 Redis S5",
            "PASS" if s5 and s5.ok else "—",
            help="1s→Redis fused→scanner→OMS，最接近实盘拓扑",
        )
        if s5:
            st.caption(s5.name)
    with c3:
        fp = profile.get("strategy_fingerprint") or ""
        st.metric("Profile hash", fp[:12] + "…" if len(fp) > 12 else fp or "—")

    st.markdown("**一键命令（盘前主路径）**")
    cfg = profile.get("profile") or {}
    scheme = cfg.get("recommended_scheme") or "single"
    prof = profile.get("path")
    st.code(
        f"""cd maga7/SHELL
# 一天 trade_log 对拍（有 Redis 走 S5，否则 --force-local）
./run_day_stream_check.sh 2026-05-28

# G2 规则对拍
python -m maga7.tools.run_stream_parity \\
  --profile {prof} --scheme {scheme} \\
  --start-date 2026-05-28 --end-date 2026-05-28 \\
  --stock-source stock_1s --tag parity_freeze_smoke

# G3 Redis S5（需 Redis DB1）
python -m maga7.tools.run_maga7_redis_sim \\
  --profile {prof} --scheme {scheme} \\
  --start-date 2026-05-28 --end-date 2026-05-28 \\
  --options --compare-offline --sync
""",
        language="bash",
    )

    parity_runs = [r for r in runs if r.stage in {"stream_parity", "redis_replay", "dry"}]
    options = run_options(parity_runs)
    if not options:
        st.info("尚无 parity / S5 / dry 结果")
        return

    selected = st.selectbox("Parity / S5 / dry run", list(options), key="parity_run")
    run = options[selected]
    st.code(str(run.path.relative_to(REPO)), language=None)
    left, right = st.columns(2)
    with left:
        st.markdown("**Summary**")
        st.json(run.summary or {})
    with right:
        st.markdown("**Compare / Parity**")
        st.json(run.parity or run.compare or {})

    frames = run_frames(run)
    tabs = st.tabs(["Trades", "Compare", "Fill audit", "Signals"])
    for tab, key in zip(tabs, ("trades", "compare", "audit", "signals")):
        with tab:
            df = frames.get(key)
            if df is None or df.empty:
                st.caption("无数据")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

    # trade_log if present (day stream check) — OPEN/CLOSE 含 bid/ask/spread_pct
    tl = run.path / "trade_log.csv"
    tl_off = run.path / "trade_log_offline.csv"
    if tl.is_file() or tl_off.is_file():
        st.markdown("**trade_log（OPEN/CLOSE + 点差）**")
        st.caption("每行开仓/平仓各自记录 bid / ask / spread / spread_pct。")
        t1, t2 = st.columns(2)
        with t1:
            if tl.is_file():
                st.caption("stream")
                st.dataframe(_show_trade_log(pd.read_csv(tl)), use_container_width=True, hide_index=True)
        with t2:
            if tl_off.is_file():
                st.caption("offline")
                st.dataframe(
                    _show_trade_log(pd.read_csv(tl_off)),
                    use_container_width=True,
                    hide_index=True,
                )

    st.success(
        "对拍通过后 → Live 页开 Shadow；不要在 Live 改一套不同的出场/选约逻辑。"
    )
