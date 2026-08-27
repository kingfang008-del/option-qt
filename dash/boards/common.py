"""Shared helpers for Mag7 dash boards."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import redis
except Exception:  # pragma: no cover
    redis = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


# Same contract as qqq_btc: Offline / Parity / Live share profile + decision path.
PARITY_LIVE_CONTRACT = """
**一致性契约（对拍 ≈ 实盘，只换数据源与成交）**

| 层 | Offline | Stream / Day 对拍 | 实盘 Shadow/Paper/Live |
|---|---|---|---|
| Profile | 同一 `strategy_profiles/*.json` | 同左 | 同左 |
| 信号 | Rule-A + TopK + regime | 同左（Scanner 因果分钟） | 同左（IB 1s→分钟） |
| 选约 / 退出 | `entry_contract` + `simulate_trade` / OMS 状态机 | 同左 | 同左（Paper/Live 真限价） |
| 数据源 | 磁盘 1s / quote | 磁盘 1s 流式打入（或 Redis S5） | IBKR 实时 |
| 成交 | 模型 fill | 模型 fill（模拟） | Shadow=模型；Paper/Live=券商 |

对拍通过 ≠ 已可真钱；G4/G5 仍要真实 session 证据。
"""


def render_contract_banner() -> None:
    st.info(PARITY_LIVE_CONTRACT)


def redis_client(host: str, port: int, db: int):
    if redis is None:
        return None
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=False,
        socket_connect_timeout=1.5,
        socket_timeout=2.0,
    )


def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def metric(value: Any, default: str = "-") -> Any:
    return default if value is None else value


def run_options(runs) -> dict:
    return {
        f"{run.stage} | {fmt_time(run.mtime)} | {run.name}": run for run in runs
    }


def render_live_ops_sidebar(
    *,
    host: str,
    port: int,
    db: int,
    profile: dict | None = None,
) -> None:
    """P0 read-only strip: env / account / arm / topology."""
    from sources import list_maga7_session_ids, live_ops_overview, resolve_live_trace_bundle

    st.header("Live Ops")
    st.caption("只读：环境 / 账户 / Arm / 拓扑")
    client = redis_client(host, int(port), int(db))
    if client is None:
        st.caption("redis 未安装")
        return
    try:
        client.ping()
    except Exception as exc:
        st.caption(f"Redis 不可用: {exc}")
        return

    probe = resolve_live_trace_bundle(client, prefer_disk=False)
    session_id = probe.get("session_id")
    session_dir = probe.get("session_dir")
    if not session_id:
        ids = list_maga7_session_ids(client)
        session_id = ids[0] if ids else None
    overview = live_ops_overview(
        client,
        session_id=session_id,
        session_dir=Path(session_dir) if session_dir else None,
        profile=profile,
    )

    st.metric("环境", overview.get("env_label") or "-")
    st.metric("Arm", overview.get("arm_label") or "-")
    eq = overview.get("equity")
    af = overview.get("available_funds")
    c1, c2 = st.columns(2)
    c1.metric("Equity", f"{float(eq):,.0f}" if eq is not None else "-")
    c2.metric("Avail", f"{float(af):,.0f}" if af is not None else "-")
    if overview.get("day_halted"):
        st.error("day_halted")
    if overview.get("reconcile_ok") is False:
        st.error("broker reconcile FAIL")
    if overview.get("data_mode") == "DELAYED_BLOCKED":
        st.error("DELAYED_BLOCKED")

    overall = overview.get("topology_overall")
    if overall:
        st.metric("数据流", overall)
    phase = overview.get("session_phase")
    if phase:
        st.caption(f"时段={phase}（PRE/POST 只验校验流+tape）")
    topo = overview.get("topology") or []
    if topo:
        st.caption("Topology")
        st.dataframe(
            pd.DataFrame(topo)[["node", "health", "age_sec"]],
            use_container_width=True,
            hide_index=True,
        )
    if session_id:
        st.caption(f"session={session_id}")
