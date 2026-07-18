"""Shared helpers for Mag7 dash boards."""
from __future__ import annotations

from datetime import datetime
from typing import Any

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
