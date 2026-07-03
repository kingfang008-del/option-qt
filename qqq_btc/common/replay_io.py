#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回放数据 I/O —— tick 报价加载、S4 风格 DB 融合。

S4 数据契约(与 s4_run_historical_replay_s2_1s 一致):
  1. 读 alpha_logs + market_bars_1s + option_snapshots_1s
  2. alpha ts += 60s 后再 merge_asof backward 到 1s 流(无 lookahead)
  3. 产出 minute_df(信号/分钟末盘口) + tick_df(全秒流) 同源

Tick parquet 列:
  timestamp / ts, exec_call_bid/ask, exec_put_bid/ask(可选)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .event_replay import prepare_minute_frame, prepare_tick_frame

ALPHA_AVAILABLE_DELAY_SECONDS = 60.0
DEFAULT_CALL_BUCKET_IDX = 2
DEFAULT_PUT_BUCKET_IDX = 0

TICK_QUOTE_COLUMNS = (
    "timestamp",
    "exec_call_bid",
    "exec_call_ask",
    "exec_put_bid",
    "exec_put_ask",
    "exec_call_spread_pct",
    "exec_put_spread_pct",
)


@dataclass
class S4ReplayBundle:
    minute_df: pd.DataFrame
    tick_df: pd.DataFrame
    merged_1s: pd.DataFrame
    alpha_delay_seconds: float = ALPHA_AVAILABLE_DELAY_SECONDS


def load_ticks(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return prepare_tick_frame(df)


def _quotes_from_buckets(
    buckets_json,
    *,
    call_idx: int = DEFAULT_CALL_BUCKET_IDX,
    put_idx: int = DEFAULT_PUT_BUCKET_IDX,
) -> dict:
    if isinstance(buckets_json, dict):
        bk = buckets_json.get("buckets", [])
    elif isinstance(buckets_json, str) and buckets_json:
        bk = json.loads(buckets_json).get("buckets", [])
    else:
        bk = []
    c_bk = bk[call_idx] if len(bk) > call_idx else []
    p_bk = bk[put_idx] if len(bk) > put_idx else []
    return {
        "exec_call_bid": float(c_bk[8]) if len(c_bk) > 8 else np.nan,
        "exec_call_ask": float(c_bk[9]) if len(c_bk) > 9 else np.nan,
        "exec_put_bid": float(p_bk[8]) if len(p_bk) > 8 else np.nan,
        "exec_put_ask": float(p_bk[9]) if len(p_bk) > 9 else np.nan,
    }


def _attach_exec_quotes(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_quotes_from_buckets(r) for r in df["buckets_json"]]
    q = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), q], axis=1)
    for leg in ("call", "put"):
        b, a = f"exec_{leg}_bid", f"exec_{leg}_ask"
        mid = (out[b] + out[a]) / 2.0
        out[f"exec_{leg}_spread_pct"] = np.where(
            (out[b] > 0) & (out[a] > out[b]),
            (out[a] - out[b]) / mid,
            np.nan,
        )
    return out


def load_s4_merged_1s_sqlite(
    db_path: str | Path,
    *,
    symbol: str,
    alpha_delay_seconds: float = ALPHA_AVAILABLE_DELAY_SECONDS,
    option_tolerance: float = 2.0,
    alpha_tolerance: float = 120.0,
) -> pd.DataFrame:
    """
    S4 等价 merge: stock 1s + option 1s + alpha(+delay) → 密集 1s 表。
    """
    db_path = Path(db_path).expanduser()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df_a = pd.read_sql(
        "SELECT ts, symbol, alpha AS alpha_score FROM alpha_logs WHERE symbol = ?",
        conn,
        params=(symbol,),
    )
    df_s = pd.read_sql(
        "SELECT ts, symbol, close FROM market_bars_1s WHERE symbol = ?",
        conn,
        params=(symbol,),
    )
    df_o = pd.read_sql(
        "SELECT ts, symbol, buckets_json FROM option_snapshots_1s WHERE symbol = ?",
        conn,
        params=(symbol,),
    )
    conn.close()

    for x in (df_a, df_s, df_o):
        x["ts"] = x["ts"].astype(float)
        x["symbol"] = x["symbol"].astype(str)

    df_a["alpha_label_ts"] = df_a["ts"]
    df_a["ts"] = df_a["ts"] + float(alpha_delay_seconds)
    df_a = df_a.sort_values("ts")
    df_s = df_s.sort_values("ts")
    df_o = df_o.sort_values("ts")

    df_market = pd.merge_asof(
        df_s, df_o, on="ts", by="symbol", direction="backward", tolerance=option_tolerance
    )
    merged = pd.merge_asof(
        df_market,
        df_a[["ts", "symbol", "alpha_score", "alpha_label_ts"]],
        on="ts",
        by="symbol",
        direction="backward",
        tolerance=alpha_tolerance,
    )
    merged = _attach_exec_quotes(merged)
    merged["timestamp"] = pd.to_datetime(merged["ts"], unit="s", utc=True)
    return merged.sort_values("ts").reset_index(drop=True)


def build_s4_bundle_from_merged(
    merged: pd.DataFrame,
    *,
    edge_col: str = "net_edge",
    alpha_to_edge: bool = True,
) -> S4ReplayBundle:
    """
    从 S4 merge 后 1s 表拆 minute_df + tick_df(同源)。

    minute_df:
      - net_edge: 分钟首 tick 的 alpha(S4 signal_packet 语义)
      - exec_*: 分钟末 tick 盘口(rails MTM)
    tick_df: 全秒 exec 报价流
    """
    if merged.empty:
        empty = pd.DataFrame()
        return S4ReplayBundle(empty, empty, merged)

    m = merged.copy()
    m["_minute_key"] = (m["ts"] // 60 * 60).astype(int)

    tick_df = m[
        ["timestamp", "ts", "exec_call_bid", "exec_call_ask", "exec_put_bid", "exec_put_ask",
         "exec_call_spread_pct", "exec_put_spread_pct", "_minute_key"]
    ].copy()

    minute_rows = []
    for mk, grp in m.groupby("_minute_key", sort=True):
        grp = grp.sort_values("ts")
        first, last = grp.iloc[0], grp.iloc[-1]
        edge = float(first.get("alpha_score", np.nan)) if alpha_to_edge else np.nan
        minute_rows.append(
            {
                "timestamp": pd.to_datetime(int(mk), unit="s", utc=True),
                "ts": float(mk),
                "alpha_label_ts": first.get("alpha_label_ts"),
                "alpha_score": first.get("alpha_score"),
                edge_col: edge,
                "net_edge": edge,
                "exec_call_bid": last["exec_call_bid"],
                "exec_call_ask": last["exec_call_ask"],
                "exec_put_bid": last.get("exec_put_bid"),
                "exec_put_ask": last.get("exec_put_ask"),
                "exec_call_spread_pct": last.get("exec_call_spread_pct"),
                "exec_put_spread_pct": last.get("exec_put_spread_pct"),
                "_minute_key": int(mk),
            }
        )

    minute_df = prepare_minute_frame(pd.DataFrame(minute_rows))
    tick_df = prepare_tick_frame(tick_df)
    return S4ReplayBundle(minute_df=minute_df, tick_df=tick_df, merged_1s=m)


def build_s4_bundle_from_sqlite(
    db_path: str | Path,
    *,
    symbol: str = "QQQ",
    alpha_delay_seconds: float = ALPHA_AVAILABLE_DELAY_SECONDS,
) -> S4ReplayBundle:
    merged = load_s4_merged_1s_sqlite(
        db_path, symbol=symbol, alpha_delay_seconds=alpha_delay_seconds
    )
    return build_s4_bundle_from_merged(merged)


def load_ticks_sqlite(
    db_path: str | Path,
    *,
    symbol: Optional[str] = None,
    bucket_call_idx: int = DEFAULT_CALL_BUCKET_IDX,
    bucket_put_idx: int = DEFAULT_PUT_BUCKET_IDX,
) -> pd.DataFrame:
    """仅 tick 流(不含 alpha merge);完整 S4 语义请用 build_s4_bundle_from_sqlite。"""
    db_path = Path(db_path).expanduser()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = "SELECT ts, symbol, buckets_json FROM option_snapshots_1s"
    params: Tuple = ()
    if symbol:
        q += " WHERE symbol = ?"
        params = (symbol,)
    raw = pd.read_sql(q, conn, params=params)
    conn.close()

    rows = []
    for r in raw.itertuples():
        qd = _quotes_from_buckets(
            r.buckets_json, call_idx=bucket_call_idx, put_idx=bucket_put_idx
        )
        rows.append({"ts": float(r.ts), **qd})
    return prepare_tick_frame(pd.DataFrame(rows))
