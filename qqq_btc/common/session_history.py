#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会话历史 carryover —— 前日 RTH tail 拼入当日,使 09:30 首 bar 即有满 seq_len 上下文。

与训练/infer 的 row_to_tensors 一致:左侧补零可用,但 carryover 让 trend/序列特征更完整。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .time_features import SESSION_TZ

SEQ_LEN = 30
DEFAULT_CARRYOVER_BARS = SEQ_LEN - 1  # 29; + 当日首 bar = 30


def _to_ny_date(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True)
    if t.dt.tz is None:
        t = t.dt.tz_localize(SESSION_TZ, ambiguous="infer")
    else:
        t = t.dt.tz_convert(SESSION_TZ)
    return t.dt.date


def session_tail(df: pd.DataFrame, n_bars: int = DEFAULT_CARRYOVER_BARS) -> pd.DataFrame:
    """取单个会话 DataFrame 末尾 n 根 bar(按 timestamp 排序)。"""
    if df.empty or n_bars <= 0:
        return df.iloc[0:0].copy()
    return df.sort_values("timestamp").tail(int(n_bars)).reset_index(drop=True)


def augment_with_session_carryover(
    df: pd.DataFrame,
    *,
    carryover_bars: int = DEFAULT_CARRYOVER_BARS,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    多日 parquet:每个新交易日开头插入上一交易日 tail,供 infer/replay 序列上下文。

    返回新 DataFrame(行数略增);原行 timestamp 不变,carryover 行带 `_carryover=True`。
    """
    if df.empty or carryover_bars <= 0:
        return df.copy()

    work = df.sort_values(ts_col).reset_index(drop=True)
    dates = _to_ny_date(work[ts_col])
    parts: list[pd.DataFrame] = []
    prev_tail: Optional[pd.DataFrame] = None

    for _, grp in work.groupby(dates, sort=True):
        grp = grp.copy()
        if prev_tail is not None and not prev_tail.empty:
            tail = prev_tail.copy()
            tail["_carryover"] = True
            parts.append(tail)
        grp["_carryover"] = False
        parts.append(grp)
        prev_tail = grp.tail(carryover_bars)

    out = pd.concat(parts, ignore_index=True)
    if "_carryover" in out.columns:
        out["_carryover"] = out["_carryover"].fillna(False).astype(bool)
    return out


def prepend_carryover(
    today_df: pd.DataFrame,
    carryover_df: Optional[pd.DataFrame],
    *,
    carryover_bars: int = DEFAULT_CARRYOVER_BARS,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """实盘/单日:在今日 history 前拼接前日 tail(仅当 today 非空)。"""
    if carryover_df is None or carryover_df.empty or today_df.empty:
        return today_df.sort_values(ts_col).reset_index(drop=True)

    tail = session_tail(carryover_df, carryover_bars)
    if tail.empty:
        return today_df.sort_values(ts_col).reset_index(drop=True)

    tail = tail.copy()
    tail["_carryover"] = True
    today = today_df.sort_values(ts_col).reset_index(drop=True).copy()
    today["_carryover"] = False
    return pd.concat([tail, today], ignore_index=True)


def real_bar_index(df: pd.DataFrame) -> int:
    """augment/prepend 后,最后一根非 carryover 行的位置(-1 若全为 carryover)。"""
    if df.empty:
        return -1
    if "_carryover" not in df.columns:
        return len(df) - 1
    real = df.index[~df["_carryover"].astype(bool)]
    return int(real[-1]) if len(real) else len(df) - 1
