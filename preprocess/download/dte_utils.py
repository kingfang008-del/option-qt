#!/usr/bin/env python3
"""DTE helpers: calendar days vs NYSE trading sessions."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

NY = "America/New_York"


@lru_cache(maxsize=1)
def _nyse_calendar():
    import pandas_market_calendars as mcal

    return mcal.get_calendar("NYSE")


def trading_sessions_between(quote_date: pd.Timestamp, exp_date: pd.Timestamp) -> int:
    """
    剩余交易 DTE：从 quote 日之后到 expiry 日（含）之间的 NYSE 交易日数。

    例：周五 quote、周一 expiry → 1（仅周一）；同天 expiry → 0。
    节假日自动跳过（NYSE schedule 不含休市日）。
    """
    q = pd.Timestamp(quote_date).normalize()
    e = pd.Timestamp(exp_date).normalize()
    if e <= q:
        return 0
    cal = _nyse_calendar()
    start = q + pd.Timedelta(days=1)
    if start > e:
        return 0
    sched = cal.schedule(start_date=start.date(), end_date=e.date())
    return int(len(sched))


def compute_dte_series(
    timestamps: pd.Series,
    expirations: pd.Series,
    *,
    use_trading_dte: bool = False,
) -> pd.Series:
    """按 anchor 配置计算 dte 列。"""
    ts = pd.to_datetime(timestamps, errors="coerce")
    exp = pd.to_datetime(expirations, errors="coerce")
    if use_trading_dte:
        if ts.dt.tz is not None:
            q_dates = ts.dt.tz_convert(NY).dt.normalize()
        else:
            q_dates = ts.dt.tz_localize(NY, ambiguous="infer").dt.normalize()
        if exp.dt.tz is not None:
            exp_dates = exp.dt.tz_convert(NY).dt.normalize()
        else:
            exp_dates = exp.dt.tz_localize(NY, ambiguous="infer").dt.normalize()

        out = pd.Series(-1, index=timestamps.index, dtype="int64")
        for qd in q_dates.dropna().unique():
            mask = q_dates == qd
            exps = exp_dates[mask].dropna().unique()
            dte_map = {e: trading_sessions_between(qd, e) for e in exps}
            mapped = exp_dates[mask].map(dte_map)
            out.loc[mask] = mapped.fillna(-1).astype(int).values
        return out.astype(int)

    if ts.dt.tz is not None:
        ts_n = ts.dt.tz_convert(NY).dt.normalize()
    else:
        ts_n = ts.dt.tz_localize(NY, ambiguous="infer").dt.normalize()
    if exp.dt.tz is not None:
        exp_n = exp.dt.tz_convert(NY).dt.normalize()
    else:
        exp_n = exp.dt.tz_localize(NY, ambiguous="infer").dt.normalize()
    return (exp_n - ts_n).dt.days.fillna(-1).astype(int)
