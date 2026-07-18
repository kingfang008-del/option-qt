#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实盘因果 VX 期限结构：仅使用前一已完成 Databento UTC 日桶。"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("qqq_btc.live.vx_term")

DEFAULT_VX_TERM = Path(
    "/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet"
)


def vx_term_path() -> Path:
    raw = os.environ.get("QQQ_BTC_VX_TERM_STRUCTURE", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_VX_TERM


@lru_cache(maxsize=2)
def _load_vx_term(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_file():
        logger.warning("VX term structure missing: %s", path)
        return pd.DataFrame(columns=["source_date", "vx_curve_slope"])
    vx = pd.read_parquet(path, columns=["date", "vx_curve_slope"]).copy()
    vx["source_date"] = pd.to_datetime(vx["date"], utc=True).dt.date
    vx = vx.sort_values("source_date").drop_duplicates("source_date", keep="last")
    return vx.reset_index(drop=True)


def clear_vx_term_cache() -> None:
    _load_vx_term.cache_clear()


def prior_vx_curve_slope(trading_day: date, *, path: Optional[Path] = None) -> Optional[float]:
    """返回 trading_day 开盘前可用的 VX2/VX1-1（source_date < trading_day）。"""
    p = path or vx_term_path()
    vx = _load_vx_term(str(p))
    if vx.empty:
        return None
    prior = vx.loc[vx["source_date"] < trading_day]
    if prior.empty:
        return None
    slope = prior.iloc[-1]["vx_curve_slope"]
    try:
        v = float(slope)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def prior_vx_curve_slope_from_ts(ts: float, *, path: Optional[Path] = None) -> Optional[float]:
    from pytz import timezone

    ny = timezone("America/New_York")
    day = datetime.fromtimestamp(float(ts), tz=ny).date()
    return prior_vx_curve_slope(day, path=path)
