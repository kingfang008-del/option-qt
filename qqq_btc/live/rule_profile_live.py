#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实盘日频 rule profile：与离线 VX selector / CHOP 对齐。

默认使用前一完成日的 VX2/VX1-1 + QQQ lookback（日前 N 个交易日）。
缺失数据时退回 TREND_PUT_OK，避免静默锁仓。
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from qqq_btc.common.rule_profiles import (
    apply_rule_profile,
    compute_qqq_lookback_stats,
    load_rule_profiles,
    prior_lookback_window,
    select_profile_name_vx,
)
from qqq_btc.common.replay_types import ReplayConfig

logger = logging.getLogger("qqq_btc.live.rule_profile")

DEFAULT_SPOT_ROOT = Path.home() / "train_data/spnq_train_resampled"


def _active_profile_value(section: str, key: str) -> Any:
    from qqq_btc.common.strategy_profile import load_active_strategy_profile

    profile = load_active_strategy_profile()
    if profile is None:
        return None
    return (profile.data.get(section) or {}).get(key)


def rule_profile_selector_enabled() -> bool:
    default = _active_profile_value("selector", "mode") or "vx"
    raw = os.environ.get("QQQ_BTC_RULE_PROFILE_SELECTOR", str(default)).strip().lower()
    if raw in ("0", "false", "no", "off", "none", ""):
        return False
    return True


def rule_profile_selector_mode() -> str:
    default = _active_profile_value("selector", "mode") or "vx"
    raw = os.environ.get("QQQ_BTC_RULE_PROFILE_SELECTOR", str(default)).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "vx"
    return raw or "vx"


def spot_root() -> Path:
    raw = os.environ.get("QQQ_BTC_SPOT_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    configured = _active_profile_value("selector", "spot_root")
    if configured:
        from qqq_btc.common.strategy_profile import load_active_strategy_profile, profile_path

        profile = load_active_strategy_profile()
        if profile is not None:
            resolved = profile_path(profile, "selector", "spot_root")
            if resolved is not None:
                return resolved
    return DEFAULT_SPOT_ROOT


@lru_cache(maxsize=4)
def _load_symbol_1m_cached(root_str: str, symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    root = Path(root_str)
    start = date.fromisoformat(start_s)
    end = date.fromisoformat(end_s)
    base = root / symbol / "regular" / "09:30-16:00" / "1min"
    if not base.is_dir():
        return pd.DataFrame()
    months = sorted({f"{d.year:04d}-{d.month:02d}" for d in pd.date_range(start, end, freq="D")})
    frames = []
    for ym in months:
        fp = base / f"{ym}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if "timestamp" not in df.columns:
            continue
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    ny = out["timestamp"].dt.tz_convert("America/New_York").dt.date
    return out.loc[(ny >= start) & (ny <= end)].reset_index(drop=True)


def clear_rule_profile_cache() -> None:
    _load_symbol_1m_cached.cache_clear()


def qqq_lookback_for_day(
    trading_day: date,
    *,
    lookback: int = 5,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """返回日前 lookback 的 QQQ 统计；缺失时返回 nan。"""
    root = root or spot_root()
    # 日历缓冲：约 2×lookback 交易日 + 周末
    start = trading_day - timedelta(days=max(21, lookback * 4))
    end = trading_day - timedelta(days=1)
    qqq = _load_symbol_1m_cached(str(root), "QQQ", start.isoformat(), end.isoformat())
    if qqq.empty or "close" not in qqq.columns:
        return {
            "qqq_up_frac": float("nan"),
            "qqq_range_mean": float("nan"),
            "window": None,
        }
    days = sorted(qqq["timestamp"].dt.tz_convert("America/New_York").dt.date.unique())
    win = prior_lookback_window(days, trading_day, lookback)
    if win is None:
        return {
            "qqq_up_frac": float("nan"),
            "qqq_range_mean": float("nan"),
            "window": None,
        }
    stats = compute_qqq_lookback_stats(qqq, win[0], win[1])
    stats["window"] = (win[0].isoformat(), win[1].isoformat())
    return stats


def select_live_profile(
    trading_day: date,
    *,
    vx_curve_slope: Optional[float],
    profiles_cfg: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """选择当日 profile；返回 (name, meta)。"""
    cfg = profiles_cfg or load_rule_profiles()
    lookback = int((cfg.get("selector") or {}).get("lookback_trading_days", 5))
    qstats = qqq_lookback_for_day(trading_day, lookback=lookback)
    slope = float(vx_curve_slope) if vx_curve_slope is not None else float("nan")
    mode = rule_profile_selector_mode()
    if mode != "vx":
        # 目前实盘只对齐离线 VX 口径；其它模式退回默认。
        name = str(cfg.get("default_profile") or "TREND_PUT_OK")
    else:
        name = select_profile_name_vx(
            slope,
            qqq_up_frac=float(qstats.get("qqq_up_frac", float("nan"))),
            qqq_range_mean=float(qstats.get("qqq_range_mean", float("nan"))),
            profiles_cfg=cfg,
        )
    meta = {
        "profile": name,
        "selector": mode,
        "vx_curve_slope": slope if np.isfinite(slope) else None,
        "qqq_up_frac": qstats.get("qqq_up_frac"),
        "qqq_range_mean": qstats.get("qqq_range_mean"),
        "qqq_lookback_window": qstats.get("window"),
    }
    return name, meta


def apply_live_rule_profile(
    base_cfg: ReplayConfig,
    trading_day: date,
    *,
    vx_curve_slope: Optional[float],
) -> tuple[ReplayConfig, dict[str, Any]]:
    if not rule_profile_selector_enabled():
        return base_cfg, {
            "profile": "DISABLED",
            "selector": "off",
            "vx_curve_slope": vx_curve_slope,
        }
    name, meta = select_live_profile(trading_day, vx_curve_slope=vx_curve_slope)
    try:
        cfg = apply_rule_profile(base_cfg, name)
    except Exception as e:
        logger.warning("apply_rule_profile(%s) failed: %s; keep base", name, e)
        return base_cfg, {**meta, "error": str(e)}
    logger.info(
        "live rule profile day=%s profile=%s vx_slope=%s qqq_up=%.3f range=%.4f",
        trading_day.isoformat(),
        name,
        meta.get("vx_curve_slope"),
        float(meta.get("qqq_up_frac") or float("nan")),
        float(meta.get("qqq_range_mean") or float("nan")),
    )
    return cfg, meta
