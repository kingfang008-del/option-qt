"""Regime → 入场规则 profile（因果、无标签）。

v2 三档（由 2026-04/05/06/07 对照提炼）：
  OPEN_DEFENSE  : 高 vix_z → 推迟开仓 + PUT 吃 q10
  CHOP_NO_TRADE : 压缩波动 + 弱上涨占比 → 当日禁新开
  TREND_PUT_OK  : 默认松门控
"""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from qqq_btc.common.replay_types import ReplayConfig

DEFAULT_PROFILES_PATH = (
    Path(__file__).resolve().parents[1] / "CONFIG" / "rule_profiles.json"
)


def load_rule_profiles(path: Optional[Path | str] = None) -> dict[str, Any]:
    p = Path(os.path.expanduser(str(path or DEFAULT_PROFILES_PATH)))
    return json.loads(p.read_text(encoding="utf-8"))


def apply_rule_profile(
    base: ReplayConfig,
    profile_name: str,
    *,
    profiles_cfg: Optional[dict[str, Any]] = None,
) -> ReplayConfig:
    cfg = profiles_cfg or load_rule_profiles()
    profiles = cfg.get("profiles") or {}
    if profile_name not in profiles:
        raise KeyError(f"unknown rule profile: {profile_name}")
    overrides = dict(profiles[profile_name].get("overrides") or {})
    known = {f.name for f in fields(ReplayConfig)}
    clean = {k: v for k, v in overrides.items() if k in known}
    return replace(base, **clean)


def compute_vix_z_mean(
    vixy_1m: pd.DataFrame,
    start: date,
    end: date,
) -> float:
    """VIXY 收盘 log-price 的因果 expanding z，再对 [start,end] 取均值。"""
    if vixy_1m.empty or "close" not in vixy_1m.columns:
        return float("nan")
    df = vixy_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    day = (
        df.groupby("d", sort=True)["close"]
        .agg(last="last")
        .reset_index()
        .rename(columns={"d": "date"})
    )
    x = np.log(pd.to_numeric(day["last"], errors="coerce").astype(float))
    mu = x.expanding(min_periods=5).mean().shift(1)
    sd = x.expanding(min_periods=5).std(ddof=0).shift(1)
    z = (x - mu) / sd.replace(0.0, np.nan)
    day["vix_z"] = z
    m = (day["date"] >= start) & (day["date"] <= end)
    vals = day.loc[m, "vix_z"].dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def compute_qqq_lookback_stats(
    qqq_1m: pd.DataFrame,
    start: date,
    end: date,
) -> dict[str, float]:
    """QQQ lookback：日收益上涨占比、日内 range 均值等。"""
    out = {
        "qqq_up_frac": float("nan"),
        "qqq_range_mean": float("nan"),
        "qqq_ret_mean": float("nan"),
        "qqq_ret_std": float("nan"),
    }
    if qqq_1m.empty:
        return out
    df = qqq_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    g = df.groupby("d", sort=True)
    day = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    )
    day = day.loc[(day.index >= start) & (day.index <= end)]
    if day.empty:
        return out
    ret = day["close"] / day["open"] - 1.0
    rng = (day["high"] - day["low"]) / day["open"]
    out["qqq_up_frac"] = float((ret > 0).mean())
    out["qqq_range_mean"] = float(rng.mean())
    out["qqq_ret_mean"] = float(ret.mean())
    out["qqq_ret_std"] = float(ret.std(ddof=0)) if len(ret) > 1 else float("nan")
    return out


def prior_lookback_window(
    trading_days: list[date],
    asof: date,
    lookback: int,
) -> Optional[tuple[date, date]]:
    """asof 日之前的 lookback 个交易日窗口。"""
    prior = [d for d in trading_days if d < asof]
    if not prior:
        return None
    chunk = prior[-lookback:]
    return chunk[0], chunk[-1]


def week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def select_profile_name(
    vix_z_mean: float,
    *,
    qqq_up_frac: float = float("nan"),
    qqq_range_mean: float = float("nan"),
    profiles_cfg: Optional[dict[str, Any]] = None,
) -> str:
    """优先级：OPEN_DEFENSE > CHOP_NO_TRADE > TREND_PUT_OK。"""
    cfg = profiles_cfg or load_rule_profiles()
    sel = cfg.get("selector") or {}
    default = str(cfg.get("default_profile") or "TREND_PUT_OK")
    open_thr = float(sel.get("open_defense_vix_z_min", 0.5))
    chop_up = float(sel.get("chop_up_frac_max", 0.5))
    chop_rng = float(sel.get("chop_range_mean_max", 0.019))

    if np.isfinite(vix_z_mean) and float(vix_z_mean) >= open_thr:
        return "OPEN_DEFENSE"

    # May/Jun 提炼：压缩波 + 弱涨占比 → 禁开（不要求高 vix）
    if (
        np.isfinite(qqq_up_frac)
        and np.isfinite(qqq_range_mean)
        and float(qqq_up_frac) <= chop_up
        and float(qqq_range_mean) < chop_rng
    ):
        # 仅在非高波防御档时启用（高波走 OPEN_DEFENSE）
        if (not np.isfinite(vix_z_mean)) or float(vix_z_mean) < open_thr:
            return "CHOP_NO_TRADE"

    return default


def select_profile_name_vx(
    vx_curve_slope: float,
    *,
    qqq_up_frac: float = float("nan"),
    qqq_range_mean: float = float("nan"),
    profiles_cfg: Optional[dict[str, Any]] = None,
) -> str:
    """用因果 VX 期限结构替代 VIXY level z 选择 profile。

    slope = VX2/VX1-1。backwardation/近乎平坦用于开盘防御；正向期限结构叠加
    QQQ 中等压缩区间用于 CHOP。缺失时退回默认 TREND，避免静默误锁。
    """
    cfg = profiles_cfg or load_rule_profiles()
    sel = cfg.get("vx_selector") or {}
    default = str(cfg.get("default_profile") or "TREND_PUT_OK")
    open_max = float(sel.get("open_defense_curve_slope_max", 0.01))
    chop_slope_min = float(sel.get("chop_curve_slope_min", 0.03))
    chop_up_max = float(sel.get("chop_up_frac_max", 0.5))
    chop_range_min = float(sel.get("chop_range_mean_min", 0.014))
    chop_range_max = float(sel.get("chop_range_mean_max", 0.019))

    if not np.isfinite(vx_curve_slope):
        return default
    if float(vx_curve_slope) <= open_max:
        return "OPEN_DEFENSE"
    if (
        float(vx_curve_slope) >= chop_slope_min
        and np.isfinite(qqq_up_frac)
        and np.isfinite(qqq_range_mean)
        and float(qqq_up_frac) <= chop_up_max
        and chop_range_min <= float(qqq_range_mean) < chop_range_max
    ):
        return "CHOP_NO_TRADE"
    return default


def assign_daily_profiles(
    trading_days: list[date],
    vixy_1m: pd.DataFrame,
    qqq_1m: pd.DataFrame,
    *,
    profiles_cfg: Optional[dict[str, Any]] = None,
    calendar_days: Optional[list[date]] = None,
) -> dict[date, dict[str, Any]]:
    """每个交易日独立用「日前 lookback」选 profile。"""
    cfg = profiles_cfg or load_rule_profiles()
    lookback = int((cfg.get("selector") or {}).get("lookback_trading_days", 5))
    days = sorted(trading_days)
    cal = sorted(calendar_days) if calendar_days is not None else days

    day_map: dict[date, dict[str, Any]] = {}
    for d in days:
        win = prior_lookback_window(cal, d, lookback)
        if win is None:
            vz = float("nan")
            qstats = {
                "qqq_up_frac": float("nan"),
                "qqq_range_mean": float("nan"),
                "qqq_ret_mean": float("nan"),
                "qqq_ret_std": float("nan"),
            }
            lookback_span = None
        else:
            vz = compute_vix_z_mean(vixy_1m, win[0], win[1])
            qstats = compute_qqq_lookback_stats(qqq_1m, win[0], win[1])
            lookback_span = [win[0].isoformat(), win[1].isoformat()]
        name = select_profile_name(
            vz,
            qqq_up_frac=qstats["qqq_up_frac"],
            qqq_range_mean=qstats["qqq_range_mean"],
            profiles_cfg=cfg,
        )
        day_map[d] = {
            "profile": name,
            "week_start": week_start_monday(d).isoformat(),
            "vix_z_mean_lookback": vz,
            "qqq_up_frac_lookback": qstats["qqq_up_frac"],
            "qqq_range_mean_lookback": qstats["qqq_range_mean"],
            "lookback": lookback_span,
        }
    return day_map


# 兼容旧名
def assign_weekly_profiles(
    trading_days: list[date],
    vixy_1m: pd.DataFrame,
    *,
    profiles_cfg: Optional[dict[str, Any]] = None,
    calendar_days: Optional[list[date]] = None,
    qqq_1m: Optional[pd.DataFrame] = None,
) -> dict[date, dict[str, Any]]:
    if qqq_1m is None:
        qqq_1m = pd.DataFrame()
    return assign_daily_profiles(
        trading_days,
        vixy_1m,
        qqq_1m,
        profiles_cfg=profiles_cfg,
        calendar_days=calendar_days,
    )
