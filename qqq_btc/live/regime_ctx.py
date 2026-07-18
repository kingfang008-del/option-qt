#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 FCS batch / 分钟 close 历史提取 replay 门控所需 regime 字段。"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import numpy as np

from qqq_btc.live.fcs_adapter import enrich_fcs_bars

# choose_entry / replay_session 门控依赖的 ctx 键(OMS item → strategy ctx)
REGIME_CTX_KEYS = (
    # bounce-cut 入场护栏依赖该原始分钟特征；必须随 alpha item 进入 OMS ctx。
    "vwap_log_return",
    # 使用与模型/FCS frame 同标签的分钟 close，不能退回领先一拍的 OMS tick price。
    "spot_close",
    "vix_reversal_count_30m",
    "spot_day_ret",
    "spot_ret_5bar",
    "spot_ret_15bar",
    "vix_ret_15bar",
    "vix_proxy_close",
    "trend_fit_ret_30m",
    "trend_fit_r2_30m",
    "spot_range_30m",
    "open30_ret",
    "open30_max_ret",
    "open30_peak_dd",
    "vix_level",
    "day_range_pos",
    "bb_width",
)

# 需要 ≥30 根 close 才能因果算满的门控键。SE 本地 history 若不足
# (开盘后未预热前日 bar),enrich 会 nan→0,盖住 FCS features_dict 真值。
_ROLLING_30_REGIME_KEYS = frozenset(
    (
        "trend_fit_ret_30m",
        "trend_fit_r2_30m",
        "spot_range_30m",
        "vix_reversal_count_30m",
    )
)
_MIN_BARS_FOR_ROLLING_30 = 30


def _last_seq_val(arr: Any, sym_idx: int) -> Optional[float]:
    if arr is None:
        return None
    try:
        row = np.asarray(arr, dtype=np.float64)[sym_idx]
    except (IndexError, TypeError, ValueError):
        return None
    if row.ndim == 0:
        v = float(row)
    else:
        v = float(row.reshape(-1)[-1])
    return v if np.isfinite(v) else None


def _seq_return(arr: Any, sym_idx: int, bars: int) -> Optional[float]:
    if arr is None:
        return None
    try:
        row = np.asarray(arr, dtype=np.float64)[sym_idx].reshape(-1)
    except (IndexError, TypeError, ValueError):
        return None
    if len(row) <= bars:
        return None
    p0, p1 = float(row[-bars - 1]), float(row[-1])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return float(p1 / p0 - 1.0)


def _spot_ret_5bar_from_frame(frame) -> Optional[float]:
    if frame is None or len(frame) < 6 or "close" not in frame.columns:
        return None
    c0 = float(frame.iloc[-6]["close"])
    c1 = float(frame.iloc[-1]["close"])
    if c0 <= 0 or not np.isfinite(c0) or not np.isfinite(c1):
        return None
    return float(c1 / c0 - 1.0)


def _ret_from_frame(frame, col: str, bars: int) -> Optional[float]:
    if frame is None or len(frame) <= bars or col not in frame.columns:
        return None
    p0 = float(frame.iloc[-bars - 1][col])
    p1 = float(frame.iloc[-1][col])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return float(p1 / p0 - 1.0)


def extract_regime_ctx(
    batch: Mapping[str, Any],
    symbols: List[str],
    *,
    history_store: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    """按 symbol 提取门控特征;优先 enriched 历史,其次 batch features_dict 末值。"""
    fd = batch.get("features_dict") or {}
    out: Dict[str, Dict[str, float]] = {}

    for i, sym in enumerate(symbols):
        vals: Dict[str, float] = {}
        hist = history_store.get(sym)
        frame = hist.to_frame() if hist is not None and hasattr(hist, "to_frame") else None
        hist_len = 0 if frame is None or frame.empty else len(frame)
        rolling30_ok = hist_len >= _MIN_BARS_FOR_ROLLING_30
        if frame is not None and not frame.empty:
            enriched = enrich_fcs_bars(frame, price_col="close")
            last = enriched.iloc[-1]
            try:
                spot_close = float(last["close"])
            except (KeyError, TypeError, ValueError):
                spot_close = float("nan")
            if np.isfinite(spot_close) and spot_close > 0:
                vals["spot_close"] = spot_close
            for key in REGIME_CTX_KEYS:
                # 短历史 rolling-30 的 0 是假值,留给 features_dict 回填
                if key in _ROLLING_30_REGIME_KEYS and not rolling30_ok:
                    continue
                if key not in last.index:
                    continue
                try:
                    v = float(last[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    vals[key] = v
            if rolling30_ok:
                sr5 = _spot_ret_5bar_from_frame(enriched)
                if sr5 is not None:
                    vals["spot_ret_5bar"] = sr5
            else:
                # 5bar 收益也需要足够历史;不足时同样走 fd
                pass
            sr15 = _ret_from_frame(enriched, "close", 15)
            vr15 = _ret_from_frame(enriched, "vix_proxy_close", 15)
            if sr15 is not None:
                vals["spot_ret_15bar"] = sr15
            if vr15 is not None:
                vals["vix_ret_15bar"] = vr15

        # 本地 history 仅保存 QQQ；VIXY proxy 从 FCS 原始序列因果计算。
        if "vix_ret_15bar" not in vals:
            vr15 = _seq_return(fd.get("vix_proxy_close"), i, 15)
            if vr15 is not None:
                vals["vix_ret_15bar"] = vr15

        for key in REGIME_CTX_KEYS:
            if key in vals:
                continue
            v = _last_seq_val(fd.get(key), i)
            if v is not None:
                vals[key] = v

        if vals:
            out[sym] = vals
    return out


def merge_regime_into_ctx(ctx: MutableMapping[str, Any], item: Mapping[str, Any]) -> None:
    """将 alpha item 上的 regime 字段合并进 OMS strategy ctx。"""
    for key in REGIME_CTX_KEYS:
        if key not in item:
            continue
        raw = item.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            ctx[key] = v
