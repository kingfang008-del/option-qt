#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 FCS batch / 分钟 close 历史提取 replay 门控所需 regime 字段。"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import numpy as np

from qqq_btc.live.fcs_adapter import enrich_fcs_bars

# choose_entry / replay_session 门控依赖的 ctx 键(OMS item → strategy ctx)
REGIME_CTX_KEYS = (
    "vix_reversal_count_30m",
    "spot_day_ret",
    "spot_ret_5bar",
    "trend_fit_ret_30m",
    "trend_fit_r2_30m",
    "spot_range_30m",
    "open30_max_ret",
    "open30_peak_dd",
    "vix_level",
    "day_range_pos",
    "bb_width",
)


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


def _spot_ret_5bar_from_frame(frame) -> Optional[float]:
    if frame is None or len(frame) < 6 or "close" not in frame.columns:
        return None
    c0 = float(frame.iloc[-6]["close"])
    c1 = float(frame.iloc[-1]["close"])
    if c0 <= 0 or not np.isfinite(c0) or not np.isfinite(c1):
        return None
    return float(c1 / c0 - 1.0)


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
        if frame is not None and not frame.empty:
            enriched = enrich_fcs_bars(frame, price_col="close")
            last = enriched.iloc[-1]
            for key in REGIME_CTX_KEYS:
                if key in last.index:
                    try:
                        v = float(last[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(v):
                        vals[key] = v
            sr5 = _spot_ret_5bar_from_frame(enriched)
            if sr5 is not None:
                vals["spot_ret_5bar"] = sr5

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
