#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal 进程特征补算 —— FCS batch 缺 time/trend 列时,用本地分钟 close 历史注入。

与 fcs_adapter.enrich_fcs_bars 同实现,保证与离线 label_pipeline 一致。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qqq_btc.common.time_features import TIME_FEATURE_NAMES
from qqq_btc.common.trend_features import OPEN30_FEATURE_NAMES, TREND_FEATURE_NAMES
from qqq_btc.live.fcs_adapter import enrich_fcs_bars

SEQ_LEN = 30
DERIVED_NAMES = frozenset(TIME_FEATURE_NAMES + TREND_FEATURE_NAMES + OPEN30_FEATURE_NAMES)


class _SymbolBarHistory:
    __slots__ = ("_ts", "_close")

    def __init__(self, maxlen: int = 200) -> None:
        self._ts: deque = deque(maxlen=maxlen)
        self._close: deque = deque(maxlen=maxlen)

    def append(self, ts: float, close: float) -> None:
        if close <= 0 or not np.isfinite(close):
            return
        self._ts.append(float(ts))
        self._close.append(float(close))

    def to_frame(self) -> pd.DataFrame:
        if not self._ts:
            return pd.DataFrame()
        ts = pd.to_datetime(list(self._ts), unit="s", utc=True)
        return pd.DataFrame({"timestamp": ts, "close": list(self._close)})


def _derived_feature_names(slow_cfg: dict) -> List[str]:
    names = []
    for f in slow_cfg.get("features", []):
        n = f.get("name")
        if n in DERIVED_NAMES:
            names.append(n)
    return names


def inject_qqq_btc_features(
    batch: dict,
    symbols: List[str],
    *,
    slow_cfg: dict,
    history_store: Dict[str, _SymbolBarHistory],
    seq_len: int = SEQ_LEN,
) -> dict:
    """就地补全 batch['features_dict'] 中缺失的 time/trend 序列。"""
    need = _derived_feature_names(slow_cfg)
    if not need or not symbols:
        return batch

    fd: Dict[str, Any] = batch.setdefault("features_dict", {})
    n_sym = len(symbols)
    ts = float(batch.get("ts", 0.0) or 0.0)
    prices = batch.get("stock_price", [0.0] * n_sym)

    for i, sym in enumerate(symbols):
        try:
            close = float(prices[i] if i < len(prices) else 0.0)
        except Exception:
            close = 0.0
        hist = history_store.setdefault(sym, _SymbolBarHistory())
        if ts > 0:
            hist.append(ts, close)

        frame = hist.to_frame()
        if frame.empty:
            continue
        enriched = enrich_fcs_bars(frame, price_col="close")

        for fname in need:
            if fname not in enriched.columns:
                continue
            seq = pd.to_numeric(enriched[fname], errors="coerce").fillna(0.0).values.astype(np.float32)
            if len(seq) > seq_len:
                seq = seq[-seq_len:]
            elif len(seq) < seq_len:
                seq = np.concatenate([np.zeros(seq_len - len(seq), dtype=np.float32), seq])

            arr = fd.get(fname)
            if arr is None:
                arr = np.zeros((n_sym, seq_len), dtype=np.float32)
            else:
                arr = np.asarray(arr, dtype=np.float32).copy()
            if arr.shape[0] < n_sym:
                pad = np.zeros((n_sym - arr.shape[0], seq_len), dtype=np.float32)
                arr = np.vstack([arr, pad])
            # FCS enrich 已写入非零时保留,避免 SE 短历史覆盖正确值
            existing = arr[i, :]
            if np.nanmax(np.abs(existing)) > 1e-8:
                continue
            arr[i, :] = seq
            fd[fname] = arr

    batch["features_dict"] = fd
    return batch
