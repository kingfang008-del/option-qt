#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因果 vix_level —— 与离线 generate_vix_level_global(1min) / FCS _compute_vix_global 同公式。

  win=60, min_periods=20, z = (last - mean) / (std + eps), ddof=1

put_gate 与模型特征都应基于同一 raw 序列;若启用 frozen_norm,再对 raw 做列变换。
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import numpy as np

EPS = 1e-6
DEFAULT_WIN = 60
DEFAULT_MIN_PERIODS = 20


def causal_vix_level(
    closes: Sequence[float],
    *,
    win: int = DEFAULT_WIN,
    min_periods: int = DEFAULT_MIN_PERIODS,
    eps: float = EPS,
) -> float:
    """对收盘价序列末值计算因果 z-score;样本不足返回 0。"""
    arr = np.asarray(closes, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < max(2, int(min_periods)):
        return 0.0
    sub = arr[-min(len(arr), int(win)) :]
    if len(sub) < max(2, int(min_periods)):
        return 0.0
    std = float(sub.std(ddof=1))
    return float((sub[-1] - sub.mean()) / (std + eps))


def causal_vix_level_series(
    closes: Sequence[float],
    *,
    win: int = DEFAULT_WIN,
    min_periods: int = DEFAULT_MIN_PERIODS,
    eps: float = EPS,
) -> np.ndarray:
    arr = np.asarray(closes, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        out[i] = causal_vix_level(arr[: i + 1], win=win, min_periods=min_periods, eps=eps)
    return out


def apply_frozen_norm_scalar(raw: float, feature: str = "vix_level") -> float:
    """若 FCS_FROZEN_NORM_PATH 存在且含该列,返回冻结归一化后的门控值。"""
    path = os.environ.get("FCS_FROZEN_NORM_PATH", "").strip()
    if not path or not Path(path).expanduser().exists():
        return float(raw)
    try:
        from qqq_btc.common.frozen_norm import FrozenNormState, _normalize_column

        state = FrozenNormState.from_npz(path)
        if feature not in state.feature_names:
            return float(raw)
        j = state.feature_names.index(feature)
        return float(_normalize_column(np.asarray([raw], dtype=np.float32), state, j)[0])
    except Exception:
        return float(raw)


class VixyCloseBuffer:
    """跨 bar 累积 VIXY close,供 put_gate 与 FCS 同口径重算。"""

    def __init__(self, maxlen: int = 4000) -> None:
        self._closes: Deque[float] = deque(maxlen=int(maxlen))

    def extend(self, closes: Sequence[float]) -> None:
        for c in closes:
            try:
                v = float(c)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                self._closes.append(v)

    def append(self, close: float) -> None:
        self.extend([close])

    def __len__(self) -> int:
        return len(self._closes)

    def raw_level(self) -> float:
        return causal_vix_level(self._closes)

    def gate_level(self) -> float:
        """put_gate 用:raw z-score,若配置了 frozen_norm 则再变换到训练口径。"""
        return apply_frozen_norm_scalar(self.raw_level(), "vix_level")


class PutGateFeature5m:
    """put_gate = 离线 5min 特征树 vix_level 的 backward-asof。

    +37.7% gold 的 PUT 门控读的是 infer 里 asof 进来的 **5min** vix_level
    (见 slow_feature 配置 resolution=5min),不是 1min FCS/VIXY z。
    本类直接读取 quote_features_test 的 5min parquet,与 offline replay 同口径。
    """

    def __init__(self) -> None:
        self._ts = np.array([], dtype="datetime64[ns]")
        self._vals = np.array([], dtype=np.float64)
        self._loaded_from: list[str] = []

    def __len__(self) -> int:
        return int(len(self._vals))

    def load(self, path: str | Path) -> int:
        """加载文件或目录下的 *.parquet(需含 timestamp,vix_level)。"""
        import pandas as pd

        root = Path(path).expanduser()
        files: list[Path]
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(root.glob("*.parquet"))
        else:
            return 0
        frames = []
        for fp in files:
            try:
                df = pd.read_parquet(fp, columns=["timestamp", "vix_level"])
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["vix_level"] = pd.to_numeric(df["vix_level"], errors="coerce")
            frames.append(df.dropna(subset=["timestamp", "vix_level"]))
            self._loaded_from.append(str(fp))
        if not frames:
            return 0
        all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        all_df = all_df.drop_duplicates(subset=["timestamp"], keep="last")
        # 用 UTC ns 做 searchsorted,避免 tz 比较坑
        self._ts = all_df["timestamp"].to_numpy(dtype="datetime64[ns]")
        self._vals = all_df["vix_level"].to_numpy(dtype=np.float64)
        return len(self._vals)

    def gate_at(self, ts) -> Optional[float]:
        """返回 ts 时刻 backward-asof 的 5min vix_level;无数据返回 None。"""
        if len(self._vals) == 0 or ts is None:
            return None
        import pandas as pd

        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("America/New_York")
            t = t.tz_convert("UTC").to_datetime64()
        except Exception:
            return None
        idx = int(np.searchsorted(self._ts, t, side="right") - 1)
        if idx < 0:
            return None
        v = float(self._vals[idx])
        return v if np.isfinite(v) else None
