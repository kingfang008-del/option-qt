#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线 1min 特征金标 → live put_trend / open30 等门控 asof。

Stream SE 本地分钟 close 与 offline quote_features_test 在趋势翻转点可差 1 个 tick,
导致 put_trend_max_ret=0 在 10:57 误放行(offline 仍 >0)。对拍时直接读金标列。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qqq_btc.live.regime_ctx import REGIME_CTX_KEYS

logger = logging.getLogger("qqq_btc.live.regime_gold")

_DEFAULT_1M = (
    Path.home()
    / "train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/1min"
)


class RegimeGold1m:
    """backward-asof 读取 offline 1min 门控列。"""

    def __init__(self) -> None:
        self._ts = np.array([], dtype="datetime64[ns]")
        self._cols: Dict[str, np.ndarray] = {}
        self._loaded_from: List[str] = []

    def __len__(self) -> int:
        return int(len(self._ts))

    def load(self, path: str | Path, columns: Optional[List[str]] = None) -> int:
        import pandas as pd

        root = Path(path).expanduser()
        files: list[Path]
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(root.glob("*.parquet"))
        else:
            return 0
        want = list(columns or REGIME_CTX_KEYS)
        frames = []
        for fp in files:
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            keep = ["timestamp"] + [c for c in want if c in df.columns]
            frames.append(df[keep])
            self._loaded_from.append(str(fp))
        if not frames:
            return 0
        all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        all_df = all_df.drop_duplicates(subset=["timestamp"], keep="last")
        self._ts = all_df["timestamp"].to_numpy(dtype="datetime64[ns]")
        self._cols = {}
        for c in want:
            if c in all_df.columns:
                self._cols[c] = pd.to_numeric(all_df[c], errors="coerce").to_numpy(
                    dtype=np.float64
                )
        return len(self._ts)

    def values_at(self, ts) -> Dict[str, float]:
        if len(self._ts) == 0 or ts is None:
            return {}
        import pandas as pd

        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("America/New_York")
            t = t.tz_convert("UTC").to_datetime64()
        except Exception:
            return {}
        idx = int(np.searchsorted(self._ts, t, side="right") - 1)
        if idx < 0:
            return {}
        out: Dict[str, float] = {}
        for c, arr in self._cols.items():
            v = float(arr[idx])
            if np.isfinite(v):
                out[c] = v
        return out


def load_regime_gold_1m() -> Optional[RegimeGold1m]:
    """仅当显式设置 QQQ_BTC_REGIME_GOLD_1M=<path> 时加载。

    空 / 未设置 / 0|off → 关闭（实盘默认必须自算 open30/trend，禁止开卷）。
    开卷诊断脚本须自己 export 金标路径。
    """
    raw = os.environ.get("QQQ_BTC_REGIME_GOLD_1M", "").strip()
    if not raw or raw.lower() in ("0", "false", "no", "off", "none"):
        return None
    path = raw
    gold = RegimeGold1m()
    n = gold.load(path)
    if n <= 0:
        logger.warning("regime gold 1m load failed: %s", path)
        return None
    logger.info("regime gold 1m loaded rows=%d cols=%s from %s", n, sorted(gold._cols), path)
    return gold
