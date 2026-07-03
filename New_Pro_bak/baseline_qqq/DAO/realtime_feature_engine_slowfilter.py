#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
试验版慢特征引擎：

- 不改动主线 RealTimeFeatureEngine 的默认行为
- 仅在 5min slow-feature 已经按 completed-bar 对齐之后，
  对“背景型/长视角”特征再施加一层额外平滑
- 方便与当前主线做 A/B 对比
"""

from __future__ import annotations

from typing import Dict, List

import torch

from realtime_feature_engine import RealTimeFeatureEngine, logger


class RealTimeFeatureEngineSlowFilterV1(RealTimeFeatureEngine):
    """
    分层 slow-filter 试验版：

    - 主线版：5min 特征只使用 completed 5min bar
    - 试验版：在此基础上，仅对更偏“背景态”的 slow 特征追加 EMA 平滑

    目标不是把全部 5min 特征重新钝化成 10min+，而是只增强：
    - 波动背景
    - 结构背景
    - 大级别 regime 约束
    """

    EXTRA_SMOOTH_SPANS: Dict[str, int] = {
        # 背景波动/结构特征：允许更慢
        "garman_klass_vol": 4,
        "adx_smooth_10": 4,
        "poc_deviation": 4,
        "bb_width": 4,
        "vix_level": 4,
        # 期权结构特征：也更适合作为背景态而不是即时扳机
        "options_struc_atm_iv": 3,
        "options_flow_skew": 3,
        "options_struc_skew": 3,
        "options_struc_term": 3,
        "options_gamma_accel": 3,
    }

    NO_EXTRA_SMOOTH = {
        # 这些虽然属于 5min 分支，但仍保留更直接的慢趋势含义
        "k",
        "cci",
        "rsi",
        "price_slope_norm_by_atr",
        "price_dist_from_ma_atr",
        "hour",
        "day_of_week",
    }

    @staticmethod
    def _ema_smooth_last_axis(series_2d: torch.Tensor, span: int) -> torch.Tensor:
        if series_2d is None or series_2d.ndim != 2:
            return series_2d
        span = max(1, int(span))
        if span <= 1 or series_2d.shape[-1] <= 1:
            return series_2d

        alpha = 2.0 / (span + 1.0)
        ema = series_2d[:, :1]
        out = [ema]
        for idx in range(1, series_2d.shape[-1]):
            curr = series_2d[:, idx:idx + 1]
            ema = alpha * curr + (1.0 - alpha) * ema
            out.append(ema)
        return torch.cat(out, dim=1)

    def _postprocess_5m_feature_tensor(self, t_5m_raw: torch.Tensor, slow_feats_5m: List[str]) -> torch.Tensor:
        if t_5m_raw is None or t_5m_raw.ndim != 3 or not slow_feats_5m:
            return t_5m_raw

        out = t_5m_raw.clone()
        applied = []
        for feat_idx, feat_name in enumerate(slow_feats_5m):
            if feat_name in self.NO_EXTRA_SMOOTH:
                continue
            span = int(self.EXTRA_SMOOTH_SPANS.get(feat_name, 1) or 1)
            if span <= 1:
                continue
            out[:, feat_idx, :] = self._ema_smooth_last_axis(out[:, feat_idx, :], span=span)
            applied.append(f"{feat_name}:ema{span}")

        if applied:
            profile_key = tuple(applied)
            if getattr(self, "_last_slowfilter_profile_key", None) != profile_key:
                logger.info("🧪 [SlowFilterV1] extra smoothing enabled for 5min features: %s", ", ".join(applied))
                self._last_slowfilter_profile_key = profile_key

        return out
