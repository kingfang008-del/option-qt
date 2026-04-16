#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BTC 期权特征引擎骨架。

目标：
1) 对齐 FeatureComputeService 期望的 compute_all_inputs 接口；
2) 作为多市场适配入口，后续只需在此文件迭代 BTC 特征逻辑；
3) 当前版本为了可运行，先回退复用现有 RealTimeFeatureEngine 计算链路。

后续替换建议：
- 合约解析与到期时间：改为交易所 BTC 期权命名规范；
- 24/7 波动率与时间特征：去掉美股 RTH 语义；
- 盘口缺失容错：按 crypto 微结构优化 Greeks/IV 与流动性特征。
"""

from __future__ import annotations

from typing import Dict


class BTCRealtimeFeatureEngine:
    """
    BTC 引擎最小骨架。
    当前实现使用 equity 引擎作为 fallback，保证框架可先跑通。
    """

    def __init__(self, *, device: str):
        from realtime_feature_engine import RealTimeFeatureEngine

        self._fallback_engine = RealTimeFeatureEngine(stats_path=None, device=device)

    def compute_all_inputs(self, **kwargs) -> Dict[str, Dict]:
        # TODO(btc): 替换为 BTC 专属特征工程。
        return self._fallback_engine.compute_all_inputs(**kwargs)

