#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容层 —— 模型底座已内化到 qqq_btc.model.backbone / losses / dataset / train,
不再依赖 New_Pro。本模块仅保留旧名字的别名,已有引用无需修改。

新代码请直接使用:
  from qqq_btc.model.backbone import DualStreamAlphaNet, freeze_for_finetune
  from qqq_btc.model.losses import NetEdgeLoss
"""
from __future__ import annotations

from .backbone import (
    QUANTILES,
    DualStreamAlphaNet,
    FINETUNE_TRAINABLE_PREFIXES,
    MonotoneQuantileHead,
    PerSymbolCalibrator,
    freeze_for_finetune,
    load_pretrain_checkpoint,
)
from .losses import NetEdgeLoss, pinball_loss

# 旧名字别名
QQQAlphaNetV2 = DualStreamAlphaNet
QQQNetEdgeLossV2 = NetEdgeLoss
freeze_for_qqq_finetune = freeze_for_finetune

__all__ = [
    "QUANTILES",
    "DualStreamAlphaNet",
    "QQQAlphaNetV2",
    "NetEdgeLoss",
    "QQQNetEdgeLossV2",
    "MonotoneQuantileHead",
    "PerSymbolCalibrator",
    "FINETUNE_TRAINABLE_PREFIXES",
    "freeze_for_finetune",
    "freeze_for_qqq_finetune",
    "load_pretrain_checkpoint",
    "pinball_loss",
]
