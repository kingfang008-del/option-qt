#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal 进程集成 —— 在保留 SignalEngineV8 Redis/FCS/ALPHA_FRAME 外壳的前提下,
替换 slow_model 为 qqq_btc DualStreamAlphaNet(checkpoint v2)。

不修改 signal_engine_v8.py;通过子类 + 启动入口 run_live_signal_qqq.py 接入。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.signal_integration")


def load_qqq_btc_slow_model(
    checkpoint: str | Path,
    config_path: str | Path = qcfg.FEATURE_CONFIG_PATH,
    device: Optional[torch.device] = None,
) -> DualStreamAlphaNet:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    caps = resolve_embedding_caps(cfg)
    model = DualStreamAlphaNet(cfg, caps).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def create_qqq_btc_signal_engine(
    SignalEngineV8,
    *,
    checkpoint: str | Path,
    config_path: str | Path = qcfg.FEATURE_CONFIG_PATH,
):
    """
    工厂:返回 SignalEngineV8 子类,slow_model 指向 qqq_btc checkpoint。
    需在 import SignalEngineV8 之后调用(通常从 New_Pro/baseline_qqq 路径 import)。
    """
    ckpt_ref = Path(checkpoint)
    cfg_ref = Path(config_path)

    ckpt_ref = Path(checkpoint)
    cfg_ref = Path(config_path)

    class QqqBtcSignalEngine(SignalEngineV8):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._qqq_btc_feature_history: dict = {}

        def _load_models(self):
            super()._load_models()
            if not ckpt_ref.exists():
                raise FileNotFoundError(
                    f"qqq_btc checkpoint required but missing: {ckpt_ref}"
                )
            if not cfg_ref.exists():
                raise FileNotFoundError(
                    f"qqq_btc feature config required but missing: {cfg_ref}"
                )
            try:
                self.slow_model = load_qqq_btc_slow_model(
                    ckpt_ref, cfg_ref, device=getattr(self, "device", None)
                )
                self.slow_cfg = json.loads(cfg_ref.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"qqq_btc slow_model load failed: {e}") from e
            logger.info("qqq_btc slow_model loaded from %s", ckpt_ref)

        async def _run_model_inference(self, batch, symbols, prices, ny_now):
            from qqq_btc.live.se_feature_bridge import inject_qqq_btc_features

            inject_qqq_btc_features(
                batch,
                symbols,
                slow_cfg=getattr(self, "slow_cfg", {}),
                history_store=self._qqq_btc_feature_history,
            )
            return await super()._run_model_inference(batch, symbols, prices, ny_now)

    return QqqBtcSignalEngine
