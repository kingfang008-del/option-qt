#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal 进程集成 —— 在保留 SignalEngineV8 Redis/FCS/ALPHA_FRAME 外壳的前提下,
替换 slow_model 为 qqq_btc DualStreamAlphaNet(checkpoint v4)。

不修改 signal_engine_v8.py;通过子类 + 启动入口 run_live_signal_qqq.py 接入。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from qqq_btc.live.regime_ctx import REGIME_CTX_KEYS, extract_regime_ctx
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

    class QqqBtcSignalEngine(SignalEngineV8):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._qqq_btc_feature_history: dict = {}
            self._qqq_btc_regime_by_sym: Dict[str, dict] = {}
            self._qqq_btc_q10_by_sym: Dict[str, float] = {}

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

        def _extract_slow_model_scores(self, model_out: dict) -> dict:
            """qqq_btc: 始终提取 call/put 双头与 net_edge_q10(不依赖 BIDIRECTIONAL env)。"""
            from strategy.regime import dual_edges_from_model_out, pick_tradable_side

            exec_cost = model_out.get("execution_cost")
            if exec_cost is not None:
                exec_cost = exec_cost.detach().cpu().numpy().reshape(-1)
            else:
                exec_cost = np.zeros(1, dtype=np.float32)

            call_edges = put_edges = None
            if "call_net_edge" in model_out and "put_net_edge" in model_out:
                try:
                    call_t, put_t = dual_edges_from_model_out(model_out)
                    call_edges = call_t.detach().cpu().numpy().reshape(-1)
                    put_edges = put_t.detach().cpu().numpy().reshape(-1)
                    edge = np.zeros_like(call_edges, dtype=np.float32)
                    th = 0.0
                    for i in range(len(call_edges)):
                        _dir, signed, _ = pick_tradable_side(
                            float(call_edges[i]), float(put_edges[i]), threshold=th,
                        )
                        edge[i] = signed if _dir != 0 else max(float(call_edges[i]), float(put_edges[i]))
                except Exception:
                    call_edges = put_edges = None

            if call_edges is None:
                scores = super()._extract_slow_model_scores(model_out)
                return scores

            q10_arr = None
            q10_t = model_out.get("net_edge_q10")
            if q10_t is not None:
                q10_arr = q10_t.detach().cpu().numpy().reshape(-1)
            self._qqq_btc_q10_arr = q10_arr

            return {
                "edge": edge,
                "execution_cost": exec_cost,
                "call_edge": call_edges,
                "put_edge": put_edges,
            }

        async def _run_model_inference(self, batch, symbols, prices, ny_now):
            from qqq_btc.live.se_feature_bridge import inject_qqq_btc_features

            inject_qqq_btc_features(
                batch,
                symbols,
                slow_cfg=getattr(self, "slow_cfg", {}),
                history_store=self._qqq_btc_feature_history,
            )
            self._qqq_btc_regime_by_sym = extract_regime_ctx(
                batch,
                symbols,
                history_store=self._qqq_btc_feature_history,
            )
            self._qqq_btc_q10_by_sym.clear()
            self._qqq_btc_q10_arr = None
            preds = await super()._run_model_inference(batch, symbols, prices, ny_now)
            q10_arr = getattr(self, "_qqq_btc_q10_arr", None)
            if q10_arr is not None:
                for i_p, s_p in enumerate(symbols):
                    if i_p < len(q10_arr):
                        self._qqq_btc_q10_by_sym[s_p] = float(q10_arr[i_p])
            return preds

        async def _publish_alpha_frame(self, *args, alpha_items=None, **kwargs):
            if alpha_items:
                for item in alpha_items:
                    sym = item.get("symbol")
                    if not sym:
                        continue
                    reg = self._qqq_btc_regime_by_sym.get(sym, {})
                    for key in REGIME_CTX_KEYS:
                        if key in reg:
                            item[key] = reg[key]
                    q10 = self._qqq_btc_q10_by_sym.get(sym)
                    if q10 is not None:
                        item["net_edge_q10"] = q10
            return await super()._publish_alpha_frame(*args, alpha_items=alpha_items, **kwargs)

    return QqqBtcSignalEngine
