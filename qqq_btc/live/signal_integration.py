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
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from qqq_btc.common.vix_level import PutGateFeature5m, VixyCloseBuffer
from qqq_btc.live.regime_ctx import REGIME_CTX_KEYS, extract_regime_ctx
from qqq_btc.live.regime_gold import load_regime_gold_1m
from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.signal_integration")

_VIXY_FALLBACK = Path.home() / "train_data/spnq_train/VIXY"
_PUT_GATE_5M_DEFAULT = (
    Path.home()
    / "train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min"
)


def _put_gate_mode() -> str:
    """feature5m | vixy_z | off。默认 feature5m(对齐 +37.7% offline put_gate)。"""
    raw = os.environ.get("QQQ_BTC_PUT_GATE_MODE", "feature5m").strip().lower()
    if raw in ("0", "false", "no", "off", "none"):
        return "off"
    if raw in ("vixy", "vixy_z", "z", "1min"):
        return "vixy_z"
    return "feature5m"


def _load_put_gate_feature_5m() -> Optional[PutGateFeature5m]:
    path = os.environ.get("QQQ_BTC_PUT_GATE_5M_FEATURE", "").strip()
    if not path:
        path = str(_PUT_GATE_5M_DEFAULT)
    gate = PutGateFeature5m()
    n = gate.load(path)
    if n <= 0:
        logger.warning("put_gate feature5m load failed: %s", path)
        return None
    logger.info(
        "put_gate feature5m loaded rows=%d from %s",
        n,
        path,
    )
    return gate


def _seed_vixy_buffer(buf: VixyCloseBuffer, *, lookback_days: int = 3) -> int:
    """从本地 VIXY 1m parquet 预热缓冲,避免开盘 put_gate 冷启动。

    Redis 对拍默认也可因果预热:只取严格早于 QQQ_BTC_VIXY_SEED_BEFORE(或今天)
    的交易日,避免灌入「未来」VIXY。设 QQQ_BTC_VIXY_SEED=0 可关闭。
    """
    import pandas as pd

    if os.environ.get("QQQ_BTC_VIXY_SEED", "1").strip().lower() in ("0", "false", "no", "off"):
        return 0

    root = Path(os.environ.get("QQQ_BTC_VIXY_1M_ROOT", str(_VIXY_FALLBACK))).expanduser()
    if not root.exists():
        return 0
    files = sorted(root.glob("*.parquet"))
    if not files:
        return 0
    # 取最近 lookback_days 个有数据的月份文件末尾若干交易日
    frames = []
    for fp in files[-2:]:
        try:
            df = pd.read_parquet(fp, columns=["timestamp", "close"])
        except Exception:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    all_df["day"] = all_df["timestamp"].dt.tz_convert("America/New_York").dt.date
    days = sorted(all_df["day"].unique())

    before_raw = os.environ.get("QQQ_BTC_VIXY_SEED_BEFORE", "").strip()
    if before_raw:
        try:
            if len(before_raw) == 8 and before_raw.isdigit():
                cutoff = pd.Timestamp(
                    f"{before_raw[:4]}-{before_raw[4:6]}-{before_raw[6:8]}"
                ).date()
            else:
                cutoff = pd.Timestamp(before_raw).date()
            days = [d for d in days if d < cutoff]
        except Exception:
            pass

    if not days:
        return 0
    use_days = days[-max(1, int(lookback_days)) :]
    part = all_df[all_df["day"].isin(use_days)]
    closes = pd.to_numeric(part["close"], errors="coerce").dropna().tolist()
    before = len(buf)
    buf.extend(closes)
    return len(buf) - before


def load_qqq_btc_slow_model(
    checkpoint: str | Path,
    config_path: str | Path = qcfg.FEATURE_CONFIG_PATH,
    device: Optional[torch.device] = None,
) -> DualStreamAlphaNet:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = bundle.get("config") if isinstance(bundle, dict) else None
    if cfg is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        logger.info("qqq_btc slow_model using config embedded in checkpoint")
    caps = resolve_embedding_caps(cfg)
    model = DualStreamAlphaNet(cfg, caps).to(device)
    ckpt = (
        bundle.get("state_dict", bundle.get("model_state_dict", bundle))
        if isinstance(bundle, dict)
        else bundle
    )
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in ckpt.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
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
            self._qqq_btc_put_gate_mode = _put_gate_mode()
            self._qqq_btc_vixy_buf = VixyCloseBuffer()
            self._qqq_btc_put_gate_5m: Optional[PutGateFeature5m] = None
            self._qqq_btc_regime_gold = load_regime_gold_1m()
            if self._qqq_btc_put_gate_mode == "feature5m":
                self._qqq_btc_put_gate_5m = _load_put_gate_feature_5m()
                if self._qqq_btc_put_gate_5m is None:
                    logger.warning("put_gate feature5m unavailable → fallback vixy_z")
                    self._qqq_btc_put_gate_mode = "vixy_z"
            if self._qqq_btc_put_gate_mode == "vixy_z":
                n_seed = _seed_vixy_buffer(self._qqq_btc_vixy_buf, lookback_days=3)
                if n_seed:
                    logger.info(
                        "VIXY put_gate buffer seeded +%d bars (len=%d, raw_z=%.3f)",
                        n_seed,
                        len(self._qqq_btc_vixy_buf),
                        self._qqq_btc_vixy_buf.raw_level(),
                    )
            logger.info(
                "put_gate mode=%s regime_gold=%s",
                self._qqq_btc_put_gate_mode,
                "on" if self._qqq_btc_regime_gold is not None else "off",
            )

        def _load_models(self, config_paths, model_paths):
            # Skip legacy AdvancedAlphaNet load; DualStream replaces slow_model.
            if not ckpt_ref.exists():
                raise FileNotFoundError(
                    f"qqq_btc checkpoint required but missing: {ckpt_ref}"
                )
            if not cfg_ref.exists():
                raise FileNotFoundError(
                    f"qqq_btc feature config required but missing: {cfg_ref}"
                )
            try:
                bundle = torch.load(
                    ckpt_ref,
                    map_location=getattr(self, "device", None) or "cpu",
                    weights_only=False,
                )
                embedded = bundle.get("config") if isinstance(bundle, dict) else None
                if embedded is not None:
                    self.slow_cfg = embedded
                else:
                    self.slow_cfg = json.loads(cfg_ref.read_text(encoding="utf-8"))
                self.slow_model = load_qqq_btc_slow_model(
                    ckpt_ref, cfg_ref, device=getattr(self, "device", None)
                )
            except Exception as e:
                raise RuntimeError(f"qqq_btc slow_model load failed: {e}") from e
            logger.info("qqq_btc slow_model loaded from %s", ckpt_ref)

        def _extract_slow_model_scores(self, model_out: dict) -> dict:
            """qqq_btc: call/put 双头 + 无符号 edge(与 strict replay 同口径,禁止 signed -put)。"""
            from strategy.regime import dual_edges_from_model_out

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
                    # 与 replay choose_entry 一致: edge 取双头非负幅度,不做 PUT→负号
                    edge = np.maximum(call_edges, put_edges).astype(np.float32)
                except Exception:
                    call_edges = put_edges = None

            if call_edges is None:
                scores = super()._extract_slow_model_scores(model_out)
                return scores

            # 模型真值头(供 audit);不要用 signed pick_tradable_side 覆盖
            net_t = model_out.get("net_edge")
            raw_t = model_out.get("net_edge_raw")
            if net_t is not None:
                net_arr = net_t.detach().cpu().numpy().reshape(-1)
            else:
                net_arr = edge
            if raw_t is not None:
                raw_arr = raw_t.detach().cpu().numpy().reshape(-1)
            else:
                raw_arr = net_arr

            q10_arr = None
            q10_t = model_out.get("net_edge_q10")
            if q10_t is not None:
                q10_arr = q10_t.detach().cpu().numpy().reshape(-1)
            self._qqq_btc_q10_arr = q10_arr
            self._qqq_btc_net_arr = net_arr
            self._qqq_btc_raw_arr = raw_arr

            return {
                "edge": edge,
                "execution_cost": exec_cost,
                "call_edge": call_edges,
                "put_edge": put_edges,
            }

        def _update_vixy_gate_buffer(self, batch, symbols, prices) -> Optional[float]:
            """从 batch 中的 VIXY 收盘价累积缓冲,返回 put_gate 用 vix_level。"""
            if not symbols:
                return None
            for i, sym in enumerate(symbols):
                if str(sym).upper() != "VIXY":
                    continue
                try:
                    px = float(prices[i]) if prices is not None and i < len(prices) else 0.0
                except Exception:
                    px = 0.0
                if px <= 0:
                    fd = batch.get("features_dict") or {}
                    close_seq = fd.get("close")
                    if close_seq is not None:
                        try:
                            row = np.asarray(close_seq, dtype=np.float64)[i]
                            px = float(row.reshape(-1)[-1])
                        except Exception:
                            px = 0.0
                if px > 0:
                    self._qqq_btc_vixy_buf.append(px)
            if len(self._qqq_btc_vixy_buf) < 20:
                return None
            return self._qqq_btc_vixy_buf.gate_level()

        def _resolve_put_gate(self, batch, symbols, prices, ny_now) -> Optional[float]:
            mode = getattr(self, "_qqq_btc_put_gate_mode", "feature5m")
            if mode == "off":
                return None
            if mode == "feature5m":
                gate5 = getattr(self, "_qqq_btc_put_gate_5m", None)
                if gate5 is None:
                    return None
                ts = ny_now
                if ts is None:
                    try:
                        ts = batch.get("alpha_label_ts") or batch.get("label_ts")
                    except Exception:
                        ts = None
                return gate5.gate_at(ts)
            return self._update_vixy_gate_buffer(batch, symbols, prices)

        async def _run_model_inference(self, batch, symbols, prices, ny_now):
            from qqq_btc.live.se_feature_bridge import inject_qqq_btc_features

            inject_qqq_btc_features(
                batch,
                symbols,
                slow_cfg=getattr(self, "slow_cfg", {}),
                history_store=self._qqq_btc_feature_history,
            )
            gate_vix = self._resolve_put_gate(batch, symbols, prices, ny_now)
            self._qqq_btc_regime_by_sym = extract_regime_ctx(
                batch,
                symbols,
                history_store=self._qqq_btc_feature_history,
            )
            # 覆盖 put_gate:默认用离线 5min vix_level asof(对齐 +37.7% replay)
            if gate_vix is not None:
                for sym in symbols:
                    if str(sym).upper() == "VIXY":
                        continue
                    reg = self._qqq_btc_regime_by_sym.setdefault(sym, {})
                    reg["vix_level"] = float(gate_vix)
            # 门控金标 1min:对齐 offline put_trend / open30(避免 SE 短历史翻转早一拍)
            gold = getattr(self, "_qqq_btc_regime_gold", None)
            if gold is not None:
                ts = ny_now
                if ts is None:
                    try:
                        ts = batch.get("alpha_label_ts") or batch.get("label_ts") or batch.get("ts")
                    except Exception:
                        ts = batch.get("ts")
                gold_vals = gold.values_at(ts)
                if gold_vals:
                    for sym in symbols:
                        if str(sym).upper() == "VIXY":
                            continue
                        reg = self._qqq_btc_regime_by_sym.setdefault(sym, {})
                        for k, v in gold_vals.items():
                            # 保留 feature5m 写入的 vix_level
                            if k == "vix_level":
                                continue
                            reg[k] = v
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
                raw_arr = getattr(self, "_qqq_btc_raw_arr", None)
                net_arr = getattr(self, "_qqq_btc_net_arr", None)
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
                    # audit/OMS: net_edge_raw 用模型真值,不用 signed alpha
                    try:
                        idx = int(item.get("batch_idx", 0) or 0)
                    except (TypeError, ValueError):
                        idx = 0
                    if raw_arr is not None and idx < len(raw_arr):
                        item["net_edge_raw"] = float(raw_arr[idx])
                    if net_arr is not None and idx < len(net_arr):
                        item["net_edge"] = float(net_arr[idx])
                        item["alpha"] = float(max(item.get("call_edge", 0.0) or 0.0, item.get("put_edge", 0.0) or 0.0))
            return await super()._publish_alpha_frame(*args, alpha_items=alpha_items, **kwargs)

    return QqqBtcSignalEngine
