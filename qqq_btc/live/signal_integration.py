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

from qqq_btc.common.vix_level import PutGateFeature5m, Vixy5mGateBuffer, VixyCloseBuffer
from qqq_btc.live.regime_ctx import REGIME_CTX_KEYS, extract_regime_ctx
from qqq_btc.live.regime_gold import load_regime_gold_1m
from qqq_btc.common.session_history import FEATURE_CARRYOVER_BARS
from qqq_btc.live.se_feature_bridge import _SymbolBarHistory
from qqq_btc.model.backbone import DualStreamAlphaNet, resolve_embedding_caps
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.signal_integration")

_VIXY_FALLBACK = Path.home() / "train_data/spnq_train/VIXY"
_VIXY_1M_RESAMPLED = (
    Path.home() / "train_data/spnq_train_resampled/VIXY/regular/09:30-16:00/1min"
)
_QQQ_1M_RESAMPLED = (
    Path.home() / "train_data/spnq_train_resampled/QQQ/regular/09:30-16:00/1min"
)
_PUT_GATE_5M_DEFAULT = (
    Path.home()
    / "train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min"
)
# 与向量化 raw5 put_gate 同序列（已修复的诚实 raw 5min）
_PUT_GATE_RAW5_DEFAULT = (
    Path.home()
    / "train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/5min"
)


def _put_gate_mode() -> str:
    """vixy_z | vixy_5m | feature5m | off。

    默认 vixy_z：因果 1min raw z（与 LIVE put_gate 阈值同标尺）。
    vixy_5m：因果 5min raw z（source=buffer）；asof 仅复现旧离线（有桶内前视）。
    feature5m：显式金标路径（开卷诊断）。
    """
    raw = os.environ.get("QQQ_BTC_PUT_GATE_MODE", "vixy_z").strip().lower()
    if raw in ("0", "false", "no", "off", "none"):
        return "off"
    if raw in ("feature5m", "offline5m", "gold5m"):
        return "feature5m"
    if raw in ("vixy_5m", "vixy5m", "5m_raw", "vixy_5min"):
        return "vixy_5m"
    if raw in ("vixy", "vixy_z", "z", "1min", "5m"):
        # 历史别名 5m 曾指向 feature5m；现仅 vixy_z。开卷请用 feature5m。
        if raw == "5m":
            return "feature5m"
        return "vixy_z"
    return "vixy_z"


def _vixy_5m_source() -> str:
    """buffer（默认，真因果）| asof（旧离线复现，有桶内前视）。"""
    raw = os.environ.get("QQQ_BTC_VIXY_5M_SOURCE", "buffer").strip().lower()
    if raw in ("asof", "file", "parquet", "offline", "leak"):
        return "asof"
    return "buffer"

def _load_put_gate_feature_5m(path: Optional[str] = None) -> Optional[PutGateFeature5m]:
    path = (path or os.environ.get("QQQ_BTC_PUT_GATE_5M_FEATURE", "")).strip()
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


def _load_raw5_asof_gate() -> Optional[PutGateFeature5m]:
    """加载与向量化相同的 raw 5min vix_level asof 源。"""
    path = os.environ.get("QQQ_BTC_PUT_GATE_RAW5", "").strip()
    if not path:
        path = str(_PUT_GATE_RAW5_DEFAULT)
    gate = _load_put_gate_feature_5m(path)
    if gate is not None:
        return gate
    # fallback: VIXY 5min 月文件（含 vix_level）
    alt = Path.home() / "train_data/spnq_train_resampled/VIXY/regular/09:30-16:00/5min"
    return _load_put_gate_feature_5m(str(alt))


def _load_vixy_seed_frame(*, lookback_days: int = 3):
    """返回预热用 VIXY 1m DataFrame(timestamp UTC, close)，无数据则 None。"""
    import pandas as pd

    if os.environ.get("QQQ_BTC_VIXY_SEED", "1").strip().lower() in ("0", "false", "no", "off"):
        return None

    root = Path(
        os.environ.get(
            "QQQ_BTC_VIXY_1M_ROOT",
            str(_VIXY_1M_RESAMPLED if _VIXY_1M_RESAMPLED.exists() else _VIXY_FALLBACK),
        )
    ).expanduser()
    if not root.exists():
        return None
    files = sorted(root.glob("*.parquet"))
    if not files:
        return None
    frames = []
    for fp in files[-2:]:
        try:
            df = pd.read_parquet(fp, columns=["timestamp", "close"])
        except Exception:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    if not frames:
        return None
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
        return None
    use_days = days[-max(1, int(lookback_days)) :]
    return all_df[all_df["day"].isin(use_days)]


def _seed_vixy_buffer(buf: VixyCloseBuffer, *, lookback_days: int = 3) -> int:
    """从本地 VIXY 1m parquet 预热缓冲,避免开盘 put_gate 冷启动。"""
    import pandas as pd

    part = _load_vixy_seed_frame(lookback_days=lookback_days)
    if part is None or part.empty:
        return 0
    closes = pd.to_numeric(part["close"], errors="coerce").dropna().tolist()
    before = len(buf)
    buf.extend(closes)
    return len(buf) - before


def _seed_vixy_5m_buffer(buf: Vixy5mGateBuffer, *, lookback_days: int = 5) -> int:
    """预热 5min put_gate：灌入带时间戳的 1m close（内部再聚合成 5min）。"""
    import pandas as pd

    part = _load_vixy_seed_frame(lookback_days=lookback_days)
    if part is None or part.empty:
        return 0
    before = len(buf)
    ts = part["timestamp"]
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts_unix = ts.view("int64") / 1e9
    else:
        ts_unix = pd.to_datetime(ts, utc=True).view("int64") / 1e9
    closes = pd.to_numeric(part["close"], errors="coerce")
    for t, c in zip(ts_unix.to_numpy(), closes.to_numpy()):
        if np.isfinite(t) and np.isfinite(c) and float(c) > 0:
            buf.append(float(t), float(c))
    return len(buf) - before


def _load_qqq_spot_seed_frame(*, lookback_days: int = 2):
    """返回预热用 QQQ 1m DataFrame(timestamp UTC, close)，供 trend_fit 跨日窗口。"""
    import pandas as pd

    if os.environ.get("QQQ_BTC_SPOT_SEED", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None

    root = Path(
        os.environ.get("QQQ_BTC_QQQ_1M_ROOT", str(_QQQ_1M_RESAMPLED))
    ).expanduser()
    if not root.exists():
        return None
    files = sorted(root.glob("*.parquet"))
    if not files:
        return None
    frames = []
    for fp in files[-3:]:
        try:
            df = pd.read_parquet(fp, columns=["timestamp", "close"])
        except Exception:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    all_df["day"] = all_df["timestamp"].dt.tz_convert("America/New_York").dt.date
    days = sorted(all_df["day"].unique())

    before_raw = os.environ.get(
        "QQQ_BTC_SPOT_SEED_BEFORE",
        os.environ.get("QQQ_BTC_VIXY_SEED_BEFORE", ""),
    ).strip()
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
        return None
    use_days = days[-max(1, int(lookback_days)) :]
    return all_df[all_df["day"].isin(use_days)]


def _seed_qqq_feature_history(
    history_store: dict,
    *,
    symbol: str = "QQQ",
    lookback_days: int = 2,
    max_bars: int = FEATURE_CARRYOVER_BARS,
) -> int:
    """预热 SE 分钟 close 历史,使开盘即有满 30/120 trend 窗口(对齐离线 carryover)。"""
    import pandas as pd

    part = _load_qqq_spot_seed_frame(lookback_days=lookback_days)
    if part is None or part.empty:
        return 0
    part = part.sort_values("timestamp").tail(int(max_bars))
    ts = pd.to_datetime(part["timestamp"], utc=True)
    ts_unix = ts.astype("int64") / 1e9
    closes = pd.to_numeric(part["close"], errors="coerce")
    hist = history_store.setdefault(symbol, _SymbolBarHistory())
    return hist.extend_seed(ts_unix.to_numpy().tolist(), closes.to_numpy().tolist())


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
            self._qqq_btc_vixy_5m_source = _vixy_5m_source()
            self._qqq_btc_vixy_buf = VixyCloseBuffer()
            self._qqq_btc_vixy_5m_buf = Vixy5mGateBuffer()
            self._qqq_btc_put_gate_5m: Optional[PutGateFeature5m] = None
            self._qqq_btc_regime_gold = load_regime_gold_1m()
            n_spot = _seed_qqq_feature_history(self._qqq_btc_feature_history)
            if n_spot:
                logger.info(
                    "QQQ trend feature history seeded +%d bars (len=%d, carryover=%d)",
                    n_spot,
                    len(self._qqq_btc_feature_history.get("QQQ", [])),
                    FEATURE_CARRYOVER_BARS,
                )
            if self._qqq_btc_put_gate_mode == "feature5m":
                self._qqq_btc_put_gate_5m = _load_put_gate_feature_5m()
                if self._qqq_btc_put_gate_5m is None:
                    logger.warning("put_gate feature5m unavailable → fallback vixy_5m")
                    self._qqq_btc_put_gate_mode = "vixy_5m"
            if self._qqq_btc_put_gate_mode == "vixy_5m":
                if self._qqq_btc_vixy_5m_source == "asof":
                    self._qqq_btc_put_gate_5m = _load_raw5_asof_gate()
                    if self._qqq_btc_put_gate_5m is None:
                        logger.warning(
                            "vixy_5m asof raw5 missing → fallback causal buffer"
                        )
                        self._qqq_btc_vixy_5m_source = "buffer"
                    else:
                        logger.info(
                            "vixy_5m source=asof rows=%d (vectorized raw5 put_gate)",
                            len(self._qqq_btc_put_gate_5m),
                        )
                if self._qqq_btc_vixy_5m_source == "buffer":
                    n_seed = _seed_vixy_5m_buffer(
                        self._qqq_btc_vixy_5m_buf, lookback_days=5
                    )
                    if n_seed:
                        logger.info(
                            "VIXY 5m put_gate buffer seeded +%d 1m bars "
                            "(len=%d, raw5_z=%.3f)",
                            n_seed,
                            len(self._qqq_btc_vixy_5m_buf),
                            self._qqq_btc_vixy_5m_buf.raw_level(),
                        )
            elif self._qqq_btc_put_gate_mode == "vixy_z":
                n_seed = _seed_vixy_buffer(self._qqq_btc_vixy_buf, lookback_days=3)
                if n_seed:
                    logger.info(
                        "VIXY put_gate buffer seeded +%d bars (len=%d, raw_z=%.3f)",
                        n_seed,
                        len(self._qqq_btc_vixy_buf),
                        self._qqq_btc_vixy_buf.raw_level(),
                    )
            logger.info(
                "put_gate mode=%s source=%s regime_gold=%s",
                self._qqq_btc_put_gate_mode,
                getattr(self, "_qqq_btc_vixy_5m_source", "-"),
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

        def _vixy_px_from_batch(self, batch, symbols, prices) -> Optional[float]:
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
                    return px
            return None

        def _batch_label_ts(self, batch, ny_now) -> Optional[float]:
            ts = ny_now
            if ts is None:
                try:
                    ts = batch.get("alpha_label_ts") or batch.get("label_ts") or batch.get("ts")
                except Exception:
                    ts = None
            if ts is None:
                return None
            try:
                import pandas as pd

                t = pd.Timestamp(ts)
                if t.tzinfo is None:
                    t = t.tz_localize("America/New_York")
                return float(t.tz_convert("UTC").timestamp())
            except Exception:
                try:
                    return float(ts)
                except Exception:
                    return None

        def _update_vixy_gate_buffer(self, batch, symbols, prices) -> Optional[float]:
            """从 batch 中的 VIXY 收盘价累积缓冲,返回 put_gate 用 vix_level。"""
            px = self._vixy_px_from_batch(batch, symbols, prices)
            if px is not None and px > 0:
                self._qqq_btc_vixy_buf.append(px)
            if len(self._qqq_btc_vixy_buf) < 20:
                return None
            return self._qqq_btc_vixy_buf.gate_level()

        def _update_vixy_5m_gate_buffer(self, batch, symbols, prices, ny_now) -> Optional[float]:
            """因果 5min raw put_gate（与离线 quote_features_raw 5min vix 同口径）。"""
            px = self._vixy_px_from_batch(batch, symbols, prices)
            ts = self._batch_label_ts(batch, ny_now)
            if px is not None and px > 0 and ts is not None:
                self._qqq_btc_vixy_5m_buf.append(ts, px)
            if len(self._qqq_btc_vixy_5m_buf) < 20:
                return None
            return self._qqq_btc_vixy_5m_buf.gate_level()

        def _resolve_put_gate(self, batch, symbols, prices, ny_now) -> Optional[float]:
            mode = getattr(self, "_qqq_btc_put_gate_mode", "vixy_z")
            if mode == "off":
                return None

            def _asof_from_file() -> Optional[float]:
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

            if mode == "feature5m":
                return _asof_from_file()
            if mode == "vixy_5m":
                src = getattr(self, "_qqq_btc_vixy_5m_source", "asof")
                if src == "asof":
                    return _asof_from_file()
                return self._update_vixy_5m_gate_buffer(batch, symbols, prices, ny_now)
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
            # 覆盖 put_gate:默认 vixy_z；开卷 feature5m 仅诊断脚本显式开启
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
