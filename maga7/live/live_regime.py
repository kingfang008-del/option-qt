"""Causal live QQQ/VIXY regime gate matching Mag7 offline rules."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.bar_agg import MultiSymbolMinuteAgg
from maga7.common.regime import RegimeDecision
from maga7.common.signals import StreamSignalState
from qqq_btc.common.regime_features import count_reversals


class LiveRegimeGate:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg or {}
        self.agg = MultiSymbolMinuteAgg(["QQQ", "VIXY"], rth_only=True)
        permissive = {
            "mf_window": 10,
            "vol_ma_window": 20,
            "window_start": "00:00",
            "window_end": "23:59",
            "streak_min": 10**9,
            "from_prev_abs": 10**9,
            "vol_z_min": 10**9,
        }
        self.qqq_state = StreamSignalState("QQQ", permissive, emit_all=True)
        self.qqq_previous_close = 0.0
        self.qqq_close = 0.0
        self.vixy_closes: deque[float] = deque(maxlen=max(60, int(self.cfg.get("vix_reversal_window", 30))))

    def on_stock_second(self, symbol: str, tick: dict[str, Any]) -> None:
        symbol = str(symbol).upper()
        if symbol not in {"QQQ", "VIXY"}:
            return
        previous_close = float(tick.get("previous_close") or 0.0)
        if symbol == "QQQ" and previous_close > 0:
            self.qqq_previous_close = previous_close
        if isinstance(tick.get("timestamp"), (int, float)):
            tick = {
                **tick,
                "timestamp": pd.Timestamp(
                    float(tick["timestamp"]), unit="s", tz="UTC"
                ).tz_convert("America/New_York"),
            }
        bar = self.agg.on_second(symbol, tick)
        if bar is None:
            return
        if symbol == "QQQ":
            self.qqq_state.on_bar(bar)
            self.qqq_close = float(bar["close"])
        else:
            self.vixy_closes.append(float(bar["close"]))

    def check(self, direction: str, ts) -> RegimeDecision:
        """Match offline Mag7RegimeGate overlays used by Watchdog experts.

        Honors ``block_directions`` / ``direction_size_scale`` /
        ``scale_dn_if_qqq_above_open`` so predictive prevention actually bites
        in shadow/live (previously these keys were ignored).
        """
        block_missing = bool(self.cfg.get("block_on_missing", False))
        direction_u = str(direction or "").upper()
        qfp = None
        if self.qqq_previous_close > 0 and self.qqq_close > 0:
            qfp = self.qqq_close / self.qqq_previous_close - 1.0
        qmf = float(self.qqq_state.mf10)
        if not np.isfinite(qmf):
            qmf = None
        vrev = None
        if len(self.vixy_closes) >= 2:
            window = int(self.cfg.get("vix_reversal_window", 30))
            values = np.asarray(list(self.vixy_closes)[-window:], dtype=float)
            vrev = float(
                count_reversals(
                    values,
                    threshold=float(self.cfg.get("vix_reversal_pct", 0.0015)),
                )
            )
        vz = None
        if len(self.vixy_closes) >= 20:
            values = np.asarray(self.vixy_closes, dtype=float)
            sd = float(np.std(values, ddof=1))
            if sd > 0:
                vz = float((values[-1] - np.mean(values)) / (sd + 1e-6))

        # Day-open for QQQ (first completed RTH bar of the session).
        qqq_open = None
        bars = getattr(self.qqq_state, "bars", None) or []
        if bars:
            try:
                qqq_open = float(bars[0]["open"])
            except Exception:
                qqq_open = None

        def decision(
            allow: bool, reason: str, *, size_scale: float = 1.0
        ) -> RegimeDecision:
            return RegimeDecision(
                allow=allow,
                reason=reason,
                qqq_from_prev=qfp,
                qqq_mf10=qmf,
                vix_reversal=vrev,
                vixy_z=vz,
                size_scale=float(size_scale),
            )

        vmax = self.cfg.get("vix_reversal_max")
        if vmax is not None and vrev is not None and vrev > float(vmax):
            return decision(False, "vix_reversal")
        if bool(self.cfg.get("qqq_align", True)):
            eps = float(self.cfg.get("qqq_from_prev_eps", 0.0))
            if qfp is None and block_missing:
                return decision(False, "qqq_missing")
            if qfp is not None:
                if direction_u == "UP" and qfp < -eps:
                    return decision(False, "qqq_align_up")
                if direction_u == "DN" and qfp > eps:
                    return decision(False, "qqq_align_dn")

        if bool(self.cfg.get("block_dn_if_qqq_above_open", False)) and direction_u == "DN":
            if (
                qqq_open is not None
                and self.qqq_close > 0
                and self.qqq_close > float(qqq_open)
            ):
                return decision(False, "qqq_above_open_dn")

        block_dirs = self.cfg.get("block_directions") or ()
        if isinstance(block_dirs, str):
            block_dirs = [x.strip() for x in block_dirs.split(",") if x.strip()]
        block_dirs_u = {str(x).upper() for x in block_dirs}
        if direction_u in block_dirs_u:
            return decision(False, f"block_dir_{direction_u.lower()}")

        if bool(self.cfg.get("qqq_mf10_align", False)):
            if qmf is None and block_missing:
                return decision(False, "qqq_mf10_missing")
            if qmf is not None:
                if direction_u == "UP" and qmf <= 0:
                    return decision(False, "qqq_mf10_up")
                if direction_u == "DN" and qmf >= 0:
                    return decision(False, "qqq_mf10_dn")
        put_min = self.cfg.get("put_vixy_z_min")
        if put_min is not None and direction_u == "DN":
            if vz is None and block_missing:
                return decision(False, "vixy_z_missing")
            if vz is not None and vz < float(put_min):
                return decision(False, "put_vixy_z")

        scale = 1.0
        reason = "ok" if qfp is not None else "regime_missing"
        scale_dn_qqq = self.cfg.get("scale_dn_if_qqq_above_open")
        if scale_dn_qqq is not None and direction_u == "DN":
            if (
                qqq_open is not None
                and self.qqq_close > 0
                and self.qqq_close > float(qqq_open)
            ):
                sc = max(0.0, min(float(scale_dn_qqq), 1.0))
                if sc <= 0.0:
                    return decision(False, "qqq_above_open_dn")
                if sc < 1.0:
                    scale *= sc
                    reason = "qqq_above_open_dn_scale"
        dir_scales = self.cfg.get("direction_size_scale") or {}
        if isinstance(dir_scales, dict) and direction_u in dir_scales:
            sc = max(0.0, min(float(dir_scales[direction_u]), 1.0))
            if sc <= 0.0:
                return decision(False, f"dir_scale_{direction_u.lower()}")
            if sc < 1.0:
                scale *= sc
                reason = (
                    f"dir_scale_{direction_u.lower()}"
                    if reason in {"ok", "regime_missing"}
                    else f"{reason}+dir_scale_{direction_u.lower()}"
                )
        return decision(True, reason, size_scale=scale)
