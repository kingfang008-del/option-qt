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
        block_missing = bool(self.cfg.get("block_on_missing", False))
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

        def decision(allow: bool, reason: str) -> RegimeDecision:
            return RegimeDecision(
                allow=allow,
                reason=reason,
                qqq_from_prev=qfp,
                qqq_mf10=qmf,
                vix_reversal=vrev,
                vixy_z=vz,
            )

        vmax = self.cfg.get("vix_reversal_max")
        if vmax is not None and vrev is not None and vrev > float(vmax):
            return decision(False, "vix_reversal")
        if bool(self.cfg.get("qqq_align", True)):
            eps = float(self.cfg.get("qqq_from_prev_eps", 0.0))
            if qfp is None and block_missing:
                return decision(False, "qqq_missing")
            if qfp is not None:
                if direction == "UP" and qfp < -eps:
                    return decision(False, "qqq_align_up")
                if direction == "DN" and qfp > eps:
                    return decision(False, "qqq_align_dn")
        if bool(self.cfg.get("qqq_mf10_align", False)):
            if qmf is None and block_missing:
                return decision(False, "qqq_mf10_missing")
            if qmf is not None:
                if direction == "UP" and qmf <= 0:
                    return decision(False, "qqq_mf10_up")
                if direction == "DN" and qmf >= 0:
                    return decision(False, "qqq_mf10_dn")
        put_min = self.cfg.get("put_vixy_z_min")
        if put_min is not None and direction == "DN":
            if vz is None and block_missing:
                return decision(False, "vixy_z_missing")
            if vz is not None and vz < float(put_min):
                return decision(False, "put_vixy_z")
        return decision(True, "ok" if qfp is not None else "regime_missing")
