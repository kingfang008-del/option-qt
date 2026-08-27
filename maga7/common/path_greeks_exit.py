"""L2 path Greeks / IV early-exit rules (winner-protective).

Designed from Jul-20 clock autopsy:
  - naive giveback (half of peak while still +20%) cuts GOOGL winners
  - iv_shock / delta_fade help toxic paths (MSFT / AMD) when gated on weak MTM

Stateful per-position evaluator; feed mid/S each tick after entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from scipy.stats import norm


@dataclass
class PathGreeksExitConfig:
    enabled: bool = False
    # giveback: only flatten when peak was real and MTM collapsed near flat/red
    giveback_peak_min: float = 0.20
    giveback_ret_max: float = 0.05  # must fall to <= this
    giveback_peak_frac: float = 0.25  # or <= peak * frac (whichever higher floor)
    # IV shock: vol up while path soft (not while strong winner)
    iv_shock_min: float = 0.03
    iv_shock_peak_dd: float = 0.10
    iv_shock_ret_max: float = 0.15
    # delta fade: only when underwater
    delta_fade_min: float = 0.15
    delta_fade_ret_max: float = 0.0
    # shared
    min_hold_sec: float = 30.0
    r: float = 0.04

    @staticmethod
    def from_trade(trade: dict[str, Any] | None) -> "PathGreeksExitConfig":
        raw = (trade or {}).get("path_greeks_exit") or {}
        if not isinstance(raw, dict):
            return PathGreeksExitConfig()
        return PathGreeksExitConfig(
            enabled=bool(raw.get("enabled", False)),
            giveback_peak_min=float(raw.get("giveback_peak_min", 0.20)),
            giveback_ret_max=float(raw.get("giveback_ret_max", 0.05)),
            giveback_peak_frac=float(raw.get("giveback_peak_frac", 0.25)),
            iv_shock_min=float(raw.get("iv_shock_min", 0.03)),
            iv_shock_peak_dd=float(raw.get("iv_shock_peak_dd", 0.10)),
            iv_shock_ret_max=float(raw.get("iv_shock_ret_max", 0.15)),
            delta_fade_min=float(raw.get("delta_fade_min", 0.15)),
            delta_fade_ret_max=float(raw.get("delta_fade_ret_max", 0.0)),
            min_hold_sec=float(raw.get("min_hold_sec", 30.0)),
            r=float(raw.get("r", 0.04)),
        )


# Named presets for ablation
PRESETS: dict[str, dict[str, Any]] = {
    "off": {"enabled": False},
    "naive": {
        "enabled": True,
        "giveback_peak_min": 0.15,
        "giveback_ret_max": 1.0,  # effectively half-peak only (see evaluator naive mode)
        "giveback_peak_frac": 0.50,
        "iv_shock_ret_max": 1.0,
        "delta_fade_ret_max": 1.0,
        "_naive_half_peak": True,
    },
    "winner_safe": {
        # giveback only after collapse to flat/red (Jul20: 5% floor still cut GOOGL)
        "enabled": True,
        "giveback_peak_min": 0.20,
        "giveback_ret_max": 0.0,
        "giveback_peak_frac": 0.0,
        "iv_shock_min": 0.03,
        "iv_shock_peak_dd": 0.10,
        "iv_shock_ret_max": 0.15,
        "delta_fade_min": 0.15,
        "delta_fade_ret_max": 0.0,
    },
    "toxic_only": {
        # no giveback; only IV/delta when soft/red
        "enabled": True,
        "giveback_peak_min": 9.0,  # disable
        "iv_shock_min": 0.03,
        "iv_shock_peak_dd": 0.10,
        "iv_shock_ret_max": 0.10,
        "delta_fade_min": 0.15,
        "delta_fade_ret_max": 0.0,
    },
}


def bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if cp == "c" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "c":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return 1.0 if (cp == "c" and S > K) else (-1.0 if cp == "p" and S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1) if cp == "c" else norm.cdf(d1) - 1.0)


def implied_vol(
    mid: float, S: float, K: float, T: float, r: float, cp: str
) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 1e-8:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        px = bs_price(S, K, T, r, m, cp)
        if px > mid:
            hi = m
        else:
            lo = m
    iv = 0.5 * (lo + hi)
    if not (0.01 < iv < 2.5):
        return None
    return float(iv)


@dataclass
class PathGreeksState:
    entry_px: float
    K: float
    cp: str
    expiry_ts: float  # unix seconds
    cfg: PathGreeksExitConfig
    naive_half_peak: bool = False
    entry_ts: float = 0.0
    iv0: float | None = None
    peak_ret: float = 0.0
    peak_delta: float | None = None
    n: int = 0

    def on_tick(
        self, *, ts: float, mid: float, S: float
    ) -> tuple[str | None, dict[str, float]]:
        """Return (reason, metrics) if exit; else (None, metrics)."""
        self.n += 1
        ret = mid / self.entry_px - 1.0 if self.entry_px > 0 else 0.0
        T = max((self.expiry_ts - ts) / 31557600.0, 1e-6)
        iv = implied_vol(mid, S, self.K, T, self.cfg.r, self.cp)
        dlt = bs_delta(S, self.K, T, self.cfg.r, iv, self.cp) if iv else None
        if self.iv0 is None and iv is not None:
            self.iv0 = iv
        if ret > self.peak_ret:
            self.peak_ret = ret
            if dlt is not None:
                self.peak_delta = dlt

        metrics = {
            "opt_ret": float(ret),
            "peak_ret": float(self.peak_ret),
            "iv": float(iv) if iv is not None else float("nan"),
            "delta": float(dlt) if dlt is not None else float("nan"),
        }
        hold = ts - self.entry_ts
        if hold < self.cfg.min_hold_sec or not self.cfg.enabled:
            return None, metrics

        peak = self.peak_ret
        # --- giveback ---
        if self.naive_half_peak:
            if peak >= 0.15 and ret <= peak - 0.5 * peak:
                return "PATH_GIVEBACK", metrics
        else:
            floor = max(self.cfg.giveback_ret_max, peak * self.cfg.giveback_peak_frac)
            if peak >= self.cfg.giveback_peak_min and ret <= floor:
                return "PATH_GIVEBACK", metrics

        # --- iv shock ---
        if (
            iv is not None
            and self.iv0 is not None
            and (iv - self.iv0) >= self.cfg.iv_shock_min
            and ret <= peak - self.cfg.iv_shock_peak_dd
            and ret <= self.cfg.iv_shock_ret_max
        ):
            return "PATH_IV_SHOCK", metrics

        # --- delta fade ---
        if (
            dlt is not None
            and self.peak_delta is not None
            and (self.peak_delta - dlt) >= self.cfg.delta_fade_min
            and ret <= self.cfg.delta_fade_ret_max
        ):
            return "PATH_DELTA_FADE", metrics

        return None, metrics


def cfg_from_preset(name: str) -> tuple[PathGreeksExitConfig, bool]:
    raw = dict(PRESETS.get(name) or PRESETS["winner_safe"])
    naive = bool(raw.pop("_naive_half_peak", False))
    trade = {"path_greeks_exit": raw}
    return PathGreeksExitConfig.from_trade(trade), naive
