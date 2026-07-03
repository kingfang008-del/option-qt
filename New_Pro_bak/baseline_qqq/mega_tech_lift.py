#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mega-cap technology lift detector and optional alpha gain helper."""

from collections import deque
import math
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MEGA_TECH_SYMBOLS = {
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO',
}


def _finite_float(value, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _safe_zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([_finite_float(v, np.nan) for v in values], dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(len(arr), dtype=float)
    fill = float(np.nanmedian(arr[valid]))
    arr = np.where(valid, arr, fill)
    std = float(np.std(arr))
    if std < 1e-9:
        return np.zeros(len(arr), dtype=float)
    return (arr - float(np.mean(arr))) / std


def _get_symbol_state(
    state: MutableMapping[str, dict],
    symbol: str,
    history_len: int,
) -> dict:
    item = state.get(symbol)
    if item is None:
        item = {
            'prices': deque(maxlen=history_len),
            'volumes': deque(maxlen=history_len),
            'top_duration': 0,
            'volz_duration': 0,
            'impulse_duration': 0,
            'joint_duration': 0,
            'last_rank': None,
            'last_score': 0.0,
        }
        state[symbol] = item
    return item


def update_mega_tech_lift_gain(
    state: MutableMapping[str, dict],
    symbols: Sequence[str],
    prices: Sequence[float],
    volumes: Optional[Sequence[float]],
    alpha_z_by_symbol: Mapping[str, float],
    vol_z_by_symbol: Mapping[str, float],
    *,
    mega_symbols: Optional[Iterable[str]] = None,
    top_n: int = 5,
    min_top_duration: int = 5,
    min_vol_duration: int = 3,
    vol_z_threshold: float = 0.5,
    vol_impulse_threshold: float = 0.5,
    gain: float = 0.25,
    max_gain: float = 0.4,
    history_len: int = 32,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Track sustained dashboard-style momentum and return optional alpha gain.

    The detector intentionally mirrors the dashboard's momentum ingredients:
    5m/15m return, alpha, vol_z, and volume impulse. It adds persistence
    counters so a symbol must stay strong for several minutes before gaining.
    """
    mega_set = set(mega_symbols or DEFAULT_MEGA_TECH_SYMBOLS)
    rows = []

    for idx, sym in enumerate(symbols):
        sym = str(sym)
        px = _finite_float(prices[idx] if idx < len(prices) else 0.0)
        vol = _finite_float(volumes[idx] if volumes is not None and idx < len(volumes) else 0.0)
        sym_state = _get_symbol_state(state, sym, history_len)
        price_hist = sym_state['prices']
        volume_hist = sym_state['volumes']

        px_5 = price_hist[-5] if len(price_hist) >= 5 else 0.0
        px_15 = price_hist[-15] if len(price_hist) >= 15 else 0.0
        vol_avg_20 = float(np.mean(list(volume_hist)[-20:])) if volume_hist else 0.0

        ret_5m = px / px_5 - 1.0 if px > 0 and px_5 > 0 else 0.0
        ret_15m = px / px_15 - 1.0 if px > 0 and px_15 > 0 else 0.0
        vol_impulse = vol / vol_avg_20 - 1.0 if vol_avg_20 > 0 else 0.0
        alpha_z = _finite_float(alpha_z_by_symbol.get(sym, 0.0))
        vol_z = _finite_float(vol_z_by_symbol.get(sym, 0.0))

        rows.append({
            'symbol': sym,
            'price': px,
            'volume': vol,
            'ret_5m': ret_5m,
            'ret_15m': ret_15m,
            'alpha_z': alpha_z,
            'vol_z': vol_z,
            'vol_impulse': vol_impulse,
            'is_mega': sym in mega_set,
        })

    if not rows:
        return {}, {}

    z_ret_5m = _safe_zscore([r['ret_5m'] for r in rows])
    z_ret_15m = _safe_zscore([r['ret_15m'] for r in rows])
    z_alpha = _safe_zscore([r['alpha_z'] for r in rows])
    z_volz = _safe_zscore([r['vol_z'] for r in rows])
    z_impulse = _safe_zscore([r['vol_impulse'] for r in rows])

    for idx, row in enumerate(rows):
        row['momentum_score'] = float(
            0.40 * z_ret_5m[idx]
            + 0.25 * z_ret_15m[idx]
            + 0.20 * z_alpha[idx]
            + 0.10 * z_volz[idx]
            + 0.05 * z_impulse[idx]
        )

    ranked = sorted(rows, key=lambda r: r['momentum_score'], reverse=True)
    rank_by_symbol = {row['symbol']: rank + 1 for rank, row in enumerate(ranked)}
    top_symbols = {row['symbol'] for row in ranked[:max(1, int(top_n))]}

    gain_by_symbol: Dict[str, float] = {}
    stats_by_symbol: Dict[str, dict] = {}
    for row in rows:
        sym = row['symbol']
        sym_state = _get_symbol_state(state, sym, history_len)

        is_top = sym in top_symbols
        is_volz_hot = row['vol_z'] >= vol_z_threshold
        is_impulse_hot = row['vol_impulse'] >= vol_impulse_threshold
        is_joint_hot = is_volz_hot and is_impulse_hot

        sym_state['top_duration'] = sym_state['top_duration'] + 1 if is_top else 0
        sym_state['volz_duration'] = sym_state['volz_duration'] + 1 if is_volz_hot else 0
        sym_state['impulse_duration'] = sym_state['impulse_duration'] + 1 if is_impulse_hot else 0
        sym_state['joint_duration'] = sym_state['joint_duration'] + 1 if is_joint_hot else 0
        sym_state['last_rank'] = rank_by_symbol.get(sym)
        sym_state['last_score'] = row['momentum_score']

        is_lift = (
            row['is_mega']
            and row['ret_15m'] > 0.0
            and sym_state['top_duration'] >= min_top_duration
            and (
                sym_state['joint_duration'] >= min_vol_duration
                or sym_state['volz_duration'] >= min_vol_duration
                or sym_state['impulse_duration'] >= min_vol_duration
            )
        )
        gain_by_symbol[sym] = min(float(gain), float(max_gain)) if is_lift else 0.0
        stats_by_symbol[sym] = {
            'rank': rank_by_symbol.get(sym),
            'momentum_score': row['momentum_score'],
            'ret_5m': row['ret_5m'],
            'ret_15m': row['ret_15m'],
            'vol_z': row['vol_z'],
            'vol_impulse': row['vol_impulse'],
            'top_duration': sym_state['top_duration'],
            'volz_duration': sym_state['volz_duration'],
            'impulse_duration': sym_state['impulse_duration'],
            'joint_duration': sym_state['joint_duration'],
            'is_lift': is_lift,
            'gain': gain_by_symbol[sym],
        }

        sym_state['prices'].append(row['price'])
        sym_state['volumes'].append(row['volume'])

    return gain_by_symbol, stats_by_symbol
