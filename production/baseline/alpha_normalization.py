#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Standalone alpha normalization helpers for research and A/B analysis."""

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ALPHA_BETA_BUCKETS = {
    'TSLA': 'high_beta', 'NVDA': 'high_beta', 'MSTR': 'high_beta',
    'COIN': 'high_beta', 'SMCI': 'high_beta', 'PLTR': 'high_beta',
    'HOOD': 'high_beta', 'CRWV': 'high_beta',

    'AAPL': 'mega_tech', 'MSFT': 'mega_tech', 'AMZN': 'mega_tech',
    'META': 'mega_tech', 'GOOGL': 'mega_tech', 'AVGO': 'mega_tech',
    'AMD': 'mega_tech', 'ORCL': 'mega_tech', 'ADBE': 'mega_tech',
    'NFLX': 'mega_tech', 'MU': 'mega_tech',

    'WMT': 'defensive', 'UNH': 'defensive', 'XOM': 'defensive',
    'NKE': 'defensive', 'DELL': 'defensive', 'INTC': 'defensive',

    'SPY': 'index', 'QQQ': 'index', 'IWM': 'index', 'GLD': 'index',
    'VIXY': 'index',
}


def _to_finite_float(value) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def _clip_alpha_z(value: float, clip: Optional[float]) -> float:
    if clip is None:
        return float(value)
    return float(max(-clip, min(clip, value)))


def _alpha_pairs(
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    exclude_symbols: Optional[Iterable[str]] = None,
) -> list[tuple[str, float]]:
    excluded = set(exclude_symbols or [])
    pairs = []
    for sym, alpha in zip(symbols, raw_alphas):
        sym = str(sym)
        if sym in excluded:
            continue
        val = _to_finite_float(alpha)
        if val is not None:
            pairs.append((sym, val))
    return pairs


def alpha_zscore_cross_section(
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    exclude_symbols: Optional[Iterable[str]] = None,
    *,
    clip: Optional[float] = 5.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """全市场截面 zscore: 当前分钟所有可交易标的一起算 mean/std."""
    pairs = _alpha_pairs(symbols, raw_alphas, exclude_symbols)
    if not pairs:
        return {}, {'mean': 0.0, 'std': 1.0, 'count': 0}

    vals = np.array([v for _, v in pairs], dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    z_by_symbol = {sym: _clip_alpha_z((val - mean) / (std + eps), clip) for sym, val in pairs}
    return z_by_symbol, {'mean': mean, 'std': std, 'count': len(pairs)}


def alpha_zscore_bucket_cross_section(
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    symbol_buckets: Optional[Mapping[str, str]] = None,
    exclude_symbols: Optional[Iterable[str]] = None,
    *,
    min_bucket_size: int = 3,
    clip: Optional[float] = 5.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """分桶截面 zscore: high beta / mega tech / defensive 等桶内单独算."""
    buckets = dict(DEFAULT_ALPHA_BETA_BUCKETS)
    if symbol_buckets:
        buckets.update({str(k): str(v) for k, v in symbol_buckets.items()})

    pairs = _alpha_pairs(symbols, raw_alphas, exclude_symbols)
    global_z, global_stats = alpha_zscore_cross_section(
        [s for s, _ in pairs],
        [v for _, v in pairs],
        exclude_symbols=None,
        clip=clip,
        eps=eps,
    )

    grouped: Dict[str, list[tuple[str, float]]] = {}
    for sym, val in pairs:
        grouped.setdefault(buckets.get(sym, 'other'), []).append((sym, val))

    z_by_symbol: Dict[str, float] = {}
    stats_by_bucket: Dict[str, Dict[str, float]] = {'__global__': global_stats}
    for bucket, items in grouped.items():
        vals = np.array([v for _, v in items], dtype=float)
        use_global = len(items) < min_bucket_size
        if use_global:
            stats_by_bucket[bucket] = {
                'mean': global_stats['mean'],
                'std': global_stats['std'],
                'count': len(items),
                'fallback_global': 1.0,
            }
            for sym, _ in items:
                z_by_symbol[sym] = global_z.get(sym, 0.0)
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        denom = std if std > eps else 1.0
        stats_by_bucket[bucket] = {
            'mean': mean,
            'std': std,
            'count': len(items),
            'fallback_global': 0.0,
        }
        for sym, val in items:
            z_by_symbol[sym] = _clip_alpha_z((val - mean) / (denom + eps), clip)

    return z_by_symbol, stats_by_bucket


def alpha_zscore_symbol_rolling(
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    history_by_symbol: Mapping[str, Sequence[float]],
    exclude_symbols: Optional[Iterable[str]] = None,
    *,
    window: int = 120,
    min_periods: int = 30,
    clip: Optional[float] = 5.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """单标的 rolling zscore: 每个 symbol 只和自己的历史 raw alpha 比."""
    z_by_symbol: Dict[str, float] = {}
    stats_by_symbol: Dict[str, Dict[str, float]] = {}
    excluded = set(exclude_symbols or [])

    for sym, alpha in zip(symbols, raw_alphas):
        sym = str(sym)
        if sym in excluded:
            continue
        val = _to_finite_float(alpha)
        if val is None:
            continue

        hist = []
        for item in list(history_by_symbol.get(sym, []))[-window:]:
            hist_val = _to_finite_float(item)
            if hist_val is not None:
                hist.append(hist_val)

        if len(hist) < min_periods:
            z_by_symbol[sym] = 0.0
            stats_by_symbol[sym] = {'mean': 0.0, 'std': 1.0, 'count': len(hist), 'cold_start': 1.0}
            continue

        arr = np.array(hist, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        denom = std if std > eps else 1.0
        z_by_symbol[sym] = _clip_alpha_z((val - mean) / (denom + eps), clip)
        stats_by_symbol[sym] = {'mean': mean, 'std': std, 'count': len(hist), 'cold_start': 0.0}

    return z_by_symbol, stats_by_symbol


def alpha_zscore_mixed(
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    history_by_symbol: Mapping[str, Sequence[float]],
    symbol_buckets: Optional[Mapping[str, str]] = None,
    exclude_symbols: Optional[Iterable[str]] = None,
    *,
    symbol_weight: float = 0.6,
    bucket_weight: float = 0.4,
    rolling_window: int = 120,
    rolling_min_periods: int = 30,
    min_bucket_size: int = 3,
    clip: Optional[float] = 5.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """混合口径: per-symbol rolling zscore + beta/风格分桶截面 zscore."""
    total_weight = symbol_weight + bucket_weight
    if total_weight <= eps:
        symbol_weight, bucket_weight, total_weight = 0.6, 0.4, 1.0

    symbol_z, symbol_stats = alpha_zscore_symbol_rolling(
        symbols,
        raw_alphas,
        history_by_symbol,
        exclude_symbols=exclude_symbols,
        window=rolling_window,
        min_periods=rolling_min_periods,
        clip=clip,
        eps=eps,
    )
    bucket_z, bucket_stats = alpha_zscore_bucket_cross_section(
        symbols,
        raw_alphas,
        symbol_buckets=symbol_buckets,
        exclude_symbols=exclude_symbols,
        min_bucket_size=min_bucket_size,
        clip=clip,
        eps=eps,
    )

    z_by_symbol: Dict[str, float] = {}
    for sym in set(symbol_z) | set(bucket_z):
        s_z = symbol_z.get(sym)
        b_z = bucket_z.get(sym)
        if s_z is None:
            mixed = b_z if b_z is not None else 0.0
        elif b_z is None:
            mixed = s_z
        else:
            mixed = (symbol_weight * s_z + bucket_weight * b_z) / total_weight
        z_by_symbol[sym] = _clip_alpha_z(mixed, clip)

    return z_by_symbol, {
        'symbol_stats': symbol_stats,
        'bucket_stats': bucket_stats,
        'symbol_weight': symbol_weight / total_weight,
        'bucket_weight': bucket_weight / total_weight,
    }


def normalize_alpha_scores(
    mode: str,
    symbols: Sequence[str],
    raw_alphas: Sequence[float],
    history_by_symbol: Optional[Mapping[str, Sequence[float]]] = None,
    symbol_buckets: Optional[Mapping[str, str]] = None,
    exclude_symbols: Optional[Iterable[str]] = None,
    *,
    rolling_window: int = 120,
    rolling_min_periods: int = 30,
    min_bucket_size: int = 3,
    symbol_weight: float = 0.6,
    bucket_weight: float = 0.4,
    clip: Optional[float] = 5.0,
    eps: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Single adapter for switching alpha zscore modes without touching callers."""
    normalized_mode = str(mode or 'cross_section').strip().lower()
    normalized_mode = normalized_mode.replace('-', '_')
    if normalized_mode in {'global', 'global_zscore', 'cross', 'cross_sectional'}:
        normalized_mode = 'cross_section'
    elif normalized_mode in {'bucket_cross_section', 'bucketed', 'tier', 'tiered'}:
        normalized_mode = 'bucket'
    elif normalized_mode in {'symbol', 'per_symbol', 'rolling_symbol'}:
        normalized_mode = 'rolling'

    global_z, global_stats = alpha_zscore_cross_section(
        symbols,
        raw_alphas,
        exclude_symbols=exclude_symbols,
        clip=clip,
        eps=eps,
    )

    if normalized_mode == 'cross_section':
        return global_z, {
            'mode': normalized_mode,
            'global_stats': global_stats,
            'count': global_stats.get('count', 0),
        }

    if normalized_mode == 'bucket':
        z_by_symbol, bucket_stats = alpha_zscore_bucket_cross_section(
            symbols,
            raw_alphas,
            symbol_buckets=symbol_buckets,
            exclude_symbols=exclude_symbols,
            min_bucket_size=min_bucket_size,
            clip=clip,
            eps=eps,
        )
        return z_by_symbol, {
            'mode': normalized_mode,
            'global_stats': global_stats,
            'bucket_stats': bucket_stats,
            'count': global_stats.get('count', 0),
        }

    if normalized_mode == 'rolling':
        z_by_symbol, symbol_stats = alpha_zscore_symbol_rolling(
            symbols,
            raw_alphas,
            history_by_symbol or {},
            exclude_symbols=exclude_symbols,
            window=rolling_window,
            min_periods=rolling_min_periods,
            clip=clip,
            eps=eps,
        )
        return z_by_symbol, {
            'mode': normalized_mode,
            'global_stats': global_stats,
            'symbol_stats': symbol_stats,
            'count': global_stats.get('count', 0),
        }

    if normalized_mode == 'mixed':
        z_by_symbol, mixed_stats = alpha_zscore_mixed(
            symbols,
            raw_alphas,
            history_by_symbol or {},
            symbol_buckets=symbol_buckets,
            exclude_symbols=exclude_symbols,
            symbol_weight=symbol_weight,
            bucket_weight=bucket_weight,
            rolling_window=rolling_window,
            rolling_min_periods=rolling_min_periods,
            min_bucket_size=min_bucket_size,
            clip=clip,
            eps=eps,
        )
        mixed_stats = dict(mixed_stats)
        mixed_stats.update({
            'mode': normalized_mode,
            'global_stats': global_stats,
            'count': global_stats.get('count', 0),
        })
        return z_by_symbol, mixed_stats

    raise ValueError(
        f"Unsupported alpha zscore mode: {mode!r}. "
        "Use cross_section, bucket, rolling, or mixed."
    )
