from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config import NY_TZ


def should_capture_feature_parity(
    *, batch_symbols: list[str], alpha_label_ts: float, target_symbol: str, target_ts: int
) -> bool:
    return target_symbol in batch_symbols and int(float(alpha_label_ts)) == int(target_ts)


def _safe_matrix_6x12(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32) if value is not None else np.zeros((0, 0), dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((6, 12), dtype=np.float32)
    out = np.zeros((6, 12), dtype=np.float32)
    rows = min(6, int(arr.shape[0]))
    cols = min(12, int(arr.shape[1]))
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def build_symbol_feature_parity_snapshot(
    *,
    symbol: str,
    symbol_idx: int,
    features_dict: Dict[str, np.ndarray],
    history_1min: Optional[pd.DataFrame],
    valid_norm_seq: Optional[np.ndarray],
    feat_name_to_idx: Dict[str, int],
    raw_mat: np.ndarray,
    raw_symbol_idx: int,
    normalizer: Any,
    batch_price: float,
    batch_fast_vol: float,
    cheat_call_iv: float,
    cheat_put_iv: float,
    source_opt_buckets: Dict[str, Any],
    source_snap_for_payload: Dict[str, Any],
    frozen_option_snapshot: Any,
    frozen_latest_opt_buckets: Any,
    valid_mask_value: bool,
    real_history_len: int,
    total_history_len: int,
    real_norm_history_len: int,
    has_cross_day_warmup: bool,
    alpha_label_ts: float,
) -> Dict[str, np.ndarray]:
    tail_ts = np.asarray([], dtype=np.int64)
    nvda_hist = history_1min if history_1min is not None else pd.DataFrame()
    if not nvda_hist.empty:
        hist_idx = nvda_hist.index
        if getattr(hist_idx, "tz", None) is None:
            hist_idx = hist_idx.tz_localize(NY_TZ)
        tail_ts = np.asarray([int(ts.timestamp()) for ts in hist_idx[-40:]], dtype=np.int64)

    feature_snapshot = {
        name: np.asarray(features_dict[name][symbol_idx], dtype=np.float32).copy()
        for name in sorted(features_dict.keys())
    }

    cci_idx = feat_name_to_idx.get("cci")
    norm_seq_tail = (
        valid_norm_seq[symbol_idx]
        if valid_norm_seq is not None and len(valid_norm_seq) > symbol_idx
        else None
    )

    raw_ohlcv: Dict[str, np.ndarray] = {}
    derived_5m = pd.DataFrame()
    if not nvda_hist.empty:
        hist_tail_30 = nvda_hist.sort_index().tail(30)
        for col in ("open", "high", "low", "close", "volume"):
            if col in hist_tail_30.columns:
                raw_ohlcv[f"hist_{col}_30"] = np.asarray(
                    hist_tail_30[col].fillna(0.0).values, dtype=np.float32
                )
        if "vwap" in hist_tail_30.columns:
            raw_ohlcv["hist_vwap_30"] = np.asarray(
                hist_tail_30["vwap"].fillna(0.0).values, dtype=np.float32
            )
        if all(col in hist_tail_30.columns for col in ("high", "low", "close")):
            tp = (
                hist_tail_30["high"].astype(np.float64)
                + hist_tail_30["low"].astype(np.float64)
                + hist_tail_30["close"].astype(np.float64)
            ) / 3.0
            tp_sma20 = tp.rolling(20, min_periods=20).mean()
            tp_mad20 = tp.rolling(20, min_periods=20).apply(
                lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
            )
            cci_raw = (tp - tp_sma20) / (0.015 * tp_mad20.replace(0.0, np.nan))
            raw_ohlcv["cci_raw_30"] = np.asarray(cci_raw.fillna(0.0).values, dtype=np.float32)

        if all(col in nvda_hist.columns for col in ("open", "high", "low", "close", "volume")):
            derived_5m = (
                nvda_hist[["open", "high", "low", "close", "volume"]]
                .sort_index()
                .resample("5min", closed="left", label="left")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna(subset=["open", "high", "low", "close"], how="any")
            )
            if not derived_5m.empty:
                tail_5m = derived_5m.tail(30)
                raw_ohlcv["hist_5m_close_30"] = np.asarray(
                    tail_5m["close"].fillna(0.0).values, dtype=np.float32
                )
                raw_ohlcv["hist_5m_high_30"] = np.asarray(
                    tail_5m["high"].fillna(0.0).values, dtype=np.float32
                )
                raw_ohlcv["hist_5m_low_30"] = np.asarray(
                    tail_5m["low"].fillna(0.0).values, dtype=np.float32
                )
                tp_5m = (
                    tail_5m["high"].astype(np.float64)
                    + tail_5m["low"].astype(np.float64)
                    + tail_5m["close"].astype(np.float64)
                ) / 3.0
                tp_5m_sma20 = tp_5m.rolling(20, min_periods=20).mean()
                tp_5m_mad20 = tp_5m.rolling(20, min_periods=20).apply(
                    lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
                )
                cci_raw_5m = (tp_5m - tp_5m_sma20) / (0.015 * tp_5m_mad20.replace(0.0, np.nan))
                raw_ohlcv["cci_raw_5m_30"] = np.asarray(
                    cci_raw_5m.fillna(0.0).values, dtype=np.float32
                )
                raw_ohlcv["history_tail_ts_5m"] = np.asarray(
                    [int(ts.timestamp()) for ts in tail_5m.index], dtype=np.int64
                )

    curr_snap = source_opt_buckets.get(symbol)
    if curr_snap is None:
        curr_snap = source_snap_for_payload.get(symbol)
    curr_snap_arr = _safe_matrix_6x12(curr_snap)
    frozen_snap_arr = _safe_matrix_6x12(frozen_option_snapshot)
    frozen_bucket_arr = _safe_matrix_6x12(frozen_latest_opt_buckets)

    cci_feature_seq = np.asarray(
        feature_snapshot.get("cci", np.zeros(30, dtype=np.float32)), dtype=np.float32
    )
    if cci_feature_seq.shape[0] != 30:
        cci_feature_seq = np.asarray(np.resize(cci_feature_seq, 30), dtype=np.float32)

    cci_norm_seq = np.zeros(30, dtype=np.float32)
    if norm_seq_tail is not None and cci_idx is not None and cci_idx < norm_seq_tail.shape[1]:
        cci_norm_seq = np.asarray(norm_seq_tail[:, cci_idx], dtype=np.float32).copy()
        if cci_norm_seq.shape[0] != 30:
            cci_norm_seq = np.asarray(np.resize(cci_norm_seq, 30), dtype=np.float32)

    cci_raw_latest = 0.0
    if cci_idx is not None and raw_symbol_idx < raw_mat.shape[0] and cci_idx < raw_mat.shape[1]:
        cci_raw_latest = float(raw_mat[raw_symbol_idx, cci_idx])

    cci_norm_mean = 0.0
    cci_norm_std = 1.0
    if normalizer is not None and cci_idx is not None:
        if hasattr(normalizer, "last_mean") and len(normalizer.last_mean) > cci_idx:
            cci_norm_mean = float(normalizer.last_mean[cci_idx])
        if hasattr(normalizer, "last_std") and len(normalizer.last_std) > cci_idx:
            cci_norm_std = float(normalizer.last_std[cci_idx])

    snapshot = {
        **feature_snapshot,
        **raw_ohlcv,
        "stock_price": np.asarray([float(batch_price)], dtype=np.float32),
        "fast_vol": np.asarray([float(batch_fast_vol)], dtype=np.float32),
        "cheat_call_iv": np.asarray([float(cheat_call_iv)], dtype=np.float32),
        "cheat_put_iv": np.asarray([float(cheat_put_iv)], dtype=np.float32),
        "option_snapshot_6x12": curr_snap_arr.copy(),
        "frozen_option_snapshot_6x12": frozen_snap_arr.copy(),
        "frozen_latest_opt_buckets_6x12": frozen_bucket_arr.copy(),
        "cci_feature_seq_30": cci_feature_seq,
        "cci_norm_seq_30": cci_norm_seq,
        "cci_raw_latest": np.asarray([cci_raw_latest], dtype=np.float32),
        "cci_norm_mean": np.asarray([cci_norm_mean], dtype=np.float32),
        "cci_norm_std": np.asarray([cci_norm_std], dtype=np.float32),
        "history_len_5m": np.asarray([int(len(derived_5m))], dtype=np.int32),
        "valid_mask": np.asarray([int(bool(valid_mask_value))], dtype=np.int32),
        "normalizer_count": np.asarray(
            [int(getattr(normalizer, "count", 0))], dtype=np.int32
        ),
        "real_history_len": np.asarray([int(real_history_len)], dtype=np.int32),
        "total_history_len": np.asarray([int(total_history_len)], dtype=np.int32),
        "real_norm_history_len": np.asarray([int(real_norm_history_len)], dtype=np.int32),
        "has_cross_day_warmup": np.asarray([int(bool(has_cross_day_warmup))], dtype=np.int32),
        "alpha_label_ts": np.asarray([int(float(alpha_label_ts))], dtype=np.int64),
        "history_tail_ts": tail_ts,
    }
    return snapshot


def save_feature_parity_snapshot(output_path: str | Path, snapshot: Dict[str, np.ndarray]) -> Path:
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **snapshot)
    return out
