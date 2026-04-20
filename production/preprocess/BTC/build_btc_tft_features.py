#!/usr/bin/env python3
"""
Build BTC TFT feature tables from CoinAPI downloader outputs.

Input directory layout is expected from:
  production/preprocess/BTC/coinapi_btc_derivatives_downloader.py

Main outputs:
  - features_1min.parquet
  - features_5min.parquet
  - feature_coverage_report.json

Design goals:
  1) Produce columns aligned with production/CONFIG/btc_slow_feature.json
  2) Keep the pipeline robust to metric schema variations
  3) Explicitly report missing data and fallback behavior
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = [
    "time",
    "timestamp",
    "time_period_end",
    "time_period_start",
    "time_close",
    "time_exchange",
    "time_coinapi",
]

METRIC_VALUE_CANDIDATES = [
    "value",
    "metric_value",
    "funding_rate",
    "open_interest",
    "price",
    "close",
    "mark_price",
    "index_price",
]

EPS = 1e-9

# Daily contract mapping rules (tuple-based, fixed in code to avoid runtime drift).
# (min_dte, max_dte), inclusive.
FRONT_DTE_WINDOW: Tuple[int, int] = (14, 60)
NEXT_DTE_WINDOW: Tuple[int, int] = (30, 90)
MIN_DAYS_TO_EXPIRY_FLOOR = 3
LIQUIDITY_PRIORITY: Tuple[str, str] = ("open_interest", "volume_1day_usd")


@dataclass
class SymbolRole:
    role: str
    symbol_id: str
    symbol_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BTC TFT features from CoinAPI raw dumps.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Downloader output dir containing symbols_selected.json and per-symbol folders.",
    )
    parser.add_argument(
        "--config-path",
        default="production/CONFIG/btc_slow_feature.json",
        help="BTC feature config json path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for feature parquet files and coverage report.",
    )
    parser.add_argument(
        "--base-frequency",
        default="1min",
        help="Base frequency for merged timeline, default 1min.",
    )
    parser.add_argument(
        "--label-horizon-minutes",
        type=int,
        default=30,
        help="Forward label horizon in minutes, default 30.",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def pick_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def normalize_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["timestamp"] = ts.dt.tz_convert(None)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out


def pick_metric_value_col(df: pd.DataFrame) -> Optional[str]:
    for col in METRIC_VALUE_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if col in TIMESTAMP_CANDIDATES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def infer_roles(symbols_selected: List[dict], input_dir: Path) -> Dict[str, SymbolRole]:
    selected = sorted(
        symbols_selected,
        key=lambda x: float(x.get("volume_1day_usd") or 0.0),
        reverse=True,
    )
    perps = [x for x in selected if x.get("symbol_type") == "PERPETUAL"]
    futures = [x for x in selected if x.get("symbol_type") == "FUTURES"]

    roles: Dict[str, SymbolRole] = {}
    if perps:
        sid = perps[0]["symbol_id"]
        roles["perp"] = SymbolRole("perp", sid, input_dir / safe_slug(sid))

    if futures:
        sid = futures[0]["symbol_id"]
        roles["front_future"] = SymbolRole("front_future", sid, input_dir / safe_slug(sid))

    if len(futures) > 1:
        sid = futures[1]["symbol_id"]
        roles["next_future"] = SymbolRole("next_future", sid, input_dir / safe_slug(sid))

    return roles


def parse_symbol_expiry(row: dict) -> Optional[pd.Timestamp]:
    """
    Parse futures expiry from metadata or symbol_id.
    Returns naive UTC-normalized date timestamp.
    """
    candidate_fields = [
        "future_delivery_time",
        "future_contract_date",
        "future_expiration_time",
        "expiration",
        "expiry",
        "data_end",
    ]
    for field in candidate_fields:
        val = row.get(field)
        if not val:
            continue
        ts = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.tz_convert(None).normalize()

    sid = str(row.get("symbol_id") or "")
    for token in re.findall(r"(\d{8}|\d{6})", sid):
        if len(token) == 8:
            ts = pd.to_datetime(token, format="%Y%m%d", errors="coerce")
        else:
            ts = pd.to_datetime(token, format="%y%m%d", errors="coerce")
        if pd.notna(ts):
            return ts.normalize()
    return None


def parse_symbol_open_interest(row: dict) -> float:
    candidates = [
        "open_interest",
        "open_interest_1day",
        "open_interest_value",
        "open_interest_contracts",
        "oi",
        "oi_contracts",
        "oi_usd",
    ]
    for field in candidates:
        if field not in row:
            continue
        try:
            val = float(row.get(field) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            return max(0.0, val)
    return 0.0


def parse_symbol_volume(row: dict) -> float:
    candidates = [
        "volume_1day_usd",
        "volume_1day",
        "volume",
        "volume_24h",
        "volume_usd",
    ]
    for field in candidates:
        if field not in row:
            continue
        try:
            val = float(row.get(field) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            return max(0.0, val)
    return 0.0


def safe_slug(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(out)


def load_ohlcv(symbol_dir: Path) -> pd.DataFrame:
    ohlcv_files = sorted(symbol_dir.glob("ohlcv_*.jsonl"))
    if not ohlcv_files:
        return pd.DataFrame()
    df = load_jsonl(ohlcv_files[0])
    if df.empty:
        return df
    ts_col = pick_timestamp_col(df)
    if ts_col is None:
        return pd.DataFrame()
    df = normalize_timestamp(df, ts_col)
    rename_map = {
        "price_open": "open",
        "price_high": "high",
        "price_low": "low",
        "price_close": "close",
        "volume_traded": "volume",
        "trades_count": "trade_count",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "trade_count"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep_cols].sort_values("timestamp")


def load_metric_series(symbol_dir: Path, metric_file: str) -> pd.Series:
    path = symbol_dir / metric_file
    df = load_jsonl(path)
    if df.empty:
        return pd.Series(dtype=float)
    ts_col = pick_timestamp_col(df)
    if ts_col is None:
        return pd.Series(dtype=float)
    df = normalize_timestamp(df, ts_col)
    value_col = pick_metric_value_col(df)
    if value_col is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.values, index=df["timestamp"].values, name=metric_file.replace(".jsonl", ""))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def build_daily_bucket_mapping(
    symbols_selected: List[dict],
    dates: pd.Series,
    min_days_to_expiry: int,
    front_dte_window: Tuple[int, int],
    next_dte_window: Tuple[int, int],
) -> pd.DataFrame:
    front_dte_min, front_dte_max = front_dte_window
    next_dte_min, next_dte_max = next_dte_window
    perps = [
        {
            "symbol_id": x.get("symbol_id"),
            "open_interest": parse_symbol_open_interest(x),
            "volume_1day_usd": parse_symbol_volume(x),
        }
        for x in symbols_selected
        if x.get("symbol_type") == "PERPETUAL"
    ]
    perps = sorted(
        perps,
        key=lambda x: (
            -x["open_interest"],
            -x["volume_1day_usd"],
            str(x["symbol_id"] or ""),
        ),
    )
    futures_raw = [x for x in symbols_selected if x.get("symbol_type") == "FUTURES"]
    futures = []
    for row in futures_raw:
        futures.append(
            {
                "symbol_id": row.get("symbol_id"),
                "expiry": parse_symbol_expiry(row),
                "open_interest": parse_symbol_open_interest(row),
                "volume_1day_usd": parse_symbol_volume(row),
            }
        )
    perp_symbol = perps[0]["symbol_id"] if perps else None

    rows: List[dict] = []
    prev_front = None
    prev_next = None
    prev_front_dte: Optional[int] = None
    prev_next_dte: Optional[int] = None
    unique_days = sorted(pd.to_datetime(dates).dt.normalize().dropna().unique())
    for day in unique_days:
        day = pd.Timestamp(day).normalize()
        eligible: List[dict] = []
        for f in futures:
            expiry = f["expiry"]
            if expiry is None:
                continue
            days_to_expiry = int((expiry - day).days)
            if days_to_expiry > max(0, min_days_to_expiry):
                eligible.append(
                    {
                        "symbol_id": f["symbol_id"],
                        "dte": days_to_expiry,
                        "open_interest": f["open_interest"],
                        "volume_1day_usd": f["volume_1day_usd"],
                    }
                )

        def _pick_best(cands: List[dict]) -> Optional[dict]:
            if not cands:
                return None
            ranked = sorted(
                cands,
                key=lambda x: (
                    -x["open_interest"],
                    -x["volume_1day_usd"],
                    x["dte"],
                    str(x["symbol_id"] or ""),
                ),
            )
            return ranked[0]

        front_pool = [
            x for x in eligible if front_dte_min <= x["dte"] <= front_dte_max
        ]
        front_pick = _pick_best(front_pool)
        if front_pick is None:
            # Fallback 1: relaxed lower bound (keep upper bound).
            front_pick = _pick_best([x for x in eligible if x["dte"] <= front_dte_max])
        if front_pick is None:
            # Fallback 2: any eligible.
            front_pick = _pick_best(eligible)

        front_symbol = front_pick["symbol_id"] if front_pick else prev_front
        front_dte = front_pick["dte"] if front_pick else prev_front_dte

        # Next must satisfy window and strict order: DTE(next) > DTE(front).
        next_pool = []
        if front_dte is not None:
            next_pool = [
                x
                for x in eligible
                if next_dte_min <= x["dte"] <= next_dte_max and x["dte"] > front_dte
            ]
        next_pick = _pick_best(next_pool)
        if next_pick is None and front_dte is not None:
            # Fallback 1: relaxed lower bound, still enforce DTE(next) > DTE(front).
            next_pick = _pick_best([x for x in eligible if x["dte"] <= next_dte_max and x["dte"] > front_dte])
        if next_pick is None and front_dte is not None:
            # Fallback 2: any future after front.
            next_pick = _pick_best([x for x in eligible if x["dte"] > front_dte])

        next_symbol = next_pick["symbol_id"] if next_pick else prev_next
        next_dte = next_pick["dte"] if next_pick else prev_next_dte
        if next_symbol is None:
            next_symbol = front_symbol
            next_dte = front_dte

        prev_front = front_symbol
        prev_next = next_symbol
        prev_front_dte = front_dte
        prev_next_dte = next_dte
        rows.append(
            {
                "date": day,
                "perp_symbol": perp_symbol,
                "front_future_symbol": front_symbol,
                "next_future_symbol": next_symbol,
                "front_dte": front_dte,
                "next_dte": next_dte,
            }
        )

    return pd.DataFrame(rows)


def merge_asof_series(
    base: pd.DataFrame,
    series: pd.Series,
    target_col: str,
    tolerance: str = "30min",
) -> pd.DataFrame:
    if series.empty:
        base[target_col] = np.nan
        return base
    right = pd.DataFrame({"timestamp": pd.to_datetime(series.index), target_col: series.values})
    base_sorted = base.sort_values("timestamp")
    right_sorted = right.sort_values("timestamp")
    merged = pd.merge_asof(
        base_sorted,
        right_sorted,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    return merged


def build_symbol_close_table(input_dir: Path, symbol_ids: Iterable[str]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for symbol_id in sorted({x for x in symbol_ids if x}):
        symbol_dir = input_dir / safe_slug(symbol_id)
        ohlcv = load_ohlcv(symbol_dir)
        if ohlcv.empty:
            continue
        part = ohlcv[["timestamp", "close"]].copy()
        part["symbol_id"] = symbol_id
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "close", "symbol_id"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["symbol_id", "timestamp"]).drop_duplicates(["symbol_id", "timestamp"], keep="last")
    return out


def build_symbol_metric_table(
    input_dir: Path,
    symbol_ids: Iterable[str],
    metric_file: str,
    value_col: str,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for symbol_id in sorted({x for x in symbol_ids if x}):
        symbol_dir = input_dir / safe_slug(symbol_id)
        series = load_metric_series(symbol_dir, metric_file)
        if series.empty:
            continue
        part = pd.DataFrame({"timestamp": pd.to_datetime(series.index), value_col: series.values})
        part["symbol_id"] = symbol_id
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["timestamp", value_col, "symbol_id"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["symbol_id", "timestamp"]).drop_duplicates(["symbol_id", "timestamp"], keep="last")
    return out


def merge_asof_by_symbol(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_symbol_col: str,
    right_symbol_col: str,
    right_value_col: str,
    out_col: str,
    tolerance: str = "30min",
) -> pd.DataFrame:
    if right.empty:
        out = left.copy()
        out[out_col] = np.nan
        return out

    work_left = left.copy().reset_index(drop=False).rename(columns={"index": "_row_id"})
    work_left["_merge_symbol"] = work_left[left_symbol_col].fillna("__MISSING__").astype(str)
    work_right = right.copy()
    work_right["_merge_symbol"] = work_right[right_symbol_col].fillna("__MISSING__").astype(str)

    merged = pd.merge_asof(
        work_left.sort_values(["_merge_symbol", "timestamp"]),
        work_right.sort_values(["_merge_symbol", "timestamp"])[["timestamp", "_merge_symbol", right_value_col]],
        on="timestamp",
        by="_merge_symbol",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    merged = merged.sort_values("_row_id")
    out = left.copy()
    out[out_col] = merged[right_value_col].values
    return out


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / (sig.replace(0, np.nan) + EPS)


def compute_features(df_1m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df_1m.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["close_log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["volume_log"] = np.log1p(out["volume"].clip(lower=0))
    out["volume_ratio"] = out["volume"] / (out["volume"].rolling(30).mean() + EPS)
    out["range_pct"] = (out["high"] - out["low"]) / (out["close"].replace(0, np.nan))
    out["trade_count_log"] = np.log1p(out["trade_count"].clip(lower=0))

    out["deriv_funding_rate"] = out["funding_rate"]
    out["deriv_funding_change"] = out["funding_rate"].diff()
    out["deriv_funding_zscore"] = rolling_zscore(out["funding_rate"], 288)

    out["deriv_open_interest"] = out["open_interest"]
    out["deriv_oi_change"] = out["open_interest"].diff()
    out["deriv_oi_momentum"] = out["open_interest"].pct_change()
    out["deriv_price_oi_divergence"] = out["close"].pct_change() - out["open_interest"].pct_change()

    out["deriv_mark_price_diff"] = (
        (out["mark_price"] - out["close"]) / (out["close"].replace(0, np.nan))
    )
    out["deriv_index_price_diff"] = (
        (out["index_price"] - out["close"]) / (out["close"].replace(0, np.nan))
    )

    out["deriv_basis_perp_vs_index"] = (
        (out["perp_price"] - out["index_price"]) / (out["index_price"].replace(0, np.nan))
    )
    out["deriv_basis_momentum"] = out["deriv_basis_perp_vs_index"].diff()
    out["deriv_current_future_basis"] = (
        (out["future_price_front"] - out["index_price"]) / (out["index_price"].replace(0, np.nan))
    )
    out["deriv_next_future_basis"] = (
        (out["future_price_next"] - out["index_price"]) / (out["index_price"].replace(0, np.nan))
    )
    out["deriv_basis_slope"] = out["deriv_next_future_basis"] - out["deriv_current_future_basis"]

    # Trade-flow and liquidation side split are not guaranteed from current downloader output.
    out["deriv_aggressive_buy_volume"] = 0.0
    out["deriv_aggressive_sell_volume"] = 0.0
    out["deriv_trade_imbalance"] = 0.0
    out["deriv_liquidation_long"] = 0.0
    out["deriv_liquidation_short"] = 0.0
    out["deriv_liquidation_imbalance"] = 0.0

    out["return_5m"] = out["close"].pct_change(5)
    out["realized_vol_5m"] = out["close_log_return"].rolling(5).std() * math.sqrt(5.0)
    out["garman_klass_vol"] = np.sqrt(
        0.5 * np.log((out["high"] / out["low"]).replace(0, np.nan)) ** 2
        - (2.0 * np.log(2.0) - 1.0) * np.log((out["close"] / out["open"]).replace(0, np.nan)) ** 2
    )
    out["bb_width"] = (
        2.0
        * out["close"].rolling(20).std()
        / (out["close"].rolling(20).mean().replace(0, np.nan))
    )
    out["volatility_expansion"] = rolling_zscore(out["realized_vol_5m"], 60)
    out["rsi"] = compute_rsi(out["close"], window=14)
    out["adx_smooth_10"] = compute_adx(out["high"], out["low"], out["close"], window=10)
    out["price_slope_norm_by_atr"] = compute_slope_over_atr(
        out["close"], out["high"], out["low"], window=14
    )

    out["hour"] = out["timestamp"].dt.hour.astype("int64")
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype("int64")

    # 1min table: keep all computed columns so downstream can select by config.
    out_1m = out.copy()

    # 5min table: right-labeled bar close convention.
    out_5m = (
        out.set_index("timestamp")
        .resample("5min", label="right", closed="right")
        .last()
        .reset_index()
    )
    return out_1m, out_5m


def add_labels(
    df: pd.DataFrame,
    horizon_steps: int,
    vol_window_steps: int,
    direction_multiplier: float = 2.5,
    event_multiplier: float = 6.0,
) -> pd.DataFrame:
    """
    参考旧股票脚本的 triple-barrier 风格标签逻辑：
      - label_return_fwd: 固定 horizon 的未来收益
      - label_direction: 0/1/2 (down/flat/up)
      - label_event: 极端波动事件
      - label_volatility: 未来窗口收益标准差
    """
    out = df.copy()
    if out.empty:
        for col, default in [
            ("label_direction", 1),
            ("label_event", 0),
            ("label_volatility", 0.0),
            ("label_return_fwd", 0.0),
        ]:
            out[col] = default
        return out

    horizon_steps = max(1, int(horizon_steps))
    vol_window_steps = max(20, int(vol_window_steps))

    price_col = "close"
    out["log_ret"] = np.log(out[price_col] / out[price_col].shift(1).replace(0, np.nan)).fillna(0.0)
    current_vol = out["log_ret"].rolling(window=vol_window_steps, min_periods=max(5, vol_window_steps // 3)).std()

    close_arr = out[price_col].to_numpy()
    high_arr = out["high"].to_numpy()
    low_arr = out["low"].to_numpy()
    n = len(out)
    valid_len = max(0, n - horizon_steps)

    if valid_len > 0 and n >= horizon_steps + 1:
        future_high = np.append(high_arr[1:], [np.nan])
        future_low = np.append(low_arr[1:], [np.nan])
        future_high = np.nan_to_num(future_high, nan=close_arr[-1])
        future_low = np.nan_to_num(future_low, nan=close_arr[-1])

        win_high = np.lib.stride_tricks.sliding_window_view(
            future_high[: n - 1], window_shape=horizon_steps
        )[:valid_len]
        win_low = np.lib.stride_tricks.sliding_window_view(
            future_low[: n - 1], window_shape=horizon_steps
        )[:valid_len]
        curr_close = close_arr[:valid_len]
        curr_vol = current_vol.to_numpy()[:valid_len]

        dir_thresh = (curr_vol * direction_multiplier).clip(min=0.002, max=0.03)
        upper_barrier = curr_close * (1.0 + dir_thresh)
        lower_barrier = curr_close * (1.0 - dir_thresh)

        hit_up = win_high >= upper_barrier[:, None]
        hit_down = win_low <= lower_barrier[:, None]
        first_up_idx = np.argmax(hit_up, axis=1)
        first_down_idx = np.argmax(hit_down, axis=1)
        any_up = np.any(hit_up, axis=1)
        any_down = np.any(hit_down, axis=1)

        is_up = any_up & (~any_down | (first_up_idx < first_down_idx))
        is_down = any_down & (~any_up | (first_down_idx < first_up_idx))

        dir_labels = np.ones(valid_len, dtype=int)
        dir_labels[is_up] = 2
        dir_labels[is_down] = 0

        evt_thresh = (curr_vol * event_multiplier).clip(min=0.005)
        max_h = np.max(win_high, axis=1)
        min_l = np.min(win_low, axis=1)
        safe_close = np.where(curr_close > 1e-9, curr_close, np.inf)
        ratio_up = max_h / safe_close - 1.0
        ratio_down = 1.0 - min_l / safe_close
        evt_labels = ((ratio_up > evt_thresh) | (ratio_down > evt_thresh)).astype(int)

        out["label_direction"] = 1
        out["label_event"] = 0
        out.iloc[:valid_len, out.columns.get_loc("label_direction")] = dir_labels
        out.iloc[:valid_len, out.columns.get_loc("label_event")] = evt_labels
    else:
        out["label_direction"] = 1
        out["label_event"] = 0

    out["label_volatility"] = out["log_ret"].rolling(window=horizon_steps).std().shift(-horizon_steps).fillna(0.0)
    out["label_return_fwd"] = (
        out[price_col].shift(-horizon_steps) / out[price_col].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out.drop(columns=["log_ret"], inplace=True, errors="ignore")
    return out


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window).mean()
    plus_di = 100.0 * pd.Series(plus_dm).rolling(window).mean() / (atr + EPS)
    minus_di = 100.0 * pd.Series(minus_dm).rolling(window).mean() / (atr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + EPS)
    return dx.rolling(window).mean()


def compute_slope_over_atr(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 14,
) -> pd.Series:
    slope = close.diff(window) / max(window, 1)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window).mean()
    return slope / (atr + EPS)


def build_base_frame(perp_ohlcv: pd.DataFrame, base_frequency: str) -> pd.DataFrame:
    if perp_ohlcv.empty:
        raise ValueError("Perp OHLCV is empty; cannot build feature timeline.")
    base = perp_ohlcv.copy()
    base = base.set_index("timestamp").sort_index()
    start, end = base.index.min(), base.index.max()
    grid = pd.date_range(start=start, end=end, freq=base_frequency)
    base = base.reindex(grid)
    base.index.name = "timestamp"
    base = base.reset_index()
    for c in ["open", "high", "low", "close"]:
        base[c] = base[c].ffill()
    for c in ["volume", "trade_count"]:
        base[c] = base[c].fillna(0.0)
    return base


def apply_config_column_selection(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = config.get("features", [])
    cols_1m = [f["name"] for f in features if f.get("resolution", "1min") == "1min"]
    cols_5m = [f["name"] for f in features if f.get("resolution", "1min") == "5min"]

    coverage = {"1min": {}, "5min": {}}
    label_keys = list((config.get("labels") or {}).keys())
    label_cols = [f"label_{k}" for k in label_keys]

    def ensure_cols(df: pd.DataFrame, cols: Iterable[str], bucket: str) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            exists = c in out.columns
            coverage[bucket][c] = "present" if exists else "missing_filled_zero"
            if not exists:
                out[c] = 0.0
        existing_labels = []
        for c in label_cols:
            exists = c in out.columns
            coverage[bucket][c] = "present" if exists else "missing_filled_zero"
            if not exists:
                out[c] = 0.0
            existing_labels.append(c)
        ordered = ["timestamp"] + list(cols) + existing_labels
        return out[ordered]

    return ensure_cols(df_1m, cols_1m, "1min"), ensure_cols(df_5m, cols_5m, "5min"), coverage


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config_path)

    symbols_path = input_dir / "symbols_selected.json"
    if not symbols_path.exists():
        raise FileNotFoundError(f"Missing {symbols_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config {config_path}")

    symbols_selected = load_json(symbols_path)
    if not isinstance(symbols_selected, list):
        raise ValueError("symbols_selected.json must be a list.")

    roles = infer_roles(symbols_selected, input_dir)
    if "perp" not in roles:
        raise ValueError("No PERPETUAL symbol found in symbols_selected.json")

    # Base timeline uses the main perp contract.
    perp_ohlcv = load_ohlcv(roles["perp"].symbol_dir)
    base = build_base_frame(perp_ohlcv, args.base_frequency)
    base["date"] = base["timestamp"].dt.normalize()

    # Daily contract-to-bucket mapping (stable semantic buckets).
    daily_mapping = build_daily_bucket_mapping(
        symbols_selected=symbols_selected,
        dates=base["date"],
        min_days_to_expiry=MIN_DAYS_TO_EXPIRY_FLOOR,
        front_dte_window=FRONT_DTE_WINDOW,
        next_dte_window=NEXT_DTE_WINDOW,
    )
    if daily_mapping.empty:
        raise ValueError("Failed to build daily bucket mapping from symbols_selected.json")
    base = base.merge(daily_mapping, on="date", how="left")

    all_mapped_symbols = set()
    for col in ["perp_symbol", "front_future_symbol", "next_future_symbol"]:
        all_mapped_symbols.update(base[col].dropna().astype(str).tolist())

    close_table = build_symbol_close_table(input_dir, all_mapped_symbols)
    base = merge_asof_by_symbol(
        base,
        close_table,
        left_symbol_col="perp_symbol",
        right_symbol_col="symbol_id",
        right_value_col="close",
        out_col="perp_price",
        tolerance="30min",
    )
    base = merge_asof_by_symbol(
        base,
        close_table,
        left_symbol_col="front_future_symbol",
        right_symbol_col="symbol_id",
        right_value_col="close",
        out_col="future_price_front",
        tolerance="30min",
    )
    base = merge_asof_by_symbol(
        base,
        close_table,
        left_symbol_col="next_future_symbol",
        right_symbol_col="symbol_id",
        right_value_col="close",
        out_col="future_price_next",
        tolerance="30min",
    )

    # Perp-side metrics follow daily perp mapping.
    perp_symbols = base["perp_symbol"].dropna().astype(str).unique().tolist()
    metric_defs = {
        "funding.jsonl": "funding_rate",
        "open_interest.jsonl": "open_interest",
        "mark_price.jsonl": "mark_price",
        "index_price.jsonl": "index_price",
        "liquidation.jsonl": "liquidation_total",
    }
    for metric_file, col_name in metric_defs.items():
        metric_table = build_symbol_metric_table(input_dir, perp_symbols, metric_file=metric_file, value_col=col_name)
        base = merge_asof_by_symbol(
            base,
            metric_table,
            left_symbol_col="perp_symbol",
            right_symbol_col="symbol_id",
            right_value_col=col_name,
            out_col=col_name,
            tolerance="30min",
        )

    # Fallbacks for critical fields.
    if "index_price" in base.columns:
        base["index_price"] = base["index_price"].fillna(base["close"])
    else:
        base["index_price"] = base["close"]

    if "mark_price" in base.columns:
        base["mark_price"] = base["mark_price"].fillna(base["close"])
    else:
        base["mark_price"] = base["close"]

    base["funding_rate"] = base.get("funding_rate", pd.Series(index=base.index, dtype=float)).fillna(0.0)
    base["open_interest"] = base.get("open_interest", pd.Series(index=base.index, dtype=float)).ffill().fillna(0.0)

    # If next future is missing, keep flat slope by copying front price.
    base["future_price_front"] = base["future_price_front"].ffill().fillna(base["close"])
    base["future_price_next"] = base["future_price_next"].ffill().fillna(base["future_price_front"])
    base["perp_price"] = base["perp_price"].ffill().fillna(base["close"])

    df_1m, df_5m = compute_features(base)

    # 标签统一目标：未来30分钟收益（两路分辨率分别折算步长）。
    label_horizon_minutes = max(1, int(args.label_horizon_minutes))
    horizon_1m = label_horizon_minutes
    horizon_5m = max(1, int(round(label_horizon_minutes / 5.0)))
    vol_window_1m = 120
    vol_window_5m = max(20, int(round(120 / 5.0)))

    df_1m = add_labels(df_1m, horizon_steps=horizon_1m, vol_window_steps=vol_window_1m)
    df_5m = add_labels(df_5m, horizon_steps=horizon_5m, vol_window_steps=vol_window_5m)

    config = load_json(config_path)
    final_1m, final_5m, coverage = apply_config_column_selection(df_1m, df_5m, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_1m = output_dir / "features_1min.parquet"
    out_5m = output_dir / "features_5min.parquet"
    out_cov = output_dir / "feature_coverage_report.json"
    out_mapping = output_dir / "daily_contract_bucket_mapping.parquet"

    final_1m.to_parquet(out_1m, index=False)
    final_5m.to_parquet(out_5m, index=False)
    daily_mapping.to_parquet(out_mapping, index=False)
    with out_cov.open("w", encoding="utf-8") as f:
        order_violation_count = int(
            (
                (daily_mapping["front_dte"].notna())
                & (daily_mapping["next_dte"].notna())
                & (daily_mapping["next_dte"] <= daily_mapping["front_dte"])
            ).sum()
        )
        json.dump(
            {
                "roles": {k: v.symbol_id for k, v in roles.items()},
                "coverage": coverage,
                "label_horizon_minutes": label_horizon_minutes,
                "label_horizon_steps": {"1min": horizon_1m, "5min": horizon_5m},
                "mapping_rules": {
                    "min_days_to_expiry_floor": int(MIN_DAYS_TO_EXPIRY_FLOOR),
                    "front_dte_window": list(FRONT_DTE_WINDOW),
                    "next_dte_window": list(NEXT_DTE_WINDOW),
                    "priority": list(LIQUIDITY_PRIORITY),
                    "strict_order": "DTE(front) < DTE(next)",
                },
                "mapping_rows": int(len(daily_mapping)),
                "mapping_unique_front": int(daily_mapping["front_future_symbol"].nunique(dropna=True)),
                "mapping_unique_next": int(daily_mapping["next_future_symbol"].nunique(dropna=True)),
                "mapping_order_violation_count": order_violation_count,
                "n_rows_1m": int(len(final_1m)),
                "n_rows_5m": int(len(final_5m)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] Saved: {out_1m}")
    print(f"[OK] Saved: {out_5m}")
    print(f"[OK] Saved: {out_mapping}")
    print(f"[OK] Saved: {out_cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
