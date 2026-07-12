"""分钟期权参考表（IV parquet）— volume/Greeks 注入，供发球机与 FCS 共用。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

NY_TZ = pytz.timezone("America/New_York")

DEFAULT_GREEK_ROOT = Path.home() / "train_data/quote_options_day_iv"
# 训练数据 (bucketed_v7 → quote_features_raw) 的真实输入是 monthly_iv 月度合并文件；
# daily 文件可能被后续重建覆盖 (行覆盖率不同)，优先 monthly 才能与离线特征对齐。
DEFAULT_MONTHLY_GREEK_ROOT = Path.home() / "train_data/quote_options_monthly_iv"


def resolve_iv_subdirs() -> list[str]:
    """
    分钟 IV parquet 子目录优先级。
    raw_1s 来自 options_databento_v3（仅 bid/ask），Greeks 从日频 IV parquet 注入。
    默认 standard/；可用 FCS_MINUTE_OPTION_IV_SUBDIR 覆盖。
    """
    env = os.environ.get("FCS_MINUTE_OPTION_IV_SUBDIR", "").strip()
    if env:
        return [p for p in env.split(",") if p]
    return ["standard"]


def _align_timestamp_series(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    if col not in df.columns and "ts" in df.columns:
        col = "ts"
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(NY_TZ).dt.round("1s")
    return pd.to_datetime(s, utc=True).dt.tz_convert(NY_TZ).dt.round("1s")


def resolve_greek_day_path(sym: str, date_iso: str, greek_root: Path | None = None) -> Path | None:
    root = Path(greek_root or DEFAULT_GREEK_ROOT)
    candidates: list[Path] = []
    for sub in resolve_iv_subdirs():
        candidates.append(root / sym / sub / f"{sym}_{date_iso}.parquet")
    candidates.append(root / sym / f"{sym}_{date_iso}.parquet")
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_greek_monthly_path(sym: str, date_iso: str) -> Path | None:
    """monthly_iv 月度文件（训练数据 bucketed_v7 的真实输入源）。"""
    month = date_iso[:7]
    env_root = os.environ.get("FCS_MINUTE_OPTION_MONTHLY_IV_ROOT", "").strip()
    root = Path(env_root).expanduser() if env_root else DEFAULT_MONTHLY_GREEK_ROOT
    for sub in resolve_iv_subdirs():
        for p in (
            root / sym / sub / f"{month}.parquet",
            root / sub / f"{month}.parquet",
            root / f"{month}.parquet",
        ):
            if p.exists():
                return p
    return None


def load_minute_option_ref(
    sym: str,
    date_iso: str,
    *,
    greek_root: Path | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """
    分钟 IV parquet → {(minute_unix_ts, bucket_id): {iv, delta, ..., volume}}。
    raw_1s 期权无 volume 列时由发球机 / FCS 注入 col 6。

    优先 monthly_iv 月度文件：它是训练特征 (bucketed_v7 → quote_features_raw)
    的真实输入；daily 文件可能被 databento 重建覆盖，分钟覆盖率与 volume 不同。
    """
    path = resolve_greek_monthly_path(sym, date_iso)
    if path is not None:
        df = pd.read_parquet(path)
        if not df.empty and "bucket_id" in df.columns:
            ts_day = _align_timestamp_series(df)
            df = df[ts_day.dt.strftime("%Y-%m-%d") == date_iso]
            if not df.empty:
                return _build_minute_ref(df)
    path = resolve_greek_day_path(sym, date_iso, greek_root)
    if path is None:
        return {}

    df = pd.read_parquet(path)
    if df.empty or "bucket_id" not in df.columns:
        return {}
    return _build_minute_ref(df)


def _build_minute_ref(df: pd.DataFrame) -> dict[tuple[int, int], dict[str, float]]:
    df = df.copy()
    df["minute_ts"] = (
        _align_timestamp_series(df)
        .dt.ceil("1min")
        .map(lambda t: int(pd.Timestamp(t).timestamp()))
    )
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        b_id = int(row.bucket_id)
        if not (0 <= b_id <= 5):
            continue
        key = (int(row.minute_ts), b_id)
        bid = float(getattr(row, "bid", 0.0) or 0.0)
        ask = float(getattr(row, "ask", 0.0) or 0.0)
        close = float(getattr(row, "close", 0.0) or 0.0)
        mid = close if close > 0.0 else (0.5 * (bid + ask) if bid > 0.0 and ask > 0.0 else 0.0)
        out[key] = {
            "iv": float(getattr(row, "iv", 0.0) or 0.0),
            "delta": float(getattr(row, "delta", 0.0) or 0.0),
            "gamma": float(getattr(row, "gamma", 0.0) or 0.0),
            "vega": float(getattr(row, "vega", 0.0) or 0.0),
            "theta": float(getattr(row, "theta", 0.0) or 0.0),
            "volume": float(getattr(row, "volume", 0.0) or 0.0),
            "spread_pct": float(getattr(row, "spread_pct", 0.0) or 0.0),
            "volume_imbalance": float(getattr(row, "volume_imbalance", 0.0) or 0.0),
            # 离线 day_iv / options_locked_feature：ceil('1min') 收盘盘口
            "bid": bid,
            "ask": ask,
            "bid_size": float(getattr(row, "bid_size", 0.0) or 0.0),
            "ask_size": float(getattr(row, "ask_size", 0.0) or 0.0),
            "close": close,
            "mid": mid,
        }
    return out


def inject_minute_volume_into_buckets(
    buckets: Any,
    minute_ts_unix: int,
    minute_ref: dict[tuple[int, int], dict[str, float]] | None,
) -> Any:
    """若 front 桶 col6 近零，从分钟参考表补成交量（对齐离线 1min VW）。"""
    import numpy as np

    if not minute_ref:
        return buckets
    arr = buckets if isinstance(buckets, np.ndarray) else np.asarray(buckets, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 6:
        return buckets
    if float(np.sum(np.maximum(arr[:4, 6], 0.0))) >= 1.0:
        return arr
    # 兼容结束标签(+60)与起点标签
    candidates = [int(minute_ts_unix), int(minute_ts_unix) + 60, int(minute_ts_unix) - 60]
    for b_id in range(min(4, arr.shape[0])):
        ref = None
        for mts in candidates:
            ref = minute_ref.get((mts, b_id))
            if ref:
                break
        if not ref:
            continue
        vol = float(ref.get("volume", 0.0) or 0.0)
        if vol > 0.0:
            arr[b_id, 6] = vol
    return arr


def _lookup_minute_ref(
    minute_ref: dict[tuple[int, int], dict[str, float]],
    minute_ts_unix: int,
    bucket_id: int,
    *,
    ffill_state: bool = True,
    max_ffill_minutes: int = 30,
) -> dict[str, float] | None:
    """按离线 options_locked_feature：盘口状态可前向填充，volume 仅当分钟存在。"""
    b = int(bucket_id)
    mts = int(minute_ts_unix)
    # 结束标签(+60)优先，再起点标签
    for key_ts in (mts + 60, mts, mts - 60):
        exact = minute_ref.get((key_ts, b))
        if exact is not None:
            return exact
    if not ffill_state:
        return None
    for lag in range(1, max_ffill_minutes + 1):
        for base in (mts + 60, mts):
            prev = minute_ref.get((base - lag * 60, b))
            if prev is None:
                continue
            # volume 缺失代表当分钟无成交，状态列沿用上一分钟
            return {
                "iv": prev.get("iv", 0.0),
                "delta": prev.get("delta", 0.0),
                "gamma": prev.get("gamma", 0.0),
                "vega": prev.get("vega", 0.0),
                "theta": prev.get("theta", 0.0),
                "spread_pct": prev.get("spread_pct", 0.0),
                "volume_imbalance": prev.get("volume_imbalance", 0.0),
                "volume": 0.0,
                "bid": prev.get("bid", 0.0),
                "ask": prev.get("ask", 0.0),
                "bid_size": prev.get("bid_size", 0.0),
                "ask_size": prev.get("ask_size", 0.0),
                "close": prev.get("close", 0.0),
                "mid": prev.get("mid", 0.0),
            }
    return None


def inject_minute_parquet_into_buckets(
    buckets: Any,
    minute_ts_unix: int,
    minute_ref: dict[tuple[int, int], dict[str, float]] | None,
    *,
    front_buckets: int = 4,
    zero_tail: bool = True,
    rewrite_microstructure: bool = False,
) -> Any:
    """
    分钟 commit 时用 IV parquet 覆写 Greeks/IV（诊断/greek-parity）。

    默认不改 bid/ask/size：实时流盘口必须来自原始秒级报价。
    仅当 rewrite_microstructure=True（离线对拍诊断）才按 spread_pct / imbalance 重建。
    volume 仅在 col6 近零时补分钟量（见 inject_minute_volume_into_buckets）。
    """
    import numpy as np

    if not minute_ref:
        return buckets
    arr = buckets if isinstance(buckets, np.ndarray) else np.asarray(buckets, dtype=np.float32)
    if arr.ndim != 2:
        return buckets
    arr = arr.copy()
    if arr.shape[0] < 6:
        pad = np.zeros((6 - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    if arr.shape[1] < 12:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)])

    mts = int(minute_ts_unix)
    for b_id in range(min(front_buckets, arr.shape[0])):
        ref = _lookup_minute_ref(minute_ref, mts, b_id)
        if not ref:
            continue
        for col, key in ((1, "delta"), (2, "gamma"), (3, "vega"), (4, "theta"), (7, "iv")):
            val = float(ref.get(key, 0.0) or 0.0)
            if np.isfinite(val):
                arr[b_id, col] = val
        vol = float(ref.get("volume", 0.0) or 0.0)
        if vol > 0.0 and float(arr[b_id, 6]) < 1e-6:
            arr[b_id, 6] = vol
        if rewrite_microstructure:
            # 优先写 day_iv 真实 bid/ask（离线对拍）；否则用 spread_pct 从 mid 反推
            bid = float(ref.get("bid", 0.0) or 0.0)
            ask = float(ref.get("ask", 0.0) or 0.0)
            mid_ref = float(ref.get("mid", 0.0) or ref.get("close", 0.0) or 0.0)
            if bid > 1e-6 and ask > 1e-6:
                arr[b_id, 8] = bid
                arr[b_id, 9] = ask
                if mid_ref > 1e-6:
                    arr[b_id, 0] = mid_ref
                else:
                    arr[b_id, 0] = 0.5 * (bid + ask)
            else:
                mid = float(arr[b_id, 0])
                sp = float(ref.get("spread_pct", 0.0) or 0.0)
                if mid > 1e-6 and np.isfinite(sp) and sp >= 0.0:
                    half = mid * sp * 0.5
                    arr[b_id, 8] = max(mid - half, 1e-6)
                    arr[b_id, 9] = max(mid + half, arr[b_id, 8])
            bs = float(ref.get("bid_size", 0.0) or 0.0)
            asz = float(ref.get("ask_size", 0.0) or 0.0)
            if bs > 0.0:
                arr[b_id, 10] = bs
            if asz > 0.0:
                arr[b_id, 11] = asz
            if float(arr[b_id, 10]) < 1e-6 and float(arr[b_id, 11]) < 1e-6:
                imb = ref.get("volume_imbalance")
                if imb is not None and np.isfinite(float(imb)):
                    imb_f = float(np.clip(imb, -0.99, 0.99))
                    base = 1000.0
                    arr[b_id, 10] = base * (1.0 + imb_f)
                    arr[b_id, 11] = base * (1.0 - imb_f)

    if zero_tail and arr.shape[0] > front_buckets:
        arr[front_buckets:, :] = 0.0
    return arr


def ensure_minute_option_ref(
    sym: str,
    date_iso: str,
    cache: dict | None,
    *,
    greek_root: Path | None = None,
) -> tuple[dict[tuple[int, int], dict[str, float]], dict]:
    """加载或复用分钟 IV 参考表。"""
    cache = cache if cache is not None else {}
    if cache.get("minute_option_ref_day") == date_iso and cache.get("minute_option_ref"):
        return cache["minute_option_ref"], cache
    ref = load_minute_option_ref(sym, date_iso, greek_root=greek_root)
    cache["minute_option_ref"] = ref
    cache["minute_option_ref_day"] = date_iso
    return ref, cache
