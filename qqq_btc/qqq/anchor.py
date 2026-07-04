#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QQQ 0DTE/1DTE 合约锚点 —— 从 New_Pro/preprocess/anchor_contract_utils.py 内化,
qqq_btc 不再依赖 New_Pro。选约逻辑逐行保留(已验证正确),清理两处耦合:
  - 默认配置路径改为 qqq_btc/CONFIG/anchor_qqq_0dte.json(与 New_Pro 同内容)
  - BUCKET_SPECS 本地定义,不再 `from config import`(那是 legacy 运行时全局)

离线锁定(get_daily_locked_contracts)、标签报价加载(load_bucket_minute_quotes)、
实盘选约(select_front_expiration)共用同一份配置,保证三处永远一致。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "CONFIG" / "anchor_qqq_0dte.json"
ANCHOR_CONFIG_PATH = _DEFAULT_CONFIG_PATH

# bucket 定义:(bucket_id, is_front, is_call, target_|delta|)
_LEGACY_6_BUCKET_TARGETS = [
    (0, True, False, 0.50),
    (1, True, False, 0.25),
    (2, True, True, 0.50),
    (3, True, True, 0.25),
    (4, False, False, 0.50),
    (5, False, True, 0.50),
]

_FRONT_4_BUCKET_TARGETS = [
    (0, True, False, 0.50),
    (1, True, False, 0.25),
    (2, True, True, 0.50),
    (3, True, True, 0.25),
]

# 实盘订阅 tag 定义(原 legacy config.BUCKET_SPECS,现为本模块常量)
BUCKET_SPECS: Dict[str, Dict[str, Any]] = {
    "PUT_ATM": {"delta": -0.50, "bucket_idx": 0},
    "PUT_OTM": {"delta": -0.25, "bucket_idx": 1},
    "CALL_ATM": {"delta": 0.50, "bucket_idx": 2},
    "CALL_OTM": {"delta": 0.25, "bucket_idx": 3},
    "NEXT_PUT_ATM": {"delta": -0.50, "bucket_idx": 4},
    "NEXT_CALL_ATM": {"delta": 0.50, "bucket_idx": 5},
}


def _expand_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).expanduser()


def load_anchor_config(path: Optional[Path] = None) -> dict:
    cfg_path = Path(path) if path else Path(
        os.environ.get("ANCHOR_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
    )
    if not cfg_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        alt = repo_root / cfg_path
        if alt.exists():
            cfg_path = alt
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(cfg_path)
    paths = cfg.get("paths") or {}
    cfg["_paths_resolved"] = {k: _expand_path(v) for k, v in paths.items()}
    return cfg


def load_qqq_anchor_config() -> dict:
    """加载本路径默认的 QQQ 0DTE 锚点配置。"""
    return load_anchor_config(_DEFAULT_CONFIG_PATH)


def select_front_dte(available_dtes: Sequence[int], cfg: dict) -> Optional[int]:
    allowed = set(cfg.get("front_allowed_dte") or [0, 1, 2])
    prefer = int(cfg.get("front_prefer_dte", 0))
    dte_min = int(cfg.get("front_min_dte", 0))
    dte_max = int(cfg.get("front_max_dte", 2))

    candidates = sorted({int(d) for d in available_dtes if dte_min <= int(d) <= dte_max and int(d) in allowed})
    if not candidates:
        fallbacks = sorted({int(d) for d in available_dtes if int(d) >= dte_min})
        if not fallbacks:
            return None
        return fallbacks[0]

    if prefer in candidates:
        return prefer
    return min(candidates, key=lambda d: (abs(d - prefer), d))


def bucket_targets(cfg: dict) -> List[Tuple[int, bool, bool, float]]:
    if cfg.get("use_next_buckets", False):
        return list(_LEGACY_6_BUCKET_TARGETS)
    return list(_FRONT_4_BUCKET_TARGETS)


def get_daily_locked_contracts(df: pd.DataFrame, cfg: dict) -> Optional[pd.DataFrame]:
    """按日锁定 front expiry 上的 4/6 个 bucket 合约。"""
    work = df.copy()
    work["date_str"] = work["timestamp"].dt.date.astype(str)
    work["abs_delta"] = work["delta"].abs()

    dte_min = int(cfg.get("front_min_dte", 0))
    dte_max = max(int(cfg.get("front_max_dte", 2)), 90 if cfg.get("use_next_buckets") else 2)
    candidates = work[(work["dte"] >= dte_min) & (work["dte"] <= dte_max)].copy()
    if candidates.empty:
        return None

    locked_map = []
    targets = bucket_targets(cfg)
    delta_tol = float(cfg.get("delta_tolerance", 0.15))
    # 锁约只允许用开盘窗口快照:与实盘一致(09:30 锁定后不换),且杜绝
    # "用盘中晚些时候的 delta 选开盘合约"的前视。旧实现用全天快照,
    # 2026 年日内波幅变大后锁出的"ATM"距开盘价中位偏离 0.85%+(2025H2
    # 约 0.25%),深 ITM/OTM 合约报价稀疏,是 bucket 覆盖率劣化的主因。
    lock_minutes = int(cfg.get("lock_window_minutes", 10))

    for date_val, daily_group in candidates.groupby("date_str"):
        day_start = daily_group["timestamp"].min()
        open_cut = day_start + pd.Timedelta(minutes=lock_minutes)
        open_group = daily_group[daily_group["timestamp"] <= open_cut]
        if not open_group.empty:
            daily_group = open_group
        available_dtes = daily_group["dte"].unique()
        selected_front_dte = select_front_dte(available_dtes, cfg)
        if selected_front_dte is None:
            continue

        selected_next_dte = selected_front_dte
        if cfg.get("use_next_buckets", False):
            next_target = selected_front_dte + 28
            min_next = selected_front_dte + 20
            max_next = selected_front_dte + 50
            next_options = [d for d in available_dtes if min_next <= d <= max_next]
            if next_options:
                selected_next_dte = min(next_options, key=lambda x: abs(x - next_target))
            else:
                fallbacks = [d for d in available_dtes if d > selected_front_dte + 15]
                if fallbacks:
                    selected_next_dte = min(fallbacks)

        for b_id, is_front, is_call, target_delta in targets:
            target_dte = selected_front_dte if is_front else selected_next_dte
            type_str = "Call" if is_call else "Put"
            mask = (daily_group["dte"] == target_dte) & (
                daily_group["contract_type"].astype(str).str.upper().str.startswith(type_str[0])
            )
            subset = daily_group[mask].copy()
            if subset.empty:
                continue

            subset["delta_dist"] = (subset["abs_delta"] - target_delta).abs()
            delta_candidates = subset[subset["delta_dist"] < delta_tol]
            if delta_candidates.empty:
                best_ticker = subset.sort_values("delta_dist").iloc[0]["contract_symbol"]
            else:
                best_ticker = delta_candidates.sort_values("delta_dist").iloc[0]["contract_symbol"]

            locked_map.append(
                {
                    "date_str": date_val,
                    "contract_symbol": best_ticker,
                    "bucket_id": b_id,
                    "front_dte": int(selected_front_dte),
                }
            )

    if not locked_map:
        return None
    return pd.DataFrame(locked_map)


def select_front_expiration(exp_dtes: Sequence[Tuple[str, int]], cfg: dict) -> Optional[Tuple[str, int]]:
    """实盘/回测:从 (expiry_str, dte) 列表选 front expiry。"""
    dte_min = int(cfg.get("front_min_dte", 0))
    dte_max = int(cfg.get("front_max_dte", 2))
    allowed = set(cfg.get("front_allowed_dte") or [0, 1, 2])

    valid = [(s, d) for s, d in exp_dtes if d >= dte_min and d in allowed and dte_min <= d <= dte_max]
    if not valid:
        valid = [(s, d) for s, d in exp_dtes if d >= dte_min]
    if not valid:
        return None

    prefer = int(cfg.get("front_prefer_dte", 0))
    prefer_hits = [x for x in valid if x[1] == prefer]
    if prefer_hits:
        return prefer_hits[0]
    return min(valid, key=lambda x: (abs(x[1] - prefer), x[1]))


def active_bucket_specs(cfg: dict) -> Dict[str, Dict[str, Any]]:
    """当前 profile 应订阅的 bucket tag 定义。"""
    specs = deepcopy(BUCKET_SPECS)
    if not cfg.get("use_next_buckets", False):
        specs = {k: v for k, v in specs.items() if "NEXT" not in k}
    return specs


def load_bucket_minute_quotes(
    symbol: str,
    timestamps: pd.Series,
    bucket_id: int,
    anchor_cfg: dict,
    prefix: str = "exec_call",
) -> pd.DataFrame:
    """
    从日级 sniper / day_iv parquet 加载指定 bucket 的 1min 报价序列。
    返回列: timestamp, {prefix}_mid, {prefix}_bid, {prefix}_ask, {prefix}_spread_pct
    """
    if timestamps.empty:
        return pd.DataFrame()

    paths = anchor_cfg.get("_paths_resolved") or {}
    roots = [
        paths.get("day_iv_dir"),
        paths.get("sniper_option_dir"),
        paths.get("raw_iv_dir"),
    ]

    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        ts = ts.dt.tz_convert("America/New_York")

    date_range = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D")
    frames = []
    for day in date_range:
        day_str = day.strftime("%Y-%m-%d")
        fp = None
        for root in roots:
            if root is None:
                continue
            for sub in ("standard", ""):
                candidate = (
                    root / symbol / sub / f"{symbol}_{day_str}.parquet"
                    if sub
                    else root / symbol / f"{symbol}_{day_str}.parquet"
                )
                if candidate.exists():
                    fp = candidate
                    break
            if fp:
                break
        if fp is None:
            continue

        day_df = pd.read_parquet(fp)
        if day_df.empty or "bucket_id" not in day_df.columns:
            continue
        day_df = day_df[day_df["bucket_id"] == bucket_id].copy()
        if day_df.empty:
            continue

        if not pd.api.types.is_datetime64_any_dtype(day_df["timestamp"]):
            day_df["timestamp"] = pd.to_datetime(day_df["timestamp"])
        if day_df["timestamp"].dt.tz is None:
            day_df["timestamp"] = day_df["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
        else:
            day_df["timestamp"] = day_df["timestamp"].dt.tz_convert("America/New_York")

        if "close" in day_df.columns:
            mid = pd.to_numeric(day_df["close"], errors="coerce")
        elif {"bid", "ask"}.issubset(day_df.columns):
            mid = (pd.to_numeric(day_df["bid"], errors="coerce") + pd.to_numeric(day_df["ask"], errors="coerce")) / 2.0
        else:
            continue

        out = pd.DataFrame(
            {
                "timestamp": day_df["timestamp"],
                f"{prefix}_mid": mid,
                f"{prefix}_bid": pd.to_numeric(day_df.get("bid", 0), errors="coerce"),
                f"{prefix}_ask": pd.to_numeric(day_df.get("ask", 0), errors="coerce"),
            }
        )
        if "spread_pct" in day_df.columns:
            out[f"{prefix}_spread_pct"] = pd.to_numeric(day_df["spread_pct"], errors="coerce")
        else:
            spread = out[f"{prefix}_ask"] - out[f"{prefix}_bid"]
            out[f"{prefix}_spread_pct"] = (spread / out[f"{prefix}_mid"].replace(0, pd.NA)).fillna(0.0)
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return quotes


def _normalize_ny_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York")
    return out


# 报价回看容差:断档超过该时长的分钟置 NaN(标签侧 label_*_valid 置无效,
# 回放侧该腿不开仓)。2026 年 bucket0 覆盖劣化,无容差会拿几十分钟前的
# 陈旧价成交,伪造 PUT 腿收益。
QUOTE_ASOF_TOLERANCE = "5min"


def merge_exec_bucket_quotes(
    df: pd.DataFrame, symbol: str, anchor_cfg: dict, tolerance: Optional[str] = QUOTE_ASOF_TOLERANCE
) -> pd.DataFrame:
    """将锁定 bucket 的分钟报价 merge_asof 进特征表。"""
    bucket_id = int(anchor_cfg.get("label_trade_bucket_id", 2))
    quotes = load_bucket_minute_quotes(symbol, df["timestamp"], bucket_id, anchor_cfg)
    if quotes.empty:
        return df

    merged = _normalize_ny_timestamp(df).sort_values("timestamp")
    quotes = quotes.sort_values("timestamp")
    merged = pd.merge_asof(
        merged,
        quotes,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
    )
    return merged


# 双腿标的 bucket:CALL ATM=2 / PUT ATM=0(见 _FRONT_4_BUCKET_TARGETS)
DUAL_LEG_BUCKETS: Dict[str, int] = {"exec_call": 2, "exec_put": 0}


def merge_dual_leg_exec_quotes(
    df: pd.DataFrame, symbol: str, anchor_cfg: dict, tolerance: Optional[str] = QUOTE_ASOF_TOLERANCE
) -> pd.DataFrame:
    """
    同时 merge CALL ATM(exec_call_*)与 PUT ATM(exec_put_*)两条腿的分钟报价。
    双向策略的标签构建与回放都需要两条腿各自的 bid/ask —— 买 PUT 不是
    "负的 CALL 收益",PUT 有自己的权利金、点差与 IV 路径。
    """
    legs = anchor_cfg.get("dual_leg_buckets") or DUAL_LEG_BUCKETS
    merged = _normalize_ny_timestamp(df).sort_values("timestamp")
    tol = pd.Timedelta(tolerance) if tolerance else None
    for prefix, bucket_id in legs.items():
        quotes = load_bucket_minute_quotes(
            symbol, merged["timestamp"], int(bucket_id), anchor_cfg, prefix=prefix
        )
        if quotes.empty:
            continue
        merged = pd.merge_asof(
            merged,
            quotes.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=tol,
        )
    return merged
