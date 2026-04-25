#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


_bootstrap_imports()
from config import PG_DB_URL  # noqa: E402
from strategy_selector import StrategyConfig  # noqa: E402

try:  # noqa: E402
    from execution_engine_v8 import compute_entry_priority_score, compute_entry_trend_quality, reserve_priority_entry_slots  # type: ignore
except Exception:  # pragma: no cover
    def compute_entry_priority_score(
        *,
        alpha: float,
        iv: float,
        roc_5m: float,
        snap_roc: float,
        macd_hist: float,
        entry_dir: int,
        cfg,
        pure_alpha_replay: bool = False,
        trend_net: float = 0.0,
        trend_efficiency: float = 0.0,
        trend_r2: float = 0.0,
        trend_observations: int = 0,
    ) -> dict[str, float]:
        alpha_abs = abs(float(alpha or 0.0))
        effective_iv = max(0.1, float(iv or 0.0))
        direction = 1 if int(entry_dir or 0) >= 0 else -1

        alpha_power = float(getattr(cfg, "ENTRY_RANK_ALPHA_POWER", 1.35))
        iv_penalty_power = float(getattr(cfg, "ENTRY_RANK_IV_PENALTY_POWER", 0.0))
        base_alpha = alpha_abs ** alpha_power
        base_score = base_alpha if pure_alpha_replay else base_alpha / (effective_iv ** iv_penalty_power)
        high_alpha_bonus = min(
            max(0.0, alpha_abs - float(getattr(cfg, "ENTRY_RANK_HIGH_ALPHA_FLOOR", 1.20)))
            * float(getattr(cfg, "ENTRY_RANK_HIGH_ALPHA_BONUS_SCALE", 0.35)),
            float(getattr(cfg, "ENTRY_RANK_HIGH_ALPHA_MAX_BONUS", 0.50)),
        )
        alpha_mult = 1.0 + high_alpha_bonus
        abs_roc_mult = 1.0 + abs(float(roc_5m or 0.0)) * float(
            getattr(cfg, "ENTRY_RANK_ROC_ABS_SCALE", 100.0) or 100.0
        )
        stock_bonus = min(
            max(0.0, float(roc_5m or 0.0) * direction)
            * float(getattr(cfg, "ENTRY_RANK_STOCK_ROC_SCALE", 120.0) or 120.0),
            float(getattr(cfg, "ENTRY_RANK_STOCK_ROC_MAX_BONUS", 0.35) or 0.35),
        )
        snap_bonus = min(
            max(0.0, float(snap_roc or 0.0) * direction)
            * float(getattr(cfg, "ENTRY_RANK_SNAP_ROC_SCALE", 200.0) or 200.0),
            float(getattr(cfg, "ENTRY_RANK_SNAP_ROC_MAX_BONUS", 0.30) or 0.30),
        )
        macd_bonus = min(
            max(0.0, float(macd_hist or 0.0) * direction)
            * float(getattr(cfg, "ENTRY_RANK_MACD_SCALE", 8.0) or 8.0),
            float(getattr(cfg, "ENTRY_RANK_MACD_MAX_BONUS", 0.30) or 0.30),
        )
        stock_ok = float(roc_5m or 0.0) * direction >= float(
            getattr(cfg, "ENTRY_PRIORITY_STOCK_ROC_FLOOR", 0.0002) or 0.0002
        )
        snap_ok = float(snap_roc or 0.0) * direction >= float(
            getattr(cfg, "ENTRY_PRIORITY_SNAP_ROC_FLOOR", 0.0) or 0.0
        )
        macd_ok = float(macd_hist or 0.0) * direction >= float(
            getattr(cfg, "ENTRY_PRIORITY_MACD_FLOOR", 0.01) or 0.01
        )
        confirmation_count = int(stock_ok) + int(snap_ok) + int(macd_ok)
        is_priority_candidate = (
            alpha_abs >= float(getattr(cfg, "ENTRY_PRIORITY_ALPHA_FLOOR", 0.9) or 0.9)
            and confirmation_count >= int(getattr(cfg, "ENTRY_PRIORITY_MIN_CONFIRMATIONS", 2) or 2)
        )
        priority_mult = 1.0
        if is_priority_candidate:
            priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_BOOST", 0.80) or 0.80)
            if stock_ok:
                priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_STOCK_BONUS", 0.25) or 0.25)
            if snap_ok:
                priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_SNAP_BONUS", 0.15) or 0.15)
            if macd_ok:
                priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_MACD_BONUS", 0.20) or 0.20)
        trend_mult = 1.0
        if bool(getattr(cfg, "ENTRY_RANK_TREND_QUALITY_ENABLED", True)):
            min_obs = int(getattr(cfg, "ENTRY_RANK_TREND_MIN_OBS", 16))
            if int(trend_observations or 0) >= min_obs:
                net_target = max(1e-6, float(getattr(cfg, "ENTRY_RANK_TREND_NET_TARGET", 0.012)))
                net_score = min(1.0, max(0.0, float(trend_net or 0.0)) / net_target)
                eff_score = min(1.0, max(0.0, float(trend_efficiency or 0.0)))
                r2_score = min(1.0, max(0.0, float(trend_r2 or 0.0)))
                trend_quality = 0.45 * net_score + 0.30 * eff_score + 0.25 * r2_score
                floor = max(1e-6, float(getattr(cfg, "ENTRY_RANK_TREND_QUALITY_FLOOR", 0.25)))
                trend_mult += float(getattr(cfg, "ENTRY_RANK_TREND_QUALITY_BOOST", 0.12)) * trend_quality
                if trend_quality < floor:
                    trend_mult -= float(getattr(cfg, "ENTRY_RANK_TREND_QUALITY_PENALTY", 0.25)) * ((floor - trend_quality) / floor)
                trend_mult = min(
                    float(getattr(cfg, "ENTRY_RANK_TREND_MAX_MULT", 1.12)),
                    max(float(getattr(cfg, "ENTRY_RANK_TREND_MIN_MULT", 0.75)), trend_mult),
                )
        score = base_score * alpha_mult * abs_roc_mult * (1.0 + stock_bonus) * (1.0 + snap_bonus) * (1.0 + macd_bonus) * priority_mult * trend_mult
        return {
            "score": float(score),
            "is_priority_candidate": float(1.0 if is_priority_candidate else 0.0),
        }

    def compute_entry_trend_quality(prices: Any, entry_dir: int, window_mins: int = 30) -> dict[str, float]:
        try:
            price_list = [float(p) for p in list(prices)[-int(window_mins + 1):] if float(p) > 0]
        except Exception:
            price_list = []
        if len(price_list) < 2:
            return {
                "trend_net": 0.0,
                "trend_efficiency": 0.0,
                "trend_r2": 0.0,
                "trend_observations": float(len(price_list)),
            }
        direction = 1 if int(entry_dir or 0) >= 0 else -1
        returns = [
            (price_list[i] - price_list[i - 1]) / price_list[i - 1]
            for i in range(1, len(price_list))
            if price_list[i - 1] > 0
        ]
        raw_net = (price_list[-1] - price_list[0]) / price_list[0] if price_list[0] > 0 else 0.0
        trend_net = raw_net * direction
        path = sum(abs(x) for x in returns)
        trend_efficiency = max(0.0, trend_net) / path if path > 0 else 0.0
        n = len(price_list)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(price_list) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        syy = sum((y - mean_y) ** 2 for y in price_list)
        if sxx > 0 and syy > 0:
            sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, price_list))
            trend_r2 = max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))
        else:
            trend_r2 = 0.0
        return {
            "trend_net": float(trend_net),
            "trend_efficiency": float(trend_efficiency),
            "trend_r2": float(trend_r2),
            "trend_observations": float(len(price_list)),
        }

    def reserve_priority_entry_slots(entry_candidates: list[dict[str, Any]], allowed_entries: int, cfg) -> list[dict[str, Any]]:
        if allowed_entries <= 0 or not entry_candidates:
            return []
        selected = list(entry_candidates[:allowed_entries])
        reserved_slots = max(0, int(getattr(cfg, "ENTRY_PRIORITY_RESERVED_SLOTS", 1) or 0))
        if reserved_slots <= 0:
            return selected

        def _is_priority_candidate(cand: dict[str, Any]) -> bool:
            return bool(cand.get("is_priority_candidate", False))

        priority_pool = [cand for cand in entry_candidates if _is_priority_candidate(cand)]
        if not priority_pool:
            return selected
        selected_priority = sum(1 for cand in selected if _is_priority_candidate(cand))
        if selected_priority >= min(reserved_slots, allowed_entries):
            return selected
        missing = min(reserved_slots, allowed_entries) - selected_priority
        selected_keys = {(cand.get("sym"), cand.get("batch_idx")) for cand in selected}
        replacements = [cand for cand in priority_pool if (cand.get("sym"), cand.get("batch_idx")) not in selected_keys][:missing]
        replaceable_idx = [idx for idx in range(len(selected) - 1, -1, -1) if not _is_priority_candidate(selected[idx])]
        for cand, idx in zip(replacements, replaceable_idx):
            selected[idx] = cand
        return sorted(selected, key=lambda x: float(x.get("alpha_strength", 0.0) or 0.0), reverse=True)


def legacy_entry_priority_score(
    *,
    alpha: float,
    iv: float,
    roc_5m: float,
    snap_roc: float,
    macd_hist: float,
    entry_dir: int,
    cfg,
    pure_alpha_replay: bool = False,
    **_: Any,
) -> dict[str, float]:
    alpha_abs = abs(float(alpha or 0.0))
    effective_iv = max(0.1, float(iv or 0.0))
    direction = 1 if int(entry_dir or 0) >= 0 else -1
    base_score = alpha_abs if pure_alpha_replay else alpha_abs / effective_iv
    abs_roc_mult = 1.0 + abs(float(roc_5m or 0.0)) * float(
        getattr(cfg, "ENTRY_RANK_ROC_ABS_SCALE", 100.0) or 100.0
    )
    stock_bonus = min(
        max(0.0, float(roc_5m or 0.0) * direction)
        * float(getattr(cfg, "ENTRY_RANK_STOCK_ROC_SCALE", 120.0) or 120.0),
        float(getattr(cfg, "ENTRY_RANK_STOCK_ROC_MAX_BONUS", 0.35) or 0.35),
    )
    snap_bonus = min(
        max(0.0, float(snap_roc or 0.0) * direction)
        * float(getattr(cfg, "ENTRY_RANK_SNAP_ROC_SCALE", 200.0) or 200.0),
        float(getattr(cfg, "ENTRY_RANK_SNAP_ROC_MAX_BONUS", 0.30) or 0.30),
    )
    macd_bonus = min(
        max(0.0, float(macd_hist or 0.0) * direction)
        * float(getattr(cfg, "ENTRY_RANK_MACD_SCALE", 8.0) or 8.0),
        float(getattr(cfg, "ENTRY_RANK_MACD_MAX_BONUS", 0.30) or 0.30),
    )
    stock_ok = float(roc_5m or 0.0) * direction >= float(
        getattr(cfg, "ENTRY_PRIORITY_STOCK_ROC_FLOOR", 0.0002) or 0.0002
    )
    snap_ok = float(snap_roc or 0.0) * direction >= float(
        getattr(cfg, "ENTRY_PRIORITY_SNAP_ROC_FLOOR", 0.0) or 0.0
    )
    macd_ok = float(macd_hist or 0.0) * direction >= float(
        getattr(cfg, "ENTRY_PRIORITY_MACD_FLOOR", 0.01) or 0.01
    )
    confirmation_count = int(stock_ok) + int(snap_ok) + int(macd_ok)
    is_priority_candidate = (
        alpha_abs >= float(getattr(cfg, "ENTRY_PRIORITY_ALPHA_FLOOR", 0.9) or 0.9)
        and confirmation_count >= int(getattr(cfg, "ENTRY_PRIORITY_MIN_CONFIRMATIONS", 2) or 2)
    )
    priority_mult = 1.0
    if is_priority_candidate:
        priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_BOOST", 0.80) or 0.80)
        if stock_ok:
            priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_STOCK_BONUS", 0.25) or 0.25)
        if snap_ok:
            priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_SNAP_BONUS", 0.15) or 0.15)
        if macd_ok:
            priority_mult += float(getattr(cfg, "ENTRY_PRIORITY_MACD_BONUS", 0.20) or 0.20)
    score = base_score * abs_roc_mult * (1.0 + stock_bonus) * (1.0 + snap_bonus) * (1.0 + macd_bonus) * priority_mult
    return {
        "score": float(score),
        "is_priority_candidate": float(1.0 if is_priority_candidate else 0.0),
    }


CONTRACT_RE = __import__("re").compile(r"(\d{6})([CP])(\d{8})")


@dataclass
class RankedCandidate:
    ts: int
    datetime_ny: str
    symbol: str
    alpha: float
    iv: float
    price: float
    roc_5m: float
    snap_roc: float
    macd_hist: float
    score: float
    is_priority_candidate: bool


@dataclass
class CandidateProfit:
    symbol: str
    option_side: str
    datetime_ny: str
    entry_ts: int
    entry_mid: float
    entry_alpha: float
    score: float
    max_roi: float
    final_roi: float
    min_roi: float
    best_ts: int
    sample_points: int


@dataclass
class OpenLot:
    symbol: str
    qty: float
    entry_ts: float
    entry_dt: str
    entry_price: float
    contract_id: str
    option_side: str
    strike: float | None
    entry_reason: str


def _lot_key(symbol: str, contract_id: str, option_side: str) -> tuple[str, str, str]:
    contract_key = (contract_id or "").replace(" ", "")
    if contract_key:
        return (symbol, "contract", contract_key)
    return (symbol, "side", option_side)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复盘全天 alpha 排序与单笔持仓利润")
    parser.add_argument("--date", required=True, help="交易日，格式 2026-04-24")
    parser.add_argument("--focus-symbols", default="", help="逗号分隔，关注的标的，例如 NVDA,MSFT,ADBE")
    parser.add_argument("--selected-per-frame", type=int, default=3, help="每分钟理论选出的候选数")
    parser.add_argument("--candidate-horizon-mins", type=int, default=30, help="理论候选向后观察的分钟数")
    parser.add_argument("--candidate-cooldown-mins", type=int, default=15, help="同一标的同方向理论候选去重冷却分钟数")
    parser.add_argument("--csv-out", default="", help="可选，输出单笔利润明细 CSV")
    parser.add_argument("--limit", type=int, default=12, help="终端输出明细条数")
    return parser.parse_args()


class MetricState:
    def __init__(self) -> None:
        self.prices: deque[float] = deque(maxlen=6)
        self.trend_prices: deque[float] = deque(maxlen=61)
        self.prev_price: float | None = None
        self.cached_min_roc: float = 0.0
        self.ema_fast_val: float | None = None
        self.ema_slow_val: float | None = None
        self.dea_val: float = 0.0
        self.prev_macd_hist: float = 0.0
        self.k_fast = 2 / (8 + 1)
        self.k_slow = 2 / (21 + 1)
        self.k_sig = 2 / (5 + 1)

    def update(self, price: float) -> tuple[float, float, float]:
        price = float(price or 0.0)
        if self.prev_price and self.prev_price > 0:
            self.cached_min_roc = (price - self.prev_price) / self.prev_price
        else:
            self.cached_min_roc = 0.0

        if self.ema_fast_val is None:
            self.ema_fast_val = price
            self.ema_slow_val = price
            self.dea_val = 0.0
        else:
            self.ema_fast_val = float(price * self.k_fast + self.ema_fast_val * (1 - self.k_fast))
            self.ema_slow_val = float(price * self.k_slow + self.ema_slow_val * (1 - self.k_slow))
            dif = self.ema_fast_val - self.ema_slow_val
            self.dea_val = float(dif * self.k_sig + self.dea_val * (1 - self.k_sig))

        macd_hist = float((self.ema_fast_val - self.ema_slow_val) - self.dea_val)
        macd_hist_slope = macd_hist - self.prev_macd_hist
        self.prev_macd_hist = macd_hist

        self.prices.append(price)
        self.trend_prices.append(price)
        self.prev_price = price

        roc_5m = 0.0
        if len(self.prices) >= 6:
            prev_5m = self.prices[0]
            if prev_5m > 0:
                roc_5m = (price - prev_5m) / prev_5m

        return roc_5m, macd_hist, self.cached_min_roc


def _fetch_rows(conn, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def load_alpha_rows(conn, date_str: str) -> list[dict[str, Any]]:
    query = """
        SELECT ts, datetime_ny, symbol, alpha, iv, price
        FROM alpha_logs
        WHERE datetime_ny LIKE %s
        ORDER BY ts ASC, symbol ASC
    """
    return _fetch_rows(conn, query, (f"{date_str}%",))


def load_order_rows(conn, date_str: str, symbols: list[str]) -> list[dict[str, Any]]:
    params: list[Any] = [f"{date_str}%"]
    where_symbols = ""
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    query = f"""
        SELECT ts, datetime_ny, symbol, action, qty, price, details_json
        FROM order_events
        WHERE datetime_ny LIKE %s
          AND action IN ('ORDER_PENDING', 'ORDER_FILLED')
          {where_symbols}
        ORDER BY ts ASC
    """
    rows = _fetch_rows(conn, query, tuple(params))
    for row in rows:
        try:
            row["details"] = json.loads(row.get("details_json") or "{}")
        except Exception:
            row["details"] = {}
        note = row["details"].get("strategy_note")
        if isinstance(note, str):
            try:
                row["details"]["strategy_note"] = json.loads(note)
            except Exception:
                row["details"]["strategy_note"] = {"reason": note}
    return rows


def load_option_paths(conn, min_ts: int, max_ts: int, symbols: list[str]) -> dict[str, dict[int, str]]:
    params: list[Any] = [min_ts, max_ts]
    where_symbols = ""
    if symbols:
        where_symbols = " AND symbol = ANY(%s)"
        params.append(symbols)
    query = f"""
        SELECT symbol, ts, buckets_json
        FROM option_snapshots_1m
        WHERE ts >= %s AND ts <= %s
          {where_symbols}
        ORDER BY symbol ASC, ts ASC
    """
    rows = _fetch_rows(conn, query, tuple(params))
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        raw_buckets = row["buckets_json"]
        if isinstance(raw_buckets, str):
            buckets_json = raw_buckets
        else:
            buckets_json = json.dumps(raw_buckets)
        out[str(row["symbol"])][int(row["ts"])] = buckets_json
    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _minute_floor(ts: float) -> int:
    return int(float(ts) // 60 * 60)


def _parse_contract(contract_id: str) -> tuple[str, float | None]:
    if not contract_id:
        return "", None
    m = CONTRACT_RE.search(contract_id.replace(" ", ""))
    if not m:
        return "", None
    return ("CALL" if m.group(2) == "C" else "PUT"), int(m.group(3)) / 1000.0


def _mid_from_bucket(bucket: list[Any]) -> float | None:
    bid = _safe_float(bucket[8] if len(bucket) > 8 else 0.0)
    ask = _safe_float(bucket[9] if len(bucket) > 9 else 0.0)
    px = _safe_float(bucket[0] if bucket else 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if px > 0:
        return px
    return None


def _pick_bucket_mid(buckets_json: str, option_side: str, strike: float | None, ref_price: float | None) -> float | None:
    try:
        buckets = json.loads(buckets_json or "{}").get("buckets", [])
    except Exception:
        return None
    side_sign = 1 if option_side == "CALL" else -1
    choices: list[tuple[float, float, float]] = []
    for bucket in buckets:
        if not isinstance(bucket, list) or len(bucket) < 10:
            continue
        delta = _safe_float(bucket[1] if len(bucket) > 1 else 0.0)
        if delta == 0 or (delta > 0) != (side_sign > 0):
            continue
        mid = _mid_from_bucket(bucket)
        if mid is None:
            continue
        bucket_strike = _safe_float(bucket[5] if len(bucket) > 5 else 0.0)
        strike_gap = abs(bucket_strike - float(strike or bucket_strike))
        price_gap = abs(mid - float(ref_price or mid))
        choices.append((strike_gap, price_gap, mid))
    if not choices:
        fallback_idx = 2 if option_side == "CALL" else 0
        if len(buckets) > fallback_idx and isinstance(buckets[fallback_idx], list):
            return _mid_from_bucket(buckets[fallback_idx])
        return None
    choices.sort(key=lambda x: (x[0], x[1]))
    return choices[0][2]


def _pick_atm_mid(buckets_json: str, option_side: str) -> float | None:
    try:
        buckets = json.loads(buckets_json or "{}").get("buckets", [])
    except Exception:
        return None
    fallback_idx = 2 if option_side == "CALL" else 0
    if len(buckets) > fallback_idx and isinstance(buckets[fallback_idx], list):
        return _mid_from_bucket(buckets[fallback_idx])
    return _pick_bucket_mid(buckets_json, option_side, strike=None, ref_price=None)


def _lookup_snapshot_near(option_series: dict[int, str], target_ts: int, max_gap_sec: int = 120) -> str:
    if target_ts in option_series:
        return option_series[target_ts]
    if not option_series:
        return ""
    nearest_ts = min(option_series.keys(), key=lambda ts: abs(int(ts) - int(target_ts)))
    if abs(int(nearest_ts) - int(target_ts)) > int(max_gap_sec):
        return ""
    return option_series.get(nearest_ts, "")


def analyze_candidate_profits(
    selected_rows: list[RankedCandidate],
    option_paths: dict[str, dict[int, str]],
    horizon_mins: int,
    cooldown_mins: int,
) -> list[CandidateProfit]:
    analyses: list[CandidateProfit] = []
    last_selected_ts: dict[tuple[str, str], int] = {}
    cooldown_sec = max(0, int(cooldown_mins)) * 60
    horizon_sec = max(1, int(horizon_mins)) * 60

    for cand in sorted(selected_rows, key=lambda x: (x.ts, -x.score)):
        option_side = "CALL" if cand.alpha >= 0 else "PUT"
        key = (cand.symbol, option_side)
        if cooldown_sec and cand.ts - last_selected_ts.get(key, -10**12) < cooldown_sec:
            continue

        option_series = option_paths.get(cand.symbol, {})
        entry_snapshot = _lookup_snapshot_near(option_series, cand.ts)
        entry_mid = _pick_atm_mid(entry_snapshot, option_side)
        if entry_mid is None or entry_mid <= 0:
            continue

        roi_path: list[tuple[int, float]] = []
        end_ts = cand.ts + horizon_sec
        for ts in sorted(option_series.keys()):
            if ts <= cand.ts or ts > end_ts:
                continue
            mid = _pick_atm_mid(option_series[ts], option_side)
            if mid is not None and mid > 0:
                roi_path.append((int(ts), (float(mid) - float(entry_mid)) / float(entry_mid)))
        if not roi_path:
            continue

        best_ts, max_roi = max(roi_path, key=lambda x: x[1])
        _, min_roi = min(roi_path, key=lambda x: x[1])
        _, final_roi = roi_path[-1]
        analyses.append(
            CandidateProfit(
                symbol=cand.symbol,
                option_side=option_side,
                datetime_ny=cand.datetime_ny,
                entry_ts=cand.ts,
                entry_mid=float(entry_mid),
                entry_alpha=cand.alpha,
                score=cand.score,
                max_roi=float(max_roi),
                final_roi=float(final_roi),
                min_roi=float(min_roi),
                best_ts=int(best_ts),
                sample_points=len(roi_path),
            )
        )
        last_selected_ts[key] = cand.ts

    return analyses


def analyze_rankings(
    alpha_rows: list[dict[str, Any]],
    cfg,
    selected_per_frame: int,
    score_func=compute_entry_priority_score,
) -> tuple[dict[str, Any], list[RankedCandidate]]:
    states: dict[str, MetricState] = defaultdict(MetricState)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    frame_top1 = Counter()
    frame_top3 = Counter()
    frame_selected = Counter()
    focus_presence = Counter()
    all_selected_rows: list[RankedCandidate] = []

    alpha_threshold = float(getattr(cfg, "ALPHA_ENTRY_THRESHOLD", 0.6) or 0.6)
    focus_syms: set[str] = set()

    for ts in sorted(grouped.keys()):
        candidates: list[dict[str, Any]] = []
        dt_ny = ""
        for row in grouped[ts]:
            dt_ny = str(row["datetime_ny"])
            sym = str(row["symbol"])
            alpha = _safe_float(row["alpha"])
            iv = _safe_float(row["iv"], 0.5)
            price = _safe_float(row["price"])
            roc_5m, macd_hist, snap_roc = states[sym].update(price)
            if abs(alpha) < alpha_threshold:
                continue
            direction = 1 if alpha >= 0 else -1
            trend_info = compute_entry_trend_quality(
                states[sym].trend_prices,
                direction,
                window_mins=int(getattr(cfg, "ENTRY_RANK_TREND_WINDOW_MINS", 30) or 30),
            )
            rank_info = score_func(
                alpha=alpha,
                iv=iv,
                roc_5m=roc_5m,
                snap_roc=snap_roc,
                macd_hist=macd_hist,
                entry_dir=direction,
                cfg=cfg,
                pure_alpha_replay=False,
                **trend_info,
            )
            candidates.append({
                "sym": sym,
                "sig": {"dir": direction},
                "batch_idx": 0,
                "alpha_strength": rank_info["score"],
                "is_priority_candidate": bool(rank_info["is_priority_candidate"]),
                "candidate": RankedCandidate(
                    ts=ts,
                    datetime_ny=dt_ny,
                    symbol=sym,
                    alpha=alpha,
                    iv=iv,
                    price=price,
                    roc_5m=roc_5m,
                    snap_roc=snap_roc,
                    macd_hist=macd_hist,
                    score=rank_info["score"],
                    is_priority_candidate=bool(rank_info["is_priority_candidate"]),
                ),
            })
        if not candidates:
            continue
        candidates.sort(key=lambda x: float(x["alpha_strength"]), reverse=True)
        frame_top1[candidates[0]["sym"]] += 1
        for cand in candidates[:3]:
            frame_top3[cand["sym"]] += 1
        selected = reserve_priority_entry_slots(candidates, selected_per_frame, cfg)
        for cand in selected:
            frame_selected[cand["sym"]] += 1
            all_selected_rows.append(cand["candidate"])

    ranking_summary = {
        "frames": sum(frame_top1.values()),
        "top1": frame_top1,
        "top3": frame_top3,
        "selected": frame_selected,
    }
    return ranking_summary, all_selected_rows


def _alpha_onset_for_trade(symbol_rows: list[dict[str, Any]], entry_ts: float, direction: int, threshold: float) -> dict[str, Any] | None:
    eligible = [row for row in symbol_rows if int(float(row["ts"])) <= _minute_floor(entry_ts)]
    if not eligible:
        return None
    streak: list[dict[str, Any]] = []
    for row in reversed(eligible):
        alpha = _safe_float(row["alpha"])
        if abs(alpha) < threshold or (alpha >= 0) != (direction >= 0):
            break
        streak.append(row)
    if not streak:
        return None
    return streak[-1]


def analyze_trades(
    order_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    option_paths: dict[str, dict[int, str]],
    cfg,
) -> list[dict[str, Any]]:
    symbol_alpha_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        symbol_alpha_map[str(row["symbol"])].append(row)

    pending_by_symbol_side: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    open_lots: dict[tuple[str, str, str], deque[OpenLot]] = defaultdict(deque)
    analyses: list[dict[str, Any]] = []
    alpha_threshold = float(getattr(cfg, "ALPHA_ENTRY_THRESHOLD", 0.6) or 0.6)

    for row in order_rows:
        details = row.get("details", {})
        side = str(details.get("side") or "").upper()
        symbol = str(row["symbol"])

        if row["action"] == "ORDER_PENDING":
            pending_by_symbol_side[(symbol, side)].append(row)
            continue

        if row["action"] != "ORDER_FILLED":
            continue

        pending = pending_by_symbol_side[(symbol, side)][-1] if pending_by_symbol_side[(symbol, side)] else None
        note = ((pending or {}).get("details") or {}).get("strategy_note", {}) if pending else {}
        reason = str((details.get("strategy_note", {}) or {}).get("reason") or note.get("reason") or "")

        if side == "BUY":
            contract_id = str(note.get("contract_id") or "")
            option_side, strike = _parse_contract(contract_id)
            if not option_side:
                tag = str(note.get("tag") or "")
                option_side = "CALL" if "CALL" in tag else "PUT"
            open_lots[_lot_key(symbol, contract_id, option_side)].append(
                OpenLot(
                    symbol=symbol,
                    qty=_safe_float(row["qty"]),
                    entry_ts=_safe_float(row["ts"]),
                    entry_dt=str(row["datetime_ny"]),
                    entry_price=_safe_float(row["price"]),
                    contract_id=contract_id,
                    option_side=option_side,
                    strike=strike,
                    entry_reason=reason,
                )
            )
            continue

        if side != "SELL":
            continue

        sell_contract_id = str(note.get("contract_id") or "")
        sell_option_side, _ = _parse_contract(sell_contract_id)
        if not sell_option_side:
            tag = str(note.get("tag") or "")
            sell_option_side = "CALL" if "CALL" in tag else ("PUT" if "PUT" in tag else "")
        lot_key = _lot_key(symbol, sell_contract_id, sell_option_side)
        if not open_lots[lot_key]:
            continue

        remaining = _safe_float(row["qty"])
        while remaining > 1e-9 and open_lots[lot_key]:
            lot = open_lots[lot_key][0]
            matched = min(remaining, lot.qty)
            remaining -= matched
            lot.qty -= matched

            direction = 1 if lot.option_side == "CALL" else -1
            onset_row = _alpha_onset_for_trade(symbol_alpha_map[symbol], lot.entry_ts, direction, alpha_threshold)
            onset_ts = int(float(onset_row["ts"])) if onset_row else None
            onset_dt = str(onset_row["datetime_ny"]) if onset_row else ""
            onset_alpha = _safe_float(onset_row["alpha"]) if onset_row else None
            onset_mid = None
            if onset_ts is not None:
                onset_mid = _pick_bucket_mid(
                    _lookup_snapshot_near(option_paths.get(symbol, {}), onset_ts),
                    lot.option_side,
                    lot.strike,
                    lot.entry_price,
                )

            entry_to_exit_roi = (_safe_float(row["price"]) - lot.entry_price) / max(0.01, lot.entry_price)
            onset_to_exit_roi = None
            if onset_mid and onset_mid > 0:
                onset_to_exit_roi = (_safe_float(row["price"]) - onset_mid) / onset_mid

            analyses.append({
                "symbol": symbol,
                "option_side": lot.option_side,
                "contract_id": lot.contract_id,
                "entry_dt": lot.entry_dt,
                "exit_dt": str(row["datetime_ny"]),
                "entry_reason": lot.entry_reason,
                "exit_reason": reason,
                "entry_price": lot.entry_price,
                "exit_price": _safe_float(row["price"]),
                "entry_to_exit_roi": entry_to_exit_roi,
                "alpha_onset_dt": onset_dt,
                "alpha_onset_alpha": onset_alpha,
                "alpha_onset_price": onset_mid,
                "alpha_onset_lead_mins": ((lot.entry_ts - onset_ts) / 60.0) if onset_ts else None,
                "alpha_onset_to_exit_roi": onset_to_exit_roi,
                "hold_mins": (_safe_float(row["ts"]) - lot.entry_ts) / 60.0,
            })

            if lot.qty <= 1e-9:
                open_lots[lot_key].popleft()

    return analyses


def _format_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def print_ranking_summary(summary: dict[str, Any], focus_symbols: list[str], limit: int, title: str = "全天排序概览") -> None:
    frames = int(summary["frames"])
    top1: Counter = summary["top1"]
    top3: Counter = summary["top3"]
    selected: Counter = summary["selected"]

    print(f"## {title}")
    print(f"有效分钟帧: {frames}")
    print("Top1 最多标的:")
    for sym, cnt in top1.most_common(limit):
        print(f"  {sym}: {cnt} 次 ({cnt / max(1, frames) * 100:.1f}%)")

    print("\n理论入选(Top3+保留名额)最多标的:")
    for sym, cnt in selected.most_common(limit):
        print(f"  {sym}: {cnt} 次 ({cnt / max(1, frames) * 100:.1f}%)")

    if focus_symbols:
        print("\n关注标的覆盖:")
        for sym in focus_symbols:
            print(
                f"  {sym}: top1={top1.get(sym, 0)} | top3={top3.get(sym, 0)} | "
                f"selected={selected.get(sym, 0)}"
            )


def _avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def print_candidate_profit_summary(
    profits: list[CandidateProfit],
    focus_symbols: list[str],
    limit: int,
    title: str,
) -> None:
    rows = profits
    if focus_symbols:
        rows = [row for row in rows if row.symbol in focus_symbols]

    print(f"\n## {title}")
    if not rows:
        print("没有足够的理论候选期权快照可分析。")
        return

    max_rois = [row.max_roi for row in rows]
    final_rois = [row.final_roi for row in rows]
    min_rois = [row.min_roi for row in rows]
    print(
        f"样本数: {len(rows)} | "
        f"平均最大ROI={_format_pct(_avg(max_rois))} | "
        f"平均最终ROI={_format_pct(_avg(final_rois))} | "
        f"平均最大回撤={_format_pct(_avg(min_rois))}"
    )
    print(
        f"maxROI>=10%命中率={sum(1 for x in max_rois if x >= 0.10) / len(max_rois) * 100:.1f}% | "
        f"maxROI>=20%命中率={sum(1 for x in max_rois if x >= 0.20) / len(max_rois) * 100:.1f}% | "
        f"最终为正={sum(1 for x in final_rois if x > 0) / len(final_rois) * 100:.1f}%"
    )

    by_symbol: dict[str, list[CandidateProfit]] = defaultdict(list)
    for row in rows:
        by_symbol[row.symbol].append(row)
    ranked = sorted(
        by_symbol.items(),
        key=lambda item: (len(item[1]), _avg([row.max_roi for row in item[1]])),
        reverse=True,
    )[:limit]
    print("按标的汇总:")
    for sym, sym_rows in ranked:
        sym_max = [row.max_roi for row in sym_rows]
        sym_final = [row.final_roi for row in sym_rows]
        print(
            f"  {sym}: n={len(sym_rows)} "
            f"avgMax={_format_pct(_avg(sym_max))} "
            f"avgFinal={_format_pct(_avg(sym_final))} "
            f"hit20={sum(1 for x in sym_max if x >= 0.20) / len(sym_max) * 100:.1f}%"
        )

    best_rows = sorted(rows, key=lambda x: x.max_roi, reverse=True)[: min(limit, 8)]
    print("最佳理论候选:")
    for row in best_rows:
        print(
            f"  {row.symbol} {row.option_side} {row.datetime_ny} "
            f"alpha={row.entry_alpha:.2f} entryMid={row.entry_mid:.2f} "
            f"max={_format_pct(row.max_roi)} final={_format_pct(row.final_roi)} "
            f"min={_format_pct(row.min_roi)}"
        )


def print_trade_summary(trades: list[dict[str, Any]], focus_symbols: list[str], limit: int) -> None:
    if focus_symbols:
        trades = [t for t in trades if t["symbol"] in focus_symbols]

    print("\n## 单笔利润分析")
    if not trades:
        print("没有匹配到完整的已平仓交易。")
        return

    onset_rois = [t["alpha_onset_to_exit_roi"] for t in trades if t["alpha_onset_to_exit_roi"] is not None]
    actual_rois = [t["entry_to_exit_roi"] for t in trades]
    print(f"平仓笔数: {len(trades)}")
    print(f"实际入场到平仓平均 ROI: {_format_pct(sum(actual_rois) / max(1, len(actual_rois)))}")
    if onset_rois:
        print(f"alpha 初段到平仓平均 ROI: {_format_pct(sum(onset_rois) / len(onset_rois))}")

    ranked = sorted(
        trades,
        key=lambda x: abs(float(x["alpha_onset_to_exit_roi"] or x["entry_to_exit_roi"] or 0.0)),
        reverse=True,
    )[:limit]
    for row in ranked:
        lead_txt = f"{row['alpha_onset_lead_mins']:.1f}m" if row["alpha_onset_lead_mins"] is not None else "n/a"
        onset_dt = row["alpha_onset_dt"] or "n/a"
        print(
            f"{row['symbol']} {row['option_side']} "
            f"entry={row['entry_dt']} exit={row['exit_dt']} "
            f"actual={_format_pct(row['entry_to_exit_roi'])} "
            f"onset={_format_pct(row['alpha_onset_to_exit_roi'])} "
            f"onset_dt={onset_dt} lead={lead_txt}"
        )
        print(f"    entry_reason={row['entry_reason']} | exit_reason={row['exit_reason']}")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    focus_symbols = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()]
    cfg = StrategyConfig()

    conn = psycopg2.connect(PG_DB_URL)
    try:
        alpha_rows = load_alpha_rows(conn, args.date)
        if not alpha_rows:
            print("当天没有 alpha_logs 数据。")
            return
        ranking_summary, selected_rows = analyze_rankings(alpha_rows, cfg, args.selected_per_frame)
        legacy_summary, legacy_selected_rows = analyze_rankings(alpha_rows, cfg, args.selected_per_frame, legacy_entry_priority_score)

        min_ts = min(int(float(r["ts"])) for r in alpha_rows)
        max_ts = max(int(float(r["ts"])) for r in alpha_rows)
        symbols_scope = sorted({str(r["symbol"]) for r in alpha_rows})
        order_rows = load_order_rows(conn, args.date, focus_symbols or symbols_scope)
        option_paths = load_option_paths(conn, min_ts - 3600, max_ts + 3600, focus_symbols or symbols_scope)
        candidate_profits = analyze_candidate_profits(
            selected_rows,
            option_paths,
            horizon_mins=args.candidate_horizon_mins,
            cooldown_mins=args.candidate_cooldown_mins,
        )
        legacy_candidate_profits = analyze_candidate_profits(
            legacy_selected_rows,
            option_paths,
            horizon_mins=args.candidate_horizon_mins,
            cooldown_mins=args.candidate_cooldown_mins,
        )
        trade_rows = analyze_trades(order_rows, alpha_rows, option_paths, cfg)
    finally:
        conn.close()

    print_ranking_summary(legacy_summary, focus_symbols, args.limit, title="旧公式排序概览(alpha/iv)")
    print()
    print_ranking_summary(ranking_summary, focus_symbols, args.limit, title="新公式排序概览(alpha核心)")
    print_candidate_profit_summary(
        legacy_candidate_profits,
        focus_symbols,
        args.limit,
        title=f"旧公式理论候选后续收益({args.candidate_horizon_mins}m)",
    )
    print_candidate_profit_summary(
        candidate_profits,
        focus_symbols,
        args.limit,
        title=f"alpha核心理论候选后续收益({args.candidate_horizon_mins}m)",
    )
    print_trade_summary(trade_rows, focus_symbols, args.limit)

    if args.csv_out:
        write_csv(args.csv_out, trade_rows)
        print(f"\nCSV 已写出: {args.csv_out}")


if __name__ == "__main__":
    main()
