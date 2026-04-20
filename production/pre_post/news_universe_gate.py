#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pre/Post 新闻门控 Universe 服务（与主链路隔离版）

功能:
1) 从 Redis 读取新闻分数与元数据；
2) 可选从 PostgreSQL `stocks_us` 按行业加载扫描池（如 Biotechnology）；
3) 在盘前/盘后筛选可交易候选，并可用扫描池轮转补位；
4) 输出 active symbol 列表到 Redis，供其他模块按需消费。

默认 Redis schema:
- ZSET prepost:news_scores
    member: SYMBOL (例如 NVDA)
    score : news_score (0~1 建议)
- HASH prepost:news_meta:<SYMBOL>
    last_ts, news_score, dollar_volume, spread_pct, quote_stability, headline, source, event_type
- STRING prepost:active_symbols
    JSON 数组，示例: ["NVDA","TSLA","AAPL"]
- HASH prepost:active_symbols_meta
    generated_ts, selected_count, reason, cap
"""

import json
import os
import time
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pytz
import redis


NY_TZ = pytz.timezone("America/New_York")


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="ignore")
        return float(v)
    except Exception:
        return float(default)


def _safe_str(v, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    return str(v)


@dataclass
class NewsGateConfig:
    redis_host: str = os.environ.get("PREPOST_REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("PREPOST_REDIS_PORT", "6379"))
    redis_db: int = int(os.environ.get("PREPOST_REDIS_DB", "0"))

    enabled: bool = _env_flag("PREPOST_NEWS_GATE_ENABLED", True)
    max_symbols: int = max(1, int(os.environ.get("PREPOST_MAX_SYMBOLS", "300")))
    refresh_seconds: int = max(5, int(os.environ.get("PREPOST_REFRESH_SECONDS", "30")))
    min_hold_seconds: int = max(30, int(os.environ.get("PREPOST_MIN_HOLD_SECONDS", "180")))

    news_lookback_minutes: int = max(1, int(os.environ.get("PREPOST_NEWS_LOOKBACK_MINUTES", "120")))
    min_news_score: float = float(os.environ.get("PREPOST_MIN_NEWS_SCORE", "0.55"))
    min_dollar_volume: float = float(os.environ.get("PREPOST_MIN_DOLLAR_VOLUME", "1500000"))
    max_spread_pct: float = float(os.environ.get("PREPOST_MAX_SPREAD_PCT", "0.025"))
    min_quote_stability: float = float(os.environ.get("PREPOST_MIN_QUOTE_STABILITY", "0.50"))

    score_zset_key: str = os.environ.get("PREPOST_NEWS_SCORE_ZSET_KEY", "prepost:news_scores")
    meta_key_prefix: str = os.environ.get("PREPOST_NEWS_META_KEY_PREFIX", "prepost:news_meta:")
    active_key: str = os.environ.get("PREPOST_ACTIVE_SYMBOLS_KEY", "prepost:active_symbols")
    active_meta_key: str = os.environ.get("PREPOST_ACTIVE_META_KEY", "prepost:active_symbols_meta")

    # PostgreSQL 行业扫描池（默认读取 stocks_us 的 Biotechnology）
    pg_universe_enabled: bool = _env_flag("PREPOST_PG_UNIVERSE_ENABLED", True)
    pg_db_url: str = os.environ.get(
        "PREPOST_PG_DB_URL",
        os.environ.get("PG_DB_URL", "dbname=quant_trade user=postgres password=postgres host=127.0.0.1 port=5432"),
    )
    pg_table: str = os.environ.get("PREPOST_PG_TABLE", "stocks_us")
    pg_symbol_col: str = os.environ.get("PREPOST_PG_SYMBOL_COL", "symbol")
    pg_industry_col: str = os.environ.get("PREPOST_PG_INDUSTRY_COL", "industry")
    pg_industry_value: str = os.environ.get("PREPOST_PG_INDUSTRY_VALUE", "Biotechnology")
    pg_refresh_seconds: int = max(60, int(os.environ.get("PREPOST_PG_REFRESH_SECONDS", "900")))

    # 当新闻候选不足时，是否用行业扫描池轮转补位
    scan_fill_enabled: bool = _env_flag("PREPOST_SCAN_FILL_ENABLED", True)


class PrePostNewsUniverseGate:
    def __init__(self, base_symbols: List[str], cfg: Optional[NewsGateConfig] = None):
        self.cfg = cfg or NewsGateConfig()
        self.base_symbols = list(dict.fromkeys([s.strip().upper() for s in base_symbols if s.strip()]))
        self.r = redis.Redis(
            host=self.cfg.redis_host,
            port=self.cfg.redis_port,
            db=self.cfg.redis_db,
            decode_responses=False,
        )
        self.last_added_ts: Dict[str, float] = {}
        self.last_selected: List[str] = list(self.base_symbols)
        self.pg_universe: List[str] = []
        self.pg_universe_set: Set[str] = set()
        self._pg_last_refresh_ts = 0.0
        self._scan_cursor = 0

    @staticmethod
    def _now_ny() -> datetime.datetime:
        return datetime.datetime.now(NY_TZ)

    def is_prepost_session(self, now_ny: Optional[datetime.datetime] = None) -> bool:
        now_ny = now_ny or self._now_ny()
        if now_ny.weekday() >= 5:
            return False
        h, m = now_ny.hour, now_ny.minute
        # 04:00-09:29 & 16:00-19:59
        if (h > 4 or (h == 4 and m >= 0)) and (h < 9 or (h == 9 and m < 30)):
            return True
        if (h > 16 or (h == 16 and m >= 0)) and h < 20:
            return True
        return False

    def _load_symbol_meta(self, sym: str) -> Dict[str, str]:
        key = f"{self.cfg.meta_key_prefix}{sym}"
        try:
            raw = self.r.hgetall(key) or {}
        except Exception:
            raw = {}
        out: Dict[str, str] = {}
        for k, v in raw.items():
            out[_safe_str(k)] = _safe_str(v)
        return out

    def _refresh_pg_universe(self, now_ts: float, force: bool = False) -> None:
        if not self.cfg.pg_universe_enabled:
            return
        if (not force) and (now_ts - self._pg_last_refresh_ts) < float(self.cfg.pg_refresh_seconds):
            return
        self._pg_last_refresh_ts = now_ts

        try:
            import psycopg2
        except Exception:
            return

        sql = (
            f"SELECT DISTINCT UPPER(TRIM({self.cfg.pg_symbol_col})) AS sym "
            f"FROM {self.cfg.pg_table} "
            f"WHERE {self.cfg.pg_industry_col} = %s "
            f"AND {self.cfg.pg_symbol_col} IS NOT NULL "
            f"AND TRIM({self.cfg.pg_symbol_col}) <> ''"
        )
        conn = None
        try:
            conn = psycopg2.connect(self.cfg.pg_db_url)
            cur = conn.cursor()
            cur.execute(sql, (self.cfg.pg_industry_value,))
            rows = cur.fetchall() or []
            syms = []
            for row in rows:
                sym = str(row[0]).strip().upper() if row and row[0] else ""
                if sym:
                    syms.append(sym)
            syms = list(dict.fromkeys(syms))
            self.pg_universe = syms
            self.pg_universe_set = set(syms)
        except Exception:
            # 保留上次成功加载的 universe，避免一次查询失败导致扫描池清空
            pass
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    def _load_ranked_candidates(
        self,
        now_ts: float,
        top_n: int = 300,
        allowed_universe: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        try:
            ranked = self.r.zrevrange(self.cfg.score_zset_key, 0, max(0, top_n - 1), withscores=True) or []
        except Exception:
            ranked = []

        max_age_sec = float(self.cfg.news_lookback_minutes * 60)
        out: List[Tuple[str, float]] = []
        for member, score in ranked:
            sym = _safe_str(member).strip().upper()
            if not sym:
                continue
            if allowed_universe is not None and sym not in allowed_universe:
                continue
            meta = self._load_symbol_meta(sym)
            last_ts = _safe_float(meta.get("last_ts"), 0.0)
            if last_ts <= 0 or (now_ts - last_ts) > max_age_sec:
                continue

            news_score = max(_safe_float(score, 0.0), _safe_float(meta.get("news_score"), 0.0))
            if news_score < self.cfg.min_news_score:
                continue

            dollar_volume = _safe_float(meta.get("dollar_volume"), 0.0)
            spread_pct = _safe_float(meta.get("spread_pct"), 0.0)
            quote_stability = _safe_float(meta.get("quote_stability"), 0.0)

            if dollar_volume > 0 and dollar_volume < self.cfg.min_dollar_volume:
                continue
            if spread_pct > 0 and spread_pct > self.cfg.max_spread_pct:
                continue
            if quote_stability > 0 and quote_stability < self.cfg.min_quote_stability:
                continue

            recency_boost = max(0.0, 1.0 - (now_ts - last_ts) / max_age_sec)
            total_score = news_score + 0.20 * recency_boost
            out.append((sym, float(total_score)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _fill_from_scan_pool(self, selected_set: Set[str], need_count: int) -> List[str]:
        if need_count <= 0:
            return []
        pool = self.pg_universe
        if not pool:
            return []
        out: List[str] = []
        n = len(pool)
        for _ in range(n):
            if len(out) >= need_count:
                break
            sym = pool[self._scan_cursor % n]
            self._scan_cursor += 1
            if sym in selected_set:
                continue
            out.append(sym)
            selected_set.add(sym)
        return out

    def _apply_min_hold(self, selected: List[str], now_ts: float) -> List[str]:
        selected_set = set(selected)
        for sym in self.last_selected:
            added_ts = self.last_added_ts.get(sym, 0.0)
            if (now_ts - added_ts) < self.cfg.min_hold_seconds:
                selected_set.add(sym)
        capped = list(selected_set)[: self.cfg.max_symbols]
        return capped

    def select_symbols(self, now_ts: Optional[float] = None) -> Tuple[List[str], Dict[str, object]]:
        now_ts = float(now_ts if now_ts is not None else time.time())
        self._refresh_pg_universe(now_ts=now_ts)
        core = list(self.base_symbols)

        if not self.cfg.enabled:
            return core, {
                "reason": "disabled",
                "selected_extra": 0,
                "cap": self.cfg.max_symbols,
                "pg_universe_size": len(self.pg_universe),
                "scan_filled": 0,
            }

        if not self.is_prepost_session():
            return core, {
                "reason": "rth",
                "selected_extra": 0,
                "cap": self.cfg.max_symbols,
                "pg_universe_size": len(self.pg_universe),
                "scan_filled": 0,
            }

        allowed_universe = set(core)
        if self.cfg.pg_universe_enabled and self.pg_universe_set:
            allowed_universe.update(self.pg_universe_set)

        ranked = self._load_ranked_candidates(now_ts=now_ts, top_n=600, allowed_universe=allowed_universe)
        selected = core[: self.cfg.max_symbols]
        selected_set = set(selected)
        for sym, _score in ranked:
            if len(selected) >= self.cfg.max_symbols:
                break
            if sym in selected_set:
                continue
            selected.append(sym)
            selected_set.add(sym)

        scan_filled = 0
        if self.cfg.scan_fill_enabled and len(selected) < self.cfg.max_symbols:
            need = self.cfg.max_symbols - len(selected)
            filler = self._fill_from_scan_pool(selected_set=selected_set, need_count=need)
            selected.extend(filler)
            scan_filled = len(filler)

        selected = self._apply_min_hold(selected, now_ts=now_ts)
        for sym in selected:
            if sym not in self.last_added_ts:
                self.last_added_ts[sym] = now_ts

        self.last_selected = list(selected)
        diag = {
            "reason": "prepost_news_gate",
            "core": len(core),
            "ranked_candidates": len(ranked),
            "selected_extra": max(0, len(selected) - min(len(core), self.cfg.max_symbols)),
            "cap": self.cfg.max_symbols,
            "pg_universe_size": len(self.pg_universe),
            "scan_filled": scan_filled,
        }
        return selected, diag

    def publish(self, symbols: List[str], diag: Dict[str, object], now_ts: Optional[float] = None) -> None:
        now_ts = float(now_ts if now_ts is not None else time.time())
        payload = json.dumps(symbols, ensure_ascii=True)
        pipe = self.r.pipeline(transaction=True)
        pipe.set(self.cfg.active_key, payload)
        pipe.hset(
            self.cfg.active_meta_key,
            mapping={
                "generated_ts": str(now_ts),
                "selected_count": str(len(symbols)),
                "reason": str(diag.get("reason", "")),
                "selected_extra": str(diag.get("selected_extra", 0)),
                "cap": str(diag.get("cap", self.cfg.max_symbols)),
                "pg_universe_size": str(diag.get("pg_universe_size", 0)),
                "scan_filled": str(diag.get("scan_filled", 0)),
            },
        )
        pipe.expire(self.cfg.active_meta_key, 3600)
        pipe.execute()

    def run_loop(self) -> None:
        print(
            f"[pre_post] gate start | enabled={self.cfg.enabled} cap={self.cfg.max_symbols} "
            f"refresh={self.cfg.refresh_seconds}s redis={self.cfg.redis_host}:{self.cfg.redis_port}/{self.cfg.redis_db} "
            f"| pg_universe={self.cfg.pg_universe_enabled} industry={self.cfg.pg_industry_value}"
        )
        while True:
            ts_now = time.time()
            symbols, diag = self.select_symbols(now_ts=ts_now)
            self.publish(symbols, diag, now_ts=ts_now)
            print(
                f"[pre_post] publish symbols={len(symbols)} reason={diag.get('reason')} "
                f"extra={diag.get('selected_extra', 0)} scan_fill={diag.get('scan_filled', 0)} "
                f"pg_pool={diag.get('pg_universe_size', 0)}"
            )
            time.sleep(self.cfg.refresh_seconds)


def _load_base_symbols_from_env() -> List[str]:
    raw = os.environ.get("PREPOST_BASE_SYMBOLS", "").strip()
    if raw:
        return [s.strip().upper() for s in raw.split(",") if s.strip()]

    # 可选复用现有 config.TARGET_SYMBOLS，不强依赖
    try:
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        baseline_dir = repo_root / "baseline"
        if str(baseline_dir) not in sys.path:
            sys.path.append(str(baseline_dir))
        from config import TARGET_SYMBOLS  # type: ignore

        return [str(s).upper() for s in TARGET_SYMBOLS]
    except Exception:
        return []


if __name__ == "__main__":
    base_symbols = _load_base_symbols_from_env()
    gate = PrePostNewsUniverseGate(base_symbols=base_symbols)
    gate.run_loop()

