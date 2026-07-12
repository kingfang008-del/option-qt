#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入场保险丝 —— 防止特征/归一化异常导致实盘失控。

两层防护(均在 decide_entry 前检查):
  1. Canary 门禁: 开盘前 canary_gate.py 用昨日数据对拍 live 栈 vs strict replay,
     写 gate JSON;不通过或过期则当日禁止新开仓(只 shadow)。
  2. 盘中 edge 分布 kill-switch: 归一化坏掉的典型形态是 edge 系统性膨胀
     (198 ENTER/天),用离线 replay 的 edge 分布参考带在线检测,越界即熔断,
     当日锁死不再开仓。

环境变量:
  QQQ_BTC_ENTRY_GUARD=0            关闭(默认 QQQ_BTC_LIVE 时开启)
  QQQ_BTC_CANARY_GATE_PATH         gate JSON(默认 ~/quant_project/shadow/canary_gate.json)
  QQQ_BTC_CANARY_GATE_REQUIRED=1   gate 缺失/过期也熔断(真实盘建议开)
  QQQ_BTC_GUARD_REF_PATH           edge 参考分布(默认 qqq_btc/CONFIG/edge_guard_ref.json)
  QQQ_BTC_GUARD_WINDOW_BARS        滚动窗口 bar 数(默认 30)
  QQQ_BTC_GUARD_MAX_ENTERS_PER_HOUR  每小时最大开仓数(默认 6)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
from pytz import timezone

logger = logging.getLogger("qqq_btc.live.entry_guard")

_NY = timezone("America/New_York")
_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_REF = _REPO / "qqq_btc" / "CONFIG" / "edge_guard_ref.json"
_GATE_RELOAD_SEC = 300.0


def entry_guard_enabled() -> bool:
    if os.environ.get("QQQ_BTC_ENTRY_GUARD", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def default_gate_path() -> Path:
    raw = os.environ.get("QQQ_BTC_CANARY_GATE_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "quant_project" / "shadow" / "canary_gate.json"


def _gate_required() -> bool:
    return os.environ.get("QQQ_BTC_CANARY_GATE_REQUIRED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ref_path() -> Path:
    raw = os.environ.get("QQQ_BTC_GUARD_REF_PATH", "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_REF


def _day_key(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=_NY).strftime("%Y-%m-%d")


class EntryGuard:
    """每个 OMS 进程一个实例(get_entry_guard 单例)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.window_bars = max(5, int(os.environ.get("QQQ_BTC_GUARD_WINDOW_BARS", "30")))
        self.max_enters_per_hour = max(
            1, int(os.environ.get("QQQ_BTC_GUARD_MAX_ENTERS_PER_HOUR", "6"))
        )
        self._edges: Deque[Tuple[float, float, float]] = deque(maxlen=self.window_bars)
        self._enter_ts: Deque[float] = deque(maxlen=256)
        self._day: Optional[str] = None
        self._kill_reason: Optional[str] = None
        self._ref: Optional[dict] = None
        self._ref_loaded = False
        self._gate_cache: Optional[dict] = None
        self._gate_loaded_at = 0.0

    # ------------------------------ refs / gate
    def _load_ref(self) -> Optional[dict]:
        if self._ref_loaded:
            return self._ref
        self._ref_loaded = True
        path = _ref_path()
        if not path.exists():
            logger.warning("[entry_guard] edge ref not found: %s (edge checks disabled)", path)
            return None
        try:
            self._ref = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("[entry_guard] failed to load ref %s: %s", path, e)
            self._ref = None
        return self._ref

    def _load_gate(self) -> Optional[dict]:
        now = time.time()
        if self._gate_cache is not None and now - self._gate_loaded_at < _GATE_RELOAD_SEC:
            return self._gate_cache
        self._gate_loaded_at = now
        path = default_gate_path()
        if not path.exists():
            self._gate_cache = None
            return None
        try:
            self._gate_cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("[entry_guard] failed to read gate %s: %s", path, e)
            self._gate_cache = {"trading_allowed": False, "reason": f"gate_unreadable:{e}"}
        return self._gate_cache

    def _check_gate(self, curr_ts: float) -> Optional[str]:
        gate = self._load_gate()
        if gate is None:
            if _gate_required():
                return "entry_guard:canary_gate_missing"
            return None
        expires = float(gate.get("expires_at", 0.0) or 0.0)
        if expires > 0 and curr_ts > expires:
            if _gate_required():
                return "entry_guard:canary_gate_expired"
            return None
        if not bool(gate.get("trading_allowed", False)):
            return "entry_guard:canary_gate_fail"
        return None

    # ------------------------------ kill latch
    def _latch_kill(self, curr_ts: float, reason: str) -> str:
        self._kill_reason = reason
        logger.error("🚨 [entry_guard] KILL latched for %s: %s", self._day, reason)
        try:
            out = (
                Path.home()
                / "quant_project"
                / "shadow"
                / f"entry_guard_kill_{self._day or _day_key(curr_ts)}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "day": self._day,
                        "ts": curr_ts,
                        "reason": reason,
                        "window_bars": self.window_bars,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        return reason

    def _maybe_reset_day(self, curr_ts: float) -> None:
        day = _day_key(curr_ts)
        if day != self._day:
            self._day = day
            self._kill_reason = None
            self._edges.clear()
            self._enter_ts.clear()

    # ------------------------------ main API
    def evaluate(
        self,
        symbol: str,
        *,
        curr_ts: float,
        call_edge: Optional[float] = None,
        put_edge: Optional[float] = None,
    ) -> Optional[str]:
        """decide_entry 前调用。返回 block reason(str) 或 None=放行。"""
        if not entry_guard_enabled():
            return None
        if curr_ts <= 0:
            return None
        with self._lock:
            self._maybe_reset_day(curr_ts)

            if self._kill_reason:
                return self._kill_reason

            gate_block = self._check_gate(curr_ts)
            if gate_block:
                return gate_block

            ce = float(call_edge) if call_edge is not None and np.isfinite(call_edge) else 0.0
            pe = float(put_edge) if put_edge is not None and np.isfinite(put_edge) else 0.0
            self._edges.append((curr_ts, ce, pe))

            ref = self._load_ref()
            if ref:
                block = self._check_edge_distribution(curr_ts, ref)
                if block:
                    return block

            enters_hour = sum(1 for t in self._enter_ts if curr_ts - t <= 3600.0)
            if enters_hour >= self.max_enters_per_hour:
                return self._latch_kill(
                    curr_ts,
                    f"entry_guard:enter_rate {enters_hour}/h > {self.max_enters_per_hour}",
                )
            return None

    def _check_edge_distribution(self, curr_ts: float, ref: dict) -> Optional[str]:
        for leg, key in (("call", 1), ("put", 2)):
            leg_ref = ref.get(f"{leg}_edge") or {}
            p99 = leg_ref.get("p99")
            hard_cap = leg_ref.get("hard_cap")
            if p99 is None and hard_cap is None:
                continue
            vals = np.array([abs(row[key]) for row in self._edges], dtype=np.float64)
            if hard_cap is not None and len(vals) and float(vals[-1]) > float(hard_cap):
                return self._latch_kill(
                    curr_ts,
                    f"entry_guard:{leg}_edge_hard_cap |{vals[-1]:.4f}| > {float(hard_cap):.4f}",
                )
            if p99 is not None and len(vals) >= self.window_bars:
                med = float(np.median(vals))
                if med > float(p99):
                    return self._latch_kill(
                        curr_ts,
                        f"entry_guard:{leg}_edge_inflation median={med:.4f} > p99={float(p99):.4f}",
                    )
        return None

    def record_enter(self, curr_ts: float) -> None:
        """decide_entry 返回 PASS 后调用。"""
        if not entry_guard_enabled() or curr_ts <= 0:
            return
        with self._lock:
            self._maybe_reset_day(curr_ts)
            self._enter_ts.append(float(curr_ts))


_GUARD: Optional[EntryGuard] = None
_GUARD_LOCK = threading.Lock()


def get_entry_guard() -> EntryGuard:
    global _GUARD
    if _GUARD is None:
        with _GUARD_LOCK:
            if _GUARD is None:
                _GUARD = EntryGuard()
    return _GUARD


def reset_entry_guard_for_tests() -> None:
    global _GUARD
    with _GUARD_LOCK:
        _GUARD = None
