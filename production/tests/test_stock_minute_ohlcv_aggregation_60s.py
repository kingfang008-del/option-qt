#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单元测试：模拟 1 分钟内 60 条秒级 stock 数据，验证分钟 OHLCV 聚合结果。

覆盖点：
1) open/high/low/close/volume/vwap 聚合是否符合 finalize_1min_bar 语义；
2) 成功写入后是否走本地 history_1min 增量更新（不触发 Redis 回灌）；
3) cleanup=True 时是否清空当前分钟缓存。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(dao_dir))


class _DummyRedis:
    def __init__(self):
        self.hset_calls = []

    def hset(self, key, field, value):
        self.hset_calls.append((key, field, value))
        return 1


class _DummyService:
    pass


def _build_ticks_60s():
    ticks = []
    for sec in range(60):
        close = 100.0 + sec * 0.1
        tick = {
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": float(sec + 1),
        }
        ticks.append(tick)
    return ticks


def test_stock_minute_ohlcv_aggregation_60s() -> None:
    _bootstrap_imports()
    from fcs_persistence_handler import FCSPersistenceHandler  # noqa: E402
    from config import NY_TZ  # noqa: E402

    sym = "NVDA"
    bar_minute = pd.Timestamp("2026-04-15 10:02:00", tz=NY_TZ)
    prev_minute = bar_minute - timedelta(minutes=1)

    svc = _DummyService()
    svc.r = _DummyRedis()
    svc.current_bars_5s = {sym: _build_ticks_60s()}
    svc.last_tick_price = {sym: 0.0}
    svc.latest_prices = {sym: 0.0}
    svc.option_snapshot = {sym: np.zeros((6, 12), dtype=np.float32)}
    svc.latest_opt_buckets = {sym: np.zeros((6, 12), dtype=np.float32)}
    svc.frozen_option_snapshot = {sym: np.zeros((6, 12), dtype=np.float32)}
    svc.frozen_latest_opt_buckets = {sym: np.zeros((6, 12), dtype=np.float32)}
    svc.latest_opt_contracts = {sym: []}
    svc.frozen_latest_opt_contracts = {sym: []}
    svc._merge_option_snapshot_with_greeks = lambda raw, enriched: np.zeros((6, 12), dtype=np.float32)
    svc._extract_tagged_atm_iv = lambda buckets: (0.0, 0.0)

    svc.sync_history_calls = 0
    svc.sync_opt_calls = 0
    svc._sync_history_from_redis = lambda _sym: setattr(svc, "sync_history_calls", svc.sync_history_calls + 1)
    svc._sync_option_history_from_redis = lambda _sym: setattr(svc, "sync_opt_calls", svc.sync_opt_calls + 1)

    # 预置上一分钟一条历史，保证走“本地增量更新”而不是全量回灌
    svc.history_1min = {
        sym: pd.DataFrame(
            [{
                "open": 99.0, "high": 99.5, "low": 98.7, "close": 99.2,
                "volume": 1000.0, "vwap": 99.1
            }],
            index=[prev_minute],
        )
    }
    svc.history_5min = {sym: pd.DataFrame()}

    handler = FCSPersistenceHandler(svc)
    ok = handler.finalize_1min_bar(sym, bar_minute, cleanup=True)
    assert ok is True, "finalize_1min_bar 未返回 True"

    # 期望值
    ticks = _build_ticks_60s()
    exp_open = float(ticks[0]["open"])
    exp_close = float(ticks[-1]["close"])
    exp_high = float(max(max(t["high"], t["close"]) for t in ticks))
    exp_low = float(min(min(t["low"], t["close"]) for t in ticks))
    exp_vol = float(sum(max(0.0, t["volume"]) for t in ticks))
    exp_vwap = float(sum(t["close"] * max(0.0, t["volume"]) for t in ticks) / (exp_vol + 1e-10))

    # 校验 history_1min 新分钟行
    out_df = svc.history_1min[sym]
    assert bar_minute in out_df.index, "history_1min 未写入目标分钟行"
    row = out_df.loc[bar_minute]
    assert abs(float(row["open"]) - exp_open) < 1e-6, "open 聚合异常"
    assert abs(float(row["high"]) - exp_high) < 1e-6, "high 聚合异常"
    assert abs(float(row["low"]) - exp_low) < 1e-6, "low 聚合异常"
    assert abs(float(row["close"]) - exp_close) < 1e-6, "close 聚合异常"
    assert abs(float(row["volume"]) - exp_vol) < 1e-6, "volume 聚合异常"
    assert abs(float(row["vwap"]) - exp_vwap) < 1e-6, "vwap 聚合异常"

    # 校验成功路径不应触发 Redis 历史回灌
    assert svc.sync_history_calls == 0, "不应触发 _sync_history_from_redis"
    assert svc.sync_opt_calls == 0, "不应触发 _sync_option_history_from_redis"

    # 校验写 Redis 的 BAR:1M payload
    bar_calls = [c for c in svc.r.hset_calls if c[0] == f"BAR:1M:{sym}"]
    assert bar_calls, "未写 BAR:1M Redis 哈希"
    bar_payload = json.loads(bar_calls[-1][2])
    assert abs(float(bar_payload["open"]) - exp_open) < 1e-6
    assert abs(float(bar_payload["close"]) - exp_close) < 1e-6

    # cleanup=True 应清空当前分钟缓存
    assert svc.current_bars_5s[sym] == [], "cleanup 后 current_bars_5s 未清空"

    print("[OK] 60秒 stock -> 1分钟 OHLCV 聚合测试通过")
    print(
        f"[INFO] minute={bar_minute} open={exp_open:.4f} high={exp_high:.4f} "
        f"low={exp_low:.4f} close={exp_close:.4f} volume={exp_vol:.0f} vwap={exp_vwap:.4f}"
    )


def main() -> None:
    test_stock_minute_ohlcv_aggregation_60s()


if __name__ == "__main__":
    main()

