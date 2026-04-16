#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：IBKR 批量发布时应丢弃过旧 bar，且 payload.ts 使用 bar 自身时间。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(dao_dir))


class _LoggerStub:
    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _SerStub:
    @staticmethod
    def pack(obj):
        return json.dumps(obj).encode("utf-8")


class _RedisStub:
    def __init__(self):
        self.calls = []

    def xadd(self, stream, mapping, maxlen=None):
        self.calls.append((stream, mapping, maxlen))


def _load_publish_snapshot():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "ibkr_connector_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    target = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "IBKRConnectorFinal":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_publish_batch_snapshot":
                    target = item
                    break
            break
    if target is None:
        raise RuntimeError("未找到 IBKRConnectorFinal._publish_batch_snapshot")
    mini = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {
        "ser": _SerStub(),
        "logger": _LoggerStub(),
        "STREAM_KEY_FUSED": "fused_market_stream",
    }
    exec(compile(mini, filename=str(src_path), mode="exec"), ns)
    return ns["_publish_batch_snapshot"]


def test_ibkr_batch_staleness_guard() -> None:
    _bootstrap_imports()
    fn = _load_publish_snapshot()

    svc = type("S", (), {})()
    svc.active_stocks = {"NVDA": object(), "AAPL": object()}
    svc._last_bar_cache = {
        "NVDA": {"open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10.0, "ts": 100.0},
        "AAPL": {"open": 2.0, "high": 2.2, "low": 1.9, "close": 2.1, "volume": 20.0, "ts": 97.0},
    }
    svc._max_batch_staleness_sec = 1.5
    svc.redis = _RedisStub()
    svc._sanitize_publish_ts = lambda ts_val, source_tag="": float(ts_val)
    svc._collect_option_buckets = lambda sym: ([], [])

    bound = fn.__get__(svc, svc.__class__)
    bound(100.0)

    assert len(svc.redis.calls) == 1, "应仅发布包含新鲜 bar 的批次"
    _, mapping, _ = svc.redis.calls[0]
    payloads = json.loads(mapping["batch"].decode("utf-8"))
    assert len(payloads) == 1 and payloads[0]["symbol"] == "NVDA", "AAPL 应被 stale 过滤"
    assert float(payloads[0]["ts"]) == 100.0, "payload.ts 应使用 bar 自身时间"
    print("[OK] ibkr batch staleness guard passed")


def main() -> None:
    test_ibkr_batch_staleness_guard()


if __name__ == "__main__":
    main()
