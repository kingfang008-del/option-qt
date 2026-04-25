#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    scripts_dir = production_dir / "scripts"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(scripts_dir))


class _FakeRedis:
    def __init__(self):
        self.hashes = {}

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def hdel(self, key, *fields):
        bucket = self.hashes.get(key, {})
        removed = 0
        for field in fields:
            if field in bucket:
                removed += 1
                bucket.pop(field, None)
        self.hashes[key] = bucket
        return removed


def test_cleanup_redis_state_removes_live_and_pending_entries() -> None:
    _bootstrap_imports()
    import repair_missing_realtime_closes as script  # noqa: E402

    fake = _FakeRedis()
    pending_key = script.namespaced_pending_orders_key(script.OMS_STATE_NAMESPACE)
    fake.hashes["oms:live_positions"] = {
        b"NVDA": json.dumps({"position": 1, "qty": 2}).encode(),
        b"AAPL": json.dumps({"position": 1, "qty": 1}).encode(),
    }
    fake.hashes[pending_key] = {
        b"k1": json.dumps({"symbol": "NVDA", "status": "SUBMITTED"}).encode(),
        b"k2": json.dumps({"symbol": "MSFT", "status": "SUBMITTED"}).encode(),
    }

    original_redis = script.redis.Redis
    script.redis.Redis = lambda **kwargs: fake
    try:
        stats = script._cleanup_redis_state(["NVDA"])
    finally:
        script.redis.Redis = original_redis

    assert stats["live_positions_removed"] == 1
    assert stats["pending_orders_removed"] == 1
    assert b"NVDA" not in fake.hashes["oms:live_positions"]
    assert b"k1" not in fake.hashes[pending_key]
    assert b"AAPL" in fake.hashes["oms:live_positions"]
    assert b"k2" in fake.hashes[pending_key]


def main() -> None:
    test_cleanup_redis_state_removes_live_and_pending_entries()
    print("[OK] repair missing realtime closes guards passed")


if __name__ == "__main__":
    main()
