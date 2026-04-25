#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


class _FakePipeline:
    def __init__(self, store: dict) -> None:
        self.store = store
        self.curr_key = None

    def delete(self, key):
        self.curr_key = key
        self.store[key] = {}
        return self

    def hset(self, key, mapping):
        self.curr_key = key
        self.store.setdefault(key, {})
        self.store[key].update(mapping)
        return self

    def expire(self, key, _ttl):
        self.curr_key = key
        return self

    def execute(self):
        return True


class _FakeRedis:
    def __init__(self) -> None:
        self.store = {}

    def pipeline(self):
        return _FakePipeline(self.store)


def test_pending_order_upsert_and_terminal_cleanup() -> None:
    _bootstrap_imports()
    import orchestrator_order_state as order_state_mod  # noqa: E402

    fake_redis = _FakeRedis()
    orch = SimpleNamespace(mode="realtime", pending_orders={})
    manager = order_state_mod.OrchestratorOrderStateManager(orch)
    manager._save_payload_async = lambda *args, **kwargs: None
    manager._get_redis = lambda: fake_redis

    order_key = "NVDA:OPEN:TEST:1"
    manager.upsert(order_key, {
        "symbol": "NVDA",
        "intent": "OPEN",
        "side": "BUY",
        "status": "SUBMITTED",
        "target_qty": 2,
        "filled_qty": 0,
        "remaining_qty": 2,
        "last_update_ts": 123.0,
    })

    redis_key = order_state_mod.namespaced_pending_orders_key(order_state_mod.OMS_STATE_NAMESPACE)
    assert order_key in orch.pending_orders
    assert redis_key in fake_redis.store
    assert order_key in fake_redis.store[redis_key]
    payload = json.loads(fake_redis.store[redis_key][order_key])
    assert payload["status"] == "SUBMITTED"

    manager.upsert(order_key, {
        "symbol": "NVDA",
        "intent": "OPEN",
        "side": "BUY",
        "status": "FILLED",
        "target_qty": 2,
        "filled_qty": 2,
        "remaining_qty": 0,
        "last_update_ts": 124.0,
        "is_terminal": True,
    })

    assert order_key not in orch.pending_orders
    assert fake_redis.store[redis_key] == {}


def main() -> None:
    test_pending_order_upsert_and_terminal_cleanup()
    print("[OK] order state manager guards passed")


if __name__ == "__main__":
    main()
