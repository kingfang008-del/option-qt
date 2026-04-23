#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import types
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    history_replay_dir = production_dir / "history_replay"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(history_replay_dir))
    sys.path.insert(0, str(dao_dir))


class _FakeRedis:
    def __init__(self, initial_owner: str | None = None) -> None:
        self.store = {}
        self.ttls = {}
        if initial_owner is not None:
            self.store["lock:oms_writer:REALTIME_DRY:db0"] = initial_owner
            self.ttls["lock:oms_writer:REALTIME_DRY:db0"] = 42

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.ttls[key] = int(ex or 0)
        return True

    def get(self, key):
        return self.store.get(key)

    def ttl(self, key):
        return self.ttls.get(key, -2)

    def expire(self, key, ttl):
        if key in self.store:
            self.ttls[key] = int(ttl)
            return True
        return False

    def eval(self, _script, _num_keys, key, expected_owner):
        if self.store.get(key) == expected_owner:
            del self.store[key]
            self.ttls.pop(key, None)
            return 1
        return 0


def _build_lock_harness(initial_owner: str, owner_alive: bool):
    sys.modules.setdefault("torch", types.ModuleType("torch"))

    ibkr_stub = types.ModuleType("ibkr_connector_v8")
    ibkr_stub.IBKRConnectorFinal = object
    sys.modules["ibkr_connector_v8"] = ibkr_stub

    with patch.object(logging, "FileHandler", lambda *_args, **_kwargs: logging.NullHandler()):
        import execution_engine_v8 as ee  # noqa: E402

    engine = SimpleNamespace(
        mode="realtime",
        use_shared_mem=False,
        r=_FakeRedis(initial_owner),
        _oms_writer_lock_key="lock:oms_writer:REALTIME_DRY:db0",
        _oms_writer_lock_value=f"currentboot:{os.getpid()}",
        _oms_writer_lock_acquired=False,
        _last_writer_lock_refresh_ts=0.0,
    )
    engine._parse_oms_writer_lock_owner = ee.ExecutionEngineV8._parse_oms_writer_lock_owner
    engine._is_local_pid_alive = lambda _pid: owner_alive
    engine._delete_oms_writer_lock_if_value_matches = MethodType(
        ee.ExecutionEngineV8._delete_oms_writer_lock_if_value_matches,
        engine,
    )
    engine._acquire_oms_writer_lock = MethodType(
        ee.ExecutionEngineV8._acquire_oms_writer_lock,
        engine,
    )
    return engine


def test_oms_writer_lock_reclaims_dead_pid_owner() -> None:
    _bootstrap_imports()

    engine = _build_lock_harness("deadboot:999999", owner_alive=False)
    with patch.dict(os.environ, {"ALLOW_MULTIPLE_OMS": ""}, clear=False):
        assert engine._acquire_oms_writer_lock(ttl_sec=60) is True

    assert engine._oms_writer_lock_acquired is True
    assert engine.r.get(engine._oms_writer_lock_key) == engine._oms_writer_lock_value


def test_oms_writer_lock_blocks_live_pid_owner() -> None:
    _bootstrap_imports()

    engine = _build_lock_harness("liveboot:12345", owner_alive=True)
    with patch.dict(os.environ, {"ALLOW_MULTIPLE_OMS": ""}, clear=False):
        assert engine._acquire_oms_writer_lock(ttl_sec=60) is False

    assert engine._oms_writer_lock_acquired is False
    assert engine.r.get(engine._oms_writer_lock_key) == "liveboot:12345"
