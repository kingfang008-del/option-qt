#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：_sync_state_from_oms 不应再触发 IS_SIMULATED 作用域错误。
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def _load_sync_state_func():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "signal_engine_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    target = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "SignalEngineV8":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_sync_state_from_oms":
                    target = item
                    break
            break
    if target is None:
        raise RuntimeError("未找到 SignalEngineV8._sync_state_from_oms")

    mini = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {"IS_SIMULATED": False, "IS_BACKTEST": False, "os": os, "logger": _LoggerStub()}
    exec(compile(mini, filename=str(src_path), mode="exec"), ns)
    return ns["_sync_state_from_oms"]


class _LoggerStub:
    def debug(self, *args, **kwargs):
        return None


class _RedisStub:
    def hgetall(self, key):
        # 返回空账本，触发 pending BUY 的阈值分支。
        return {}


class _StateStub:
    def __init__(self):
        self.position = 0
        self.is_pending = True
        self.pending_action = "BUY"
        self._pending_frames = 0
        self.entry_price = 1.0
        self.max_roi = 0.0


def test_se_sync_state_scope_guard() -> None:
    _bootstrap_imports()
    fn = _load_sync_state_func()

    svc = type("S", (), {})()
    svc.use_shared_mem = False
    svc.r = _RedisStub()
    svc.states = {"NVDA": _StateStub()}

    # 旧实现会在这里抛 UnboundLocalError；新实现不应抛异常。
    bound = fn.__get__(svc, svc.__class__)
    bound()

    st = svc.states["NVDA"]
    assert st._pending_frames == 1, "pending 帧计数应正常递增"
    assert st.is_pending is True, "首轮容忍期内不应立即解锁"
    print("[OK] _sync_state_from_oms scope guard passed")


def main() -> None:
    test_se_sync_state_scope_guard()


if __name__ == "__main__":
    main()
