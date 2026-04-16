#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：IBKR 连接器发布时间戳不应超前 wall-clock。
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from unittest.mock import patch


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


def _load_sanitize_publish_ts():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "ibkr_connector_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    target = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "IBKRConnectorFinal":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_sanitize_publish_ts":
                    target = item
                    break
            break
    if target is None:
        raise RuntimeError("未找到 IBKRConnectorFinal._sanitize_publish_ts")
    mini = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {"time": time, "logger": _LoggerStub()}
    exec(compile(mini, filename=str(src_path), mode="exec"), ns)
    return ns["_sanitize_publish_ts"]


def test_ibkr_publish_ts_guard() -> None:
    _bootstrap_imports()
    fn = _load_sanitize_publish_ts()

    svc = type("S", (), {})()
    svc._max_ts_lead_sec = 1.0
    bound = fn.__get__(svc, svc.__class__)

    wall_ts = 1000.0
    with patch("time.time", return_value=wall_ts):
        out = float(bound(1012.0, source_tag="unit_test"))
    assert abs(out - wall_ts) < 1e-9, "超前时间戳应被钳制到 wall-clock"

    with patch("time.time", return_value=wall_ts):
        out2 = float(bound(1000.5, source_tag="unit_test"))
    assert abs(out2 - 1000.5) < 1e-9, "正常时间戳不应被改写"

    print("[OK] ibkr publish ts guard passed")


def main() -> None:
    test_ibkr_publish_ts_guard()


if __name__ == "__main__":
    main()
