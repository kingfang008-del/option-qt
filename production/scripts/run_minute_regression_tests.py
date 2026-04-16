#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一键回归脚本：分钟聚合/门控稳定性测试。

优先级：
1) 若环境有 pytest：使用 pytest 运行并输出简洁结果；
2) 若环境无 pytest：回退为逐个 python 执行测试文件。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


TEST_FILES = [
    "production/tests/test_option_minute_aggregation_60s.py",
    "production/tests/test_stock_minute_ohlcv_aggregation_60s.py",
    "production/tests/test_valid_mask_3min_seconds.py",
    "production/tests/test_fcs_parity_snapshot_thresholds.py",
    "production/tests/test_fcs_minute_boundary_fallback.py",
    "production/tests/test_fcs_preagg_mode_guard.py",
    "production/tests/test_fcs_realtime_ts_guard.py",
    "production/tests/test_se_sync_state_scope_guard.py",
    "production/tests/test_orchestrator_execution_guards.py",
    "production/tests/test_ibkr_publish_ts_guard.py",
    "production/tests/test_ibkr_batch_staleness_guard.py",
]


def _run(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    pytest_path = shutil.which("pytest")
    pytest_module_ok = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

    if pytest_path and pytest_module_ok:
        print("[INFO] using pytest runner")
        cmd = [sys.executable, "-m", "pytest", "-q"] + TEST_FILES
        return _run(cmd, repo_root)

    print("[WARN] pytest not found, fallback to python scripts")
    for rel in TEST_FILES:
        print(f"[RUN] {rel}")
        code = _run([sys.executable, rel], repo_root)
        if code != 0:
            print(f"[FAIL] {rel} exited with {code}")
            return code
    print("[OK] all minute regression tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

