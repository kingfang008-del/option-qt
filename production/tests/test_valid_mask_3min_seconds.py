#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单元测试：连续 3 分钟 * 60 秒，验证 valid_mask/gate 在秒级不会被重新触发抖动。

目标：
1) 仅在 is_new_minute=True 时更新 gate 状态；
2) 秒级帧复用上一分钟 ready，不因秒级缺失快照而改变 valid；
3) 复现一个 3 分钟状态序列并断言其稳定性。
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(dao_dir))


def _load_method_from_fcs(method_name: str):
    """
    从 feature_compute_service_v8.py 中提取 FeatureComputeService 的目标方法。
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "feature_compute_service_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    target_fn = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureComputeService":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    target_fn = sub
                    break
    if target_fn is None:
        raise RuntimeError(f"未找到方法: {method_name}")

    mini_mod = ast.Module(body=[target_fn], type_ignores=[])
    ast.fix_missing_locations(mini_mod)
    ns = {}
    exec(compile(mini_mod, filename=str(src_path), mode="exec"), ns)
    return ns[method_name]


class _Svc:
    pass


def test_valid_mask_3min_seconds() -> None:
    _bootstrap_imports()
    from fcs_support_handler import FCSSupportHandler  # noqa: E402

    resolve_gate_status = _load_method_from_fcs("_resolve_gate_status")

    svc = _Svc()
    svc.option_gate_state = {"NVDA": {"pass_streak": 0, "fail_streak": 0, "ready": False, "grace_until_ts": None}}
    svc.option_gate_min_pass = 2
    svc.option_gate_max_fail = 1
    svc.option_gate_grace_minutes = 0
    svc.option_gate_require_frame_consistency = True

    # 绑定 support handler 的真实状态机逻辑
    support = FCSSupportHandler(svc)
    svc._update_option_gate_state = types.MethodType(
        lambda self, sym, snapshot_ok, frame_ok, is_new_minute, gate_minute_ts=None: support.update_option_gate_state(
            sym=sym,
            snapshot_ok=snapshot_ok,
            frame_ok=frame_ok,
            is_new_minute=is_new_minute,
            gate_minute_ts=gate_minute_ts,
        ),
        svc,
    )

    # frame 一致性在本测试中恒为 True（关注点是秒级不重判）
    svc._is_option_frame_consistent = types.MethodType(lambda self, sym, gate_minute_ts: True, svc)
    # snapshot_ok 由输入 candidate_buckets["snapshot_ok"] 决定
    svc._is_option_snapshot_complete = types.MethodType(
        lambda self, candidate_buckets, candidate_contracts=None, min_iv=None: bool(
            isinstance(candidate_buckets, dict) and candidate_buckets.get("snapshot_ok", False)
        ),
        svc,
    )
    svc._resolve_gate_status = types.MethodType(resolve_gate_status, svc)

    sym = "NVDA"
    start_ts = 1776260400  # 09:40:00（示意）

    # 分钟边界快照质量计划：
    # M0: True -> pass_streak=1, ready=False, allow=False
    # M1: True -> pass_streak=2, ready=True,  allow=True
    # M2: False -> fail_streak=1, ready=False, allow=False
    minute_boundary_plan = {0: True, 1: True, 2: False}

    allows = []
    ready_series = []

    for sec in range(180):
        minute_idx = sec // 60
        is_new_minute = (sec % 60 == 0)
        gate_ts = float(start_ts + sec)

        if is_new_minute:
            marker = {"snapshot_ok": minute_boundary_plan[minute_idx]}
        else:
            # 秒级帧故意传“坏输入”（无快照），期望不影响 allow（只复用 ready）
            marker = None

        _, _, allow, gate_state = svc._resolve_gate_status(
            sym,
            marker,
            [],
            is_new_minute=is_new_minute,
            gate_minute_ts=gate_ts,
        )
        allows.append(bool(allow))
        ready_series.append(bool(gate_state.get("ready", False)))

    # 断言分钟边界状态
    assert allows[0] is False, "M0 边界应为 False（仅第1次通过，尚未 ready）"
    assert allows[60] is True, "M1 边界应为 True（第2次通过进入 ready）"
    assert allows[120] is False, "M2 边界应为 False（边界失败触发降级）"

    # 断言秒级稳定性：每个分钟内部，allow 恒定不抖动
    assert len(set(allows[1:60])) == 1 and allows[1] is False, "M0 秒级帧不应抖动"
    assert len(set(allows[61:120])) == 1 and allows[61] is True, "M1 秒级帧不应抖动"
    assert len(set(allows[121:180])) == 1 and allows[121] is False, "M2 秒级帧不应抖动"

    print("[OK] 3分钟*60秒 valid_mask/gate 稳定性测试通过")
    print(
        "[INFO] boundary_allows="
        f"{[allows[0], allows[60], allows[120]]} "
        f"boundary_ready={ [ready_series[0], ready_series[60], ready_series[120]] }"
    )


def main() -> None:
    test_valid_mask_3min_seconds()


if __name__ == "__main__":
    main()

