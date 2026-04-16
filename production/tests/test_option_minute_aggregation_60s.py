#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单元测试：模拟 1 分钟内 60 条秒级期权数据，验证分钟聚合结果是否正确。

验证点：
1) finalize 返回的分钟快照应采用该分钟最后一条秒级快照语义；
2) volume 列（idx=6）必须被强制重算为 bid_size + ask_size（idx=10+11）；
3) contracts 与 update_ts 应为最后一次 update 的值；
4) 非目标分钟调用 finalize 应返回空结果。
"""

from __future__ import annotations

import sys
import ast
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


def _load_option_minute_aggregator_cls():
    """
    从 feature_compute_service_v8.py 源码中提取 OptionMinuteAggregator 类定义并执行，
    避免测试环境必须安装 torch/redis 等重依赖。
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "feature_compute_service_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    cls_node = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "OptionMinuteAggregator":
            cls_node = node
            break
    if cls_node is None:
        raise RuntimeError("未在 feature_compute_service_v8.py 中找到 OptionMinuteAggregator")

    mini_mod = ast.Module(body=[cls_node], type_ignores=[])
    ast.fix_missing_locations(mini_mod)
    ns = {"np": np}
    exec(compile(mini_mod, filename=str(src_path), mode="exec"), ns)
    return ns["OptionMinuteAggregator"]


def _build_snapshot(sec: int) -> np.ndarray:
    """
    构造一条秒级 6x12 期权快照：
    - 价格/盘口随秒数变化，便于验证“最后一条覆盖”；
    - volume 列故意填错误值，验证聚合器会重算覆盖。
    """
    arr = np.zeros((6, 12), dtype=np.float32)
    base_price = 1.0 + sec * 0.01
    for i in range(6):
        strike = 95.0 + i * 5.0
        bid = base_price + i * 0.02
        ask = bid + 0.03
        bid_sz = float(10 + sec + i)
        ask_sz = float(20 + sec + i)
        arr[i, 0] = 0.0  # 价格列由其他模块决定，这里不做约束
        arr[i, 5] = strike
        arr[i, 6] = -999.0  # 故意写错，验证 update/finalize 会覆盖成 size_sum
        arr[i, 8] = bid
        arr[i, 9] = ask
        arr[i, 10] = bid_sz
        arr[i, 11] = ask_sz
    return arr


def _build_contracts(sec: int) -> list[str]:
    # 每秒制造一个可追踪 contracts，便于验证 finalize 返回最后版本
    suffix = f"{sec:02d}"
    return [
        f"O:NVDA260620P00095000_{suffix}",
        f"O:NVDA260620P00100000_{suffix}",
        f"O:NVDA260620C00100000_{suffix}",
        f"O:NVDA260620C00105000_{suffix}",
        f"O:NVDA260620P00110000_{suffix}",
        f"O:NVDA260620C00115000_{suffix}",
    ]


def test_option_minute_aggregation_60s() -> None:
    _bootstrap_imports()
    from config import NY_TZ  # noqa: E402

    sym = "NVDA"
    OptionMinuteAggregator = _load_option_minute_aggregator_cls()
    agg = OptionMinuteAggregator(rows=6, cols=12)
    minute_dt = pd.Timestamp("2026-04-15 10:02:00", tz=NY_TZ)

    last_snap = None
    last_contracts = None
    last_ts = None

    # 模拟 1 分钟内 60 条秒级输入（10:02:00 ~ 10:02:59）
    for sec in range(60):
        snap = _build_snapshot(sec)
        contracts = _build_contracts(sec)
        ts = float((minute_dt + timedelta(seconds=sec)).timestamp())
        agg.update(sym, minute_dt, snap, contracts, update_ts=ts)
        last_snap = snap
        last_contracts = contracts
        last_ts = ts

    out_snap, out_contracts, out_ts = agg.finalize(sym, minute_dt)
    if out_snap is None:
        raise RuntimeError("finalize 返回空，期望得到分钟快照")

    # 1) 维度
    assert out_snap.shape == (6, 12), f"shape 异常: {out_snap.shape}"

    # 2) contracts / ts 必须是最后一次 update
    assert out_contracts == last_contracts, "contracts 未采用最后一秒版本"
    assert abs(float(out_ts) - float(last_ts)) < 1e-6, "update_ts 未采用最后一秒时间戳"

    # 3) 核心语义：分钟快照应匹配最后一秒（除 volume 强制重算）
    # 先对比不受重算影响的列
    for col in (0, 5, 8, 9, 10, 11):
        lhs = out_snap[:, col]
        rhs = last_snap[:, col]
        if not np.allclose(lhs, rhs, atol=1e-6):
            raise AssertionError(f"列 {col} 与最后一秒不一致")

    # 4) volume 必须等于 bid_size + ask_size
    expected_vol = np.maximum(last_snap[:, 10], 0.0) + np.maximum(last_snap[:, 11], 0.0)
    if not np.allclose(out_snap[:, 6], expected_vol, atol=1e-6):
        raise AssertionError("volume 列未按 bid_size+ask_size 重算")

    # 5) 非目标分钟 finalize 应返回空结果
    wrong_minute = minute_dt + timedelta(minutes=1)
    bad_snap, bad_contracts, bad_ts = agg.finalize(sym, wrong_minute)
    assert bad_snap is None and bad_contracts == [] and bad_ts is None, "跨分钟 finalize 行为异常"

    print("[OK] 60秒 -> 1分钟 聚合测试通过")
    print(f"[INFO] symbol={sym} minute={minute_dt} last_ts={int(last_ts)}")
    print(f"[INFO] row0 volume={out_snap[0,6]:.2f}, bid_size+ask_size={expected_vol[0]:.2f}")


def main() -> None:
    test_option_minute_aggregation_60s()


if __name__ == "__main__":
    main()

