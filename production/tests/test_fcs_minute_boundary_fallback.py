#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回归测试：FCS 分钟触发不应依赖“恰好 :00 秒到达”。
"""

from __future__ import annotations

import sys
import ast
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))
    sys.path.insert(0, str(dao_dir))


def _build_hist(ts: float) -> pd.DataFrame:
    from config import NY_TZ  # noqa: E402

    idx = pd.date_range(
        end=pd.Timestamp(ts, unit="s", tz=NY_TZ).floor("1min"),
        periods=40,
        freq="1min",
    )
    return pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)


def _load_align_inference_timing():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "production" / "baseline" / "DAO" / "feature_compute_service_v8.py"
    src = src_path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(src_path))
    target = None
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "FeatureComputeService":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_align_inference_timing":
                    target = item
                    break
            break
    if target is None:
        raise RuntimeError("未找到 FeatureComputeService._align_inference_timing")
    mini = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {
        "datetime": datetime,
        "NY_TZ": None,  # 会在测试内注入真实值
        "time": time,
        "logger": logging.getLogger("test_fcs_minute_boundary_fallback"),
    }
    exec(compile(mini, filename=str(src_path), mode="exec"), ns)
    return ns


def test_fcs_minute_boundary_fallback() -> None:
    _bootstrap_imports()
    from fcs_market_profile import build_market_profile  # noqa: E402
    from config import NY_TZ  # noqa: E402

    ns = _load_align_inference_timing()
    ns["NY_TZ"] = NY_TZ

    svc = type("S", (), {})()
    svc._align_inference_timing = ns["_align_inference_timing"].__get__(svc, svc.__class__)
    svc.symbols = ["NVDA"]
    svc.history_1min = {"NVDA": _build_hist(1776261720.0)}
    svc.market_profile = build_market_profile(
        "equity_us",
        ny_tz=NY_TZ,
        warmup_required_len=31,
        non_tradable_symbols=(),
    )
    svc.last_model_minute_ts = 0

    # 首帧（非整分）：只建立锚点，不触发分钟事件。
    first = svc._align_inference_timing(1776261725.0)
    assert first is None, "首帧应仅建锚点"
    anchor = int(svc.last_model_minute_ts)
    assert anchor == (int(1776261725.0) // 60) * 60

    # 同一分钟内后续帧：不应重复触发。
    second = svc._align_inference_timing(1776261735.0)
    assert second is not None
    _, _, is_new_minute_second, _, _ = second
    assert is_new_minute_second is False

    # 下一分钟的首个 tick 不是 :00（例如 :02），也必须触发分钟事件。
    third = svc._align_inference_timing(1776261782.0)  # xx:03:02
    assert third is not None
    alpha_label_ts, _, is_new_minute_third, _, _ = third
    assert is_new_minute_third is True, "分钟跨越应触发 is_new_minute=True"
    assert int(alpha_label_ts) == ((int(1776261782.0) // 60) * 60 - 60)

    print("[OK] minute boundary fallback works without exact :00 tick")


def main() -> None:
    test_fcs_minute_boundary_fallback()


if __name__ == "__main__":
    main()
