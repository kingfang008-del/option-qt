#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QQQ 0DTE 官方特征契约：prefer_primary_gapfill。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
CONTRACT_PATH = _REPO / "qqq_btc/CONFIG/feature_contract_0dte_prefer_primary.json"


def load_contract(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONTRACT_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def expand_home(p: str | Path) -> Path:
    return Path(str(p)).expanduser()


def dynamic_map_path(contract: dict[str, Any] | None = None) -> Path:
    c = contract or load_contract()
    return expand_home(c["lock_map"]["path"])


def assert_not_v3_only_map(map_path: Path) -> None:
    """阻止误把 v3 单合约 map 当默认锁约图。"""
    name = map_path.name.lower()
    if "0dte_v3" in name and "dynamic" not in name:
        raise ValueError(
            f"拒绝使用单合约 map 作为 0DTE 默认锁约图: {map_path}. "
            "请使用 locked_targets_map_0dte_dynamic.parquet（prefer_primary）。"
        )
