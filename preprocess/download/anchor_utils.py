#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1 锚点配置加载 —— profile / JSON 参数化，选约逻辑复用 qqq_btc.qqq.anchor。

目录约定(preprocess 位于项目根目录):
  preprocess/CONFIG/anchor_qqq_0dte.json
  preprocess/CONFIG/anchor_legacy_9dte.json

内置 profile:
  qqq_0dte     → 0/1/2 DTE, 4 bucket (与 qqq_btc 对齐)
  legacy_9dte  → ~9 DTE + 次月, 6 bucket

也可 --config 指定任意 JSON(字段与 qqq_btc/CONFIG/anchor_qqq_0dte.json 同 schema)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_DOWNLOAD_DIR = Path(__file__).resolve().parent
_PREPROCESS_ROOT = _DOWNLOAD_DIR.parent
_REPO_ROOT = _PREPROCESS_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qqq_btc.qqq.anchor import get_daily_locked_contracts, load_anchor_config

# 优先 preprocess/CONFIG，其次 qqq_btc/CONFIG(与训练链一致)
_CONFIG_CANDIDATES = [
    _PREPROCESS_ROOT / "CONFIG",
    _REPO_ROOT / "qqq_btc" / "CONFIG",
    _REPO_ROOT / "production" / "CONFIG",
]


def _config_dir() -> Path:
    for d in _CONFIG_CANDIDATES:
        if (d / "anchor_qqq_0dte.json").exists():
            return d
    return _CONFIG_CANDIDATES[0]


_CONFIG_DIR = _config_dir()

BUILTIN_PROFILES = {
    "qqq_0dte": _CONFIG_DIR / "anchor_qqq_0dte.json",
    "legacy_9dte": _CONFIG_DIR / "anchor_legacy_9dte.json",
}


def resolve_anchor_config(
    profile: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> dict:
    """
    解析锚点配置。优先级: --config > --profile > ANCHOR_CONFIG_PATH >
    OPTION_ANCHOR_PROFILE > 默认 qqq_0dte。
    """
    if config_path:
        return load_anchor_config(Path(config_path))

    prof = profile or os.environ.get("OPTION_ANCHOR_PROFILE")
    if prof:
        if prof not in BUILTIN_PROFILES:
            known = ", ".join(BUILTIN_PROFILES)
            raise ValueError(f"未知 profile {prof!r},可选: {known}")
        path = BUILTIN_PROFILES[prof]
        if not path.exists():
            raise FileNotFoundError(f"profile {prof!r} 配置文件不存在: {path}")
        return load_anchor_config(path)

    env_path = os.environ.get("ANCHOR_CONFIG_PATH")
    if env_path:
        return load_anchor_config(Path(env_path))

    default = BUILTIN_PROFILES["qqq_0dte"]
    if not default.exists():
        raise FileNotFoundError(
            f"默认 0DTE 配置不存在: {default}\n"
            f"请创建 preprocess/CONFIG/anchor_qqq_0dte.json 或使用 --config"
        )
    return load_anchor_config(default)


def resolve_paths(cfg: dict, raw_dir: Optional[str] = None, output: Optional[str] = None) -> tuple[Path, Path]:
    paths = cfg.get("_paths_resolved") or {}
    raw = Path(raw_dir).expanduser() if raw_dir else paths.get("raw_iv_dir")
    out = Path(output).expanduser() if output else paths.get("locked_targets_output")
    if raw is None:
        raw = Path.home() / "train_data/nq_options_day_iv"
    if out is None:
        out = Path.home() / "train_data/locked_targets_map.parquet"
    return Path(raw), Path(out)


def resolve_symbols(cfg: dict, cli_symbols: Optional[str] = None) -> list[str]:
    if cli_symbols:
        return [s.strip().upper() for s in cli_symbols.split(",") if s.strip()]
    cfg_syms = cfg.get("symbols") or []
    if cfg_syms:
        return [str(s).upper() for s in cfg_syms]

    for baseline in (
        _REPO_ROOT / "production" / "baseline",
        _REPO_ROOT / "New_Pro" / "baseline_qqq",
    ):
        try:
            if str(baseline) not in sys.path:
                sys.path.insert(0, str(baseline))
            from config import TARGET_SYMBOLS  # noqa: WPS433

            return list(TARGET_SYMBOLS)
        except ImportError:
            continue
    return ["QQQ"]


__all__ = [
    "BUILTIN_PROFILES",
    "REPO_ROOT",
    "PREPROCESS_ROOT",
    "get_daily_locked_contracts",
    "resolve_anchor_config",
    "resolve_paths",
    "resolve_symbols",
]

REPO_ROOT = _REPO_ROOT
PREPROCESS_ROOT = _PREPROCESS_ROOT
