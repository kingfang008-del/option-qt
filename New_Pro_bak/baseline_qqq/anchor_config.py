"""baseline_qqq 实盘锚点配置加载（对齐 New_Pro/CONFIG/anchor_qqq_0dte.json）。"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

_NEW_PRO_ROOT = Path(__file__).resolve().parent.parent
_PREPROCESS = _NEW_PRO_ROOT / "preprocess"
if str(_PREPROCESS) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS))

from anchor_contract_utils import (  # noqa: E402
    active_bucket_specs,
    load_anchor_config,
    select_front_expiration,
)

_DEFAULT_ANCHOR_PATH = _NEW_PRO_ROOT / "CONFIG" / "anchor_qqq_0dte.json"


@lru_cache(maxsize=1)
def get_anchor_config() -> dict:
    env_path = os.environ.get("ANCHOR_CONFIG_PATH")
    path = Path(env_path) if env_path else _DEFAULT_ANCHOR_PATH
    return load_anchor_config(path)


def anchor_profile_active() -> bool:
    profile = os.environ.get("OPTION_ANCHOR_PROFILE", "qqq_0dte").strip().lower()
    return profile in {
        "qqq_0dte", "qqq_hybrid", "0dte", "hybrid", "1", "true", "yes", "on",
    }


def get_live_bucket_specs() -> Dict[str, dict]:
    if not anchor_profile_active():
        from config import BUCKET_SPECS

        return dict(BUCKET_SPECS)
    return active_bucket_specs(get_anchor_config())


def pick_front_expiration(exp_dtes) -> Optional[Tuple[str, int]]:
    return select_front_expiration(exp_dtes, get_anchor_config())


def min_front_dte_for_restore() -> int:
    """PG 恢复锁时 front 合约允许的最小 DTE（0DTE profile = 0）。"""
    if not anchor_profile_active():
        return 5
    return int(get_anchor_config().get("front_min_dte", 0))


def get_required_lock_tags() -> list:
    """当前 profile 应订阅/持久化的 bucket tag 列表。"""
    return list(get_live_bucket_specs().keys())


def use_next_buckets() -> bool:
    if not anchor_profile_active():
        return True
    return bool(get_anchor_config().get("use_next_buckets", False))
