from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import redis

from config import LIVE_TRADING_CAPITAL_LIMIT, REDIS_CFG, RUN_MODE, TRADING_ENABLED


RUNTIME_TRADING_CONTROLS_KEY = "meta:runtime_trading_controls"


def _redis_client():
    return redis.Redis(
        host=REDIS_CFG["host"],
        port=REDIS_CFG["port"],
        db=REDIS_CFG["db"],
        decode_responses=False,
    )


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value
    return value


def read_runtime_trading_controls(r=None) -> Dict[str, Any]:
    client = r or _redis_client()
    raw = client.hgetall(RUNTIME_TRADING_CONTROLS_KEY) or {}
    return {_decode(k): _decode(v) for k, v in raw.items()}


def get_runtime_live_trading_capital_limit(
    default_value: Optional[float] = None,
    r=None,
    return_meta: bool = False,
) -> Any:
    default_cap = float(LIVE_TRADING_CAPITAL_LIMIT if default_value is None else default_value)
    meta = {
        "source": "config",
        "updated_at": None,
        "updated_by": None,
        "run_mode": RUN_MODE,
    }
    try:
        mapping = read_runtime_trading_controls(r=r)
        raw = mapping.get("live_trading_capital_limit")
        if raw is not None and str(raw).strip() != "":
            cap = max(0.0, float(raw))
            meta.update({
                "source": "redis",
                "updated_at": mapping.get("updated_at"),
                "updated_by": mapping.get("updated_by"),
                "run_mode": mapping.get("run_mode") or RUN_MODE,
            })
            return (cap, meta) if return_meta else cap
    except Exception:
        pass
    return (default_cap, meta) if return_meta else default_cap


def set_runtime_live_trading_capital_limit(value: float, r=None, source: str = "dashboard") -> float:
    client = r or _redis_client()
    cap = max(0.0, float(value or 0.0))
    client.hset(
        RUNTIME_TRADING_CONTROLS_KEY,
        mapping={
            "live_trading_capital_limit": f"{cap:.8f}",
            "updated_at": f"{time.time():.3f}",
            "updated_by": str(source or "dashboard"),
            "run_mode": RUN_MODE,
        },
    )
    client.expire(RUNTIME_TRADING_CONTROLS_KEY, 30 * 24 * 3600)
    return cap


def clear_runtime_live_trading_capital_limit(r=None, source: str = "dashboard_reset") -> None:
    client = r or _redis_client()
    client.hdel(RUNTIME_TRADING_CONTROLS_KEY, "live_trading_capital_limit")
    client.hset(
        RUNTIME_TRADING_CONTROLS_KEY,
        mapping={
            "updated_at": f"{time.time():.3f}",
            "updated_by": str(source or "dashboard_reset"),
            "run_mode": RUN_MODE,
        },
    )
    client.expire(RUNTIME_TRADING_CONTROLS_KEY, 30 * 24 * 3600)


def get_runtime_trading_enabled(
    default_value: Optional[bool] = None,
    r=None,
    return_meta: bool = False,
) -> Any:
    default_enabled = bool(TRADING_ENABLED if default_value is None else default_value)
    meta = {
        "source": "config",
        "updated_at": None,
        "updated_by": None,
        "run_mode": RUN_MODE,
    }
    # Safety rule: runtime override can never force-enable a mode that config already disabled.
    if not default_enabled:
        return (False, meta) if return_meta else False
    try:
        mapping = read_runtime_trading_controls(r=r)
        raw = mapping.get("trading_enabled")
        if raw is not None and str(raw).strip() != "":
            enabled = str(raw).strip().lower() in {"1", "true", "yes", "on"}
            meta.update({
                "source": "redis",
                "updated_at": mapping.get("updated_at"),
                "updated_by": mapping.get("updated_by"),
                "run_mode": mapping.get("run_mode") or RUN_MODE,
            })
            return (enabled, meta) if return_meta else enabled
    except Exception:
        pass
    return (default_enabled, meta) if return_meta else default_enabled


def set_runtime_trading_enabled(value: bool, r=None, source: str = "dashboard") -> bool:
    client = r or _redis_client()
    enabled = bool(value)
    client.hset(
        RUNTIME_TRADING_CONTROLS_KEY,
        mapping={
            "trading_enabled": "1" if enabled else "0",
            "updated_at": f"{time.time():.3f}",
            "updated_by": str(source or "dashboard"),
            "run_mode": RUN_MODE,
        },
    )
    client.expire(RUNTIME_TRADING_CONTROLS_KEY, 30 * 24 * 3600)
    return enabled


def clear_runtime_trading_enabled(r=None, source: str = "dashboard_reset") -> None:
    client = r or _redis_client()
    client.hdel(RUNTIME_TRADING_CONTROLS_KEY, "trading_enabled")
    client.hset(
        RUNTIME_TRADING_CONTROLS_KEY,
        mapping={
            "updated_at": f"{time.time():.3f}",
            "updated_by": str(source or "dashboard_reset"),
            "run_mode": RUN_MODE,
        },
    )
    client.expire(RUNTIME_TRADING_CONTROLS_KEY, 30 * 24 * 3600)
