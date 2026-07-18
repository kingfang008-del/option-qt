"""Mag7 Redis fused-market helpers — same bus as New_Pro IBKR connector.

Reuses:
  - stream name ``fused_market_stream``
  - msgpack batch payload shape ``{symbol, ts, stock, option_contracts?}``
  - ``replay:current_ts`` clock

Does **not** start FCS / TFT / QQQ OMS.
"""
from __future__ import annotations

import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import redis

logger = logging.getLogger("maga7.live.redis_fused")

STREAM_FUSED_MARKET = "fused_market_stream"
GROUP_MAG7 = "maga7_scanner_group"
HASH_OPTION_SNAPSHOT = "live_option_snapshot"

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"


def _ensure_baseline_path() -> None:
    if str(_BASELINE) not in sys.path and _BASELINE.is_dir():
        sys.path.insert(0, str(_BASELINE))


def pack_obj(obj: Any) -> bytes:
    """Pack list or dict via New_Pro msgpack / pickle."""
    _ensure_baseline_path()
    try:
        from utils import serialization_utils as ser  # type: ignore

        return ser.pack(obj)
    except Exception:
        import pickle

        return pickle.dumps(obj)


def pack_batch(batch: list[dict[str, Any]]) -> bytes:
    return pack_obj(batch)


def unpack_obj(raw: bytes | Any) -> Any:
    if raw is None or isinstance(raw, (dict, list)):
        return raw
    _ensure_baseline_path()
    try:
        from utils import serialization_utils as ser  # type: ignore

        return ser.unpack(raw)
    except Exception:
        import pickle

        return pickle.loads(raw)


def unpack_batch(raw: bytes | Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        obj = unpack_obj(raw)
        if isinstance(obj, list):
            return obj
        return [obj] if isinstance(obj, dict) else []
    except Exception:
        import pickle

        obj = pickle.loads(raw)
        if isinstance(obj, list):
            return obj
        return [obj] if isinstance(obj, dict) else []


def redis_client(
    *,
    host: str = "127.0.0.1",
    port: int = 6379,
    db: int = 1,
) -> redis.Redis:
    """db=1 matches New_Pro / qqq_btc replay convention (0=live)."""
    return redis.Redis(host=host, port=port, db=db, decode_responses=False)


def run_keys(run_id: str) -> dict[str, str]:
    """Redis keys isolated to one S5 replay run."""
    rid = str(run_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", rid):
        raise ValueError(f"invalid Mag7 run_id: {rid!r}")
    return {
        "stream": f"{STREAM_FUSED_MARKET}:maga7:{rid}",
        "group": f"{GROUP_MAG7}:{rid}",
        "option_snapshot": f"{HASH_OPTION_SNAPSHOT}:maga7:{rid}",
        "clock": f"replay:current_ts:{rid}",
        "status": f"replay:status:{rid}",
        "ack_ts": f"sync:maga7_done:{rid}",
        "ack_frame": f"sync:maga7_done_frame_id:{rid}",
    }


def init_maga7_redis(
    r: redis.Redis,
    *,
    run_id: str | None = None,
    reset: bool = True,
) -> str:
    """Create an isolated stream/group; never mutate the shared FCS bus."""
    run_id = run_id or str(uuid.uuid4())[:8]
    if not reset:
        return run_id
    keys = run_keys(run_id)

    # A repeated explicit run_id is reset only inside its own namespace.
    try:
        r.delete(
            keys["stream"],
            keys["option_snapshot"],
            keys["clock"],
            keys["status"],
            keys["ack_ts"],
            keys["ack_frame"],
        )
    except Exception:
        pass

    try:
        r.xgroup_create(keys["stream"], keys["group"], id="0-0", mkstream=True)
    except Exception:
        r.xgroup_destroy(keys["stream"], keys["group"])
        r.xgroup_create(keys["stream"], keys["group"], id="0-0", mkstream=True)

    r.set(keys["status"], "INIT")
    logger.info(
        "Mag7 Redis init OK | run_id=%s stream=%s group=%s",
        run_id,
        keys["stream"],
        keys["group"],
    )
    return run_id


def ack_maga7_frame(
    r: redis.Redis,
    *,
    run_id: str,
    ts_val: float,
    frame_id: str,
) -> None:
    """ACK exactly one frame without spoofing FCS/Orch ACK keys."""
    keys = run_keys(run_id)
    pipe = r.pipeline(transaction=True)
    pipe.set(keys["ack_ts"], str(ts_val))
    pipe.set(keys["ack_frame"], frame_id)
    pipe.execute()
