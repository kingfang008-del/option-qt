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
import uuid
from typing import Any

import msgpack
import redis

logger = logging.getLogger("maga7.live.redis_fused")

STREAM_FUSED_MARKET = "fused_market_stream"
GROUP_MAG7 = "maga7_scanner_group"
HASH_OPTION_SNAPSHOT = "live_option_snapshot"

def pack_obj(obj: Any) -> bytes:
    """Pack a versioned, code-execution-safe msgpack envelope."""
    return msgpack.packb(
        {"__maga7_wire__": 1, "payload": obj},
        use_bin_type=True,
    )


def pack_batch(batch: list[dict[str, Any]]) -> bytes:
    return pack_obj(batch)


def unpack_obj(raw: bytes | Any) -> Any:
    if raw is None or isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError(f"unsupported Redis payload type: {type(raw).__name__}")
    value = msgpack.unpackb(
        bytes(raw),
        raw=False,
        strict_map_key=False,
    )
    if isinstance(value, dict) and "__maga7_wire__" in value:
        if value.get("__maga7_wire__") != 1 or "payload" not in value:
            raise ValueError("unsupported Mag7 wire schema")
        return value["payload"]
    # Safe compatibility for pre-hardening msgpack records. Pickle is never read.
    return value


def unpack_batch(raw: bytes | Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    obj = unpack_obj(raw)
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
        # RTH authority stream consumed by Scanner/OMS (full-symbol frames only).
        "stream": f"{STREAM_FUSED_MARKET}:maga7:{rid}",
        # Stock-only publisher stream (option_contracts empty); options process ingests.
        "stream_stock": f"{STREAM_FUSED_MARKET}:maga7:{rid}:stock",
        "group_stock": f"{GROUP_MAG7}:{rid}:stock",
        # Pre/post validation streams (partial frames OK; not consumed by OMS).
        "stream_pre": f"{STREAM_FUSED_MARKET}:maga7:{rid}:pre",
        "stream_post": f"{STREAM_FUSED_MARKET}:maga7:{rid}:post",
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
    md_role: str = "combined",
) -> str:
    """Create an isolated stream/group; never mutate the shared FCS bus."""
    run_id = run_id or str(uuid.uuid4())[:8]
    if not reset:
        return run_id
    keys = run_keys(run_id)
    role = str(md_role or "combined").lower()

    delete_keys = []
    if role in {"combined", "stock"}:
        delete_keys.extend(
            [
                keys["stream_stock"],
                keys.get("stream_pre"),
                keys.get("stream_post"),
            ]
        )
    if role in {"combined", "options"}:
        delete_keys.extend(
            [
                keys["stream"],
                keys["option_snapshot"],
                keys["clock"],
                keys["status"],
                keys["ack_ts"],
                keys["ack_frame"],
            ]
        )
        if role == "combined":
            delete_keys.extend([keys.get("stream_pre"), keys.get("stream_post")])
    try:
        r.delete(*[key for key in delete_keys if key])
    except Exception:
        pass

    if role in {"combined", "options"}:
        try:
            r.xgroup_create(keys["stream"], keys["group"], id="$", mkstream=True)
        except Exception:
            r.xgroup_destroy(keys["stream"], keys["group"])
            r.xgroup_create(keys["stream"], keys["group"], id="$", mkstream=True)
        r.set(keys["status"], "INIT")
    if role in {"combined", "stock"}:
        try:
            r.xgroup_create(
                keys["stream_stock"], keys["group_stock"], id="$", mkstream=True
            )
        except Exception:
            try:
                r.xgroup_destroy(keys["stream_stock"], keys["group_stock"])
            except Exception:
                pass
            r.xgroup_create(
                keys["stream_stock"], keys["group_stock"], id="$", mkstream=True
            )

    logger.info(
        "Mag7 Redis init OK | run_id=%s role=%s stream=%s stock=%s",
        run_id,
        role,
        keys["stream"],
        keys["stream_stock"],
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
