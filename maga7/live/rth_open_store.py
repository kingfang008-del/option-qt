"""Trade-date scoped RTH open persistence (survives mid-day process restarts).

Session ``scanner_state.json`` is tied to ``session_id``. A fresh ``start dry``
creates a new session and previously lost 09:30 opens, letting AM FO latch a
late first bar. This store is keyed only by ``trade_date`` so any later session
can reseeds ``day_open`` before bars arrive.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("maga7.live.rth_open_store")

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


def rth_opens_path(live_root: Path | str, trade_date: str) -> Path:
    return Path(live_root) / str(trade_date) / "rth_opens.json"


def redis_key(trade_date: str) -> str:
    return f"maga7:rth_opens:{trade_date}"


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_rth_opens(live_root: Path | str, trade_date: str) -> dict[str, float]:
    path = rth_opens_path(live_root, trade_date)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to read %s", path)
        return {}
    opens = raw.get("opens") if isinstance(raw, dict) else None
    if not isinstance(opens, dict):
        return {}
    out: dict[str, float] = {}
    for symbol, value in opens.items():
        try:
            px = float(value)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out[str(symbol).upper()] = px
    return out


def load_rth_opens_redis(redis_client: Any, trade_date: str) -> dict[str, float]:
    if redis_client is None:
        return {}
    try:
        raw = redis_client.get(redis_key(trade_date))
    except Exception:
        logger.exception("failed to read rth opens from redis")
        return {}
    if not raw:
        return {}
    try:
        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
        payload = json.loads(text)
    except Exception:
        logger.exception("failed to parse rth opens redis payload")
        return {}
    opens = payload.get("opens") if isinstance(payload, dict) else None
    if not isinstance(opens, dict):
        return {}
    out: dict[str, float] = {}
    for symbol, value in opens.items():
        try:
            px = float(value)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out[str(symbol).upper()] = px
    return out


def merge_rth_opens(*maps: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for mapping in maps:
        for symbol, value in (mapping or {}).items():
            px = float(value or 0.0)
            if px > 0:
                out[str(symbol).upper()] = px
    return out


def _with_file_lock(path: Path, fn: Any) -> Any:
    """Serialize read-modify-write across stock MD / options / hist backfill."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path(path)
    handle = open(lock_file, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return fn()
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def save_rth_opens(
    live_root: Path | str,
    trade_date: str,
    opens: dict[str, float],
    *,
    redis_client: Any | None = None,
    source: str = "",
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    cleaned = {
        str(symbol).upper(): float(value)
        for symbol, value in (opens or {}).items()
        if float(value or 0.0) > 0
    }
    path = rth_opens_path(live_root, trade_date)

    def _write() -> dict[str, Any]:
        existing = load_rth_opens(live_root, trade_date)
        if overwrite:
            merged = merge_rth_opens(existing, cleaned)
        else:
            # Never clobber an already-recorded official open with a later guess.
            merged = dict(existing)
            for symbol, px in cleaned.items():
                if symbol not in merged or float(merged[symbol] or 0.0) <= 0:
                    merged[symbol] = px
        payload = {
            "schema_version": 1,
            "trade_date": str(trade_date),
            "updated_at": time.time(),
            "source": str(source or ""),
            "opens": merged,
            "metadata": metadata or {},
        }
        _atomic_write(path, payload)
        if redis_client is not None:
            try:
                redis_client.set(
                    redis_key(trade_date), json.dumps(payload, ensure_ascii=False)
                )
            except Exception:
                logger.exception("failed to publish rth opens to redis")
        return payload

    return _with_file_lock(path, _write)


def upsert_rth_open(
    live_root: Path | str,
    trade_date: str,
    symbol: str,
    open_px: float,
    *,
    redis_client: Any | None = None,
    source: str = "live_bar",
) -> dict[str, float]:
    px = float(open_px or 0.0)
    if px <= 0:
        return load_rth_opens(live_root, trade_date)
    symbol_u = str(symbol).upper()
    save_rth_opens(
        live_root,
        trade_date,
        {symbol_u: px},
        redis_client=redis_client,
        source=source,
        overwrite=False,
    )
    return load_rth_opens(live_root, trade_date)


def _row_hhmm_ny(row: dict[str, Any]) -> str:
    """Return HH:MM in America/New_York for a tape row."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    if row.get("ts") is not None:
        try:
            dt = datetime.fromtimestamp(float(row["ts"]), tz=ny)
            return dt.strftime("%H:%M")
        except Exception:
            pass
    iso = str(row.get("timestamp") or "").strip()
    if not iso:
        return ""
    try:
        text = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ny)
        else:
            dt = dt.astimezone(ny)
        return dt.strftime("%H:%M")
    except Exception:
        return ""


def recover_rth_opens_from_tapes(
    live_root: Path | str,
    trade_date: str,
    *,
    symbols: list[str] | None = None,
) -> dict[str, float]:
    """Best-effort: scan same-day session tapes for the 09:30 open print."""
    root = Path(live_root) / str(trade_date)
    if not root.is_dir():
        return {}
    want = {str(sym).upper() for sym in (symbols or [])} or None
    found: dict[str, float] = {}
    # Newest sessions first — later sessions may include IB-hist-seeded tapes.
    for session_dir in sorted(root.glob("live_*"), reverse=True):
        tape_root = session_dir / "tape"
        if not tape_root.is_dir():
            continue
        for path in tape_root.rglob("*_*.jsonl"):
            if "options" in path.parts:
                continue
            name = path.name
            if not name.endswith(".jsonl"):
                continue
            symbol = name.split("_", 1)[0].upper()
            if want is not None and symbol not in want:
                continue
            if symbol in found:
                continue
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if _row_hhmm_ny(row) != "09:30":
                            continue
                        px = float(row.get("open") or row.get("close") or 0.0)
                        if px > 0:
                            found[symbol] = px
                            break
            except Exception:
                continue
    return found


def resolve_rth_opens(
    live_root: Path | str,
    trade_date: str,
    *,
    redis_client: Any | None = None,
    symbols: list[str] | None = None,
    recover_tapes: bool = True,
) -> dict[str, float]:
    disk = load_rth_opens(live_root, trade_date)
    redis_opens = load_rth_opens_redis(redis_client, trade_date)
    # Disk/Redis are authoritative. Tape only fills still-missing symbols so a
    # bad late session cannot overwrite an already-recorded official open.
    merged = merge_rth_opens(disk, redis_opens)
    if recover_tapes:
        tape_opens = recover_rth_opens_from_tapes(
            live_root, trade_date, symbols=symbols
        )
        for symbol, px in tape_opens.items():
            if symbol not in merged or float(merged[symbol] or 0.0) <= 0:
                merged[symbol] = px
    if merged and merged != disk:
        save_rth_opens(
            live_root,
            trade_date,
            merged,
            redis_client=redis_client,
            source="resolve_merge",
            overwrite=False,
        )
    return merged


def seed_scanner_day_opens(
    scanner: Any,
    opens: dict[str, float],
    *,
    force: bool = True,
) -> list[str]:
    """Seed StreamSignalState + AM pulse scouts from durable trade-date opens.

    ``force=True`` (default) lets a restored official open replace an empty or
    late pseudo latch so mid-day restarts keep FO anchors consistent.
    """
    seeded: list[str] = []
    states = getattr(scanner, "states", None) or {}
    for symbol, open_px in (opens or {}).items():
        px = float(open_px or 0.0)
        if px <= 0:
            continue
        symbol_u = str(symbol).upper()
        state = states.get(symbol_u)
        if state is not None:
            current = getattr(state, "day_open", None)
            current_px = float(current or 0.0) if current is not None else 0.0
            if current_px <= 0 or (force and abs(current_px - px) > 1e-9):
                state.day_open = px
                seeded.append(symbol_u)
        for lane in ("am_pulse", "am_pulse_extension"):
            scout = getattr(scanner, f"_{lane}_scout", None)
            if scout is not None and hasattr(scout, "seed_day_open"):
                scout.seed_day_open(symbol_u, px, force=force)
        pending = getattr(scanner, "_pending_rth_opens", None)
        if isinstance(pending, dict):
            pending[symbol_u] = px
    cleaned = {
        str(symbol).upper(): float(value)
        for symbol, value in (opens or {}).items()
        if float(value or 0.0) > 0
    }
    if not hasattr(scanner, "_pending_rth_opens"):
        try:
            setattr(scanner, "_pending_rth_opens", dict(cleaned))
        except Exception:
            pass
    else:
        pending = getattr(scanner, "_pending_rth_opens")
        if isinstance(pending, dict):
            pending.update(cleaned)
    return seeded


def missing_rth_open_symbols(
    opens: dict[str, float],
    symbols: list[str] | None,
) -> list[str]:
    have = {
        str(symbol).upper()
        for symbol, value in (opens or {}).items()
        if float(value or 0.0) > 0
    }
    want = [str(symbol).upper() for symbol in (symbols or []) if str(symbol).strip()]
    return [symbol for symbol in want if symbol not in have]
