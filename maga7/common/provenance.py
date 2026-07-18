"""Deterministic strategy/live code fingerprints for evidence gating."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STRATEGY_FILES = (
    "maga7/common/provenance.py",
    "maga7/common/bar_agg.py",
    "maga7/common/config.py",
    "maga7/common/contract_select.py",
    "maga7/common/entry_contract.py",
    "maga7/common/fills.py",
    "maga7/common/open_lock.py",
    "maga7/common/position_size.py",
    "maga7/common/reentry.py",
    "maga7/common/regime.py",
    "maga7/common/replay.py",
    "maga7/common/signals.py",
    "maga7/common/stream_engine.py",
    "maga7/live/oms_fill_session.py",
    "maga7/live/oms_stub.py",
    "maga7/live/redis_consumer.py",
    "maga7/live/redis_fused.py",
    "maga7/live/redis_pitcher.py",
    "maga7/live/redis_quotes.py",
    "maga7/live/scanner.py",
    "maga7/tools/run_maga7_redis_sim.py",
    "maga7/tools/run_stream_parity.py",
)

LIVE_FILES = STRATEGY_FILES + (
    "maga7/live/broker_oms.py",
    "maga7/live/ibkr_connector.py",
    "maga7/live/live_contract_lock.py",
    "maga7/live/live_engine.py",
    "maga7/live/live_regime.py",
    "maga7/live/scanner_state.py",
    "maga7/tools/run_live_session.py",
)


def code_fingerprint(profile_path: str | Path, *, live: bool = False) -> str:
    digest = hashlib.sha256()
    profile = Path(profile_path).expanduser().resolve()
    files = LIVE_FILES if live else STRATEGY_FILES
    for path in (profile, *(REPO / relative for relative in files)):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            label = str(path.relative_to(REPO))
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
