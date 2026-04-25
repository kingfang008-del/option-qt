#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def test_shadow_router_disabled_is_noop() -> None:
    _bootstrap_imports()
    from Domain.shadow_router import DomainShadowRouter  # noqa: E402

    router = DomainShadowRouter(enabled=False)
    router.on_alpha_frame({"action": "ALPHA_FRAME", "ts": 1, "items": []})
    router.on_execution_quote("NVDA", {"ts": 1.0, "call_price": 1.0})
    router.on_state_snapshot({"NVDA": {"symbol": "NVDA", "position": 1, "qty": 1, "entry_price": 1.0, "entry_ts": 1.0}})

    stats = router.stats()
    assert stats["alpha_frame"]["ok"] == 0 and stats["alpha_frame"]["error"] == 0
    assert stats["execution_quote"]["ok"] == 0 and stats["execution_quote"]["error"] == 0
    assert stats["position_state"]["ok"] == 0 and stats["position_state"]["error"] == 0


def test_shadow_router_collects_ok_counts_and_dumps_samples() -> None:
    _bootstrap_imports()
    from Domain.shadow_router import DomainShadowRouter  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp_dir:
        router = DomainShadowRouter(
            enabled=True,
            dump_dir=tmp_dir,
            dump_payloads=True,
            ok_log_every=1,
        )
        router.on_alpha_frame(
            {
                "action": "ALPHA_FRAME",
                "source": "alpha_engine_v8",
                "ts": 1_710_000_060,
                "frame_id": "frame-shadow",
                "items": [
                    {
                        "symbol": "NVDA",
                        "alpha": 1.0,
                        "alpha_label_ts": 1_710_000_000,
                        "alpha_available_ts": 1_710_000_060,
                        "opt_data": {
                            "has_feed": True,
                            "call_price": 10.0,
                            "call_bid": 9.9,
                            "call_ask": 10.1,
                            "call_id": "NVDA_CALL",
                            "ts": 1_710_000_059.0,
                        },
                    }
                ],
            }
        )
        router.on_execution_quote(
            "NVDA",
            {
                "ts": 1_710_000_061.0,
                "call_price": 10.1,
                "call_bid": 10.0,
                "call_ask": 10.2,
            },
            legacy_position=1,
        )
        router.on_state_snapshot(
            {
                "NVDA": {
                    "symbol": "NVDA",
                    "position": 1,
                    "qty": 2,
                    "entry_price": 9.5,
                    "entry_ts": 1_710_000_065.0,
                    "contract_id": "NVDA_CALL",
                    "opt_type": "call",
                    "strike_price": 900,
                }
            },
            namespace="test_ns",
            run_mode="REALTIME_DRY",
        )

        stats = router.stats()
        assert stats["alpha_frame"]["ok"] == 1
        assert stats["execution_quote"]["ok"] == 1
        assert stats["position_state"]["ok"] == 1

        dump_root = Path(tmp_dir)
        assert any((dump_root / "alpha_frame").iterdir())
        assert any((dump_root / "execution_quote").iterdir())
        assert any((dump_root / "position_state").iterdir())


def test_shadow_router_records_invalid_payloads_as_errors() -> None:
    _bootstrap_imports()
    from Domain.shadow_router import DomainShadowRouter  # noqa: E402

    router = DomainShadowRouter(enabled=True, dump_payloads=False, ok_log_every=1)
    router.on_execution_quote(
        "NVDA",
        {
            "ts": 1_710_000_061.0,
            "call_bid": 10.3,
            "call_ask": 10.1,
            "call_price": 10.2,
        },
        legacy_position=1,
    )

    stats = router.stats()
    assert stats["execution_quote"]["error"] == 1
