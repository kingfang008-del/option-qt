#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def test_pick_alpha_frame_payload_from_nested_wrapper() -> None:
    _bootstrap_imports()
    from scripts.domain_adapter_dryrun import _pick_alpha_frame_payload  # noqa: E402

    raw = {
        "payload": {
            "action": "ALPHA_FRAME",
            "ts": 100,
            "frame_id": "f1",
            "items": [],
        }
    }
    payload = _pick_alpha_frame_payload(raw)

    assert payload["frame_id"] == "f1"


def test_run_dryrun_with_quotes_and_state() -> None:
    _bootstrap_imports()
    from scripts.domain_adapter_dryrun import run_dryrun  # noqa: E402

    alpha_frame_payload = {
        "action": "ALPHA_FRAME",
        "source": "alpha_engine_v8",
        "ts": 1_710_000_060,
        "frame_id": "frame-9",
        "items": [
            {
                "symbol": "NVDA",
                "batch_idx": 0,
                "stock_price": 900.0,
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
    quotes_payload = {
        "NVDA": {
            "ts": 1_710_000_061.0,
            "call_price": 10.1,
            "call_bid": 10.0,
            "call_ask": 10.2,
        }
    }
    state_payload = [
        {
            "symbol": "NVDA",
            "position": 1,
            "qty": 2,
            "entry_price": 9.5,
            "entry_ts": 1_710_000_065.0,
            "contract_id": "NVDA_CALL",
            "opt_type": "call",
            "strike_price": 900,
            "last_valid_iv": 0.4,
        }
    ]

    result = run_dryrun(alpha_frame_payload, quotes_payload=quotes_payload, state_payload=state_payload)

    assert result["frame_errors"] == []
    assert result["window_errors"] == []
    assert result["position_errors"] == []
    assert result["frame"].frame_id == "frame-9"
    assert result["window"] is not None
    assert len(result["positions"]) == 1


def test_main_returns_zero_for_valid_payload() -> None:
    _bootstrap_imports()
    from scripts.domain_adapter_dryrun import main  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp_dir:
        alpha_path = Path(tmp_dir) / "alpha.json"
        alpha_path.write_text(
            json.dumps(
                {
                    "action": "ALPHA_FRAME",
                    "ts": 1_710_000_060,
                    "frame_id": "frame-11",
                    "items": [],
                }
            ),
            encoding="utf-8",
        )

        code = main(["--alpha-frame", str(alpha_path), "--strict"])

        assert code == 0
