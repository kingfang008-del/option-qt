#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
旁路演练脚本：把旧主线里的 JSON payload 送进 Domain adapters，再输出转换与 validate 结果。

用法示例：

python3 production/scripts/domain_adapter_dryrun.py \
  --alpha-frame /path/to/alpha_frame.json \
  --quotes /path/to/quotes.json \
  --state /path/to/state.json \
  --strict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = production_dir / "baseline"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(baseline_dir))


def _load_json_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    return json.loads(text)


def _candidate_payloads(raw: Any) -> List[Mapping[str, Any]]:
    candidates: List[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        candidates.append(raw)
        for key in ("payload", "data", "message", "frame", "alpha_frame"):
            nested = raw.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                candidates.append(item)
    return candidates


def _pick_alpha_frame_payload(raw: Any) -> Mapping[str, Any]:
    candidates = _candidate_payloads(raw)
    for payload in candidates:
        if str(payload.get("action", "") or "").upper() == "ALPHA_FRAME":
            return payload
    for payload in candidates:
        if "items" in payload and "ts" in payload:
            return payload
    raise ValueError("no ALPHA_FRAME-like payload found")


def _pick_mapping_payload(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, list):
        return raw
    raise ValueError("quotes/state payload must be a JSON object or array")


def _format_errors(errors: Iterable[str], limit: int = 20) -> str:
    lines = []
    for idx, msg in enumerate(errors):
        if idx >= limit:
            lines.append(f"... truncated after {limit} errors")
            break
        lines.append(f"- {msg}")
    return "\n".join(lines) if lines else "(none)"


def run_dryrun(
    alpha_frame_payload: Mapping[str, Any],
    quotes_payload: Optional[Any] = None,
    state_payload: Optional[Any] = None,
) -> Dict[str, Any]:
    _bootstrap_imports()
    from Domain import (  # noqa: E402
        alpha_frame_from_legacy,
        execution_window_from_legacy,
        position_snapshot_from_legacy_state,
    )

    frame = alpha_frame_from_legacy(alpha_frame_payload)
    frame_errors = frame.validate()

    result: Dict[str, Any] = {
        "frame": frame,
        "frame_errors": frame_errors,
        "window": None,
        "window_errors": [],
        "positions": [],
        "position_errors": [],
    }

    if quotes_payload is not None:
        window = execution_window_from_legacy(alpha_frame_payload, quotes_payload=quotes_payload)
        result["window"] = window
        result["window_errors"] = window.validate()

    if state_payload is not None:
        raw_positions = state_payload if isinstance(state_payload, list) else [state_payload]
        for idx, item in enumerate(raw_positions):
            if not isinstance(item, Mapping):
                result["position_errors"].append(f"state[{idx}] is not an object")
                continue
            pos = position_snapshot_from_legacy_state(item)
            result["positions"].append(pos)
            for msg in pos.validate():
                result["position_errors"].append(f"state[{idx}].{msg}")

    return result


def _print_result(result: Dict[str, Any]) -> None:
    frame = result["frame"]
    window = result["window"]
    positions = result["positions"]
    frame_errors = result["frame_errors"]
    window_errors = result["window_errors"]
    position_errors = result["position_errors"]

    print("== Domain Adapter Dry Run ==")
    print(f"frame_id: {frame.frame_id or '(empty)'}")
    print(f"minute_ts: {frame.minute_ts}")
    print(f"items: {len(frame.items)}")
    print(f"frame validation errors: {len(frame_errors)}")
    if window is not None:
        print(f"window quotes: {len(window.quotes_1s)}")
        print(f"window validation errors: {len(window_errors)}")
    if positions:
        print(f"positions converted: {len(positions)}")
        print(f"position validation errors: {len(position_errors)}")

    print("\n[Frame Errors]")
    print(_format_errors(frame_errors))

    if window is not None:
        print("\n[Window Errors]")
        print(_format_errors(window_errors))

    if positions:
        print("\n[Position Errors]")
        print(_format_errors(position_errors))

    if frame.items:
        sample = frame.items[0]
        print("\n[Sample Item]")
        print(
            json.dumps(
                {
                    "symbol": sample.symbol,
                    "instrument_kind": sample.instrument_traits.instrument_kind.value,
                    "alpha": sample.alpha,
                    "decision_quote_contract_id": (
                        "" if sample.decision_quote is None else sample.decision_quote.contract_id
                    ),
                    "decision_quote_source_kind": (
                        ""
                        if sample.decision_quote is None
                        else sample.decision_quote.source_kind.value
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run legacy OMS payloads through Domain adapters without touching mainline logic."
    )
    parser.add_argument("--alpha-frame", required=True, help="Path to ALPHA_FRAME JSON file.")
    parser.add_argument("--quotes", help="Optional path to execution quote JSON file.")
    parser.add_argument("--state", help="Optional path to position/state JSON file.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation errors are found.",
    )
    args = parser.parse_args(argv)

    alpha_raw = _load_json_file(Path(args.alpha_frame))
    alpha_frame_payload = _pick_alpha_frame_payload(alpha_raw)

    quotes_payload = None
    if args.quotes:
        quotes_payload = _pick_mapping_payload(_load_json_file(Path(args.quotes)))

    state_payload = None
    if args.state:
        state_payload = _pick_mapping_payload(_load_json_file(Path(args.state)))

    result = run_dryrun(alpha_frame_payload, quotes_payload=quotes_payload, state_payload=state_payload)
    _print_result(result)

    total_errors = (
        len(result["frame_errors"])
        + len(result["window_errors"])
        + len(result["position_errors"])
    )
    if args.strict and total_errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
