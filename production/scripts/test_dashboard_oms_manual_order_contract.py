#!/usr/bin/env python3
"""Dry-run validator for dashboard manual OMS order payloads.

Default mode does not write to Redis. Use --send --confirm SEND_TO_OMS only
when you intentionally want OMS to consume the generated test signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from config import REDIS_CFG, RUN_MODE, STREAM_ORCH_SIGNAL  # noqa: E402
from utils import serialization_utils as ser  # noqa: E402


def _direction_from_tag(tag: str) -> int:
    return 1 if str(tag or "").upper().startswith("CALL") else -1


def build_manual_open_payload(
    *,
    symbol: str,
    tag: str,
    qty: int,
    price: float,
    stock_price: float,
    contract_id: str,
) -> Dict[str, Any]:
    symbol = symbol.strip().upper()
    tag = tag.strip().upper()
    now_ts = time.time()
    request_id = f"dashboard-open-test-{symbol}-{tag}-{int(now_ts * 1000)}"
    direction = _direction_from_tag(tag)
    return {
        "source": "dashboard_manual_open",
        "action": "BUY",
        "ts": now_ts,
        "symbol": symbol,
        "stock_price": float(stock_price or 0.0),
        "batch_idx": -1,
        "frame_id": request_id,
        "sig": {
            "action": "BUY",
            "dir": direction,
            "tag": tag,
            "price": max(float(price), 0.01),
            "market_price": max(float(price), 0.01),
            "bid": max(float(price) - 0.01, 0.01),
            "ask": max(float(price) + 0.01, 0.01),
            "bid_size": float(qty),
            "ask_size": float(qty),
            "reason": f"DASHBOARD_MANUAL_OPEN:{request_id}",
            "meta": {
                "manual_open": True,
                "request_id": request_id,
                "requested_qty": int(qty),
                "contract_id": str(contract_id or ""),
                "strike": 0.0,
                "iv": 0.25,
                "bid": max(float(price) - 0.01, 0.01),
                "ask": max(float(price) + 0.01, 0.01),
                "bid_size": float(qty),
                "ask_size": float(qty),
                "alpha_z": float(direction),
                "spy_roc": 0.0,
                "index_trend": direction,
            },
        },
    }


def build_manual_close_payload(
    *,
    symbol: str,
    position: int,
    qty: float,
    price: float,
    stock_price: float,
    contract_id: str,
) -> Dict[str, Any]:
    symbol = symbol.strip().upper()
    now_ts = time.time()
    request_id = f"dashboard-close-test-{symbol}-{int(now_ts * 1000)}"
    return {
        "source": "dashboard_manual_close",
        "action": "SELL",
        "ts": now_ts,
        "symbol": symbol,
        "stock_price": float(stock_price or 0.0),
        "batch_idx": -1,
        "frame_id": request_id,
        "sig": {
            "action": "SELL",
            "dir": int(position),
            "target_side": int(position),
            "original_position": int(position),
            "price": max(float(price), 0.01),
            "market_price": max(float(price), 0.01),
            "bid": 0.0,
            "ask": 0.0,
            "bid_size": float(qty),
            "ask_size": float(qty),
            "reason": f"DASHBOARD_FORCE_CLOSE:{request_id}",
            "meta": {
                "contract_id": str(contract_id or ""),
                "manual_close": True,
                "request_id": request_id,
            },
        },
    }


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    action = str(payload.get("action", "")).upper()
    source = str(payload.get("source", ""))
    sig = payload.get("sig")
    if action not in {"BUY", "SELL"}:
        errors.append("payload.action must be BUY or SELL")
    if source not in {"dashboard_manual_open", "dashboard_manual_close"}:
        errors.append("payload.source must be dashboard_manual_open/dashboard_manual_close")
    if not payload.get("symbol"):
        errors.append("payload.symbol is required")
    if not payload.get("frame_id"):
        errors.append("payload.frame_id/request id is required")
    if not isinstance(sig, dict):
        errors.append("payload.sig must be a dict")
        return errors
    if str(sig.get("action", "")).upper() != action:
        errors.append("sig.action must match payload.action")
    if float(sig.get("price", 0.0) or 0.0) <= 0.01:
        errors.append("sig.price must be > 0.01")
    if action == "BUY":
        meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        if source != "dashboard_manual_open":
            errors.append("BUY source must be dashboard_manual_open")
        if not meta.get("manual_open"):
            errors.append("BUY meta.manual_open must be true")
        if int(meta.get("requested_qty", 0) or 0) <= 0:
            errors.append("BUY meta.requested_qty must be > 0")
        if str(sig.get("tag", "")).upper() not in {"CALL_ATM", "CALL_OTM", "PUT_ATM", "PUT_OTM"}:
            errors.append("BUY sig.tag must be supported dashboard tag")
        if int(sig.get("dir", 0) or 0) not in {-1, 1}:
            errors.append("BUY sig.dir must be +/-1")
    if action == "SELL":
        meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        if source != "dashboard_manual_close":
            errors.append("SELL source must be dashboard_manual_close")
        if not meta.get("manual_close"):
            errors.append("SELL meta.manual_close must be true")
        if "DASHBOARD_FORCE_CLOSE" not in str(sig.get("reason", "")).upper():
            errors.append("SELL reason must contain DASHBOARD_FORCE_CLOSE")
        if int(sig.get("dir", 0) or 0) not in {-1, 1}:
            errors.append("SELL sig.dir must be +/-1")
        if float(sig.get("bid_size", 0.0) or 0.0) <= 0:
            errors.append("SELL qty/bid_size must be > 0")
    return errors


def send_to_redis(payloads: List[Dict[str, Any]]) -> List[str]:
    import redis

    rds = redis.Redis(
        **{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]},
        decode_responses=False,
    )
    ids = []
    for payload in payloads:
        msg_id = rds.xadd(STREAM_ORCH_SIGNAL, {"data": ser.pack(payload)}, maxlen=10000)
        if isinstance(msg_id, bytes):
            msg_id = msg_id.decode("utf-8", errors="ignore")
        ids.append(str(msg_id))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["open", "close", "both"], default="both")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--tag", default="CALL_ATM", choices=["CALL_ATM", "CALL_OTM", "PUT_ATM", "PUT_OTM"])
    parser.add_argument("--qty", type=float, default=1.0)
    parser.add_argument("--price", type=float, default=1.23)
    parser.add_argument("--stock-price", type=float, default=500.0)
    parser.add_argument("--contract-id", default="SPY 260515C00500000")
    parser.add_argument("--send", action="store_true", help="Write generated payloads to Redis OMS stream.")
    parser.add_argument("--confirm", default="", help="Must equal SEND_TO_OMS when --send is used.")
    args = parser.parse_args()

    payloads: List[Dict[str, Any]] = []
    if args.case in {"open", "both"}:
        payloads.append(
            build_manual_open_payload(
                symbol=args.symbol,
                tag=args.tag,
                qty=int(args.qty),
                price=args.price,
                stock_price=args.stock_price,
                contract_id=args.contract_id,
            )
        )
    if args.case in {"close", "both"}:
        payloads.append(
            build_manual_close_payload(
                symbol=args.symbol,
                position=_direction_from_tag(args.tag),
                qty=float(args.qty),
                price=args.price,
                stock_price=args.stock_price,
                contract_id=args.contract_id,
            )
        )

    print(f"RUN_MODE={RUN_MODE} stream={STREAM_ORCH_SIGNAL}")
    all_errors: List[str] = []
    for payload in payloads:
        errors = validate_payload(payload)
        all_errors.extend([f"{payload.get('action')} {payload.get('symbol')}: {e}" for e in errors])
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))

    if all_errors:
        print("VALIDATION_FAILED")
        for err in all_errors:
            print(f"- {err}")
        return 1
    print("VALIDATION_OK")

    if args.send:
        if args.confirm != "SEND_TO_OMS":
            print("--send requires --confirm SEND_TO_OMS")
            return 2
        ids = send_to_redis(payloads)
        print(f"SENT_TO_OMS_STREAM ids={ids}")
    else:
        print("DRY_RUN_ONLY: no Redis write. Add --send --confirm SEND_TO_OMS for end-to-end OMS consumption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
