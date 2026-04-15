#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向 OMS 的 Redis Stream 手动推送标准测试信号。

用途：
1. 配合 quant_system.sh 启动的 ExecutionEngine / Dashboard 联调
2. 验证 Dashboard trade log 是否与控制台交易日志一致
3. 不依赖 SignalEngine，直接模拟 SE -> OMS 的真实 payload

示例:
  python3.10 production/tests/push_mock_oms_signal.py \
    --action BUY \
    --symbol NVDA \
    --contract-id NVDA260403C00180000 \
    --tag CALL_ATM \
    --stock-price 177.009 \
    --option-price 4.25 \
    --bid 4.20 \
    --ask 4.30 \
    --iv 0.42 \
    --alpha 1.5 \
    --strike 180

  python3.10 production/tests/push_mock_oms_signal.py \
    --action SELL \
    --symbol NVDA \
    --contract-id NVDA260403C00180000 \
    --stock-price 177.600 \
    --option-price 4.60 \
    --bid 4.55 \
    --ask 4.65 \
    --reason MANUAL_EXIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import redis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "baseline"
UTILS_DIR = PROJECT_ROOT / "utils"

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from config import REDIS_CFG, STREAM_ORCH_SIGNAL, get_redis_db  # type: ignore
from serialization_utils import pack  # type: ignore


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Push a mock OMS signal into Redis stream.")
    p.add_argument("--action", choices=["BUY", "SELL", "SYNC"], required=True)
    p.add_argument("--symbol", default="NVDA")
    p.add_argument("--tag", default="CALL_ATM")
    p.add_argument("--dir", type=int, default=1, help="BUY=1, SELL can still use original dir")
    p.add_argument("--contract-id", default="NVDA260403C00180000")
    p.add_argument("--ts", type=float, default=0.0, help="NY logical timestamp in seconds; default=now")
    p.add_argument("--stock-price", type=float, default=177.009)
    p.add_argument("--option-price", type=float, default=4.25)
    p.add_argument("--market-price", type=float, default=0.0)
    p.add_argument("--bid", type=float, default=4.20)
    p.add_argument("--ask", type=float, default=4.30)
    p.add_argument("--bid-size", type=float, default=50.0)
    p.add_argument("--ask-size", type=float, default=50.0)
    p.add_argument("--iv", type=float, default=0.42)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--strike", type=float, default=180.0)
    p.add_argument("--spy-roc", type=float, default=0.0010)
    p.add_argument("--index-trend", type=int, default=1)
    p.add_argument("--batch-idx", type=int, default=0)
    p.add_argument("--reason", default="MANUAL_TEST")
    p.add_argument("--run-mode", default=os.environ.get("RUN_MODE", "REALTIME"))
    p.add_argument("--redis-db", type=int, default=None)
    p.add_argument("--stream", default=STREAM_ORCH_SIGNAL)
    return p


def _resolve_ts(ts: float) -> float:
    return float(ts) if ts and ts > 0 else float(time.time())


def _build_signal_payload(args: argparse.Namespace) -> dict:
    ts_val = _resolve_ts(args.ts)
    market_price = float(args.market_price) if args.market_price > 0 else float(args.option_price)
    direction = int(args.dir)

    if args.action == "BUY":
        signal = {
            "action": "BUY",
            "dir": direction,
            "tag": args.tag,
            "price": float(args.option_price),
            "reason": args.reason,
            "meta": {
                "contract_id": args.contract_id,
                "strike": float(args.strike),
                "iv": float(args.iv),
                "bid": float(args.bid),
                "ask": float(args.ask),
                "bid_size": float(args.bid_size),
                "ask_size": float(args.ask_size),
                "spy_roc": float(args.spy_roc),
                "index_trend": int(args.index_trend),
                "alpha_z": float(args.alpha),
                "alpha_label_ts": ts_val - 60.0,
                "alpha_available_ts": ts_val,
            },
        }
        return {
            "action": "BUY",
            "symbol": args.symbol,
            "ts": ts_val,
            "stock_price": float(args.stock_price),
            "batch_idx": int(args.batch_idx),
            "sig": signal,
        }

    if args.action == "SELL":
        signal = {
            "action": "SELL",
            "dir": direction,
            "tag": args.tag,
            "price": float(args.option_price),
            "market_price": market_price,
            "bid": float(args.bid),
            "ask": float(args.ask),
            "bid_size": float(args.bid_size),
            "ask_size": float(args.ask_size),
            "reason": args.reason,
            "original_position": direction,
        }
        return {
            "action": "SELL",
            "symbol": args.symbol,
            "ts": ts_val,
            "stock_price": float(args.stock_price),
            "batch_idx": int(args.batch_idx),
            "sig": signal,
        }

    return {
        "action": "SYNC",
        "ts": ts_val,
        "prices": {},
        "payload": {},
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    os.environ["RUN_MODE"] = str(args.run_mode).upper()
    redis_db = int(args.redis_db) if args.redis_db is not None else int(get_redis_db())

    payload = _build_signal_payload(args)
    redis_client = redis.Redis(host=REDIS_CFG["host"], port=REDIS_CFG["port"], db=redis_db)
    msg_id = redis_client.xadd(args.stream, {"data": pack(payload)}, maxlen=5000)

    printable = {
        "stream": args.stream,
        "redis_db": redis_db,
        "message_id": msg_id.decode("utf-8") if isinstance(msg_id, bytes) else str(msg_id),
        "payload": payload,
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
