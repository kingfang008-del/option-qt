#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURR = Path(__file__).resolve()
PROD_ROOT = CURR.parent.parent
sys.path.append(str(PROD_ROOT))

from utils.greeks_math import calculate_bucket_greeks

try:
    from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False


DEFAULT_TS = 1767366780
DEFAULT_CONTRACT = "O:NVDA260206C00195000"
DEFAULT_S = 191.09
DEFAULT_PRICE = 7.50
DEFAULT_STRIKE = 195.0
DEFAULT_BID = 7.45
DEFAULT_ASK = 7.55
DEFAULT_R = 0.0365
DEFAULT_T = 0.09648452

IDX_PRICE = 0
IDX_DELTA = 1
IDX_GAMMA = 2
IDX_VEGA = 3
IDX_THETA = 4
IDX_STRIKE = 5
IDX_VOLUME = 6
IDX_IV = 7
IDX_BID = 8
IDX_ASK = 9
IDX_BID_SIZE = 10
IDX_ASK_SIZE = 11


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.8f}"
    return str(v)


def print_section(title, rows):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    width = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f"{k:<{width}} {fmt(v)}")


def load_from_runtime_audit(path: Path, bucket_id: int):
    data = json.loads(path.read_text())
    pre = data.get("pre_supplement_greeks_input", {})
    row = pre.get("row_before_or_after", {})
    contract = pre.get("contract") or data.get("contract") or DEFAULT_CONTRACT
    return {
        "timestamp": float(pre.get("timestamp", data.get("ts", DEFAULT_TS))),
        "contract": contract,
        "S": float(pre.get("stock_price_input", data.get("stock_price", DEFAULT_S))),
        "price": float(row.get("price", DEFAULT_PRICE)),
        "strike": float(row.get("strike", DEFAULT_STRIKE)),
        "bid": float(row.get("bid", DEFAULT_BID)),
        "ask": float(row.get("ask", DEFAULT_ASK)),
        "bid_size": float(row.get("bid_size", 0.0)),
        "ask_size": float(row.get("ask_size", 0.0)),
        "volume": float(row.get("volume", 0.0)),
        "r": float(pre.get("r", DEFAULT_R)),
        "T": float(pre.get("T_years", DEFAULT_T)),
        "bucket_id": bucket_id,
    }


def build_case(args):
    if args.runtime_audit:
        return load_from_runtime_audit(Path(args.runtime_audit).expanduser().resolve(), args.bucket_id)
    return {
        "timestamp": float(args.timestamp),
        "contract": args.contract,
        "S": float(args.stock_price),
        "price": float(args.price),
        "strike": float(args.strike),
        "bid": float(args.bid),
        "ask": float(args.ask),
        "bid_size": float(args.bid_size),
        "ask_size": float(args.ask_size),
        "volume": float(args.volume),
        "r": float(args.r),
        "T": float(args.t_years),
        "bucket_id": int(args.bucket_id),
    }


def direct_vollib(case):
    if not HAS_VOLLIB:
        return {"error": "py_vollib_vectorized not installed"}
    contract = str(case["contract"]).replace("O:", "")
    opt_type = "p" if "P" in contract.split() else "c"
    if len(contract) >= 7:
        marker = contract[6:7]
        if marker in ("C", "P"):
            opt_type = marker.lower()
    iv_res = vectorized_implied_volatility(
        np.array([case["price"]], dtype=np.float64),
        np.array([case["S"]], dtype=np.float64),
        np.array([case["strike"]], dtype=np.float64),
        np.array([case["T"]], dtype=np.float64),
        np.array([case["r"]], dtype=np.float64),
        opt_type,
        return_as="numpy",
        on_error="ignore",
    )
    iv = float(iv_res[0]) if len(iv_res) else 0.0
    out = {"raw_iv_result": iv_res.tolist(), "iv": iv, "opt_type": opt_type}
    if iv > 0.0:
        g = get_all_greeks(
            opt_type,
            np.array([case["S"]], dtype=np.float64),
            np.array([case["strike"]], dtype=np.float64),
            np.array([case["T"]], dtype=np.float64),
            np.array([case["r"]], dtype=np.float64),
            np.array([iv], dtype=np.float64),
            return_as="dict",
        )
        out.update(
            {
                "delta": float(g["delta"][0]),
                "gamma": float(g["gamma"][0]),
                "vega": float(g["vega"][0]),
                "theta": float(g["theta"][0]),
            }
        )
    return out


def run_case(case):
    buckets = np.zeros((6, 12), dtype=np.float64)
    row = case["bucket_id"]
    buckets[row, IDX_PRICE] = case["price"]
    buckets[row, IDX_STRIKE] = case["strike"]
    buckets[row, IDX_VOLUME] = case["volume"]
    buckets[row, IDX_BID] = case["bid"]
    buckets[row, IDX_ASK] = case["ask"]
    buckets[row, IDX_BID_SIZE] = case["bid_size"]
    buckets[row, IDX_ASK_SIZE] = case["ask_size"]

    contracts = [""] * 6
    contracts[row] = case["contract"]

    before = buckets.copy()
    after = calculate_bucket_greeks(
        buckets.copy(),
        S=case["S"],
        T=case["T"],
        r=case["r"],
        contracts=contracts,
        current_ts=case["timestamp"],
    )
    return before[row], after[row]


def main():
    parser = argparse.ArgumentParser(description="Standalone repro for greeks_math.calculate_bucket_greeks")
    parser.add_argument("--runtime-audit", help="Optional runtime audit JSON to source the exact pre-supplement inputs")
    parser.add_argument("--bucket-id", type=int, default=5)
    parser.add_argument("--timestamp", type=float, default=DEFAULT_TS)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--stock-price", type=float, default=DEFAULT_S)
    parser.add_argument("--price", type=float, default=DEFAULT_PRICE)
    parser.add_argument("--strike", type=float, default=DEFAULT_STRIKE)
    parser.add_argument("--bid", type=float, default=DEFAULT_BID)
    parser.add_argument("--ask", type=float, default=DEFAULT_ASK)
    parser.add_argument("--bid-size", type=float, default=39.0)
    parser.add_argument("--ask-size", type=float, default=373.0)
    parser.add_argument("--volume", type=float, default=412.0)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--t-years", type=float, default=DEFAULT_T)
    args = parser.parse_args()

    case = build_case(args)
    before, after = run_case(case)
    vollib = direct_vollib(case)

    ts_ny = pd.Timestamp(case["timestamp"], unit="s", tz="UTC").tz_convert("America/New_York")
    print_section(
        "复现输入",
        [
            ("timestamp", case["timestamp"]),
            ("timestamp_ny", ts_ny),
            ("contract", case["contract"]),
            ("bucket_id", case["bucket_id"]),
            ("stock_price", case["S"]),
            ("price", case["price"]),
            ("strike", case["strike"]),
            ("bid", case["bid"]),
            ("ask", case["ask"]),
            ("r", case["r"]),
            ("T_years", case["T"]),
        ],
    )

    print_section(
        "调用 calculate_bucket_greeks 前的 row",
        [
            ("price", before[IDX_PRICE]),
            ("delta", before[IDX_DELTA]),
            ("gamma", before[IDX_GAMMA]),
            ("vega", before[IDX_VEGA]),
            ("theta", before[IDX_THETA]),
            ("strike", before[IDX_STRIKE]),
            ("volume", before[IDX_VOLUME]),
            ("iv", before[IDX_IV]),
            ("bid", before[IDX_BID]),
            ("ask", before[IDX_ASK]),
        ],
    )

    print_section(
        "调用 calculate_bucket_greeks 后的 row",
        [
            ("price", after[IDX_PRICE]),
            ("delta", after[IDX_DELTA]),
            ("gamma", after[IDX_GAMMA]),
            ("vega", after[IDX_VEGA]),
            ("theta", after[IDX_THETA]),
            ("strike", after[IDX_STRIKE]),
            ("volume", after[IDX_VOLUME]),
            ("iv", after[IDX_IV]),
            ("bid", after[IDX_BID]),
            ("ask", after[IDX_ASK]),
        ],
    )

    print_section(
        "直接调用 py_vollib_vectorized",
        [(k, v) for k, v in vollib.items()],
    )


if __name__ == "__main__":
    main()
