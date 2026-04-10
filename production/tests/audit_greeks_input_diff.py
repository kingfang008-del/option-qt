#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURR = Path(__file__).resolve()
PROD_ROOT = CURR.parent.parent
sys.path.append(str(PROD_ROOT))

from utils.greeks_math import calculate_bucket_greeks


ROW_NAMES = {
    0: "PUT_ATM",
    1: "PUT_OTM",
    2: "CALL_ATM",
    3: "CALL_OTM",
    4: "NEXT_PUT_ATM",
    5: "NEXT_CALL_ATM",
}

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


def find_rfr_file():
    candidates = [
        PROD_ROOT.parent / "risk_free_rates.parquet",
        Path("/home/kingfang007/risk_free_rates.parquet"),
        Path("risk_free_rates.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Risk-free rate file not found. Checked: {candidates}")


def load_rfr_series():
    path = find_rfr_file()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index).normalize()
    if "DGS3MO" not in df.columns:
        raise KeyError(f"DGS3MO column missing in {path}")
    series = df["DGS3MO"].copy()
    series = series / 100.0 if float(series.dropna().abs().max()) > 1.0 else series
    return path, series


def load_offline_row(parquet_path: Path, ts_str: str, bucket_id: int):
    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns:
        raise KeyError(f"'timestamp' column missing in {parquet_path}")
    ts_target = pd.Timestamp(ts_str)
    if ts_target.tzinfo is None:
        ts_target = ts_target.tz_localize("America/New_York")
    else:
        ts_target = ts_target.tz_convert("America/New_York")

    ts_series = pd.to_datetime(df["timestamp"])
    if getattr(ts_series.dt, "tz", None) is None:
        ts_series = ts_series.dt.tz_localize("America/New_York")
    else:
        ts_series = ts_series.dt.tz_convert("America/New_York")

    df = df.copy()
    df["timestamp"] = ts_series

    hit = df[(df["timestamp"] == ts_target) & (df["bucket_id"] == bucket_id)]
    if hit.empty:
        raise ValueError(
            f"No offline row found for timestamp={ts_target} bucket_id={bucket_id} in {parquet_path}"
        )
    if len(hit) > 1:
        hit = hit.iloc[[0]]
    return hit.iloc[0]


def extract_contract_expiry(contract: str):
    m = re.search(r"(\d{6})", contract.replace("O:", ""))
    if not m:
        raise ValueError(f"Cannot parse expiry from contract: {contract}")
    expiry = pd.to_datetime(m.group(1), format="%y%m%d").tz_localize("America/New_York")
    return expiry + pd.Timedelta(hours=16)


def compute_offline_inputs(row, rfr_series):
    ts = pd.Timestamp(row["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    expiry_ts = pd.Timestamp(row["expiration_date"])
    if expiry_ts.tzinfo is None:
        expiry_ts = expiry_ts.tz_localize("America/New_York")
    else:
        expiry_ts = expiry_ts.tz_convert("America/New_York")
    expiry_ts = expiry_ts + pd.Timedelta(hours=16)
    t_years = max(1e-6, (expiry_ts - ts).total_seconds() / 31557600.0)
    r_key = ts.normalize().tz_localize(None)
    if r_key not in rfr_series.index:
        raise ValueError(f"RFR missing exact date for offline row: {r_key}")
    r_val = float(rfr_series.loc[r_key])
    return {
        "timestamp_ny": ts,
        "expiry_ny": expiry_ts,
        "t_years": t_years,
        "r": r_val,
        "price": float(row["close"]),
        "spot": float(row["stock_close"]),
        "strike": float(row["strike_price"]),
        "contract": str(row["ticker"]).replace("O:", ""),
        "stored_iv": float(row["iv"]),
        "stored_theta": float(row["theta"]),
        "stored_delta": float(row["delta"]),
        "stored_gamma": float(row["gamma"]),
        "stored_vega": float(row["vega"]),
        "bid": float(row.get("bid", 0.0)),
        "ask": float(row.get("ask", 0.0)),
    }


def compute_runtime_equivalent(off):
    contract = off["contract"]
    runtime_ts = off["timestamp_ny"].floor("1min")
    t_years = max(1e-6, (extract_contract_expiry(contract) - runtime_ts).total_seconds() / 31557600.0)
    bucket = np.zeros((1, 12), dtype=np.float64)
    bucket[0, IDX_PRICE] = off["price"]
    bucket[0, IDX_STRIKE] = off["strike"]
    bucket[0, IDX_BID] = off["bid"]
    bucket[0, IDX_ASK] = off["ask"]
    out = calculate_bucket_greeks(
        bucket.copy(),
        S=off["spot"],
        T=t_years,
        r=off["r"],
        contracts=[contract],
        current_ts=float(runtime_ts.timestamp()),
    )
    return {
        "runtime_ts_anchor_ny": runtime_ts,
        "t_years": t_years,
        "r": off["r"],
        "price": float(out[0, IDX_PRICE]),
        "strike": float(out[0, IDX_STRIKE]),
        "iv": float(out[0, IDX_IV]),
        "delta": float(out[0, IDX_DELTA]),
        "gamma": float(out[0, IDX_GAMMA]),
        "vega": float(out[0, IDX_VEGA]),
        "theta": float(out[0, IDX_THETA]),
    }


def load_runtime_payload(path: Path):
    data = json.loads(path.read_text())
    return data


def extract_runtime_actual(data, bucket_id: int, fallback_contract: str = "", snapshot_key: str = "buckets"):
    buckets_raw = data[snapshot_key]
    buckets = np.asarray(buckets_raw, dtype=np.float64)
    contracts = list(data.get("contracts", []))
    if bucket_id >= len(buckets):
        raise IndexError(f"bucket_id={bucket_id} out of range for runtime payload key={snapshot_key}")
    row = buckets[bucket_id]
    contract = contracts[bucket_id] if bucket_id < len(contracts) else fallback_contract
    ts = data.get("ts")
    stock_price = data.get("stock_price")
    if stock_price is None:
        stock_price = data.get("close")
    return {
        "timestamp": float(ts) if ts is not None else None,
        "contract": contract,
        "stock_price": float(stock_price) if stock_price is not None else None,
        "price": float(row[IDX_PRICE]),
        "strike": float(row[IDX_STRIKE]),
        "iv": float(row[IDX_IV]),
        "delta": float(row[IDX_DELTA]),
        "gamma": float(row[IDX_GAMMA]),
        "vega": float(row[IDX_VEGA]),
        "theta": float(row[IDX_THETA]),
        "bid": float(row[IDX_BID]),
        "ask": float(row[IDX_ASK]),
    }


def extract_runtime_from_audit_json(data, bucket_id: int, fallback_contract: str = ""):
    contracts = list(data.get("contracts", []))
    out = {}
    for key in [
        "option_snapshot",
        "frozen_option_snapshot",
        "payload_option_snapshot",
        "latest_opt_buckets",
        "frozen_latest_opt_buckets",
    ]:
        if key not in data:
            continue
        row = extract_runtime_actual(
            {
                "ts": data.get("ts"),
                "stock_price": data.get("stock_price"),
                "contracts": contracts,
                key: data.get(key, []),
            },
            bucket_id=bucket_id,
            fallback_contract=fallback_contract,
            snapshot_key=key,
        )
        out[key] = row
    return out


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


def flatten_dict(prefix, data):
    rows = []
    if not isinstance(data, dict):
        return rows
    for k, v in data.items():
        key = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.extend(flatten_dict(f"{key}.", v))
        else:
            rows.append((key, v))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Audit offline-vs-runtime Greek inputs for a specific bucket/time."
    )
    parser.add_argument("--offline-parquet", required=True, help="Offline day parquet from ask_bid Greeks output")
    parser.add_argument("--timestamp", required=True, help="Timestamp in NY time, e.g. 2026-01-02 10:13:00")
    parser.add_argument("--bucket-id", type=int, default=5, help="Bucket row id to inspect; default 5 => NEXT_CALL_ATM")
    parser.add_argument("--runtime-payload", help="Optional JSON file with runtime buckets/contracts/ts/stock_price")
    args = parser.parse_args()

    offline_path = Path(args.offline_parquet).expanduser().resolve()
    runtime_path = Path(args.runtime_payload).expanduser().resolve() if args.runtime_payload else None

    rfr_path, rfr_series = load_rfr_series()
    row = load_offline_row(offline_path, args.timestamp, args.bucket_id)
    off = compute_offline_inputs(row, rfr_series)
    rt_formula = compute_runtime_equivalent(off)

    print_section(
        "定位目标",
        [
            ("bucket_id", args.bucket_id),
            ("bucket_name", ROW_NAMES.get(args.bucket_id, "UNKNOWN")),
            ("target_cell", f"({args.bucket_id}, {IDX_THETA})"),
            ("target_field", "theta"),
            ("offline_parquet", str(offline_path)),
            ("rfr_file", str(rfr_path)),
        ],
    )

    print_section(
        "离线输入与存量结果",
        [
            ("contract", off["contract"]),
            ("timestamp_ny", off["timestamp_ny"]),
            ("expiry_ny", off["expiry_ny"]),
            ("price_close", off["price"]),
            ("spot_stock_close", off["spot"]),
            ("strike", off["strike"]),
            ("bid", off["bid"]),
            ("ask", off["ask"]),
            ("r", off["r"]),
            ("T_years_offline", off["t_years"]),
            ("stored_iv_offline", off["stored_iv"]),
            ("stored_delta_offline", off["stored_delta"]),
            ("stored_gamma_offline", off["stored_gamma"]),
            ("stored_vega_offline", off["stored_vega"]),
            ("stored_theta_offline", off["stored_theta"]),
        ],
    )

    print_section(
        "按 realtime 公式重算后的结果（使用离线同一份 price / S / K / r）",
        [
            ("runtime_ts_anchor_ny", rt_formula["runtime_ts_anchor_ny"]),
            ("T_years_runtime", rt_formula["t_years"]),
            ("r", rt_formula["r"]),
            ("runtime_iv", rt_formula["iv"]),
            ("runtime_delta", rt_formula["delta"]),
            ("runtime_gamma", rt_formula["gamma"]),
            ("runtime_vega", rt_formula["vega"]),
            ("runtime_theta", rt_formula["theta"]),
        ],
    )

    print_section(
        "离线存量 vs realtime公式重算 差异",
        [
            ("iv_diff", rt_formula["iv"] - off["stored_iv"]),
            ("delta_diff", rt_formula["delta"] - off["stored_delta"]),
            ("gamma_diff", rt_formula["gamma"] - off["stored_gamma"]),
            ("vega_diff", rt_formula["vega"] - off["stored_vega"]),
            ("theta_diff", rt_formula["theta"] - off["stored_theta"]),
            ("T_diff", rt_formula["t_years"] - off["t_years"]),
        ],
    )

    if runtime_path:
        runtime_payload = load_runtime_payload(runtime_path)
        if "buckets" in runtime_payload and "contracts" in runtime_payload:
            rt_actual = extract_runtime_actual(runtime_payload, args.bucket_id, fallback_contract=off["contract"])
            print_section(
                "实际 runtime payload 中该 bucket 的值",
                [
                    ("runtime_payload", str(runtime_path)),
                    ("contract", rt_actual["contract"]),
                    ("timestamp", rt_actual["timestamp"]),
                    ("stock_price", rt_actual["stock_price"]),
                    ("price", rt_actual["price"]),
                    ("strike", rt_actual["strike"]),
                    ("bid", rt_actual["bid"]),
                    ("ask", rt_actual["ask"]),
                    ("iv", rt_actual["iv"]),
                    ("delta", rt_actual["delta"]),
                    ("gamma", rt_actual["gamma"]),
                    ("vega", rt_actual["vega"]),
                    ("theta", rt_actual["theta"]),
                ],
            )
            print_section(
                "实际 runtime payload vs 离线存量 差异",
                [
                    ("price_diff", rt_actual["price"] - off["price"]),
                    ("spot_diff", (rt_actual["stock_price"] or 0.0) - off["spot"] if rt_actual["stock_price"] is not None else "N/A"),
                    ("strike_diff", rt_actual["strike"] - off["strike"]),
                    ("iv_diff", rt_actual["iv"] - off["stored_iv"]),
                    ("delta_diff", rt_actual["delta"] - off["stored_delta"]),
                    ("gamma_diff", rt_actual["gamma"] - off["stored_gamma"]),
                    ("vega_diff", rt_actual["vega"] - off["stored_vega"]),
                    ("theta_diff", rt_actual["theta"] - off["stored_theta"]),
                ],
            )
        else:
            runtime_layers = extract_runtime_from_audit_json(runtime_payload, args.bucket_id, fallback_contract=off["contract"])
            print_section(
                "runtime 审计 JSON 概览",
                [
                    ("runtime_audit_json", str(runtime_path)),
                    ("symbol", runtime_payload.get("symbol")),
                    ("ts", runtime_payload.get("ts")),
                    ("stock_price", runtime_payload.get("stock_price")),
                    ("latest_stock_price", runtime_payload.get("latest_stock_price")),
                    ("last_stock_update_ts", runtime_payload.get("last_stock_update_ts")),
                    ("last_option_update_ts", runtime_payload.get("last_option_update_ts")),
                    ("contract_bucket", runtime_payload.get("contract")),
                ],
            )
            if runtime_payload.get("pre_supplement_greeks_input"):
                print_section(
                    "补 Greeks 前实际入参审计",
                    flatten_dict("", runtime_payload.get("pre_supplement_greeks_input", {})),
                )
            if runtime_payload.get("post_supplement_greeks_input"):
                print_section(
                    "补 Greeks 后结果审计",
                    flatten_dict("", runtime_payload.get("post_supplement_greeks_input", {})),
                )
            for layer_name, rt_actual in runtime_layers.items():
                print_section(
                    f"{layer_name} 中该 bucket 的值",
                    [
                        ("contract", rt_actual["contract"]),
                        ("timestamp", rt_actual["timestamp"]),
                        ("stock_price", rt_actual["stock_price"]),
                        ("price", rt_actual["price"]),
                        ("strike", rt_actual["strike"]),
                        ("bid", rt_actual["bid"]),
                        ("ask", rt_actual["ask"]),
                        ("iv", rt_actual["iv"]),
                        ("delta", rt_actual["delta"]),
                        ("gamma", rt_actual["gamma"]),
                        ("vega", rt_actual["vega"]),
                        ("theta", rt_actual["theta"]),
                    ],
                )
                print_section(
                    f"{layer_name} vs 离线存量 差异",
                    [
                        ("price_diff", rt_actual["price"] - off["price"]),
                        ("spot_diff", (rt_actual["stock_price"] or 0.0) - off["spot"] if rt_actual["stock_price"] is not None else "N/A"),
                        ("strike_diff", rt_actual["strike"] - off["strike"]),
                        ("iv_diff", rt_actual["iv"] - off["stored_iv"]),
                        ("delta_diff", rt_actual["delta"] - off["stored_delta"]),
                        ("gamma_diff", rt_actual["gamma"] - off["stored_gamma"]),
                        ("vega_diff", rt_actual["vega"] - off["stored_vega"]),
                        ("theta_diff", rt_actual["theta"] - off["stored_theta"]),
                    ],
                )

    print("\n结论提示:")
    print("1. 如果“离线存量 vs realtime公式重算”已经接近 0，说明公式本身基本一致，问题更可能在 runtime 输入。")
    print("2. 如果这里 theta 差异已经很大，优先检查 timestamp 锚点、price 口径、rfr 取法。")
    print("3. 如果提供的是 runtime 审计 JSON，先看 pre_supplement_greeks_input，再看差异首次出现在哪一层：option_snapshot / frozen_option_snapshot / payload_option_snapshot。")


if __name__ == "__main__":
    main()
