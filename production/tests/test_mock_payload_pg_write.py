#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构造模拟 payload，验证“payload 生成后 -> PG 写库”流程。

设计目标：
1) 不依赖本机 Greeks 实时计算环境；
2) 直接复用 DataPersistenceServicePG.process_feature_data；
3) 使用 config.py 中的 PG_DB_URL 连接；
4) 写入后回查 option_snapshots_1s / option_snapshots_1m，验证 Greeks 是否落库。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psycopg2


def _bootstrap_imports() -> None:
    """将 baseline / DAO 目录加入 sys.path，便于直接导入现有模块。"""
    repo_root = Path(__file__).resolve().parents[2]
    production_dir = repo_root / "production"
    baseline_dir = repo_root / "production" / "baseline"
    dao_dir = baseline_dir / "DAO"
    sys.path.insert(0, str(production_dir))
    sys.path.insert(0, str(dao_dir))
    sys.path.insert(0, str(baseline_dir))


def _build_mock_buckets(seed: float = 1.0) -> List[List[float]]:
    """
    构造 6x12 的期权桶矩阵，包含非零 Greeks 与 IV。
    列约定（项目内）：
    0=price, 1~4=Greeks, 5=strike, 6=volume, 7=iv, 8=bid, 9=ask, 10/11=size
    """
    arr = np.zeros((6, 12), dtype=np.float32)
    strikes = [95, 100, 100, 105, 110, 115]
    for i in range(6):
        arr[i, 0] = 1.0 + seed * 0.1 + i * 0.05
        arr[i, 1] = (-0.2 + 0.08 * i)  # delta
        arr[i, 2] = 0.01 + 0.002 * i   # gamma
        arr[i, 3] = 0.12 + 0.01 * i    # vega
        arr[i, 4] = -0.03 - 0.002 * i  # theta
        arr[i, 5] = float(strikes[i])
        arr[i, 6] = 100 + i * 10
        arr[i, 7] = 0.25 + 0.01 * i    # iv
        arr[i, 8] = arr[i, 0] - 0.02
        arr[i, 9] = arr[i, 0] + 0.02
        arr[i, 10] = 50 + i
        arr[i, 11] = 60 + i
    return arr.tolist()


def _build_payload(ts: int, symbols: List[str], include_not_ready: bool) -> Dict:
    live_options = {}
    for idx, sym in enumerate(symbols):
        ready = True
        if include_not_ready and idx == len(symbols) - 1:
            ready = False
        live_options[sym] = {
            "buckets": _build_mock_buckets(seed=1.0 + idx),
            "contracts": [
                f"O:{sym}260620P00095000",
                f"O:{sym}260620P00100000",
                f"O:{sym}260620C00100000",
                f"O:{sym}260620C00105000",
                f"O:{sym}260620P00110000",
                f"O:{sym}260620C00115000",
            ],
            "greeks_ready": ready,
        }

    # 这里只测 option write 流程，feature_logs 可以关闭写入
    payload = {
        "ts": float(ts),
        "symbols": symbols,
        "live_options": live_options,
        "is_new_minute": True,
    }
    return payload


def _fetch_rows(conn, table: str, ts: int, symbols: List[str]):
    with conn.cursor() as c:
        c.execute(
            f"""
            SELECT symbol, ts, buckets_json
            FROM {table}
            WHERE ts = %s AND symbol = ANY(%s)
            ORDER BY symbol
            """,
            (float(ts), symbols),
        )
        return c.fetchall()


def _greeks_sum_from_json(buckets_json: str) -> float:
    try:
        obj = json.loads(buckets_json) if isinstance(buckets_json, str) else buckets_json
        buckets = obj.get("buckets", []) if isinstance(obj, dict) else []
        arr = np.array(buckets, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 5:
            return 0.0
        return float(np.sum(np.abs(arr[:, 1:5])))
    except Exception:
        return 0.0


def main():
    _bootstrap_imports()
    from config import PG_DB_URL  # noqa: E402
    from data_persistence_service_v8_pg import DataPersistenceServicePG  # noqa: E402

    parser = argparse.ArgumentParser(description="Mock payload -> PG option snapshot write check")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AAPL"], help="测试 symbols")
    parser.add_argument("--ts", type=int, default=None, help="写入 ts（默认当前分钟）")
    parser.add_argument("--include-not-ready", action="store_true", help="最后一个 symbol 标记 greeks_ready=False，验证 1m gate")
    parser.add_argument("--disable-1m-gate", action="store_true", help="关闭 option_1m_wait_greeks_ready，验证强制写 1m")
    parser.add_argument("--cleanup", action="store_true", help="验证后删除本次 ts 测试数据")
    args = parser.parse_args()

    ts = args.ts if args.ts is not None else int(time.time()) // 60 * 60
    payload = _build_payload(ts=ts, symbols=args.symbols, include_not_ready=args.include_not_ready)

    # 避免依赖 Redis，绕过 __init__：只挂载 PG 连接与必要属性
    svc = DataPersistenceServicePG.__new__(DataPersistenceServicePG)
    svc.conn = psycopg2.connect(PG_DB_URL)
    svc.conn.autocommit = False
    svc.current_date = None
    svc.bar_buffer_1s = {}
    svc.bar_buffer = {}
    svc.option_buffer = {}
    svc.option_buffer_1s = {}
    svc.bar_buffer_5m = {}
    svc.option_buffer_5m = {}
    svc.acc_5m = {}
    svc.last_synthesized = {}
    svc.trade_buffer = []
    svc.option_1m_wait_greeks_ready = not args.disable_1m_gate

    # 初始化主表（确保 option_snapshots_* 存在）
    svc._init_master_tables()
    svc._check_date_rotation(ts)

    print(f"[INFO] write ts={ts}, symbols={args.symbols}, 1m_gate={svc.option_1m_wait_greeks_ready}")
    svc.process_feature_data(payload, write_feature_logs=False, write_option_snapshots=True)
    svc.conn.commit()

    rows_1s = _fetch_rows(svc.conn, "option_snapshots_1s", ts, args.symbols)
    rows_1m = _fetch_rows(svc.conn, "option_snapshots_1m", ts, args.symbols)

    print(f"[RESULT] option_snapshots_1s rows={len(rows_1s)}")
    for sym, row_ts, buckets_json in rows_1s:
        print(f"  - 1s {sym} ts={int(row_ts)} greeks_sum={_greeks_sum_from_json(buckets_json):.6f}")

    print(f"[RESULT] option_snapshots_1m rows={len(rows_1m)}")
    for sym, row_ts, buckets_json in rows_1m:
        print(f"  - 1m {sym} ts={int(row_ts)} greeks_sum={_greeks_sum_from_json(buckets_json):.6f}")

    if args.cleanup:
        with svc.conn.cursor() as c:
            c.execute("DELETE FROM option_snapshots_1s WHERE ts=%s AND symbol = ANY(%s)", (float(ts), args.symbols))
            c.execute("DELETE FROM option_snapshots_1m WHERE ts=%s AND symbol = ANY(%s)", (float(ts), args.symbols))
        svc.conn.commit()
        print("[CLEANUP] deleted test rows from option_snapshots_1s/1m")

    svc.conn.close()
    print("[DONE] mock payload PG write check completed.")


if __name__ == "__main__":
    main()

