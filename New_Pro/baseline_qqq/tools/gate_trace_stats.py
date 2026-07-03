#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V0 / OMS 决策门控统计 — 从 Redis meta:gate_trace:* 与 meta:gate_counter:* 汇总。

用法:
  cd New_Pro/baseline_qqq
  python tools/gate_trace_stats.py
  python tools/gate_trace_stats.py --date 20260703 --top 20
  python tools/gate_trace_stats.py --redis-url redis://127.0.0.1:6379/0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import redis
except ImportError:
    redis = None  # type: ignore

_BASE = Path(__file__).resolve().parents[1]
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
import baseline_paths  # noqa: E402,F401


def _default_redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


def _ny_date_str(d: Optional[str] = None) -> str:
    if d:
        return d.replace("-", "")
    try:
        from pytz import timezone

        return datetime.now(timezone("America/New_York")).strftime("%Y%m%d")
    except Exception:
        return datetime.utcnow().strftime("%Y%m%d")


def _decode_hash(raw: Dict[Any, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (raw or {}).items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        val = v.decode() if isinstance(v, bytes) else str(v)
        out[key] = val
    return out


def fetch_gate_counter(r, ny_date: str) -> Counter:
    key = f"meta:gate_counter:{ny_date}"
    raw = r.hgetall(key) or {}
    decoded = _decode_hash(raw)
    cnt = Counter()
    for gate, val in decoded.items():
        try:
            cnt[gate] = int(val)
        except (TypeError, ValueError):
            pass
    return cnt


def fetch_gate_traces(r) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        keys = list(r.scan_iter(match="meta:gate_trace:*", count=200))
    except Exception:
        keys = []
    for key in keys:
        sym = key.decode().split(":", 2)[-1] if isinstance(key, bytes) else str(key).split(":", 2)[-1]
        h = _decode_hash(r.hgetall(key) or {})
        if not h:
            continue
        trace: List[dict] = []
        if h.get("trace_json"):
            try:
                trace = json.loads(h["trace_json"])
            except json.JSONDecodeError:
                trace = []
        rows.append(
            {
                "symbol": sym,
                "kind": h.get("kind", ""),
                "result": h.get("result", ""),
                "last_block": h.get("last_block", ""),
                "ts": h.get("ts", ""),
                "trace": trace,
            }
        )
    return rows


def aggregate_trace_blocks(traces: List[Dict[str, Any]]) -> Counter:
    """从 trace_json 全量统计各 gate 的 block 次数(比 counter 更细,含 pass/skip)。"""
    blocks = Counter()
    for row in traces:
        for step in row.get("trace") or []:
            if step.get("status") == "block":
                gate = str(step.get("gate") or "unknown")
                blocks[gate] += 1
    return blocks


def aggregate_by_kind(traces: List[Dict[str, Any]]) -> Dict[str, Counter]:
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    for row in traces:
        kind = row.get("kind") or "unknown"
        lb = row.get("last_block") or ""
        if lb:
            by_kind[kind][lb] += 1
        res = row.get("result") or ""
        if res.startswith("REJECT:"):
            by_kind[kind][res.replace("REJECT:", "", 1)] += 1
    return by_kind


def print_report(
    *,
    ny_date: str,
    counter: Counter,
    traces: List[Dict[str, Any]],
    top: int,
) -> None:
    print(f"=== Gate 统计 (NY date={ny_date}) ===\n")

    if counter:
        print(f"--- meta:gate_counter:{ny_date} (OMS 日累计, Top {top}) ---")
        total = sum(counter.values())
        for gate, n in counter.most_common(top):
            pct = 100.0 * n / total if total else 0.0
            print(f"  {gate:32s}  {n:6d}  ({pct:5.1f}%)")
        print(f"  {'TOTAL':32s}  {total:6d}\n")
    else:
        print(f"--- meta:gate_counter:{ny_date}: (空) ---\n")

    if traces:
        print(f"--- meta:gate_trace:* 快照 ({len(traces)} symbols) ---")
        for row in sorted(traces, key=lambda x: x["symbol"]):
            print(
                f"  {row['symbol']:6s}  kind={row.get('kind','?'):5s}  "
                f"result={row.get('result','')}  last_block={row.get('last_block','')}"
            )
        print()

        by_kind = aggregate_by_kind(traces)
        for kind, cnt in sorted(by_kind.items()):
            print(f"--- last_block by kind={kind} ---")
            for gate, n in cnt.most_common(top):
                print(f"  {gate:32s}  {n:6d}")
            print()

        trace_blocks = aggregate_trace_blocks(traces)
        if trace_blocks:
            print(f"--- trace_json block 累计 (Top {top}) ---")
            for gate, n in trace_blocks.most_common(top):
                print(f"  {gate:32s}  {n:6d}")
            print()
    else:
        print("--- meta:gate_trace:*: (空) ---\n")

    print("提示: counter 仅在 last_block 变化时 +1; trace_json 含完整 E* 链。")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Redis V0 gate trace 统计")
    parser.add_argument("--redis-url", default=_default_redis_url())
    parser.add_argument("--date", default=None, help="NY 交易日 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)

    if redis is None:
        print("需要 redis 包: pip install redis", file=sys.stderr)
        return 1

    ny_date = _ny_date_str(args.date)
    try:
        r = redis.from_url(args.redis_url, decode_responses=False)
        r.ping()
    except Exception as e:
        print(f"Redis 连接失败 ({args.redis_url}): {e}", file=sys.stderr)
        return 1

    counter = fetch_gate_counter(r, ny_date)
    traces = fetch_gate_traces(r)
    print_report(ny_date=ny_date, counter=counter, traces=traces, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
