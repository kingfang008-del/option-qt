#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
接回 production/baseline 老版对拍工具:

  - verify_parity_raw.py      → 分钟期权桶 (BAR_OPT / 离线 IV 参考)
  - verify_parity_thresholds.py → FCS NPZ 阈值门禁

供 compare_stream_replay_day 分层验收调用,避免再手跑旧脚本。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_PROD_BASELINE = _REPO / "production" / "baseline"
if str(_PROD_BASELINE) not in sys.path:
    sys.path.insert(0, str(_PROD_BASELINE))

try:
    from verify_parity_thresholds import (  # type: ignore
        DEFAULT_THRESHOLDS,
        evaluate_threshold_diffs,
    )
except Exception:  # pragma: no cover - 仓库布局异常时的兜底
    DEFAULT_THRESHOLDS = {
        "hist_vwap_30": 0.06,
        "vwap_diff": 0.02,
        "options_iv_divergence": 0.005,
        "option_snapshot_6x12": 0.60,
        "frozen_option_snapshot_6x12": 0.60,
        "frozen_latest_opt_buckets_6x12": 0.60,
    }

    def evaluate_threshold_diffs(left_map, right_map, thresholds, strict=False):  # type: ignore
        raise RuntimeError("verify_parity_thresholds import failed")


# bucket 列语义与 verify_parity_raw / greeks_math 一致
_BUCKET_FIELDS = (
    ("last", 0, 0.05),
    ("delta", 1, 0.05),
    ("gamma", 2, 0.05),
    ("vega", 3, 0.50),
    ("theta", 4, 0.50),
    ("strike", 5, 0.01),
    ("volume", 6, 50.0),
    ("iv", 7, 0.05),
    ("bid", 8, 0.05),
    ("ask", 9, 0.05),
)

# 离线 IV 参考表没有 bid/ask/last 时,只对希腊值/IV/volume 门禁
_REF_GATE_FIELDS = ("delta", "gamma", "vega", "theta", "iv", "volume")


def _as_6x12(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64) if raw is not None else np.zeros((0, 0))
    out = np.zeros((6, 12), dtype=np.float64)
    if arr.ndim != 2:
        return out
    r = min(6, int(arr.shape[0]))
    c = min(12, int(arr.shape[1]))
    if r and c:
        out[:r, :c] = arr[:r, :c]
    return out


def buckets_from_payload(payload: Any) -> np.ndarray:
    """Redis BAR_OPT JSON / sqlite buckets_json → (6,12)."""
    if payload is None:
        return np.zeros((6, 12), dtype=np.float64)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        buckets = payload.get("buckets", payload)
    else:
        buckets = payload
    return _as_6x12(buckets)


def load_offline_ref_buckets(
    symbol: str,
    date: str,
    ts_unix: int,
    *,
    greek_root: Path | str | None = None,
) -> np.ndarray:
    """用 quote_options_*_iv 分钟参考表拼 6x12(无价时 last/bid/ask=0)。"""
    from qqq_btc.common.option_minute_ref import load_minute_option_ref

    root = Path(greek_root).expanduser() if greek_root else None
    lookup = load_minute_option_ref(symbol, date, greek_root=root)
    minute_ts = int(pd.Timestamp(ts_unix, unit="s", tz="UTC").floor("min").timestamp())
    out = np.zeros((6, 12), dtype=np.float64)
    for b_id in range(6):
        g = lookup.get((minute_ts, b_id)) or lookup.get((ts_unix, b_id))
        if not g:
            # ceil 口径(option_minute_ref) vs floor:再试 ±60s
            for adj in (60, -60):
                g = lookup.get((minute_ts + adj, b_id))
                if g:
                    break
        if not g:
            continue
        out[b_id, 1] = float(g.get("delta", 0.0) or 0.0)
        out[b_id, 2] = float(g.get("gamma", 0.0) or 0.0)
        out[b_id, 3] = float(g.get("vega", 0.0) or 0.0)
        out[b_id, 4] = float(g.get("theta", 0.0) or 0.0)
        out[b_id, 6] = float(g.get("volume", 0.0) or 0.0)
        out[b_id, 7] = float(g.get("iv", 0.0) or 0.0)
    return out


def compare_option_buckets(
    stream_buckets: np.ndarray,
    ref_buckets: np.ndarray,
    *,
    gate_fields: tuple[str, ...] = _REF_GATE_FIELDS,
    atm_only: bool = True,
) -> dict:
    """
    对齐 verify_parity_raw.compare_buckets 精神:按固定槽位比 PUT/CALL ATM 等。
    atm_only=True 时只门禁 bucket 0(PUT_ATM) 与 2(CALL_ATM)。
    """
    stream = _as_6x12(stream_buckets)
    ref = _as_6x12(ref_buckets)
    bucket_ids = (0, 2) if atm_only else tuple(range(6))
    field_tol = {name: tol for name, _idx, tol in _BUCKET_FIELDS}
    field_idx = {name: idx for name, idx, _tol in _BUCKET_FIELDS}

    rows = []
    for b_id in bucket_ids:
        side = {0: "PUT_ATM", 2: "CALL_ATM"}.get(b_id, f"B{b_id}")
        for name in gate_fields:
            if name not in field_idx:
                continue
            idx = field_idx[name]
            s_v = float(stream[b_id, idx])
            r_v = float(ref[b_id, idx])
            # 参考侧全 0 且流侧也 0 → skip(该分钟无合约)
            if abs(r_v) < 1e-12 and abs(s_v) < 1e-12:
                continue
            diff = s_v - r_v
            tol = float(field_tol.get(name, 0.05))
            rows.append(
                {
                    "bucket": int(b_id),
                    "side": side,
                    "field": name,
                    "stream": s_v,
                    "ref": r_v,
                    "diff": float(diff),
                    "abs_diff": float(abs(diff)),
                    "tol": tol,
                    "ok": bool(abs(diff) <= tol),
                }
            )
    n = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    return {
        "n_checked": n,
        "n_ok": n_ok,
        "ok": bool(n > 0 and n_ok == n),
        "rows": sorted(rows, key=lambda r: r["abs_diff"], reverse=True),
    }


def report_redis_option_bucket_parity(
    *,
    symbol: str,
    date: str,
    ts_unix: int,
    greek_root: Path | str | None = None,
    redis_client: Any = None,
    sqlite_db: Path | str | None = None,
) -> dict:
    """
    Redis BAR_OPT:1M vs 离线分钟 IV 参考(或 sqlite option_snapshots_1m)。
    对应老版 verify_parity_raw 的 OPTION 段。
    """
    key = f"BAR_OPT:1M:{symbol}"
    stream_raw = None
    source = "redis"
    if redis_client is not None:
        stream_raw = redis_client.hget(key, str(int(ts_unix)))
        if stream_raw is None:
            # 偶发 key 用 floor 分钟
            minute_ts = int(pd.Timestamp(ts_unix, unit="s", tz="UTC").floor("min").timestamp())
            stream_raw = redis_client.hget(key, str(minute_ts))
            ts_unix = minute_ts

    ref_buckets = None
    ref_source = "option_minute_ref"
    if sqlite_db is not None:
        db = Path(sqlite_db).expanduser()
        if db.exists():
            import sqlite3

            conn = sqlite3.connect(str(db))
            try:
                row = pd.read_sql(
                    "SELECT buckets_json FROM option_snapshots_1m WHERE symbol=? AND ts=? LIMIT 1",
                    conn,
                    params=(symbol, int(ts_unix)),
                )
            finally:
                conn.close()
            if not row.empty:
                ref_buckets = buckets_from_payload(row.iloc[0]["buckets_json"])
                ref_source = f"sqlite:{db.name}"

    if ref_buckets is None:
        ref_buckets = load_offline_ref_buckets(
            symbol, date, int(ts_unix), greek_root=greek_root
        )

    if stream_raw is None:
        return {
            "ok": False,
            "symbol": symbol,
            "date": date,
            "ts_unix": int(ts_unix),
            "stream_source": source,
            "ref_source": ref_source,
            "error": f"missing Redis {key}@{ts_unix}",
            "n_checked": 0,
            "n_ok": 0,
            "rows": [],
        }

    stream_buckets = buckets_from_payload(stream_raw)
    cmp = compare_option_buckets(stream_buckets, ref_buckets)
    out = {
        "symbol": symbol,
        "date": date,
        "ts_unix": int(ts_unix),
        "stream_source": source,
        "ref_source": ref_source,
        **cmp,
    }
    print(
        f"\n=== Option bucket parity (verify_parity_raw) @ {date} ts={ts_unix} ==="
    )
    print(f"  stream={source}  ref={ref_source}  checked={cmp['n_checked']} ok={cmp['n_ok']}")
    for r in cmp["rows"][:12]:
        tag = "ok" if r["ok"] else "GAP"
        print(
            f"  {r['side']:8s} {r['field']:8s} "
            f"stream={r['stream']:+.5f} ref={r['ref']:+.5f} "
            f"d={r['diff']:+.5f} tol={r['tol']:.3f} [{tag}]"
        )
    print(f"  bucket parity: {'PASS' if cmp['ok'] else 'FAIL'}")
    return out


def build_ref_npz_from_option_buckets(
    stream_npz: Path,
    ref_buckets: np.ndarray,
    out_path: Path,
) -> Path:
    """
    以 stream NPZ 为底,仅用离线参考桶覆盖希腊值/IV/volume 列。
    保留 strike/bid/ask/last,避免 threshold 审计被 747 vs 0 误杀。
    """
    left = np.load(stream_npz, allow_pickle=False)
    payload = {k: left[k] for k in left.files}
    ref = _as_6x12(ref_buckets).astype(np.float32)
    greek_cols = (1, 2, 3, 4, 6, 7)  # delta,gamma,vega,theta,volume,iv
    for key in (
        "option_snapshot_6x12",
        "frozen_option_snapshot_6x12",
        "frozen_latest_opt_buckets_6x12",
    ):
        if key not in payload:
            continue
        base = _as_6x12(payload[key]).astype(np.float32)
        for c in greek_cols:
            # 参考侧有值才覆盖;否则保留 stream(避免把有效成交量打成 0)
            mask = np.abs(ref[:, c]) > 1e-12
            base[mask, c] = ref[mask, c]
        payload[key] = base
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def report_threshold_npz_parity(
    *,
    left_npz: Path,
    right_npz: Path,
    thresholds: Optional[dict] = None,
    strict: bool = False,
) -> dict:
    """直接复用 verify_parity_thresholds.evaluate_threshold_diffs。"""
    left_path = Path(left_npz).expanduser()
    right_path = Path(right_npz).expanduser()
    if not left_path.exists():
        print(f"[threshold-parity] left missing: {left_path}")
        return {"ok": False, "error": f"left missing: {left_path}"}
    if not right_path.exists():
        print(f"[threshold-parity] right missing: {right_path}")
        return {"ok": False, "error": f"right missing: {right_path}"}

    left = np.load(left_path, allow_pickle=False)
    right = np.load(right_path, allow_pickle=False)
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update({str(k): float(v) for k, v in thresholds.items()})

    result = evaluate_threshold_diffs(
        left_map={k: left[k] for k in left.files},
        right_map={k: right[k] for k in right.files},
        thresholds=thr,
        strict=bool(strict),
    )
    failures = list(result.get("failures") or [])
    listed = result.get("listed_results") or []
    ok = len(failures) == 0
    print(f"\n=== Threshold NPZ parity (verify_parity_thresholds) ===")
    print(f"  LEFT : {left_path}")
    print(f"  RIGHT: {right_path}")
    print(f"  common={len(result.get('common') or [])} exact={result.get('exact_match_count', 0)}")
    for key, diff, threshold, passed in listed:
        if passed:
            continue
        print(f"  {key:36s} max_diff={diff:.6f} thr={threshold:.6f} [FAIL]")
    print(f"  threshold parity: {'PASS' if ok else 'FAIL'}")
    if failures and not listed:
        for item in failures[:8]:
            print(f"  - {item}")
    return {
        "ok": bool(ok),
        "left": str(left_path),
        "right": str(right_path),
        "failures": failures,
        "listed_results": [
            {
                "key": k,
                "max_diff": float(d),
                "threshold": float(t),
                "ok": bool(p),
            }
            for k, d, t, p in listed
        ],
        "exact_match_count": int(result.get("exact_match_count") or 0),
        "n_common": len(result.get("common") or []),
        "other_diffs_top": [
            {"key": k, "max_diff": float(d)}
            for k, d in sorted(result.get("other_diffs") or [], key=lambda x: x[1], reverse=True)[:15]
        ],
    }
