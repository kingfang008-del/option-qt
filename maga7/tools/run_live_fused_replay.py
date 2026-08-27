#!/usr/bin/env python3
"""Replay an existing live session fused Redis stream with current code.

Reads ``fused_market_stream:maga7:<session_id>`` (stocks + option_contracts),
runs Mag7Scanner + Mag7OmsStub (prefer Redis quotes), writes artifacts under
``<session_dir>/fused_replay_<tag>/``.

Usage:
  python -m maga7.tools.run_live_fused_replay \\
    --session-dir /mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e

  # research baseline scheme (profile recommended_scheme)
  python -m maga7.tools.run_live_fused_replay --session-dir ... --scheme single
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from maga7.common.config import load_profile
from maga7.common.provenance import code_fingerprint
from maga7.common.replay import month_list, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files, resolve_mf_fast_window
from maga7.live.oms_stub import Mag7OmsStub
from maga7.live.redis_fused import redis_client, run_keys, unpack_batch
from maga7.live.scanner import Mag7Scanner, write_signal_audit
from maga7.tools.run_live_tape_parity import _scanner_from_locks
from maga7.tools.run_oms_dry_run import _seed_prev_closes, _stock_load_start

logger = logging.getLogger("maga7.fused_replay")
NY = ZoneInfo("America/New_York")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_live_closes(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "order_events.jsonl"
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if str(row.get("kind") or "").upper() != "POSITION_CLOSE":
            continue
        ts = row.get("ts")
        try:
            t = pd.Timestamp(float(ts), unit="s", tz="UTC").tz_convert(NY)
        except Exception:
            t = None
        out.append(
            {
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "contract": row.get("contract"),
                "reason": row.get("reason"),
                "ret": row.get("ret"),
                "exit_ts": t.isoformat() if t is not None else None,
                "entry_ts": row.get("entry_ts"),
            }
        )
    return out


def _compare_trades(
    replay: list[Any], live_closes: list[dict[str, Any]]
) -> dict[str, Any]:
    def key(sym: str, direction: str, contract: str) -> tuple[str, str, str]:
        return (str(sym or "").upper(), str(direction or "").upper(), str(contract or "").replace("O:", "").strip())

    r_map = {
        key(t.symbol, t.direction, t.contract): t for t in replay
    }
    l_map = {
        key(c.get("symbol"), c.get("direction"), c.get("contract")): c for c in live_closes
    }
    matched = sorted(set(r_map) & set(l_map))
    only_r = sorted(set(r_map) - set(l_map))
    only_l = sorted(set(l_map) - set(r_map))
    return {
        "matched": len(matched),
        "only_replay": [list(k) for k in only_r],
        "only_live": [list(k) for k in only_l],
        "replay_n": len(r_map),
        "live_n": len(l_map),
    }


def _preload_stock_lookback(scanner: Mag7Scanner, profile: dict[str, Any], trade_date: str) -> dict[str, Any]:
    """Seed prior sessions for mf_idio / peer / from_prev; leave stock_by unfrozen for tape day."""
    load_start = _stock_load_start(profile, trade_date)
    # Disk may lag the live trade_date — preload through last available day before trade_date.
    end = (pd.Timestamp(trade_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if end < load_start:
        return {"load_start": load_start, "end": end, "n_symbols": 0, "skipped": True}
    sig = profile.get("signal") or {}
    paths = profile["_paths"]
    months = month_list(load_start, end)
    load_syms = list(
        dict.fromkeys(
            list(profile.get("symbols") or [])
            + list(sig.get("peer_symbols") or [])
            + ["QQQ"]
        )
    )
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in load_syms:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= load_start) & (raw["date"] <= end)]
        if raw.empty:
            continue
        stock_by[sym] = attach_mf_features(
            raw,
            mf_window=int(sig.get("mf_window", 10)),
            vol_ma_window=int(sig.get("vol_ma_window", 20)),
            mf_fast_window=resolve_mf_fast_window(sig),
        )
    scanner.stock_by = stock_by
    scanner.stock_by_frozen = False
    _seed_prev_closes(scanner, trade_date)
    return {
        "load_start": load_start,
        "end": end,
        "n_symbols": len(stock_by),
        "n_rows": {k: int(len(v)) for k, v in stock_by.items()},
        "skipped": False,
    }


def run_fused_replay(
    session_dir: Path,
    *,
    scheme: str | None = None,
    redis_host: str = "127.0.0.1",
    redis_port: int = 6379,
    redis_db: int = 0,
    disable_prevention: bool = False,
    disable_am_pulse: bool = False,
    tag: str | None = None,
    trade_overrides: dict[str, Any] | None = None,
    fill_overrides: dict[str, Any] | None = None,
    profile_path_override: str | Path | None = None,
) -> dict[str, Any]:
    session_dir = Path(session_dir)
    manifest = _read_json(session_dir / "manifest.json")
    locks = _read_json(session_dir / "locks.json")
    session_id = str(manifest.get("session_id") or session_dir.name)
    profile_path = profile_path_override or manifest.get("profile_path")
    if not profile_path or not Path(profile_path).is_file():
        raise SystemExit(f"profile_path missing: {profile_path}")
    profile = load_profile(profile_path)
    trade_date = str(
        manifest.get("trade_date")
        or locks.get("trade_date")
        or "2026-07-20"
    )
    profile["date_range"] = {"start": trade_date, "end": trade_date}
    if trade_overrides:
        trade = profile.setdefault("trade", {})
        for k, v in trade_overrides.items():
            trade[k] = v
    if fill_overrides:
        fill = profile.setdefault("fill", {})
        for k, v in fill_overrides.items():
            fill[k] = v
    if disable_prevention:
        wd = profile.get("watchdog")
        if isinstance(wd, dict) and isinstance(wd.get("prevention"), dict):
            wd["prevention"]["enabled"] = False
    if disable_am_pulse:
        for lane in ("am_pulse", "am_pulse_extension"):
            block = profile.get(lane)
            if isinstance(block, dict):
                block["enabled"] = False

    scheme_u = str(
        scheme
        or profile.get("recommended_scheme")
        or "m5_circuit"
    ).strip()
    fingerprint = code_fingerprint(profile["_profile_path"])

    r = redis_client(host=redis_host, port=redis_port, db=redis_db)
    keys = run_keys(session_id)
    stream = keys["stream"]
    n_stream = int(r.xlen(stream))
    if n_stream <= 0:
        raise SystemExit(f"empty fused stream: {stream}")

    scanner = _scanner_from_locks(profile, locks, scheme=scheme_u)
    lookback_info = _preload_stock_lookback(scanner, profile, trade_date)
    stub = Mag7OmsStub.from_profile(
        profile,
        prefer_redis_quotes=True,
        redis_publish=False,
        scanner=scanner,
    )
    # Block re-fire while stub still has an unresolved pending for the symbol.
    def _active(sym: str) -> bool:
        sym_u = str(sym).upper()
        sess = stub._session
        if sess is None:
            return False
        return any(str(p.sig.symbol).upper() == sym_u for p in (sess.pending or []))

    scanner.is_symbol_active = _active

    n_frames = 0
    n_signals = 0
    t0 = time.time()
    # Inclusive chunks; advance with exclusive '(' start (Redis 6.2+).
    cursor = "-"
    while True:
        batch_rows = r.xrange(stream, min=cursor, max="+", count=500)
        if not batch_rows:
            break
        for mid, fields in batch_rows:
            raw = fields.get(b"batch") if isinstance(fields, dict) else None
            if raw is None:
                continue
            batch = unpack_batch(raw)
            if not batch:
                continue
            ts_values = {
                float(p["ts"])
                for p in batch
                if isinstance(p, dict) and p.get("ts") is not None
            }
            if len(ts_values) != 1:
                continue
            ts_val = next(iter(ts_values))

            # Options first (same as live_engine / redis_consumer).
            for payload in batch:
                if not isinstance(payload, dict):
                    continue
                stub.ingest_option_contracts(
                    str(payload.get("symbol") or ""),
                    float(payload.get("ts") or ts_val),
                    list(payload.get("option_contracts") or []),
                    resolve_pending=False,
                )

            stub.try_resolve_pending(asof_ts=ts_val)

            n_sig_before = len(scanner.signals)
            signals = []
            # References (QQQ/VIXY) before Mag7 names.
            for payload in batch:
                if not isinstance(payload, dict):
                    continue
                sym = str(payload.get("symbol") or "").upper()
                stock = payload.get("stock") or {}
                if not stock:
                    continue
                tick = {
                    "timestamp": pd.Timestamp(ts_val, unit="s", tz="UTC").tz_convert(NY),
                    "open": float(stock.get("open") or stock.get("close") or 0.0),
                    "high": float(stock.get("high") or stock.get("close") or 0.0),
                    "low": float(stock.get("low") or stock.get("close") or 0.0),
                    "close": float(stock.get("close") or 0.0),
                    "volume": float(stock.get("volume") or 0.0),
                    "previous_close": float(stock.get("previous_close") or 0.0),
                }
                if sym not in scanner.states:
                    gate = getattr(scanner, "regime_gate", None)
                    if gate is not None and hasattr(gate, "on_stock_second"):
                        gate.on_stock_second(sym, tick)
                    if hasattr(scanner, "on_reference_second"):
                        scanner.on_reference_second(sym, tick)

            for payload in batch:
                if not isinstance(payload, dict):
                    continue
                sym = str(payload.get("symbol") or "").upper()
                if sym not in scanner.states:
                    continue
                stock = payload.get("stock") or {}
                if not stock:
                    continue
                tick = {
                    "timestamp": pd.Timestamp(ts_val, unit="s", tz="UTC").tz_convert(NY),
                    "open": float(stock.get("open") or stock.get("close") or 0.0),
                    "high": float(stock.get("high") or stock.get("close") or 0.0),
                    "low": float(stock.get("low") or stock.get("close") or 0.0),
                    "close": float(stock.get("close") or 0.0),
                    "volume": float(stock.get("volume") or 0.0),
                    "previous_close": float(stock.get("previous_close") or 0.0),
                }
                st = scanner.states.get(sym)
                if (
                    st is not None
                    and getattr(st, "prev_close", None) is None
                    and tick["previous_close"] > 0
                ):
                    st.prev_close = tick["previous_close"]
                sig = scanner.on_stock_second(sym, tick)
                if sig is not None:
                    signals.append(sig)

            for sig in scanner.signals[n_sig_before:]:
                if sig not in signals:
                    signals.append(sig)
            frame_ts = pd.Timestamp(ts_val, unit="s", tz="UTC").tz_convert(NY)
            for sig in scanner.drain_hunts(frame_ts):
                if sig not in signals:
                    signals.append(sig)

            for sig in signals:
                stub.process_one(sig)
                n_signals += 1
            n_frames += 1

        last_mid = batch_rows[-1][0]
        cursor = b"(" + last_mid if isinstance(last_mid, (bytes, bytearray)) else "(" + str(last_mid)
        if len(batch_rows) < 500:
            break

    stub.flush_pending()
    elapsed = time.time() - t0
    stamp = datetime.now(tz=NY).strftime("%H%M%S")
    out_tag = tag or (
        f"{scheme_u}"
        f"{'_noprev' if disable_prevention else ''}"
        f"{'_nopulse' if disable_am_pulse else ''}"
        f"_{stamp}"
    )
    out_dir = session_dir / f"fused_replay_{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = stub.finalize_summary(n_signals=len(scanner.signals))
    live_closes = _load_live_closes(session_dir)
    cmp = _compare_trades(stub.trades, live_closes)
    summary.update(
        {
            "mode": "LIVE_FUSED_REPLAY",
            "session_id": session_id,
            "session_dir": str(session_dir),
            "stream": stream,
            "scheme": scheme_u,
            "disable_prevention": bool(disable_prevention),
            "disable_am_pulse": bool(disable_am_pulse),
            "stock_lookback": lookback_info,
            "profile_path": str(profile_path),
            "code_fingerprint": fingerprint,
            "trade_date": trade_date,
            "n_stream": n_stream,
            "n_frames": n_frames,
            "n_scanner_signals": len(scanner.signals),
            "n_process_calls": n_signals,
            "elapsed_sec": round(elapsed, 2),
            "watchdog_state": getattr(scanner, "_watchdog_state", None),
            "watchdog_reason": getattr(scanner, "_watchdog_reason", None),
            "n_regime_block": int(getattr(scanner, "n_regime_block", 0) or 0),
            "n_peer_block": int(getattr(scanner, "n_peer_block", 0) or 0),
            "vs_live": cmp,
        }
    )
    stub.write(out_dir)
    write_signal_audit(scanner.signals, out_dir / "signals.jsonl")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # compact trade table
    if stub.trades:
        pd.DataFrame([t.__dict__ for t in stub.trades]).to_csv(
            out_dir / "trades.csv", index=False
        )
    logger.info(
        "fused replay done frames=%d signals=%d trades=%d → %s",
        n_frames,
        len(scanner.signals),
        len(stub.trades),
        out_dir,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument(
        "--scheme",
        default="",
        help="single | m5 | m5_circuit (default: profile recommended_scheme or m5_circuit)",
    )
    ap.add_argument("--redis-host", default="127.0.0.1")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--redis-db", type=int, default=0)
    ap.add_argument(
        "--disable-prevention",
        action="store_true",
        help="turn off watchdog.prevention for fair compare to pre-prevention live book",
    )
    ap.add_argument(
        "--disable-am-pulse",
        action="store_true",
        help="turn off am_pulse + am_pulse_extension sleeves (CORE-only fused replay)",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    summary = run_fused_replay(
        Path(args.session_dir),
        scheme=args.scheme or None,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        disable_prevention=bool(args.disable_prevention),
        disable_am_pulse=bool(args.disable_am_pulse),
        tag=args.tag or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
