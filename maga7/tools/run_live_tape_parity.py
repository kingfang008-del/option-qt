#!/usr/bin/env python3
"""Intraday sentinel: replay session tape/rth → Scanner, compare to live artifacts.

Default: process-local (no Redis). Writes ``tape_parity.json`` under the session dir.

Usage:
  python -m maga7.tools.run_live_tape_parity
  python -m maga7.tools.run_live_tape_parity --session-dir /mnt/s990/data/maga7/live_sessions/...
  python -m maga7.tools.run_live_tape_parity --loop-seconds 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.bar_agg import MultiSymbolMinuteAgg
from maga7.common.config import load_profile, resolve_live_sessions_dir
from maga7.common.contract_select import lock_policy_from_profile
from maga7.common.entry_contract import ContractBooks
from maga7.common.open_lock import resolve_otm_rungs
from maga7.common.replay import to_ny
from maga7.common.signals import StreamSignalState
from maga7.live.live_regime import LiveRegimeGate
from maga7.live.scanner import Mag7Scanner, ScannerSignal

NY = "America/New_York"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _discover_latest_session(root: Path | None = None) -> Path | None:
    root = root or resolve_live_sessions_dir()
    if not root.is_dir():
        return None
    manifests = sorted(root.rglob("manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0].parent if manifests else None


def _load_tape_seconds(
    session_dir: Path,
    *,
    phases: tuple[str, ...] = ("rth",),
    asof_ts: float | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Return {unix_sec: {symbol: ohlcv}} from tape jsonl."""
    frames: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for phase in phases:
        tape_dir = session_dir / "tape" / phase
        if not tape_dir.is_dir():
            continue
        for path in sorted(tape_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(row, dict):
                            continue
                        try:
                            ts = int(float(row["ts"]))
                            sym = str(row["symbol"]).upper()
                            close = float(row["close"])
                        except Exception:
                            continue
                        if asof_ts is not None and ts > float(asof_ts):
                            continue
                        frames[ts][sym] = {
                            "open": float(row.get("open", close)),
                            "high": float(row.get("high", close)),
                            "low": float(row.get("low", close)),
                            "close": close,
                            "volume": float(row.get("volume") or 0.0),
                            "previous_close": float(row.get("previous_close") or 0.0),
                        }
            except OSError:
                continue
    return dict(frames)


def _scanner_from_locks(
    profile: dict[str, Any],
    locks_payload: dict[str, Any],
    *,
    scheme: str,
) -> Mag7Scanner:
    trade = profile.get("trade") or {}
    mode = str(trade.get("contract_mode", "open_ladder")).lower()
    prefer_dte, allowed = lock_policy_from_profile(profile)
    trade_date = str(locks_payload.get("trade_date") or "")
    multi: dict[tuple[str, str], dict[int, dict[int, str]]] = {}
    for symbol, rows in (locks_payload.get("locks") or {}).items():
        by_dte: dict[int, dict[int, str]] = {}
        for lock in rows or []:
            if not isinstance(lock, dict):
                continue
            dte = int(lock.get("front_dte") or 0)
            bucket = int(lock.get("bucket_id") or 0)
            local = str(lock.get("local_symbol") or "").strip()
            if local:
                by_dte.setdefault(dte, {})[bucket] = local
        multi[(str(symbol).upper(), trade_date)] = by_dte
    emit_all = str(scheme).startswith("m5")
    states = {
        symbol: StreamSignalState(symbol, profile["signal"], emit_all=emit_all)
        for symbol in profile["symbols"]
    }
    books = ContractBooks(
        mode=mode,
        multi_idx=multi,
        prefer_dte=prefer_dte,
        allowed_dte=list(allowed),
        clear_otm_thresh=(
            float(trade["clear_otm_ban_0dte_pct"])
            if trade.get("clear_otm_ban_0dte_pct") is not None
            else None
        ),
        ladder=True,
        otm_rungs=resolve_otm_rungs(profile, default=3),
    )
    regime_gate = None
    if bool((profile.get("regime") or {}).get("enabled")):
        regime_gate = LiveRegimeGate(profile.get("regime") or {})
    watchdog = None
    watchdog_snap: dict = {}
    if bool((profile.get("watchdog") or {}).get("enabled")):
        try:
            from maga7.common.watchdog import RegimeWatchdog, snapshot_regime

            watchdog = RegimeWatchdog.from_profile(profile)
            if watchdog is not None and regime_gate is not None:
                watchdog_snap = snapshot_regime(regime_gate.cfg)
        except Exception:
            watchdog = None
            watchdog_snap = {}
    return Mag7Scanner(
        profile=profile,
        states=states,
        books=books,
        minute_agg=MultiSymbolMinuteAgg(profile["symbols"], rth_only=True),
        regime_gate=regime_gate,
        watchdog=watchdog,
        _watchdog_snap=watchdog_snap,
        stock_by={},
        emit_all=emit_all,
    )


def _as_ny_ts(raw: Any) -> pd.Timestamp | None:
    """Parse iso / pandas / unix-seconds into America/New_York timestamp."""
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)) or (
            isinstance(raw, str) and raw.replace(".", "", 1).isdigit()
        ):
            val = float(raw)
            # Heuristic: values in seconds since 2001..2100
            if 1_000_000_000 <= val <= 4_000_000_000:
                return pd.Timestamp(val, unit="s", tz="UTC").tz_convert(NY)
        return to_ny(raw)
    except Exception:
        return None


def _sig_key(sig: Any) -> tuple:
    if isinstance(sig, ScannerSignal):
        ts = to_ny(sig.sig_ts)
        return (
            ts.strftime("%Y-%m-%d %H:%M"),
            str(sig.symbol).upper(),
            str(sig.direction).upper(),
            str(sig.contract or ""),
        )
    if isinstance(sig, dict):
        ts = _as_ny_ts(sig.get("sig_ts") or sig.get("ts") or sig.get("signal_ts"))
        minute = ts.strftime("%Y-%m-%d %H:%M") if ts is not None else str(
            sig.get("sig_ts") or sig.get("ts") or ""
        )
        return (
            minute,
            str(sig.get("symbol") or "").upper(),
            str(sig.get("direction") or sig.get("dir") or "").upper(),
            str(sig.get("contract") or sig.get("local_symbol") or ""),
        )
    return ("", "", "", "")


def _normalize_live_signal(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize orch / scanner / order_events shapes for _sig_key."""
    out = dict(item)
    if not out.get("direction") and out.get("dir"):
        out["direction"] = out["dir"]
    if not out.get("sig_ts"):
        out["sig_ts"] = out.get("ts") or out.get("signal_ts")
    ts = _as_ny_ts(out.get("sig_ts"))
    if ts is not None:
        out["sig_ts"] = ts.isoformat()
    return out


def _load_live_signals(session_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def _add(item: dict[str, Any]) -> None:
        norm = _normalize_live_signal(item)
        key = _sig_key(norm)
        if not key[0] or not key[1] or key[0].startswith("1970-") or key in seen:
            return
        seen.add(key)
        rows.append(norm)

    for name in ("signals.jsonl", "signals.csv"):
        path = session_dir / name
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        _add(item)
        else:
            try:
                df = pd.read_csv(path)
                for item in df.to_dict("records"):
                    if isinstance(item, dict):
                        _add(item)
            except Exception:
                pass

    state = _read_json(session_dir / "scanner_state.json")
    # Prefer actually-emitted signals. ``day_fires`` can include Rule-A seats that
    # later failed regime/peer and never entered ``signals`` / OMS.
    for item in state.get("signals") or []:
        if isinstance(item, dict):
            _add(item)

    # Mid-session: signals.jsonl may be empty until DONE; ENTRY_INTENT embeds signal.
    events_path = session_dir / "order_events.jsonl"
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = str(event.get("kind") or event.get("event") or "").upper()
                nested = event.get("signal")
                if kind == "ENTRY_INTENT" and isinstance(nested, dict):
                    _add(nested)
    # Fallback: day_fires only when nothing else recorded yet (early session).
    if not rows:
        for item in state.get("day_fires") or []:
            if isinstance(item, dict):
                _add(item)
    return rows


def _fill_feedback_events(
    session_dir: Path,
) -> list[tuple[float, str, bool]]:
    """POSITION_CLOSE timeline → (unix_ts, symbol, won) for scanner.record_fill."""
    path = session_dir / "order_events.jsonl"
    if not path.is_file():
        return []
    out: list[tuple[float, str, bool]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("kind") or "").upper() != "POSITION_CLOSE":
                continue
            sym = str(event.get("symbol") or "").upper()
            if not sym:
                continue
            ts = _as_ny_ts(event.get("ts") or event.get("exit_ts"))
            if ts is None:
                continue
            ret = event.get("ret")
            try:
                won = float(ret) > 0.0 if ret is not None else False
            except Exception:
                won = False
            out.append((float(ts.timestamp()), sym, bool(won)))
    out.sort(key=lambda row: row[0])
    return out


def _open_symbols_by_second(session_dir: Path) -> list[tuple[float, set[str]]]:
    """Timeline of open underlyings from order_events (for OMS-aware replay)."""
    path = session_dir / "order_events.jsonl"
    if not path.is_file():
        return []
    events: list[tuple[float, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            kind = str(event.get("kind") or "").upper()
            sym = str(event.get("symbol") or "").upper()
            if not sym:
                continue
            ts = _as_ny_ts(
                event.get("ts")
                or event.get("entry_ts")
                or event.get("created_at")
                or event.get("signal_ts")
            )
            if ts is None:
                continue
            unix = float(ts.timestamp())
            if kind == "POSITION_OPEN":
                events.append((unix, "open", sym))
            elif kind in {"POSITION_CLOSE", "POSITION_CLOSED", "FLAT"}:
                events.append((unix, "close", sym))
    events.sort(key=lambda row: row[0])
    timeline: list[tuple[float, set[str]]] = []
    open_syms: set[str] = set()
    for unix, action, sym in events:
        if action == "open":
            open_syms.add(sym)
        else:
            open_syms.discard(sym)
        timeline.append((unix, set(open_syms)))
    return timeline


def _active_symbols_at(
    timeline: list[tuple[float, set[str]]], when: float
) -> set[str]:
    active: set[str] = set()
    for unix, syms in timeline:
        if unix <= when:
            active = syms
        else:
            break
    return active


def _health_snapshot(session_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    locks = _read_json(session_dir / "locks.json")
    tape_pre = list((session_dir / "tape" / "pre").glob("*.jsonl")) if (session_dir / "tape" / "pre").is_dir() else []
    tape_rth = list((session_dir / "tape" / "rth").glob("*.jsonl")) if (session_dir / "tape" / "rth").is_dir() else []

    def _fresh(paths: list[Path]) -> float | None:
        ages = []
        for path in paths:
            try:
                ages.append(now - path.stat().st_mtime)
            except OSError:
                pass
        return min(ages) if ages else None

    return {
        "manifest_state": manifest.get("state"),
        "lock_status": locks.get("status"),
        "n_lock_symbols": len(locks.get("locks") or {}),
        "tape_pre_files": len(tape_pre),
        "tape_rth_files": len(tape_rth),
        "tape_pre_fresh_sec": _fresh(tape_pre),
        "tape_rth_fresh_sec": _fresh(tape_rth),
        "has_scanner_state": (session_dir / "scanner_state.json").is_file(),
        "has_oms_state": (session_dir / "oms_state.json").is_file(),
        "has_order_events": (session_dir / "order_events.jsonl").is_file(),
    }


def run_parity(
    session_dir: Path,
    *,
    scheme: str = "m5_circuit",
    asof_ts: float | None = None,
    include_pre: bool = False,
    disable_prevention: bool = False,
) -> dict[str, Any]:
    session_dir = Path(session_dir)
    manifest = _read_json(session_dir / "manifest.json")
    locks = _read_json(session_dir / "locks.json")
    health = _health_snapshot(session_dir, manifest)
    report: dict[str, Any] = {
        "ok": False,
        "ts": time.time(),
        "session_dir": str(session_dir),
        "session_id": manifest.get("session_id") or session_dir.name,
        "health": health,
        "issues": [],
        "replay_signals": 0,
        "live_signals": 0,
        "matched": 0,
        "only_live": [],
        "only_replay": [],
    }

    # L0/L1 gates before expensive replay
    if health.get("manifest_state") not in {"RUNNING", "DONE", "PREPARED", "STARTING"}:
        report["issues"].append(f"unexpected_manifest_state={health.get('manifest_state')}")
    if not locks or locks.get("status") not in {"LOCKED", "PARTIAL"}:
        report["issues"].append("locks_missing_or_unlocked")
        report["ok"] = False
        report["stage"] = "pre_lock"
        # Still useful: mark PRE healthy if tape/pre fresh
        if (health.get("tape_pre_files") or 0) >= 1 and (
            health.get("tape_pre_fresh_sec") is None
            or float(health["tape_pre_fresh_sec"]) < 120
        ):
            report["pre_ok"] = True
            report["ok"] = True
            report["note"] = "pre_lock_tape_fresh"
        else:
            report["pre_ok"] = False
            report["issues"].append("pre_tape_stale_or_missing")
        _write_report(session_dir, report)
        return report

    profile_path = manifest.get("profile_path")
    if not profile_path or not Path(profile_path).is_file():
        report["issues"].append("profile_path_missing")
        _write_report(session_dir, report)
        return report
    profile = load_profile(profile_path)
    if disable_prevention:
        # Fair compare vs sessions that traded before prevention was armed.
        import copy

        profile = copy.deepcopy(profile)
        wd = profile.get("watchdog")
        if isinstance(wd, dict) and isinstance(wd.get("prevention"), dict):
            wd["prevention"]["enabled"] = False
    phases = ("pre", "rth") if include_pre else ("rth",)
    frames = _load_tape_seconds(session_dir, phases=phases, asof_ts=asof_ts)
    if not frames:
        report["issues"].append("no_tape_seconds")
        report["stage"] = "no_tape"
        _write_report(session_dir, report)
        return report

    scanner = _scanner_from_locks(profile, locks, scheme=scheme)
    # Mirror live OMS: while a symbol is open, scanner must not re-fire (m5).
    open_timeline = _open_symbols_by_second(session_dir)
    fill_events = _fill_feedback_events(session_dir)
    fill_i = 0
    active_now: set[str] = set()
    scanner.is_symbol_active = lambda sym, _box=active_now: str(sym).upper() in _box
    refs = {"QQQ", "VIXY"}
    trade_syms = {str(s).upper() for s in profile.get("symbols") or []}
    for ts in sorted(frames):
        # Apply closed fills before this second so cooldown / only_win match live.
        while fill_i < len(fill_events) and fill_events[fill_i][0] <= float(ts):
            _, fill_sym, won = fill_events[fill_i]
            scanner.record_fill(
                fill_sym,
                exit_ts=pd.Timestamp(fill_events[fill_i][0], unit="s", tz="UTC").tz_convert(
                    NY
                ),
                won=won,
            )
            fill_i += 1
        active_now.clear()
        active_now.update(_active_symbols_at(open_timeline, float(ts)))
        batch = frames[ts]
        ts_ny = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(NY)
        # references first
        for sym in sorted(set(batch) & refs):
            bar = batch[sym]
            tick = {
                "timestamp": ts_ny,
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
            # Mirror live_engine: feed LiveRegimeGate before Mag7 decisions so
            # qqq_align sees completed QQQ/VIXY (parity previously left
            # regime_missing → phantom DN that live correctly blocked).
            if bar.get("previous_close"):
                tick["previous_close"] = float(bar["previous_close"])
            gate = getattr(scanner, "regime_gate", None)
            if gate is not None and hasattr(gate, "on_stock_second"):
                gate.on_stock_second(sym, tick)
            if hasattr(scanner, "on_reference_second"):
                scanner.on_reference_second(sym, tick)
        for sym in sorted(set(batch) & trade_syms):
            bar = batch[sym]
            tick = {
                "timestamp": ts_ny,
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
            st = scanner.states.get(sym)
            if st is not None and st.prev_close is None and bar.get("previous_close"):
                st.prev_close = float(bar["previous_close"])
            scanner.on_stock_second(sym, tick)

    replay_sigs = [_sig_row(s) for s in scanner.signals]
    live_sigs = _load_live_signals(session_dir)
    # Filter live to asof
    if asof_ts is not None:
        filtered = []
        for row in live_sigs:
            t = row.get("sig_ts") or row.get("ts")
            try:
                if float(pd.Timestamp(t).timestamp()) <= float(asof_ts) + 1.0:
                    filtered.append(row)
            except Exception:
                filtered.append(row)
        live_sigs = filtered

    replay_map = {_sig_key(s): s for s in scanner.signals}
    live_map = {_sig_key(s): s for s in live_sigs}
    only_replay = [list(k) for k in replay_map if k not in live_map]
    only_live = [list(k) for k in live_map if k not in replay_map]
    matched = len(set(replay_map) & set(live_map))

    report.update(
        {
            "stage": "signal_parity",
            "replay_signals": len(replay_map),
            "live_signals": len(live_map),
            "matched": matched,
            "only_replay": only_replay[:50],
            "only_live": only_live[:50],
            "tape_seconds": len(frames),
            "asof_ts": asof_ts,
            "ok": len(only_replay) == 0 and len(only_live) == 0,
        }
    )
    if only_replay or only_live:
        report["issues"].append(
            f"signal_mismatch matched={matched} only_replay={len(only_replay)} only_live={len(only_live)}"
        )
    # Empty-empty is OK (pre-window)
    if not replay_map and not live_map:
        report["ok"] = True
        report["issues"] = [i for i in report["issues"] if "signal_mismatch" not in i]
        report["note"] = "no_signals_yet_both_empty"
    report["fill_feedback_closes"] = len(fill_events)
    report["scheme"] = str(scheme)
    report["disable_prevention"] = bool(disable_prevention)
    report["watchdog_state"] = getattr(scanner, "_watchdog_state", None)
    report["watchdog_reason"] = getattr(scanner, "_watchdog_reason", None)
    report["n_regime_block"] = int(getattr(scanner, "n_regime_block", 0) or 0)
    report["n_peer_block"] = int(getattr(scanner, "n_peer_block", 0) or 0)
    # residual only_replay after fill feedback usually means live suppressed
    # (peer/regime/resume bar truncation) while cold tape still Rule-A fires.
    if only_replay and not only_live and matched > 0:
        report["note"] = (
            "only_replay_residual: live accepted subset; replay still sees extra "
            "Rule-A after OMS fills (peer/regime/resume gaps). Not a feed outage."
        )
    if (
        not replay_map
        and live_map
        and str(report.get("watchdog_reason") or "").startswith("prevention:")
    ):
        report["note"] = (
            "replay_empty_under_prevention: current profile prevention blocks UP; "
            "live book was taken before that overlay. Re-run with --disable-prevention "
            "for fair signal parity on this session."
        )
        report["issues"].append("prevention_profile_vs_live_book")
    if only_live and not only_replay and matched > 0:
        report["note"] = (
            (report.get("note") + " | ") if report.get("note") else ""
        ) + (
            "only_live_residual: often resume skipped max_entries/n_done, or live "
            "EOD-flatten then re-entered (check GOOGL 11:48 EOD + 12:43)."
        )

    _write_report(session_dir, report)
    # also write replay signals for audit
    out = session_dir / "tape_parity_signals.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in replay_sigs:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return report


def _sig_row(sig: ScannerSignal) -> dict[str, Any]:
    return {
        "sig_ts": to_ny(sig.sig_ts).isoformat(),
        "symbol": sig.symbol,
        "direction": sig.direction,
        "contract": sig.contract,
        "spot": sig.spot,
        "bucket_id": getattr(sig, "bucket_id", None),
        "meta": getattr(sig, "meta", None),
    }


def _write_report(session_dir: Path, report: dict[str, Any]) -> None:
    path = Path(session_dir) / "tape_parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", type=str, default="")
    ap.add_argument("--scheme", type=str, default="m5_circuit")
    ap.add_argument("--include-pre", action="store_true")
    ap.add_argument("--asof-now", action="store_true", default=True)
    ap.add_argument("--loop-seconds", type=float, default=0.0, help="0=once; e.g. 600")
    ap.add_argument("--live-root", type=str, default="")
    ap.add_argument(
        "--disable-prevention",
        action="store_true",
        help="Ignore watchdog.prevention for fair compare vs pre-prevention live books",
    )
    args = ap.parse_args(argv)

    def _once() -> int:
        root = Path(args.live_root) if args.live_root else None
        session_dir = (
            Path(args.session_dir) if args.session_dir else _discover_latest_session(root)
        )
        if session_dir is None or not session_dir.is_dir():
            print("ERROR: no live session dir found", file=sys.stderr)
            return 2
        asof = time.time() if args.asof_now else None
        report = run_parity(
            session_dir,
            scheme=args.scheme,
            asof_ts=asof,
            include_pre=bool(args.include_pre),
            disable_prevention=bool(args.disable_prevention),
        )
        print(json.dumps({k: report[k] for k in (
            "ok", "stage", "session_id", "issues", "replay_signals", "live_signals",
            "matched", "health", "note", "pre_ok", "watchdog_reason", "n_regime_block",
        ) if k in report}, ensure_ascii=False, indent=2, default=str))
        print(f"report={session_dir / 'tape_parity.json'}")
        return 0 if report.get("ok") else 1

    if args.loop_seconds and args.loop_seconds > 0:
        while True:
            code = _once()
            print(f"--- loop sleep {args.loop_seconds}s (last_exit={code}) ---", flush=True)
            time.sleep(float(args.loop_seconds))
    return _once()


if __name__ == "__main__":
    raise SystemExit(main())
