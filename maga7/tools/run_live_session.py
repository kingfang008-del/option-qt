#!/usr/bin/env python3
"""Run one auditable Mag7 IBKR Shadow/Paper/Live session."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.bar_agg import MultiSymbolMinuteAgg
from maga7.common.config import load_profile
from maga7.common.contract_select import lock_policy_from_profile
from maga7.common.entry_contract import ContractBooks
from maga7.common.open_lock import resolve_otm_rungs
from maga7.common.provenance import code_fingerprint
from maga7.common.signals import StreamSignalState
from maga7.live.broker_oms import Mag7BrokerOms, _atomic_json
from maga7.live.ibkr_connector import Mag7IbkrConfig, Mag7IbkrConnector
from maga7.live.live_engine import Mag7LiveFrameEngine
from maga7.live.live_regime import LiveRegimeGate
from maga7.live.redis_fused import unpack_obj
from maga7.common.replay import to_ny
from maga7.live.scanner import Mag7Scanner, ScannerSignal, write_signal_audit
from maga7.live.scanner_state import restore_scanner

NY = "America/New_York"


def _now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def _consumes_baseline_topk(signal: ScannerSignal | dict[str, Any]) -> bool:
    if isinstance(signal, dict):
        meta = dict(signal.get("meta") or {})
    else:
        meta = dict(getattr(signal, "meta", None) or {})
    source = str(meta.get("event_source") or "").strip().lower()
    route = str(meta.get("route") or "").strip().lower()
    if source:
        return source == "baseline"
    return route not in {
        "am_pulse",
        "am_pulse_extension",
        "qqq_open_cont",
        "hunt",
        "hunter",
        "satellite",
    }


def _sanitize_resume_topk(scanner: Mag7Scanner) -> int:
    """Remove legacy satellite entries that old resume code put in baseline seats."""
    fires = list(scanner.day_fires or [])
    satellite = [signal for signal in fires if not _consumes_baseline_topk(signal)]
    if not satellite:
        return 0
    baseline = [signal for signal in fires if _consumes_baseline_topk(signal)]
    satellite_symbols = {str(signal.symbol).upper() for signal in satellite}
    baseline_symbols = {str(signal.symbol).upper() for signal in baseline}
    polluted_only = satellite_symbols - baseline_symbols
    scanner.day_fires = baseline
    scanner.day_topk_syms.difference_update(polluted_only)
    hunt_symbols = set(getattr(scanner, "day_hunt_symbols", None) or set())
    for symbol in polluted_only:
        if symbol not in hunt_symbols:
            scanner.n_done[symbol] = sum(
                1 for signal in baseline if str(signal.symbol).upper() == symbol
            )
        state = scanner.states.get(symbol)
        if (
            state is not None
            and bool(getattr(state, "fired_today", False))
            and getattr(state, "first_fire", None)
        ):
            # The raw Rule-A detector fired while a false satellite seat blocked
            # downstream baseline processing. Require a new post-resume bar.
            state.fired_today = False
            state.first_fire = None
    return len(satellite)


def _log_am_pulse_lanes(profile: dict[str, Any]) -> None:
    """Startup self-check so AM / AM_EXT cannot silently stay unarmed."""
    from maga7.common.am_pulse_scout import (
        am_pulse_lane_enabled,
        load_am_pulse_lane_cfg,
    )

    for lane in ("am_pulse", "am_pulse_extension"):
        enabled = am_pulse_lane_enabled(profile, lane)
        if not enabled:
            print(f"AM_PULSE_LANE lane={lane} enabled=false", flush=True)
            continue
        cfg = load_am_pulse_lane_cfg(profile, lane)
        print(
            "AM_PULSE_LANE "
            f"lane={lane} enabled=true "
            f"execute_mode={cfg.get('execute_mode')} "
            f"window={cfg.get('window_start')}-{cfg.get('window_end')} "
            f"flatten_before={cfg.get('flatten_before')} "
            f"arm={cfg.get('arm')} dirs={cfg.get('dirs')}",
            flush=True,
        )


def _seed_scanner_from_order_events(scanner: Mag7Scanner, session_dir: Path) -> None:
    """After resume, rebuild day_fires/n_done from ENTRY_INTENT so m5 won't re-fire."""
    path = Path(session_dir) / "order_events.jsonl"
    if not path.is_file():
        return
    existing = {
        (
            to_ny(sig.sig_ts).strftime("%Y-%m-%d %H:%M"),
            str(sig.symbol).upper(),
            str(sig.direction).upper(),
            str(sig.contract or ""),
        )
        for sig in (scanner.day_fires or [])
    }
    added = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("kind") or "").upper() != "ENTRY_INTENT":
                continue
            raw = event.get("signal")
            if not isinstance(raw, dict):
                continue
            if not _consumes_baseline_topk(raw):
                continue
            try:
                sig = ScannerSignal(
                    date=str(raw.get("date") or ""),
                    symbol=str(raw.get("symbol") or "").upper(),
                    direction=str(raw.get("direction") or "").upper(),
                    sig_ts=to_ny(raw.get("sig_ts")),
                    spot=float(raw.get("spot") or 0.0),
                    rank=int(raw.get("rank") or 0),
                    bucket_id=int(raw.get("bucket_id") or 0),
                    contract=str(raw.get("contract") or "") or None,
                    moneyness=str(raw.get("moneyness") or ""),
                    meta=dict(raw.get("meta") or {}),
                )
            except Exception:
                continue
            key = (
                to_ny(sig.sig_ts).strftime("%Y-%m-%d %H:%M"),
                sig.symbol,
                sig.direction,
                str(sig.contract or ""),
            )
            if key in existing:
                continue
            existing.add(key)
            scanner.day_fires.append(sig)
            if sig not in scanner.signals:
                scanner.signals.append(sig)
            scanner.n_done[sig.symbol] = int(scanner.n_done.get(sig.symbol, 0)) + 1
            scanner.day_topk_syms.add(sig.symbol)
            added += 1
    if added:
        print(f"RESUME seed scanner day_fires from order_events n={added}", flush=True)


def _clock_today(value: str, trade_date: str) -> pd.Timestamp:
    return pd.Timestamp(f"{trade_date} {value}:00", tz=NY)


async def _sleep_until(target: pd.Timestamp, stop: asyncio.Event) -> None:
    while not stop.is_set():
        delay = target.timestamp() - time.time()
        if delay <= 0:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(delay, 30.0))
        except asyncio.TimeoutError:
            pass


def _scanner_from_live_locks(
    profile: dict[str, Any],
    connector: Mag7IbkrConnector,
    *,
    scheme: str,
) -> Mag7Scanner:
    trade = profile.get("trade") or {}
    mode = str(trade.get("contract_mode", "open_ladder")).lower()
    prefer_dte, allowed = lock_policy_from_profile(profile)
    multi: dict[tuple[str, str], dict[int, dict[int, str]]] = {}
    for symbol, locks in connector.locks.items():
        by_dte: dict[int, dict[int, str]] = {}
        for lock in locks:
            by_dte.setdefault(lock.front_dte, {})[lock.bucket_id] = lock.local_symbol
        multi[(symbol, connector.trade_date)] = by_dte
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
    wd_cfg = profile.get("watchdog") or {}
    if bool(wd_cfg.get("enabled")):
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


def _manifest(
    *,
    session_id: str,
    trade_date: str,
    mode: str,
    profile: dict[str, Any],
    state: str,
    connector: Mag7IbkrConnector | None = None,
    engine: Mag7LiveFrameEngine | None = None,
    oms: Mag7BrokerOms | None = None,
    error: str = "",
) -> dict[str, Any]:
    gate = (False, "not_initialized") if oms is None else oms.live_gate()
    g4_evidence = None
    if connector is not None:
        session_dir = Path(connector.session_dir)
        metrics = engine.metrics if engine is not None else None
        required = {
            name: (session_dir / name).is_file()
            for name in (
                "locks.json",
                "signals.jsonl",
                "signals.csv",
                "order_events.jsonl",
            )
        }
        position_closes = 0
        events_path = session_dir / "order_events.jsonl"
        if state in {"DONE", "DONE_WITH_OPEN_POSITIONS", "FAILED"} and events_path.is_file():
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        if json.loads(line).get("kind") == "POSITION_CLOSE":
                            position_closes += 1
                    except (json.JSONDecodeError, AttributeError):
                        continue
        option_tape_files = list((session_dir / "tape").glob("*/options/*.jsonl"))
        checks = {
            "state_done": state == "DONE",
            "mode_shadow": mode == "shadow",
            "data_live": connector.data_mode == "LIVE",
            "lock_complete": connector.lock_status == "LOCKED",
            "frames_positive": bool(metrics is not None and metrics.frames > 0),
            "foreign_zero": bool(metrics is not None and metrics.foreign == 0),
            "rejected_zero": bool(metrics is not None and metrics.rejected == 0),
            "duplicates_zero": bool(metrics is not None and metrics.duplicates == 0),
            "positions_zero": bool(oms is not None and not oms.positions),
            "artifacts_complete": all(required.values()),
            "option_quote_tape": bool(
                option_tape_files
                and int(getattr(connector, "option_tape_quotes", 0)) > 0
            ),
            "round_trip_present": position_closes > 0,
        }
        g4_evidence = {
            "pass": all(checks.values()),
            "checks": checks,
            "required_artifacts": required,
            "option_tape_files": len(option_tape_files),
            "option_tape_frames": int(
                getattr(connector, "option_tape_frames", 0)
            ),
            "option_tape_quotes": int(
                getattr(connector, "option_tape_quotes", 0)
            ),
            "position_closes": position_closes,
        }
    return {
        "schema_version": 1,
        "session_id": session_id,
        "trade_date": trade_date,
        "mode": mode,
        "state": state,
        "updated_at": time.time(),
        "profile_path": profile.get("_profile_path"),
        "live_fingerprint": profile.get("_live_fingerprint"),
        "symbols": profile.get("symbols"),
        "connector": (
            connector._status_payload(state) if connector is not None else None
        ),
        "engine_metrics": engine.metrics.__dict__ if engine is not None else None,
        "oms": (
            {
                "positions": len(oms.positions),
                "active_intents": len(
                    [
                        item
                        for item in oms.intents.values()
                        if item.status not in {"FILLED", "CANCELLED", "ERROR", "REJECTED"}
                    ]
                ),
                "reconcile_ok": oms.reconcile_ok,
                "gate_ok": gate[0],
                "gate_reason": gate[1],
                "profile_hash": oms.profile_hash,
            }
            if oms is not None
            else None
        ),
        "g4_evidence": g4_evidence,
        "error": error,
    }


async def run(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    if args.allow_code_drift:
        profile["_allow_code_drift"] = True
    profile["_live_fingerprint"] = code_fingerprint(
        profile["_profile_path"],
        live=True,
    )
    _log_am_pulse_lanes(profile)
    trade_date = args.date or _now_ny().strftime("%Y-%m-%d")
    if trade_date != _now_ny().strftime("%Y-%m-%d") and not args.prepare_only:
        raise SystemExit("live session date must equal current America/New_York date")
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=trade_date,
        end_date=trade_date,
    )
    if schedule.empty:
        raise SystemExit(f"{trade_date} is not an NYSE trading session")
    market_open = pd.Timestamp(schedule.iloc[0]["market_open"]).tz_convert(NY)
    market_close = pd.Timestamp(schedule.iloc[0]["market_close"]).tz_convert(NY)
    if args.resume and not args.session_id:
        raise SystemExit("--resume requires --session-id")
    session_id = args.session_id or (
        f"live_{trade_date.replace('-', '')}_{_now_ny().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    live_root = Path(
        profile["_paths"].get("live_sessions_dir")
        or Path(profile["_paths"]["results_dir"]) / "live_sessions"
    )
    session_dir = live_root / trade_date / session_id
    previous = {}
    if args.resume:
        if not session_dir.is_dir():
            raise SystemExit(f"resume session does not exist: {session_dir}")
        previous = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("trade_date") != trade_date:
            raise SystemExit("resume manifest trade date mismatch")
        if previous.get("mode") != args.mode:
            raise SystemExit("resume manifest mode mismatch")
        previous_profile = str(previous.get("profile_path") or "")
        current_profile = str(profile.get("_profile_path") or "")
        if (
            previous_profile
            and current_profile
            and Path(previous_profile).resolve() != Path(current_profile).resolve()
        ):
            raise SystemExit("resume manifest profile path mismatch")
        previous_fp = str(previous.get("live_fingerprint") or "")
        current_fp = str(profile.get("_live_fingerprint") or "")
        if (
            previous_fp
            and current_fp
            and previous_fp != current_fp
            and not args.allow_code_drift
        ):
            raise SystemExit(
                "resume code fingerprint mismatch; review the diff and pass "
                "--allow-code-drift explicitly"
            )
    else:
        session_dir.mkdir(parents=True, exist_ok=False)
    lock_handle = (session_dir / ".session.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"session already running: {session_id}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()
    manifest_path = session_dir / "manifest.json"
    _atomic_json(
        manifest_path,
        _manifest(
            session_id=session_id,
            trade_date=trade_date,
            mode=args.mode,
            profile=profile,
            state="RESUMING" if args.resume else "STARTING",
        ),
    )

    if args.mode == "live" and not args.live_orders:
        raise SystemExit("--mode live additionally requires --live-orders")
    if args.mode in {"paper", "live"} and not args.account:
        raise SystemExit("--mode paper/live requires explicit --account")
    default_port = 4001 if args.mode == "live" else 4002
    config = Mag7IbkrConfig(
        host=args.ib_host,
        port=args.ib_port or default_port,
        client_id=args.client_id,
        account=args.account,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        market_data_type=args.market_data_type,
        max_option_subscriptions=args.max_option_subscriptions,
        preferred_dte=int((profile.get("trade") or {}).get("prefer_dte", 0)),
        md_role=str(args.md_role or "combined"),
    )
    _, allowed = lock_policy_from_profile(profile)
    otm_rungs = resolve_otm_rungs(profile, default=3)
    connector = Mag7IbkrConnector(
        session_id=session_id,
        symbols=list(profile["symbols"]),
        reference_symbols=[
            symbol
            for symbol in ("QQQ", "VIXY")
            if (
                symbol == "QQQ"
                and (
                    bool((profile.get("regime") or {}).get("qqq_align", False))
                    or bool((profile.get("regime") or {}).get("qqq_mf10_align", False))
                    # Watchdog Halt/Hunt needs QQQ morning bars in stock_by
                    or bool((profile.get("watchdog") or {}).get("enabled", False))
                    # Hold Watchdog mid-hold flatten also needs live QQQ levels
                    or bool(
                        ((profile.get("trade") or {}).get("hold_watchdog") or {}).get(
                            "enabled", False
                        )
                    )
                )
            )
            or (
                symbol == "VIXY"
                and (
                    (profile.get("regime") or {}).get("vix_reversal_max") is not None
                    or (profile.get("regime") or {}).get("put_vixy_z_min") is not None
                )
            )
        ],
        trade_date=trade_date,
        session_dir=session_dir,
        config=config,
        allowed_dte=tuple(allowed),
        otm_rungs=otm_rungs,
        resume=args.resume,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    scanner = None
    oms = None
    engine = None
    tasks: list[asyncio.Task] = []
    try:
        await connector.connect(retries=args.connect_retries)
        await connector.subscribe_stocks()
        # Capture the full RTH stock path while contract discovery is running;
        # the fresh consumer group will replay these buffered frames after lock.
        tasks = [
            asyncio.create_task(connector.publish_loop(), name="market-publisher"),
            asyncio.create_task(connector.heartbeat_loop(), name="ibkr-heartbeat"),
        ]
        if str(getattr(connector, "md_role", "combined")).lower() == "options":
            tasks.append(
                asyncio.create_task(
                    connector.stock_ingest_loop(), name="stock-ingest"
                )
            )
            # Wait briefly for external stock publisher ticks before lock.
            deadline = time.time() + float(args.stock_ready_timeout)
            while time.time() < deadline and not stop.is_set():
                live_n = sum(
                    1
                    for symbol in connector.symbols
                    if float(connector.last_stock_tick.get(symbol, 0.0) or 0.0) > 0
                )
                if live_n >= max(1, len(connector.symbols) // 2):
                    print(
                        f"EXTERNAL_STOCK_READY live_symbols={live_n}/{len(connector.symbols)}",
                        flush=True,
                    )
                    break
                await asyncio.sleep(0.5)
            else:
                if not stop.is_set():
                    print(
                        "WARN external stock publisher not ready — "
                        "lock may wait/fail on spots",
                        flush=True,
                    )
        restored = await connector.restore_locks() if args.resume else False
        if not restored:
            lock_at = (
                market_open
                if args.lock_time == "auto"
                else _clock_today(args.lock_time, trade_date)
            )
            if args.prelock_time != "off":
                prelock_at = (
                    lock_at - pd.Timedelta(minutes=10)
                    if args.prelock_time == "auto"
                    else _clock_today(args.prelock_time, trade_date)
                )
                if prelock_at >= lock_at:
                    raise RuntimeError(
                        f"prelock time must be before lock time: {prelock_at} >= {lock_at}"
                    )
                await _sleep_until(prelock_at, stop)
                if stop.is_set():
                    return 130
                await connector.prepare_contract_candidates()
            await _sleep_until(lock_at, stop)
            if stop.is_set():
                return 130
            await connector.lock_and_subscribe(
                timeout_sec=args.spot_timeout,
                min_tick_ts=lock_at.timestamp(),
            )
        shadow_unavailable_only = (
            args.mode == "shadow"
            and connector.lock_status == "PARTIAL"
            and bool(connector.locks)
            and bool(connector.errors)
            and all(
                reason == "empty contract universe"
                for reason in connector.errors.values()
            )
        )
        if connector.lock_status != "LOCKED" and not shadow_unavailable_only:
            raise RuntimeError(
                f"open lock incomplete: {connector.lock_status} {connector.errors}"
            )
        if shadow_unavailable_only:
            print(
                "SHADOW_LOCK_PARTIAL unavailable symbols excluded: "
                f"{connector.errors}",
                flush=True,
            )
        if args.prepare_only:
            _atomic_json(
                manifest_path,
                _manifest(
                    session_id=session_id,
                    trade_date=trade_date,
                    mode=args.mode,
                    profile=profile,
                    state="PREPARED",
                    connector=connector,
                ),
            )
            return 0

        scanner = _scanner_from_live_locks(profile, connector, scheme=args.scheme)
        if not hasattr(scanner, "drain_am_pulse_extension"):
            raise RuntimeError(
                "scanner missing drain_am_pulse_extension — AM_EXT cannot arm; "
                "restart with current maga7.live.scanner"
            )
        from maga7.live.rth_open_store import (
            merge_rth_opens,
            missing_rth_open_symbols,
            resolve_rth_opens,
            seed_scanner_day_opens,
        )

        scanner._rth_open_live_root = str(live_root)
        scanner._rth_open_redis = connector.redis
        rth_symbols = [str(s).upper() for s in (profile.get("symbols") or [])] + ["QQQ"]
        rth_opens = resolve_rth_opens(
            live_root,
            trade_date,
            redis_client=connector.redis,
            symbols=rth_symbols,
            recover_tapes=True,
        )
        missing_opens = missing_rth_open_symbols(rth_opens, rth_symbols)
        if missing_opens:
            try:
                hist_opens = await connector.fetch_rth_opens_historical(missing_opens)
            except Exception as exc:
                hist_opens = {}
                print(f"WARN RTH_OPEN_HIST_FAILED err={exc}", flush=True)
            if hist_opens:
                rth_opens = merge_rth_opens(rth_opens, hist_opens)
                print(
                    f"RTH_OPEN_HIST n={len(hist_opens)} symbols={sorted(hist_opens)}",
                    flush=True,
                )
            still_missing = missing_rth_open_symbols(rth_opens, rth_symbols)
            if still_missing:
                print(
                    f"WARN RTH_OPEN_MISSING symbols={still_missing}",
                    flush=True,
                )
        from maga7.common.event_calendar import resolve_live_event_blackout

        event_blackout, event_meta = resolve_live_event_blackout(
            profile,
            trade_date=trade_date,
            redis_client=connector.redis,
        )
        scanner.set_event_blackout(event_blackout, event_meta)
        if event_meta.get("active_today_full") or event_meta.get("active_today"):
            print(
                f"EVENT_BLACKOUT CORE today={trade_date} "
                f"sources={event_meta.get('sources')} "
                f"dates={event_meta.get('blackout_dates')}",
                flush=True,
            )
        elif event_meta.get("active_today_symbols"):
            print(
                f"EVENT_BLACKOUT SYMBOL today={trade_date} "
                f"symbols={event_meta.get('active_today_symbols')} "
                f"sources={event_meta.get('sources')}",
                flush=True,
            )
        if args.resume:
            raw_state = connector.redis.get(f"maga7:scanner_state:{session_id}")
            redis_state = unpack_obj(raw_state) if raw_state is not None else None
            state_path = session_dir / "scanner_state.json"
            disk_state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else None
            )
            candidates = [
                state
                for state in (redis_state, disk_state)
                if isinstance(state, dict)
                and state.get("session_id") == session_id
            ]
            scanner_state = (
                max(candidates, key=lambda state: float(state.get("frame_ts") or 0.0))
                if candidates
                else None
            )
            if isinstance(scanner_state, dict):
                saved_fp = str(scanner_state.get("live_fingerprint") or "")
                live_fp = str(profile["_live_fingerprint"] or "")
                if saved_fp and live_fp and saved_fp != live_fp:
                    if not args.allow_code_drift:
                        raise RuntimeError(
                            "scanner state live fingerprint mismatch"
                        )
                    print(
                        "WARN resume fingerprint mismatch "
                        f"saved={saved_fp[:12]} live={live_fp[:12]} "
                        "(explicit --allow-code-drift)",
                        flush=True,
                    )
                restore_scanner(scanner, scanner_state)
                sanitized = _sanitize_resume_topk(scanner)
                if sanitized:
                    print(
                        "RESUME removed satellite signals from baseline TopK "
                        f"n={sanitized}",
                        flush=True,
                    )
            elif previous.get("state") not in {"PREPARED", "STARTING"}:
                raise RuntimeError("resume requires scanner state")
        # Seed AFTER restore so durable 09:30 opens overwrite snapshot null /
        # late pseudo latch (mid-day restart / restart-options).
        if rth_opens:
            seeded = seed_scanner_day_opens(scanner, rth_opens, force=True)
            print(
                f"RTH_OPEN_RESTORE n={len(rth_opens)} seeded={seeded} "
                f"symbols={sorted(rth_opens)}",
                flush=True,
            )
        oms = Mag7BrokerOms(
            profile=profile,
            scanner=scanner,
            connector=connector,
            session_id=session_id,
            trade_date=trade_date,
            session_dir=session_dir,
            mode=args.mode,
            max_qty=args.max_qty,
            equity=args.shadow_equity,
        )
        connector.on_reconnect = oms.on_feed_reconnected
        if args.resume:
            _seed_scanner_from_order_events(scanner, session_dir)
        # The profile-level macro calendar is a CORE baseline gate. A/B sleeves
        # remain independent and may opt in with their own event_calendar_block.
        if event_meta.get("active_today_full") or event_meta.get("active_today"):
            oms._event(
                "EVENT_BLACKOUT",
                {
                    "trade_date": trade_date,
                    "scope": "core",
                    "sources": event_meta.get("sources"),
                    "blackout_dates": event_meta.get("blackout_dates"),
                },
            )
        elif event_meta.get("active_today_symbols"):
            oms._event(
                "EVENT_BLACKOUT",
                {
                    "trade_date": trade_date,
                    "scope": "symbol",
                    "symbols": event_meta.get("active_today_symbols"),
                    "sources": event_meta.get("sources"),
                    "symbol_blackout": event_meta.get("symbol_blackout"),
                },
            )
        await oms.initialize_account()
        await oms.recover_broker_activity()
        await oms.reconcile()
        if args.mode in {"paper", "live"}:
            gate_ok, gate_reason = oms.live_gate()
            if not gate_ok:
                raise RuntimeError(f"{args.mode} trading gate blocked: {gate_reason}")
        engine = Mag7LiveFrameEngine(
            redis_client=connector.redis,
            session_id=session_id,
            scanner=scanner,
            oms=oms,
            connector=connector,
            consumer_name=f"maga7_live_{os.getpid()}",
        )
        tasks.extend(
            [
                asyncio.create_task(engine.run(), name="frame-engine"),
                asyncio.create_task(oms.reconcile_loop(), name="broker-reconcile"),
                asyncio.create_task(oms.order_watchdog_loop(), name="order-watchdog"),
                asyncio.create_task(
                    oms.broker_recovery_loop(),
                    name="broker-recovery",
                ),
            ]
        )
        _atomic_json(
            manifest_path,
            _manifest(
                session_id=session_id,
                trade_date=trade_date,
                mode=args.mode,
                profile=profile,
                state="RUNNING",
                connector=connector,
                engine=engine,
                oms=oms,
            ),
        )

        end_at = (
            _now_ny() + pd.Timedelta(seconds=args.run_seconds)
            if args.run_seconds > 0
            else (
                market_close - pd.Timedelta(minutes=5)
                if args.end_time == "auto"
                else _clock_today(args.end_time, trade_date)
            )
        )
        monitor = asyncio.create_task(_sleep_until(end_at, stop), name="session-clock")
        watched = tasks + [monitor]
        while not stop.is_set():
            done, _ = await asyncio.wait(
                watched,
                timeout=5.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is monitor:
                    stop.set()
                    break
                exc = task.exception()
                if exc is not None:
                    raise exc
            _atomic_json(
                manifest_path,
                _manifest(
                    session_id=session_id,
                    trade_date=trade_date,
                    mode=args.mode,
                    profile=profile,
                    state="RUNNING",
                    connector=connector,
                    engine=engine,
                    oms=oms,
                ),
            )

        flatten_deadline = time.time() + args.eod_fill_wait
        while oms.positions and time.time() < flatten_deadline:
            oms.force_flatten("EOD")
            await asyncio.sleep(1.0)
        await oms.reconcile()
        write_signal_audit(scanner.signals, session_dir / "signals.jsonl")
        state = "DONE" if not oms.positions else "DONE_WITH_OPEN_POSITIONS"
        _atomic_json(
            manifest_path,
            _manifest(
                session_id=session_id,
                trade_date=trade_date,
                mode=args.mode,
                profile=profile,
                state=state,
                connector=connector,
                engine=engine,
                oms=oms,
            ),
        )
        return 0 if state == "DONE" else 2
    except Exception as exc:
        _atomic_json(
            manifest_path,
            _manifest(
                session_id=session_id,
                trade_date=trade_date,
                mode=args.mode,
                profile=profile,
                state="FAILED",
                connector=connector,
                engine=engine,
                oms=oms,
                error=str(exc),
            ),
        )
        raise
    finally:
        if oms is not None:
            oms.cancel_open_orders()
        if engine is not None:
            engine.stop()
        connector.stop()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--mode", choices=["shadow", "paper", "live"], default="shadow")
    parser.add_argument("--live-orders", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--scheme", default="m5")
    parser.add_argument(
        "--prelock-time",
        default="auto",
        help="auto=10 minutes before lock; HH:MM explicit; off=legacy one-phase lock",
    )
    parser.add_argument("--lock-time", default="auto")
    parser.add_argument("--end-time", default="auto")
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=212)
    parser.add_argument(
        "--md-role",
        choices=["combined", "options", "stock"],
        default="combined",
        help="combined=legacy one process; options=external stock MD publisher",
    )
    parser.add_argument(
        "--stock-ready-timeout",
        type=float,
        default=20.0,
        help="When md-role=options, seconds to wait for external stock ticks",
    )
    parser.add_argument("--account", default="")
    parser.add_argument("--market-data-type", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--spot-timeout", type=float, default=60.0)
    parser.add_argument("--max-option-subscriptions", type=int, default=90)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--max-qty", type=int, default=1)
    parser.add_argument("--shadow-equity", type=float, default=100_000.0)
    parser.add_argument("--eod-fill-wait", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
