#!/usr/bin/env python3
"""Run one auditable Mag7 IBKR Shadow/Paper/Live session."""
from __future__ import annotations

import argparse
import asyncio
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
from maga7.live.scanner import Mag7Scanner, write_signal_audit
from maga7.live.scanner_state import restore_scanner

NY = "America/New_York"


def _now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


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
        otm_rungs=resolve_otm_rungs(profile, default=5),
    )
    regime_gate = None
    if bool((profile.get("regime") or {}).get("enabled")):
        regime_gate = LiveRegimeGate(profile.get("regime") or {})
    return Mag7Scanner(
        profile=profile,
        states=states,
        books=books,
        minute_agg=MultiSymbolMinuteAgg(profile["symbols"], rth_only=True),
        regime_gate=regime_gate,
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
        "error": error,
    }


async def run(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    profile["_live_fingerprint"] = code_fingerprint(
        profile["_profile_path"],
        live=True,
    )
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
    results_root = Path(profile["_paths"]["results_dir"])
    session_dir = results_root / "live_sessions" / trade_date / session_id
    previous = {}
    if args.resume:
        if not session_dir.is_dir():
            raise SystemExit(f"resume session does not exist: {session_dir}")
        previous = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("trade_date") != trade_date:
            raise SystemExit("resume manifest trade date mismatch")
        if previous.get("mode") != args.mode:
            raise SystemExit("resume manifest mode mismatch")
    else:
        session_dir.mkdir(parents=True, exist_ok=False)
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
    )
    _, allowed = lock_policy_from_profile(profile)
    otm_rungs = resolve_otm_rungs(profile, default=5)
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
        restored = await connector.restore_locks() if args.resume else False
        if not restored:
            lock_at = (
                market_open
                if args.lock_time == "auto"
                else _clock_today(args.lock_time, trade_date)
            )
            await _sleep_until(lock_at, stop)
            if stop.is_set():
                return 130
            await connector.lock_and_subscribe(
                timeout_sec=args.spot_timeout,
                min_tick_ts=lock_at.timestamp(),
            )
        if connector.lock_status != "LOCKED":
            raise RuntimeError(
                f"open lock incomplete: {connector.lock_status} {connector.errors}"
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
        from maga7.common.event_calendar import resolve_live_event_blackout

        event_blackout, event_meta = resolve_live_event_blackout(
            profile,
            trade_date=trade_date,
            redis_client=connector.redis,
        )
        scanner.set_event_blackout(event_blackout, event_meta)
        if event_meta.get("active_today"):
            print(
                f"EVENT_BLACKOUT active today={trade_date} "
                f"sources={event_meta.get('sources')} "
                f"dates={event_meta.get('blackout_dates')}",
                flush=True,
            )
        if args.resume:
            raw_state = connector.redis.get(f"maga7:scanner_state:{session_id}")
            scanner_state = unpack_obj(raw_state) if raw_state is not None else None
            if not isinstance(scanner_state, dict):
                state_path = session_dir / "scanner_state.json"
                scanner_state = (
                    json.loads(state_path.read_text(encoding="utf-8"))
                    if state_path.is_file()
                    else None
                )
            if isinstance(scanner_state, dict):
                if scanner_state.get("session_id") != session_id:
                    raise RuntimeError("scanner state session mismatch")
                if (
                    scanner_state.get("live_fingerprint")
                    != profile["_live_fingerprint"]
                ):
                    raise RuntimeError("scanner state live fingerprint mismatch")
                restore_scanner(scanner, scanner_state)
            elif previous.get("state") not in {"PREPARED", "STARTING"}:
                raise RuntimeError("resume requires scanner state")
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
        if event_meta.get("active_today"):
            oms.day_halted = True
            oms._event(
                "EVENT_BLACKOUT",
                {
                    "trade_date": trade_date,
                    "sources": event_meta.get("sources"),
                    "blackout_dates": event_meta.get("blackout_dates"),
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
    parser.add_argument("--scheme", default="m5")
    parser.add_argument("--lock-time", default="auto")
    parser.add_argument("--end-time", default="auto")
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=212)
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
