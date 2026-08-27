#!/usr/bin/env python3
"""Pre-open Mag7 lock/subscription readiness probe.

Healthy handling when front DTE is missing:
  auto-fall back to the nearest standard-class expiry (not FAIL).

This probe still FAIL-closes on structural problems:
  empty universe, empty lock, adjusted stub class (2MSFT).

If NBBO is missing after a healthy nearest fallback, it auto-diagnoses the
upstream cause instead of treating "no bid/ask" as the root issue.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock

from maga7.common.config import load_profile
from maga7.live.live_contract_lock import LiveOpenLadderLockService
from maga7.live.option_quote_diagnose import (
    diagnose_missing_option_quotes,
    is_adjusted_local_symbol,
)

NY = ZoneInfo("America/New_York")


@dataclass
class SymbolProbe:
    symbol: str
    ok: bool
    spot: float = 0.0
    n_locks: int = 0
    front_dte: int | None = None
    expiry: str | None = None
    sample_locals: list[str] | None = None
    nearest_fallback: bool = False
    adjusted_class: bool = False
    quoted: bool = False
    quote_detail: str = ""
    diagnosis_code: str = ""
    diagnosis_detail: str = ""
    actionable: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _ny_now() -> datetime:
    return datetime.now(tz=NY)


async def _wait_stock_spot(ib: IB, stock: Any, *, timeout: float) -> float:
    ticker = ib.reqMktData(stock, "", False, False)
    deadline = asyncio.get_running_loop().time() + float(timeout)
    try:
        while asyncio.get_running_loop().time() < deadline:
            for attr in ("marketPrice", "last", "close"):
                raw = getattr(ticker, attr, None)
                if callable(raw):
                    raw = raw()
                if _positive(raw):
                    return float(raw)
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            if _positive(bid) and _positive(ask) and float(ask) >= float(bid):
                return (float(bid) + float(ask)) / 2.0
            await asyncio.sleep(0.2)
        raise TimeoutError("stock spot timeout")
    finally:
        try:
            ib.cancelMktData(stock)
        except Exception:
            pass


async def _probe_option_ticker(
    ib: IB, contract: Any, *, timeout: float
) -> dict[str, Any]:
    ticker = ib.reqMktData(contract, "100,101,106", False, False)
    deadline = asyncio.get_running_loop().time() + float(timeout)
    snap = {
        "bid": None,
        "ask": None,
        "close": None,
        "has_model": False,
        "quoted": False,
    }
    try:
        while asyncio.get_running_loop().time() < deadline:
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            snap = {
                "bid": bid,
                "ask": ask,
                "close": getattr(ticker, "close", None),
                "has_model": bool(getattr(ticker, "modelGreeks", None)),
                "quoted": False,
            }
            if _positive(bid) and _positive(ask) and float(ask) >= float(bid):
                snap["quoted"] = True
                return snap
            await asyncio.sleep(0.2)
        return snap
    finally:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass


async def probe_symbol(
    ib: IB,
    *,
    symbol: str,
    trade_date: str,
    allowed_dte: tuple[int, ...],
    otm_rungs: int,
    spot_timeout: float,
    quote_timeout: float,
    require_quotes: bool,
) -> SymbolProbe:
    symbol = str(symbol).upper()
    try:
        stock = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(stock)
        if not qualified:
            return SymbolProbe(symbol=symbol, ok=False, error="stock_qualify_failed")
        spot = await _wait_stock_spot(ib, stock, timeout=spot_timeout)
        service = LiveOpenLadderLockService(
            ib,
            allowed_dte=allowed_dte,
            otm_rungs=otm_rungs,
            request_concurrency=1,
        )
        prepared = await service.prepare_symbol(
            stock,
            symbol=symbol,
            trade_date=trade_date,
        )
        if not prepared:
            return SymbolProbe(
                symbol=symbol,
                ok=False,
                spot=spot,
                error="empty_contract_universe",
                diagnosis_code="empty_contract_universe",
                actionable="check IB chain / symbol class availability",
            )
        locks, contracts = await service.lock_symbol(
            stock,
            symbol=symbol,
            trade_date=trade_date,
            spot=spot,
            prepared_contracts=prepared,
        )
        if not locks:
            return SymbolProbe(
                symbol=symbol,
                ok=False,
                spot=spot,
                error="empty_lock",
                diagnosis_code="empty_lock",
                actionable="check nearest-expiry fallback and strike ladder",
            )
        locals_ = [lock.local_symbol for lock in locks]
        adjusted = any(is_adjusted_local_symbol(symbol, local) for local in locals_)
        dtes = sorted({int(lock.front_dte) for lock in locks})
        allowed = {int(value) for value in allowed_dte}
        nearest_fallback = bool(dtes) and not set(dtes).issubset(allowed)
        probe = SymbolProbe(
            symbol=symbol,
            ok=not adjusted,
            spot=spot,
            n_locks=len(locks),
            front_dte=dtes[0] if dtes else None,
            expiry=locks[0].expiry,
            sample_locals=locals_[:4],
            nearest_fallback=nearest_fallback,
            adjusted_class=adjusted,
            error="adjusted_trading_class" if adjusted else "",
        )
        if adjusted:
            diagnosis = diagnose_missing_option_quotes(
                symbol=symbol,
                locks=locks,
                allowed_dte=allowed_dte,
            )
            probe.diagnosis_code = diagnosis.code
            probe.diagnosis_detail = diagnosis.detail
            probe.actionable = diagnosis.actionable
            return probe

        atm = [
            lock
            for lock in locks
            if int(lock.ladder_rung) == 0 and int(lock.con_id) in contracts
        ]
        quoted = 0
        details: list[str] = []
        option_quotes: dict[tuple[str, str], dict[str, float]] = {}
        ticker_snapshots: dict[int, dict[str, Any]] = {}
        subscribed: list[int] = []
        for lock in atm[:2]:
            contract = contracts.get(int(lock.con_id))
            if contract is None:
                continue
            subscribed.append(int(lock.con_id))
            snap = await _probe_option_ticker(ib, contract, timeout=quote_timeout)
            ticker_snapshots[int(lock.con_id)] = snap
            if snap.get("quoted"):
                quoted += 1
                bid = float(snap["bid"])
                ask = float(snap["ask"])
                option_quotes[(symbol, lock.local_symbol)] = {
                    "bid": bid,
                    "ask": ask,
                    "ts": 0.0,
                }
                details.append(f"{lock.local_symbol}:{bid:.2f}/{ask:.2f}")
            else:
                details.append(
                    f"{lock.local_symbol}:NO_NBBO(bid={snap.get('bid')},ask={snap.get('ask')},"
                    f"model={snap.get('has_model')})"
                )
        probe.quoted = quoted > 0
        probe.quote_detail = "; ".join(details)
        diagnosis = diagnose_missing_option_quotes(
            symbol=symbol,
            locks=locks,
            option_quotes=option_quotes,
            subscribed_con_ids=subscribed,
            allowed_dte=allowed_dte,
            ticker_snapshots=ticker_snapshots,
        )
        probe.diagnosis_code = diagnosis.code
        probe.diagnosis_detail = diagnosis.detail
        probe.actionable = diagnosis.actionable
        # Nearest fallback is healthy. Missing NBBO is diagnosed, not a DTE failure.
        if require_quotes and not probe.quoted:
            probe.ok = False
            probe.error = diagnosis.code or "no_option_quotes"
        elif diagnosis.exclude:
            probe.ok = False
            probe.error = diagnosis.code
        return probe
    except Exception as exc:
        return SymbolProbe(symbol=symbol, ok=False, error=str(exc))


async def run(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    lock_cfg = profile.get("lock") or {}
    live_cfg = (profile.get("live") or {}) if isinstance(profile.get("live"), dict) else {}
    allowed = tuple(
        int(value)
        for value in (
            lock_cfg.get("allowed_dte")
            or live_cfg.get("allowed_dte")
            or [0, 1, 2]
        )
    )
    otm_rungs = int(lock_cfg.get("otm_rungs") or live_cfg.get("otm_rungs") or 3)
    symbols = [str(sym).upper() for sym in (profile.get("symbols") or [])]
    if args.symbols:
        symbols = [str(sym).upper() for sym in args.symbols]
    trade_date = args.trade_date or _ny_now().strftime("%Y-%m-%d")
    require_quotes = bool(args.require_quotes)

    ib = IB()
    probes: list[SymbolProbe] = []
    try:
        await ib.connectAsync(
            args.host,
            args.port,
            clientId=args.client_id,
            readonly=True,
            timeout=args.timeout,
        )
        ib.reqMarketDataType(int(args.market_data_type))
        for symbol in symbols:
            probes.append(
                await probe_symbol(
                    ib,
                    symbol=symbol,
                    trade_date=trade_date,
                    allowed_dte=allowed,
                    otm_rungs=otm_rungs,
                    spot_timeout=args.timeout,
                    quote_timeout=args.quote_timeout,
                    require_quotes=require_quotes,
                )
            )
    finally:
        if ib.isConnected():
            ib.disconnect()

    blockers = [row for row in probes if not row.ok]
    warnings = [
        row
        for row in probes
        if row.ok
        and (
            row.nearest_fallback
            or (not row.quoted and row.diagnosis_code not in {"ok_quoted", "nearest_fallback_quoted"})
        )
    ]
    payload = {
        "ok": not blockers,
        "trade_date": trade_date,
        "require_quotes": require_quotes,
        "allowed_dte": list(allowed),
        "blockers": len(blockers),
        "warnings": len(warnings),
        "symbols": [row.to_dict() for row in probes],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    for row in probes:
        flag = "PASS" if row.ok else "FAIL"
        print(
            f"{flag} {row.symbol} spot={row.spot:.2f} locks={row.n_locks} "
            f"dte={row.front_dte} expiry={row.expiry} diagnosis={row.diagnosis_code or row.error or 'ok'} "
            f"{row.quote_detail}",
            flush=True,
        )
        if row.actionable and row.actionable != "none":
            print(f"  actionable: {row.actionable}", flush=True)
    if blockers:
        print(
            f"MAG7_LOCK_PREFLIGHT_FAIL blockers={len(blockers)} "
            f"symbols={[row.symbol for row in blockers]}",
            flush=True,
        )
        return 2
    print(
        f"MAG7_LOCK_PREFLIGHT_OK symbols={len(probes)} warnings={len(warnings)} "
        f"require_quotes={require_quotes}",
        flush=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--client-id", type=int, default=913)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--quote-timeout", type=float, default=6.0)
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument(
        "--require-quotes",
        action="store_true",
        help="Optional hard fail when ATM NBBO is missing (off by default).",
    )
    parser.add_argument("--json-out", default="")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
