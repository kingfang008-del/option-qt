"""Live-feasible Mag7 open-ladder contract lock and persistence."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from maga7.common.contract_select import trading_dte
from maga7.common.open_lock import ladder_bucket_id

logger = logging.getLogger("maga7.live.contract_lock")


@dataclass(frozen=True)
class LockedContract:
    symbol: str
    date: str
    expiry: str
    front_dte: int
    right: str
    strike: float
    ladder_rung: int
    bucket_id: int
    local_symbol: str
    con_id: int
    exchange: str = "SMART"
    currency: str = "USD"
    lock_spot: float = 0.0
    lock_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expiry_date(raw: str) -> date | None:
    text = str(raw or "")[:8]
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def local_symbol_expiry(raw: str) -> date | None:
    """Parse the OCC YYMMDD expiry encoded at the end of a local symbol."""
    compact = "".join(str(raw or "").upper().split())
    match = re.search(r"(\d{6})[CP]\d{8}$", compact)
    if match is None:
        return None
    text = match.group(1)
    try:
        return date(2000 + int(text[:2]), int(text[2:4]), int(text[4:6]))
    except ValueError:
        return None


def locked_contract_identity_ok(lock: LockedContract) -> bool:
    """Require manifest expiry and OCC local-symbol expiry to identify one contract."""
    expiry = _expiry_date(lock.expiry)
    local_expiry = local_symbol_expiry(lock.local_symbol)
    return expiry is not None and local_expiry is not None and expiry == local_expiry


def _contract_expiry(contract: Any) -> date | None:
    canonical = _expiry_date(getattr(contract, "_maga7_expiry", ""))
    if canonical is not None:
        return canonical
    raw = _expiry_date(getattr(contract, "lastTradeDateOrContractMonth", ""))
    local = local_symbol_expiry(getattr(contract, "localSymbol", ""))
    # Without ContractDetails.realExpirationDate, accept only an exact identity.
    return raw if raw is not None and raw == local else None


def _prefer_trading_class(symbol: str, contracts: Iterable[Any]) -> str:
    """Prefer the standard equity class matching ``symbol`` over adjusted stubs."""
    values = list(contracts)
    if not values:
        return ""
    exact = str(symbol or "").upper()
    classes = [
        str(getattr(contract, "tradingClass", "") or "").upper()
        for contract in values
    ]
    if exact and exact in classes:
        return exact
    counted = Counter(classes)
    return counted.most_common(1)[0][0] if counted else ""


def select_ladder_contracts(
    contracts: Iterable[Any],
    *,
    symbol: str,
    trade_date: str,
    spot: float,
    allowed_dte: Iterable[int] = (0, 1, 2),
    otm_rungs: int = 3,
    lock_ts: float | None = None,
    fallback_nearest: bool = True,
) -> list[LockedContract]:
    """Select ATM + strict OTM1..N on both sides for each exact trading DTE.

    When none of ``allowed_dte`` are present (META/MSFT midweek case), optionally
    fall back to the nearest available trading DTE so the symbol stays tradeable.
    """
    if spot <= 0:
        return []
    wanted = {int(value) for value in allowed_dte}
    all_rows: list[tuple[Any, int, str, float]] = []
    for contract in contracts:
        expiry = _contract_expiry(contract)
        local = str(getattr(contract, "localSymbol", "") or "").strip()
        local_expiry = local_symbol_expiry(local)
        right = str(getattr(contract, "right", "") or "").upper()
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        if (
            expiry is None
            or local_expiry is None
            or local_expiry != expiry
            or right not in {"C", "P"}
            or strike <= 0
        ):
            if expiry is not None and local_expiry is not None and local_expiry != expiry:
                logger.error(
                    "reject contract identity mismatch symbol=%s expiry=%s local=%s",
                    symbol,
                    expiry.isoformat(),
                    local,
                )
            continue
        dte = int(trading_dte(expiry, trade_date))
        if dte < 0:
            continue
        all_rows.append((contract, dte, right, strike))

    rows = [row for row in all_rows if row[1] in wanted]
    if not rows and fallback_nearest and all_rows:
        nearest = min(row[1] for row in all_rows)
        rows = [row for row in all_rows if row[1] == nearest]
        wanted = {nearest}
        logger.warning(
            "lock fallback nearest dte symbol=%s allowed=%s nearest=%s",
            symbol,
            sorted(int(value) for value in allowed_dte),
            nearest,
        )

    selected: list[LockedContract] = []
    ts = float(lock_ts or time.time())
    for dte in sorted(wanted):
        for right in ("P", "C"):
            side = "p" if right == "P" else "c"
            candidates = [row for row in rows if row[1] == dte and row[2] == right]
            if not candidates:
                continue
            # Existing offline semantics: ATM is nearest strike by absolute distance.
            atm = min(candidates, key=lambda row: abs(row[3] - spot))
            picks = [atm]
            if right == "C":
                otm = sorted(
                    (row for row in candidates if row[3] > spot and row[0] is not atm),
                    key=lambda row: row[3],
                )
            else:
                otm = sorted(
                    (row for row in candidates if row[3] < spot and row[0] is not atm),
                    key=lambda row: -row[3],
                )
            seen = {float(atm[3])}
            for row in otm:
                if float(row[3]) in seen:
                    continue
                seen.add(float(row[3]))
                picks.append(row)
                if len(picks) >= int(otm_rungs) + 1:
                    break

            for rung, (contract, _, _, strike) in enumerate(picks):
                contract_expiry = _contract_expiry(contract)
                if contract_expiry is None:
                    continue
                selected.append(
                    LockedContract(
                        symbol=symbol,
                        date=str(trade_date),
                        expiry=contract_expiry.strftime("%Y%m%d"),
                        front_dte=dte,
                        right=right,
                        strike=float(strike),
                        ladder_rung=rung,
                        bucket_id=ladder_bucket_id(side, rung),
                        local_symbol=str(getattr(contract, "localSymbol", "") or "").strip(),
                        con_id=int(getattr(contract, "conId", 0) or 0),
                        exchange=str(getattr(contract, "exchange", "") or "SMART"),
                        currency=str(getattr(contract, "currency", "") or "USD"),
                        lock_spot=float(spot),
                        lock_ts=ts,
                    )
                )
    return selected


def atomic_write_lock_manifest(
    path: Path,
    *,
    session_id: str,
    trade_date: str,
    locks: dict[str, list[LockedContract]],
    status: str,
    errors: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "trade_date": trade_date,
        "status": status,
        "updated_at": time.time(),
        "locks": {
            symbol: [contract.to_dict() for contract in contracts]
            for symbol, contracts in locks.items()
        },
        "errors": errors or {},
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return payload


class LiveOpenLadderLockService:
    """Query IBKR option chains, lock the ladder, qualify selected contracts."""

    def __init__(
        self,
        ib: Any,
        *,
        allowed_dte: Iterable[int] = (0, 1, 2),
        otm_rungs: int = 3,
        request_concurrency: int = 2,
    ):
        self.ib = ib
        self.allowed_dte = tuple(int(value) for value in allowed_dte)
        self.otm_rungs = int(otm_rungs)
        self._sem = asyncio.Semaphore(max(1, int(request_concurrency)))

    async def _discover_symbol_contracts(
        self,
        stock_contract: Any,
        *,
        symbol: str,
        trade_date: str,
    ) -> list[Any]:
        """Fetch the eligible contract universe without using an entry spot."""
        from ib_insync import Option

        chains = await self.ib.reqSecDefOptParamsAsync(
            stock_contract.symbol,
            "",
            "STK",
            int(stock_contract.conId or 0),
        )
        symbol_u = str(symbol or "").upper()
        preferred_chains = [
            chain
            for chain in (chains or [])
            if str(getattr(chain, "tradingClass", "") or "").upper() == symbol_u
        ]
        # Prefer standard equity class; if that yields nothing, fall back to all.
        for attempt, scan_chains in enumerate(
            (
                preferred_chains,
                list(chains or []),
            )
            if preferred_chains
            else (list(chains or []),)
        ):
            if not scan_chains:
                continue
            contracts = await self._contracts_from_chains(
                scan_chains,
                symbol=symbol_u,
                trade_date=trade_date,
                option_cls=Option,
            )
            if contracts:
                if attempt > 0:
                    logger.warning(
                        "discover fell back to all tradingClasses symbol=%s n=%s",
                        symbol_u,
                        len(contracts),
                    )
                return contracts
        return []

    async def _contracts_from_chains(
        self,
        scan_chains: list[Any],
        *,
        symbol: str,
        trade_date: str,
        option_cls: Any,
    ) -> list[Any]:
        expirations: set[str] = set()
        for chain in scan_chains:
            expirations.update(getattr(chain, "expirations", set()) or set())
        dated: list[tuple[int, str]] = []
        for expiry in sorted(expirations):
            exp_date = _expiry_date(expiry)
            if exp_date is None:
                continue
            dte = int(trading_dte(exp_date, trade_date))
            if dte < 0:
                continue
            dated.append((dte, expiry))
        allowed = {int(value) for value in self.allowed_dte}
        chosen_expiries = [expiry for dte, expiry in dated if dte in allowed]
        if not chosen_expiries and dated:
            nearest_dte, nearest_expiry = min(dated, key=lambda row: row[0])
            chosen_expiries = [nearest_expiry]
            logger.warning(
                "discover fallback nearest expiry symbol=%s allowed=%s nearest_dte=%s expiry=%s",
                symbol,
                sorted(allowed),
                nearest_dte,
                nearest_expiry,
            )

        details = []
        for expiry in chosen_expiries:
            for right in ("P", "C"):
                query = option_cls(symbol, expiry, right=right, exchange="SMART")
                try:
                    details.extend(await self.ib.reqContractDetailsAsync(query))
                except Exception as exc:
                    logger.warning(
                        "contract details failed %s %s %s: %s",
                        symbol,
                        expiry,
                        right,
                        exc,
                    )
        grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for detail in details:
            contract = detail.contract
            real_expiry = _expiry_date(getattr(detail, "realExpirationDate", ""))
            local_expiry = local_symbol_expiry(
                getattr(contract, "localSymbol", "")
            )
            raw_expiry = _expiry_date(
                getattr(contract, "lastTradeDateOrContractMonth", "")
            )
            if (
                real_expiry is not None
                and local_expiry is not None
                and real_expiry != local_expiry
            ):
                logger.error(
                    "reject real/local expiry mismatch symbol=%s real=%s local=%s",
                    symbol,
                    real_expiry.isoformat(),
                    getattr(contract, "localSymbol", ""),
                )
                continue
            canonical_expiry = real_expiry or (
                local_expiry if local_expiry == raw_expiry else None
            )
            if canonical_expiry is None:
                logger.error(
                    "reject contract without canonical expiry symbol=%s last=%s local=%s",
                    symbol,
                    getattr(contract, "lastTradeDateOrContractMonth", ""),
                    getattr(contract, "localSymbol", ""),
                )
                continue
            setattr(contract, "_maga7_expiry", canonical_expiry.strftime("%Y%m%d"))
            multiplier = str(getattr(contract, "multiplier", "100") or "100")
            if multiplier not in {"100", "100.0"}:
                continue
            key = (
                canonical_expiry.strftime("%Y%m%d"),
                str(getattr(contract, "right", "") or "").upper(),
            )
            grouped[key].append(contract)
        contracts = []
        for values in grouped.values():
            main_class = _prefer_trading_class(symbol, values)
            preferred = [
                contract
                for contract in values
                if str(getattr(contract, "tradingClass", "") or "").upper() == main_class
            ]
            contracts.extend(preferred or list(values))
        return contracts

    async def prepare_symbol(
        self,
        stock_contract: Any,
        *,
        symbol: str,
        trade_date: str,
    ) -> list[Any]:
        """Pre-open metadata phase; final strike selection remains an RTH decision."""
        async with self._sem:
            return await self._discover_symbol_contracts(
                stock_contract,
                symbol=symbol,
                trade_date=trade_date,
            )

    async def lock_symbol(
        self,
        stock_contract: Any,
        *,
        symbol: str,
        trade_date: str,
        spot: float,
        prepared_contracts: Iterable[Any] | None = None,
    ) -> tuple[list[LockedContract], dict[int, Any]]:
        async with self._sem:
            contracts = (
                list(prepared_contracts)
                if prepared_contracts is not None
                else await self._discover_symbol_contracts(
                    stock_contract,
                    symbol=symbol,
                    trade_date=trade_date,
                )
            )
            locks = select_ladder_contracts(
                contracts,
                symbol=symbol,
                trade_date=trade_date,
                spot=spot,
                allowed_dte=self.allowed_dte,
                otm_rungs=self.otm_rungs,
            )

            by_local = {
                str(getattr(contract, "localSymbol", "") or "").strip(): contract
                for contract in contracts
            }
            selected_contracts = [
                by_local[lock.local_symbol]
                for lock in locks
                if lock.local_symbol in by_local
            ]
            # reqContractDetails already returns qualified contracts. Only fall
            # back to a second round trip when IB omitted conId.
            unqualified = [
                contract
                for contract in selected_contracts
                if int(getattr(contract, "conId", 0) or 0) <= 0
            ]
            if unqualified:
                await self.ib.qualifyContractsAsync(*unqualified)
            by_con_id = {
                int(getattr(contract, "conId", 0) or 0): contract
                for contract in selected_contracts
                if int(getattr(contract, "conId", 0) or 0) > 0
            }
            return locks, by_con_id
