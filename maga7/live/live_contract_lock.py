"""Live-feasible Mag7 open-ladder contract lock and persistence."""
from __future__ import annotations

import asyncio
import json
import logging
import os
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


def select_ladder_contracts(
    contracts: Iterable[Any],
    *,
    symbol: str,
    trade_date: str,
    spot: float,
    allowed_dte: Iterable[int] = (0, 1, 2),
    otm_rungs: int = 5,
    lock_ts: float | None = None,
) -> list[LockedContract]:
    """Select ATM + strict OTM1..N on both sides for each exact trading DTE."""
    if spot <= 0:
        return []
    wanted = {int(value) for value in allowed_dte}
    rows: list[tuple[Any, int, str, float]] = []
    for contract in contracts:
        expiry = _expiry_date(getattr(contract, "lastTradeDateOrContractMonth", ""))
        right = str(getattr(contract, "right", "") or "").upper()
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        if expiry is None or right not in {"C", "P"} or strike <= 0:
            continue
        dte = trading_dte(expiry, trade_date)
        if dte in wanted:
            rows.append((contract, int(dte), right, strike))

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
                selected.append(
                    LockedContract(
                        symbol=symbol,
                        date=str(trade_date),
                        expiry=str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8],
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
        otm_rungs: int = 5,
        request_concurrency: int = 2,
    ):
        self.ib = ib
        self.allowed_dte = tuple(int(value) for value in allowed_dte)
        self.otm_rungs = int(otm_rungs)
        self._sem = asyncio.Semaphore(max(1, int(request_concurrency)))

    async def lock_symbol(
        self,
        stock_contract: Any,
        *,
        symbol: str,
        trade_date: str,
        spot: float,
    ) -> tuple[list[LockedContract], dict[int, Any]]:
        from ib_insync import Option

        async with self._sem:
            chains = await self.ib.reqSecDefOptParamsAsync(
                stock_contract.symbol,
                "",
                "STK",
                int(stock_contract.conId or 0),
            )
            expirations = set()
            for chain in chains or []:
                expirations.update(getattr(chain, "expirations", set()) or set())
            chosen_expiries = []
            for expiry in sorted(expirations):
                exp_date = _expiry_date(expiry)
                if exp_date is None:
                    continue
                if trading_dte(exp_date, trade_date) in self.allowed_dte:
                    chosen_expiries.append(expiry)

            details = []
            for expiry in chosen_expiries:
                for right in ("P", "C"):
                    query = Option(symbol, expiry, right=right, exchange="SMART")
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
                multiplier = str(getattr(contract, "multiplier", "100") or "100")
                if multiplier not in {"100", "100.0"}:
                    continue
                key = (
                    str(getattr(contract, "lastTradeDateOrContractMonth", ""))[:8],
                    str(getattr(contract, "right", "") or "").upper(),
                )
                grouped[key].append(contract)
            contracts = []
            for values in grouped.values():
                classes = Counter(
                    str(getattr(contract, "tradingClass", "") or "")
                    for contract in values
                )
                main_class = classes.most_common(1)[0][0] if classes else ""
                contracts.extend(
                    contract
                    for contract in values
                    if str(getattr(contract, "tradingClass", "") or "") == main_class
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
            if selected_contracts:
                await self.ib.qualifyContractsAsync(*selected_contracts)
            by_con_id = {
                int(getattr(contract, "conId", 0) or 0): contract
                for contract in selected_contracts
                if int(getattr(contract, "conId", 0) or 0) > 0
            }
            return locks, by_con_id
