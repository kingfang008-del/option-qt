import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from maga7.common.contract_select import trading_dte
from maga7.live.live_contract_lock import (
    LockedContract,
    local_symbol_expiry,
    locked_contract_identity_ok,
    select_ladder_contracts,
)


@dataclass
class FakeContract:
    lastTradeDateOrContractMonth: str
    right: str
    strike: float
    localSymbol: str
    conId: int
    exchange: str = "SMART"
    currency: str = "USD"


def test_select_ladder_contracts_matches_open_ladder_semantics():
    contracts = []
    con_id = 1
    for expiry in ("20260716", "20260717", "20260720"):
        for right in ("P", "C"):
            for strike in range(96, 106):
                contracts.append(
                    FakeContract(
                        expiry,
                        right,
                        float(strike),
                        f"TEST {expiry[2:]}{right}{strike:08d}",
                        con_id,
                    )
                )
                con_id += 1

    locks = select_ladder_contracts(
        contracts,
        symbol="TEST",
        trade_date="2026-07-16",
        spot=100.4,
        allowed_dte=(0, 1, 2),
        otm_rungs=2,
        lock_ts=1.0,
    )

    assert len(locks) == 18
    day0 = [lock for lock in locks if lock.front_dte == 0]
    call = sorted(
        (lock for lock in day0 if lock.right == "C"),
        key=lambda lock: lock.ladder_rung,
    )
    put = sorted(
        (lock for lock in day0 if lock.right == "P"),
        key=lambda lock: lock.ladder_rung,
    )
    assert [(lock.strike, lock.bucket_id) for lock in call] == [
        (100.0, 2),
        (101.0, 3),
        (102.0, 5),
    ]
    assert [(lock.strike, lock.bucket_id) for lock in put] == [
        (100.0, 0),
        (99.0, 1),
        (98.0, 4),
    ]


def test_quote_diagnosis_adjusted_stub_vs_nearest_fallback():
    from maga7.live.option_quote_diagnose import (
        diagnose_missing_option_quotes,
        is_adjusted_local_symbol,
    )

    assert is_adjusted_local_symbol("MSFT", "2MSFT 260728C00400000")
    assert not is_adjusted_local_symbol("MSFT", "MSFT  260731C00395000")

    stub = diagnose_missing_option_quotes(
        symbol="MSFT",
        locks=[
            LockedContract(
                symbol="MSFT",
                date="2026-07-28",
                expiry="20260728",
                front_dte=0,
                right="C",
                strike=400.0,
                ladder_rung=0,
                bucket_id=2,
                local_symbol="2MSFT 260728C00400000",
                con_id=1,
            )
        ],
        allowed_dte=(0, 1, 2),
        subscribed_con_ids=(1,),
        ticker_snapshots={1: {"bid": -1, "ask": -1, "has_model": True}},
    )
    assert stub.code == "adjusted_stub_class"
    assert stub.exclude

    nearest = diagnose_missing_option_quotes(
        symbol="MSFT",
        locks=[
            LockedContract(
                symbol="MSFT",
                date="2026-07-28",
                expiry="20260731",
                front_dte=3,
                right="C",
                strike=395.0,
                ladder_rung=0,
                bucket_id=2,
                local_symbol="MSFT  260731C00395000",
                con_id=2,
            )
        ],
        option_quotes={
            ("MSFT", "MSFT  260731C00395000"): {"bid": 1.0, "ask": 1.1, "ts": 1.0}
        },
        allowed_dte=(0, 1, 2),
        subscribed_con_ids=(2,),
    )
    assert nearest.code == "nearest_fallback_quoted"
    assert not nearest.exclude

    alive = diagnose_missing_option_quotes(
        symbol="MSFT",
        locks=[
            LockedContract(
                symbol="MSFT",
                date="2026-07-28",
                expiry="20260731",
                front_dte=3,
                right="C",
                strike=395.0,
                ladder_rung=0,
                bucket_id=2,
                local_symbol="MSFT  260731C00395000",
                con_id=2,
            )
        ],
        allowed_dte=(0, 1, 2),
        subscribed_con_ids=(2,),
        ticker_snapshots={2: {"bid": -1.0, "ask": -1.0, "has_model": True}},
    )
    assert alive.code == "ticker_alive_no_nbbo"
    assert not alive.exclude


def test_diagnose_locked_option_quotes_excludes_only_stub_class():
    import asyncio
    from types import SimpleNamespace

    from maga7.live.ibkr_connector import Mag7IbkrConnector

    connector = Mag7IbkrConnector.__new__(Mag7IbkrConnector)
    connector.locks = {
        "MSFT": [
            LockedContract(
                symbol="MSFT",
                date="2026-07-28",
                expiry="20260728",
                front_dte=0,
                right="C",
                strike=400.0,
                ladder_rung=0,
                bucket_id=2,
                local_symbol="2MSFT 260728C00400000",
                con_id=1,
            )
        ],
        "AMD": [
            LockedContract(
                symbol="AMD",
                date="2026-07-28",
                expiry="20260729",
                front_dte=1,
                right="C",
                strike=450.0,
                ladder_rung=0,
                bucket_id=2,
                local_symbol="AMD   260729C00450000",
                con_id=2,
            )
        ],
    }
    connector.errors = {}
    connector.option_quotes = {}
    connector.subscribed_option_ids = {1, 2}
    connector.option_contracts = {
        1: SimpleNamespace(conId=1),
        2: SimpleNamespace(conId=2),
    }
    connector.lock_service = SimpleNamespace(allowed_dte=(0, 1, 2))

    class _Ticker:
        def __init__(self, *, model=False):
            self.bid = -1.0
            self.ask = -1.0
            self.close = 0.0
            self.modelGreeks = object() if model else None

    class _Ib:
        def ticker(self, contract):
            if int(contract.conId) == 1:
                return _Ticker(model=True)
            return _Ticker(model=False)

    connector.ib = _Ib()

    async def run():
        return await Mag7IbkrConnector._diagnose_locked_option_quotes(
            connector, timeout_sec=0.4
        )

    diagnostics = asyncio.run(run())
    assert connector.locks["MSFT"] == []
    assert connector.errors["MSFT"] == "adjusted_stub_class"
    assert connector.locks["AMD"]
    assert "AMD" not in connector.errors
    assert diagnostics["MSFT"]["code"] == "adjusted_stub_class"
    assert diagnostics["AMD"]["code"] in {
        "awaiting_nbbo",
        "ticker_alive_no_nbbo",
    }


def test_select_ladder_falls_back_to_nearest_available_dte():
    contracts = []
    con_id = 1
    for right in ("P", "C"):
        for strike in (390.0, 395.0, 400.0, 405.0):
            contracts.append(
                FakeContract(
                    "20260731",
                    right,
                    float(strike),
                    f"MSFT  260731{right}{int(strike * 1000):08d}",
                    con_id,
                )
            )
            con_id += 1
            setattr(contracts[-1], "_maga7_expiry", "20260731")

    locks = select_ladder_contracts(
        contracts,
        symbol="MSFT",
        trade_date="2026-07-28",
        spot=395.1,
        allowed_dte=(0, 1, 2),
        otm_rungs=1,
        lock_ts=1.0,
    )
    assert locks
    assert {lock.front_dte for lock in locks} == {3}
    assert {lock.expiry for lock in locks} == {"20260731"}


def test_discover_prefers_symbol_class_and_nearest_expiry():
    class FakeIb:
        async def reqSecDefOptParamsAsync(self, *_):
            return [
                SimpleNamespace(
                    tradingClass="2MSFT",
                    expirations={"20260728"},
                    multiplier="100",
                ),
                SimpleNamespace(
                    tradingClass="MSFT",
                    expirations={"20260731"},
                    multiplier="100",
                ),
            ]

        async def reqContractDetailsAsync(self, query):
            assert query.lastTradeDateOrContractMonth == "20260731"
            right = str(query.right)
            rows = []
            for strike in (390.0, 395.0, 400.0):
                local = f"MSFT  260731{right}{int(strike * 1000):08d}"
                rows.append(
                    SimpleNamespace(
                        realExpirationDate="20260731",
                        contract=SimpleNamespace(
                            lastTradeDateOrContractMonth="20260731",
                            right=right,
                            strike=strike,
                            localSymbol=local,
                            conId=int(strike) + (1 if right == "C" else 2),
                            exchange="SMART",
                            currency="USD",
                            multiplier="100",
                            tradingClass="MSFT",
                        ),
                    )
                )
            return rows

        async def qualifyContractsAsync(self, *contracts):
            return list(contracts)

    async def run():
        from maga7.live.live_contract_lock import LiveOpenLadderLockService

        ib = FakeIb()
        service = LiveOpenLadderLockService(ib, allowed_dte=(0, 1, 2), otm_rungs=1)
        stock = SimpleNamespace(symbol="MSFT", conId=272093)
        prepared = await service.prepare_symbol(
            stock,
            symbol="MSFT",
            trade_date="2026-07-28",
        )
        locks, _ = await service.lock_symbol(
            stock,
            symbol="MSFT",
            trade_date="2026-07-28",
            spot=395.1,
            prepared_contracts=prepared,
        )
        assert locks
        assert {lock.expiry for lock in locks} == {"20260731"}
        assert {lock.front_dte for lock in locks} == {3}
        assert all(lock.local_symbol.startswith("MSFT") for lock in locks)

    class FakeOption:
        def __init__(self, symbol, expiry, *, right, exchange):
            self.symbol = symbol
            self.lastTradeDateOrContractMonth = expiry
            self.right = right
            self.exchange = exchange

    fake_ib_insync = ModuleType("ib_insync")
    fake_ib_insync.Option = FakeOption
    with patch.dict(sys.modules, {"ib_insync": fake_ib_insync}):
        asyncio.run(run())


def test_contract_lock_rejects_expiry_local_symbol_mismatch():
    bad = FakeContract(
        lastTradeDateOrContractMonth="20260728",
        right="P",
        strike=200.0,
        localSymbol="NVDA  260727P00200000",
        conId=900722877,
    )
    locks = select_ladder_contracts(
        [bad],
        symbol="NVDA",
        trade_date="2026-07-27",
        spot=201.03,
        allowed_dte=(0, 1, 2),
    )
    assert locks == []


def test_restored_lock_identity_fails_closed_on_expiry_mismatch():
    lock = LockedContract(
        symbol="NVDA",
        date="2026-07-27",
        expiry="20260728",
        front_dte=1,
        right="P",
        strike=200.0,
        ladder_rung=3,
        bucket_id=6,
        local_symbol="NVDA  260727P00200000",
        con_id=900722877,
    )
    assert local_symbol_expiry(lock.local_symbol).isoformat() == "2026-07-27"
    assert not locked_contract_identity_ok(lock)


def test_preopen_prepare_reuses_metadata_at_rth_lock():
    class FakeIb:
        def __init__(self):
            self.secdef_calls = 0
            self.detail_calls = 0
            self.qualify_calls = 0

        async def reqSecDefOptParamsAsync(self, *_):
            self.secdef_calls += 1
            return [SimpleNamespace(expirations={"20260716"})]

        async def reqContractDetailsAsync(self, query):
            self.detail_calls += 1
            right = str(query.right)
            rows = []
            for strike in (98.0, 99.0, 100.0, 101.0, 102.0):
                local = f"TEST  260716{right}{int(strike * 1000):08d}"
                rows.append(
                    SimpleNamespace(
                        realExpirationDate="20260716",
                        contract=SimpleNamespace(
                            # IB may expose the next calendar day here while
                            # realExpirationDate/OCC local symbol carry expiry.
                            lastTradeDateOrContractMonth="20260717",
                            right=right,
                            strike=strike,
                            localSymbol=local,
                            conId=int(strike * 10) + (1 if right == "C" else 2),
                            exchange="SMART",
                            currency="USD",
                            multiplier="100",
                            tradingClass="TEST",
                        )
                    )
                )
            return rows

        async def qualifyContractsAsync(self, *contracts):
            self.qualify_calls += 1
            return list(contracts)

    async def run():
        from maga7.live.live_contract_lock import LiveOpenLadderLockService

        ib = FakeIb()
        service = LiveOpenLadderLockService(
            ib,
            allowed_dte=(0,),
            otm_rungs=1,
        )
        stock = SimpleNamespace(symbol="TEST", conId=123)
        prepared = await service.prepare_symbol(
            stock,
            symbol="TEST",
            trade_date="2026-07-16",
        )
        assert prepared
        calls_after_prepare = (ib.secdef_calls, ib.detail_calls)
        locks, contracts = await service.lock_symbol(
            stock,
            symbol="TEST",
            trade_date="2026-07-16",
            spot=100.4,
            prepared_contracts=prepared,
        )
        assert len(locks) == 4
        assert {lock.expiry for lock in locks} == {"20260716"}
        assert {lock.front_dte for lock in locks} == {0}
        assert contracts
        assert (ib.secdef_calls, ib.detail_calls) == calls_after_prepare
        assert ib.qualify_calls == 0

    class FakeOption:
        def __init__(self, symbol, expiry, *, right, exchange):
            self.symbol = symbol
            self.expiry = expiry
            self.right = right
            self.exchange = exchange

    fake_ib_insync = ModuleType("ib_insync")
    fake_ib_insync.Option = FakeOption
    with patch.dict(sys.modules, {"ib_insync": fake_ib_insync}):
        asyncio.run(run())
