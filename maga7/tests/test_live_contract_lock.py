from dataclasses import dataclass

from maga7.common.contract_select import trading_dte
from maga7.live.live_contract_lock import select_ladder_contracts


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


def test_trading_dte_skips_nyse_good_friday():
    assert trading_dte("2026-04-06", "2026-04-02") == 1
