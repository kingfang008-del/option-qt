"""Lane-level prefer_dte / allowed_dte overrides on resolve_entry_contract."""
from __future__ import annotations

from maga7.common.entry_contract import ContractBooks, resolve_entry_contract


def test_allowed_dte_0_only_skips_1dte_fallback() -> None:
    day = "2026-07-24"
    # Only 1DTE in the book for DN ATM → 0DTE-only must fail closed.
    books = ContractBooks(
        mode="open_ladder",
        multi_idx={
            ("TSLA", day): {
                1: {0: "TSLA260725P00320000"},  # dte=1 put ATM bucket
            }
        },
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
        ladder=True,
        otm_rungs=3,
    )
    # With default allowed [0,1,2] may pick 1DTE; with [0] must return none.
    pick0 = resolve_entry_contract(
        books,
        symbol="TSLA",
        date=day,
        direction="DN",
        moneyness="ATM",
        sig_ts=f"{day} 10:45:00",
        spot=320.0,
        prefer_dte=0,
        allowed_dte=[0],
    )
    assert pick0.ticker is None

    pick1 = resolve_entry_contract(
        books,
        symbol="TSLA",
        date=day,
        direction="DN",
        moneyness="ATM",
        sig_ts=f"{day} 10:45:00",
        spot=320.0,
        prefer_dte=0,
        allowed_dte=[0, 1],
    )
    assert pick1.ticker == "TSLA260725P00320000"
    assert pick1.dte == 1


def test_0dte_preferred_when_both_present() -> None:
    day = "2026-07-24"
    books = ContractBooks(
        mode="open_ladder",
        multi_idx={
            ("TSLA", day): {
                0: {0: "TSLA260724P00320000"},
                1: {0: "TSLA260725P00320000"},
            }
        },
        prefer_dte=0,
        allowed_dte=[0, 1, 2],
        ladder=True,
        otm_rungs=3,
    )
    pick = resolve_entry_contract(
        books,
        symbol="TSLA",
        date=day,
        direction="DN",
        moneyness="ATM",
        sig_ts=f"{day} 10:45:00",
        spot=320.0,
        prefer_dte=0,
        allowed_dte=[0],
    )
    assert pick.ticker == "TSLA260724P00320000"
    assert pick.dte == 0
