"""Shared entry-contract resolution for offline / stream / live scanner.

Keeps contract_mode + clear_otm + lock map loading in one place so stream
parity matches run_offline_replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.contract_select import (
    DayIvChainCache,
    lock_policy_from_profile,
    resolve_contract,
    trading_dte,
)
from maga7.common.open_lock import (
    is_clearly_otm,
    load_multidte_lock_index,
    resolve_open_lock_contract,
    strike_from_occ,
)
from maga7.common.replay import BUCKET_MAP, load_lock_index


@dataclass
class EntryContract:
    ticker: str | None
    dte: int | None
    source: str
    bucket_id: int
    strike: float | None = None


@dataclass
class ContractBooks:
    """Preloaded lock indexes for a profile."""

    mode: str
    flat_idx: dict[tuple[str, str], dict[int, str]] | None = None
    multi_idx: dict[tuple[str, str], dict[int, dict[int, str]]] | None = None
    chain_cache: DayIvChainCache | None = None
    prefer_dte: int = 0
    allowed_dte: list[int] | None = None
    clear_otm_thresh: float | None = None
    ladder: bool = False
    otm_rungs: int = 1

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "ContractBooks":
        trade = profile.get("trade") or {}
        mode = str(trade.get("contract_mode", "day_lock")).lower()
        prefer, allowed = lock_policy_from_profile(profile)
        clear = trade.get("clear_otm_ban_0dte_pct")
        clear_otm = float(clear) if clear is not None else None
        paths = profile.get("_paths") or {}
        ladder = mode in ("open_ladder", "open_lock_ladder")
        from maga7.common.open_lock import resolve_otm_rungs

        otm_rungs = resolve_otm_rungs(profile, default=2 if ladder else 1)

        flat = None
        multi = None
        chain = None
        if mode in ("open_lock", "open", "open_ladder", "open_lock_ladder"):
            lock_path = paths.get("open_locked_map") or paths.get("locked_map")
            multi = load_multidte_lock_index(Path(lock_path))
        else:
            lock_path = paths.get("locked_map")
            flat = load_lock_index(Path(lock_path))
        if mode == "signal_atm" or str(trade.get("quote_source", "1s")).lower() in ("day_iv", "auto"):
            day_iv = paths.get("day_iv_root")
            if day_iv:
                chain = DayIvChainCache(Path(day_iv))
        elif mode in ("open_lock", "open", "open_ladder", "open_lock_ladder"):
            day_iv = paths.get("day_iv_root")
            if day_iv and str(trade.get("quote_source", "1s")).lower() in ("day_iv", "auto"):
                chain = DayIvChainCache(Path(day_iv))

        return cls(
            mode=mode,
            flat_idx=flat,
            multi_idx=multi,
            chain_cache=chain,
            prefer_dte=prefer,
            allowed_dte=list(allowed),
            clear_otm_thresh=clear_otm,
            ladder=ladder or bool(trade.get("open_ladder")),
            otm_rungs=otm_rungs,
        )


def resolve_entry_contract(
    books: ContractBooks,
    *,
    symbol: str,
    date: str,
    direction: str,
    moneyness: str,
    sig_ts,
    spot: float | None,
) -> EntryContract:
    bid = BUCKET_MAP[(direction, moneyness)]
    mode = books.mode

    if mode in ("open_lock", "open", "open_ladder", "open_lock_ladder"):
        by_dte = books.multi_idx.get((symbol, date)) if books.multi_idx else None
        ticker, dte, source = resolve_open_lock_contract(
            by_dte,
            direction=direction,
            moneyness=moneyness,
            spot=spot,
            prefer_dte=books.prefer_dte,
            allowed_dte=books.allowed_dte or [0, 1, 2],
            clear_otm_thresh=books.clear_otm_thresh,
            ladder=books.ladder or mode in ("open_ladder", "open_lock_ladder"),
            otm_rungs=books.otm_rungs,
        )
        strike = strike_from_occ(ticker) if ticker else None
        return EntryContract(ticker=ticker, dte=dte, source=source, bucket_id=bid, strike=strike)

    day_ticker = None
    if books.flat_idx:
        day_ticker = (books.flat_idx.get((symbol, date)) or {}).get(bid)

    if mode == "signal_atm":
        chain = books.chain_cache.get(symbol, date) if books.chain_cache is not None else None
        pick = resolve_contract(
            mode="signal_atm",
            chain=chain,
            date=str(date),
            direction=direction,
            sig_ts=sig_ts,
            spot=spot,
            day_lock_ticker=day_ticker,
            prefer_dte=books.prefer_dte,
            allowed_dte=books.allowed_dte or [0, 1, 2],
            fallback_day_lock=True,
        )
        if pick is None:
            return EntryContract(None, None, "none", bid)
        ticker = pick.ticker
        dte = pick.dte if pick.dte >= 0 else None
        source = pick.source
        strike = pick.strike if np.isfinite(pick.strike) else strike_from_occ(ticker)
        if (
            books.clear_otm_thresh is not None
            and dte == 0
            and spot is not None
            and strike is not None
            and is_clearly_otm(direction, float(spot), float(strike), thresh=books.clear_otm_thresh)
        ):
            pick2 = resolve_contract(
                mode="signal_atm",
                chain=chain,
                date=str(date),
                direction=direction,
                sig_ts=sig_ts,
                spot=spot,
                day_lock_ticker=day_ticker,
                prefer_dte=1,
                allowed_dte=[d for d in (books.allowed_dte or [1, 2]) if int(d) >= 1] or [1, 2],
                fallback_day_lock=False,
            )
            if pick2 is not None:
                return EntryContract(
                    pick2.ticker,
                    pick2.dte if pick2.dte >= 0 else None,
                    "signal_atm_skip0_clear_otm",
                    bid,
                    pick2.strike if np.isfinite(pick2.strike) else strike_from_occ(pick2.ticker),
                )
        return EntryContract(ticker, dte, source, bid, strike)

    # day_lock
    if not day_ticker:
        return EntryContract(None, None, "no_lock", bid)
    strike = strike_from_occ(day_ticker)
    dte = None
    import re

    m = re.search(r"(\d{6})[CP]", str(day_ticker).replace("O:", ""))
    if m:
        exp = pd.Timestamp("20" + m.group(1)).date()
        dte = trading_dte(exp, date)
    if (
        books.clear_otm_thresh is not None
        and dte == 0
        and spot is not None
        and is_clearly_otm(direction, float(spot), strike, thresh=books.clear_otm_thresh)
    ):
        return EntryContract(None, dte, "day_lock_skip0_clear_otm", bid, strike)
    return EntryContract(day_ticker, dte, "day_lock", bid, strike)
