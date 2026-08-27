#!/usr/bin/env python3
"""Read-only IBKR readiness smoke: account, live QQQ stock, and option quote."""
from __future__ import annotations

import argparse
import asyncio
import math
from datetime import datetime

from ib_insync import IB, Option, Stock


def _positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


async def _wait_quote(ticker, *, timeout: float) -> tuple[float, float]:
    deadline = asyncio.get_running_loop().time() + float(timeout)
    while asyncio.get_running_loop().time() < deadline:
        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        if _positive(bid) and _positive(ask) and float(ask) >= float(bid):
            return float(bid), float(ask)
        await asyncio.sleep(0.2)
    raise TimeoutError("bid/ask timeout")


async def run(args: argparse.Namespace) -> None:
    ib = IB()
    try:
        await ib.connectAsync(
            args.host,
            args.port,
            clientId=args.client_id,
            readonly=True,
            timeout=args.timeout,
        )
        accounts = list(ib.managedAccounts() or [])
        if not accounts:
            raise RuntimeError("no managed account returned")
        ib.reqMarketDataType(1)

        stock = Stock("QQQ", "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(stock)
        if not qualified:
            raise RuntimeError("QQQ stock qualification failed")
        stock_ticker = ib.reqMktData(stock, "", False, False)
        stock_bid, stock_ask = await _wait_quote(stock_ticker, timeout=args.timeout)
        if int(getattr(stock_ticker, "marketDataType", 0) or 0) != 1:
            raise RuntimeError(
                f"QQQ market data is not live/frozen: "
                f"type={getattr(stock_ticker, 'marketDataType', None)}"
            )
        spot = (stock_bid + stock_ask) / 2.0

        chains = await ib.reqSecDefOptParamsAsync("QQQ", "", "STK", stock.conId)
        chain = next(
            (
                item
                for item in chains
                if str(item.exchange).upper() in {"SMART", ""}
                and item.expirations
                and item.strikes
            ),
            None,
        )
        if chain is None:
            chain = next(
                (item for item in chains if item.expirations and item.strikes),
                None,
            )
        if chain is None:
            raise RuntimeError("QQQ option chain unavailable")
        today = datetime.now().strftime("%Y%m%d")
        expiry = min((value for value in chain.expirations if value >= today), default=None)
        if expiry is None:
            raise RuntimeError("QQQ option expiry unavailable")
        strike = min((float(value) for value in chain.strikes), key=lambda value: abs(value - spot))
        option = Option(
            "QQQ",
            expiry,
            strike,
            "C",
            "SMART",
            currency="USD",
            tradingClass=str(chain.tradingClass or "QQQ"),
        )
        qualified = await ib.qualifyContractsAsync(option)
        if not qualified:
            raise RuntimeError("QQQ option qualification failed")
        option_ticker = ib.reqMktData(option, "100,101,106", False, False)
        option_bid, option_ask = await _wait_quote(option_ticker, timeout=args.timeout)
        print(
            "IBKR_SMOKE_OK "
            f"accounts={len(accounts)} stock={stock_bid:.2f}/{stock_ask:.2f} "
            f"option={option.localSymbol} {option_bid:.2f}/{option_ask:.2f}",
            flush=True,
        )
    finally:
        if ib.isConnected():
            ib.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--client-id", type=int, default=912)
    parser.add_argument("--timeout", type=float, default=8.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
