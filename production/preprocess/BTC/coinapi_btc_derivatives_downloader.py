#!/usr/bin/env python3
"""
CoinAPI BTC derivatives downloader for "Scheme A" (perpetual + futures).

What it does:
1. Discover BTC perpetual / futures symbols on a chosen exchange.
2. Download OHLCV history for selected symbols.
3. Discover available metrics for BTC on that exchange.
4. Try to download funding / open interest / liquidation / mark/index metrics.
5. Fetch current quote and current order book depth.
6. Emit a coverage report that tells you which fields are available and which are missing.

Tested against CoinAPI REST API structure documented as of 2026-04.
You still need to verify exact metric IDs returned by your account / exchange universe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE_URL = "https://rest.coinapi.io"
DEFAULT_TIMEOUT = 30


class CoinAPIError(RuntimeError):
    pass


@dataclass
class CoverageRow:
    symbol_id: str
    symbol_type: str
    ohlcv: bool = False
    funding: bool = False
    open_interest: bool = False
    liquidation: bool = False
    mark_price: bool = False
    index_price: bool = False
    current_quote: bool = False
    current_book_depth: bool = False


class CoinAPIClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-CoinAPI-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "btc-derivatives-downloader/1.0",
        })

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise CoinAPIError(
                f"GET {url} failed: {resp.status_code} {resp.text[:500]}"
            )
        if not resp.text.strip():
            return None
        return resp.json()

    # -------- Metadata --------
    def list_active_symbols(
        self,
        exchange_id: str,
        filter_asset_id: Optional[str] = None,
        filter_symbol_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if filter_asset_id:
            params["filter_asset_id"] = filter_asset_id
        if filter_symbol_id:
            params["filter_symbol_id"] = filter_symbol_id
        return self.get(f"/v1/symbols/{exchange_id}/active", params=params)

    # -------- OHLCV --------
    def get_ohlcv_history(
        self,
        symbol_id: str,
        period_id: str,
        time_start: str,
        time_end: str,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        return self.get(
            f"/v1/ohlcv/{symbol_id}/history",
            params={
                "period_id": period_id,
                "time_start": time_start,
                "time_end": time_end,
                "limit": limit,
            },
        )

    # -------- Metrics --------
    def list_available_metrics_for_asset(
        self,
        exchange_id: str,
        asset_id: str,
    ) -> List[Dict[str, Any]]:
        return self.get(
            "/v1/metrics/asset/listing",
            params={
                "exchange_id": exchange_id,
                "asset_id": asset_id,
            },
        )

    def get_historical_symbol_metrics(
        self,
        symbol_id: str,
        metric_id: str,
        time_start: str,
        time_end: str,
    ) -> List[Dict[str, Any]]:
        # CoinAPI docs expose "Historical metrics for symbol" in MetricsV1.
        # This path follows the documented naming convention.
        return self.get(
            "/v1/metrics/symbol/history",
            params={
                "symbol_id": symbol_id,
                "metric_id": metric_id,
                "time_start": time_start,
                "time_end": time_end,
            },
        )

    # -------- Current quote / book --------
    def get_current_quote(self, symbol_id: str) -> Dict[str, Any]:
        return self.get(f"/v1/quotes/{symbol_id}/current")

    def get_current_orderbook_depth(self, symbol_id: str, limit_levels: int = 20) -> Dict[str, Any]:
        return self.get(
            f"/v1/orderbooks/{symbol_id}/depth/current",
            params={"limit_levels": limit_levels},
        )


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_btc_derivative_symbols(
    symbols: List[Dict[str, Any]],
    asset_id: str = "BTC",
    preferred_quote_assets: Tuple[str, ...] = ("USDT", "USD", "USDC"),
) -> List[Dict[str, Any]]:
    """
    Pick BTC perpetuals and futures, sorted with PERP first and then by volume.
    """
    filtered = []
    for s in symbols:
        if s.get("asset_id_base") != asset_id:
            continue
        if s.get("symbol_type") not in {"PERPETUAL", "FUTURES"}:
            continue
        if preferred_quote_assets and s.get("asset_id_quote") not in preferred_quote_assets:
            continue
        filtered.append(s)

    def score(x: Dict[str, Any]) -> Tuple[int, float]:
        typ = x.get("symbol_type")
        vol = float(x.get("volume_1day_usd") or 0.0)
        type_rank = 0 if typ == "PERPETUAL" else 1
        return (type_rank, -vol)

    filtered.sort(key=score)
    return filtered


def classify_metric(metric_id: str) -> Optional[str]:
    """
    Coarse grouping based on metric ID text.
    We do not hardcode one exact metric name because support can vary by exchange.
    """
    m = metric_id.upper()
    if "FUNDING" in m:
        return "funding"
    if "OPEN_INTEREST" in m:
        return "open_interest"
    if "LIQUIDATION" in m:
        return "liquidation"
    if "MARK_PRICE" in m:
        return "mark_price"
    if "INDEX_PRICE" in m:
        return "index_price"
    return None


def build_symbol_metric_map(
    metric_listing: List[Dict[str, Any]],
    symbols_of_interest: Iterable[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
        {
          symbol_id: {
            "funding": [metric_id1, ...],
            "open_interest": [...],
            ...
          }
        }
    """
    wanted = set(symbols_of_interest)
    out: Dict[str, Dict[str, List[str]]] = {sid: {} for sid in wanted}

    for row in metric_listing:
        sid = row.get("symbol_id")
        metric_id = row.get("metric_id")
        if not sid or sid not in wanted or not metric_id:
            continue
        category = classify_metric(metric_id)
        if not category:
            continue
        out[sid].setdefault(category, []).append(metric_id)

    return out


def fetch_metric_category(
    client: CoinAPIClient,
    symbol_id: str,
    metric_ids: List[str],
    time_start: str,
    time_end: str,
    sleep_sec: float = 0.15,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Try metric IDs in order and return the first one that successfully yields data.
    """
    last_error: Optional[Exception] = None
    for metric_id in metric_ids:
        try:
            rows = client.get_historical_symbol_metrics(
                symbol_id=symbol_id,
                metric_id=metric_id,
                time_start=time_start,
                time_end=time_end,
            )
            if rows:
                return metric_id, rows
        except Exception as exc:  # keep going
            last_error = exc
        time.sleep(sleep_sec)

    if last_error:
        print(f"[WARN] metric fetch failed for {symbol_id} candidates={metric_ids}: {last_error}", file=sys.stderr)
    return None, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BTC perpetual/futures data from CoinAPI.")
    parser.add_argument("--api-key", default=os.getenv("COINAPI_KEY"), help="CoinAPI API key or set COINAPI_KEY")
    parser.add_argument("--exchange", default="BINANCEFTS", help="CoinAPI exchange_id, e.g. BINANCEFTS, OKEX, DERIBIT")
    parser.add_argument("--asset", default="BTC", help="Base asset, default BTC")
    parser.add_argument("--period", default="1MIN", help="OHLCV period_id, e.g. 1MIN, 5MIN, 1HRS")
    parser.add_argument("--time-start", required=True, help="ISO8601, e.g. 2026-04-01T00:00:00")
    parser.add_argument("--time-end", required=True, help="ISO8601, e.g. 2026-04-03T00:00:00")
    parser.add_argument("--top-n", type=int, default=3, help="How many derivative symbols to download")
    parser.add_argument("--outdir", default="./coinapi_btc_data", help="Output directory")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Pass --api-key or set COINAPI_KEY", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    client = CoinAPIClient(api_key=args.api_key)

    # 1) Discover derivative symbols
    print(f"[INFO] Discovering active symbols on exchange={args.exchange} asset={args.asset}")
    symbols = client.list_active_symbols(exchange_id=args.exchange, filter_asset_id=args.asset)
    derivative_symbols = select_btc_derivative_symbols(symbols, asset_id=args.asset)

    if not derivative_symbols:
        print(f"[ERROR] No BTC perpetual/futures symbols found on {args.exchange}", file=sys.stderr)
        return 1

    selected = derivative_symbols[: args.top_n]
    selected_ids = [s["symbol_id"] for s in selected]

    write_json(outdir / "symbols_selected.json", selected)
    print("[INFO] Selected symbols:")
    for s in selected:
        print(f"  - {s['symbol_id']} ({s['symbol_type']}, quote={s.get('asset_id_quote')}, vol_1day_usd={s.get('volume_1day_usd')})")

    # 2) Discover actual metric availability
    print(f"[INFO] Discovering metric availability for exchange={args.exchange}, asset={args.asset}")
    metric_listing = client.list_available_metrics_for_asset(exchange_id=args.exchange, asset_id=args.asset)
    write_json(outdir / "metric_listing_asset.json", metric_listing)

    symbol_metric_map = build_symbol_metric_map(metric_listing, selected_ids)
    write_json(outdir / "symbol_metric_map.json", symbol_metric_map)

    coverage: List[CoverageRow] = []

    # 3) Download per-symbol datasets
    for sym in selected:
        symbol_id = sym["symbol_id"]
        symbol_type = sym["symbol_type"]
        print(f"[INFO] Downloading for {symbol_id}")
        row = CoverageRow(symbol_id=symbol_id, symbol_type=symbol_type)

        symbol_dir = outdir / safe_slug(symbol_id)
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # OHLCV
        try:
            ohlcv = client.get_ohlcv_history(
                symbol_id=symbol_id,
                period_id=args.period,
                time_start=args.time_start,
                time_end=args.time_end,
            )
            write_jsonl(symbol_dir / f"ohlcv_{args.period}.jsonl", ohlcv)
            row.ohlcv = len(ohlcv) > 0
        except Exception as exc:
            print(f"[WARN] OHLCV failed for {symbol_id}: {exc}", file=sys.stderr)

        # Current quote
        try:
            quote = client.get_current_quote(symbol_id)
            write_json(symbol_dir / "current_quote.json", quote)
            row.current_quote = bool(quote)
        except Exception as exc:
            print(f"[WARN] quote failed for {symbol_id}: {exc}", file=sys.stderr)

        # Current orderbook depth
        try:
            book = client.get_current_orderbook_depth(symbol_id, limit_levels=20)
            write_json(symbol_dir / "current_orderbook_depth.json", book)
            row.current_book_depth = bool(book)
        except Exception as exc:
            print(f"[WARN] orderbook depth failed for {symbol_id}: {exc}", file=sys.stderr)

        # Metrics categories
        categories = symbol_metric_map.get(symbol_id, {})
        for category in ["funding", "open_interest", "liquidation", "mark_price", "index_price"]:
            metric_ids = categories.get(category, [])
            if not metric_ids:
                continue

            metric_id_used, metric_rows = fetch_metric_category(
                client=client,
                symbol_id=symbol_id,
                metric_ids=metric_ids,
                time_start=args.time_start,
                time_end=args.time_end,
            )
            if metric_rows:
                write_jsonl(symbol_dir / f"{category}.jsonl", metric_rows)
                write_json(symbol_dir / f"{category}_meta.json", {
                    "symbol_id": symbol_id,
                    "category": category,
                    "metric_id_used": metric_id_used,
                    "count": len(metric_rows),
                })
                setattr(row, category, True)

        coverage.append(row)

    # 4) Coverage report
    coverage_json = [asdict(x) for x in coverage]
    write_json(outdir / "coverage_report.json", coverage_json)

    print("\n=== COVERAGE REPORT ===")
    for r in coverage:
        print(asdict(r))

    # 5) Final missing-field summary
    required_fields = [
        "ohlcv",
        "funding",
        "open_interest",
        "liquidation",
        "mark_price",
        "index_price",
        "current_quote",
        "current_book_depth",
    ]

    missing_summary = {}
    for r in coverage:
        missing = [f for f in required_fields if not getattr(r, f)]
        missing_summary[r.symbol_id] = missing

    write_json(outdir / "missing_fields_by_symbol.json", missing_summary)

    print("\n=== MISSING FIELDS BY SYMBOL ===")
    for symbol_id, missing in missing_summary.items():
        print(f"{symbol_id}: {missing if missing else 'NONE'}")

    print(f"\n[INFO] Output saved to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# export COINAPI_KEY="你的_key"

# python coinapi_btc_derivatives_downloader.py \
#   --exchange BINANCEFTS \
#   --asset BTC \
#   --period 1MIN \
#   --time-start 2026-04-01T00:00:00 \
#   --time-end   2026-04-03T00:00:00 \
#   --top-n 3 \
#   --outdir ./coinapi_btc_data