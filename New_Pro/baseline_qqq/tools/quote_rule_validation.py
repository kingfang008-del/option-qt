#!/usr/bin/env python3
"""Validate option execution/routing rules on raw 1s option quotes.

This is a quote-only research harness.  It deliberately avoids TFT/model
signals and does not touch live OMS code.  Its job is narrower: given raw NBBO
quotes for a day, test whether early-open quote gates, fills, exits and option
structures are sane enough to deserve a richer signal layer later.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ENTRY_FRAC = 0.775
EXIT_FRAC = 0.775
COMMISSION_PER_CONTRACT = 0.65
MULTIPLIER = 100.0


@dataclass(frozen=True)
class LegQuote:
    ticker: str
    bid: float
    ask: float
    mid: float
    spread_pct: float
    bid_size: float
    ask_size: float


@dataclass(frozen=True)
class StructureSpec:
    name: str
    legs: tuple[tuple[str, int], ...]
    hard_stop: float
    trail_trigger: float
    trail_keep: float
    time_stop_seconds: int
    max_hold_seconds: int
    risk_frac: float


@dataclass
class OpenPosition:
    strategy: str
    structure: StructureSpec
    entry_ts: pd.Timestamp
    entry_i: int
    entry_debit: float
    contracts: int
    max_roi: float = 0.0


CALL_DEBIT = StructureSpec(
    "CALL_DEBIT_SPREAD",
    (("call_itm", 1), ("call_otm", -1)),
    hard_stop=-0.45,
    trail_trigger=0.35,
    trail_keep=0.55,
    time_stop_seconds=25 * 60,
    max_hold_seconds=70 * 60,
    risk_frac=0.010,
)
PUT_DEBIT = StructureSpec(
    "PUT_DEBIT_SPREAD",
    (("put_itm", 1), ("put_otm", -1)),
    hard_stop=-0.45,
    trail_trigger=0.35,
    trail_keep=0.55,
    time_stop_seconds=25 * 60,
    max_hold_seconds=70 * 60,
    risk_frac=0.010,
)
CALL_RUNNER = StructureSpec(
    "CALL_OTM_RUNNER",
    (("call_otm", 1),),
    hard_stop=-0.35,
    trail_trigger=0.30,
    trail_keep=0.55,
    time_stop_seconds=35 * 60,
    max_hold_seconds=150 * 60,
    risk_frac=0.004,
)
PUT_RUNNER = StructureSpec(
    "PUT_OTM_RUNNER",
    (("put_otm", 1),),
    hard_stop=-0.35,
    trail_trigger=0.30,
    trail_keep=0.55,
    time_stop_seconds=35 * 60,
    max_hold_seconds=150 * 60,
    risk_frac=0.004,
)
STRANGLE = StructureSpec(
    "CALL_PUT_STRANGLE",
    (("call_otm", 1), ("put_otm", 1)),
    hard_stop=-0.28,
    trail_trigger=0.30,
    trail_keep=0.55,
    time_stop_seconds=25 * 60,
    max_hold_seconds=80 * 60,
    risk_frac=0.006,
)
TIGHT_CALL = StructureSpec(
    "LONG_CALL_TIGHT",
    (("call_otm", 1),),
    hard_stop=-0.18,
    trail_trigger=0.25,
    trail_keep=0.60,
    time_stop_seconds=18 * 60,
    max_hold_seconds=45 * 60,
    risk_frac=0.012,
)
TIGHT_PUT = StructureSpec(
    "LONG_PUT_TIGHT",
    (("put_otm", 1),),
    hard_stop=-0.18,
    trail_trigger=0.25,
    trail_keep=0.60,
    time_stop_seconds=18 * 60,
    max_hold_seconds=45 * 60,
    risk_frac=0.012,
)


def _buy_fill(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0 or ask < bid:
        return float("nan")
    return float(bid + ENTRY_FRAC * (ask - bid))


def _sell_fill(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0 or ask < bid:
        return float("nan")
    return float(ask - EXIT_FRAC * (ask - bid))


def _parse_contract(ticker: str) -> tuple[str, str, float]:
    m = re.match(r"AAPL(\d{6})([CP])(\d{8})", ticker)
    if not m:
        raise ValueError(f"Unsupported ticker: {ticker}")
    expiry = "20" + m.group(1)[:2] + "-" + m.group(1)[2:4] + "-" + m.group(1)[4:6]
    strike = int(m.group(3)) / 1000.0
    return expiry, m.group(2), strike


def load_quotes(path: Path, expiry: Optional[str]) -> tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_parquet(path)
    parsed = df["ticker"].map(_parse_contract)
    df["expiry"] = [x[0] for x in parsed]
    df["right"] = [x[1] for x in parsed]
    df["strike"] = [x[2] for x in parsed]
    if expiry is None:
        expiries = sorted(df["expiry"].unique())
        expiry = expiries[0]
    df = df[df["expiry"] == expiry].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"]) & (df["mid"] > 0)].copy()

    calls = df[df["right"] == "C"].sort_values("strike")["ticker"].drop_duplicates().tolist()
    puts = df[df["right"] == "P"].sort_values("strike")["ticker"].drop_duplicates().tolist()
    if len(calls) < 2 or len(puts) < 2:
        raise ValueError("Need at least two calls and two puts for spread/runner validation.")
    mapping = {
        "call_itm": calls[0],
        "call_otm": calls[-1],
        "put_otm": puts[0],
        "put_itm": puts[-1],
    }

    pivot_cols = ["bid", "ask", "mid", "spread_pct", "bid_size", "ask_size"]
    wide = df.pivot_table(index="timestamp", columns="ticker", values=pivot_cols, aggfunc="last")
    wide = wide.sort_index().ffill()
    wide.columns = [f"{ticker}_{field}" for field, ticker in wide.columns]
    wide = wide.reset_index()
    wide["tod"] = wide["timestamp"].dt.strftime("%H:%M:%S")
    wide["sec_from_open"] = (
        (wide["timestamp"].dt.hour - 9) * 3600
        + (wide["timestamp"].dt.minute - 30) * 60
        + wide["timestamp"].dt.second
    )
    return wide, mapping


def _quote(row: pd.Series, ticker: str) -> LegQuote:
    bid = float(row[f"{ticker}_bid"])
    ask = float(row[f"{ticker}_ask"])
    mid = float(row[f"{ticker}_mid"])
    return LegQuote(
        ticker=ticker,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_pct=float(row[f"{ticker}_spread_pct"]),
        bid_size=float(row[f"{ticker}_bid_size"]),
        ask_size=float(row[f"{ticker}_ask_size"]),
    )


def _quality_ok(row: pd.Series, mapping: Dict[str, str], aliases: Iterable[str], max_spread_pct: float) -> bool:
    for alias in aliases:
        q = _quote(row, mapping[alias])
        if not (np.isfinite(q.spread_pct) and q.spread_pct <= max_spread_pct):
            return False
        if min(q.bid_size, q.ask_size) < 2:
            return False
    return True


def _open_debit(row: pd.Series, mapping: Dict[str, str], spec: StructureSpec) -> float:
    total = 0.0
    for alias, qty in spec.legs:
        q = _quote(row, mapping[alias])
        px = _buy_fill(q.bid, q.ask) if qty > 0 else _sell_fill(q.bid, q.ask)
        if not np.isfinite(px):
            return float("nan")
        total += qty * px
    return float(total)


def _close_value(row: pd.Series, mapping: Dict[str, str], spec: StructureSpec) -> float:
    total = 0.0
    for alias, qty in spec.legs:
        q = _quote(row, mapping[alias])
        px = _sell_fill(q.bid, q.ask) if qty > 0 else _buy_fill(q.bid, q.ask)
        if not np.isfinite(px):
            return float("nan")
        total += qty * px
    return float(max(total, 0.0))


def _mark_value(row: pd.Series, mapping: Dict[str, str], spec: StructureSpec) -> float:
    total = 0.0
    for alias, qty in spec.legs:
        q = _quote(row, mapping[alias])
        total += qty * q.mid
    return float(max(total, 0.0))


def _contracts(account: float, spec: StructureSpec, debit: float) -> int:
    risk_budget = account * spec.risk_frac
    risk_per_contract = abs(spec.hard_stop) * debit * MULTIPLIER
    if risk_per_contract <= 0:
        return 0
    return max(1, int(risk_budget / risk_per_contract))


def _commission_drag(debit: float, legs: int) -> float:
    notional = debit * MULTIPLIER
    if notional <= 0:
        return 0.0
    return (2.0 * legs * COMMISSION_PER_CONTRACT) / notional


def _exit_reason(pos: OpenPosition, row: pd.Series, mapping: Dict[str, str], i: int) -> Optional[str]:
    mark = _mark_value(row, mapping, pos.structure)
    if not np.isfinite(mark) or mark <= 0:
        return None
    roi = mark / pos.entry_debit - 1.0
    pos.max_roi = max(pos.max_roi, roi)
    held = int(row["sec_from_open"] - pos.entry_i)
    if roi <= pos.structure.hard_stop:
        return "HARD_STOP"
    if held >= pos.structure.max_hold_seconds:
        return "MAX_HOLD"
    if held >= pos.structure.time_stop_seconds and roi < 0.03:
        return "TIME_STOP"
    if pos.max_roi >= pos.structure.trail_trigger and roi < pos.max_roi * pos.structure.trail_keep:
        return "TRAILING"
    return None


def _trade_dict(
    pos: OpenPosition,
    row: pd.Series,
    mapping: Dict[str, str],
    reason: str,
) -> dict:
    exit_value = _close_value(row, mapping, pos.structure)
    legs = sum(abs(qty) for _, qty in pos.structure.legs)
    gross_roi = exit_value / pos.entry_debit - 1.0
    net_roi = gross_roi - _commission_drag(pos.entry_debit, legs)
    pnl = net_roi * pos.entry_debit * MULTIPLIER * pos.contracts
    return {
        "strategy": pos.strategy,
        "structure": pos.structure.name,
        "entry_ts": str(pos.entry_ts),
        "exit_ts": str(row["timestamp"]),
        "seconds_held": int(row["sec_from_open"] - pos.entry_i),
        "entry_debit": round(pos.entry_debit, 4),
        "exit_value": round(exit_value, 4),
        "contracts": pos.contracts,
        "gross_roi": round(gross_roi, 4),
        "net_roi": round(net_roi, 4),
        "max_mark_roi": round(pos.max_roi, 4),
        "pnl": round(pnl, 2),
        "exit_reason": reason,
    }


def _call_ret(wide: pd.DataFrame, mapping: Dict[str, str], alias: str, seconds: int) -> pd.Series:
    col = f"{mapping[alias]}_mid"
    return wide[col] / wide[col].shift(seconds) - 1.0


def _run_strategy(
    wide: pd.DataFrame,
    mapping: Dict[str, str],
    name: str,
    signal_fn: Callable[[pd.DataFrame, Dict[str, str], int], Optional[StructureSpec]],
    account: float,
) -> List[dict]:
    pos: Optional[OpenPosition] = None
    trades: List[dict] = []
    cooldown_until = -1
    for i, row in wide.iterrows():
        sec = int(row["sec_from_open"])
        if sec < 3:
            continue
        if pos is not None:
            reason = _exit_reason(pos, row, mapping, i)
            if reason is None:
                continue
            trade = _trade_dict(pos, row, mapping, reason)
            trades.append(trade)
            cooldown_until = sec + (10 * 60 if trade["pnl"] < 0 else 3 * 60)
            pos = None
            continue
        if sec < cooldown_until or sec > 11_700:
            continue
        spec = signal_fn(wide, mapping, i)
        if spec is None:
            continue
        max_spread = 0.08 if sec < 60 else 0.06
        if not _quality_ok(row, mapping, (alias for alias, qty in spec.legs if qty > 0), max_spread):
            continue
        debit = _open_debit(row, mapping, spec)
        if not np.isfinite(debit) or debit <= 0:
            continue
        contracts = _contracts(account, spec, debit)
        if contracts <= 0:
            continue
        pos = OpenPosition(
            strategy=name,
            structure=spec,
            entry_ts=row["timestamp"],
            entry_i=sec,
            entry_debit=debit,
            contracts=contracts,
        )
    if pos is not None:
        trades.append(_trade_dict(pos, wide.iloc[-1], mapping, "EOD_CLOSE"))
    return trades


def _sig_naive_call(wide: pd.DataFrame, mapping: Dict[str, str], i: int) -> Optional[StructureSpec]:
    sec = int(wide.iloc[i]["sec_from_open"])
    return TIGHT_CALL if 3 <= sec <= 20 else None


def _sig_tight_momentum(wide: pd.DataFrame, mapping: Dict[str, str], i: int) -> Optional[StructureSpec]:
    if i < 20:
        return None
    row = wide.iloc[i]
    sec = int(row["sec_from_open"])
    if sec > 2 * 3600:
        return None
    call_r5 = _call_ret(wide, mapping, "call_otm", 5).iloc[i]
    put_r5 = _call_ret(wide, mapping, "put_otm", 5).iloc[i]
    if call_r5 >= 0.035 and call_r5 > put_r5 + 0.025:
        return TIGHT_CALL
    if put_r5 >= 0.035 and put_r5 > call_r5 + 0.025:
        return TIGHT_PUT
    return None


def _sig_router(wide: pd.DataFrame, mapping: Dict[str, str], i: int) -> Optional[StructureSpec]:
    if i < 20:
        return None
    row = wide.iloc[i]
    sec = int(row["sec_from_open"])
    if sec > 2 * 3600:
        return None
    call_r5 = _call_ret(wide, mapping, "call_otm", 5).iloc[i]
    put_r5 = _call_ret(wide, mapping, "put_otm", 5).iloc[i]
    call_r20 = _call_ret(wide, mapping, "call_otm", 20).iloc[i]
    put_r20 = _call_ret(wide, mapping, "put_otm", 20).iloc[i]

    # Ambiguous volatility burst: both wings firm up, buy small convexity.
    if call_r5 >= 0.025 and put_r5 >= 0.025:
        return STRANGLE
    # Opening-drive runner: keep convexity when one wing accelerates cleanly.
    # A debit spread is safer but sells the exact tail we want in a 10x+ move.
    if sec <= 180 and call_r5 >= 0.045 and call_r5 > put_r5 + 0.035:
        return CALL_RUNNER
    if sec <= 180 and put_r5 >= 0.045 and put_r5 > call_r5 + 0.035:
        return PUT_RUNNER
    if call_r5 >= 0.06 and call_r20 >= 0.08 and call_r5 > put_r5 + 0.035:
        return CALL_RUNNER
    if put_r5 >= 0.06 and put_r20 >= 0.08 and put_r5 > call_r5 + 0.035:
        return PUT_RUNNER
    if call_r5 >= 0.035 and call_r5 > put_r5 + 0.025:
        return CALL_DEBIT
    if put_r5 >= 0.035 and put_r5 > call_r5 + 0.025:
        return PUT_DEBIT
    return None


def summarize(trades: List[dict], account: float) -> dict:
    pnl = sum(float(t["pnl"]) for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(float(t["pnl"]) for t in wins)
    gross_loss = -sum(float(t["pnl"]) for t in losses)
    structures = {}
    exits = {}
    if trades:
        structures = {str(k): int(v) for k, v in pd.Series([t["structure"] for t in trades]).value_counts().items()}
        exits = {str(k): int(v) for k, v in pd.Series([t["exit_reason"] for t in trades]).value_counts().items()}
    return {
        "trades": len(trades),
        "pnl": round(pnl, 2),
        "roi_on_account": round(pnl / account, 4),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "worst_trade": min((float(t["pnl"]) for t in trades), default=0.0),
        "best_trade": max((float(t["pnl"]) for t in trades), default=0.0),
        "structures": structures,
        "exit_reasons": exits,
    }


def _run_aapl_legacy(args: argparse.Namespace) -> dict:
    quote_path = Path(args.quotes).expanduser()
    wide, mapping = load_quotes(quote_path, args.expiry)
    strategies = {
        "naive_0930_call": _sig_naive_call,
        "tight_momentum_single": _sig_tight_momentum,
        "router_structured": _sig_router,
    }
    result = {
        "mode": "aapl_legacy",
        "quote_path": str(quote_path),
        "rows": int(len(wide)),
        "contracts": mapping,
        "fill_model": {
            "entry_frac": ENTRY_FRAC,
            "exit_frac": EXIT_FRAC,
            "commission_per_contract": COMMISSION_PER_CONTRACT,
        },
        "strategies": {},
    }
    for name, fn in strategies.items():
        trades = _run_strategy(wide, mapping, name, fn, args.account)
        result["strategies"][name] = {
            "summary": summarize(trades, args.account),
            "trades": trades,
        }
    return result


def _run_raw1s_batch(args: argparse.Namespace) -> dict:
    import sys

    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from raw1s_rule_validation import (
        QuoteGateConfig,
        discover_raw1s_days,
        validate_raw1s_batch,
        write_batch_reports,
    )

    raw_dir = Path(args.raw_1s_dir).expanduser()
    files = discover_raw1s_days(
        raw_dir,
        args.symbol,
        glob_pattern=args.glob,
        batch_days=args.batch_days,
    )
    if not files:
        raise FileNotFoundError(
            f"No parquet files for {args.symbol} under {raw_dir} "
            f"(options/ | options_databento/ | {{symbol}}/)"
        )

    fill_fracs = tuple(float(x) for x in args.fill_sensitivity.split(","))
    gate_grid = [
        QuoteGateConfig(3, 0.06, 2, 0),
        QuoteGateConfig(3, 0.04, 2, 0),
        QuoteGateConfig(5, 0.06, 2, 0),
        QuoteGateConfig(3, 0.06, 5, 0),
        QuoteGateConfig(3, 0.06, 2, 3),
        QuoteGateConfig(5, 0.06, 2, 3),
    ]

    import sys

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from qqq_btc.common.fill_model import OptionSpreadFillModel
    from qqq_btc.qqq import config as qcfg

    fill_model = OptionSpreadFillModel(
        entry_frac=args.entry_frac,
        exit_frac=args.exit_frac,
    )
    result = validate_raw1s_batch(
        raw_dir=raw_dir,
        symbol=args.symbol,
        bucket_id=args.bucket,
        files=files,
        fill_model=fill_model,
        rails_cfg=qcfg.EXIT_RAILS,
        replay_cfg=qcfg.REPLAY,
        gate_grid=gate_grid,
        run_sensitivity=args.sensitivity,
        fill_sensitivity=fill_fracs,
    )
    out = Path(args.out)
    write_batch_reports(result, out, write_per_day=not args.no_per_day)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate option rules on raw 1s quotes (AAPL legacy or QQQ raw_1s batch)."
    )
    parser.add_argument(
        "--raw-1s-dir",
        default=None,
        help="Server raw_1s root, e.g. /mnt/s990/data/raw_1s (enables QQQ bucket batch mode)",
    )
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument(
        "--bucket",
        type=int,
        default=2,
        help="Option bucket_id (QQQ CALL ATM = 2)",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Filename glob under options/{symbol}/, e.g. QQQ_2026-*.parquet",
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=None,
        help="Use last N trading-day parquet files",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run qqq_btc rails / fill sensitivity grid (slower)",
    )
    parser.add_argument(
        "--fill-sensitivity",
        default="0.775",
        help="Comma-separated fill fracs when --sensitivity (e.g. 0.65,0.775,0.90)",
    )
    parser.add_argument("--entry-frac", type=float, default=ENTRY_FRAC)
    parser.add_argument("--exit-frac", type=float, default=EXIT_FRAC)
    parser.add_argument("--no-per-day", action="store_true", help="Skip per-day JSON subfolder")

    parser.add_argument("--quotes", default="~/Downloads/AAPL_2026-03-18.parquet")
    parser.add_argument("--expiry", default=None)
    parser.add_argument("--account", type=float, default=50_000.0)
    parser.add_argument(
        "--out",
        default=None,
        help="Aggregate JSON path (default depends on mode)",
    )
    args = parser.parse_args()

    if args.raw_1s_dir:
        if args.out is None:
            tag = f"{args.symbol.lower()}_bucket{args.bucket}"
            if args.batch_days:
                tag += f"_{args.batch_days}d"
            args.out = f"New_Pro/baseline_qqq/reports/{tag}_raw1s_rule_validation.json"
        result = _run_raw1s_batch(args)
        print(
            json.dumps(
                {
                    "aggregate": result.get("aggregate"),
                    "sensitivity_keys": list(result.get("sensitivity", {}).keys()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Wrote {args.out}")
        if not args.no_per_day:
            print(f"Per-day reports: {Path(args.out).parent / (Path(args.out).stem + '_days')}")
        return 0

    if args.out is None:
        args.out = "New_Pro/baseline_qqq/reports/aapl_2026-03-18_quote_rule_validation.json"
    result = _run_aapl_legacy(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in result["strategies"].items()}, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
