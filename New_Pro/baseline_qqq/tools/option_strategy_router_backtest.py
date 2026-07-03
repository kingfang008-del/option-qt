#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-only option structure router backtest.

This is a shadow research harness: it does not touch live OMS code.  It compares
the current directional proxy style (always long ATM call on a bullish rule
signal) with a simple option structure router:

  - LONG_CALL for clean, high-quality trend
  - CALL_DEBIT_SPREAD when trend exists but false-break risk is elevated
  - STRADDLE for high-range, direction-ambiguous open-drive setups

The cached AAPL v3 data currently has CALL_ATM, CALL_OTM and PUT_ATM.  It does
not include PUT_OTM, so this harness intentionally avoids PUT_DEBIT_SPREAD until
that leg is available in the cache/build path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

_BASELINE = Path(__file__).resolve().parents[1]
_REPO = _BASELINE.parents[1]
for p in (str(_REPO), str(_BASELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("QQQ_BTC_LIVE", "1")
os.environ.setdefault("FAST_GATE_ENABLED", "0")

import baseline_paths  # noqa: E402,F401
from qqq_btc.common.exit_rails import ExitRailsConfig, PositionState, check_exit  # noqa: E402
from qqq_btc.qqq import config as qcfg  # noqa: E402
from tools.prototype_rule_stack import (  # noqa: E402
    CACHE_DIR,
    V3Config,
    _dual_scores,
    add_features,
    load_day_v3,
)
from tools.validate_gates_realday import _polygon_key  # noqa: E402


DEFAULT_DATES = ["2026-06-26", "2026-06-29", "2026-07-01", "2026-07-02"]


SINGLE_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.18,
    soft_stop_roi=-0.12,
    early_stop_bars=5,
    early_stop_roi=-0.08,
    time_stop_bars=18,
    time_stop_min_roi=0.03,
    max_hold_bars=45,
    trailing_trigger_roi=0.25,
    trailing_keep_ratio=0.60,
    ladder=((0.10, 0.04), (0.18, 0.10), (0.30, 0.20), (0.50, 0.35)),
    flash_trigger_roi=0.10,
    flash_exit_roi=0.02,
    eod_close_bar_index=380,
)


SPREAD_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.45,
    soft_stop_roi=-0.30,
    early_stop_bars=8,
    early_stop_roi=-0.20,
    time_stop_bars=25,
    time_stop_min_roi=0.02,
    max_hold_bars=55,
    trailing_trigger_roi=0.35,
    trailing_keep_ratio=0.55,
    ladder=((0.15, 0.06), (0.25, 0.12), (0.40, 0.25), (0.65, 0.45)),
    flash_trigger_roi=0.18,
    flash_exit_roi=0.03,
    eod_close_bar_index=380,
)


STRADDLE_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.25,
    soft_stop_roi=-0.18,
    early_stop_bars=6,
    early_stop_roi=-0.12,
    time_stop_bars=18,
    time_stop_min_roi=0.04,
    max_hold_bars=35,
    trailing_trigger_roi=0.25,
    trailing_keep_ratio=0.55,
    ladder=((0.12, 0.04), (0.22, 0.12), (0.35, 0.22)),
    flash_trigger_roi=0.12,
    flash_exit_roi=0.02,
    eod_close_bar_index=380,
)


@dataclass
class RouterConfig:
    account: float = 50_000.0
    max_trades_per_day: int = 3
    cooldown_loss_bars: int = 20
    cooldown_win_bars: int = 8
    daily_loss_stop_usd: float = -1_500.0
    risk_frac_single: float = 0.012
    risk_frac_spread: float = 0.010
    risk_frac_straddle: float = 0.006
    warmup_bars: int = 20
    entry_end_bar: int = 330
    score_threshold: float = 0.34
    strong_score_threshold: float = 0.72
    z_min: float = 0.85
    ambiguity_max: float = 0.22
    straddle_or_range_pct: float = 0.004
    max_single_spread_pct: float = 0.20
    max_leg_spread_pct: float = 0.55
    max_debit_fraction_of_long: float = 0.88


@dataclass
class Position:
    structure: str
    entry_time: pd.Timestamp
    entry_bar: int
    entry_debit: float
    contracts: int
    rails: ExitRailsConfig
    state: PositionState
    meta: Dict[str, float]


def _session_bar(t: pd.Timestamp) -> int:
    return int((t.hour - 9) * 60 + t.minute - 30)


def _leg_spread(row: pd.Series, leg: str) -> float:
    return float(row.get(f"{leg}_spread", np.nan))


def _safe_fill(px: float, fallback: float) -> float:
    if np.isfinite(px) and px > 0:
        return float(px)
    return float(fallback) if np.isfinite(fallback) and fallback > 0 else float("nan")


def _long_call_open(row: pd.Series) -> float:
    return _safe_fill(
        qcfg.FILL_MODEL.entry_fill(row["call_atm_bid"], row["call_atm_ask"]),
        row["call_atm_mid"],
    )


def _long_call_close(row: pd.Series) -> float:
    return _safe_fill(
        qcfg.FILL_MODEL.exit_fill(row["call_atm_bid"], row["call_atm_ask"]),
        row["call_atm_mid"],
    )


def _call_spread_open(row: pd.Series) -> float:
    long_buy = _long_call_open(row)
    short_sell = _safe_fill(
        qcfg.FILL_MODEL.exit_fill(row["call_otm_bid"], row["call_otm_ask"]),
        row["call_otm_mid"],
    )
    return long_buy - short_sell


def _call_spread_close(row: pd.Series) -> float:
    long_sell = _long_call_close(row)
    short_buy = _safe_fill(
        qcfg.FILL_MODEL.entry_fill(row["call_otm_bid"], row["call_otm_ask"]),
        row["call_otm_mid"],
    )
    return max(long_sell - short_buy, 0.0)


def _straddle_open(row: pd.Series) -> float:
    call_buy = _long_call_open(row)
    put_buy = _safe_fill(
        qcfg.FILL_MODEL.entry_fill(row["put_atm_bid"], row["put_atm_ask"]),
        row["put_atm_mid"],
    )
    return call_buy + put_buy


def _straddle_close(row: pd.Series) -> float:
    call_sell = _long_call_close(row)
    put_sell = _safe_fill(
        qcfg.FILL_MODEL.exit_fill(row["put_atm_bid"], row["put_atm_ask"]),
        row["put_atm_mid"],
    )
    return call_sell + put_sell


def _mark(row: pd.Series, structure: str) -> float:
    if structure == "CALL_DEBIT_SPREAD":
        return max(float(row["call_atm_mid"]) - float(row["call_otm_mid"]), 0.0)
    if structure == "STRADDLE":
        return float(row["call_atm_mid"]) + float(row["put_atm_mid"])
    return float(row["call_atm_mid"])


def _open_debit(row: pd.Series, structure: str) -> float:
    if structure == "CALL_DEBIT_SPREAD":
        return _call_spread_open(row)
    if structure == "STRADDLE":
        return _straddle_open(row)
    return _long_call_open(row)


def _close_value(row: pd.Series, structure: str) -> float:
    if structure == "CALL_DEBIT_SPREAD":
        return _call_spread_close(row)
    if structure == "STRADDLE":
        return _straddle_close(row)
    return _long_call_close(row)


def _round_trip_commission_drag(entry_debit: float, legs: int) -> float:
    notional = entry_debit * qcfg.FILL_MODEL.contract_multiplier
    if notional <= 0:
        return float("nan")
    return float((2 * legs * qcfg.FILL_MODEL.commission_per_contract) / notional)


def _contracts(cfg: RouterConfig, structure: str, entry_debit: float, rails: ExitRailsConfig) -> int:
    if entry_debit <= 0:
        return 0
    risk_frac = cfg.risk_frac_single
    if structure == "CALL_DEBIT_SPREAD":
        risk_frac = cfg.risk_frac_spread
    elif structure == "STRADDLE":
        risk_frac = cfg.risk_frac_straddle
    budget = cfg.account * risk_frac
    risk_per_contract = abs(rails.hard_stop_roi) * entry_debit * 100.0
    if risk_per_contract <= 0:
        return 0
    return max(1, int(budget / risk_per_contract))


def _or_range_pct(d: pd.DataFrame, i: int, bars: int) -> float:
    start = max(0, i - bars + 1)
    win = d.iloc[start : i + 1]
    base = max(float(win["stock_close"].iloc[0]), 1.0)
    return float((win["stock_high"].max() - win["stock_low"].min()) / base)


def _signal(d: pd.DataFrame, i: int, cfg: RouterConfig) -> Optional[dict]:
    row = d.iloc[i]
    sb = int(row["session_bar"])
    if i < cfg.warmup_bars or sb > cfg.entry_end_bar:
        return None
    if not np.isfinite(row.get("z5", np.nan)) or float(row["z5"]) < cfg.z_min:
        return None

    call_score, put_score, meta = _dual_scores(d, i, V3Config().or_bars)
    spread_call = _leg_spread(row, "call_atm")
    spread_otm = _leg_spread(row, "call_otm")
    spread_put = _leg_spread(row, "put_atm")
    er = float(row.get("er30", np.nan))
    above_frac = float(row.get("above_vwap_frac30", np.nan))
    close = float(row["stock_close"])
    vwap = float(row["vwap"])

    ambiguous = abs(call_score - put_score) <= cfg.ambiguity_max
    high_range = _or_range_pct(d, i, V3Config().or_bars) >= cfg.straddle_or_range_pct
    if (
        ambiguous
        and high_range
        and spread_call <= cfg.max_single_spread_pct
        and spread_put <= cfg.max_single_spread_pct
    ):
        return {
            "baseline": "LONG_CALL",
            "routed": "STRADDLE",
            "reason": "ambiguous_high_range",
            "call_score": call_score,
            "put_score": put_score,
        }

    bullish = (
        call_score >= cfg.score_threshold
        and call_score > put_score
        and close > vwap
        and np.isfinite(spread_call)
        and spread_call <= cfg.max_single_spread_pct
    )
    if not bullish:
        return None

    false_break_risk = (
        not np.isfinite(er)
        or er < 0.34
        or above_frac < 0.68
        or sb < 45
        or spread_call > 0.12
    )
    clean_trend = (
        call_score >= cfg.strong_score_threshold
        and np.isfinite(er)
        and er >= 0.34
        and above_frac >= 0.72
        and spread_call <= 0.12
    )
    routed = "LONG_CALL"
    if false_break_risk and not clean_trend:
        debit = _call_spread_open(row)
        long_debit = _long_call_open(row)
        spread_ok = (
            np.isfinite(debit)
            and debit > 0.03
            and np.isfinite(long_debit)
            and debit <= long_debit * cfg.max_debit_fraction_of_long
            and spread_otm <= cfg.max_leg_spread_pct
        )
        if spread_ok:
            routed = "CALL_DEBIT_SPREAD"

    return {
        "baseline": "LONG_CALL",
        "routed": routed,
        "reason": "clean_trend" if routed == "LONG_CALL" else "false_break_risk",
        "call_score": call_score,
        "put_score": put_score,
        "slope": float(meta.get("slope", 0.0)),
        "er30": er,
    }


def _rails(structure: str) -> ExitRailsConfig:
    if structure == "CALL_DEBIT_SPREAD":
        return SPREAD_RAILS
    if structure == "STRADDLE":
        return STRADDLE_RAILS
    return SINGLE_RAILS


def _legs(structure: str) -> int:
    return 2 if structure in {"CALL_DEBIT_SPREAD", "STRADDLE"} else 1


def _run_mode(d: pd.DataFrame, cfg: RouterConfig, mode: str) -> dict:
    position: Optional[Position] = None
    trades: List[dict] = []
    blocks: Dict[str, int] = {}
    cooldown_until = -1
    halted = False

    def block(reason: str) -> None:
        blocks[reason] = blocks.get(reason, 0) + 1

    for i in range(len(d)):
        row = d.iloc[i]
        sb = int(row["session_bar"])

        if position is not None:
            mtm = _mark(row, position.structure)
            reason = check_exit(position.rails, position.state, mtm, i, sb)
            if reason:
                exit_value = _close_value(row, position.structure)
                drag = _round_trip_commission_drag(position.entry_debit, _legs(position.structure))
                net = exit_value / position.entry_debit - 1.0 - drag
                pnl = position.contracts * 100.0 * position.entry_debit * net
                trades.append({
                    "structure": position.structure,
                    "entry": position.entry_time.isoformat(),
                    "exit": row["time"].isoformat(),
                    "entry_debit": position.entry_debit,
                    "exit_value": exit_value,
                    "contracts": position.contracts,
                    "net": net,
                    "pnl": pnl,
                    "exit_reason": reason,
                    **position.meta,
                })
                cooldown_until = i + (cfg.cooldown_loss_bars if pnl < 0 else cfg.cooldown_win_bars)
                position = None
                if sum(t["pnl"] for t in trades) <= cfg.daily_loss_stop_usd:
                    halted = True
            continue

        if halted:
            block("halted")
            continue
        if len(trades) >= cfg.max_trades_per_day:
            block("max_trades")
            continue
        if i <= cooldown_until:
            block("cooldown")
            continue

        sig = _signal(d, i, cfg)
        if sig is None:
            block("no_signal")
            continue
        structure = sig[mode]
        rails = _rails(structure)
        entry_debit = _open_debit(row, structure)
        if not np.isfinite(entry_debit) or entry_debit <= 0:
            block("bad_debit")
            continue
        contracts = _contracts(cfg, structure, entry_debit, rails)
        if contracts <= 0:
            block("no_contracts")
            continue
        position = Position(
            structure=structure,
            entry_time=row["time"],
            entry_bar=i,
            entry_debit=float(entry_debit),
            contracts=contracts,
            rails=rails,
            state=PositionState(entry_price=float(entry_debit), entry_bar=i),
            meta={
                "route_reason": sig["reason"],
                "call_score": float(sig["call_score"]),
                "put_score": float(sig["put_score"]),
                "stock_entry": float(row["stock_close"]),
            },
        )

    if position is not None:
        row = d.iloc[-1]
        exit_value = _close_value(row, position.structure)
        drag = _round_trip_commission_drag(position.entry_debit, _legs(position.structure))
        net = exit_value / position.entry_debit - 1.0 - drag
        pnl = position.contracts * 100.0 * position.entry_debit * net
        trades.append({
            "structure": position.structure,
            "entry": position.entry_time.isoformat(),
            "exit": row["time"].isoformat(),
            "entry_debit": position.entry_debit,
            "exit_value": exit_value,
            "contracts": position.contracts,
            "net": net,
            "pnl": pnl,
            "exit_reason": "FORCE_EOD",
            **position.meta,
        })

    pnl = float(sum(t["pnl"] for t in trades))
    return {
        "trades": trades,
        "pnl": pnl,
        "roi": pnl / cfg.account,
        "blocks": blocks,
    }


def run_day(symbol: str, date: str, cfg: RouterConfig, key: str) -> dict:
    df = load_day_v3(symbol, date, key)
    d = add_features(df)
    d["session_bar"] = [_session_bar(t) for t in d["time"]]
    baseline = _run_mode(d, cfg, "baseline")
    routed = _run_mode(d, cfg, "routed")
    return {
        "date": date,
        "rows": int(len(d)),
        "baseline_single": baseline,
        "routed": routed,
    }


def _summary(results: Iterable[dict], account: float) -> dict:
    out = {}
    for mode in ("baseline_single", "routed"):
        pnls = [float(r[mode]["pnl"]) for r in results]
        trades = [t for r in results for t in r[mode]["trades"]]
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        by_structure: Dict[str, int] = {}
        for t in trades:
            by_structure[t["structure"]] = by_structure.get(t["structure"], 0) + 1
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = -sum(t["pnl"] for t in losses)
        out[mode] = {
            "days": len(pnls),
            "trades": len(trades),
            "pnl": float(sum(pnls)),
            "roi": float(sum(pnls) / account),
            "avg_day_roi": float(sum(pnls) / max(len(pnls), 1) / account),
            "worst_day_roi": float(min(pnls, default=0.0) / account),
            "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
            "by_structure": by_structure,
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AAPL rule-only option structure router backtest")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--json-out", default=str(CACHE_DIR.parent / "option_router_aapl_recent_week.json"))
    args = ap.parse_args(argv)

    cfg = RouterConfig()
    key = _polygon_key()
    results = []
    for date in args.dates:
        try:
            results.append(run_day(args.symbol.upper(), date, cfg, key))
        except Exception as exc:
            print(f"{date} SKIP: {exc}")

    summary = _summary(results, cfg.account)
    payload = {"symbol": args.symbol.upper(), "dates": [r["date"] for r in results], "summary": summary, "days": results}
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {args.symbol.upper()} option structure router recent-week test ===")
    print(f"dates: {', '.join(payload['dates'])}")
    for mode in ("baseline_single", "routed"):
        s = summary[mode]
        pf = "inf" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
        print(
            f"{mode:16s} trades={s['trades']:2d} pnl=${s['pnl']:+8.0f} "
            f"roi={s['roi']:+.2%} worst_day={s['worst_day_roi']:+.2%} "
            f"win={s['win_rate']:.0%} pf={pf} structures={s['by_structure']}"
        )
    print("\nPer day:")
    for r in results:
        b = r["baseline_single"]
        ro = r["routed"]
        print(
            f"{r['date']} rows={r['rows']:3d} "
            f"baseline={b['roi']:+.2%}/{len(b['trades'])}t "
            f"routed={ro['roi']:+.2%}/{len(ro['trades'])}t"
        )
    print(f"\njson: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
