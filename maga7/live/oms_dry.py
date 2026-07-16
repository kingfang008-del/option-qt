"""Mag7 OMS dry-run — Scanner signals → fill_model limits → 1s quote sim (no IBKR).

Reuses ``qqq_btc.live.oms_adapter.limit_price_from_quote`` / ``audit_fill`` with Mag7
``FillSpec`` (default frac=0.8). Does **not** route through QQQ TFT entry selection.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import (
    load_quotes,
    path_for_ticker,
    simulate_trade,
    to_ny,
)
from maga7.live.scanner import Mag7Scanner, ScannerSignal
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.live.oms_adapter import audit_fill, limit_price_from_quote

NY = "America/New_York"


@dataclass
class DryOrder:
    """Paper order intent + simulated fill."""

    ts: str
    date: str
    symbol: str
    contract: str
    side: str  # BUY / SELL
    action: str  # OPEN / CLOSE
    limit_px: float
    fill_px: float
    bid: float
    ask: float
    fill_spread_frac: float
    model_frac: float
    rank: int
    direction: str
    reason: str = ""
    mode: str = "DRY_RUN"
    source: str = "maga7_mf10_top2"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        meta = d.pop("meta", {}) or {}
        d.update({f"meta_{k}": v for k, v in meta.items() if not isinstance(v, (dict, list))})
        return d


@dataclass
class DryTrade:
    date: str
    symbol: str
    direction: str
    contract: str
    rank: int
    entry: float
    exit: float
    ret: float
    reason: str
    entry_ts: str
    exit_ts: str
    entry_bid: float
    entry_ask: float
    exit_bid: float
    exit_ask: float
    qty_frac: float
    pnl_equity: float


class Mag7OmsDryRunner:
    """Consume ScannerSignal list → dry orders + equity path (single TopK scheme)."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        fill_cfg = profile.get("fill") or {}
        self.fill = FillSpec(
            entry_frac=float(fill_cfg.get("entry_frac", 0.8)),
            exit_frac=float(fill_cfg.get("exit_frac", 0.8)),
        )
        self.fm = OptionSpreadFillModel(
            entry_frac=self.fill.entry_frac,
            exit_frac=self.fill.exit_frac,
        )
        self.quote_root = profile["_paths"]["quote_1s_root"]
        self.quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        self.orders: list[DryOrder] = []
        self.trades: list[DryTrade] = []
        self.skipped: list[dict[str, Any]] = []

    def _get_q(self, sym: str, date: str):
        k = (sym, date)
        if k not in self.quote_cache:
            self.quote_cache[k] = load_quotes(self.quote_root, sym, date)
        return self.quote_cache[k]

    def _quote_at(self, path: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float, pd.Timestamp] | None:
        after = path[path["timestamp"] >= ts]
        if after.empty:
            return None
        r = after.iloc[0]
        bid, ask = float(r["bid"]), float(r["ask"])
        if not (np.isfinite(bid) and np.isfinite(ask) and ask >= bid > 0):
            return None
        return bid, ask, to_ny(r["timestamp"])

    def process_signals(self, signals: Iterable[ScannerSignal]) -> dict[str, Any]:
        trade_cfg = self.profile["trade"]
        tp = float(trade_cfg.get("tp_mult", 1.6))
        sl = float(trade_cfg.get("sl_mult", 0.4))
        hold = int(trade_cfg.get("hold_minutes", 30))
        pos_frac = float(trade_cfg.get("position_frac", 0.25))
        top_k = max(int(self.profile["signal"].get("top_k", 2)), 1)
        qty_frac = pos_frac / top_k

        eq = 100.0
        peak = 100.0
        maxdd = 0.0
        daily: dict[str, float] = {}
        sig_list = list(signals)
        n_sig = len(sig_list)

        for sig in sig_list:
            if not sig.contract:
                self.skipped.append(
                    {"date": sig.date, "symbol": sig.symbol, "reason": "no_contract", "sig_ts": str(sig.sig_ts)}
                )
                continue
            qdf = self._get_q(sig.symbol, sig.date)
            path = path_for_ticker(qdf, sig.contract)
            if path is None or path.empty:
                self.skipped.append(
                    {
                        "date": sig.date,
                        "symbol": sig.symbol,
                        "contract": sig.contract,
                        "reason": "no_quote_path",
                        "sig_ts": str(sig.sig_ts),
                    }
                )
                continue

            sim = simulate_trade(
                path,
                sig.sig_ts,
                fill=self.fill,
                tp_mult=tp,
                sl_mult=sl,
                hold_minutes=hold,
            )
            if sim is None:
                self.skipped.append(
                    {
                        "date": sig.date,
                        "symbol": sig.symbol,
                        "contract": sig.contract,
                        "reason": "sim_failed",
                        "sig_ts": str(sig.sig_ts),
                    }
                )
                continue

            entry_q = self._quote_at(path, to_ny(sig.sig_ts))
            if entry_q is None:
                self.skipped.append(
                    {"date": sig.date, "symbol": sig.symbol, "reason": "no_entry_quote", "sig_ts": str(sig.sig_ts)}
                )
                continue
            e_bid, e_ask, e_ts = entry_q
            # OMS limit = fill_model (same as assumed dry fill)
            entry_limit = limit_price_from_quote(e_bid, e_ask, "BUY", self.fm)
            entry_audit = audit_fill(e_bid, e_ask, sim.entry, "BUY", self.fm)

            exit_slice = path[path["timestamp"] >= to_ny(sim.exit_ts)]
            if exit_slice.empty:
                exit_slice = path.iloc[[-1]]
            x_bid = float(exit_slice.iloc[0]["bid"])
            x_ask = float(exit_slice.iloc[0]["ask"])
            # Prefer quote exactly at exit_ts
            at_exit = path[path["timestamp"] == to_ny(sim.exit_ts)]
            if not at_exit.empty:
                x_bid = float(at_exit.iloc[0]["bid"])
                x_ask = float(at_exit.iloc[0]["ask"])
            exit_limit = limit_price_from_quote(x_bid, x_ask, "SELL", self.fm)
            exit_audit = audit_fill(x_bid, x_ask, sim.exit, "SELL", self.fm)

            self.orders.append(
                DryOrder(
                    ts=e_ts.isoformat(),
                    date=sig.date,
                    symbol=sig.symbol,
                    contract=sig.contract,
                    side="BUY",
                    action="OPEN",
                    limit_px=entry_limit,
                    fill_px=float(sim.entry),
                    bid=e_bid,
                    ask=e_ask,
                    fill_spread_frac=float(entry_audit.fill_spread_frac),
                    model_frac=float(entry_audit.model_entry_frac),
                    rank=sig.rank,
                    direction=sig.direction,
                    reason="ENTRY",
                    meta={"sig_ts": sig.sig_ts.isoformat(), "fill_frac": self.fill.entry_frac},
                )
            )
            self.orders.append(
                DryOrder(
                    ts=to_ny(sim.exit_ts).isoformat(),
                    date=sig.date,
                    symbol=sig.symbol,
                    contract=sig.contract,
                    side="SELL",
                    action="CLOSE",
                    limit_px=exit_limit,
                    fill_px=float(sim.exit),
                    bid=x_bid,
                    ask=x_ask,
                    fill_spread_frac=float(exit_audit.fill_spread_frac),
                    model_frac=float(self.fill.exit_frac),
                    rank=sig.rank,
                    direction=sig.direction,
                    reason=sim.reason,
                    meta={"entry": sim.entry, "ret": sim.ret},
                )
            )

            pnl = eq * qty_frac * sim.ret
            eq = eq + pnl
            peak = max(peak, eq)
            maxdd = min(maxdd, eq / peak - 1.0)
            daily[sig.date] = eq

            self.trades.append(
                DryTrade(
                    date=sig.date,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    contract=sig.contract,
                    rank=sig.rank,
                    entry=float(sim.entry),
                    exit=float(sim.exit),
                    ret=float(sim.ret),
                    reason=sim.reason,
                    entry_ts=to_ny(sim.entry_ts).isoformat(),
                    exit_ts=to_ny(sim.exit_ts).isoformat(),
                    entry_bid=e_bid,
                    entry_ask=e_ask,
                    exit_bid=x_bid,
                    exit_ask=x_ask,
                    qty_frac=qty_frac,
                    pnl_equity=float(pnl),
                )
            )

        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.ret > 0)
        summary = {
            "mode": "OMS_DRY_RUN",
            "fill_frac": self.fill.entry_frac,
            "n_signals": n_sig,
            "n_trades": n,
            "n_orders": len(self.orders),
            "n_skipped": len(self.skipped),
            "total_ret": eq / 100.0 - 1.0,
            "equity_end": eq,
            "maxdd": maxdd,
            "trade_win": (wins / n) if n else float("nan"),
            "trade_exp": float(np.mean([t.ret for t in self.trades])) if n else float("nan"),
            "source": "maga7_mf10_top2",
        }
        self.summary = summary
        self.daily = [{"date": d, "equity": daily[d]} for d in sorted(daily)]
        return summary

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = getattr(self, "summary", {})
        daily = getattr(self, "daily", [])
        with (out_dir / "orders_dry.jsonl").open("w", encoding="utf-8") as f:
            for o in self.orders:
                f.write(json.dumps(o.to_row(), default=str) + "\n")
        pd.DataFrame([o.to_row() for o in self.orders]).to_csv(out_dir / "orders_dry.csv", index=False)
        pd.DataFrame([asdict(t) for t in self.trades]).to_csv(out_dir / "trades.csv", index=False)
        # fill audit compatible subset
        audit_rows = []
        for o in self.orders:
            audit_rows.append(
                {
                    "ts": o.ts,
                    "symbol": o.symbol,
                    "action": o.action,
                    "side": o.side,
                    "qty": 1.0,
                    "fill_px": o.fill_px,
                    "bid": o.bid,
                    "ask": o.ask,
                    "spread_pct": (o.ask - o.bid) / ((o.ask + o.bid) / 2.0) if o.ask > 0 else "",
                    "fill_spread_frac": o.fill_spread_frac,
                    "model_frac": o.model_frac,
                    "delta_frac": o.fill_spread_frac - o.model_frac
                    if np.isfinite(o.fill_spread_frac)
                    else "",
                    "reason": o.reason,
                    "exit_reason": o.reason if o.action == "CLOSE" else "",
                    "mode": o.mode,
                    "leg": o.contract,
                    "session_bar": "",
                    "net_return": "",
                }
            )
        pd.DataFrame(audit_rows).to_csv(out_dir / "fill_audit.csv", index=False)
        if daily:
            pd.DataFrame(daily).to_csv(out_dir / "daily.csv", index=False)
        if self.skipped:
            pd.DataFrame(self.skipped).to_csv(out_dir / "skipped.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_oms_dry_from_scanner(
    profile: dict[str, Any],
    *,
    signals: list[ScannerSignal],
    out_dir: Path,
) -> dict[str, Any]:
    runner = Mag7OmsDryRunner(profile)
    # process_signals consumes iterable once — pass list
    summary = runner.process_signals(list(signals))
    summary["n_signals"] = len(signals)
    runner.summary = summary
    runner.write(out_dir)
    return summary
