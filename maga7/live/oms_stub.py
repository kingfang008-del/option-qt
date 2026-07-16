"""Mag7 OMS stub (S4) — small-qty shadow OMS, no QQQ TFT / no IBKR required.

Uses ``oms_adapter.limit_price_from_quote`` (frac=0.8) + 1s quote path for fills.
Optional Redis publish of mapped BUY/SELL payloads (default OFF).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.fills import FillSpec
from maga7.common.replay import load_quotes, path_for_ticker, simulate_trade, to_ny
from maga7.live.oms_dry import DryOrder, DryTrade
from maga7.live.scanner import ScannerSignal
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.live.oms_adapter import audit_fill, limit_price_from_quote

logger = logging.getLogger("maga7.live.oms_stub")

FILL_AUDIT_HEADER = (
    "ts",
    "symbol",
    "action",
    "side",
    "qty",
    "fill_px",
    "bid",
    "ask",
    "spread_pct",
    "fill_spread_frac",
    "model_frac",
    "delta_frac",
    "reason",
    "exit_reason",
    "mode",
    "leg",
    "session_bar",
    "net_return",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def append_mag7_fill_audit(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    line = {k: row.get(k, "") for k in FILL_AUDIT_HEADER}
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FILL_AUDIT_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(line)


@dataclass
class StubFill:
    action: str
    side: str
    limit_px: float
    fill_px: float
    bid: float
    ask: float
    qty: int
    ts: pd.Timestamp
    fill_spread_frac: float
    model_frac: float


@dataclass
class Mag7OmsStub:
    """Independent Mag7 OMS: dry-submit + fill_audit + optional Redis xadd."""

    profile: dict[str, Any]
    max_qty: int = 1
    mode: str = "MAG7_SHADOW"
    redis_publish: bool = False
    redis_stream: str = "orch_trade_signals"
    fill_audit_path: Path | None = None
    equity: float = 100_000.0  # notional dollars for qty sizing (paper)
    orders: list[DryOrder] = field(default_factory=list)
    trades: list[DryTrade] = field(default_factory=list)
    published: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    _redis: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        fill_cfg = self.profile.get("fill") or {}
        self.fill = FillSpec(
            entry_frac=float(fill_cfg.get("entry_frac", 0.8)),
            exit_frac=float(fill_cfg.get("exit_frac", 0.8)),
        )
        self.fm = OptionSpreadFillModel(
            entry_frac=self.fill.entry_frac,
            exit_frac=self.fill.exit_frac,
        )
        self.quote_root = self.profile["_paths"]["quote_1s_root"]
        self.quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
        if self.fill_audit_path is None:
            env_p = os.environ.get("MAG7_FILL_AUDIT_PATH", "").strip()
            if env_p:
                self.fill_audit_path = Path(env_p).expanduser()
        if self.max_qty <= 0:
            self.max_qty = _env_int("MAG7_MAX_QTY", 1)
        if not self.redis_publish:
            self.redis_publish = _env_bool("MAG7_REDIS_PUBLISH", False)

    @classmethod
    def from_profile(cls, profile: dict[str, Any], **kwargs) -> "Mag7OmsStub":
        max_qty = int(kwargs.pop("max_qty", _env_int("MAG7_MAX_QTY", 1)))
        mode = str(kwargs.pop("mode", os.environ.get("MAG7_OMS_MODE", "MAG7_SHADOW")))
        redis_publish = bool(kwargs.pop("redis_publish", _env_bool("MAG7_REDIS_PUBLISH", False)))
        return cls(
            profile=profile,
            max_qty=max_qty,
            mode=mode,
            redis_publish=redis_publish,
            **kwargs,
        )

    def _get_q(self, sym: str, date: str):
        k = (sym, date)
        if k not in self.quote_cache:
            self.quote_cache[k] = load_quotes(self.quote_root, sym, date)
        return self.quote_cache[k]

    def _connect_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis
        except ImportError:
            logger.warning("redis package missing; publish disabled")
            self.redis_publish = False
            return None
        host = os.environ.get("REDIS_HOST", "127.0.0.1")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        self._redis = redis.Redis(host=host, port=port, decode_responses=True)
        return self._redis

    def _publish(self, payload: dict[str, Any]) -> None:
        self.published.append(payload)
        if not self.redis_publish:
            return
        r = self._connect_redis()
        if r is None:
            return
        r.xadd(self.redis_stream, {"data": json.dumps(payload, default=str)})
        logger.info("redis xadd %s action=%s %s", self.redis_stream, payload.get("action"), payload.get("symbol"))

    def _audit_row(self, order: DryOrder, qty: int, net_return: Any = "") -> dict[str, Any]:
        spread_pct = (order.ask - order.bid) / ((order.ask + order.bid) / 2.0) if order.ask > 0 else ""
        return {
            "ts": order.ts,
            "symbol": order.symbol,
            "action": order.action,
            "side": order.side,
            "qty": qty,
            "fill_px": order.fill_px,
            "bid": order.bid,
            "ask": order.ask,
            "spread_pct": spread_pct,
            "fill_spread_frac": order.fill_spread_frac,
            "model_frac": order.model_frac,
            "delta_frac": order.fill_spread_frac - order.model_frac
            if np.isfinite(order.fill_spread_frac)
            else "",
            "reason": order.reason,
            "exit_reason": order.reason if order.action == "CLOSE" else "",
            "mode": self.mode,
            "leg": order.contract,
            "session_bar": "",
            "net_return": net_return,
        }

    def _write_audit(self, order: DryOrder, qty: int, net_return: Any = "") -> None:
        if self.fill_audit_path is None:
            return
        append_mag7_fill_audit(self._audit_row(order, qty, net_return), self.fill_audit_path)

    def size_qty(self, limit_px: float) -> int:
        """Sleeve = position_frac; concurrent split only when open_until is known."""
        if limit_px <= 0:
            return 1
        trade = self.profile.get("trade") or {}
        from maga7.common.position_size import resolve_size_frac

        open_until = getattr(self, "open_until", None) or {}
        size_frac, _, _, allow, _ = resolve_size_frac(
            trade,
            top_k=max(int(self.profile["signal"].get("top_k", 2)), 1),
            open_until=open_until,
            symbol=None,
            entry_ts=pd.Timestamp.now(tz="America/New_York"),
        )
        if not allow:
            return 0
        notional = self.equity * size_frac
        raw = int(notional // (limit_px * 100.0))
        return max(1, min(self.max_qty, max(raw, 1)))

    def submit_buy(
        self,
        *,
        bid: float,
        ask: float,
        qty: int,
        ts: pd.Timestamp,
        sig: ScannerSignal,
    ) -> StubFill:
        limit = limit_price_from_quote(bid, ask, "BUY", self.fm)
        # Shadow: assume limit fills at model price
        fill_px = float(limit)
        rec = audit_fill(bid, ask, fill_px, "BUY", self.fm)
        fill = StubFill(
            action="OPEN",
            side="BUY",
            limit_px=limit,
            fill_px=fill_px,
            bid=bid,
            ask=ask,
            qty=qty,
            ts=to_ny(ts),
            fill_spread_frac=float(rec.fill_spread_frac),
            model_frac=float(rec.model_entry_frac),
        )
        self._publish(
            sig.to_oms_exec_payload(
                action="BUY", bid=bid, ask=ask, limit_px=limit, qty=qty, ts=fill.ts
            )
        )
        return fill

    def submit_sell(
        self,
        *,
        bid: float,
        ask: float,
        qty: int,
        ts: pd.Timestamp,
        sig: ScannerSignal,
        reason: str,
    ) -> StubFill:
        limit = limit_price_from_quote(bid, ask, "SELL", self.fm)
        fill_px = float(limit)
        rec = audit_fill(bid, ask, fill_px, "SELL", self.fm)
        fill = StubFill(
            action="CLOSE",
            side="SELL",
            limit_px=limit,
            fill_px=fill_px,
            bid=bid,
            ask=ask,
            qty=qty,
            ts=to_ny(ts),
            fill_spread_frac=float(rec.fill_spread_frac),
            model_frac=float(self.fill.exit_frac),
        )
        self._publish(
            sig.to_oms_exec_payload(
                action="SELL", bid=bid, ask=ask, limit_px=limit, qty=qty, ts=fill.ts
            )
        )
        return fill

    def run_signals(self, signals: list[ScannerSignal]) -> dict[str, Any]:
        """Process TopK signals: stub entry/exit via 1s path (parity with S3/offline)."""
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

        for sig in signals:
            if not sig.contract:
                self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "no_contract"})
                continue
            path = path_for_ticker(self._get_q(sig.symbol, sig.date), sig.contract)
            if path is None or path.empty:
                self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "no_quote"})
                continue
            sim = simulate_trade(path, sig.sig_ts, fill=self.fill, tp_mult=tp, sl_mult=sl, hold_minutes=hold)
            if sim is None:
                self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "sim_failed"})
                continue

            after = path[path["timestamp"] >= to_ny(sig.sig_ts)]
            e_bid, e_ask = float(after.iloc[0]["bid"]), float(after.iloc[0]["ask"])
            qty = self.size_qty(sim.entry)
            buy = self.submit_buy(bid=e_bid, ask=e_ask, qty=qty, ts=sim.entry_ts, sig=sig)
            # Align fill to sim (1s path) for offline parity; limit still from adapter.
            buy.fill_px = float(sim.entry)
            buy_rec = audit_fill(e_bid, e_ask, buy.fill_px, "BUY", self.fm)
            buy.fill_spread_frac = float(buy_rec.fill_spread_frac)

            at_exit = path[path["timestamp"] == to_ny(sim.exit_ts)]
            if at_exit.empty:
                at_exit = path[path["timestamp"] >= to_ny(sim.exit_ts)]
            x_bid = float(at_exit.iloc[0]["bid"])
            x_ask = float(at_exit.iloc[0]["ask"])
            sell = self.submit_sell(
                bid=x_bid, ask=x_ask, qty=qty, ts=sim.exit_ts, sig=sig, reason=sim.reason
            )
            sell.fill_px = float(sim.exit)
            sell_rec = audit_fill(x_bid, x_ask, sell.fill_px, "SELL", self.fm)
            sell.fill_spread_frac = float(sell_rec.fill_spread_frac)

            open_o = DryOrder(
                ts=buy.ts.isoformat(),
                date=sig.date,
                symbol=sig.symbol,
                contract=sig.contract,
                side="BUY",
                action="OPEN",
                limit_px=buy.limit_px,
                fill_px=buy.fill_px,
                bid=e_bid,
                ask=e_ask,
                fill_spread_frac=buy.fill_spread_frac,
                model_frac=buy.model_frac,
                rank=sig.rank,
                direction=sig.direction,
                reason="ENTRY",
                mode=self.mode,
                meta={"qty": qty},
            )
            close_o = DryOrder(
                ts=sell.ts.isoformat(),
                date=sig.date,
                symbol=sig.symbol,
                contract=sig.contract,
                side="SELL",
                action="CLOSE",
                limit_px=sell.limit_px,
                fill_px=sell.fill_px,
                bid=x_bid,
                ask=x_ask,
                fill_spread_frac=sell.fill_spread_frac,
                model_frac=sell.model_frac,
                rank=sig.rank,
                direction=sig.direction,
                reason=sim.reason,
                mode=self.mode,
                meta={"qty": qty, "ret": sim.ret},
            )
            self.orders.extend([open_o, close_o])
            self._write_audit(open_o, qty)
            self._write_audit(close_o, qty, net_return=float(sim.ret))

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
            "mode": self.mode,
            "fill_frac": self.fill.entry_frac,
            "max_qty": self.max_qty,
            "redis_publish": self.redis_publish,
            "n_signals": len(signals),
            "n_trades": n,
            "n_orders": len(self.orders),
            "n_published": len(self.published),
            "n_skipped": len(self.skipped),
            "total_ret": eq / 100.0 - 1.0,
            "equity_end": eq,
            "maxdd": maxdd,
            "trade_win": (wins / n) if n else float("nan"),
            "trade_exp": float(np.mean([t.ret for t in self.trades])) if n else float("nan"),
            "source": "maga7_mf10_top2",
            "fill_audit_path": str(self.fill_audit_path) if self.fill_audit_path else None,
        }
        self.summary = summary
        self.daily = [{"date": d, "equity": daily[d]} for d in sorted(daily)]
        return summary

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "orders_stub.jsonl").open("w", encoding="utf-8") as f:
            for o in self.orders:
                f.write(json.dumps(o.to_row(), default=str) + "\n")
        pd.DataFrame([o.to_row() for o in self.orders]).to_csv(out_dir / "orders_stub.csv", index=False)
        pd.DataFrame([asdict(t) for t in self.trades]).to_csv(out_dir / "trades.csv", index=False)
        audit_rows = [self._audit_row(o, int((o.meta or {}).get("qty", 1)), o.meta.get("ret", "")) for o in self.orders]
        pd.DataFrame(audit_rows).to_csv(out_dir / "fill_audit.csv", index=False)
        # also mirror to session audit path if set and different
        if self.fill_audit_path and Path(self.fill_audit_path).resolve() != (out_dir / "fill_audit.csv").resolve():
            # already appended live; copy snapshot
            pass
        if getattr(self, "daily", None):
            pd.DataFrame(self.daily).to_csv(out_dir / "daily.csv", index=False)
        if self.skipped:
            pd.DataFrame(self.skipped).to_csv(out_dir / "skipped.csv", index=False)
        if self.published:
            with (out_dir / "redis_payloads.jsonl").open("w", encoding="utf-8") as f:
                for p in self.published:
                    f.write(json.dumps(p, default=str) + "\n")
        (out_dir / "summary.json").write_text(
            json.dumps(getattr(self, "summary", {}), indent=2), encoding="utf-8"
        )
