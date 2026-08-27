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
from maga7.common.replay import to_ny
from maga7.live.oms_dry import DryOrder, DryTrade
from maga7.live.oms_fill_session import PendingRedisSignal, QuoteSimSession
from maga7.live.scanner import Mag7Scanner, ScannerSignal
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
    scanner: Mag7Scanner | None = None
    prefer_redis_quotes: bool = False
    orders: list[DryOrder] = field(default_factory=list)
    trades: list[DryTrade] = field(default_factory=list)
    published: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    _redis: Any = field(default=None, repr=False)
    _session: QuoteSimSession | None = field(default=None, repr=False)
    _eq: float = 100.0
    _peak: float = 100.0
    _maxdd: float = 0.0
    _daily: dict[str, float] = field(default_factory=dict)
    _n_sig: int = 0

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
        self._session = QuoteSimSession(self.profile, prefer_redis=self.prefer_redis_quotes)
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

    def size_qty(
        self,
        limit_px: float,
        *,
        symbol: str,
        entry_ts: pd.Timestamp,
        regime_scale: float = 1.0,
    ) -> int:
        """Sleeve = position_frac; concurrent split via QuoteSimSession.open_until."""
        if limit_px <= 0 or self._session is None:
            return 1
        size_frac, allow, _ = self._session.size_frac_for(
            symbol, entry_ts, regime_scale=regime_scale
        )
        if not allow or size_frac <= 0:
            return 0
        from maga7.common.session_risk_budget import (
            current_drawdown,
            parse_session_risk_budget,
            resolve_session_risk_budget,
        )
        from maga7.common.position_size import apply_size_scale

        bud = parse_session_risk_budget((self.profile.get("trade") or {}).get("session_risk_budget"))
        bud_sc, _ = resolve_session_risk_budget(
            bud, current_dd=current_drawdown(self._eq, self._peak)
        )
        size_frac = apply_size_scale(size_frac, bud_sc)
        if size_frac <= 0:
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

    def ingest_option_contracts(
        self,
        symbol: str,
        ts: float,
        contracts: list[dict[str, Any]],
        *,
        resolve_pending: bool = True,
    ) -> None:
        assert self._session is not None
        self._session.ingest_redis_contracts(symbol, ts, contracts)
        if self.prefer_redis_quotes and resolve_pending:
            self.try_resolve_pending(asof_ts=ts)

    def _sync_live_stock_into_session(self) -> None:
        """Prefer scanner-accumulated 1m bars when disk stock_by lacks the session date.

        Offline stock roots may end before a live stress day (e.g. 2026-07-20);
        STOCK_REV / DELTA_STOP need underlying path from the fused/live tape.
        """
        assert self._session is not None
        live = getattr(self.scanner, "stock_by", None) if self.scanner is not None else None
        if not isinstance(live, dict) or not live:
            return
        for sym, df in live.items():
            if df is None or getattr(df, "empty", True):
                continue
            self._session.stock_by[str(sym).upper()] = df

    def try_resolve_pending(self, *, asof_ts: float | pd.Timestamp | None = None) -> list[DryTrade]:
        """Resolve deferred Redis signals once quote book covers hold / early exit."""
        assert self._session is not None
        if not self._session.pending:
            return []
        self._sync_live_stock_into_session()
        asof = None
        if asof_ts is not None:
            if isinstance(asof_ts, (int, float)) and not isinstance(asof_ts, bool):
                asof = pd.Timestamp(float(asof_ts), unit="s", tz="UTC").tz_convert("America/New_York")
            else:
                asof = to_ny(asof_ts)
        done: list[DryTrade] = []
        still: list[PendingRedisSignal] = []
        for pend in self._session.pending:
            sig = pend.sig
            path, src = self._session.get_path(
                sig.symbol, sig.date, sig.contract or "", allow_disk_fallback=False
            )
            if path is None or path.empty or src != "redis":
                still.append(pend)
                continue
            sim = self._session.simulate_on_path(sig, path)
            if sim is None:
                still.append(pend)
                continue
            hold_end = to_ny(sig.sig_ts) + pd.Timedelta(minutes=pend.hold_minutes)
            path_max = to_ny(path["timestamp"].iloc[-1])
            rsn = str(sim.reason or "")
            # Time-based exits need full hold coverage (partial Redis path may pin SEC_MAX/T+).
            time_exit = rsn == "SEC_MAX" or rsn.startswith("T+")
            early = (not time_exit) and (
                rsn
                in {
                    "TP",
                    "SL",
                    "MF_FLIP",
                    "STREAK0",
                    "TRAIL",
                    "TRAIL_LADDER",
                    "PROFIT_STALL",
                    "DELTA_STOP",
                    "STOCK_REV",
                    "ADVERSE_SOFT",
                    "STALE_CUT",
                    "MTM_FLOOR",
                    "MAE_CUT",
                    "HOLD_SHOCK",
                    "TRADE_TOX",
                    "FLOW_DIE",
                    "FLOW_MTM",
                }
                or rsn.startswith("SL_LADDER")
                or rsn.startswith("TP_LADDER")
                or rsn.startswith("ROI_TIME")
                or rsn.startswith("TRADE_TOX")
            )
            ready = False
            if early and (asof is None or to_ny(sim.exit_ts) <= asof):
                ready = True
            elif path_max >= hold_end and (asof is None or asof >= hold_end):
                ready = True
            if not ready:
                still.append(pend)
                continue
            trade = self._commit_sim(sig, sim, path=path, quote_source="redis")
            if trade is not None:
                done.append(trade)
        self._session.pending = still
        return done

    def flush_pending(self) -> list[DryTrade]:
        """Force-resolve remaining pendings (end of day / stream)."""
        assert self._session is not None
        if not self._session.pending:
            return []
        self._sync_live_stock_into_session()
        # allow resolve without asof gate
        done: list[DryTrade] = []
        still: list[PendingRedisSignal] = []
        for pend in self._session.pending:
            sig = pend.sig
            path, src = self._session.get_path(
                sig.symbol, sig.date, sig.contract or "", allow_disk_fallback=False
            )
            if path is None or path.empty:
                self.skipped.append(
                    {"date": sig.date, "symbol": sig.symbol, "reason": "redis_path_missing"}
                )
                still.append(pend)
                continue
            sim = self._session.simulate_on_path(sig, path)
            if sim is None:
                self.skipped.append(
                    {"date": sig.date, "symbol": sig.symbol, "reason": "sim_failed_flush"}
                )
                continue
            trade = self._commit_sim(sig, sim, path=path, quote_source=src)
            if trade is not None:
                done.append(trade)
        self._session.pending = still
        return done

    def _commit_sim(
        self,
        sig: ScannerSignal,
        sim: Any,
        *,
        path: pd.DataFrame,
        quote_source: str,
    ) -> DryTrade | None:
        assert self._session is not None
        from maga7.common.position_size import regime_scale_from_meta

        r_scale = regime_scale_from_meta(sig.meta)
        qty_frac, allow, _ = self._session.size_frac_for(
            sig.symbol, sim.entry_ts, regime_scale=r_scale
        )
        if not allow:
            self.skipped.append(
                {"date": sig.date, "symbol": sig.symbol, "reason": "max_concurrent"}
            )
            return None

        after = path[path["timestamp"] >= to_ny(sig.sig_ts)]
        if after.empty:
            self.skipped.append(
                {"date": sig.date, "symbol": sig.symbol, "reason": "no_entry_quote"}
            )
            return None
        e_bid, e_ask = float(after.iloc[0]["bid"]), float(after.iloc[0]["ask"])
        qty = self.size_qty(
            sim.entry, symbol=sig.symbol, entry_ts=sim.entry_ts, regime_scale=r_scale
        )
        if qty <= 0:
            self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "qty_zero"})
            return None

        buy = self.submit_buy(bid=e_bid, ask=e_ask, qty=qty, ts=sim.entry_ts, sig=sig)
        buy.fill_px = float(sim.entry)
        buy_rec = audit_fill(e_bid, e_ask, buy.fill_px, "BUY", self.fm)
        buy.fill_spread_frac = float(buy_rec.fill_spread_frac)

        at_exit = path[path["timestamp"] == to_ny(sim.exit_ts)]
        if at_exit.empty:
            at_exit = path[path["timestamp"] >= to_ny(sim.exit_ts)]
        if at_exit.empty:
            at_exit = path.iloc[[-1]]
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
            contract=sig.contract or "",
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
            meta={"qty": qty, "size_frac": qty_frac, "quote_source": quote_source},
        )
        close_o = DryOrder(
            ts=sell.ts.isoformat(),
            date=sig.date,
            symbol=sig.symbol,
            contract=sig.contract or "",
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
            meta={"qty": qty, "ret": sim.ret, "size_frac": qty_frac, "quote_source": quote_source},
        )
        self.orders.extend([open_o, close_o])
        self._write_audit(open_o, qty)
        self._write_audit(close_o, qty, net_return=float(sim.ret))

        pnl = self._eq * qty_frac * sim.ret
        self._eq = self._eq + pnl
        self._peak = max(self._peak, self._eq)
        self._maxdd = min(self._maxdd, self._eq / self._peak - 1.0)
        self._daily[sig.date] = self._eq
        self._session.mark_closed(sig.symbol, sim.exit_ts)
        if self.scanner is not None:
            self.scanner.record_fill(sig.symbol, exit_ts=sim.exit_ts, won=sim.ret > 0)

        trade = DryTrade(
            date=sig.date,
            symbol=sig.symbol,
            direction=sig.direction,
            contract=sig.contract or "",
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
        self.trades.append(trade)
        return trade

    def process_one(self, sig: ScannerSignal) -> DryTrade | None:
        """Fill one signal; call from scanner.on_signal for interleaved m5."""
        assert self._session is not None
        self._n_sig += 1
        if not sig.contract:
            self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "no_contract"})
            return None

        # Redis-prefer: defer until the causal quote book covers an exit.
        if self.prefer_redis_quotes:
            hold = self._session.hold_minutes()
            self._session.pending.append(PendingRedisSignal(sig=sig, hold_minutes=hold))
            # Block same-symbol reentry until hold window (scanner cooldown uses last_exit).
            if self.scanner is not None:
                self.scanner.last_exit[sig.symbol] = to_ny(sig.sig_ts) + pd.Timedelta(minutes=hold)
            # Try immediately in case book already has coverage (warm or pitcher ahead).
            done = self.try_resolve_pending(asof_ts=sig.sig_ts)
            return done[-1] if done else None

        sim = self._session.simulate_signal(sig)
        if sim is None:
            self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "sim_failed"})
            return None
        path, src = self._session.get_path(sig.symbol, sig.date, sig.contract)
        if path is None or path.empty:
            self.skipped.append({"date": sig.date, "symbol": sig.symbol, "reason": "no_quote"})
            return None
        return self._commit_sim(sig, sim, path=path, quote_source=src)

    def run_signals(self, signals: list[ScannerSignal]) -> dict[str, Any]:
        """Process signals: stub entry/exit via 1s path (mf_flip + concurrent)."""
        for sig in signals:
            self.process_one(sig)
        if self.prefer_redis_quotes:
            self.flush_pending()
        return self.finalize_summary(n_signals=len(signals))

    def finalize_summary(self, *, n_signals: int | None = None) -> dict[str, Any]:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t.ret > 0)
        trade_cfg = self.profile.get("trade") or {}
        sess = self._session
        summary = {
            "mode": self.mode,
            "fill_frac": self.fill.entry_frac,
            "exit_mode": str(trade_cfg.get("exit_mode") or "none"),
            "position_sizing": str(trade_cfg.get("position_sizing") or "topk"),
            "prefer_redis_quotes": self.prefer_redis_quotes,
            "n_path_redis": int(getattr(sess, "n_path_redis", 0) or 0),
            "n_path_disk": int(getattr(sess, "n_path_disk", 0) or 0),
            "n_redis_quote_updates": int(getattr(getattr(sess, "quote_book", None), "n_updates", 0) or 0),
            "n_pending_left": int(len(getattr(sess, "pending", []) or [])),
            "max_qty": self.max_qty,
            "redis_publish": self.redis_publish,
            "n_signals": n_signals if n_signals is not None else self._n_sig,
            "n_trades": n,
            "n_orders": len(self.orders),
            "n_published": len(self.published),
            "n_skipped": len(self.skipped),
            "total_ret": self._eq / 100.0 - 1.0,
            "equity_end": self._eq,
            "maxdd": self._maxdd,
            "trade_win": (wins / n) if n else float("nan"),
            "trade_exp": float(np.mean([t.ret for t in self.trades])) if n else float("nan"),
            "source": "maga7_mf10_top2",
            "fill_audit_path": str(self.fill_audit_path) if self.fill_audit_path else None,
        }
        self.summary = summary
        self.daily = [{"date": d, "equity": self._daily[d]} for d in sorted(self._daily)]
        return summary

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "orders_stub.jsonl").open("w", encoding="utf-8") as f:
            for o in self.orders:
                f.write(json.dumps(o.to_row(), default=str) + "\n")
        pd.DataFrame([o.to_row() for o in self.orders]).to_csv(out_dir / "orders_stub.csv", index=False)
        pd.DataFrame([asdict(t) for t in self.trades]).to_csv(out_dir / "trades.csv", index=False)
        from maga7.common.trade_log import write_trade_log

        write_trade_log(self.trades, out_dir)
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
