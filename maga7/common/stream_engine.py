"""Streaming engine for Rule-A TopK scalp — causal bar/tick path for parity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from maga7.common.contract_select import day_iv_path_as_quotes
from maga7.common.entry_contract import ContractBooks, resolve_entry_contract
from maga7.common.fills import FillSpec
from maga7.common.position_size import resolve_size_frac
from maga7.common.reentry import resolve_only_win_reenter
from maga7.common.replay import load_quotes, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import StreamSignalState


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    ticker: str
    entry: float
    entry_ts: pd.Timestamp
    exit_deadline: pd.Timestamp
    tp: float
    sl: float
    n_in_day: int


@dataclass
class StreamEngine:
    """
    Causal engine:
      1) ingest 1m stock bars → Rule-A fires
      2) day-level TopK by earliest fire time
      3) resolve contract (day_lock / open_lock / signal_atm + clear_otm)
      4) optional regime gate
      5) enter/exit on 1s (or day_iv) quotes with FillSpec
    """

    profile: dict[str, Any]
    fill: FillSpec
    books: ContractBooks
    quote_cache: dict = field(default_factory=dict)
    states: dict[str, StreamSignalState] = field(default_factory=dict)
    day_fires: list[dict[str, Any]] = field(default_factory=list)
    pending_fires: list[dict[str, Any]] = field(default_factory=list)
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    current_date: str | None = None
    last_exit: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    last_win: dict[str, bool] = field(default_factory=dict)
    n_done: dict[str, int] = field(default_factory=dict)
    equity: float = 100.0
    day_start_equity: float = 100.0
    peak: float = 100.0
    maxdd: float = 0.0
    day_halt: bool = False
    scheme: str = "single"
    regime_gate: Any = None
    n_regime_block: int = 0
    n_skip0_clear_otm: int = 0
    stock_by: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: dict[str, Any], *, scheme: str = "single") -> "StreamEngine":
        fill = FillSpec(
            entry_frac=float(profile["fill"].get("entry_frac", 0.8)),
            exit_frac=float(profile["fill"].get("exit_frac", 0.8)),
        )
        books = ContractBooks.from_profile(profile)
        emit_all = scheme.startswith("m5")
        states = {
            s: StreamSignalState(s, profile["signal"], emit_all=emit_all) for s in profile["symbols"]
        }
        regime_gate = None
        try:
            from maga7.common.regime import Mag7RegimeGate
            from maga7.common.replay import month_list

            start = profile["date_range"]["start"]
            end = profile["date_range"]["end"]
            regime_gate = Mag7RegimeGate.from_profile(profile, months=month_list(start, end))
        except Exception:
            regime_gate = None
        return cls(
            profile=profile,
            fill=fill,
            books=books,
            states=states,
            scheme=scheme,
            regime_gate=regime_gate,
        )

    def _roll_day(self, date: str) -> None:
        if self.current_date == date:
            return
        self.current_date = date
        self.day_fires = []
        self.pending_fires = []
        self.positions = {}
        self.day_start_equity = self.equity
        self.day_halt = False
        self.n_done = {s: 0 for s in self.profile["symbols"]}
        self.last_exit = {s: None for s in self.profile["symbols"]}
        self.last_win = {s: True for s in self.profile["symbols"]}

    def _get_q(self, sym: str, date: str):
        k = (sym, date)
        if k not in self.quote_cache:
            self.quote_cache[k] = load_quotes(self.profile["_paths"]["quote_1s_root"], sym, date)
        return self.quote_cache[k]

    def _get_path(self, sym: str, date: str, ticker: str):
        trade = self.profile.get("trade") or {}
        quote_source = str(trade.get("quote_source", "1s")).lower()
        half = float(trade.get("day_iv_half_spread_frac", 0.01))
        if quote_source in ("1s", "auto"):
            path = path_for_ticker(self._get_q(sym, date), ticker)
            if path is not None and not path.empty:
                return path, "1s"
            if quote_source == "1s":
                return None, "none"
        if quote_source in ("day_iv", "auto") and self.books.chain_cache is not None:
            path = day_iv_path_as_quotes(
                self.books.chain_cache.get(sym, date), ticker, half_spread_frac=half
            )
            if path is not None and not path.empty:
                return path, "day_iv"
        return None, "none"

    def _topk(self) -> int:
        return int(self.profile["signal"].get("top_k", 2))

    def _try_enter(self, fire: dict[str, Any]) -> None:
        if self.day_halt:
            return
        trade = self.profile["trade"]
        sym = fire["symbol"]
        direction = fire["dir"]
        date = fire["date"]
        ts = to_ny(fire["sig_ts"])
        money = str(trade.get("moneyness", "ATM"))
        use_reentry = self.scheme.startswith("m5")
        max_n = int(trade.get("max_entries_per_symbol", 5)) if use_reentry else 1
        cooldown = int(trade.get("cooldown_minutes", 5))
        only_win = resolve_only_win_reenter(trade)

        if self.n_done.get(sym, 0) >= max_n:
            return
        if sym in self.positions:
            return
        if self.last_exit.get(sym) is not None and ts < self.last_exit[sym] + pd.Timedelta(minutes=cooldown):
            return
        if only_win and self.n_done.get(sym, 0) > 0 and not self.last_win.get(sym, True):
            return

        if self.regime_gate is not None:
            dec = self.regime_gate.check(direction, ts)
            if not dec.allow:
                self.n_regime_block += 1
                self.events.append({"type": "REGIME_BLOCK", **fire, "reason": dec.reason})
                return

        spot = float(fire["spot"]) if fire.get("spot") is not None else None
        pick = resolve_entry_contract(
            self.books,
            symbol=sym,
            date=date,
            direction=direction,
            moneyness=money,
            sig_ts=ts,
            spot=spot,
        )
        if pick.ticker is None:
            self.events.append({"type": "NO_CONTRACT", **fire, "source": pick.source})
            return
        if "skip0_clear_otm" in pick.source:
            self.n_skip0_clear_otm += 1

        path, qsrc = self._get_path(sym, date, pick.ticker)
        sdf = self.stock_by.get(sym)
        stock_day = None
        if sdf is not None and not getattr(sdf, "empty", True):
            stock_day = sdf[sdf["date"] == date]
        exit_mode = str(trade.get("exit_mode") or trade.get("stock_exit") or "none")
        sim = simulate_trade(
            path,
            ts,
            fill=self.fill,
            tp_mult=float(trade.get("tp_mult", 1.6)),
            sl_mult=float(trade.get("sl_mult", 0.4)),
            hold_minutes=int(trade.get("hold_minutes", 30)),
            direction=direction,
            stock_day=stock_day,
            exit_mode=exit_mode,
            exit_mf_grace_seconds=int(trade.get("exit_mf_grace_seconds", 60)),
        )
        if sim is None:
            return

        open_until = {s: t for s, t in self.last_exit.items() if t is not None}
        size_frac, sizing_mode, n_conc, allow, size_reason = resolve_size_frac(
            trade,
            top_k=self._topk(),
            open_until=open_until,
            symbol=sym,
            entry_ts=sim.entry_ts,
        )
        if not allow:
            self.events.append(
                {
                    "type": "SKIP_MAX_CONCURRENT",
                    **fire,
                    "n_concurrent": n_conc,
                    "reason": size_reason,
                }
            )
            return
        self.equity *= 1.0 + size_frac * sim.ret
        self.peak = max(self.peak, self.equity)
        self.maxdd = min(self.maxdd, self.equity / self.peak - 1.0)
        self.n_done[sym] = self.n_done.get(sym, 0) + 1
        self.last_exit[sym] = sim.exit_ts
        self.last_win[sym] = sim.ret > 0
        row = {
            "date": date,
            "symbol": sym,
            "dir": direction,
            "moneyness": money,
            "ticker": pick.ticker,
            "contract_source": pick.source,
            "quote_source": qsrc,
            "sig_spot": spot,
            "sig_strike": pick.strike,
            "sig_dte": pick.dte,
            "sig_ts": ts,
            "n_in_day": self.n_done[sym],
            "entry": sim.entry,
            "exit": sim.exit,
            "ret": sim.ret,
            "reason": sim.reason,
            "entry_ts": sim.entry_ts,
            "exit_ts": sim.exit_ts,
            "size_frac": size_frac,
            "n_concurrent": n_conc,
            "position_sizing": sizing_mode,
            "size_reason": size_reason,
            "source": "stream",
        }
        self.trades.append(row)
        self.events.append({"type": "TRADE", **row})
        circuit = trade.get("day_circuit") if "circuit" in self.scheme else None
        if circuit is not None and (self.equity / self.day_start_equity - 1.0) <= float(circuit):
            self.day_halt = True

    def on_stock_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        ts = to_ny(bar["timestamp"])
        date = ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        st = self.states.get(symbol)
        if st is None:
            return
        fire = st.on_bar(bar)
        if fire is None:
            return
        self.events.append({"type": "SIGNAL", **fire})
        accepted_syms = {f["symbol"] for f in self.day_fires}
        if fire["symbol"] in accepted_syms:
            if self.scheme.startswith("m5"):
                self._try_enter(fire)
            return
        if len(self.day_fires) < self._topk():
            self.day_fires.append(fire)
            self.events.append({"type": "TOPK_ACCEPT", **fire, "rank": len(self.day_fires)})
            self._try_enter(fire)
        else:
            self.events.append({"type": "TOPK_REJECT", **fire})

    def summary(self) -> dict[str, Any]:
        tdf = pd.DataFrame(self.trades)
        return {
            "scheme": self.scheme,
            "contract_mode": self.books.mode,
            "n_trades": int(len(tdf)),
            "total_ret": float(self.equity / 100.0 - 1.0),
            "maxdd": float(self.maxdd),
            "trade_win": float((tdf["ret"] > 0).mean()) if len(tdf) else float("nan"),
            "trade_exp": float(tdf["ret"].mean()) if len(tdf) else float("nan"),
            "end_equity": float(self.equity),
            "n_events": len(self.events),
            "n_regime_block": int(self.n_regime_block),
            "n_skip0_clear_otm": int(self.n_skip0_clear_otm),
        }


def run_stream_replay(profile: dict[str, Any], *, scheme: str = "single") -> dict[str, Any]:
    """Drive StreamEngine with chronological 1m bars across all symbols."""
    from maga7.common.replay import month_list
    from maga7.common.signals import attach_mf_features, load_stock_month_files

    paths = profile["_paths"]
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    eng = StreamEngine.from_profile(profile, scheme=scheme)
    sig = profile.get("signal") or {}

    frames = []
    stock_by: dict[str, Any] = {}
    for sym in profile["symbols"]:
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
        feat = attach_mf_features(
            raw,
            mf_window=int(sig.get("mf_window", 10)),
            vol_ma_window=int(sig.get("vol_ma_window", 20)),
        )
        stock_by[sym] = feat
        feat = feat.copy()
        feat["symbol"] = sym
        frames.append(feat[["symbol", "timestamp", "open", "high", "low", "close", "volume", "date"]])
    eng.stock_by = stock_by
    if not frames:
        return {"summary": eng.summary(), "trades": pd.DataFrame(), "events": eng.events}
    all_bars = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    for r in all_bars.itertuples(index=False):
        eng.on_stock_bar(
            r.symbol,
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            },
        )
    return {
        "summary": eng.summary(),
        "trades": pd.DataFrame(eng.trades),
        "events": eng.events,
        "daily_equity": eng.equity,
    }
