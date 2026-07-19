"""Streaming engine for Rule-A TopK scalp — causal bar/tick path for parity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from maga7.common.contract_select import day_iv_path_as_quotes
from maga7.common.entry_contract import ContractBooks, resolve_entry_contract
from maga7.common.fills import FillSpec
from maga7.common.position_size import (
    block_same_dir_after_win_enabled,
    is_symbol_dir_big_win,
    post_win_cooldown_action,
    post_win_cooldown_sessions,
    resolve_size_frac,
)
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
    prev_day_ret: float | None = None
    post_win_left: int = 0
    post_win_mode: str = ""
    post_win_scale: float = 1.0
    prev_big_win_dirs: set = field(default_factory=set)
    day_big_win_dirs: set = field(default_factory=set)
    n_same_dir_win_block: int = 0
    scheme: str = "single"
    regime_gate: Any = None
    n_regime_block: int = 0
    n_regime_scale: int = 0
    n_event_block: int = 0
    event_blackout: set = field(default_factory=set)
    n_peer_block: int = 0
    n_skip0_clear_otm: int = 0
    n_skip_max_entry_otm: int = 0
    n_confirm_block: int = 0
    stock_by: dict[str, Any] = field(default_factory=dict)
    # Full frames for Watchdog breadth / QQQ (may include refs beyond profile.symbols)
    stock_by_full: dict[str, Any] = field(default_factory=dict)
    watchdog: Any = None
    regime_snap: dict[str, Any] | None = None
    day_watchdog_state: str = "off"
    day_watchdog_reason: str = "off"
    n_watchdog_days: int = 0
    watchdog_state_counts: dict = field(default_factory=dict)
    loss_streak: int = 0
    n_mf_idio_block: int = 0
    n_mf_idio_scale: int = 0
    mf_idio_beta_cache: dict = field(default_factory=dict)
    pending_hunts: list = field(default_factory=list)
    day_hunt_symbols: set = field(default_factory=set)
    day_hunt_dirs: set = field(default_factory=set)
    n_hunt_signals: int = 0
    n_hunt_trades: int = 0
    n_hunt_budget_skip: int = 0
    n_hunt_mutex_skip: int = 0

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

    def _mf_idio_cfg(self) -> dict[str, Any]:
        sig = self.profile.get("signal") or {}
        mode = str(sig.get("mf_idio_mode") or "off").strip().lower()
        on = mode not in {"", "off", "none", "false", "0"}
        action = str(sig.get("mf_idio_action") or "block").strip().lower()
        if action not in {"block", "scale", "size", "half"}:
            action = "block"
        if action in {"size", "half"}:
            action = "scale"
        scale = max(0.0, min(float(sig.get("mf_idio_scale", 0.5) or 0.5), 1.0))
        als = sig.get("mf_idio_after_loss_streak")
        return {
            "on": on,
            "mode": mode,
            "action": action,
            "scale": scale,
            "min_frac": float(sig.get("mf_idio_min_frac", 0.0) or 0.0),
            "beta_days": int(sig.get("mf_idio_beta_days", 5) or 5),
            "beta_on": str(sig.get("mf_idio_beta_on") or "ret").strip().lower(),
            "block_missing": bool(sig.get("mf_idio_block_missing", False)),
            "after_loss_streak": int(als) if als is not None else None,
        }

    def _mf_idio_armed(self, cfg: dict[str, Any]) -> bool:
        if not cfg["on"]:
            return False
        als = cfg["after_loss_streak"]
        if als is None or int(als) <= 0:
            return True
        return int(self.loss_streak) >= int(als)

    def _mf_idio_fail(
        self,
        sym: str,
        direction: str,
        feature_ts: pd.Timestamp,
        date: str,
        cfg: dict[str, Any],
    ) -> bool:
        if not self._mf_idio_armed(cfg):
            return False
        from maga7.common.signals import mf_idio_ok, rolling_idio_beta

        full = self.stock_by_full or self.stock_by
        key = (str(sym).upper(), str(date))
        if key not in self.mf_idio_beta_cache and cfg["mode"] not in {
            "diff_pos",
            "diff",
            "mf_diff",
        }:
            self.mf_idio_beta_cache[key] = rolling_idio_beta(
                full.get(sym),
                full.get("QQQ"),
                asof_date=str(date),
                n_days=cfg["beta_days"],
                on=cfg["beta_on"],
            )
        ok, _meta = mf_idio_ok(
            full.get(sym),
            full.get("QQQ"),
            date=str(date),
            asof_ts=feature_ts,
            direction=str(direction),
            mode=cfg["mode"],
            min_frac=cfg["min_frac"],
            beta_days=cfg["beta_days"],
            beta_on=cfg["beta_on"],
            beta=self.mf_idio_beta_cache.get(key),
            block_missing=cfg["block_missing"],
        )
        return not ok

    def _roll_day(self, date: str) -> None:
        if self.current_date == date:
            return
        trade = self.profile.get("trade") or {}
        if self.current_date is not None and self.day_start_equity > 0:
            self.prev_day_ret = self.equity / self.day_start_equity - 1.0
            if self.prev_day_ret < 0:
                self.loss_streak += 1
            else:
                self.loss_streak = 0
            thr = trade.get("post_win_cooldown_day_ret", 0.10)
            mode = str(trade.get("post_win_cooldown_mode") or "off").lower()
            if (
                mode not in {"", "off", "none", "false", "0"}
                and thr is not None
                and self.prev_day_ret >= float(thr)
            ):
                self.post_win_left = post_win_cooldown_sessions(trade)
            elif self.post_win_left > 0:
                self.post_win_left -= 1
            self.prev_big_win_dirs = set(self.day_big_win_dirs)
        self.current_date = date
        self.day_fires = []
        self.pending_fires = []
        self.positions = {}
        self.day_start_equity = self.equity
        self.day_halt = False
        self.day_big_win_dirs = set()
        self.post_win_mode, self.post_win_scale = post_win_cooldown_action(
            trade,
            prev_day_ret=self.prev_day_ret,
            cooldown_left=self.post_win_left,
        )
        if self.post_win_mode == "skip":
            self.day_halt = True
        if not self.event_blackout:
            self.event_blackout = self._build_event_blackout()
        if date in self.event_blackout:
            self.day_halt = True
            self.n_event_block += 1
        self.n_done = {s: 0 for s in self.profile["symbols"]}
        self.last_exit = {s: None for s in self.profile["symbols"]}
        self.last_win = {s: True for s in self.profile["symbols"]}
        self.pending_hunts = []
        self.day_hunt_symbols = set()
        self.day_hunt_dirs = set()
        # Watchdog Degrade/Halt + schedule Hunt candidates (mirror offline)
        self.day_watchdog_state = "off"
        self.day_watchdog_reason = "off"
        if self.watchdog is not None and self.regime_gate is not None and self.regime_snap is not None:
            full = self.stock_by_full or self.stock_by
            wd_dec = self.watchdog.begin_day(
                str(date),
                stock_by=full,
                qqq_df=full.get("QQQ"),
                symbols=list(self.profile.get("symbols") or []),
            )
            self.watchdog.apply_to_regime(self.regime_gate.cfg, self.regime_snap)
            self.day_watchdog_state = wd_dec.state.value
            self.day_watchdog_reason = wd_dec.reason
            if wd_dec.state.value != "normal":
                self.n_watchdog_days += 1
                st = self.day_watchdog_state
                self.watchdog_state_counts[st] = int(self.watchdog_state_counts.get(st, 0)) + 1
            self._schedule_hunts(str(date))

    def _schedule_hunts(self, date: str) -> None:
        """Queue Hunt fires at feature_ts + bar_delay (same clock as offline)."""
        if self.day_halt or self.watchdog is None:
            return
        if not bool(getattr(self.watchdog, "hunt_armed", False)):
            return
        trade = self.profile.get("trade") or {}
        bar_delay = int(trade.get("bar_availability_delay_seconds", 0) or 0)
        for hc in list(getattr(self.watchdog, "hunt_candidates", None) or []):
            self.n_hunt_signals += 1
            feature_ts = to_ny(hc.sig_ts)
            entry_ts = feature_ts + pd.Timedelta(seconds=bar_delay)
            if entry_ts > to_ny(hc.armed_until):
                continue
            self.pending_hunts.append(
                {
                    "entry_ts": entry_ts,
                    "feature_ts": feature_ts,
                    "symbol": str(hc.symbol),
                    "dir": str(hc.direction),
                    "date": date,
                    "event_source": "hunt",
                    "detector": str(getattr(hc, "detector", "") or ""),
                }
            )
        self.pending_hunts.sort(key=lambda x: (x["entry_ts"], x["symbol"]))

    def _spot_at(self, sym: str, date: str, feature_ts: pd.Timestamp) -> float | None:
        full = self.stock_by_full or self.stock_by
        sdf = full.get(sym)
        if sdf is None or getattr(sdf, "empty", True):
            return None
        day = sdf[sdf["date"].astype(str) == str(date)]
        if day.empty:
            return None
        ts = pd.to_datetime(day["timestamp"])
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("America/New_York")
        else:
            ts = ts.dt.tz_convert("America/New_York")
        day = day.copy()
        day["_ts"] = ts
        upto = day[day["_ts"] <= feature_ts]
        if upto.empty:
            return None
        try:
            return float(upto.iloc[-1]["close"])
        except Exception:
            return None

    def _drain_hunts(self, ts: pd.Timestamp) -> None:
        if not self.pending_hunts or self.day_halt:
            return
        due = [h for h in self.pending_hunts if h["entry_ts"] <= ts]
        if not due:
            return
        self.pending_hunts = [h for h in self.pending_hunts if h["entry_ts"] > ts]
        for h in due:
            fire = {
                "symbol": h["symbol"],
                "dir": h["dir"],
                "date": h["date"],
                "sig_ts": h["feature_ts"],
                "spot": self._spot_at(h["symbol"], h["date"], h["feature_ts"]),
                "event_source": "hunt",
                "route": "hunt",
            }
            self.events.append({"type": "HUNT_SIGNAL", **fire, "entry_ts": h["entry_ts"]})
            self._try_enter(fire)

    def _build_event_blackout(self) -> set:
        from maga7.common.event_calendar import resolve_event_blackout

        trade = self.profile.get("trade") or {}
        reg = self.profile.get("regime") or {}
        cfg = {
            **{k: reg[k] for k in (
                "event_calendar_block", "event_calendar", "event_dates", "event_blackout_sessions",
            ) if k in reg},
            **{k: trade[k] for k in (
                "event_calendar_block", "event_calendar", "event_dates", "event_blackout_sessions",
            ) if k in trade},
        }
        sessions: list[str] = []
        for sdf in self.stock_by.values():
            if sdf is not None and not getattr(sdf, "empty", True) and "date" in sdf.columns:
                sessions.extend(str(x) for x in sdf["date"].unique())
        if not sessions:
            start = str(self.profile.get("date_range", {}).get("start", ""))
            end = str(self.profile.get("date_range", {}).get("end", ""))
            if start and end:
                sessions = [
                    d.strftime("%Y-%m-%d")
                    for d in pd.bdate_range(start, end)
                ]
        return resolve_event_blackout(cfg, session_dates=sessions)

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
        is_hunt = str(fire.get("event_source") or fire.get("route") or "baseline") == "hunt"
        wd = self.watchdog
        wd_cfg = getattr(wd, "cfg", None) if wd is not None else None

        if is_hunt and wd is not None and wd.hunt_budget_remaining() <= 0:
            self.n_hunt_budget_skip += 1
            self.events.append({"type": "HUNT_BUDGET_SKIP", **fire})
            return

        # Mutex: prior Hunt blocks same-symbol (or same dir) baseline unless opp allowed.
        if (
            (not is_hunt)
            and wd_cfg is not None
            and bool(getattr(wd_cfg, "hunter_mutex_with_baseline", False))
        ):
            scope = str(getattr(wd_cfg, "hunter_mutex_scope", "symbol") or "symbol").lower()
            if scope in {"symbol_dir", "dir", "same_dir"}:
                mutex_hit = (str(sym), str(direction).upper()) in self.day_hunt_dirs
            else:
                mutex_hit = str(sym) in self.day_hunt_symbols
            if mutex_hit:
                self.n_hunt_mutex_skip += 1
                self.events.append({"type": "HUNT_MUTEX_SKIP", **fire})
                return

        if block_same_dir_after_win_enabled(trade) and (
            str(sym).upper(),
            str(direction).upper(),
        ) in self.prev_big_win_dirs:
            self.n_same_dir_win_block += 1
            self.events.append(
                {
                    "type": "SAME_DIR_WIN_BLOCK",
                    **fire,
                    "blocked": f"{sym}:{direction}",
                }
            )
            return
        feature_ts = to_ny(fire["sig_ts"])
        bar_delay_seconds = int(
            trade.get("bar_availability_delay_seconds", 0) or 0
        )
        ts = feature_ts + pd.Timedelta(seconds=bar_delay_seconds)
        money = str(trade.get("moneyness", "ATM"))
        use_reentry = self.scheme.startswith("m5")
        max_n = int(trade.get("max_entries_per_symbol", 5)) if use_reentry else 1
        cooldown = int(trade.get("cooldown_minutes", 5))
        only_win = resolve_only_win_reenter(trade)

        if self.n_done.get(sym, 0) >= max_n:
            allow_opp = (
                (not is_hunt)
                and wd_cfg is not None
                and bool(getattr(wd_cfg, "hunter_allow_baseline_opposite", False))
                and str(sym) in self.day_hunt_symbols
                and (str(sym), str(direction).upper()) not in self.day_hunt_dirs
                and any(s == str(sym) and d != str(direction).upper() for s, d in self.day_hunt_dirs)
            )
            if not allow_opp:
                return
        if sym in self.positions:
            return
        if self.last_exit.get(sym) is not None and ts < self.last_exit[sym] + pd.Timedelta(minutes=cooldown):
            return
        if only_win and self.n_done.get(sym, 0) > 0 and not self.last_win.get(sym, True):
            return

        regime_scale = 1.0
        skip_qqq = bool(
            is_hunt and wd_cfg is not None and getattr(wd_cfg, "hunter_skip_qqq_align", False)
        )
        if self.regime_gate is not None and not skip_qqq:
            dec = self.regime_gate.check(direction, feature_ts)
            if not dec.allow:
                self.n_regime_block += 1
                self.events.append({"type": "REGIME_BLOCK", **fire, "reason": dec.reason})
                return
            regime_scale = float(getattr(dec, "size_scale", 1.0) or 1.0)

        sig_cfg = self.profile.get("signal") or {}
        peer_min = sig_cfg.get("peer_align_min")
        skip_peer = bool(
            is_hunt and wd_cfg is not None and getattr(wd_cfg, "hunter_skip_peer", False)
        )
        if (not skip_peer) and peer_min is not None and int(peer_min) > 0 and self.stock_by:
            from maga7.common.signals import count_peer_align

            peers = list(sig_cfg.get("peer_symbols") or self.profile.get("symbols") or [])
            peer_n = count_peer_align(
                self.stock_by,
                date=str(date),
                asof_ts=feature_ts,
                direction=str(direction),
                peer_symbols=peers,
                mode=str(sig_cfg.get("peer_align_mode", "mf10")),
                streak_min=int(sig_cfg.get("streak_min", 8)),
            )
            if peer_n < int(peer_min):
                self.n_peer_block += 1
                self.events.append(
                    {"type": "PEER_BLOCK", **fire, "peer_align_n": peer_n, "peer_align_min": int(peer_min)}
                )
                return

        idio_cfg = self._mf_idio_cfg()
        # Hunt skips mf_idio entry block (same as offline); scale still applies below.
        if (not is_hunt) and idio_cfg["on"] and idio_cfg["action"] == "block":
            if self._mf_idio_fail(sym, direction, feature_ts, str(date), idio_cfg):
                self.n_mf_idio_block += 1
                self.events.append({"type": "MF_IDIO_BLOCK", **fire})
                return

        # Entry confirm (optional; may be weekday-gated — parity with offline replay).
        confirm_bars_raw = trade.get("entry_confirm_bars") or (self.profile.get("signal") or {}).get(
            "entry_confirm_bars"
        )
        confirm_bars_n = int(confirm_bars_raw) if confirm_bars_raw is not None else 0
        confirm_mode = str(
            trade.get("entry_confirm_mode")
            or (self.profile.get("signal") or {}).get("entry_confirm_mode")
            or "mf"
        ).strip().lower()
        confirm_wd_raw = trade.get("entry_confirm_weekdays") or (self.profile.get("signal") or {}).get(
            "entry_confirm_weekdays"
        )
        use_confirm = confirm_bars_n > 0
        if use_confirm and confirm_wd_raw is not None:
            if isinstance(confirm_wd_raw, str):
                confirm_wds = {
                    int(x.strip()) for x in confirm_wd_raw.split(",") if str(x).strip() != ""
                }
            else:
                confirm_wds = {int(x) for x in confirm_wd_raw}
            try:
                wd0 = int(pd.Timestamp(str(date)).weekday())
            except Exception:
                wd0 = -1
            use_confirm = wd0 in confirm_wds
        if use_confirm:
            from maga7.common.replay import entry_confirm_ok

            sdf_c = self.stock_by.get(sym)
            stock_day_c = None
            if sdf_c is not None and not getattr(sdf_c, "empty", True):
                stock_day_c = sdf_c[sdf_c["date"] == date]
            ok_c, confirm_ft, _, _, _ = entry_confirm_ok(
                stock_day_c,
                direction=str(direction),
                feature_ts=feature_ts,
                confirm_bars=confirm_bars_n,
                mode=confirm_mode,
            )
            if not ok_c:
                self.n_confirm_block += 1
                self.events.append({"type": "ENTRY_CONFIRM_BLOCK", **fire})
                return
            ts = to_ny(confirm_ft) + pd.Timedelta(seconds=bar_delay_seconds)

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
            if pick.source == "skip_max_entry_otm":
                self.n_skip_max_entry_otm += 1
                self.events.append({"type": "MAX_OTM_SKIP", **fire, "source": pick.source})
            else:
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
        sim_kw: dict[str, Any] = {
            "fill": self.fill,
            "tp_mult": float(trade.get("tp_mult", 1.6)),
            "sl_mult": float(trade.get("sl_mult", 0.4)),
            "hold_minutes": int(trade.get("hold_minutes", 30)),
            "exit_mf_grace_seconds": int(trade.get("exit_mf_grace_seconds", 60)),
            "exit_min_hold_minutes": trade.get("exit_min_hold_minutes"),
            "mtm_floor_ret": trade.get("mtm_floor_ret"),
            "flow_cum_floor": trade.get("flow_cum_floor"),
            "stock_bar_delay_seconds": bar_delay_seconds,
            "trail_activate": trade.get("trail_activate"),
            "trail_dd": trade.get("trail_dd"),
            "hold_extend_minutes": trade.get("hold_extend_minutes"),
            "hold_extend_mtm_min": trade.get("hold_extend_mtm_min"),
            "hold_extend_require_mf": bool(trade.get("hold_extend_require_mf", True)),
            "early_exit_mode": trade.get("early_exit_mode"),
            "mae_cut_ret": trade.get("mae_cut_ret"),
            "mae_cut_mfe_bypass": trade.get("mae_cut_mfe_bypass"),
            "mae_cut_min_hold_minutes": trade.get("mae_cut_min_hold_minutes"),
            "dyn_max_hold_minutes": trade.get("dyn_max_hold_minutes"),
            "dyn_min_hold_minutes": trade.get("dyn_min_hold_minutes"),
            "dyn_trail_start_minutes": trade.get("dyn_trail_start_minutes"),
            "dyn_slope_lookback": trade.get("dyn_slope_lookback"),
            "dyn_fast_opp_bars": trade.get("dyn_fast_opp_bars"),
            "dyn_fast_pct": trade.get("dyn_fast_pct"),
            "dyn_require_price_break": bool(trade.get("dyn_require_price_break", True)),
        }
        from maga7.common.hold_watchdog import hold_watchdog_from_trade

        hwd = hold_watchdog_from_trade(trade)
        sim_kw["hold_watchdog"] = hwd
        qqq_day = None
        if hwd.enabled:
            qdf = self.stock_by.get("QQQ") or self.stock_by_full.get("QQQ")
            if qdf is not None and not getattr(qdf, "empty", True):
                qqq_day = qdf[qdf["date"].astype(str) == str(date)]
        hunt_pos_frac = None
        if is_hunt and wd is not None:
            from maga7.common.watchdog import hunt_trade_overrides

            hov = hunt_trade_overrides(wd.cfg)
            hunt_pos_frac = hov.pop("position_frac", None)
            if "exit_mode" in hov:
                exit_mode = str(hov.pop("exit_mode"))
            sim_kw.update(hov)
        sim = simulate_trade(
            path,
            ts,
            direction=direction,
            stock_day=stock_day,
            exit_mode=exit_mode,
            qqq_day=qqq_day,
            **sim_kw,
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
        if hunt_pos_frac is not None and float(hunt_pos_frac) > 0:
            size_frac = float(hunt_pos_frac)
            size_reason = f"{size_reason}+hunt_frac:{float(hunt_pos_frac):.2f}"
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
        if self.post_win_mode == "scale" and self.post_win_scale < 1.0:
            size_frac = float(size_frac) * float(self.post_win_scale)
            size_reason = f"{size_reason}+post_win_scale"
        if regime_scale < 1.0:
            size_frac = float(size_frac) * float(regime_scale)
            size_reason = f"{size_reason}+regime_scale"
            self.n_regime_scale += 1
        if idio_cfg["on"] and idio_cfg["action"] == "scale":
            if self._mf_idio_fail(sym, direction, feature_ts, str(date), idio_cfg):
                idio_mult = float(idio_cfg["scale"])
                size_frac = float(size_frac) * idio_mult
                size_reason = f"{size_reason}+mf_idio_scale:{idio_mult:.2f}"
                self.n_mf_idio_scale += 1
        self.equity *= 1.0 + size_frac * sim.ret
        self.peak = max(self.peak, self.equity)
        self.maxdd = min(self.maxdd, self.equity / self.peak - 1.0)
        self.n_done[sym] = self.n_done.get(sym, 0) + 1
        self.last_exit[sym] = sim.exit_ts
        self.last_win[sym] = sim.ret > 0
        if is_hunt and wd is not None:
            wd.note_hunt_entry()
            self.day_hunt_symbols.add(str(sym))
            self.day_hunt_dirs.add((str(sym), str(direction).upper()))
            self.n_hunt_trades += 1
            h_circ = getattr(wd.cfg, "hunter_day_circuit_ret", None)
            if h_circ is not None and float(sim.ret) <= float(h_circ):
                self.day_halt = True
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
            "route": "hunt" if is_hunt else "baseline",
            "event_source": "hunt" if is_hunt else "baseline",
            "watchdog_state": self.day_watchdog_state,
            "watchdog_reason": self.day_watchdog_reason,
        }
        self.trades.append(row)
        self.events.append({"type": "TRADE", **row})
        if block_same_dir_after_win_enabled(trade) and is_symbol_dir_big_win(
            ret=float(sim.ret), reason=sim.reason, trade=trade
        ):
            self.day_big_win_dirs.add((str(sym).upper(), str(direction).upper()))
        circuit_raw = trade.get("day_circuit", None)
        if circuit_raw is not None and circuit_raw is not False:
            if (self.equity / self.day_start_equity - 1.0) <= float(circuit_raw):
                self.day_halt = True

    def on_stock_bar(self, symbol: str, bar: dict[str, Any]) -> None:
        ts = to_ny(bar["timestamp"])
        date = ts.strftime("%Y-%m-%d")
        self._roll_day(date)
        # Hunt fires are time-driven (not TopK); drain before Rule-A at this clock.
        self._drain_hunts(ts)
        st = self.states.get(symbol)
        if st is None:
            return
        fire = st.on_bar(bar)
        if fire is None:
            return
        fire = {**fire, "event_source": "baseline", "route": "baseline"}
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
            "n_regime_scale": int(self.n_regime_scale),
            "n_event_block": int(self.n_event_block),
            "n_skip0_clear_otm": int(self.n_skip0_clear_otm),
            "n_skip_max_entry_otm": int(self.n_skip_max_entry_otm),
            "n_confirm_block": int(self.n_confirm_block),
            "n_mf_idio_block": int(self.n_mf_idio_block),
            "n_mf_idio_scale": int(self.n_mf_idio_scale),
            "watchdog_enabled": bool(self.watchdog is not None),
            "n_watchdog_days": int(self.n_watchdog_days),
            "watchdog_state_counts": dict(self.watchdog_state_counts),
            "n_hunt_signals": int(self.n_hunt_signals),
            "n_hunt_trades": int(self.n_hunt_trades),
            "n_hunt_budget_skip": int(self.n_hunt_budget_skip),
            "n_hunt_mutex_skip": int(self.n_hunt_mutex_skip),
        }


def run_stream_replay(
    profile: dict[str, Any],
    *,
    scheme: str = "single",
    stock_by: dict[str, Any] | None = None,
    regime_gate=None,
) -> dict[str, Any]:
    """Drive StreamEngine with chronological 1m bars across all symbols.

    ``stock_by`` / ``regime_gate``: optional prebuilt frames (e.g. from stock 1s).
    When omitted, loads pre-aggregated ``stock_root`` month files (research cache).
    """
    from maga7.common.replay import month_list
    from maga7.common.signals import (
        attach_mf_features,
        load_stock_month_files,
        resolve_mf_fast_window,
    )

    paths = profile["_paths"]
    start = profile["date_range"]["start"]
    end = profile["date_range"]["end"]
    months = month_list(start, end)
    eng = StreamEngine.from_profile(profile, scheme=scheme)
    if regime_gate is not None:
        eng.regime_gate = regime_gate
    sig = profile.get("signal") or {}

    frames = []
    if stock_by is None:
        stock_by = {}
        for sym in profile["symbols"]:
            raw = load_stock_month_files(paths["stock_root"], sym, months)
            if raw.empty:
                continue
            raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
            feat = attach_mf_features(
                raw,
                mf_window=int(sig.get("mf_window", 10)),
                vol_ma_window=int(sig.get("vol_ma_window", 20)),
                mf_fast_window=resolve_mf_fast_window(sig),
            )
            stock_by[sym] = feat
    for sym, feat in stock_by.items():
        if sym not in profile["symbols"]:
            continue
        if feat is None or feat.empty:
            continue
        f = feat.copy()
        f["symbol"] = sym
        keep = [c for c in ("symbol", "timestamp", "open", "high", "low", "close", "volume", "date") if c in f.columns]
        frames.append(f[keep])
    eng.stock_by = {s: stock_by[s] for s in profile["symbols"] if s in stock_by}
    eng.stock_by_full = dict(stock_by)
    # Watchdog: Degrade/Halt overlay + Hunt schedule (drained on bar clock).
    try:
        from maga7.common.watchdog import RegimeWatchdog, snapshot_regime

        eng.watchdog = RegimeWatchdog.from_profile(profile)
        if eng.watchdog is not None and eng.regime_gate is not None:
            eng.regime_snap = snapshot_regime(eng.regime_gate.cfg)
    except Exception:
        eng.watchdog = None
        eng.regime_snap = None
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
    # Flush any Hunt still pending after last bar (should be rare).
    if eng.pending_hunts:
        last_ts = to_ny(all_bars["timestamp"].iloc[-1])
        eng._drain_hunts(last_ts + pd.Timedelta(days=1))
    return {
        "summary": eng.summary(),
        "trades": pd.DataFrame(eng.trades),
        "events": eng.events,
        "daily_equity": eng.equity,
    }
