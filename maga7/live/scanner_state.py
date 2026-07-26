"""Compact, restart-safe state for live Scanner and minute aggregators."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from maga7.common.replay import to_ny
from maga7.live.scanner import ScannerSignal

# Full RTH day (~390 1m bars) + small buffer for Watchdog/Hunt after resume.
STOCK_BY_KEEP = 450
STOCK_BY_COLS = (
    "timestamp",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "net$",
    "mf10",
    "mf_fast",
    "mf_short",
    "streak_up",
    "streak_dn",
    "prev_close",
    "from_prev",
    "cum",
    "vol_ma",
    "vol_z",
)


def _signal_to_dict(signal: ScannerSignal) -> dict[str, Any]:
    return {
        **signal.__dict__,
        "sig_ts": to_ny(signal.sig_ts).isoformat(),
    }


def _signal_from_dict(payload: dict[str, Any]) -> ScannerSignal:
    return ScannerSignal(**{**payload, "sig_ts": to_ny(payload["sig_ts"])})


def _json_num(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        # numpy / pandas scalars
        if hasattr(value, "item"):
            value = value.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
    except Exception:
        pass
    return value


def _stock_by_to_dict(stock_by: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(stock_by, dict):
        return out
    for symbol, frame in stock_by.items():
        if frame is None or getattr(frame, "empty", True):
            continue
        cols = [c for c in STOCK_BY_COLS if c in frame.columns]
        if not cols:
            continue
        tail = frame[cols].tail(STOCK_BY_KEEP)
        rows: list[dict[str, Any]] = []
        for raw in tail.to_dict("records"):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key == "timestamp" and value is not None:
                    row[key] = to_ny(value).isoformat()
                else:
                    row[key] = _json_num(value)
            rows.append(row)
        if rows:
            out[str(symbol)] = rows
    return out


def _stock_by_from_dict(payload: dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    restored: dict[str, pd.DataFrame] = {}
    if not isinstance(payload, dict):
        return restored
    for symbol, rows in payload.items():
        if not rows:
            continue
        frame = pd.DataFrame(list(rows))
        if "timestamp" in frame.columns:
            frame["timestamp"] = frame["timestamp"].map(to_ny)
        restored[str(symbol)] = frame
    return restored


def scanner_snapshot(scanner: Any) -> dict[str, Any]:
    states = {}
    for symbol, state in scanner.states.items():
        keep = max(int(state.mf_window), int(state.vol_ma_window), 20)
        bars = []
        for bar in state.bars[-keep:]:
            bars.append(
                {
                    **bar,
                    "timestamp": to_ny(bar["timestamp"]).isoformat(),
                }
            )
        first_fire = state.first_fire
        if isinstance(first_fire, dict) and first_fire.get("sig_ts") is not None:
            first_fire = {
                **first_fire,
                "sig_ts": to_ny(first_fire["sig_ts"]).isoformat(),
            }
        states[symbol] = {
            "bars": bars,
            "prev_close": state.prev_close,
            "day_open": state.day_open,
            "date": state.date,
            "cum": state.cum,
            "mf10": state.mf10,
            "streak_up": state.streak_up,
            "streak_dn": state.streak_dn,
            "fired_today": state.fired_today,
            "first_fire": first_fire,
        }
    aggs = {}
    if scanner.minute_agg is not None:
        for symbol, agg in scanner.minute_agg.aggs.items():
            aggs[symbol] = {
                "cur_minute": (
                    to_ny(agg.cur_minute).isoformat()
                    if agg.cur_minute is not None
                    else None
                ),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
            }
    regime = None
    gate = getattr(scanner, "regime_gate", None)
    if gate is not None and hasattr(gate, "qqq_state"):
        qqq = gate.qqq_state
        regime_aggs = {}
        for symbol, agg in gate.agg.aggs.items():
            regime_aggs[symbol] = {
                "cur_minute": (
                    to_ny(agg.cur_minute).isoformat()
                    if agg.cur_minute is not None
                    else None
                ),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
            }
        regime = {
            "qqq_previous_close": gate.qqq_previous_close,
            "qqq_close": gate.qqq_close,
            "vixy_closes": list(gate.vixy_closes),
            "qqq_state": {
                "bars": [
                    {**bar, "timestamp": to_ny(bar["timestamp"]).isoformat()}
                    for bar in qqq.bars[-20:]
                ],
                "prev_close": qqq.prev_close,
                "day_open": qqq.day_open,
                "date": qqq.date,
                "cum": qqq.cum,
                "mf10": qqq.mf10,
                "streak_up": qqq.streak_up,
                "streak_dn": qqq.streak_dn,
                "fired_today": qqq.fired_today,
            },
            "minute_aggs": regime_aggs,
        }
    prevention = None
    wd = getattr(scanner, "watchdog", None)
    if wd is not None:
        try:
            from maga7.common.predictive_prevention import prevention_from_watchdog

            date = str(
                getattr(scanner, "current_date", None)
                or getattr(wd, "_current_date", None)
                or ""
            )
            prevention = prevention_from_watchdog(wd, date=date).to_dict()
            prevention["watchdog_state"] = getattr(scanner, "_watchdog_state", None)
            prevention["watchdog_reason"] = getattr(scanner, "_watchdog_reason", None)
            prevention["watchdog_route"] = getattr(scanner, "_watchdog_route", None)
        except Exception:
            prevention = None

    pending = list(getattr(scanner, "pending_hunts", None) or [])
    hunt_cands = list(getattr(wd, "hunt_candidates", None) or []) if wd is not None else []
    watchdog_snap = {
        "state": getattr(scanner, "_watchdog_state", "off"),
        "reason": getattr(scanner, "_watchdog_reason", "off"),
        "route": getattr(scanner, "_watchdog_route", "baseline"),
        "closed": bool(getattr(scanner, "_watchdog_closed", False)),
        "day_halt": bool(getattr(scanner, "_day_halt", False)),
        "hunt_armed": bool(getattr(wd, "hunt_armed", False)) if wd is not None else False,
        "hunt_budget_remaining": (
            int(wd.hunt_budget_remaining())
            if wd is not None and hasattr(wd, "hunt_budget_remaining")
            else None
        ),
        "n_hunt_candidates": int(len(hunt_cands)),
        "n_hunt_signals": int(getattr(scanner, "n_hunt_signals", 0) or 0),
        "n_hunt_emitted": int(getattr(scanner, "n_hunt_emitted", 0) or 0),
        "n_hunt_budget_skip": int(getattr(scanner, "n_hunt_budget_skip", 0) or 0),
        "n_hunt_mutex_skip": int(getattr(scanner, "n_hunt_mutex_skip", 0) or 0),
        "pending_hunts": int(len(pending)),
        "day_hunt_symbols": sorted(str(s) for s in (getattr(scanner, "day_hunt_symbols", None) or set())),
        "candidates": [
            {
                "symbol": str(getattr(hc, "symbol", "")),
                "direction": str(getattr(hc, "direction", "")),
                "detector": str(getattr(hc, "detector", "") or ""),
                "sig_ts": to_ny(hc.sig_ts).isoformat() if getattr(hc, "sig_ts", None) is not None else None,
                "armed_until": (
                    to_ny(hc.armed_until).isoformat()
                    if getattr(hc, "armed_until", None) is not None
                    else None
                ),
            }
            for hc in hunt_cands[:8]
        ],
        "pending": [
            {
                "symbol": str(h.get("symbol")),
                "dir": str(h.get("dir")),
                "entry_ts": str(h.get("entry_ts")),
                "detector": str(h.get("detector") or ""),
            }
            for h in pending[:8]
        ],
    }

    return {
        "schema_version": 1,
        "current_date": scanner.current_date,
        "states": states,
        "minute_aggs": aggs,
        # Watchdog/Hunt morning path: day 1m bars + mf features.
        "stock_by": _stock_by_to_dict(getattr(scanner, "stock_by", None)),
        "day_fires": [_signal_to_dict(signal) for signal in scanner.day_fires],
        "signals": [_signal_to_dict(signal) for signal in scanner.signals],
        "n_done": scanner.n_done,
        "last_exit": {
            symbol: to_ny(value).isoformat() if value is not None else None
            for symbol, value in scanner.last_exit.items()
        },
        "last_win": scanner.last_win,
        "regime": regime,
        "prevention": prevention,
        "watchdog": watchdog_snap,
        "qqq_open_cont": {
            "enabled": bool(
                ((getattr(scanner, "profile", None) or {}).get("qqq_open_cont") or {}).get(
                    "enabled", False
                )
            ),
            "done_date": getattr(scanner, "_open_cont_done_date", None),
            "n_signals": int(getattr(scanner, "n_open_cont_signals", 0) or 0),
            "n_emitted": int(getattr(scanner, "n_open_cont_emitted", 0) or 0),
            "n_skip": int(getattr(scanner, "n_open_cont_skip", 0) or 0),
            "qqq_rth_open": getattr(scanner, "_qqq_rth_open", None),
            "qqq_last_px": getattr(scanner, "_qqq_last_px", None),
        },
        "am_pulse": {
            "enabled": bool(
                ((getattr(scanner, "profile", None) or {}).get("am_pulse") or {}).get(
                    "enabled", False
                )
            ),
            "execute_mode": (
                ((getattr(scanner, "profile", None) or {}).get("am_pulse") or {}).get(
                    "execute_mode", "shadow"
                )
            ),
            "n_signals": int(getattr(scanner, "n_am_pulse_signals", 0) or 0),
            "n_emitted": int(getattr(scanner, "n_am_pulse_emitted", 0) or 0),
            "n_skip": int(getattr(scanner, "n_am_pulse_skip", 0) or 0),
            "n_shadow": int(getattr(scanner, "n_am_pulse_shadow", 0) or 0),
            "pending": int(len(getattr(scanner, "pending_am_pulse", []) or [])),
        },
    }


def restore_scanner(scanner: Any, payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version") or 0) != 1:
        raise RuntimeError("unsupported scanner state schema")
    for symbol, saved in (payload.get("states") or {}).items():
        state = scanner.states.get(symbol)
        if state is None:
            continue
        state.bars = [
            {**bar, "timestamp": to_ny(bar["timestamp"])}
            for bar in (saved.get("bars") or [])
        ]
        state.prev_close = saved.get("prev_close")
        state.day_open = saved.get("day_open")
        state.date = saved.get("date")
        state.cum = float(saved.get("cum") or 0.0)
        state.mf10 = float(saved.get("mf10", float("nan")))
        state.streak_up = int(saved.get("streak_up") or 0)
        state.streak_dn = int(saved.get("streak_dn") or 0)
        state.fired_today = bool(saved.get("fired_today", False))
        first_fire = saved.get("first_fire")
        if isinstance(first_fire, dict) and first_fire.get("sig_ts") is not None:
            first_fire = {
                **first_fire,
                "sig_ts": to_ny(first_fire["sig_ts"]),
            }
        state.first_fire = first_fire
    if scanner.minute_agg is not None:
        for symbol, saved in (payload.get("minute_aggs") or {}).items():
            agg = scanner.minute_agg.aggs.get(symbol)
            if agg is None:
                continue
            raw_minute = saved.get("cur_minute")
            agg.cur_minute = to_ny(raw_minute) if raw_minute else None
            agg.open = saved.get("open")
            agg.high = saved.get("high")
            agg.low = saved.get("low")
            agg.close = saved.get("close")
            agg.volume = float(saved.get("volume") or 0.0)
    scanner.current_date = payload.get("current_date")
    scanner.day_fires = [
        _signal_from_dict(value) for value in (payload.get("day_fires") or [])
    ]
    scanner.signals = [
        _signal_from_dict(value) for value in (payload.get("signals") or [])
    ]
    scanner.n_done = {
        str(key): int(value) for key, value in (payload.get("n_done") or {}).items()
    }
    scanner.last_exit = {
        str(key): to_ny(value) if value else None
        for key, value in (payload.get("last_exit") or {}).items()
    }
    scanner.last_win = {
        str(key): bool(value)
        for key, value in (payload.get("last_win") or {}).items()
    }
    # Optional for older snapshots; empty dict means "not yet accumulated".
    if "stock_by" in payload:
        restored_stock = _stock_by_from_dict(payload.get("stock_by"))
        if getattr(scanner, "stock_by_frozen", False):
            pass
        else:
            scanner.stock_by = restored_stock
    regime = payload.get("regime")
    gate = getattr(scanner, "regime_gate", None)
    if isinstance(regime, dict) and gate is not None and hasattr(gate, "qqq_state"):
        gate.qqq_previous_close = float(regime.get("qqq_previous_close") or 0.0)
        gate.qqq_close = float(regime.get("qqq_close") or 0.0)
        gate.vixy_closes.clear()
        gate.vixy_closes.extend(regime.get("vixy_closes") or [])
        qqq_saved = regime.get("qqq_state") or {}
        qqq = gate.qqq_state
        qqq.bars = [
            {**bar, "timestamp": to_ny(bar["timestamp"])}
            for bar in (qqq_saved.get("bars") or [])
        ]
        qqq.prev_close = qqq_saved.get("prev_close")
        qqq.day_open = qqq_saved.get("day_open")
        qqq.date = qqq_saved.get("date")
        qqq.cum = float(qqq_saved.get("cum") or 0.0)
        qqq.mf10 = float(qqq_saved.get("mf10", float("nan")))
        qqq.streak_up = int(qqq_saved.get("streak_up") or 0)
        qqq.streak_dn = int(qqq_saved.get("streak_dn") or 0)
        qqq.fired_today = bool(qqq_saved.get("fired_today", False))
        for symbol, saved in (regime.get("minute_aggs") or {}).items():
            agg = gate.agg.aggs.get(symbol)
            if agg is None:
                continue
            raw_minute = saved.get("cur_minute")
            agg.cur_minute = to_ny(raw_minute) if raw_minute else None
            agg.open = saved.get("open")
            agg.high = saved.get("high")
            agg.low = saved.get("low")
            agg.close = saved.get("close")
            agg.volume = float(saved.get("volume") or 0.0)


def write_scanner_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
