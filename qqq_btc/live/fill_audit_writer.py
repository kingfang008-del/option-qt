#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实盘 fill 审计 CSV —— 每笔 OPEN/CLOSE 成交追加一行,供 parity_audit fill 对账。

环境变量:
  QQQ_BTC_FILL_AUDIT_PATH  默认 ~/quant_project/shadow/fill_audit.csv
  QQQ_BTC_FILL_AUDIT=0     可关闭写入
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from qqq_btc.live.oms_adapter import audit_fill
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.live.fill_audit")

_LOCK = threading.Lock()
_HEADER = (
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


def fill_audit_enabled() -> bool:
    if os.environ.get("QQQ_BTC_FILL_AUDIT", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def default_audit_path() -> Path:
    raw = os.environ.get("QQQ_BTC_FILL_AUDIT_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "quant_project" / "shadow" / "fill_audit.csv"


def append_fill_audit_row(row: Dict[str, Any], path: Optional[Path] = None) -> None:
    if not fill_audit_enabled():
        return
    path = path or default_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {k: row.get(k, "") for k in _HEADER}
    with _LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_HEADER)
            if write_header:
                w.writeheader()
            w.writerow(line)


def record_fill_audit(
    *,
    symbol: str,
    side: str,
    fill_px: float,
    bid: float,
    ask: float,
    qty: float = 1.0,
    action: str = "FILL",
    ts: Optional[float] = None,
    reason: str = "",
    exit_reason: str = "",
    mode: str = "",
    leg: str = "",
    session_bar: Any = "",
    net_return: Any = "",
) -> Optional[Dict[str, Any]]:
    """audit_fill + 写 CSV;返回行 dict 便于单测。"""
    if not fill_audit_enabled():
        return None
    if bid <= 0 or ask < bid or fill_px <= 0:
        logger.debug("fill audit skip %s: invalid quote bid=%s ask=%s fill=%s", symbol, bid, ask, fill_px)
        return None
    rec = audit_fill(bid, ask, fill_px, side, qcfg.FILL_MODEL)
    row = {
        "ts": float(ts if ts is not None else time.time()),
        "symbol": str(symbol),
        "action": action,
        "side": rec.side,
        "qty": float(qty),
        "fill_px": rec.fill_px,
        "bid": rec.bid,
        "ask": rec.ask,
        "spread_pct": round(rec.spread_pct, 6) if rec.spread_pct == rec.spread_pct else "",
        "fill_spread_frac": round(rec.fill_spread_frac, 6) if rec.fill_spread_frac == rec.fill_spread_frac else "",
        "model_frac": rec.model_entry_frac,
        "delta_frac": round(rec.fill_spread_frac - rec.model_entry_frac, 6)
        if rec.fill_spread_frac == rec.fill_spread_frac
        else "",
        "reason": reason,
        "exit_reason": exit_reason,
        "mode": mode,
        "leg": str(leg or ""),
        "session_bar": session_bar if session_bar != "" else "",
        "net_return": net_return if net_return != "" else "",
    }
    append_fill_audit_row(row)
    logger.info(
        "fill_audit %s %s %s frac=%.3f model=%.3f delta=%.3f",
        symbol,
        action,
        side,
        float(row["fill_spread_frac"] or 0),
        rec.model_entry_frac,
        float(row["delta_frac"] or 0),
    )
    return row


def apply_fill_audit_patch() -> None:
    """Monkey-patch OrchestratorAccounting OPEN/EXIT 成交后写 shadow CSV。"""
    import orchestrator_accounting as oac

    if getattr(oac.OrchestratorAccounting, "_qqq_btc_fill_audit_patched", False):
        return

    _orig_open = oac.OrchestratorAccounting._process_open_accounting
    _orig_exit = oac.OrchestratorAccounting._process_exit_accounting

    def _wrapped_open(self, sym, st, filled_qty, fill_price, stock_price, entry_ts, sig, *args, **kwargs):
        _orig_open(self, sym, st, filled_qty, fill_price, stock_price, entry_ts, sig, *args, **kwargs)
        if filled_qty <= 0:
            return
        try:
            import pandas as pd
            from pytz import timezone as tz

            from qqq_btc.common.time_features import session_minute
            from qqq_btc.live.session_governor import get_session_governor

            ny = tz("America/New_York")
            ts = pd.Series([pd.Timestamp.fromtimestamp(float(entry_ts), tz=ny)])
            st.qqq_btc_entry_bar = int(session_minute(ts).iloc[0])
            rails, vol_scale = get_session_governor().scaled_exit_rails(sym)
            st.qqq_btc_exit_rails = rails
            st.qqq_btc_vol_scale = float(vol_scale)
        except Exception as e:
            logger.warning("qqq_btc entry rails scale skipped: %s", e)
        meta = dict(sig.get("meta", {}) or {})
        execution_meta = kwargs.get("execution_meta") or {}
        if isinstance(execution_meta, dict):
            meta.update({k: v for k, v in execution_meta.items() if v not in (None, "")})
        leg = str(sig.get("leg") or meta.get("leg") or "")
        if not leg:
            # reason 形如 QQQ_BTC_ENTRY|CALL|E:0.15
            for part in str(sig.get("reason", "") or "").split("|"):
                p = part.strip().upper()
                if p in ("CALL", "PUT", "STRADDLE"):
                    leg = p
                    break
        record_fill_audit(
            symbol=sym,
            side="BUY",
            fill_px=float(fill_price),
            bid=float(meta.get("bid", st.entry_fill_bid if hasattr(st, "entry_fill_bid") else 0) or 0),
            ask=float(meta.get("ask", st.entry_fill_ask if hasattr(st, "entry_fill_ask") else 0) or 0),
            qty=float(filled_qty),
            action="OPEN",
            ts=float(entry_ts),
            reason=str(sig.get("reason", "") or ""),
            mode=str(kwargs.get("mode_override") or getattr(self.orch, "mode", "") or ""),
            leg=leg,
            session_bar=getattr(st, "qqq_btc_entry_bar", ""),
        )

    def _wrapped_exit(
        self, sym, st, filled_qty, fill_price, stock_price, curr_ts, reason, duration, ratio,
        original_position=None, execution_meta=None,
    ):
        entry_px = float(getattr(st, "entry_price", 0.0) or 0.0)
        pos_dir = int(original_position if original_position is not None else getattr(st, "position", 0) or 0)
        _orig_exit(
            self, sym, st, filled_qty, fill_price, stock_price, curr_ts, reason, duration, ratio,
            original_position=original_position, execution_meta=execution_meta,
        )
        if filled_qty <= 0:
            return
        meta = execution_meta if isinstance(execution_meta, dict) else {}
        leg = "PUT" if pos_dir < 0 else "CALL"
        nr = ""
        sb = ""
        try:
            from qqq_btc.common.time_features import session_minute
            from qqq_btc.live.session_governor import net_return_from_prices
            import pandas as pd
            from pytz import timezone as tz

            ny = tz("America/New_York")
            ts_s = pd.Series([pd.Timestamp.fromtimestamp(float(curr_ts), tz=ny)])
            sb = int(session_minute(ts_s).iloc[0])
            comm = 0.0
            try:
                comm = float(qcfg.FILL_MODEL.commission_return_drag(entry_px))
            except Exception:
                pass
            nr = float(net_return_from_prices(entry_px, float(fill_price), commission_drag=comm))
        except Exception:
            pass
        record_fill_audit(
            symbol=sym,
            side="SELL",
            fill_px=float(fill_price),
            bid=float(meta.get("bid", 0) or 0),
            ask=float(meta.get("ask", 0) or 0),
            qty=float(filled_qty),
            action="CLOSE",
            ts=float(curr_ts),
            reason="",
            exit_reason=str(reason or ""),
            mode=str(getattr(self.orch, "mode", "") or ""),
            leg=leg,
            session_bar=sb,
            net_return=nr,
        )
        try:
            from qqq_btc.live.session_governor import get_session_governor, net_return_from_prices

            comm = 0.0
            try:
                comm = float(qcfg.FILL_MODEL.commission_return_drag(entry_px))
            except Exception:
                pass
            nr2 = net_return_from_prices(entry_px, float(fill_price), commission_drag=comm)
            # 与 entry 共用同一 singleton governor（含 LIVE_REPLAY env）
            streak_until = get_session_governor().record_trade_close(
                sym, net_ret=nr2, curr_ts=float(curr_ts), leg=leg,
            )
            if streak_until > 0 and hasattr(st, "cooldown_until"):
                st.cooldown_until = max(float(getattr(st, "cooldown_until", 0.0) or 0.0), streak_until)
        except Exception as e:
            logger.warning("session_governor trade_close skipped: %s", e)

    oac.OrchestratorAccounting._process_open_accounting = _wrapped_open
    oac.OrchestratorAccounting._process_exit_accounting = _wrapped_exit
    oac.OrchestratorAccounting._qqq_btc_fill_audit_patched = True
    logger.info("patched OrchestratorAccounting → fill_audit CSV %s", default_audit_path())
