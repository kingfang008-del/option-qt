"""Production-style trade_log (OPEN/CLOSE rows) for Mag7 day-stream checks.

Mirrors the spirit of ``production`` ``_emit_trade_log`` / ``replay_trades_*.csv``:
one row per open, one row per close, easy to diff entry/exit clocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def trades_to_trade_log(trades: Iterable[Any]) -> pd.DataFrame:
    """Expand DryTrade / StubTrade-like objects into OPEN+CLOSE trade_log rows."""
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(trades):
        if isinstance(t, dict):
            d = t
        else:
            d = {
                "date": getattr(t, "date", None),
                "symbol": getattr(t, "symbol", None),
                "direction": getattr(t, "dir", None) or getattr(t, "direction", None),
                "contract": getattr(t, "contract", None) or getattr(t, "ticker", None),
                "rank": getattr(t, "rank", None),
                "entry": getattr(t, "entry", None),
                "exit": getattr(t, "exit", None),
                "ret": getattr(t, "ret", None),
                "reason": getattr(t, "reason", None),
                "entry_ts": getattr(t, "entry_ts", None),
                "exit_ts": getattr(t, "exit_ts", None),
                "qty_frac": getattr(t, "qty_frac", None) or getattr(t, "size_frac", None),
                "pnl_equity": getattr(t, "pnl_equity", None),
            }
        trade_id = f"{d.get('date')}|{d.get('symbol')}|{d.get('entry_ts')}|{i}"
        common = {
            "trade_id": trade_id,
            "date": d.get("date"),
            "symbol": d.get("symbol"),
            "dir": d.get("direction") or d.get("dir"),
            "contract": d.get("contract"),
            "rank": d.get("rank"),
            "qty_frac": d.get("qty_frac"),
        }
        rows.append(
            {
                **common,
                "action": "OPEN",
                "ts": d.get("entry_ts"),
                "px": d.get("entry"),
                "ret": None,
                "reason": "ENTRY",
                "pnl_equity": None,
            }
        )
        rows.append(
            {
                **common,
                "action": "CLOSE",
                "ts": d.get("exit_ts"),
                "px": d.get("exit"),
                "ret": d.get("ret"),
                "reason": d.get("reason"),
                "pnl_equity": d.get("pnl_equity"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "date",
                "symbol",
                "dir",
                "contract",
                "rank",
                "qty_frac",
                "action",
                "ts",
                "px",
                "ret",
                "reason",
                "pnl_equity",
            ]
        )
    return pd.DataFrame(rows)


def write_trade_log(trades: Iterable[Any], out_dir: Path, *, name: str = "trade_log.csv") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    trades_to_trade_log(trades).to_csv(path, index=False)
    return path


def offline_trades_to_trade_log(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Convert offline replay trades.csv frame to trade_log schema."""
    if trades_df is None or trades_df.empty:
        return trades_to_trade_log([])
    rows = []
    for r in trades_df.itertuples(index=False):
        rows.append(
            {
                "date": getattr(r, "date", None),
                "symbol": getattr(r, "symbol", None),
                "direction": getattr(r, "dir", None) or getattr(r, "direction", None),
                "contract": getattr(r, "ticker", None) or getattr(r, "contract", None),
                "rank": getattr(r, "rank", None),
                "entry": getattr(r, "entry", None),
                "exit": getattr(r, "exit", None),
                "ret": getattr(r, "ret", None),
                "reason": getattr(r, "reason", None),
                "entry_ts": getattr(r, "entry_ts", None),
                "exit_ts": getattr(r, "exit_ts", None),
                "qty_frac": getattr(r, "size_frac", None) or getattr(r, "qty_frac", None),
                "pnl_equity": getattr(r, "pnl_equity", None),
            }
        )
    return trades_to_trade_log(rows)
