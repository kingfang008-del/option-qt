"""Option trade prints (1s OHLCV aggregates) for mark-to-trade path signals.

Layout (Polygon OPRA trades → optional 1s agg)::
  {option_trades_root}/{SYM}/{SYM}_{YYYY-MM-DD}.parquet
  columns: ticker, timestamp, o, h, l, c, v, n, t

Used for early toxic-path detection (MFE/MAE on last trade ``c``); exits still
fill on quotes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class TradeToxicConfig:
    enabled: bool = False
    cut_ret: float = 0.25
    mfe_bypass: float = 0.05
    min_hold_seconds: int = 60
    # Asymmetric guards (default off / unlimited): require dig to persist, and/or
    # only allow TRADE_TOX within the first max_cut_seconds after fill.
    persist_seconds: int = 0
    max_cut_seconds: int | None = None
    # Optional: also require quote sell MTM <= -quote_confirm_ret (None = off).
    quote_confirm_ret: float | None = None
    # Soft MFE bypass when underlying barely moves against the trade (divergence).
    # If stock adverse ret < div_stock_adverse_max, allow peak MFE < div_mfe_bypass.
    div_mfe_bypass: float | None = None
    div_stock_adverse_max: float | None = None
    # When OPRA trade prints are missing, mark toxic on quote sell path (live parity).
    quote_fallback: bool = False
    # Optional dig threshold used only on the quote-fallback mark (prints missing).
    # Do NOT apply in parallel with prints — dual-window FAIL (strong keep≈0.52).
    quote_fallback_cut_ret: float | None = None


def _opt_float(v: Any) -> float | None:
    if v in (None, "", False):
        return None
    return float(v)


def trade_toxic_from_trade(trade: dict[str, Any] | None) -> TradeToxicConfig:
    raw = (trade or {}).get("trade_toxic") or {}
    if not isinstance(raw, dict):
        # Also allow early_exit_mode=trade_toxic with mae-style flat keys.
        early = str((trade or {}).get("early_exit_mode") or "").strip().lower()
        if early not in {"trade_toxic", "trade_mae", "toxic_trade"}:
            return TradeToxicConfig(enabled=False)
        max_cut = (trade or {}).get("trade_toxic_max_cut_seconds")
        qconf = (trade or {}).get("trade_toxic_quote_confirm_ret")
        return TradeToxicConfig(
            enabled=True,
            cut_ret=float((trade or {}).get("trade_toxic_cut_ret", (trade or {}).get("mae_cut_ret", 0.25)) or 0.25),
            mfe_bypass=float(
                (trade or {}).get("trade_toxic_mfe_bypass", (trade or {}).get("mae_cut_mfe_bypass", 0.05))
                or 0.05
            ),
            min_hold_seconds=int(
                (trade or {}).get("trade_toxic_min_hold_seconds", 60) or 60
            ),
            persist_seconds=int((trade or {}).get("trade_toxic_persist_seconds", 0) or 0),
            max_cut_seconds=int(max_cut) if max_cut not in (None, "", False) else None,
            quote_confirm_ret=_opt_float(qconf),
            div_mfe_bypass=_opt_float((trade or {}).get("trade_toxic_div_mfe_bypass")),
            div_stock_adverse_max=_opt_float((trade or {}).get("trade_toxic_div_stock_adverse_max")),
            quote_fallback=bool((trade or {}).get("trade_toxic_quote_fallback", False)),
            quote_fallback_cut_ret=_opt_float((trade or {}).get("trade_toxic_quote_fallback_cut_ret")),
        )
    max_cut = raw.get("max_cut_seconds")
    return TradeToxicConfig(
        enabled=bool(raw.get("enabled", False)),
        cut_ret=float(raw.get("cut_ret", 0.25) or 0.25),
        mfe_bypass=float(raw.get("mfe_bypass", 0.05) or 0.05),
        min_hold_seconds=int(raw.get("min_hold_seconds", 60) or 60),
        persist_seconds=int(raw.get("persist_seconds", 0) or 0),
        max_cut_seconds=int(max_cut) if max_cut not in (None, "", False) else None,
        quote_confirm_ret=_opt_float(raw.get("quote_confirm_ret")),
        div_mfe_bypass=_opt_float(raw.get("div_mfe_bypass")),
        div_stock_adverse_max=_opt_float(raw.get("div_stock_adverse_max")),
        quote_fallback=bool(raw.get("quote_fallback", False)),
        quote_fallback_cut_ret=_opt_float(raw.get("quote_fallback_cut_ret")),
    )


def load_option_trades(trades_root: Path | str | None, symbol: str, date: str) -> pd.DataFrame | None:
    if trades_root is None:
        return None
    root = Path(trades_root).expanduser()
    p = root / symbol / f"{symbol}_{date}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty or "timestamp" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NY)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NY)
    return df


def path_for_ticker_trades(tdf: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    """Return ``timestamp, last`` (from ``c``) for one OCC ticker."""
    if tdf is None or tdf.empty:
        return None
    t = str(ticker).replace("O:", "")
    sub = tdf[tdf["ticker"].astype(str).str.replace("O:", "", regex=False) == t].sort_values(
        "timestamp"
    )
    if sub.empty:
        return None
    sub = sub.drop_duplicates("timestamp", keep="last")
    px_col = "c" if "c" in sub.columns else ("price" if "price" in sub.columns else None)
    if px_col is None:
        return None
    out = pd.DataFrame(
        {
            "timestamp": sub["timestamp"].to_numpy(),
            "last": sub[px_col].astype(float).to_numpy(),
        }
    )
    out = out[np.isfinite(out["last"]) & (out["last"] > 0)].reset_index(drop=True)
    return None if out.empty else out


def prepare_trade_mark_arrays(
    trade_path: pd.DataFrame | None,
    entry_ts: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Causal trade marks at/after entry.

    Returns ``(ts_ns, last_px, trade_entry)`` or None if no prints.
    """
    if trade_path is None or trade_path.empty:
        return None
    et = pd.Timestamp(entry_ts)
    if et.tzinfo is None:
        et = et.tz_localize(NY)
    else:
        et = et.tz_convert(NY)
    after = trade_path[trade_path["timestamp"] >= et]
    if after.empty:
        return None
    px = after["last"].astype(float).to_numpy()
    ts = pd.to_datetime(after["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    entry = float(px[0])
    if not np.isfinite(entry) or entry <= 0:
        return None
    ts_ns = np.array([int(pd.Timestamp(x).value) for x in ts], dtype=np.int64)
    return ts_ns, px, entry


def prepare_quote_mark_arrays(
    timestamps: list | np.ndarray | pd.Series,
    sell_px: list | np.ndarray | pd.Series,
    *,
    entry_px: float,
    fill_ts: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Causal quote-sell marks at/after fill (live OMS parity when prints missing).

    Returns ``(ts_ns, sell_px, entry_px)`` or None.
    """
    entry = float(entry_px)
    if not np.isfinite(entry) or entry <= 0:
        return None
    ft = pd.Timestamp(fill_ts)
    if ft.tzinfo is None:
        ft = ft.tz_localize(NY)
    else:
        ft = ft.tz_convert(NY)
    ft_ns = int(ft.value)
    ts_out: list[int] = []
    px_out: list[float] = []
    for raw_t, raw_p in zip(timestamps, sell_px):
        t = pd.Timestamp(raw_t)
        if t.tzinfo is None:
            t = t.tz_localize(NY)
        else:
            t = t.tz_convert(NY)
        p = float(raw_p)
        if int(t.value) < ft_ns or (not np.isfinite(p)) or p <= 0:
            continue
        ts_out.append(int(t.value))
        px_out.append(p)
    if not ts_out:
        return None
    return np.asarray(ts_out, dtype=np.int64), np.asarray(px_out, dtype=float), entry


def trade_mtm_asof(
    ts_ns: np.ndarray,
    px: np.ndarray,
    trade_entry: float,
    asof_ts: pd.Timestamp,
) -> float | None:
    """Last trade MTM at/before ``asof_ts`` relative to ``trade_entry``."""
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    i = int(np.searchsorted(ts_ns, int(asof.value), side="right") - 1)
    if i < 0:
        return None
    last = float(px[i])
    if not np.isfinite(last) or last <= 0 or trade_entry <= 0:
        return None
    return last / trade_entry - 1.0


def trade_peak_mfe_asof(
    ts_ns: np.ndarray,
    px: np.ndarray,
    trade_entry: float,
    asof_ts: pd.Timestamp,
) -> float | None:
    """Running max trade MTM of all prints at/before ``asof_ts``."""
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    i = int(np.searchsorted(ts_ns, int(asof.value), side="right") - 1)
    if i < 0 or trade_entry <= 0:
        return None
    window = px[: i + 1].astype(float)
    window = window[np.isfinite(window) & (window > 0)]
    if window.size == 0:
        return None
    return float(window.max() / trade_entry - 1.0)


def trade_toxic_mfe_limit(
    cfg: TradeToxicConfig,
    *,
    stock_adverse: float | None = None,
) -> float:
    """Effective peak-MFE bypass threshold (includes stock-divergence soft MFE)."""
    mfe_lim = float(cfg.mfe_bypass)
    if (
        cfg.div_mfe_bypass is not None
        and cfg.div_stock_adverse_max is not None
        and stock_adverse is not None
        and np.isfinite(float(stock_adverse))
        and float(stock_adverse) < float(cfg.div_stock_adverse_max)
    ):
        mfe_lim = max(mfe_lim, float(cfg.div_mfe_bypass))
    return mfe_lim


def trade_toxic_in_cut_window(
    held_seconds: float,
    cfg: TradeToxicConfig,
    *,
    bypass_max_cut: bool = False,
) -> bool:
    """True when held long enough and still inside the asymmetric cut window."""
    if held_seconds < float(cfg.min_hold_seconds or 0):
        return False
    if bypass_max_cut:
        return True
    max_cut = cfg.max_cut_seconds
    if max_cut is None:
        return True
    return held_seconds <= float(max_cut)


def trade_toxic_cut_ret(
    cfg: TradeToxicConfig,
    *,
    mark_source: str,
) -> float:
    """Resolve the toxic threshold for an OPRA print or quote-fallback mark."""
    if (
        str(mark_source).strip().lower() == "quote"
        and cfg.quote_fallback
        and cfg.quote_fallback_cut_ret is not None
    ):
        return float(cfg.quote_fallback_cut_ret)
    return float(cfg.cut_ret)


def trade_toxic_is_dig(
    *,
    mtm_ret: float,
    peak_mfe: float,
    cfg: TradeToxicConfig,
    stock_adverse: float | None = None,
    cut_ret: float | None = None,
) -> bool:
    """Core dig predicate: peak MFE below bypass and MTM ≤ −cut_ret."""
    if not np.isfinite(float(mtm_ret)) or not np.isfinite(float(peak_mfe)):
        return False
    mfe_lim = trade_toxic_mfe_limit(cfg, stock_adverse=stock_adverse)
    cut = float(cfg.cut_ret) if cut_ret is None else float(cut_ret)
    return float(peak_mfe) < mfe_lim and float(mtm_ret) <= -cut
