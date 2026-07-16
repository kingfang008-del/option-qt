"""Rule A money-flow signals + TopK earliest ranking."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

NY = "America/New_York"


def _to_ny(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts)
    if getattr(out.dt, "tz", None) is None:
        return out.dt.tz_localize("UTC").dt.tz_convert(NY)
    return out.dt.tz_convert(NY)


def load_stock_month_files(stock_root, symbol: str, months: Iterable[str]) -> pd.DataFrame:
    frames = []
    root = stock_root
    for m in months:
        p = root / symbol / f"{m}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = _to_ny(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    t = df["timestamp"].dt.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())].copy()
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    return df


def attach_mf_features(
    df: pd.DataFrame,
    mf_window: int = 10,
    vol_ma_window: int = 20,
    *,
    mf_confirm_bars: int = 3,
) -> pd.DataFrame:
    """Causal mf10 / streak / from_prev / vol_z on RTH 1m bars (left-label stock OK).

    Also builds short-horizon ``mf_short`` (last ``mf_confirm_bars`` net$) so entry can
    require recent flow still aligned — reduces late fires into V-reversals.
    """
    out = df.copy()
    hl = (out["high"] - out["low"]).replace(0, np.nan)
    buy = ((out["close"] - out["low"]) / hl).fillna(0.5) * out["volume"]
    sell = ((out["high"] - out["close"]) / hl).fillna(0.5) * out["volume"]
    out["net$"] = (buy - sell) * out["close"]
    out["mf10"] = out.groupby("date")["net$"].transform(lambda s: s.rolling(mf_window, min_periods=mf_window).sum())
    confirm_n = max(1, int(mf_confirm_bars))
    out["mf_short"] = out.groupby("date")["net$"].transform(
        lambda s: s.rolling(confirm_n, min_periods=confirm_n).sum()
    )
    out["cum"] = out.groupby("date")["net$"].cumsum()
    day_close = out.groupby("date")["close"].last()
    prev = day_close.shift(1)
    out["prev_close"] = out["date"].map(prev).fillna(out.groupby("date")["open"].transform("first"))
    out["from_prev"] = out["close"] / out["prev_close"] - 1.0
    out["vol_ma"] = out.groupby("date")["volume"].transform(
        lambda s: s.rolling(vol_ma_window, min_periods=5).mean()
    )
    out["vol_z"] = (out["volume"] / out["vol_ma"]).replace([np.inf, -np.inf], np.nan)

    def _streak(mask: pd.Series) -> pd.Series:
        x = mask.astype(int)
        parts = []
        for _, s in x.groupby(out["date"]):
            c = 0
            vals = []
            for v in s:
                c = c + 1 if v else 0
                vals.append(c)
            parts.append(pd.Series(vals, index=s.index))
        return pd.concat(parts).sort_index()

    out["streak_up"] = _streak(out["mf10"] > 0)
    out["streak_dn"] = _streak(out["mf10"] < 0)
    return out


def _rule_a_mask(
    w: pd.DataFrame,
    *,
    direction: str,
    streak_min: int,
    from_prev_abs: float,
    vol_z_min: float,
    streak_max: int | None,
    require_mf_short_align: bool,
) -> pd.Series:
    if direction == "UP":
        m = (
            (w["streak_up"] >= streak_min)
            & (w["cum"] > 0)
            & (w["from_prev"] >= from_prev_abs)
            & (w["vol_z"] >= vol_z_min)
        )
        if streak_max is not None:
            m = m & (w["streak_up"] <= int(streak_max))
        if require_mf_short_align and "mf_short" in w.columns:
            m = m & (w["mf_short"] > 0)
        return m
    m = (
        (w["streak_dn"] >= streak_min)
        & (w["cum"] < 0)
        & (w["from_prev"] <= -from_prev_abs)
        & (w["vol_z"] >= vol_z_min)
    )
    if streak_max is not None:
        m = m & (w["streak_dn"] <= int(streak_max))
    if require_mf_short_align and "mf_short" in w.columns:
        m = m & (w["mf_short"] < 0)
    return m


def first_rule_a_day(
    g: pd.DataFrame,
    *,
    window_start: str = "10:30",
    window_end: str = "14:00",
    streak_min: int = 8,
    from_prev_abs: float = 0.02,
    vol_z_min: float = 1.0,
    streak_max: int | None = None,
    require_mf_short_align: bool = False,
) -> dict[str, Any] | None:
    """First Rule-A fire of the day (UP or DN, earlier wins)."""
    w = g[(g["tod"] >= window_start) & (g["tod"] <= window_end)].sort_values("timestamp")
    up = w[
        _rule_a_mask(
            w,
            direction="UP",
            streak_min=streak_min,
            from_prev_abs=from_prev_abs,
            vol_z_min=vol_z_min,
            streak_max=streak_max,
            require_mf_short_align=require_mf_short_align,
        )
    ]
    dn = w[
        _rule_a_mask(
            w,
            direction="DN",
            streak_min=streak_min,
            from_prev_abs=from_prev_abs,
            vol_z_min=vol_z_min,
            streak_max=streak_max,
            require_mf_short_align=require_mf_short_align,
        )
    ]
    cand: list[tuple[str, pd.Timestamp, float]] = []
    if len(up):
        r = up.iloc[0]
        cand.append(("UP", r["timestamp"], float(r["close"])))
    if len(dn):
        r = dn.iloc[0]
        cand.append(("DN", r["timestamp"], float(r["close"])))
    if not cand:
        return None
    cand.sort(key=lambda x: x[1])
    d, ts, spot = cand[0]
    return {"dir": d, "sig_ts": ts, "spot": spot}


def all_rule_a_times(
    g: pd.DataFrame,
    direction: str,
    *,
    window_start: str = "10:30",
    window_end: str = "14:00",
    streak_min: int = 8,
    from_prev_abs: float = 0.02,
    vol_z_min: float = 1.0,
    streak_max: int | None = None,
    require_mf_short_align: bool = False,
) -> list[pd.Timestamp]:
    w = g[(g["tod"] >= window_start) & (g["tod"] <= window_end)].sort_values("timestamp")
    hits = w[
        _rule_a_mask(
            w,
            direction=direction,
            streak_min=streak_min,
            from_prev_abs=from_prev_abs,
            vol_z_min=vol_z_min,
            streak_max=streak_max,
            require_mf_short_align=require_mf_short_align,
        )
    ]
    return list(hits["timestamp"])

def sync_index(
    stock_by_symbol: dict[str, pd.DataFrame],
    *,
    date: str,
    asof_ts: pd.Timestamp,
    peer_symbols: list[str],
) -> float | None:
    """Cross-sectional money-flow sync index in [-1, 1].

    ``SI = mean(sign(mf10))`` over peers with finite mf10 at/before ``asof_ts``.
    """
    if not peer_symbols:
        return None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    signs: list[float] = []
    for sym in peer_symbols:
        df = stock_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        day = df[df["date"] == date]
        if day.empty:
            continue
        bar = day[day["timestamp"] <= asof].tail(1)
        if bar.empty:
            continue
        mf = float(bar.iloc[0]["mf10"]) if "mf10" in bar.columns else float("nan")
        if not np.isfinite(mf) or mf == 0:
            continue
        signs.append(1.0 if mf > 0 else -1.0)
    if not signs:
        return None
    return float(np.mean(signs))


def price_efficiency_ok(
    stock_df: pd.DataFrame | None,
    *,
    asof_ts: pd.Timestamp,
    direction: str,
    window: int = 10,
    min_ratio: float = 0.5,
    lookback_bars: int = 780,
) -> tuple[bool, float | None, float | None]:
    """Price-volume efficiency gate (absorption / iceberg filter).

    ``PE = |Δprice| / sum(|net$|)`` over ``window`` bars ending at ``asof_ts``.
    Block when favorable 10m flow is present and
    ``PE < min_ratio ×`` causal rolling mean of PE (prior ``lookback_bars``).

    Pass multi-day ``stock_df`` so the PE mean has history. Missing → allow.
    """
    if stock_df is None or stock_df.empty or direction not in ("UP", "DN"):
        return True, None, None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    if "net$" not in stock_df.columns or "close" not in stock_df.columns:
        return True, None, None
    # Cache PE columns on the frame once (mutates copy-safe via assigned cols).
    df = stock_df
    if "_pe_abs" not in df.columns or "_pe_flow_up" not in df.columns:
        close = df["close"].astype(float)
        net = df["net$"].astype(float)
        abs_net = net.abs()
        ret = close / close.shift(int(window)) - 1.0
        denom = abs_net.rolling(int(window), min_periods=int(window)).sum()
        pe_abs = ret.abs() / denom.replace(0, np.nan)
        flow10 = net.rolling(int(window), min_periods=int(window)).sum()
        pe_ma = pe_abs.shift(1).rolling(
            int(lookback_bars), min_periods=max(30, int(lookback_bars) // 10)
        ).mean()
        df = df.copy()
        df["_pe_abs"] = pe_abs
        df["_pe_flow10"] = flow10
        df["_pe_ma"] = pe_ma
        # write back so later calls reuse (caller holds stock_by ref)
        stock_df["_pe_abs"] = pe_abs
        stock_df["_pe_flow10"] = flow10
        stock_df["_pe_ma"] = pe_ma
    else:
        df = stock_df

    mask = df["timestamp"] <= asof
    if not mask.any():
        return True, None, None
    row = df.loc[mask].iloc[-1]
    pe = float(row["_pe_abs"]) if np.isfinite(row["_pe_abs"]) else float("nan")
    ma = float(row["_pe_ma"]) if np.isfinite(row["_pe_ma"]) else float("nan")
    flow10 = float(row["_pe_flow10"]) if np.isfinite(row["_pe_flow10"]) else float("nan")
    align = 1.0 if direction == "UP" else -1.0
    flow = flow10 * align if np.isfinite(flow10) else float("nan")
    if not np.isfinite(pe) or not np.isfinite(ma) or ma <= 0:
        return True, (pe if np.isfinite(pe) else None), (ma if np.isfinite(ma) else None)
    if not np.isfinite(flow) or flow <= 0:
        return True, pe, ma
    return bool(pe >= float(min_ratio) * ma), pe, ma


def count_peer_align(
    stock_by_symbol: dict[str, pd.DataFrame],
    *,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    peer_symbols: list[str],
    mode: str = "mf10",
    streak_min: int = 8,
) -> int:
    """Count peers aligned with ``direction`` at/before ``asof_ts`` (causal).

    ``mode``:
      - ``mf10``: mf10 sign matches (UP>0 / DN<0)
      - ``streak``: streak_up/dn >= streak_min
      - ``mf_fp``: mf10 aligned AND from_prev same-sign
    """
    if direction not in ("UP", "DN") or not peer_symbols:
        return 0
    mode = str(mode or "mf10").strip().lower()
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    n = 0
    for sym in peer_symbols:
        df = stock_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        day = df[df["date"] == date]
        if day.empty:
            continue
        bar = day[day["timestamp"] <= asof].tail(1)
        if bar.empty:
            continue
        row = bar.iloc[0]
        mf = float(row["mf10"]) if "mf10" in row.index and np.isfinite(row["mf10"]) else float("nan")
        if mode == "mf10":
            ok = (direction == "UP" and mf > 0) or (direction == "DN" and mf < 0)
        elif mode == "streak":
            su = int(row["streak_up"]) if "streak_up" in row.index else 0
            sd = int(row["streak_dn"]) if "streak_dn" in row.index else 0
            ok = (direction == "UP" and su >= streak_min) or (direction == "DN" and sd >= streak_min)
        elif mode in ("mf_fp", "mf10_fp"):
            fp = float(row["from_prev"]) if "from_prev" in row.index and np.isfinite(row["from_prev"]) else float("nan")
            ok = (
                (direction == "UP" and mf > 0 and np.isfinite(fp) and fp > 0)
                or (direction == "DN" and mf < 0 and np.isfinite(fp) and fp < 0)
            )
        else:
            ok = (direction == "UP" and mf > 0) or (direction == "DN" and mf < 0)
        if ok:
            n += 1
    return n


def build_topk_signals(
    stock_by_symbol: dict[str, pd.DataFrame],
    cfg_signal: dict[str, Any],
) -> pd.DataFrame:
    """Per-symbol first Rule-A, then TopK earliest within each date."""
    streak_max = cfg_signal.get("streak_max")
    streak_max_i = int(streak_max) if streak_max is not None else None
    require_short = bool(cfg_signal.get("require_mf_short_align", False))
    rows = []
    for sym, df in stock_by_symbol.items():
        if df.empty:
            continue
        for date, g in df.groupby("date"):
            hit = first_rule_a_day(
                g,
                window_start=str(cfg_signal.get("window_start", "10:30")),
                window_end=str(cfg_signal.get("window_end", "14:00")),
                streak_min=int(cfg_signal.get("streak_min", 8)),
                from_prev_abs=float(cfg_signal.get("from_prev_abs", 0.02)),
                vol_z_min=float(cfg_signal.get("vol_z_min", 1.0)),
                streak_max=streak_max_i,
                require_mf_short_align=require_short,
            )
            if hit is None:
                continue
            rows.append({"date": date, "symbol": sym, **hit})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "dir", "sig_ts", "spot", "rank"])
    sig = pd.DataFrame(rows).sort_values(["date", "sig_ts", "symbol"])
    top_k = int(cfg_signal.get("top_k", 2))
    sig["rank"] = sig.groupby("date").cumcount() + 1
    return sig[sig["rank"] <= top_k].reset_index(drop=True)

class StreamSignalState:
    """Causal per-symbol Rule-A state for streaming parity."""

    def __init__(self, symbol: str, cfg_signal: dict[str, Any], *, emit_all: bool = False):
        self.symbol = symbol
        self.cfg = cfg_signal
        self.emit_all = emit_all  # True → every Rule-A bar (reentry); False → first only
        self.mf_window = int(cfg_signal.get("mf_window", 10))
        self.vol_ma_window = int(cfg_signal.get("vol_ma_window", 20))
        self.mf_confirm_bars = max(1, int(cfg_signal.get("mf_confirm_bars", 3)))
        self.bars: list[dict[str, Any]] = []
        self.prev_close: float | None = None
        self.day_open: float | None = None
        self.date: str | None = None
        self.cum = 0.0
        self.mf10 = float("nan")
        self.mf_short = float("nan")
        self.streak_up = 0
        self.streak_dn = 0
        self.fired_today = False
        self.first_fire: dict[str, Any] | None = None

    def on_bar(self, bar: dict[str, Any]) -> dict[str, Any] | None:
        """Ingest one 1m bar; return Rule-A fire once per day if triggers."""
        ts = pd.Timestamp(bar["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY)
        else:
            ts = ts.tz_convert(NY)
        date = ts.strftime("%Y-%m-%d")
        if self.date != date:
            if self.bars:
                self.prev_close = float(self.bars[-1]["close"])
            self.date = date
            self.bars = []
            self.cum = 0.0
            self.mf10 = float("nan")
            self.mf_short = float("nan")
            self.streak_up = 0
            self.streak_dn = 0
            self.fired_today = False
            self.first_fire = None
            self.day_open = float(bar["open"])

        o, h, l, c, v = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"])
        if self.day_open is None:
            self.day_open = o
        hl = h - l
        buy = ((c - l) / hl * v) if hl > 0 else 0.5 * v
        sell = ((h - c) / hl * v) if hl > 0 else 0.5 * v
        net = (buy - sell) * c
        self.cum += net
        self.bars.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v, "net$": net})

        # mf10 + short confirm window
        nets = [b["net$"] for b in self.bars]
        mf10 = float(np.sum(nets[-self.mf_window:])) if len(nets) >= self.mf_window else np.nan
        self.mf10 = mf10
        self.mf_short = (
            float(np.sum(nets[-self.mf_confirm_bars:]))
            if len(nets) >= self.mf_confirm_bars
            else np.nan
        )
        if np.isfinite(mf10) and mf10 > 0:
            self.streak_up += 1
            self.streak_dn = 0
        elif np.isfinite(mf10) and mf10 < 0:
            self.streak_dn += 1
            self.streak_up = 0
        else:
            self.streak_up = 0
            self.streak_dn = 0

        vols = [b["volume"] for b in self.bars]
        vol_ma = float(np.mean(vols[-self.vol_ma_window:])) if len(vols) >= 5 else np.nan
        vol_z = (v / vol_ma) if vol_ma and vol_ma > 0 else np.nan
        prev = self.prev_close if self.prev_close is not None else float(self.day_open)
        from_prev = c / prev - 1.0
        tod = ts.strftime("%H:%M")

        if self.fired_today and not self.emit_all:
            return None
        ws = str(self.cfg.get("window_start", "10:30"))
        we = str(self.cfg.get("window_end", "14:00"))
        if not (ws <= tod <= we):
            return None
        streak_min = int(self.cfg.get("streak_min", 8))
        streak_max = self.cfg.get("streak_max")
        streak_max_i = int(streak_max) if streak_max is not None else None
        fp = float(self.cfg.get("from_prev_abs", 0.02))
        vz_min = float(self.cfg.get("vol_z_min", 1.0))
        need_short = bool(self.cfg.get("require_mf_short_align", False))
        direction = None
        if self.streak_up >= streak_min and self.cum > 0 and from_prev >= fp and vol_z >= vz_min:
            if streak_max_i is not None and self.streak_up > streak_max_i:
                pass
            elif need_short and (not np.isfinite(self.mf_short) or self.mf_short <= 0):
                pass
            else:
                direction = "UP"
        elif self.streak_dn >= streak_min and self.cum < 0 and from_prev <= -fp and vol_z >= vz_min:
            if streak_max_i is not None and self.streak_dn > streak_max_i:
                pass
            elif need_short and (not np.isfinite(self.mf_short) or self.mf_short >= 0):
                pass
            else:
                direction = "DN"
        if direction is None:
            return None
        fire = {"date": date, "symbol": self.symbol, "dir": direction, "sig_ts": ts, "spot": c}
        if not self.fired_today:
            self.first_fire = fire
        self.fired_today = True
        return fire