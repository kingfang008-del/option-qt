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


def attach_mf_features(df: pd.DataFrame, mf_window: int = 10, vol_ma_window: int = 20) -> pd.DataFrame:
    """Causal mf10 / streak / from_prev / vol_z on RTH 1m bars (left-label stock OK)."""
    out = df.copy()
    hl = (out["high"] - out["low"]).replace(0, np.nan)
    buy = ((out["close"] - out["low"]) / hl).fillna(0.5) * out["volume"]
    sell = ((out["high"] - out["close"]) / hl).fillna(0.5) * out["volume"]
    out["net$"] = (buy - sell) * out["close"]
    out["mf10"] = out.groupby("date")["net$"].transform(lambda s: s.rolling(mf_window, min_periods=mf_window).sum())
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


def first_rule_a_day(
    g: pd.DataFrame,
    *,
    window_start: str = "10:30",
    window_end: str = "14:00",
    streak_min: int = 8,
    from_prev_abs: float = 0.02,
    vol_z_min: float = 1.0,
) -> dict[str, Any] | None:
    """First Rule-A fire of the day (UP or DN, earlier wins)."""
    w = g[(g["tod"] >= window_start) & (g["tod"] <= window_end)].sort_values("timestamp")
    up = w[
        (w["streak_up"] >= streak_min)
        & (w["cum"] > 0)
        & (w["from_prev"] >= from_prev_abs)
        & (w["vol_z"] >= vol_z_min)
    ]
    dn = w[
        (w["streak_dn"] >= streak_min)
        & (w["cum"] < 0)
        & (w["from_prev"] <= -from_prev_abs)
        & (w["vol_z"] >= vol_z_min)
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
) -> list[pd.Timestamp]:
    w = g[(g["tod"] >= window_start) & (g["tod"] <= window_end)].sort_values("timestamp")
    if direction == "UP":
        hits = w[
            (w["streak_up"] >= streak_min)
            & (w["cum"] > 0)
            & (w["from_prev"] >= from_prev_abs)
            & (w["vol_z"] >= vol_z_min)
        ]
    else:
        hits = w[
            (w["streak_dn"] >= streak_min)
            & (w["cum"] < 0)
            & (w["from_prev"] <= -from_prev_abs)
            & (w["vol_z"] >= vol_z_min)
        ]
    return list(hits["timestamp"])


def build_topk_signals(
    stock_by_symbol: dict[str, pd.DataFrame],
    cfg_signal: dict[str, Any],
) -> pd.DataFrame:
    """Per-symbol first Rule-A, then TopK earliest within each date."""
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
        self.bars: list[dict[str, Any]] = []
        self.prev_close: float | None = None
        self.day_open: float | None = None
        self.date: str | None = None
        self.cum = 0.0
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

        # mf10
        nets = [b["net$"] for b in self.bars]
        mf10 = float(np.sum(nets[-self.mf_window:])) if len(nets) >= self.mf_window else np.nan
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
        fp = float(self.cfg.get("from_prev_abs", 0.02))
        vz_min = float(self.cfg.get("vol_z_min", 1.0))
        direction = None
        if self.streak_up >= streak_min and self.cum > 0 and from_prev >= fp and vol_z >= vz_min:
            direction = "UP"
        elif self.streak_dn >= streak_min and self.cum < 0 and from_prev <= -fp and vol_z >= vz_min:
            direction = "DN"
        if direction is None:
            return None
        fire = {"date": date, "symbol": self.symbol, "dir": direction, "sig_ts": ts, "spot": c}
        if not self.fired_today:
            self.first_fire = fire
        self.fired_today = True
        return fire
