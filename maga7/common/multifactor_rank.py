"""Multi-factor ranking for late/obvious movers — research only, default off.

Scores Mag7 names causally on 1m bars. Intended to **re-rank** (or backfill)
TopK, not replace Rule-A entry logic in freeze.

Factors (UP; DN uses signed flip):
  - vol_x: cum$ / same-TOD 10d median
  - fp: from_prev
  - accel: fp − fp_lookback (short-horizon acceleration)
  - rs: fp − cross-sectional median fp (pool relative strength)
  - qqq_div: fp − QQQ from_prev (idiosyncratic vs index)
  - reclaim: for UP, max(0, fp − morning_min_fp) after having been soft-red

Cross-section rank score = sum of winsorized z-scores × weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from maga7.common.macro_unusual import build_tod_median_curve, prepare_day

NY = "America/New_York"
Direction = Literal["UP", "DN"]


@dataclass(frozen=True)
class MultiFactorConfig:
    lookback_days: int = 10
    accel_minutes: int = 15
    window_start: str = "10:30"
    window_end: str = "14:00"
    fp_gate: float = 0.005  # soft gate to enter ranking pool
    require_above_open: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "vol_x": 0.15,
            "fp": 0.25,
            "accel": 0.25,
            "rs": 0.20,
            "qqq_div": 0.10,
            "reclaim": 0.05,
        }
    )


@dataclass(frozen=True)
class FactorSnap:
    symbol: str
    date: str
    direction: Direction
    asof: pd.Timestamp
    tod: str
    fp: float
    vol_x: float
    accel: float
    rs: float
    qqq_div: float
    reclaim: float
    score: float
    rank: int = 0


def _winsor_z(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanpercentile(x, [10, 90])
    x = np.clip(x, lo, hi)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _fp_at(day: pd.DataFrame, tod: str) -> float | None:
    upto = day[day["tod"] <= tod]
    if upto.empty:
        return None
    v = upto.iloc[-1]["from_prev"]
    return float(v) if np.isfinite(v) else None


def _cum_at(day: pd.DataFrame, tod: str) -> float | None:
    upto = day[day["tod"] <= tod]
    if upto.empty:
        return None
    return float(upto.iloc[-1]["cum_dvol"])


def score_universe_at(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    asof_tod: str,
    direction: Direction,
    cfg: MultiFactorConfig,
    tod_median_by_sym: dict[str, dict[str, float]],
    day_cache: dict[str, pd.DataFrame] | None = None,
) -> list[FactorSnap]:
    """Cross-section score at one clock. Returns sorted by score desc."""
    day_cache = day_cache if day_cache is not None else {}
    qqq = stock_by.get("QQQ")
    if "QQQ" not in day_cache and qqq is not None:
        day_cache["QQQ"] = prepare_day(qqq, date)
    qday = day_cache.get("QQQ")
    qfp = _fp_at(qday, asof_tod) if qday is not None and not qday.empty else None

    raw: list[dict[str, Any]] = []
    for sym in symbols:
        sdf = stock_by.get(sym)
        if sdf is None:
            continue
        if sym not in day_cache:
            day_cache[sym] = prepare_day(sdf, date)
        day = day_cache[sym]
        if day.empty:
            continue
        upto = day[day["tod"] <= asof_tod]
        if upto.empty:
            continue
        row = upto.iloc[-1]
        if str(row["tod"]) < cfg.window_start or str(row["tod"]) > cfg.window_end:
            continue
        fp = float(row["from_prev"]) if np.isfinite(row["from_prev"]) else float("nan")
        if not np.isfinite(fp):
            continue
        if direction == "UP" and fp < cfg.fp_gate:
            continue
        if direction == "DN" and fp > -cfg.fp_gate:
            continue
        if cfg.require_above_open:
            if direction == "UP" and float(row["close"]) < float(row["day_open"]):
                continue
            if direction == "DN" and float(row["close"]) > float(row["day_open"]):
                continue

        # accel: fp now vs fp N minutes earlier
        ts = pd.Timestamp(row["_ts"])
        prev_ts = ts - pd.Timedelta(minutes=int(cfg.accel_minutes))
        prev = upto[upto["_ts"] <= prev_ts]
        fp_prev = float(prev.iloc[-1]["from_prev"]) if len(prev) else fp
        accel = fp - fp_prev

        cum = float(row["cum_dvol"])
        med = tod_median_by_sym.get(sym, {}).get(str(row["tod"]))
        vol_x = (cum / med) if med and med > 0 else 1.0

        # morning reclaim: how much fp recovered from session min fp so far
        min_fp = float(np.nanmin(upto["from_prev"].astype(float).to_numpy()))
        if direction == "UP":
            reclaim = max(0.0, fp - min_fp) if np.isfinite(min_fp) else 0.0
        else:
            reclaim = max(0.0, min_fp - fp) if np.isfinite(min_fp) else 0.0  # how far extended down

        qqq_div = (fp - qfp) if qfp is not None else 0.0

        raw.append(
            {
                "symbol": sym,
                "fp": fp,
                "vol_x": vol_x,
                "accel": accel if direction == "UP" else -accel,
                "qqq_div": qqq_div if direction == "UP" else -qqq_div,
                "reclaim": reclaim,
                "asof": ts,
                "tod": str(row["tod"]),
            }
        )

    if not raw:
        return []

    # relative strength vs pool median fp (signed for direction)
    fps = np.array([r["fp"] for r in raw], dtype=float)
    med_fp = float(np.nanmedian(fps))
    for r in raw:
        r["rs"] = (r["fp"] - med_fp) if direction == "UP" else (med_fp - r["fp"])

    keys = ["vol_x", "fp", "accel", "rs", "qqq_div", "reclaim"]
    # for DN, use -fp in z so more negative ranks higher
    mats = {}
    for k in keys:
        if k == "fp":
            mats[k] = np.array(
                [r["fp"] if direction == "UP" else -r["fp"] for r in raw], dtype=float
            )
        else:
            mats[k] = np.array([r[k] for r in raw], dtype=float)

    z = {k: _winsor_z(mats[k]) for k in keys}
    w = cfg.weights
    wsum = sum(float(w.get(k, 0.0)) for k in keys) or 1.0
    out: list[FactorSnap] = []
    for i, r in enumerate(raw):
        score = 0.0
        for k in keys:
            score += float(w.get(k, 0.0)) / wsum * float(z[k][i])
        out.append(
            FactorSnap(
                symbol=str(r["symbol"]),
                date=str(date),
                direction=direction,
                asof=pd.Timestamp(r["asof"]),
                tod=str(r["tod"]),
                fp=float(r["fp"]),
                vol_x=float(r["vol_x"]),
                accel=float(r["accel"]),
                rs=float(r["rs"]),
                qqq_div=float(r["qqq_div"]),
                reclaim=float(r["reclaim"]),
                score=float(score),
            )
        )
    out.sort(key=lambda s: (-s.score, s.symbol))
    ranked = [
        FactorSnap(
            symbol=s.symbol,
            date=s.date,
            direction=s.direction,
            asof=s.asof,
            tod=s.tod,
            fp=s.fp,
            vol_x=s.vol_x,
            accel=s.accel,
            rs=s.rs,
            qqq_div=s.qqq_div,
            reclaim=s.reclaim,
            score=s.score,
            rank=i + 1,
        )
        for i, s in enumerate(out)
    ]
    return ranked


def order_events_topk_then_mf(
    top2: pd.DataFrame,
    all_first: pd.DataFrame,
    stock_by: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    cfg: MultiFactorConfig | None = None,
    tod_median_by_sym_date: dict[tuple[str, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Earliest TopK first, then remaining first-Rule-A by multi-factor score.

    Causal: each residual row scored at its own ``sig_ts`` clock.
    """
    cfg = cfg or MultiFactorConfig()
    if all_first is None or len(all_first) == 0:
        return top2.copy() if top2 is not None else all_first
    if top2 is None or len(top2) == 0:
        top2 = all_first.iloc[0:0].copy()

    top_keys = {
        (str(r.date), str(r.symbol).upper(), str(r.dir).upper())
        for r in top2.itertuples(index=False)
    }
    parts: list[pd.DataFrame] = []
    dates = sorted(set(all_first["date"].astype(str).unique()) | set(top2["date"].astype(str).unique()))
    for date in dates:
        head = top2[top2["date"].astype(str) == date].sort_values(["sig_ts", "symbol"])
        rest = all_first[all_first["date"].astype(str) == date].copy()
        if len(rest):
            mask = [
                (str(d), str(s).upper(), str(di).upper()) not in top_keys
                for d, s, di in zip(rest["date"], rest["symbol"], rest["dir"])
            ]
            rest = rest.loc[mask].copy()
        scored_rows: list[pd.Series] = []
        day_cache: dict[str, pd.DataFrame] = {}
        # per-date volume medians (shared across residual rows)
        med: dict[str, dict[str, float]] = {}
        if tod_median_by_sym_date is not None:
            for sym in symbols:
                med[sym] = tod_median_by_sym_date.get((sym, str(date)), {})
        else:
            for sym in symbols:
                sdf = stock_by.get(sym)
                if sdf is None:
                    med[sym] = {}
                else:
                    med[sym] = build_tod_median_curve(
                        sdf, before_date=str(date), lookback_days=cfg.lookback_days
                    )
        if len(rest):
            for _, row in rest.iterrows():
                tod = pd.Timestamp(row["sig_ts"]).strftime("%H:%M")
                direction = str(row["dir"]).upper()
                if direction not in {"UP", "DN"}:
                    continue
                ranked = score_universe_at(
                    stock_by,
                    date=str(date),
                    symbols=symbols,
                    asof_tod=tod,
                    direction=direction,  # type: ignore[arg-type]
                    cfg=cfg,
                    tod_median_by_sym=med,
                    day_cache=day_cache,
                )
                sc = -1e9
                for s in ranked:
                    if s.symbol == str(row["symbol"]):
                        sc = float(s.score)
                        break
                rr = row.copy()
                rr["_mf_score"] = sc
                scored_rows.append(rr)
            if scored_rows:
                tail = pd.DataFrame(scored_rows)
                tail = tail.sort_values(["_mf_score", "sig_ts", "symbol"], ascending=[False, True, True])
            else:
                tail = rest.iloc[0:0]
        else:
            tail = rest
        chunk = pd.concat([head, tail], ignore_index=True)
        parts.append(chunk)
    if not parts:
        return top2.copy()
    out = pd.concat(parts, ignore_index=True)
    if "_mf_score" in out.columns:
        out = out.drop(columns=["_mf_score"])
    return out


def _day_clocks(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: MultiFactorConfig,
    step_minutes: int = 1,
    day_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    day_cache = day_cache if day_cache is not None else {}
    sample = None
    for s in symbols:
        if s not in stock_by:
            continue
        if s not in day_cache:
            day_cache[s] = prepare_day(stock_by[s], date)
        if sample is None and not day_cache[s].empty:
            sample = day_cache[s]
    if sample is None or sample.empty:
        return [], day_cache
    clocks = sample[
        (sample["tod"] >= cfg.window_start) & (sample["tod"] <= cfg.window_end)
    ]["tod"].tolist()
    if step_minutes > 1:
        clocks = clocks[:: int(step_minutes)]
    return [str(t) for t in clocks], day_cache


def first_top2_entry(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    symbol: str,
    direction: Direction,
    cfg: MultiFactorConfig,
    tod_median_by_sym: dict[str, dict[str, float]],
    step_minutes: int = 1,
) -> FactorSnap | None:
    """Earliest asof where ``symbol`` is in multi-factor top2 for direction."""
    clocks, day_cache = _day_clocks(
        stock_by, date=date, symbols=symbols, cfg=cfg, step_minutes=step_minutes
    )
    for tod in clocks:
        ranked = score_universe_at(
            stock_by,
            date=date,
            symbols=symbols,
            asof_tod=str(tod),
            direction=direction,
            cfg=cfg,
            tod_median_by_sym=tod_median_by_sym,
            day_cache=day_cache,
        )
        for s in ranked:
            if s.symbol == symbol and s.rank <= 2:
                return s
    return None


def iter_first_top2_entries(
    stock_by: dict[str, pd.DataFrame],
    *,
    date: str,
    symbols: list[str],
    cfg: MultiFactorConfig,
    tod_median_by_sym: dict[str, dict[str, float]],
    step_minutes: int = 1,
    directions: tuple[Direction, ...] = ("UP", "DN"),
    stable_bars: int = 1,
) -> list[FactorSnap]:
    """Chronological first-time top2 memberships (each symbol×direction once).

    ``stable_bars``>1 requires the name to remain in top2 for that many consecutive
    clocks before the entry is emitted (still causal; entry clock is the confirm bar).
    """
    clocks, day_cache = _day_clocks(
        stock_by, date=date, symbols=symbols, cfg=cfg, step_minutes=step_minutes
    )
    seen: set[tuple[str, str]] = set()
    streak: dict[tuple[str, str], int] = {}
    out: list[FactorSnap] = []
    need = max(1, int(stable_bars))
    for tod in clocks:
        for direction in directions:
            ranked = score_universe_at(
                stock_by,
                date=date,
                symbols=symbols,
                asof_tod=str(tod),
                direction=direction,
                cfg=cfg,
                tod_median_by_sym=tod_median_by_sym,
                day_cache=day_cache,
            )
            in_top = {s.symbol: s for s in ranked if s.rank <= 2}
            for sym in list(symbols):
                key = (sym, direction)
                if key in seen:
                    continue
                if sym in in_top:
                    streak[key] = streak.get(key, 0) + 1
                    if streak[key] >= need:
                        seen.add(key)
                        out.append(in_top[sym])
                else:
                    streak[key] = 0
    return out
