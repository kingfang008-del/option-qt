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


def resolve_mf_fast_window(cfg_signal: dict[str, Any] | None = None, *, default: int = 3) -> int:
    """Fast companion window length (minutes). Prefers ``mf_fast_window`` over ``mf_confirm_bars``."""
    cfg = cfg_signal or {}
    raw = cfg.get("mf_fast_window", cfg.get("mf_confirm_bars", default))
    return max(1, int(raw if raw is not None else default))


def attach_mf_features(
    df: pd.DataFrame,
    mf_window: int = 10,
    vol_ma_window: int = 20,
    *,
    mf_confirm_bars: int = 3,
    mf_fast_window: int | None = None,
) -> pd.DataFrame:
    """Causal mf10 / streak / from_prev / vol_z on RTH 1m bars (left-label stock OK).

    Also builds a fast companion window ``mf_fast`` / ``mf_short`` (last N minutes of
    net$) used for early entry when ``early_on_mf_fast`` is enabled, or as a late-fire
    confirm when ``require_mf_short_align`` is enabled.
    """
    out = df.copy()
    hl = (out["high"] - out["low"]).replace(0, np.nan)
    buy = ((out["close"] - out["low"]) / hl).fillna(0.5) * out["volume"]
    sell = ((out["high"] - out["close"]) / hl).fillna(0.5) * out["volume"]
    out["net$"] = (buy - sell) * out["close"]
    out["mf10"] = out.groupby("date")["net$"].transform(lambda s: s.rolling(mf_window, min_periods=mf_window).sum())
    fast_n = max(1, int(mf_fast_window if mf_fast_window is not None else mf_confirm_bars))
    out["mf_fast"] = out.groupby("date")["net$"].transform(
        lambda s: s.rolling(fast_n, min_periods=fast_n).sum()
    )
    # Backward-compatible alias used by older profiles / diagnostics.
    out["mf_short"] = out["mf_fast"]
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


def _mf_fast_col(w: pd.DataFrame) -> str | None:
    if "mf_fast" in w.columns:
        return "mf_fast"
    if "mf_short" in w.columns:
        return "mf_short"
    return None


def _rule_a_mask(
    w: pd.DataFrame,
    *,
    direction: str,
    streak_min: int,
    from_prev_abs: float,
    vol_z_min: float,
    streak_max: int | None,
    require_mf_short_align: bool,
    early_on_mf_fast: bool = False,
    streak_min_fast: int | None = None,
) -> pd.Series:
    """Standard path: streak>=streak_min. Early path: streak_min_fast..streak_min-1 + mf_fast align."""
    fast_col = _mf_fast_col(w)
    streak_col = "streak_up" if direction == "UP" else "streak_dn"
    if direction == "UP":
        base = (
            (w["cum"] > 0)
            & (w["from_prev"] >= from_prev_abs)
            & (w["vol_z"] >= vol_z_min)
        )
        fast_ok = (w[fast_col] > 0) if fast_col else pd.Series(False, index=w.index)
    else:
        base = (
            (w["cum"] < 0)
            & (w["from_prev"] <= -from_prev_abs)
            & (w["vol_z"] >= vol_z_min)
        )
        fast_ok = (w[fast_col] < 0) if fast_col else pd.Series(False, index=w.index)

    streak = w[streak_col]
    standard = streak >= int(streak_min)
    if streak_max is not None:
        standard = standard & (streak <= int(streak_max))
    if require_mf_short_align:
        standard = standard & fast_ok

    early = pd.Series(False, index=w.index)
    if early_on_mf_fast and streak_min_fast is not None:
        s_fast = int(streak_min_fast)
        s_full = int(streak_min)
        if 1 <= s_fast < s_full and fast_col is not None:
            early = (streak >= s_fast) & (streak < s_full) & fast_ok
            if streak_max is not None:
                early = early & (streak <= int(streak_max))

    return base & (standard | early)


def _rule_a_kwargs_from_cfg(cfg_signal: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg_signal or {}
    streak_max = cfg.get("streak_max")
    streak_min_fast = cfg.get("streak_min_fast")
    return {
        "window_start": str(cfg.get("window_start", "10:30")),
        "window_end": str(cfg.get("window_end", "14:00")),
        "streak_min": int(cfg.get("streak_min", 8)),
        "from_prev_abs": float(cfg.get("from_prev_abs", 0.02)),
        "vol_z_min": float(cfg.get("vol_z_min", 1.0)),
        "streak_max": int(streak_max) if streak_max is not None else None,
        "require_mf_short_align": bool(cfg.get("require_mf_short_align", False)),
        "early_on_mf_fast": bool(cfg.get("early_on_mf_fast", False)),
        "streak_min_fast": int(streak_min_fast) if streak_min_fast is not None else None,
    }


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
    early_on_mf_fast: bool = False,
    streak_min_fast: int | None = None,
) -> dict[str, Any] | None:
    """First Rule-A fire of the day (UP or DN, earlier wins)."""
    w = g[(g["tod"] >= window_start) & (g["tod"] <= window_end)].sort_values("timestamp")
    common = dict(
        streak_min=streak_min,
        from_prev_abs=from_prev_abs,
        vol_z_min=vol_z_min,
        streak_max=streak_max,
        require_mf_short_align=require_mf_short_align,
        early_on_mf_fast=early_on_mf_fast,
        streak_min_fast=streak_min_fast,
    )
    up = w[_rule_a_mask(w, direction="UP", **common)]
    dn = w[_rule_a_mask(w, direction="DN", **common)]
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
    bar = up.iloc[0] if d == "UP" else dn.iloc[0]
    out: dict[str, Any] = {"dir": d, "sig_ts": ts, "spot": spot}
    if "from_prev" in bar.index and pd.notna(bar["from_prev"]):
        out["from_prev"] = float(bar["from_prev"])
    if "mf10" in bar.index and pd.notna(bar["mf10"]):
        out["mf10"] = float(bar["mf10"])
    if d == "UP" and "streak_up" in bar.index and pd.notna(bar["streak_up"]):
        out["streak"] = int(bar["streak_up"])
    elif d == "DN" and "streak_dn" in bar.index and pd.notna(bar["streak_dn"]):
        out["streak"] = int(bar["streak_dn"])
    return out


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
    early_on_mf_fast: bool = False,
    streak_min_fast: int | None = None,
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
            early_on_mf_fast=early_on_mf_fast,
            streak_min_fast=streak_min_fast,
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


def tod_mf_z_ok(
    stock_df: pd.DataFrame | None,
    *,
    asof_ts: pd.Timestamp,
    direction: str,
    lookback_days: int = 20,
    z_min: float = 2.0,
    min_periods: int = 5,
    mf_col: str = "mf10",
) -> tuple[bool, float | None]:
    """Time-of-day normalized mf10 z-score gate (inprove2 §3).

    For the bar's clock minute ``HH:MM``, compare current ``mf10`` to the
    causal rolling mean/std of the same minute over the prior ``lookback_days``
    sessions. Require ``Z > z_min`` (UP) or ``Z < -z_min`` (DN).

    Missing history → allow (fail-open). Caches ``_tod_mf_z`` on ``stock_df``.
    """
    if stock_df is None or stock_df.empty or direction not in ("UP", "DN"):
        return True, None
    if mf_col not in stock_df.columns or "tod" not in stock_df.columns:
        return True, None
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)

    cache_key = f"_tod_mf_z_{mf_col}_{int(lookback_days)}_{int(min_periods)}"
    if cache_key not in stock_df.columns:
        z = pd.Series(np.nan, index=stock_df.index, dtype=float)
        for _, g in stock_df.groupby("tod", sort=False):
            # one row per session at this clock minute
            g2 = g.sort_values("timestamp")
            if "date" in g2.columns:
                g2 = g2.drop_duplicates("date", keep="last")
            mf = pd.to_numeric(g2[mf_col], errors="coerce")
            mu = mf.shift(1).rolling(int(lookback_days), min_periods=int(min_periods)).mean()
            sd = mf.shift(1).rolling(int(lookback_days), min_periods=int(min_periods)).std(ddof=1)
            zz = (mf - mu) / (sd + 1e-6)
            z.loc[g2.index] = zz.to_numpy()
        stock_df[cache_key] = z

    mask = stock_df["timestamp"] <= asof
    if not mask.any():
        return True, None
    row = stock_df.loc[mask].iloc[-1]
    zv = float(row[cache_key]) if np.isfinite(row[cache_key]) else float("nan")
    if not np.isfinite(zv):
        return True, None
    if direction == "UP":
        return bool(zv >= float(z_min)), zv
    return bool(zv <= -float(z_min)), zv


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


def _ols_beta(x: np.ndarray, y: np.ndarray, *, min_n: int = 40) -> float | None:
    """OLS slope of y on x; None if underpowered / degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < int(min_n):
        return None
    x = x[m]
    y = y[m]
    vx = float(np.var(x))
    if vx < 1e-18:
        return None
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return cov / vx


def rolling_idio_beta(
    stock_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    *,
    asof_date: str,
    n_days: int = 5,
    on: str = "ret",
    min_n: int = 40,
) -> float | None:
    """Causal beta of stock vs QQQ over prior complete sessions (excludes ``asof_date``)."""
    if stock_df is None or qqq_df is None or stock_df.empty or qqq_df.empty:
        return None
    on = str(on or "ret").strip().lower()
    dates = sorted(str(d) for d in stock_df["date"].unique() if str(d) < str(asof_date))
    use = dates[-int(n_days) :] if n_days > 0 else dates
    if len(use) < max(2, int(n_days) // 2):
        return None
    s = stock_df[stock_df["date"].astype(str).isin(use)]
    q = qqq_df[qqq_df["date"].astype(str).isin(use)]
    if s.empty or q.empty:
        return None
    if on in ("mf", "mf10"):
        if "mf10" not in s.columns or "mf10" not in q.columns:
            return None
        merged = s[["timestamp", "mf10"]].merge(
            q[["timestamp", "mf10"]].rename(columns={"mf10": "mf_q"}),
            on="timestamp",
            how="inner",
        )
        return _ols_beta(merged["mf_q"].to_numpy(), merged["mf10"].to_numpy(), min_n=min_n)
    # default: 1m return beta
    s = s.sort_values("timestamp").copy()
    q = q.sort_values("timestamp").copy()
    if "ret1" not in s.columns:
        s["ret1"] = s.groupby("date")["close"].pct_change()
    if "ret1" not in q.columns:
        q["ret1"] = q.groupby("date")["close"].pct_change()
    merged = s[["timestamp", "ret1"]].merge(
        q[["timestamp", "ret1"]].rename(columns={"ret1": "ret_q"}),
        on="timestamp",
        how="inner",
    )
    return _ols_beta(merged["ret_q"].to_numpy(), merged["ret1"].to_numpy(), min_n=min_n)


def mf_idio_ok(
    stock_df: pd.DataFrame | None,
    qqq_df: pd.DataFrame | None,
    *,
    date: str,
    asof_ts: pd.Timestamp,
    direction: str,
    mode: str = "pos",
    min_frac: float = 0.0,
    beta_days: int = 5,
    beta_on: str = "ret",
    beta: float | None = None,
    block_missing: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Idiosyncratic residual money-flow gate (improve6 scheme 1).

    ``mf_idio = mf_stock - beta * mf_qqq`` at/before ``asof_ts``.

    ``mode``:
      - ``off``: always allow
      - ``pos``: signed residual must be > 0
      - ``frac``: signed residual >= ``min_frac`` * |mf_stock|
      - ``diff_pos``: signed (mf_stock - mf_qqq) > 0 (no beta)
    """
    mode = str(mode or "off").strip().lower()
    meta: dict[str, Any] = {"mode": mode}
    if mode in {"", "off", "none", "false", "0"}:
        return True, meta
    if stock_df is None or qqq_df is None or stock_df.empty or qqq_df.empty:
        return (not block_missing), {**meta, "reason": "mf_idio_missing_frame"}
    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    sday = stock_df[stock_df["date"].astype(str) == str(date)]
    qday = qqq_df[qqq_df["date"].astype(str) == str(date)]
    if sday.empty or qday.empty:
        return (not block_missing), {**meta, "reason": "mf_idio_missing_day"}
    sbar = sday[sday["timestamp"] <= asof].tail(1)
    qbar = qday[qday["timestamp"] <= asof].tail(1)
    if sbar.empty or qbar.empty:
        return (not block_missing), {**meta, "reason": "mf_idio_missing_bar"}
    mf_s = float(sbar.iloc[0]["mf10"]) if "mf10" in sbar.columns else float("nan")
    mf_q = float(qbar.iloc[0]["mf10"]) if "mf10" in qbar.columns else float("nan")
    if not (np.isfinite(mf_s) and np.isfinite(mf_q)):
        return (not block_missing), {**meta, "reason": "mf_idio_nan"}
    dir_sign = 1.0 if str(direction).upper() == "UP" else -1.0
    meta.update({"mf_s": mf_s, "mf_q": mf_q, "dir_sign": dir_sign})

    if mode in {"diff_pos", "diff", "mf_diff"}:
        resid = mf_s - mf_q
        signed = resid * dir_sign
        meta.update({"mf_idio": resid, "mf_idio_signed": signed, "beta": None})
        ok = bool(signed > 0)
        meta["reason"] = "ok" if ok else "mf_idio_diff"
        return ok, meta

    b = beta
    if b is None:
        b = rolling_idio_beta(
            stock_df, qqq_df, asof_date=str(date), n_days=int(beta_days), on=beta_on
        )
    if b is None or not np.isfinite(b):
        return (not block_missing), {**meta, "reason": "mf_idio_beta_missing"}
    resid = mf_s - float(b) * mf_q
    signed = resid * dir_sign
    meta.update({"mf_idio": resid, "mf_idio_signed": signed, "beta": float(b)})
    if mode in {"pos", "positive", "gt0"}:
        ok = bool(signed > 0)
        meta["reason"] = "ok" if ok else "mf_idio_pos"
        return ok, meta
    if mode in {"frac", "min_frac", "fraction"}:
        thr = float(min_frac) * abs(mf_s)
        ok = bool(signed >= thr)
        meta["reason"] = "ok" if ok else "mf_idio_frac"
        meta["thr"] = thr
        return ok, meta
    return True, {**meta, "reason": "mf_idio_unknown_mode"}


def build_all_first_rule_a_signals(
    stock_by_symbol: dict[str, pd.DataFrame],
    cfg_signal: dict[str, Any],
) -> pd.DataFrame:
    """Per-symbol first Rule-A fire each day (no TopK trim)."""
    rule_kw = _rule_a_kwargs_from_cfg(cfg_signal)
    rows = []
    for sym, df in stock_by_symbol.items():
        if df.empty:
            continue
        for date, g in df.groupby("date"):
            hit = first_rule_a_day(g, **rule_kw)
            if hit is None:
                continue
            rows.append({"date": date, "symbol": sym, **hit})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "dir", "sig_ts", "spot", "rank"])
    sig = pd.DataFrame(rows).sort_values(["date", "sig_ts", "symbol"])
    sig["rank"] = sig.groupby("date").cumcount() + 1
    return sig.reset_index(drop=True)


def build_topk_signals(
    stock_by_symbol: dict[str, pd.DataFrame],
    cfg_signal: dict[str, Any],
) -> pd.DataFrame:
    """Per-symbol first Rule-A, then TopK earliest within each date."""
    sig = build_all_first_rule_a_signals(stock_by_symbol, cfg_signal)
    if sig.empty:
        return sig
    top_k = int(cfg_signal.get("top_k", 2))
    return sig[sig["rank"] <= top_k].reset_index(drop=True)

class StreamSignalState:
    """Causal per-symbol Rule-A state for streaming parity."""

    def __init__(self, symbol: str, cfg_signal: dict[str, Any], *, emit_all: bool = False):
        self.symbol = symbol
        self.cfg = cfg_signal
        self.emit_all = emit_all  # True → every Rule-A bar (reentry); False → first only
        self.mf_window = int(cfg_signal.get("mf_window", 10))
        self.vol_ma_window = int(cfg_signal.get("vol_ma_window", 20))
        self.mf_confirm_bars = resolve_mf_fast_window(cfg_signal)
        self.mf_fast_window = self.mf_confirm_bars
        self.bars: list[dict[str, Any]] = []
        self.prev_close: float | None = None
        self.day_open: float | None = None
        self.date: str | None = None
        self.cum = 0.0
        self.mf10 = float("nan")
        self.mf_fast = float("nan")
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
            self.mf_fast = float("nan")
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

        # mf10 + fast companion window
        nets = [b["net$"] for b in self.bars]
        mf10 = float(np.sum(nets[-self.mf_window:])) if len(nets) >= self.mf_window else np.nan
        self.mf10 = mf10
        self.mf_fast = (
            float(np.sum(nets[-self.mf_fast_window:]))
            if len(nets) >= self.mf_fast_window
            else np.nan
        )
        self.mf_short = self.mf_fast
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
        streak_min_fast_raw = self.cfg.get("streak_min_fast")
        streak_min_fast = int(streak_min_fast_raw) if streak_min_fast_raw is not None else None
        early_on = bool(self.cfg.get("early_on_mf_fast", False))
        fp = float(self.cfg.get("from_prev_abs", 0.02))
        vz_min = float(self.cfg.get("vol_z_min", 1.0))
        need_short = bool(self.cfg.get("require_mf_short_align", False))

        def _gates(streak: int, *, want_up: bool) -> bool:
            if want_up:
                if not (self.cum > 0 and from_prev >= fp and vol_z >= vz_min):
                    return False
            else:
                if not (self.cum < 0 and from_prev <= -fp and vol_z >= vz_min):
                    return False
            if streak_max_i is not None and streak > streak_max_i:
                return False
            return True

        def _fast_aligned(want_up: bool) -> bool:
            if not np.isfinite(self.mf_fast):
                return False
            return self.mf_fast > 0 if want_up else self.mf_fast < 0

        direction = None
        # Standard path at full streak_min.
        if self.streak_up >= streak_min and _gates(self.streak_up, want_up=True):
            if need_short and not _fast_aligned(True):
                pass
            else:
                direction = "UP"
        elif self.streak_dn >= streak_min and _gates(self.streak_dn, want_up=False):
            if need_short and not _fast_aligned(False):
                pass
            else:
                direction = "DN"
        # Early path: mf10 streak building + fast window already aligned.
        elif (
            early_on
            and streak_min_fast is not None
            and 1 <= streak_min_fast < streak_min
        ):
            if (
                streak_min_fast <= self.streak_up < streak_min
                and _gates(self.streak_up, want_up=True)
                and _fast_aligned(True)
            ):
                direction = "UP"
            elif (
                streak_min_fast <= self.streak_dn < streak_min
                and _gates(self.streak_dn, want_up=False)
                and _fast_aligned(False)
            ):
                direction = "DN"
        if direction is None:
            return None
        fire = {"date": date, "symbol": self.symbol, "dir": direction, "sig_ts": ts, "spot": c}
        if not self.fired_today:
            self.first_fire = fire
        self.fired_today = True
        return fire