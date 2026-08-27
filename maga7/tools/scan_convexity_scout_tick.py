#!/usr/bin/env python3
"""Causal Mag7 downside convexity scout on stock 1s + S3 option trade ticks.

Research only: S3 tick contains prints, not bid/ask. The scan uses robust
per-second median trade prices and reports upper-bound convexity probabilities;
quote validation is still required before any live wiring.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.contract_select import trading_dte
from maga7.common.option_flow import (
    DEFAULT_TICK_ROOT,
    load_option_tick_day,
    prepare_option_flow_day,
    put_flow_features_at,
    tick_dates,
)
from maga7.common.replay import to_ny
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class SignalSpec:
    ret3: float
    ret15: float
    accel_mult: float
    peer_min: int
    min_vol_z: float

    @property
    def key(self) -> str:
        return (
            f"r3_{self.ret3:.4f}_r15_{self.ret15:.4f}_"
            f"a{self.accel_mult:.1f}_p{self.peer_min}_vz{self.min_vol_z:.1f}"
        )


def _grid(date: str, start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(f"{date} {start}:00", tz=NY),
        pd.Timestamp(f"{date} {end}:00", tz=NY),
        freq="1s",
        inclusive="left",
    )


def _stock_features(
    stock_root: Path,
    symbols: list[str],
    date: str,
    *,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    grid = _grid(date, start, end)
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = load_stock_1s_day(stock_root, symbol, date)
        if raw.empty:
            continue
        local = raw.set_index("timestamp").reindex(grid)
        close = local["close"].astype(float).ffill(limit=2)
        volume = local.get("volume", pd.Series(0.0, index=grid)).fillna(0.0).astype(float)
        r3 = close / close.shift(3) - 1.0
        r15 = close / close.shift(15) - 1.0
        r30 = close / close.shift(30) - 1.0
        prior12 = close.shift(3) / close.shift(15) - 1.0
        current_rate = -r3 / 3.0
        prior_rate = -prior12 / 12.0
        vol_mean = volume.rolling(120, min_periods=60).mean()
        vol_std = volume.rolling(120, min_periods=60).std(ddof=0)
        vol_z = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
        out[symbol] = pd.DataFrame(
            {
                "close": close,
                "r3": r3,
                "r15": r15,
                "r30": r30,
                "current_rate": current_rate,
                "prior_rate": prior_rate,
                "vol_z": vol_z,
            },
            index=grid,
        )
    return out


def _day_signals(
    features: dict[str, pd.DataFrame],
    date: str,
    specs: list[SignalSpec],
    *,
    scan_start: str,
    scan_end: str,
) -> list[dict[str, Any]]:
    if not features:
        return []
    symbols = sorted(features)
    index = next(iter(features.values())).index
    peer = pd.concat(
        [(features[symbol]["r30"] < 0).rename(symbol) for symbol in symbols],
        axis=1,
    ).sum(axis=1)
    lo = pd.Timestamp(f"{date} {scan_start}:00", tz=NY)
    hi = pd.Timestamp(f"{date} {scan_end}:00", tz=NY)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        candidates = []
        for symbol in symbols:
            feat = features[symbol]
            accel_ok = feat["current_rate"] >= (
                feat["prior_rate"].clip(lower=0.0) * float(spec.accel_mult)
            )
            gate = (
                (feat.index >= lo)
                & (feat.index < hi)
                & (feat["r3"] <= -float(spec.ret3))
                & (feat["r15"] <= -float(spec.ret15))
                & (feat["current_rate"] > 0)
                & accel_ok
                & (feat["vol_z"] >= float(spec.min_vol_z))
                & (peer >= int(spec.peer_min))
            )
            hit_idx = np.flatnonzero(gate.to_numpy())
            if not len(hit_idx):
                continue
            i = int(hit_idx[0])
            r = feat.iloc[i]
            accel_ratio = float(r["current_rate"]) / max(
                float(r["prior_rate"]), 1e-8
            )
            score = (
                abs(float(r["r3"])) * 10_000.0
                + abs(float(r["r15"])) * 5_000.0
                + min(accel_ratio, 10.0) * 5.0
                + float(peer.iloc[i]) * 2.0
                + max(float(r["vol_z"]), 0.0)
            )
            candidates.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "signal_ts": index[i],
                    "spot": float(r["close"]),
                    "r3": float(r["r3"]),
                    "r15": float(r["r15"]),
                    "r30": float(r["r30"]),
                    "accel_ratio": accel_ratio,
                    "vol_z": float(r["vol_z"]),
                    "peer_n": int(peer.iloc[i]),
                    "score": float(score),
                    "spec_key": spec.key,
                }
            )
        if candidates:
            earliest = min(row["signal_ts"] for row in candidates)
            same_clock = [row for row in candidates if row["signal_ts"] == earliest]
            rows.append(max(same_clock, key=lambda row: row["score"]))
    return rows


def _occ_meta(ticker: str, date: str) -> tuple[str, int, float] | None:
    clean = str(ticker).replace("O:", "")
    match = OCC.match(clean)
    if not match or match.group("right") != "P":
        return None
    expiry = "20" + match.group("expiry")
    try:
        dte = int(trading_dte(expiry, date))
    except Exception:
        return None
    strike = int(match.group("strike")) / 1000.0
    return clean, dte, float(strike)


def _second_median_path(
    ts_ns: np.ndarray, prices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    seconds = np.asarray(ts_ns, dtype=np.int64) // 1_000_000_000
    frame = pd.DataFrame({"second": seconds, "price": prices})
    grouped = frame.groupby("second", sort=True)["price"].median()
    return (
        grouped.index.to_numpy(dtype=np.int64) * 1_000_000_000,
        grouped.to_numpy(dtype=np.float64),
    )


def _prepare_contract_paths(
    paths: dict[str, tuple[np.ndarray, np.ndarray]],
    date: str,
) -> list[dict[str, Any]]:
    prepared = []
    for ticker, (ts_ns, prices) in paths.items():
        meta = _occ_meta(ticker, date)
        if meta is None:
            continue
        clean, dte, strike = meta
        if dte not in {0, 1}:
            continue
        robust_ts, robust_px = _second_median_path(ts_ns, prices)
        prepared.append(
            {
                "ticker": clean,
                "dte": dte,
                "strike": strike,
                "raw_ts": ts_ns,
                "robust_ts": robust_ts,
                "robust_px": robust_px,
            }
        )
    return prepared


def _entry_snapshot(
    raw_ts: np.ndarray,
    robust_ts: np.ndarray,
    robust_px: np.ndarray,
    signal_ts: pd.Timestamp,
) -> dict[str, float] | None:
    t0 = int(to_ny(signal_ts).value)
    i = int(np.searchsorted(raw_ts, t0, side="left"))
    if i >= len(raw_ts) or (int(raw_ts[i]) - t0) / 1e9 > 5.0:
        return None
    prior_lo = int(np.searchsorted(raw_ts, t0 - 60_000_000_000, side="left"))
    prior_hi = int(np.searchsorted(raw_ts, t0, side="left"))
    j = int(np.searchsorted(robust_ts, t0 // 1_000_000_000 * 1_000_000_000, side="left"))
    if j >= len(robust_ts) or (int(robust_ts[j]) - t0) / 1e9 > 5.0:
        return None
    entry = float(robust_px[j]) * 1.01
    if not (0.03 <= entry <= 20.0):
        return None
    return {
        "entry": entry,
        "entry_i": float(j),
        "prior_prints_60": float(max(0, prior_hi - prior_lo)),
        "entry_lag_sec": float((int(robust_ts[j]) - t0) / 1e9),
    }


def _adaptive_gap(signal: dict[str, Any]) -> float:
    impulse = max(abs(float(signal["r3"])), abs(float(signal["r15"])) * 0.6)
    if impulse >= 0.010:
        return 0.030
    if impulse >= 0.007:
        return 0.020
    if impulse >= 0.004:
        return 0.010
    return 0.005


def _flow_snapshot(
    flow: dict[str, Any] | None,
    signal_ts: pd.Timestamp,
) -> dict[str, float]:
    if not flow:
        return {}
    i = int(
        np.searchsorted(
            flow["ts_ns"],
            int(to_ny(signal_ts).value),
            side="right",
        )
        - 1
    )
    if i < 0:
        return {}
    out: dict[str, float] = {}
    for window_sec in (30, 60, 120):
        feat = put_flow_features_at(flow, i=i, window_sec=window_sec)
        if feat is None:
            continue
        share, z, put_v, call_v = feat
        out.update(
            {
                f"put_share_{window_sec}": share,
                f"put_z_{window_sec}": z,
                f"put_v_{window_sec}": put_v,
                f"call_v_{window_sec}": call_v,
            }
        )
    return out


def _select_contracts(
    prepared_paths: list[dict[str, Any]],
    signal: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    spot = float(signal["spot"])
    choices: list[dict[str, Any]] = []
    for path in prepared_paths:
        clean = str(path["ticker"])
        dte = int(path["dte"])
        strike = float(path["strike"])
        gap = (spot - strike) / spot
        if gap < -0.005 or gap > 0.05:
            continue
        snap = _entry_snapshot(
            path["raw_ts"],
            path["robust_ts"],
            path["robust_px"],
            signal["signal_ts"],
        )
        if snap is None or snap["prior_prints_60"] < 2:
            continue
        choices.append(
            {
                "ticker": clean,
                "dte": dte,
                "strike": strike,
                "gap": float(gap),
                "ts_ns": path["robust_ts"],
                "prices": path["robust_px"],
                **snap,
            }
        )
    if not choices:
        return {}

    def choose(target: float, *, cheap: bool) -> dict[str, Any] | None:
        pool = [
            row
            for row in choices
            if not cheap or float(row["entry"]) <= min(1.0, spot * 0.003)
        ]
        if not pool:
            return None
        return min(
            pool,
            key=lambda row: (
                abs(float(row["gap"]) - target)
                + 0.0015 * int(row["dte"])
                + (0.001 if float(row["prior_prints_60"]) < 5 else 0.0)
            ),
        )

    target = _adaptive_gap(signal)
    selected = {
        "ATM_LIQ": choose(0.0, cheap=False),
        "ADAPTIVE": choose(target, cheap=False),
        "ADAPTIVE_CHEAP": choose(target, cheap=True),
    }
    return {key: value for key, value in selected.items() if value is not None}


def _score_path(
    contract: dict[str, Any],
    signal_ts: pd.Timestamp,
    *,
    max_hold_sec: int,
) -> dict[str, Any]:
    ts_ns = contract["ts_ns"]
    prices = contract["prices"]
    entry = float(contract["entry"])
    i0 = int(contract["entry_i"])
    t0 = int(to_ny(signal_ts).value)
    end = min(
        t0 + int(max_hold_sec) * 1_000_000_000,
        int(to_ny(f"{to_ny(signal_ts).strftime('%Y-%m-%d')} 15:55:00").value),
    )
    i1 = int(np.searchsorted(ts_ns, end, side="right") - 1)
    if i1 <= i0:
        return {}
    sell = prices[i0 : i1 + 1] * 0.99
    rets = sell / entry - 1.0
    mfe = float(np.nanmax(rets))
    mae = float(np.nanmin(rets))
    close_ret = float(rets[-1])
    tp = 2.0  # 3x premium
    sl = -0.50
    hit = np.flatnonzero((rets >= tp) | (rets <= sl))
    exit_i = int(hit[0]) if len(hit) else len(rets) - 1
    return {
        "mfe": mfe,
        "mae": mae,
        "close_ret": close_ret,
        "rule_ret": float(rets[exit_i]),
        "rule_reason": (
            "tp3x"
            if rets[exit_i] >= tp
            else ("sl50" if rets[exit_i] <= sl else "max_hold")
        ),
        "rule_hold_sec": float(
            (int(ts_ns[i0 + exit_i]) - int(ts_ns[i0])) / 1e9
        ),
        "hit_2x": bool(mfe >= 1.0),
        "hit_3x": bool(mfe >= 2.0),
        "hit_5x": bool(mfe >= 4.0),
        "hit_10x": bool(mfe >= 9.0),
        "hit_20x": bool(mfe >= 19.0),
    }


def _summary(trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if trades.empty:
        return rows
    for (window, spec, policy), group in trades.groupby(
        ["window", "spec_key", "policy"], sort=False
    ):
        rows.append(
            {
                "window": window,
                "spec_key": spec,
                "policy": policy,
                "n": int(len(group)),
                "days": int(group["date"].nunique()),
                "mean_rule_ret": float(group["rule_ret"].mean()),
                "median_rule_ret": float(group["rule_ret"].median()),
                "win_rate": float((group["rule_ret"] > 0).mean()),
                "mean_mfe": float(group["mfe"].mean()),
                "p_2x": float(group["hit_2x"].mean()),
                "p_3x": float(group["hit_3x"].mean()),
                "p_5x": float(group["hit_5x"].mean()),
                "p_10x": float(group["hit_10x"].mean()),
                "p_20x": float(group["hit_20x"].mean()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    parser.add_argument("--start-date", default="2026-02-02")
    parser.add_argument("--end-date", default="2026-07-23")
    parser.add_argument("--scan-start", default="09:35")
    parser.add_argument("--scan-end", default="14:00")
    parser.add_argument("--max-hold-sec", type=int, default=7200)
    parser.add_argument("--tag", default="research_convexity_scout_tick_v1")
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    symbols = list(profile.get("symbols") or [])
    stock_root = Path(
        profile["_paths"].get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks"
    ).expanduser()
    tick_root = Path(args.tick_root)
    dates = [
        date
        for date in tick_dates(tick_root)
        if args.start_date <= date <= args.end_date
    ]
    specs = [
        SignalSpec(r3, r15, accel, peer, vol_z)
        for r3 in (0.0015, 0.0025, 0.0040)
        for r15 in (0.0030, 0.0050, 0.0080)
        for accel in (1.2, 1.8)
        for peer in (3, 5)
        for vol_z in (0.5, 1.5)
        if r15 >= r3
    ]
    print(
        f"convexity scout dates={len(dates)} specs={len(specs)} "
        f"symbols={len(symbols)} tick={tick_root}",
        flush=True,
    )

    signals: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        features = _stock_features(
            stock_root,
            symbols,
            date,
            start="09:30",
            end=args.scan_end,
        )
        signals.extend(
            _day_signals(
                features,
                date,
                specs,
                scan_start=args.scan_start,
                scan_end=args.scan_end,
            )
        )
        if i % 10 == 0:
            print(f"[signal] {i + 1}/{len(dates)} rows={len(signals)}", flush=True)

    unique: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for signal in signals:
        key = (
            signal["date"],
            signal["symbol"],
            int(to_ny(signal["signal_ts"]).value),
        )
        unique.setdefault(key, []).append(signal)
    print(f"signals={len(signals)} unique_events={len(unique)}; pricing", flush=True)

    priced: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    by_symbol_day: dict[tuple[str, str], list[tuple[tuple[str, str, int], dict[str, Any]]]] = {}
    for key, rows in unique.items():
        by_symbol_day.setdefault((key[0], key[1]), []).append((key, rows[0]))
    for i, ((date, symbol), events) in enumerate(by_symbol_day.items()):
        tick = load_option_tick_day(tick_root, symbol, date)
        if tick is None or tick.empty:
            continue
        if "correction" in tick.columns:
            tick = tick[pd.to_numeric(tick["correction"], errors="coerce").fillna(0) == 0]
        flow = prepare_option_flow_day(tick)
        paths = _paths_by_ticker(tick)
        prepared_paths = _prepare_contract_paths(paths, date)
        for key, signal in events:
            flow_meta = _flow_snapshot(flow, signal["signal_ts"])
            for policy, contract in _select_contracts(prepared_paths, signal).items():
                result = _score_path(
                    contract,
                    signal["signal_ts"],
                    max_hold_sec=args.max_hold_sec,
                )
                if result:
                    priced[(*key, policy)] = {
                        "policy": policy,
                        **flow_meta,
                        "ticker": contract["ticker"],
                        "dte": contract["dte"],
                        "strike": contract["strike"],
                        "otm_gap": contract["gap"],
                        "entry_price": contract["entry"],
                        "prior_prints_60": contract["prior_prints_60"],
                        "entry_lag_sec": contract["entry_lag_sec"],
                        **result,
                    }
        if i % 20 == 0:
            print(f"[price] {i + 1}/{len(by_symbol_day)} priced={len(priced)}", flush=True)

    trades: list[dict[str, Any]] = []
    for signal in signals:
        key0 = (
            signal["date"],
            signal["symbol"],
            int(to_ny(signal["signal_ts"]).value),
        )
        for policy in ("ATM_LIQ", "ADAPTIVE", "ADAPTIVE_CHEAP"):
            contract = priced.get((*key0, policy))
            if contract is None:
                continue
            window = "TRAIN_FEB_APR" if signal["date"] <= "2026-04-30" else "OOS_MAY_JUL"
            trades.append(
                {
                    **signal,
                    "signal_ts": str(to_ny(signal["signal_ts"])),
                    "window": window,
                    **contract,
                }
            )

    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    trade_frame = pd.DataFrame(trades)
    trade_frame.to_csv(out / "trades.csv", index=False)
    summary = _summary(trade_frame)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "research_only": True,
        "data_warning": "S3 trade ticks have no bid/ask; quote validation required",
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "n_dates": len(dates),
        "n_signal_rows": len(signals),
        "n_unique_events": len(unique),
        "n_priced_rows": len(trades),
        "symbols": symbols,
        "specs": [asdict(spec) | {"key": spec.key} for spec in specs],
        "summary": summary,
    }
    (out / "scoreboard.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"wrote {out} trades={len(trades)} summary={len(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
