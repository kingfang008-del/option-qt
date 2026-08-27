#!/usr/bin/env python3
"""Offline verification of proposed L1 (SVI) + L2 (path Greeks) layers.

P0: fit raw SVI on Mag7/QQQ ``nq_options_day_iv`` front-expiry @10:30.
P1: Jul-20 clock fused trades — reconstruct option mid path from live Redis
    stream (db=0), invert BS IV / delta, test early-exit rules vs T+30/SL.

Outputs under ``/mnt/s990/data/maga7/results/vol_surface_layer_verify/``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.svi_raw import fit_svi_raw, svi_iv
from maga7.live.redis_fused import redis_client, run_keys, unpack_batch

NY = "America/New_York"
OUT_DEFAULT = Path("/mnt/s990/data/maga7/results/vol_surface_layer_verify")
DAY_IV = Path.home() / "train_data/nq_options_day_iv"
SESSION = Path(
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e"
)
CLOCK_TRADES = SESSION / "fused_replay_lit_m5_v1_clock/trades.csv"


def _bs_price(S, K, T, r, sigma, cp: str) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if cp == "c" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "c":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_delta(S, K, T, r, sigma, cp: str) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return 1.0 if (cp == "c" and S > K) else (-1.0 if cp == "p" and S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1) if cp == "c" else norm.cdf(d1) - 1.0)


def _implied_vol(mid: float, S: float, K: float, T: float, r: float, cp: str) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 1e-8:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(60):
        mid_s = 0.5 * (lo + hi)
        px = _bs_price(S, K, T, r, mid_s, cp)
        if px > mid:
            hi = mid_s
        else:
            lo = mid_s
    iv = 0.5 * (lo + hi)
    if not (0.01 < iv < 2.5):
        return None
    return float(iv)


def _parse_occ(contract: str) -> tuple[str, str, float, str]:
    """Return (symbol, yymmdd, strike, cp)."""
    s = str(contract).replace(" ", "")
    # e.g. AMD260720C00517500
    import re

    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", s)
    if not m:
        raise ValueError(f"bad contract {contract!r}")
    sym, ymd, cp, strike_s = m.groups()
    return sym, ymd, float(strike_s) / 1000.0, cp.lower()


def _clean_slice(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "bid" in d.columns and "ask" in d.columns:
        mid = (pd.to_numeric(d["bid"], errors="coerce") + pd.to_numeric(d["ask"], errors="coerce")) / 2.0
        spread = pd.to_numeric(d["ask"], errors="coerce") - pd.to_numeric(d["bid"], errors="coerce")
        ok_spread = spread <= np.maximum(0.10, mid * 0.10)
        d = d.loc[ok_spread.fillna(False)]
    if "volume" in d.columns:
        # keep if volume>0 OR iv already present (day_iv often aggregates)
        vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
        d = d.loc[(vol > 0) | (pd.to_numeric(d["iv"], errors="coerce") > 0.01)]
    iv = pd.to_numeric(d["iv"], errors="coerce")
    d = d.loc[iv.between(0.05, 2.0)]
    # IV outlier vs neighbors by strike
    d = d.sort_values("strike_price")
    if len(d) >= 5:
        med = d["iv"].rolling(5, center=True, min_periods=1).median()
        d = d.loc[(d["iv"] - med).abs() <= 0.05]
    return d


def run_p0(dates: list[str], symbols: list[str], asof: str = "10:30") -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for date in dates:
            path = DAY_IV / sym / f"{sym}_{date}.parquet"
            if not path.is_file():
                rows.append({"symbol": sym, "date": date, "ok": False, "error": "missing_day_iv"})
                continue
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
            day = df[df["timestamp"].dt.strftime("%Y-%m-%d") == date]
            snap = day[day["timestamp"].dt.strftime("%H:%M") == asof]
            if snap.empty:
                # nearest <= asof
                asof_ts = pd.Timestamp(f"{date} {asof}", tz=NY)
                snap = day[day["timestamp"] <= asof_ts]
                if snap.empty:
                    rows.append({"symbol": sym, "date": date, "ok": False, "error": "no_asof"})
                    continue
                t_last = snap["timestamp"].max()
                snap = snap[snap["timestamp"] == t_last]
            # front expiry by min T with enough points
            snap = snap.copy()
            snap["expiration_date"] = pd.to_datetime(snap["expiration_date"])
            S = float(pd.to_numeric(snap["stock_close"], errors="coerce").dropna().iloc[-1])
            best = None
            for exp, g in snap.groupby(snap["expiration_date"].dt.strftime("%Y-%m-%d")):
                g = _clean_slice(g)
                if len(g) < 8:
                    continue
                T_years = max(
                    (pd.Timestamp(exp, tz=NY).normalize() + pd.Timedelta(hours=16) - pd.Timestamp(f"{date} {asof}", tz=NY)).total_seconds()
                    / 31557600.0,
                    1e-4,
                )
                F = S  # short-dated approx
                k = np.log(pd.to_numeric(g["strike_price"], errors="coerce").values / F)
                iv = pd.to_numeric(g["iv"], errors="coerce").values
                vol = pd.to_numeric(g["volume"], errors="coerce").fillna(1.0).values if "volume" in g else None
                try:
                    params, met = fit_svi_raw(k, iv, T_years, weights=vol)
                except Exception as exc:
                    continue
                score = met["rmse_iv"]
                cand = {
                    "symbol": sym,
                    "date": date,
                    "expiry": exp,
                    "ok": True,
                    "S": S,
                    "T": T_years,
                    "n_clean": int(met["n"]),
                    "rmse_iv": met["rmse_iv"],
                    "mae_iv": met["mae_iv"],
                    "frac_neg_butterfly": met["frac_neg_butterfly"],
                    "iv_atm_mkt": met["iv_atm_mkt"],
                    "iv_atm_svi": met["iv_atm_svi"],
                    **params.as_dict(),
                }
                if best is None or score < best["rmse_iv"]:
                    best = cand
            if best is None:
                rows.append({"symbol": sym, "date": date, "ok": False, "error": "fit_failed"})
            else:
                rows.append(best)
    return pd.DataFrame(rows)


def _extract_paths_from_redis(
    *,
    session_id: str,
    wants: dict[str, tuple[str, pd.Timestamp, pd.Timestamp]],
) -> dict[str, pd.DataFrame]:
    """One-pass scan: contract_key -> path rows.

    ``wants``: cleaned_contract -> (symbol, entry_ts, exit_ts)
    """
    r = redis_client(db=0)
    stream = run_keys(session_id)["stream"]
    t_min = min(v[1] for v in wants.values()) - pd.Timedelta(seconds=2)
    t_max = max(v[2] for v in wants.values()) + pd.Timedelta(seconds=2)
    buckets: dict[str, list[dict]] = {k: [] for k in wants}
    # suffix index for fuzzy OCC match
    suffixes = {k: k[-15:] for k in wants}
    cursor = "-"
    n_frames = 0
    while True:
        chunk = r.xrange(stream, min=cursor, max="+", count=500)
        if not chunk:
            break
        for mid, fields in chunk:
            n_frames += 1
            raw = fields.get(b"batch") if isinstance(fields, dict) else None
            if raw is None:
                continue
            batch = unpack_batch(raw)
            for payload in batch:
                if not isinstance(payload, dict):
                    continue
                sym = str(payload.get("symbol") or "").upper()
                ts = float(payload.get("ts") or 0.0)
                t = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(NY)
                if t < t_min or t > t_max:
                    continue
                stock = payload.get("stock") or {}
                S = float(stock.get("close") or 0.0)
                for oc in list(payload.get("option_contracts") or []):
                    if not isinstance(oc, dict):
                        continue
                    c = str(
                        oc.get("localSymbol")
                        or oc.get("contract")
                        or oc.get("ticker")
                        or oc.get("symbol")
                        or ""
                    ).replace(" ", "")
                    hit = None
                    for key, suf in suffixes.items():
                        if wants[key][0].upper() != sym:
                            continue
                        if c == key or suf in c or (len(c) >= 15 and c[-15:] == suf):
                            hit = key
                            break
                    if hit is None:
                        continue
                    bid = float(oc.get("bid") or 0.0)
                    ask = float(oc.get("ask") or 0.0)
                    mid_px = float(oc.get("mid") or 0.0)
                    if mid_px <= 0:
                        if bid > 0 and ask > 0:
                            mid_px = 0.5 * (bid + ask)
                        else:
                            mid_px = bid or ask
                    if mid_px <= 0:
                        continue
                    buckets[hit].append(
                        {"ts": t, "S": S, "bid": bid, "ask": ask, "mid": mid_px}
                    )
            cursor = "(" + (mid.decode() if isinstance(mid, bytes) else str(mid))
        if len(chunk) < 500:
            break
    print(f"  redis scanned frames={n_frames}", flush=True)
    out: dict[str, pd.DataFrame] = {}
    for key, rows in buckets.items():
        if not rows:
            out[key] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
        _, entry_ts, exit_ts = wants[key]
        out[key] = df.loc[(df["ts"] >= entry_ts) & (df["ts"] <= exit_ts)].reset_index(
            drop=True
        )
    return out


def run_p1(r: float = 0.04) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(CLOCK_TRADES)
    session_id = SESSION.name
    wants: dict[str, tuple[str, pd.Timestamp, pd.Timestamp]] = {}
    meta = []
    for _, tr in trades.iterrows():
        contract = str(tr["contract"])
        sym, ymd, K, cp = _parse_occ(contract)
        entry_ts = pd.Timestamp(tr["entry_ts"]).tz_convert(NY)
        exit_ts = pd.Timestamp(tr["exit_ts"]).tz_convert(NY)
        key = contract.replace(" ", "")
        wants[key] = (sym, entry_ts, exit_ts)
        meta.append((tr, sym, ymd, K, cp, key, entry_ts, exit_ts))
    print("  extracting option paths from Redis db=0...", flush=True)
    paths = _extract_paths_from_redis(session_id=session_id, wants=wants)

    detail_frames = []
    summary_rows = []
    for tr, sym, ymd, K, cp, key, entry_ts, exit_ts in meta:
        contract = str(tr["contract"])
        expiry = pd.Timestamp(f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", tz=NY) + pd.Timedelta(
            hours=16
        )
        path = paths.get(key, pd.DataFrame())
        if path.empty:
            summary_rows.append(
                {
                    "symbol": sym,
                    "contract": contract,
                    "clock_ret": float(tr["ret"]),
                    "clock_reason": str(tr["reason"]),
                    "ok": False,
                    "error": "no_path",
                }
            )
            continue
        entry_px = float(tr["entry"])
        ivs, deltas, rets = [], [], []
        for _, row in path.iterrows():
            T = max((expiry - row["ts"]).total_seconds() / 31557600.0, 1e-6)
            iv = _implied_vol(float(row["mid"]), float(row["S"]), K, T, r, cp)
            if iv is None:
                ivs.append(np.nan)
                deltas.append(np.nan)
            else:
                ivs.append(iv)
                deltas.append(_bs_delta(float(row["S"]), K, T, r, iv, cp))
            rets.append(float(row["mid"]) / entry_px - 1.0)
        path = path.copy()
        path["iv"] = ivs
        path["delta"] = deltas
        path["opt_ret"] = rets
        path["symbol"] = sym
        path["contract"] = contract
        detail_frames.append(path)

        peak = float(np.nanmax(path["opt_ret"]))
        peak_i = int(np.nanargmax(path["opt_ret"].values))
        iv0 = float(path["iv"].iloc[0]) if np.isfinite(path["iv"].iloc[0]) else np.nan
        # L2-style rules after peak
        giveback_hit = None
        iv_shock_hit = None
        delta_fade_hit = None
        for i in range(peak_i, len(path)):
            ret = float(path["opt_ret"].iloc[i])
            iv = float(path["iv"].iloc[i]) if np.isfinite(path["iv"].iloc[i]) else np.nan
            dlt = float(path["delta"].iloc[i]) if np.isfinite(path["delta"].iloc[i]) else np.nan
            # giveback: was +15% MFE, now gave back half of peak
            if giveback_hit is None and peak >= 0.15 and ret <= peak - 0.5 * peak:
                giveback_hit = i
            # IV shock: IV up > 3 vol points from entry while opt_ret falling from peak
            if iv_shock_hit is None and np.isfinite(iv) and np.isfinite(iv0) and (iv - iv0) >= 0.03 and ret < peak - 0.05:
                iv_shock_hit = i
            # delta fade for long call: delta drops > 0.15 from peak-bar delta
            d_peak = float(path["delta"].iloc[peak_i]) if np.isfinite(path["delta"].iloc[peak_i]) else np.nan
            if delta_fade_hit is None and np.isfinite(dlt) and np.isfinite(d_peak) and (d_peak - dlt) >= 0.15 and ret < peak - 0.05:
                delta_fade_hit = i

        def _cf(i):
            if i is None:
                return None
            return {
                "t": str(path["ts"].iloc[i]),
                "ret": float(path["opt_ret"].iloc[i]),
                "hold_sec": float((path["ts"].iloc[i] - entry_ts).total_seconds()),
                "iv": float(path["iv"].iloc[i]) if np.isfinite(path["iv"].iloc[i]) else None,
                "delta": float(path["delta"].iloc[i]) if np.isfinite(path["delta"].iloc[i]) else None,
            }

        first = None
        first_name = None
        for name, idx in [("giveback", giveback_hit), ("iv_shock", iv_shock_hit), ("delta_fade", delta_fade_hit)]:
            if idx is None:
                continue
            if first is None or idx < first:
                first = idx
                first_name = name
        cf = _cf(first)
        summary_rows.append(
            {
                "symbol": sym,
                "contract": contract,
                "clock_ret": float(tr["ret"]),
                "clock_reason": str(tr["reason"]),
                "ok": True,
                "n_path": int(len(path)),
                "mfe": peak,
                "mae": float(np.nanmin(path["opt_ret"])),
                "iv0": iv0,
                "iv_end": float(path["iv"].iloc[-1]) if np.isfinite(path["iv"].iloc[-1]) else np.nan,
                "delta0": float(path["delta"].iloc[0]) if np.isfinite(path["delta"].iloc[0]) else np.nan,
                "delta_end": float(path["delta"].iloc[-1]) if np.isfinite(path["delta"].iloc[-1]) else np.nan,
                "l2_rule": first_name,
                "l2_ret": None if cf is None else cf["ret"],
                "l2_hold_sec": None if cf is None else cf["hold_sec"],
                "lift_vs_clock": None if cf is None else float(cf["ret"] - float(tr["ret"])),
            }
        )
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--p1-only", action="store_true")
    ap.add_argument(
        "--dates",
        default="2026-07-14,2026-07-15,2026-07-16",
        help="P0 dates (nq day_iv)",
    )
    ap.add_argument("--symbols", default="AMD,NVDA,QQQ")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    if not args.p1_only:
        dates = [x.strip() for x in args.dates.split(",") if x.strip()]
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        print("=== P0 SVI ===", flush=True)
        p0 = run_p0(dates, symbols)
        p0.to_csv(out / "p0_svi_fits.csv", index=False)
        ok = p0[p0.get("ok") == True] if "ok" in p0.columns else p0  # noqa: E712
        report["p0"] = {
            "n_ok": int(ok.shape[0]),
            "n_all": int(len(p0)),
            "rmse_median": float(ok["rmse_iv"].median()) if len(ok) and "rmse_iv" in ok else None,
            "rmse_p90": float(ok["rmse_iv"].quantile(0.9)) if len(ok) and "rmse_iv" in ok else None,
            "butterfly_neg_median": float(ok["frac_neg_butterfly"].median())
            if len(ok) and "frac_neg_butterfly" in ok
            else None,
        }
        print(p0.to_string(index=False), flush=True)
        print("P0 summary", report["p0"], flush=True)

    if not args.p0_only:
        print("=== P1 path Greeks Jul20 clock ===", flush=True)
        p1, detail = run_p1()
        p1.to_csv(out / "p1_path_greeks_summary.csv", index=False)
        if not detail.empty:
            detail.to_csv(out / "p1_path_greeks_detail.csv", index=False)
        ok = p1[p1["ok"] == True] if "ok" in p1.columns else p1  # noqa: E712
        lifts = ok["lift_vs_clock"].dropna() if "lift_vs_clock" in ok.columns else pd.Series(dtype=float)
        report["p1"] = {
            "n_ok": int(ok.shape[0]),
            "n_all": int(len(p1)),
            "n_l2_fire": int(ok["l2_rule"].notna().sum()) if "l2_rule" in ok else 0,
            "sum_clock_ret": float(ok["clock_ret"].sum()) if len(ok) else None,
            "sum_l2_ret": float(ok["l2_ret"].fillna(ok["clock_ret"]).sum()) if len(ok) else None,
            "sum_lift": float(lifts.sum()) if len(lifts) else None,
        }
        print(p1.to_string(index=False), flush=True)
        print("P1 summary", report["p1"], flush=True)

    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
