#!/usr/bin/env python3
"""AM foresight profit map → then VWAP feature lift (rule discovery order).

Phase 1 (oracle): on a dense TOD grid, buy ATM call+put independently from
option trade lasts; record clock/oracle ret inside H∈{300,600,900}s.
No entry rule — answers *when* Mag7 AM options had reachable profit.

Phase 2 (features): at each probe, snapshot causal 10/20/30s trailing VWAP
vs RTH open (and session VWAP diff). Contrast edge probes
(oracle_ret ≥ thr) vs non-edge to see which VWAP signatures lift.

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_vwap_foresight_map \\
    --tag research_am_vwap_foresight_map_may_jul
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import prepare_day_arrays, rolling_vwap_at
from maga7.common.stock_1s import session_dates
from maga7.tools.scan_session_horizon_foresight import (
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _spot_at_arr,
    _stock_arrays,
)

NY = "America/New_York"
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")

SESSIONS = (
    ("AM_A_0930_1030", "09:30", "10:30"),
    ("AM_B_1030_1130", "10:30", "11:30"),
)

WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _hhmm_to_min(hhmm: str) -> int:
    a, b = str(hhmm).split(":")
    return int(a) * 60 + int(b)


def _tod_bucket(ts: pd.Timestamp, bucket_min: int = 5) -> str:
    t = to_ny(ts)
    m = (t.hour * 60 + t.minute) // int(bucket_min) * int(bucket_min)
    return f"{m // 60:02d}:{m % 60:02d}"


def _window_of(date: str) -> str | None:
    for name, a, b in WINDOWS:
        if a <= date <= b:
            return name
    return None


def _feat_at(
    arr: dict[str, np.ndarray],
    t: pd.Timestamp,
    *,
    day_open: float,
) -> dict[str, float] | None:
    ts_ns = arr["ts_ns"]
    t_ns = int(to_ny(t).value)
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    if i < 30:
        return None
    if abs(int(ts_ns[i]) - t_ns) > 5_000_000_000:
        return None
    px = float(arr["close"][i])
    if not np.isfinite(px) or px <= 0 or day_open <= 0:
        return None
    sess = float(arr["sess_vwap"][i]) if np.isfinite(arr["sess_vwap"][i]) else float("nan")
    out: dict[str, float] = {
        "px": px,
        "from_open_px": px / day_open - 1.0,
        "vwap_diff": (px / sess - 1.0) if sess and sess > 0 else float("nan"),
    }
    for w in (10, 20, 30):
        vw = rolling_vwap_at(arr, i, w)
        out[f"vwap{w}"] = float(vw) if np.isfinite(vw) else float("nan")
        out[f"fo_vwap{w}"] = (
            float(vw) / day_open - 1.0 if np.isfinite(vw) and vw > 0 else float("nan")
        )
    # accel: |fo10| vs |fo30|
    f10, f30 = out.get("fo_vwap10"), out.get("fo_vwap30")
    if np.isfinite(f10) and np.isfinite(f30) and f10 * f30 > 0:
        out["accel_10_30"] = abs(float(f10)) - abs(float(f30))
    else:
        out["accel_10_30"] = float("nan")
    return out


def _lift_table(
    edge: pd.DataFrame,
    ctrl: pd.DataFrame,
    *,
    feat: str,
    thrs: list[float],
    signed: bool,
) -> list[dict[str, Any]]:
    """P(edge | |feat|>=thr) / P(edge) style lift using same-tod pooled probes."""
    rows: list[dict[str, Any]] = []
    if edge.empty and ctrl.empty:
        return rows
    all_df = pd.concat([edge.assign(_edge=1), ctrl.assign(_edge=0)], ignore_index=True)
    base = float(all_df["_edge"].mean()) if len(all_df) else 0.0
    if base <= 0:
        return rows
    for thr in thrs:
        if signed:
            # same-sign extension: |feat|>=thr
            mask = all_df[feat].abs() >= float(thr)
        else:
            mask = all_df[feat] >= float(thr)
        sub = all_df.loc[mask]
        if len(sub) < 20:
            continue
        rate = float(sub["_edge"].mean())
        rows.append(
            {
                "feat": feat,
                "thr": float(thr),
                "n_gate": int(len(sub)),
                "edge_rate": rate,
                "base_rate": base,
                "lift": rate / base if base > 0 else float("nan"),
                "edge_n": int(sub["_edge"].sum()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_am_vwap_foresight_map")
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--stride-sec", type=int, default=30)
    ap.add_argument("--horizons", default="300,600,900")
    ap.add_argument("--edge-oracle", type=float, default=0.15, help="oracle_ret ≥ this = edge")
    ap.add_argument("--edge-horizon", type=int, default=900)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--bucket-min", type=int, default=5)
    ap.add_argument("--sessions", default="", help="Comma subset of session names")
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    edge_h = int(args.edge_horizon)
    if edge_h not in horizons:
        horizons.append(edge_h)
    if str(args.sessions).strip():
        want = {x.strip() for x in str(args.sessions).split(",") if x.strip()}
        sessions = tuple(s for s in SESSIONS if s[0] in want)
    else:
        sessions = SESSIONS

    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date
    ]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"am foresight map {args.start_date}..{args.end_date} "
        f"stride={args.stride_sec}s H={horizons} edge@{edge_h}>={args.edge_oracle} "
        f"sessions={[s[0] for s in sessions]}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 5 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) probes={len(rows)}", flush=True)
        cal = _window_of(date)
        if cal is None:
            continue
        for sym in symbols:
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            day1s = load_stock_1s_day(stock_1s, sym, date)
            if day1s is None or day1s.empty:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            ts_ns, px = _stock_arrays(day1s)
            arr = prepare_day_arrays(day1s)
            day_open = float(arr["day_open"])
            if not np.isfinite(day_open) or day_open <= 0:
                continue

            for sess_name, w0, w1 in sessions:
                t_cursor = to_ny(pd.Timestamp(f"{date} {w0}", tz=NY))
                t_end = to_ny(pd.Timestamp(f"{date} {w1}", tz=NY))
                while t_cursor < t_end:
                    spot = _spot_at_arr(ts_ns, px, t_cursor)
                    if spot is None:
                        t_cursor += pd.Timedelta(seconds=int(args.stride_sec))
                        continue
                    feat = _feat_at(arr, t_cursor, day_open=day_open)
                    if feat is None:
                        t_cursor += pd.Timedelta(seconds=int(args.stride_sec))
                        continue
                    for d in ("UP", "DN"):
                        ticker, dte, _ = resolve_open_lock_contract(
                            by_dte,
                            direction=d,
                            moneyness="ATM",
                            spot=float(spot),
                            prefer_dte=0,
                            allowed_dte=[0, 1, 2] if sess_name.startswith("AM_A") else [0],
                            clear_otm_thresh=0.01,
                            ladder=True,
                            otm_rungs=otm,
                        )
                        if not ticker:
                            continue
                        path = tpaths.get(str(ticker).replace("O:", ""))
                        if path is None:
                            continue
                        fwds = _fwd_trade_rets_arr(
                            path[0],
                            path[1],
                            t_cursor,
                            horizons,
                            slip=float(args.slip),
                        )
                        if not fwds:
                            continue
                        by_h = {int(x["horizon_sec"]): x for x in fwds}
                        edge_row = by_h.get(edge_h)
                        if edge_row is None:
                            continue
                        rec: dict[str, Any] = {
                            "date": date,
                            "calendar": cal,
                            "session": sess_name,
                            "symbol": sym,
                            "dir": d,
                            "entry_ts": str(t_cursor),
                            "tod_bucket": _tod_bucket(t_cursor, args.bucket_min),
                            "ticker": ticker,
                            "dte": dte,
                            "spot": float(spot),
                            "day_open": float(day_open),
                            "oracle_ret": float(edge_row["oracle_ret"]),
                            "clock_ret": float(edge_row["clock_ret"]),
                            "oracle_hold_sec": float(edge_row["oracle_hold_sec"]),
                            "mfe": float(edge_row["mfe"]),
                            "mae": float(edge_row["mae"]),
                            "is_edge": bool(
                                float(edge_row["oracle_ret"]) + 1e-12 >= float(args.edge_oracle)
                            ),
                        }
                        for h, fr in by_h.items():
                            rec[f"oracle_h{h}"] = float(fr["oracle_ret"])
                            rec[f"clock_h{h}"] = float(fr["clock_ret"])
                        rec.update(feat)
                        rows.append(rec)
                    t_cursor += pd.Timedelta(seconds=int(args.stride_sec))

    if not rows:
        print("no probes", flush=True)
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(out / "probes.csv", index=False)
    print(f"probes={len(df)} edge_rate={df['is_edge'].mean():.3f}", flush=True)

    # --- TOD map ---
    tod = (
        df.groupby(["calendar", "session", "tod_bucket", "dir"], observed=False)
        .agg(
            n=("is_edge", "size"),
            edge_rate=("is_edge", "mean"),
            mean_oracle=("oracle_ret", "mean"),
            mean_clock=("clock_ret", "mean"),
            p50_oracle=("oracle_ret", "median"),
            frac_clock_pos=("clock_ret", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values(["session", "tod_bucket", "dir"])
    )
    tod.to_csv(out / "tod_map.csv", index=False)

    # best TOD pockets (discover only)
    disc = tod[tod["calendar"] == "may_jul09"].copy()
    pockets = disc[(disc["n"] >= 40) & (disc["edge_rate"] >= 0.25)].sort_values(
        "edge_rate", ascending=False
    )
    pockets.to_csv(out / "tod_pockets_discover.csv", index=False)

    # --- Feature lift on discover ---
    disc_df = df[df["calendar"] == "may_jul09"]
    edge = disc_df[disc_df["is_edge"]]
    ctrl = disc_df[~disc_df["is_edge"]]
    lift_rows: list[dict[str, Any]] = []
    thrs = [0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015]
    for feat in ("fo_vwap10", "fo_vwap20", "fo_vwap30", "from_open_px", "vwap_diff", "accel_10_30"):
        # signed extension: for FO feats, require feat sign matches dir
        sub_e = edge.copy()
        sub_c = ctrl.copy()
        if feat.startswith("fo_vwap") or feat == "from_open_px":
            # align: DN wants negative fo, UP wants positive
            def _signed(frame: pd.DataFrame) -> pd.DataFrame:
                f = frame.copy()
                sign = np.where(f["dir"].to_numpy() == "UP", 1.0, -1.0)
                f[feat] = f[feat].astype(float) * sign
                return f

            sub_e, sub_c = _signed(sub_e), _signed(sub_c)
            lift_rows.extend(_lift_table(sub_e, sub_c, feat=feat, thrs=thrs, signed=False))
        else:
            lift_rows.extend(_lift_table(sub_e, sub_c, feat=feat, thrs=thrs, signed=False))

    lift = pd.DataFrame(lift_rows)
    if not lift.empty:
        lift = lift.sort_values(["lift", "edge_n"], ascending=[False, False])
        lift.to_csv(out / "feature_lift_discover.csv", index=False)

    # Per-session / dir edge summary
    summ = (
        df.groupby(["calendar", "session", "dir"], observed=False)
        .agg(
            n=("is_edge", "size"),
            edge_rate=("is_edge", "mean"),
            mean_oracle=("oracle_ret", "mean"),
            mean_clock=("clock_ret", "mean"),
        )
        .reset_index()
    )
    summ.to_csv(out / "session_dir_summary.csv", index=False)

    # Top lift candidates
    top_lift = lift.head(12).to_dict("records") if not lift.empty else []
    top_pockets = pockets.head(12).to_dict("records") if not pockets.empty else []

    # Blind check: same top feat thr on jul10_23
    blind_checks: list[dict[str, Any]] = []
    blind = df[df["calendar"] == "jul10_23"]
    for cand in top_lift[:5]:
        feat, thr = cand["feat"], float(cand["thr"])
        b = blind.copy()
        if feat.startswith("fo_vwap") or feat == "from_open_px":
            sign = np.where(b["dir"].to_numpy() == "UP", 1.0, -1.0)
            b[feat] = b[feat].astype(float) * sign
        gate = b[b[feat] >= thr]
        if len(gate) < 10:
            continue
        blind_checks.append(
            {
                "feat": feat,
                "thr": thr,
                "n": int(len(gate)),
                "edge_rate": float(gate["is_edge"].mean()),
                "mean_oracle": float(gate["oracle_ret"].mean()),
                "mean_clock": float(gate["clock_ret"].mean()),
                "discover_lift": float(cand["lift"]),
            }
        )

    verdict = {
        "protocol": "foresight_map_then_feature_lift",
        "edge_def": f"oracle_ret@{edge_h}s >= {args.edge_oracle}",
        "stride_sec": int(args.stride_sec),
        "n_probes": int(len(df)),
        "edge_rate_all": float(df["is_edge"].mean()),
        "session_dir_summary": summ.to_dict("records"),
        "top_tod_pockets_discover": top_pockets,
        "top_feature_lift_discover": top_lift,
        "blind_jul_top_feats": blind_checks,
        "verdict": (
            "FORESIGHT_HAS_POCKETS"
            if len(top_pockets) >= 3 and any(float(x.get("lift") or 0) >= 1.2 for x in top_lift)
            else (
                "FORESIGHT_WEAK_LIFT"
                if len(top_pockets) >= 1
                else "FORESIGHT_NO_EDGE"
            )
        ),
        "next": (
            "Only distill causal FO/VWAP rules inside TOD pockets with "
            "discover lift≥1.2 and non-collapsing blind edge_rate."
        ),
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("\n=== VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str)[:3500], flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
