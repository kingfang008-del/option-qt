#!/usr/bin/env python3
"""Foresight wave scan: big stock dumps × option tick flow signatures.

Oracle question (not a promote gate):
  Given large forward stock dumps on days with option tick prints, is there a
  *causal* put-flow signature *before* the wave that could be distilled?

Steps per symbol-day:
  1) Stock 1s: find non-overlapping waves where fwd ret over H <= -wave_thr
  2) At wave start t0, read causal tick put_share / put_vol_z (lookback only)
  3) Oracle: buy ATM put at t0 (tick last±slip), best exit inside H (and clock@H)
  4) Contrast: random same-session probes without requiring a dump

Example:
  PYTHONPATH=. python -m maga7.tools.scan_option_flow_tick_foresight_waves \\
    --tag research_option_flow_tick_foresight_waves_jul10_23
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
from maga7.common.option_flow import (
    DEFAULT_TICK_ROOT,
    load_option_tick_day,
    prepare_option_flow_day,
    put_flow_features_at,
    tick_dates,
)
from maga7.common.replay import to_ny
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
SESSIONS = (
    ("AM_0935_1030", "09:35", "10:30"),
    ("CORE_1030_1200", "10:30", "12:00"),
    ("MID_1200_1400", "12:00", "14:00"),
    ("PM_1400_1530", "14:00", "15:30"),
)


def _idx_at_or_before(ts_ns: np.ndarray, t_ns: int) -> int | None:
    i = int(np.searchsorted(ts_ns, t_ns, side="right") - 1)
    return i if i >= 0 else None


def _find_dump_waves(
    ts_ns: np.ndarray,
    px: np.ndarray,
    *,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    horizon_sec: int,
    wave_thr: float,
    stride_sec: int,
    min_gap_sec: int,
) -> list[dict[str, Any]]:
    """Non-overlapping times where forward stock ret over H <= -wave_thr."""
    t0 = to_ny(t_start)
    t1 = to_ny(t_end)
    stride = pd.Timedelta(seconds=int(stride_sec))
    gap = pd.Timedelta(seconds=int(min_gap_sec))
    h_ns = int(horizon_sec) * 1_000_000_000
    out: list[dict[str, Any]] = []
    t = t0
    next_ok = t0
    while t < t1:
        if t >= next_ok:
            i0 = _idx_at_or_before(ts_ns, int(t.value))
            if i0 is not None:
                end_ns = int(ts_ns[i0]) + h_ns
                i1 = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
                if i1 > i0:
                    a = float(px[i0])
                    # wave depth = min close in window (worst dump) and clock end
                    win = px[i0 : i1 + 1]
                    b_min = float(np.nanmin(win))
                    b_end = float(px[i1])
                    if a > 0 and np.isfinite(a) and np.isfinite(b_min):
                        depth = b_min / a - 1.0
                        clock = b_end / a - 1.0
                        if depth <= -float(wave_thr):
                            out.append(
                                {
                                    "t0": t,
                                    "stock_depth": float(depth),
                                    "stock_clock": float(clock),
                                    "i0": int(i0),
                                }
                            )
                            next_ok = t + gap
        t += stride
    return out


def _flow_at_ts(flow: dict[str, Any], t: pd.Timestamp, window_sec: int) -> dict[str, float] | None:
    i = _idx_at_or_before(flow["ts_ns"], int(to_ny(t).value))
    if i is None:
        return None
    feat = put_flow_features_at(flow, i=i, window_sec=int(window_sec))
    if feat is None:
        return None
    share, z, pv, cv = feat
    return {
        "put_share": float(share),
        "put_vol_z": float(z),
        "put_v": float(pv),
        "call_v": float(cv),
    }


def _summarize(rows: list[dict], *, prefix: str) -> dict[str, Any]:
    if not rows:
        return {f"{prefix}_n": 0}
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {f"{prefix}_n": int(len(df))}
    for col in ("put_share", "put_vol_z", "oracle_ret", "clock_ret", "stock_depth"):
        if col in df.columns and df[col].notna().any():
            s = df[col].astype(float)
            out[f"{prefix}_{col}_mean"] = float(s.mean())
            out[f"{prefix}_{col}_p50"] = float(s.median())
    if "put_share" in df.columns:
        for thr in (0.50, 0.55, 0.60, 0.65):
            out[f"{prefix}_frac_share_ge_{thr}"] = float((df["put_share"] >= thr).mean())
    if "put_vol_z" in df.columns:
        for z in (1.5, 2.0, 3.0):
            out[f"{prefix}_frac_z_ge_{z}"] = float((df["put_vol_z"] >= z).mean())
    if "oracle_ret" in df.columns:
        out[f"{prefix}_oracle_pos_frac"] = float((df["oracle_ret"] > 0).mean())
        out[f"{prefix}_oracle_mean"] = float(df["oracle_ret"].mean())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_option_flow_tick_foresight_waves_jul10_23")
    ap.add_argument("--tick-root", default=str(DEFAULT_TICK_ROOT))
    ap.add_argument("--start-date", default="")
    ap.add_argument("--end-date", default="")
    ap.add_argument("--wave-thr", default="0.005,0.008,0.012")
    ap.add_argument("--horizons", default="300,600,900")
    ap.add_argument("--flow-sec", default="60,120")
    ap.add_argument("--stride-sec", type=int, default=30)
    ap.add_argument("--min-gap-sec", type=int, default=300)
    ap.add_argument("--control-stride-sec", type=int, default=300)
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument(
        "--sessions",
        default="AM_0935_1030,CORE_1030_1200,MID_1200_1400,PM_1400_1530",
    )
    args = ap.parse_args(argv)

    tick_root = Path(args.tick_root)
    dates = tick_dates(tick_root)
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    if args.end_date:
        dates = [d for d in dates if d <= args.end_date]
    if not dates:
        print(f"no tick dates under {tick_root}", flush=True)
        return 2

    wave_thrs = [float(x) for x in args.wave_thr.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    flow_secs = [int(x) for x in args.flow_sec.split(",") if x.strip()]
    want_sess = {x.strip() for x in args.sessions.split(",") if x.strip()}
    sessions = tuple(s for s in SESSIONS if s[0] in want_sess)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof.get("symbols") or [])
    stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"foresight waves {dates[0]}..{dates[-1]} days={len(dates)} "
        f"thr={wave_thrs} H={horizons} flow={flow_secs}",
        flush=True,
    )

    wave_rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []

    for di, date in enumerate(dates):
        if di % 2 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) waves={len(wave_rows)}", flush=True)
        for sym in symbols:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            ts_ns, px = _stock_arrays(day)
            tday = load_option_tick_day(tick_root, sym, date)
            flow = prepare_option_flow_day(tday)
            if flow is None:
                continue
            tpaths = _paths_by_ticker(tday)
            by_dte = lock.get((sym, date))
            if not by_dte or not tpaths:
                continue

            for sess_name, s0, s1 in sessions:
                t_start = pd.Timestamp(f"{date} {s0}:00", tz=NY)
                t_end = pd.Timestamp(f"{date} {s1}:00", tz=NY)
                for H in horizons:
                    for wthr in wave_thrs:
                        waves = _find_dump_waves(
                            ts_ns,
                            px,
                            t_start=t_start + pd.Timedelta(seconds=max(flow_secs)),
                            t_end=t_end - pd.Timedelta(seconds=H),
                            horizon_sec=H,
                            wave_thr=wthr,
                            stride_sec=int(args.stride_sec),
                            min_gap_sec=int(args.min_gap_sec),
                        )
                        for w in waves:
                            t0 = w["t0"]
                            spot = _spot_at_arr(ts_ns, px, t0)
                            ticker, dte, _ = resolve_open_lock_contract(
                                by_dte,
                                direction="DN",
                                moneyness="ATM",
                                spot=spot,
                                prefer_dte=0,
                                allowed_dte=[0, 1, 2],
                                clear_otm_thresh=0.01,
                                ladder=True,
                                otm_rungs=otm,
                            )
                            if not ticker:
                                continue
                            arr = tpaths.get(str(ticker).replace("O:", ""))
                            if arr is None:
                                continue
                            fwd = _fwd_trade_rets_arr(
                                arr[0], arr[1], t0, [H], slip=float(args.slip)
                            )
                            if not fwd:
                                continue
                            fr = fwd[0]
                            for fw in flow_secs:
                                feat = _flow_at_ts(flow, t0, fw)
                                if feat is None:
                                    continue
                                wave_rows.append(
                                    {
                                        "date": date,
                                        "symbol": sym,
                                        "session": sess_name,
                                        "kind": "wave",
                                        "t0": str(to_ny(t0)),
                                        "horizon_sec": int(H),
                                        "wave_thr": float(wthr),
                                        "flow_sec": int(fw),
                                        "stock_depth": w["stock_depth"],
                                        "stock_clock": w["stock_clock"],
                                        "ticker": ticker,
                                        "dte": dte,
                                        "oracle_ret": fr["oracle_ret"],
                                        "clock_ret": fr["clock_ret"],
                                        "oracle_hold_sec": fr["oracle_hold_sec"],
                                        **feat,
                                    }
                                )

                # controls: session probes without dump requirement
                t = t_start + pd.Timedelta(seconds=max(flow_secs))
                t_lim = t_end - pd.Timedelta(seconds=max(horizons))
                cstride = pd.Timedelta(seconds=int(args.control_stride_sec))
                H0 = int(horizons[0])
                while t < t_lim:
                    spot = _spot_at_arr(ts_ns, px, t)
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction="DN",
                        moneyness="ATM",
                        spot=spot,
                        prefer_dte=0,
                        allowed_dte=[0, 1, 2],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if ticker:
                        arr = tpaths.get(str(ticker).replace("O:", ""))
                        if arr is not None:
                            fwd = _fwd_trade_rets_arr(
                                arr[0], arr[1], t, [H0], slip=float(args.slip)
                            )
                            if fwd:
                                fr = fwd[0]
                                i0 = _idx_at_or_before(ts_ns, int(to_ny(t).value))
                                depth = float("nan")
                                if i0 is not None:
                                    end_ns = int(ts_ns[i0]) + H0 * 1_000_000_000
                                    i1 = int(np.searchsorted(ts_ns, end_ns, side="right") - 1)
                                    if i1 > i0 and float(px[i0]) > 0:
                                        depth = float(np.nanmin(px[i0 : i1 + 1]) / float(px[i0]) - 1.0)
                                for fw in flow_secs:
                                    feat = _flow_at_ts(flow, t, fw)
                                    if feat is None:
                                        continue
                                    ctrl_rows.append(
                                        {
                                            "date": date,
                                            "symbol": sym,
                                            "session": sess_name,
                                            "kind": "control",
                                            "t0": str(to_ny(t)),
                                            "horizon_sec": H0,
                                            "wave_thr": None,
                                            "flow_sec": int(fw),
                                            "stock_depth": depth,
                                            "ticker": ticker,
                                            "dte": dte,
                                            "oracle_ret": fr["oracle_ret"],
                                            "clock_ret": fr["clock_ret"],
                                            **feat,
                                        }
                                    )
                    t += cstride

    waves_df = pd.DataFrame(wave_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    waves_df.to_csv(out / "waves.csv", index=False)
    ctrl_df.to_csv(out / "controls.csv", index=False)

    score_rows: list[dict[str, Any]] = []
    for H in horizons:
        for wthr in wave_thrs:
            for fw in flow_secs:
                w = waves_df[
                    (waves_df.horizon_sec == H)
                    & (waves_df.wave_thr == wthr)
                    & (waves_df.flow_sec == fw)
                ]
                c = ctrl_df[(ctrl_df.horizon_sec == H) & (ctrl_df.flow_sec == fw)]
                # if control horizon differs, still compare flow distributions
                if c.empty:
                    c = ctrl_df[ctrl_df.flow_sec == fw]
                row: dict[str, Any] = {
                    "horizon_sec": H,
                    "wave_thr": wthr,
                    "flow_sec": fw,
                }
                row.update(_summarize(w.to_dict("records"), prefix="wave"))
                row.update(_summarize(c.to_dict("records"), prefix="ctrl"))
                # lift: P(share>=0.55|wave) / P(share>=0.55|ctrl)
                for thr in (0.55, 0.60):
                    wk = f"wave_frac_share_ge_{thr}"
                    ck = f"ctrl_frac_share_ge_{thr}"
                    if row.get(wk) is not None and row.get(ck) not in (None, 0, 0.0):
                        row[f"lift_share_{thr}"] = float(row[wk]) / float(row[ck])
                    else:
                        row[f"lift_share_{thr}"] = None
                for z in (1.5, 2.0):
                    wk = f"wave_frac_z_ge_{z}"
                    ck = f"ctrl_frac_z_ge_{z}"
                    if row.get(wk) is not None and row.get(ck) not in (None, 0, 0.0):
                        row[f"lift_z_{z}"] = float(row[wk]) / float(row[ck])
                    else:
                        row[f"lift_z_{z}"] = None
                # distillibility: need lift>1.2 and wave oracle mean>0 and enough n
                lift = row.get("lift_share_0.55")
                row["distill_ok"] = bool(
                    int(row.get("wave_n") or 0) >= 15
                    and lift is not None
                    and float(lift) >= 1.2
                    and float(row.get("wave_oracle_mean") or 0) > 0
                    and float(row.get("wave_frac_share_ge_0.55") or 0) >= 0.45
                )
                score_rows.append(row)

    score = pd.DataFrame(score_rows)
    score.to_csv(out / "scoreboard.csv", index=False)
    distill = score[score.distill_ok].sort_values("lift_share_0.55", ascending=False)
    verdict = "FORESIGHT_DISTILL" if len(distill) else "FORESIGHT_NO_DISTILL"

    # best economic oracle pocket regardless of distill
    best = None
    if len(score):
        best = score.sort_values("wave_oracle_mean", ascending=False).iloc[0].to_dict()

    summary = {
        "expert_kind": "option_flow_tick_foresight_waves",
        "tick_root": str(tick_root),
        "dates": dates,
        "n_wave_rows": int(len(waves_df)),
        "n_ctrl_rows": int(len(ctrl_df)),
        "verdict": verdict,
        "distill_n": int(len(distill)),
        "best_oracle_cell": best,
        "champion_distill": distill.iloc[0].to_dict() if len(distill) else None,
        "note": (
            "Oracle labels big stock dumps then inspects causal tick put-flow at t0. "
            "DISTILL requires lift(share>=0.55)>=1.2, wave_n>=15, oracle_mean>0, "
            "frac_share>=0.45. Not a live promote gate."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if len(distill):
        distill.to_csv(out / "distill_pass.csv", index=False)

    print("\n=== verdict", verdict, "distill_n=", len(distill), flush=True)
    cols = [
        c
        for c in [
            "horizon_sec",
            "wave_thr",
            "flow_sec",
            "wave_n",
            "wave_put_share_mean",
            "wave_frac_share_ge_0.55",
            "ctrl_frac_share_ge_0.55",
            "lift_share_0.55",
            "wave_oracle_mean",
            "distill_ok",
        ]
        if c in score.columns
    ]
    show = score.sort_values(
        ["distill_ok", "lift_share_0.55", "wave_oracle_mean"],
        ascending=[False, False, False],
    ).head(15)
    print(show[cols].to_string(index=False), flush=True)
    if best:
        print(
            f"\nbest oracle cell H={best.get('horizon_sec')} thr={best.get('wave_thr')} "
            f"flow={best.get('flow_sec')} n={best.get('wave_n')} "
            f"oracle_mean={best.get('wave_oracle_mean')} "
            f"share_mean={best.get('wave_put_share_mean')} "
            f"lift55={best.get('lift_share_0.55')}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)
    return 0 if verdict == "FORESIGHT_DISTILL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
