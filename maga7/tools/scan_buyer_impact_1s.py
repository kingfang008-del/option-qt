#!/usr/bin/env python3
"""Validate 'buyer only in ~10% impact windows' using stock 1s → option oracle.

Thesis: option sellers win most of the time; buyers need rare stock-impact
bursts. Before L2/OBI, ask whether *second-level stock* features can mark
those bursts causally.

Protocol:
  1) Stride AM seconds; attach causal stock 1s features at t
  2) Label forward ATM option oracle in the stock-move direction (H=30/60/120)
  3) Define buyer-good = oracle >= thr (e.g. +15%/+25%)
  4) Score rarity + precision/recall/lift of impact gates vs base rate / FO

Example:
  PYTHONPATH=. python -m maga7.tools.scan_buyer_impact_1s \\
    --tag research_buyer_impact_1s
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
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.scan_session_horizon_foresight import (
    _fwd_trade_rets_arr,
    _paths_by_ticker,
    _spot_at_arr,
)

DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"


def _impact_row(feat: dict[str, Any]) -> dict[str, float]:
    """Causal impact proxies from stock 1s snapshot."""
    r5 = float(feat.get("ret_15") or np.nan)  # use 15 as short; 5 may be missing
    r15 = float(feat.get("ret_15") or np.nan)
    r30 = float(feat.get("ret_30") or np.nan)
    r60 = float(feat.get("ret_60") or np.nan)
    volz = float(feat.get("vol_z") or np.nan)
    volr = float(feat.get("volume_ratio_60") or np.nan)
    mf = float(feat.get("mf100") or np.nan)
    # prefer true short window if present
    if "ret_15" in feat:
        r15 = float(feat.get("ret_15") or np.nan)
    out = {
        "abs_ret15": abs(r15) if np.isfinite(r15) else float("nan"),
        "abs_ret30": abs(r30) if np.isfinite(r30) else float("nan"),
        "abs_ret60": abs(r60) if np.isfinite(r60) else float("nan"),
        "vol_z": volz,
        "volume_ratio_60": volr,
        "abs_mf100": abs(mf) if np.isfinite(mf) else float("nan"),
        "ret15": r15,
        "ret30": r30,
        "ret60": r60,
        "mf100": mf,
        "from_open": float(feat.get("from_open") or np.nan),
    }
    # composite: large short ret + volume expansion
    parts = []
    if np.isfinite(out["abs_ret15"]):
        parts.append(out["abs_ret15"] / 0.001)  # scale ~1 per 10bp
    if np.isfinite(out["vol_z"]):
        parts.append(out["vol_z"])
    if np.isfinite(out["volume_ratio_60"]):
        parts.append(out["volume_ratio_60"])
    out["impact_score"] = float(np.nanmean(parts)) if parts else float("nan")
    return out


def _dir_from_ret(ret: float) -> str | None:
    if not np.isfinite(ret) or ret == 0:
        return None
    return "UP" if ret > 0 else "DN"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--tag", default="research_buyer_impact_1s")
    ap.add_argument("--window-start", default="09:30")
    ap.add_argument("--window-end", default="11:30")
    ap.add_argument("--stride-sec", type=int, default=10)
    ap.add_argument("--horizons", default="30,60,120")
    ap.add_argument("--oracle-thrs", default="0.10,0.15,0.25,0.40")
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--max-days", type=int, default=0, help="0=all; debug cap")
    ap.add_argument("--slip", type=float, default=0.01)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stock_1s = Path(prof["_paths"]["stock_1s_root"])
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(prof["_paths"]["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(prof, default=3)
    symbols = list(prof.get("symbols") or [])
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    oracle_thrs = [float(x) for x in args.oracle_thrs.split(",") if x.strip()]

    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date
    ]
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]
    print(
        f"buyer-impact 1s {args.window_start}-{args.window_end} "
        f"days={len(dates)} stride={args.stride_sec} H={horizons}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for di, date in enumerate(dates):
        if di % 5 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) probes={len(rows)}", flush=True)
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s, sym, date)
            if raw is None or raw.empty:
                continue
            sarr = prepare_day_arrays(raw)
            by_dte = lock.get((sym, date))
            if not by_dte:
                continue
            tday = load_option_trades(trades_root, sym, date)
            if tday is None or tday.empty:
                continue
            tpaths = _paths_by_ticker(tday)
            if not tpaths:
                continue
            ts_ns = sarr["ts_ns"]
            t0 = to_ny(pd.Timestamp(f"{date} {args.window_start}", tz=NY))
            t1 = to_ny(pd.Timestamp(f"{date} {args.window_end}", tz=NY))
            # leave room for longest horizon
            t_lim = t1 - pd.Timedelta(seconds=max(horizons))
            i0 = int(np.searchsorted(ts_ns, int(t0.value), side="left"))
            i1 = int(np.searchsorted(ts_ns, int(t_lim.value), side="right") - 1)
            if i1 <= i0:
                continue
            stride = max(1, int(args.stride_sec))
            for i in range(i0, i1 + 1, stride):
                t = pd.Timestamp(int(ts_ns[i]), tz="UTC").tz_convert(NY)
                feat = features_at(sarr, t)
                if feat is None:
                    continue
                imp = _impact_row(feat)
                # direction from short stock ret (causal)
                direction = _dir_from_ret(imp["ret15"])
                if direction is None:
                    direction = _dir_from_ret(imp["ret30"])
                if direction is None:
                    continue
                spot = float(feat.get("px") or np.nan)
                if not np.isfinite(spot):
                    spot_v = _spot_at_arr(sarr["ts_ns"], sarr["close"], t)
                    spot = float(spot_v) if spot_v is not None else float("nan")
                if not np.isfinite(spot):
                    continue
                ticker, dte, _ = resolve_open_lock_contract(
                    by_dte,
                    direction=direction,
                    moneyness="ATM",
                    spot=spot,
                    prefer_dte=0,
                    allowed_dte=(0, 1, 2),
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
                    t,
                    horizons,
                    slip=float(args.slip),
                )
                if not fwds:
                    continue
                by_h = {int(x["horizon_sec"]): x for x in fwds}
                fo = float(feat.get("from_open") or np.nan)
                # FO-like: |from_open| in direction
                fav_fo = fo if direction == "UP" else (-fo if np.isfinite(fo) else float("nan"))
                rec: dict[str, Any] = {
                    "date": date,
                    "symbol": sym,
                    "ts": str(t),
                    "dir": direction,
                    "ticker": str(ticker),
                    "dte": dte,
                    "fav_from_open": fav_fo,
                    **imp,
                }
                for H in horizons:
                    fr = by_h.get(H)
                    if fr is None:
                        continue
                    rec[f"oracle_h{H}"] = float(fr["oracle_ret"])
                    rec[f"clock_h{H}"] = float(fr["clock_ret"])
                    rec[f"mfe_h{H}"] = float(fr["mfe"])
                rows.append(rec)

    if not rows:
        print("no probes", flush=True)
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(out / "probes.csv", index=False)
    print(f"probes={len(df)}", flush=True)

    # --- base rates & lift tables ---
    lift_rows: list[dict[str, Any]] = []
    # gates: rarity targets ~ top 10% / 5% / 2% by impact_score, plus hard thresholds
    for H in horizons:
        col = f"oracle_h{H}"
        if col not in df.columns:
            continue
        sub = df[np.isfinite(df[col]) & np.isfinite(df["impact_score"])].copy()
        if sub.empty:
            continue
        base_mean = float(sub[col].mean())
        for thr in oracle_thrs:
            base_rate = float((sub[col] >= thr).mean())
            # percentile gates on impact_score
            for pct in (90, 95, 98):
                cut = float(sub["impact_score"].quantile(pct / 100.0))
                g = sub[sub["impact_score"] >= cut]
                if g.empty:
                    continue
                hit = float((g[col] >= thr).mean())
                lift_rows.append(
                    {
                        "horizon": H,
                        "oracle_thr": thr,
                        "gate": f"impact_p{pct}",
                        "cut": cut,
                        "n_gate": int(len(g)),
                        "frac_time": float(len(g) / len(sub)),
                        "base_rate": base_rate,
                        "hit_rate": hit,
                        "lift": hit / base_rate if base_rate > 0 else float("nan"),
                        "mean_oracle": float(g[col].mean()),
                        "base_mean_oracle": base_mean,
                        "recall": float(
                            ((sub[col] >= thr) & (sub["impact_score"] >= cut)).sum()
                            / max(1, (sub[col] >= thr).sum())
                        ),
                    }
                )
            # hard stock gates
            hard = [
                ("abs_ret15>=10bp", sub["abs_ret15"] >= 0.001),
                ("abs_ret15>=20bp", sub["abs_ret15"] >= 0.002),
                ("abs_ret30>=20bp", sub["abs_ret30"] >= 0.002),
                ("abs_ret30>=40bp", sub["abs_ret30"] >= 0.004),
                ("vol_z>=2", sub["vol_z"] >= 2.0),
                ("vol_z>=3", sub["vol_z"] >= 3.0),
                ("volr>=1.5", sub["volume_ratio_60"] >= 1.5),
                ("volr>=2", sub["volume_ratio_60"] >= 2.0),
                ("abs_ret15>=10bp+volz2", (sub["abs_ret15"] >= 0.001) & (sub["vol_z"] >= 2.0)),
                ("abs_ret15>=20bp+volz2", (sub["abs_ret15"] >= 0.002) & (sub["vol_z"] >= 2.0)),
                ("abs_ret30>=20bp+volr15", (sub["abs_ret30"] >= 0.002) & (sub["volume_ratio_60"] >= 1.5)),
            ]
            for name, mask in hard:
                g = sub[mask.fillna(False)]
                if len(g) < 20:
                    continue
                hit = float((g[col] >= thr).mean())
                lift_rows.append(
                    {
                        "horizon": H,
                        "oracle_thr": thr,
                        "gate": name,
                        "cut": float("nan"),
                        "n_gate": int(len(g)),
                        "frac_time": float(len(g) / len(sub)),
                        "base_rate": base_rate,
                        "hit_rate": hit,
                        "lift": hit / base_rate if base_rate > 0 else float("nan"),
                        "mean_oracle": float(g[col].mean()),
                        "base_mean_oracle": base_mean,
                        "recall": float(
                            ((sub[col] >= thr) & mask.fillna(False)).sum()
                            / max(1, (sub[col] >= thr).sum())
                        ),
                    }
                )
            # FO-like control: fav_from_open >= 0.8%
            fo_mask = sub["fav_from_open"] >= 0.008
            g = sub[fo_mask.fillna(False)]
            if len(g) >= 20:
                hit = float((g[col] >= thr).mean())
                lift_rows.append(
                    {
                        "horizon": H,
                        "oracle_thr": thr,
                        "gate": "FO>=0.8%",
                        "cut": 0.008,
                        "n_gate": int(len(g)),
                        "frac_time": float(len(g) / len(sub)),
                        "base_rate": base_rate,
                        "hit_rate": hit,
                        "lift": hit / base_rate if base_rate > 0 else float("nan"),
                        "mean_oracle": float(g[col].mean()),
                        "base_mean_oracle": base_mean,
                        "recall": float(
                            ((sub[col] >= thr) & fo_mask.fillna(False)).sum()
                            / max(1, (sub[col] >= thr).sum())
                        ),
                    }
                )

    lift = pd.DataFrame(lift_rows)
    lift.to_csv(out / "lift_table.csv", index=False)

    # Among buyer-good events, distribution of impact_score percentile
    recall_diag: list[dict[str, Any]] = []
    for H in horizons:
        col = f"oracle_h{H}"
        if col not in df.columns:
            continue
        sub = df[np.isfinite(df[col]) & np.isfinite(df["impact_score"])].copy()
        sub["impact_pct"] = sub["impact_score"].rank(pct=True)
        for thr in oracle_thrs:
            good = sub[sub[col] >= thr]
            if good.empty:
                continue
            recall_diag.append(
                {
                    "horizon": H,
                    "oracle_thr": thr,
                    "n_good": int(len(good)),
                    "frac_good": float(len(good) / len(sub)),
                    "p50_impact_pct": float(good["impact_pct"].median()),
                    "p25_impact_pct": float(good["impact_pct"].quantile(0.25)),
                    "frac_in_top10": float((good["impact_pct"] >= 0.90).mean()),
                    "frac_in_top5": float((good["impact_pct"] >= 0.95).mean()),
                    "frac_in_top2": float((good["impact_pct"] >= 0.98).mean()),
                    "frac_FO08": float((good["fav_from_open"] >= 0.008).mean()),
                }
            )
    recall_df = pd.DataFrame(recall_diag)
    recall_df.to_csv(out / "good_event_impact_pct.csv", index=False)

    # Pick best rare gates: frac_time in [0.02, 0.15], lift high, recall decent
    soft = lift[
        (lift["frac_time"] >= 0.02)
        & (lift["frac_time"] <= 0.15)
        & (lift["n_gate"] >= 50)
        & (lift["lift"] >= 1.3)
        & (lift["oracle_thr"] >= 0.15)
    ].sort_values(["lift", "recall"], ascending=[False, False])

    # summary for H=60 thr=0.25
    focus = lift[(lift.horizon == 60) & (lift.oracle_thr == 0.25)].sort_values(
        "lift", ascending=False
    )

    verdict = {
        "protocol": "stock_1s_impact_marks_option_buyer_windows",
        "n_probes": int(len(df)),
        "thesis": "Buyers need rare impact; can 1s stock features mark them?",
        "good_event_diag": recall_df.to_dict(orient="records"),
        "focus_h60_o25": focus.head(15).to_dict(orient="records"),
        "best_rare_lift": soft.head(20).to_dict(orient="records") if len(soft) else [],
    }
    (out / "summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    print("\n=== Among buyer-good events, where was impact_score? ===", flush=True)
    print(recall_df.to_string(index=False), flush=True)
    print("\n=== H=60 oracle>=25% gates by lift ===", flush=True)
    cols = [
        c
        for c in [
            "gate",
            "frac_time",
            "n_gate",
            "base_rate",
            "hit_rate",
            "lift",
            "recall",
            "mean_oracle",
        ]
        if c in focus.columns
    ]
    print(focus[cols].head(15).to_string(index=False), flush=True)
    print("\n=== Best rare lifts (2–15% time, lift>=1.3, thr>=15%) ===", flush=True)
    print(
        soft[
            [
                "horizon",
                "oracle_thr",
                "gate",
                "frac_time",
                "hit_rate",
                "lift",
                "recall",
                "mean_oracle",
            ]
        ].head(15).to_string(index=False)
        if len(soft)
        else "(none)",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
