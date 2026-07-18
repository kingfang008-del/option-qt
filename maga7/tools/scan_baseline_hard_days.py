#!/usr/bin/env python3
"""Scan freeze-baseline hard days and attribute failure day_types.

Writes under ``maga7/results/regime_router/``:
  - baseline_daily_scan.csv
  - bad_day_features.csv
  - bad_day_scan_summary.json

See ``maga7/docs/regime_router_research.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import month_list, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files


def _load_daily() -> pd.DataFrame:
    paths = [
        ROOT / "maga7/results/research_extend_mtm_full_day_peer3_daily_by_month/daily_2025h2.csv",
        ROOT / "maga7/results/research_extend_mtm_full_day_peer3_daily_by_month/daily_all.csv",
        ROOT / "maga7/results/research_extend_mtm_full_day_peer3_may_jul_to_0717/daily.csv",
    ]
    frames = []
    prio = {"daily.csv": 3, "daily_all.csv": 2, "daily_2025h2.csv": 1}
    for p in paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["date"] = d["date"].astype(str)
        d["src"] = p.name
        d["_p"] = prio.get(p.name, 0)
        frames.append(d[["date", "equity", "day_ret", "n", "day_halt", "src", "_p"]])
    if not frames:
        raise SystemExit("no daily csv found")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["date", "_p"]).drop_duplicates("date", keep="last")
    return all_df.sort_values("date").reset_index(drop=True)


def _session_feats(df: pd.DataFrame, date: str, asof_hhmm: str = "10:30") -> dict | None:
    day = df[df["date"].astype(str) == str(date)].sort_values("timestamp")
    if day.empty:
        return None
    ts = pd.to_datetime(day["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    day = day.copy()
    day["_ts"] = ts
    open_px = float(day.iloc[0]["open"] if "open" in day.columns else day.iloc[0]["close"])
    asof = pd.Timestamp(f"{date} {asof_hhmm}", tz="America/New_York")
    upto = day[day["_ts"] <= asof]
    if upto.empty:
        return None
    close = pd.to_numeric(upto["close"], errors="coerce")
    px = float(close.iloc[-1])
    lo = float(pd.to_numeric(upto["low"] if "low" in upto.columns else upto["close"], errors="coerce").min())
    hi = float(pd.to_numeric(upto["high"] if "high" in upto.columns else upto["close"], errors="coerce").max())
    day_close = float(day.iloc[-1]["close"])
    bounce = px / lo - 1.0 if lo > 0 else np.nan
    return {
        "qqq_open_to_1030": px / open_px - 1.0,
        "qqq_above_open_1030": int(px > open_px),
        "qqq_bounce_lod_1030": bounce,
        "qqq_range_1030": (hi - lo) / open_px if open_px else np.nan,
        "qqq_day_ret": day_close / open_px - 1.0,
    }


def _day_type(r: pd.Series) -> str:
    if r.qqq_above_open_1030 and r.qqq_bounce_lod_1030 >= 0.005 and r.n_dn > 0 and r.sum_ret_dn < 0:
        return "rebound_trap_dn"
    if r.n_dn > 0 and r.sum_ret_dn < -0.2 and r.sum_ret_up > -0.05:
        return "dn_toxic"
    if r.n_up > 0 and r.sum_ret_up < -0.2 and r.sum_ret_dn >= -0.05:
        return "up_toxic"
    if pd.notna(r.qqq_range_1030) and float(r.qqq_range_1030) >= 0.015:
        return "wide_chop"
    return "other_loss"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bad-thr", type=float, default=-0.03)
    ap.add_argument("--out", default="maga7/results/regime_router")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    daily = _load_daily()
    daily.to_csv(out / "baseline_daily_scan.csv", index=False)

    bad = daily[(daily["day_ret"] <= float(args.bad_thr)) & (daily["n"] > 0)].copy()
    ok = daily[(daily["day_ret"] >= abs(float(args.bad_thr))) & (daily["n"] > 0)].copy()

    trade_paths = list((ROOT / "maga7/results").glob("research_extend_mtm_full_day_peer3*/trades.csv"))
    tr_frames = []
    for p in trade_paths:
        t = pd.read_csv(p)
        t["date"] = t["date"].astype(str)
        t["_src"] = p.parent.name
        tr_frames.append(t)
    trades = pd.concat(tr_frames, ignore_index=True) if tr_frames else pd.DataFrame()
    if not trades.empty:
        trades["_p"] = trades["_src"].map(lambda s: 3 if "to_0717" in s else 1)
        trades = trades.sort_values(["date", "_p"]).drop_duplicates(
            ["date", "symbol", "entry_ts"], keep="last"
        )

    prof = load_profile(
        "maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
    )
    start, end = str(daily["date"].min()), str(daily["date"].max())
    qqq_raw = load_stock_month_files(prof["_paths"]["stock_root"], "QQQ", month_list(start, end))
    qqq_raw = qqq_raw[(qqq_raw["date"] >= start) & (qqq_raw["date"] <= end)]
    qqq = attach_mf_features(qqq_raw, mf_window=10, vol_ma_window=20)

    rows = []
    ok_sample = ok.sample(n=min(len(ok), max(len(bad) * 2, 1)), random_state=7) if len(ok) else ok
    for label, part in [("bad", bad), ("ok", ok_sample)]:
        for r in part.itertuples():
            f = _session_feats(qqq, r.date)
            if f is None:
                continue
            td = trades[trades["date"] == r.date] if not trades.empty else pd.DataFrame()
            n_up = int((td["dir"] == "UP").sum()) if len(td) else 0
            n_dn = int((td["dir"] == "DN").sum()) if len(td) else 0
            ret_up = float(td.loc[td["dir"] == "UP", "ret"].sum()) if n_up else 0.0
            ret_dn = float(td.loc[td["dir"] == "DN", "ret"].sum()) if n_dn else 0.0
            rows.append(
                {
                    "date": r.date,
                    "label": label,
                    "day_ret": float(r.day_ret),
                    "n": int(r.n),
                    **f,
                    "n_up": n_up,
                    "n_dn": n_dn,
                    "sum_ret_up": ret_up,
                    "sum_ret_dn": ret_dn,
                }
            )
    feat = pd.DataFrame(rows)
    if not feat.empty:
        mask = feat["label"] == "bad"
        feat.loc[mask, "day_type"] = feat.loc[mask].apply(_day_type, axis=1)
        feat.loc[~mask, "day_type"] = "ok"
    feat.to_csv(out / "bad_day_features.csv", index=False)

    bad_f = feat[feat["label"] == "bad"] if not feat.empty else pd.DataFrame()
    meta = {
        "range": [start, end],
        "n_days": int(len(daily)),
        "n_trade_days": int((daily["n"] > 0).sum()),
        "bad_thr": float(args.bad_thr),
        "n_bad": int(len(bad)),
        "bad_type_counts": bad_f["day_type"].value_counts().to_dict() if len(bad_f) else {},
        "rebound_trap_dates": sorted(bad_f.loc[bad_f["day_type"] == "rebound_trap_dn", "date"].tolist())
        if len(bad_f)
        else [],
        "doc": "maga7/docs/regime_router_research.md",
    }
    (out / "bad_day_scan_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
