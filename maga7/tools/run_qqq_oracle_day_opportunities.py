#!/usr/bin/env python3
"""Single-day QQQ 0DTE oracle opportunity map (foresight → rule design).

For one RTH day:
  1) Scan every ``stride_sec`` from ``scan_start``..``scan_end``
  2) At each t, price ATM call (UP) and put (DN) with fill costs
  3) Oracle: best (direction, horizon) forward option ret (no look-ahead in fill,
     but choice of which window is foresight)
  4) Greedy non-overlap packing → how many distinct swings a perfect picker catches
  5) Characterize winning windows to suggest causal rule features

Example:
  python -m maga7.tools.run_qqq_oracle_day_opportunities --date 2026-06-25
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
from maga7.common.fills import FillSpec
from maga7.tools.run_morning_sec_qqq_dte1 import BUCKET_ATM, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice

NY = "America/New_York"
OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
STOCK = Path("/mnt/s990/data/raw_1s/stocks")


def _prep_path(path: pd.DataFrame | None) -> pd.DataFrame | None:
    if path is None or path.empty:
        return None
    out = path.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.dt.tz_convert(NY)
    out["timestamp"] = ts
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out["mid"] = (out["bid"].astype(float) + out["ask"].astype(float)) * 0.5
    return out.reset_index(drop=True)


def _forward_fill_ret(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    hold_sec: int,
    fill: FillSpec,
) -> dict[str, Any] | None:
    after = path[path["timestamp"] >= entry_ts]
    if after.empty:
        return None
    r0 = after.iloc[0]
    lag = (r0["timestamp"] - entry_ts).total_seconds()
    if lag > 3:
        return None
    entry = fill.buy(float(r0["bid"]), float(r0["ask"]))
    if not np.isfinite(entry) or entry <= 0:
        return None
    end = entry_ts + pd.Timedelta(seconds=int(hold_sec))
    win = after[after["timestamp"] <= end]
    if win.empty:
        return None
    # oracle within window: best exit sell (foresight) AND clock exit
    sells = fill.sell_series(win["bid"].astype(float), win["ask"].astype(float))
    sells = np.asarray(sells, dtype=float)
    clock_exit = float(sells[-1])
    best_i = int(np.nanargmax(sells))
    best_exit = float(sells[best_i])
    best_ts = win.iloc[best_i]["timestamp"]
    mids = win["mid"].astype(float).to_numpy()
    mfe = float(np.nanmax(mids) / entry - 1.0) if entry > 0 else np.nan
    mae = float(np.nanmin(mids) / entry - 1.0) if entry > 0 else np.nan
    return {
        "entry": entry,
        "entry_ts": r0["timestamp"],
        "entry_lag": float(lag),
        "clock_ret": float(clock_exit / entry - 1.0),
        "oracle_ret": float(best_exit / entry - 1.0),
        "oracle_exit_ts": best_ts,
        "oracle_hold_sec": float((best_ts - r0["timestamp"]).total_seconds()),
        "mfe": mfe,
        "mae": mae,
        "n_quotes": int(len(win)),
    }


def _greedy_pack(cands: pd.DataFrame, ret_col: str, min_ret: float) -> pd.DataFrame:
    """Non-overlapping pack by descending ret; occupancy = [entry, entry+hold]."""
    if cands.empty:
        return cands.iloc[0:0]
    df = cands[cands[ret_col] >= min_ret].sort_values(ret_col, ascending=False)
    picked = []
    busy: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, r in df.iterrows():
        a = pd.Timestamp(r["t"])
        b = a + pd.Timedelta(seconds=float(r["hold_sec"]))
        if any(not (b <= x0 or a >= x1) for x0, x1 in busy):
            continue
        picked.append(r)
        busy.append((a, b))
    return pd.DataFrame(picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--scan-start", default="09:35")
    ap.add_argument("--scan-end", default="15:30")
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--horizons", default="60,120,180,300,600")
    ap.add_argument("--min-oracle-ret", type=float, default=0.08)
    ap.add_argument("--min-clock-ret", type=float, default=0.05)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/qqq_oracle_day_opportunities_v1",
    )
    args = ap.parse_args()
    date = str(args.date)
    out = Path(args.out) / date
    out.mkdir(parents=True, exist_ok=True)

    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    stock = _morning_slice(load_stock_1s_day(STOCK, "QQQ", date), start="09:30", end="16:00")
    if stock.empty:
        raise SystemExit(f"no stock {date}")
    stock = stock.copy()
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True).dt.tz_convert(NY)
    stock = stock.sort_values("timestamp").reset_index(drop=True)
    s_ts = pd.DatetimeIndex(stock["timestamp"])
    s_px = stock["close"].astype(float).to_numpy()
    open_px = float(s_px[0])

    paths = {}
    for direction in ("UP", "DN"):
        p, ticker, strike = _load_atm_path(OPT, date, direction)
        paths[direction] = {
            "path": _prep_path(p),
            "ticker": ticker,
            "strike": strike,
            "bucket": BUCKET_ATM[direction],
        }
        if paths[direction]["path"] is None:
            raise SystemExit(f"missing ATM {direction} path for {date}")

    t0 = pd.Timestamp(f"{date} {args.scan_start}", tz=NY)
    t1 = pd.Timestamp(f"{date} {args.scan_end}", tz=NY)
    grid = pd.date_range(t0, t1, freq=f"{int(args.stride_sec)}s", tz=NY)

    rows: list[dict[str, Any]] = []
    print(
        f"scan {date} n_grid={len(grid)} horizons={horizons} stride={args.stride_sec}s",
        flush=True,
    )
    for k, t in enumerate(grid):
        j = int(s_ts.searchsorted(t, side="right")) - 1
        if j < 30:
            continue
        S = float(s_px[j])
        # causal stock features at t (no foresight)
        look = min(j, 300)
        seg = s_px[j - look : j + 1]
        ret_30 = float(s_px[j] / s_px[j - 30] - 1.0) if j >= 30 else np.nan
        ret_60 = float(s_px[j] / s_px[j - 60] - 1.0) if j >= 60 else np.nan
        ret_180 = float(s_px[j] / s_px[j - 180] - 1.0) if j >= 180 else np.nan
        from_open = float(S / open_px - 1.0)
        # short-horizon realized vol proxy
        if look >= 60:
            r = np.diff(np.log(np.maximum(seg[-61:], 1e-9)))
            vol60 = float(np.std(r) * np.sqrt(390 * 60))  # rough annualized-ish
        else:
            vol60 = np.nan

        best_clock = None
        best_oracle = None
        for direction in ("UP", "DN"):
            path = paths[direction]["path"]
            for H in horizons:
                sim = _forward_fill_ret(path, t, H, fill)
                if sim is None:
                    continue
                rec = {
                    "t": t,
                    "tod": t.strftime("%H:%M:%S"),
                    "direction": direction,
                    "hold_sec": H,
                    "S": S,
                    "from_open": from_open,
                    "ret_30": ret_30,
                    "ret_60": ret_60,
                    "ret_180": ret_180,
                    "vol60": vol60,
                    "ticker": paths[direction]["ticker"],
                    "strike": paths[direction]["strike"],
                    **sim,
                }
                rows.append(rec)
                if best_clock is None or rec["clock_ret"] > best_clock["clock_ret"]:
                    best_clock = rec
                if best_oracle is None or rec["oracle_ret"] > best_oracle["oracle_ret"]:
                    best_oracle = rec
        if (k + 1) % 500 == 0:
            print(f"  grid {k+1}/{len(grid)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no sims")
    df.to_parquet(out / "all_windows.parquet", index=False)

    # per-t best summary
    best_clock_by_t = (
        df.sort_values("clock_ret", ascending=False)
        .groupby("t", as_index=False)
        .first()
    )
    best_oracle_by_t = (
        df.sort_values("oracle_ret", ascending=False)
        .groupby("t", as_index=False)
        .first()
    )
    best_clock_by_t.to_csv(out / "best_clock_by_t.csv", index=False)
    best_oracle_by_t.to_csv(out / "best_oracle_by_t.csv", index=False)

    pack_clock = _greedy_pack(best_clock_by_t, "clock_ret", float(args.min_clock_ret))
    pack_oracle = _greedy_pack(best_oracle_by_t, "oracle_ret", float(args.min_oracle_ret))
    pack_clock.to_csv(out / "pack_clock.csv", index=False)
    pack_oracle.to_csv(out / "pack_oracle.csv", index=False)

    # hour buckets
    def hour_stats(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        f = frame.copy()
        f["hour"] = pd.to_datetime(f["t"]).dt.strftime("%H")
        g = f.groupby("hour")[col].agg(["count", "mean", "median", lambda x: float((x > 0).mean())])
        g.columns = ["n", "mean", "median", "win"]
        return g.reset_index()

    # causal alignment: among oracle winners, how often stock ret_30 agrees with dir
    ow = best_oracle_by_t[best_oracle_by_t["oracle_ret"] >= float(args.min_oracle_ret)].copy()
    if not ow.empty:
        ow["stock_sign"] = np.sign(ow["ret_30"].astype(float))
        ow["dir_sign"] = ow["direction"].map({"UP": 1.0, "DN": -1.0})
        ow["align_30"] = ow["stock_sign"] * ow["dir_sign"] > 0
        ow["align_60"] = np.sign(ow["ret_60"]) * ow["dir_sign"] > 0
        ow["align_open"] = np.sign(ow["from_open"]) * ow["dir_sign"] > 0

    # open_cont_0945 baseline for this day
    t0945 = pd.Timestamp(f"{date} 09:45", tz=NY)
    j = int(s_ts.searchsorted(t0945, side="right")) - 1
    fo = float(s_px[j] / open_px - 1.0) if j >= 0 else np.nan
    oc_dir = "UP" if fo > 0 else "DN"
    oc_rows = []
    for H in horizons:
        sim = _forward_fill_ret(paths[oc_dir]["path"], t0945, H, fill)
        if sim:
            oc_rows.append({"rule": "open_cont_0945", "direction": oc_dir, "from_open": fo, "hold_sec": H, **sim})

    # rule candidates from foresight: enter when |ret_30| large and aligned, hold 180
    rule_hits = []
    for _, r in best_clock_by_t.iterrows():
        # candidate causal gate
        r30 = float(r["ret_30"]) if np.isfinite(r["ret_30"]) else 0.0
        if abs(r30) < 0.0015:  # 15bp / 30s
            continue
        direction = "UP" if r30 > 0 else "DN"
        # use the clock ret for that direction/hold preferentially H=180
        sub = df[(df["t"] == r["t"]) & (df["direction"] == direction) & (df["hold_sec"] == 180)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if float(row["clock_ret"]) < float(args.min_clock_ret):
            continue
        rule_hits.append(row.to_dict())
    rule_df = pd.DataFrame(rule_hits)
    pack_rule = _greedy_pack(rule_df, "clock_ret", float(args.min_clock_ret)) if not rule_df.empty else rule_df

    summary = {
        "date": date,
        "scan_start": args.scan_start,
        "scan_end": args.scan_end,
        "stride_sec": int(args.stride_sec),
        "horizons": horizons,
        "n_grid_evaluated": int(best_clock_by_t["t"].nunique()),
        "n_window_sims": int(len(df)),
        "stock_open": open_px,
        "stock_close_ret": float(s_px[-1] / open_px - 1.0),
        "stock_range_pct": float((s_px.max() - s_px.min()) / open_px),
        "oracle": {
            "min_ret": float(args.min_oracle_ret),
            "n_t_above": int((best_oracle_by_t["oracle_ret"] >= args.min_oracle_ret).sum()),
            "pack_n": int(len(pack_oracle)),
            "pack_sum_ret": float(pack_oracle["oracle_ret"].sum()) if len(pack_oracle) else 0.0,
            "pack_mean_ret": float(pack_oracle["oracle_ret"].mean()) if len(pack_oracle) else None,
            "tod_list": pack_oracle["tod"].tolist() if len(pack_oracle) else [],
            "dirs": pack_oracle["direction"].value_counts().to_dict() if len(pack_oracle) else {},
            "holds": pack_oracle["hold_sec"].value_counts().to_dict() if len(pack_oracle) else {},
        },
        "clock": {
            "min_ret": float(args.min_clock_ret),
            "n_t_above": int((best_clock_by_t["clock_ret"] >= args.min_clock_ret).sum()),
            "pack_n": int(len(pack_clock)),
            "pack_sum_ret": float(pack_clock["clock_ret"].sum()) if len(pack_clock) else 0.0,
            "pack_mean_ret": float(pack_clock["clock_ret"].mean()) if len(pack_clock) else None,
            "tod_list": pack_clock["tod"].tolist() if len(pack_clock) else [],
        },
        "open_cont_0945": oc_rows,
        "causal_impulse_rule": {
            "desc": "|ret_30|>=15bp then trade that dir for H=180 if clock_ret>=min",
            "n_raw_hits": int(len(rule_df)),
            "pack_n": int(len(pack_rule)),
            "pack_sum_clock_ret": float(pack_rule["clock_ret"].sum()) if len(pack_rule) else 0.0,
            "tod_list": pack_rule["tod"].tolist() if len(pack_rule) else [],
        },
        "oracle_winner_alignment": {
            "n": int(len(ow)),
            "align_ret30": float(ow["align_30"].mean()) if len(ow) else None,
            "align_ret60": float(ow["align_60"].mean()) if len(ow) else None,
            "align_from_open": float(ow["align_open"].mean()) if len(ow) else None,
            "median_abs_ret30": float(ow["ret_30"].abs().median()) if len(ow) else None,
            "median_abs_from_open": float(ow["from_open"].abs().median()) if len(ow) else None,
            "hour_hist": ow.assign(hour=pd.to_datetime(ow["t"]).dt.strftime("%H"))
            .groupby("hour")
            .size()
            .to_dict()
            if len(ow)
            else {},
        },
        "rule_design_notes": [],
    }

    # design notes from data
    notes = summary["rule_design_notes"]
    notes.append(
        f"Oracle pack catches {summary['oracle']['pack_n']} non-overlap swings "
        f"(sum oracle_ret={summary['oracle']['pack_sum_ret']:+.2f}) vs open_cont single-shot."
    )
    if summary["oracle_winner_alignment"]["align_ret30"] is not None:
        a30 = summary["oracle_winner_alignment"]["align_ret30"]
        notes.append(
            f"Among oracle winners (≥{args.min_oracle_ret:.0%}), ret_30 direction aligns {a30:.0%} — "
            + ("impulse-follow is the right family." if a30 >= 0.6 else "impulse-follow is weak; need other gates.")
        )
    notes.append(
        f"Clock pack (fixed hold, no perfect exit) still finds {summary['clock']['pack_n']} swings "
        f"sum={summary['clock']['pack_sum_ret']:+.2f}; this is the realistic upper band for rules."
    )
    notes.append(
        f"Causal probe |ret_30|≥15bp→H180 packs {summary['causal_impulse_rule']['pack_n']} "
        f"sum_clock={summary['causal_impulse_rule']['pack_sum_clock_ret']:+.2f}."
    )
    if oc_rows:
        best_oc = max(oc_rows, key=lambda x: x["clock_ret"])
        notes.append(
            f"open_cont_0945 on this day: dir={best_oc['direction']} fo={fo:+.3%} "
            f"best clock H{best_oc['hold_sec']} ret={best_oc['clock_ret']:+.3%} "
            f"(one shot vs pack_n={summary['clock']['pack_n']})."
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if len(pack_rule):
        pack_rule.to_csv(out / "pack_causal_impulse.csv", index=False)
    if len(ow):
        ow.to_csv(out / "oracle_winners.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
