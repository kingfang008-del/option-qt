#!/usr/bin/env python3
"""Causal A/B rules vs oracle pack on a single QQQ 0DTE day.

Rules (from foresight_rule_design on 2026-06-25):
  A_am_continuation:
    After ``am_start``, if |from_open| >= fo_min and short window still extends
    in the same direction, enter that dir once; hold H / trail.
  B_stretch_fade:
    After ``fade_start``, if day still stretched (|from_open| >= fo_fade) and
    short window reverses against the open (bounce), fade toward open once.

Compares catches vs oracle pack (≥50% / ≥100%) from
``qqq_oracle_day_opportunities_v1/<date>/``.
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
from maga7.common.replay import simulate_trade, to_ny
from maga7.tools.run_morning_sec_qqq_dte1 import BUCKET_ATM, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice

NY = "America/New_York"
OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
STOCK = Path("/mnt/s990/data/raw_1s/stocks")
ORACLE_ROOT = Path("/mnt/s990/data/maga7/results/qqq_oracle_day_opportunities_v1")


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
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _sim(path, entry_ts, *, direction, hold_sec, fill, tp, sl, trail: bool):
    kw: dict[str, Any] = dict(
        path=path,
        entry_ts=entry_ts,
        fill=fill,
        tp_mult=tp,
        sl_mult=sl,
        hold_minutes=max(1, int(np.ceil(hold_sec / 60.0))),
        direction=direction,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(hold_sec)),
        trade_toxic={"enabled": False},
        stock_bar_delay_seconds=0,
    )
    if trail:
        kw["exit_mode"] = "mtm_trail"
        kw["trail_activate"] = 0.15
        kw["trail_dd"] = 0.08
    else:
        kw["exit_mode"] = None
    return simulate_trade(**kw)


def _overlap(a0, a1, b0, b1) -> bool:
    return not (a1 <= b0 or a0 >= b1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2026-06-25")
    ap.add_argument("--am-start", default="09:40")
    ap.add_argument("--am-end", default="10:15")
    ap.add_argument("--fade-start", default="10:30")
    ap.add_argument("--fade-end", default="15:00")
    ap.add_argument("--fo-min", type=float, default=0.008)
    ap.add_argument("--fo-fade", type=float, default=0.010)
    ap.add_argument("--extend-bp", type=float, default=0.0010, help="30s extend confirm")
    ap.add_argument("--fade-bp", type=float, default=0.0010, help="30s reverse confirm")
    ap.add_argument("--hold-sec", type=int, default=600)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--max-trades", type=int, default=3)
    ap.add_argument("--trail", action="store_true", default=True)
    ap.add_argument("--no-trail", action="store_true")
    ap.add_argument("--tp-mult", type=float, default=1.6)
    ap.add_argument("--sl-mult", type=float, default=0.45)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/qqq_foresight_rules_day_v1",
    )
    args = ap.parse_args()
    date = str(args.date)
    use_trail = bool(args.trail) and not bool(args.no_trail)
    out = Path(args.out) / date
    out.mkdir(parents=True, exist_ok=True)

    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    stock = _morning_slice(load_stock_1s_day(STOCK, "QQQ", date), start="09:30", end="16:00")
    stock = stock.copy()
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True).dt.tz_convert(NY)
    stock = stock.sort_values("timestamp").reset_index(drop=True)
    s_ts = pd.DatetimeIndex(stock["timestamp"])
    s_px = stock["close"].astype(float).to_numpy()
    open_px = float(s_px[0])

    paths = {}
    for d in ("UP", "DN"):
        p, ticker, strike = _load_atm_path(OPT, date, d)
        paths[d] = {"path": _prep_path(p), "ticker": ticker, "strike": strike}
        if paths[d]["path"] is None:
            raise SystemExit(f"missing path {d}")

    am0 = pd.Timestamp(f"{date} {args.am_start}", tz=NY)
    am1 = pd.Timestamp(f"{date} {args.am_end}", tz=NY)
    f0 = pd.Timestamp(f"{date} {args.fade_start}", tz=NY)
    f1 = pd.Timestamp(f"{date} {args.fade_end}", tz=NY)
    grid = pd.date_range(am0, f1, freq=f"{int(args.stride_sec)}s", tz=NY)

    trades: list[dict] = []
    did_A = False
    did_B = False
    cooldown_until: pd.Timestamp | None = None

    for t in grid:
        if len(trades) >= int(args.max_trades):
            break
        if cooldown_until is not None and t < cooldown_until:
            continue
        j = int(s_ts.searchsorted(t, side="right")) - 1
        if j < 60:
            continue
        S = float(s_px[j])
        fo = S / open_px - 1.0
        r30 = S / float(s_px[j - 30]) - 1.0
        family = None
        direction = None

        # --- Rule A: AM continuation ---
        if (not did_A) and (am0 <= t <= am1) and abs(fo) >= float(args.fo_min):
            # still extending with open
            if fo > 0 and r30 >= float(args.extend_bp):
                family, direction = "A_am_continuation", "UP"
            elif fo < 0 and r30 <= -float(args.extend_bp):
                family, direction = "A_am_continuation", "DN"

        # --- Rule B: stretch fade (against open) ---
        if (
            family is None
            and (not did_B)
            and (f0 <= t <= f1)
            and abs(fo) >= float(args.fo_fade)
        ):
            # day still red/green but short window reverses toward open
            if fo < 0 and r30 >= float(args.fade_bp):
                family, direction = "B_stretch_fade", "UP"
            elif fo > 0 and r30 <= -float(args.fade_bp):
                family, direction = "B_stretch_fade", "DN"

        if family is None or direction is None:
            continue

        path = paths[direction]["path"]
        after = path[path["timestamp"] >= t]
        if after.empty:
            continue
        lag = (to_ny(after.iloc[0]["timestamp"]) - t).total_seconds()
        if lag > 3:
            continue

        sim = _sim(
            path,
            t,
            direction=direction,
            hold_sec=int(args.hold_sec),
            fill=fill,
            tp=float(args.tp_mult),
            sl=float(args.sl_mult),
            trail=use_trail,
        )
        if sim is None:
            continue
        reason = str(sim.reason)
        if reason == "DISPLACE":
            reason = f"H{args.hold_sec}"
        exit_ts = to_ny(sim.exit_ts)
        trades.append(
            {
                "date": date,
                "family": family,
                "tod": t.strftime("%H:%M:%S"),
                "entry_ts": t,
                "exit_ts": exit_ts,
                "direction": direction,
                "from_open": float(fo),
                "ret_30": float(r30),
                "ticker": paths[direction]["ticker"],
                "strike": paths[direction]["strike"],
                "ret": float(sim.ret),
                "reason": reason,
                "held_sec": float((exit_ts - t).total_seconds()),
                "entry": float(sim.entry),
                "exit": float(sim.exit),
            }
        )
        if family.startswith("A"):
            did_A = True
        if family.startswith("B"):
            did_B = True
        cooldown_until = exit_ts  # no overlap

    trdf = pd.DataFrame(trades)
    trdf.to_csv(out / "trades.csv", index=False)

    # load oracle packs if present
    odir = ORACLE_ROOT / date
    packs = {}
    for name in ("pack_oracle_ge_50", "pack_oracle_ge_100", "pack_clock_ge_50"):
        p = odir / f"{name}.csv"
        if p.is_file():
            packs[name] = pd.read_csv(p)
        else:
            # fallback names from earlier run
            alt = {
                "pack_oracle_ge_50": "pack_oracle.csv",
                "pack_clock_ge_50": "pack_clock.csv",
            }.get(name)
            if alt and (odir / alt).is_file():
                packs[name] = pd.read_csv(odir / alt)

    def match_pack(pack: pd.DataFrame, tol_sec: float = 180.0) -> dict:
        if pack is None or pack.empty or trdf.empty:
            return {"n_oracle": 0 if pack is None else int(len(pack)), "n_caught": 0, "caught": [], "missed": []}
        pack = pack.copy()
        pack["t"] = pd.to_datetime(pack["t"])
        caught = []
        missed = []
        used = set()
        for i, o in pack.iterrows():
            ot = pd.Timestamp(o["t"])
            if ot.tzinfo is None:
                ot = ot.tz_localize(NY)
            else:
                ot = ot.tz_convert(NY)
            hit = None
            for j, tr in trdf.iterrows():
                if j in used:
                    continue
                et = pd.Timestamp(tr["entry_ts"])
                if et.tzinfo is None:
                    et = et.tz_localize(NY)
                if abs((et - ot).total_seconds()) <= tol_sec and tr["direction"] == o.get("direction"):
                    hit = tr
                    used.add(j)
                    break
            row = {
                "oracle_tod": ot.strftime("%H:%M:%S"),
                "oracle_dir": o.get("direction"),
                "oracle_ret": float(o.get("oracle_ret") or o.get("clock_ret") or 0),
            }
            if hit is not None:
                row.update(
                    {
                        "caught": True,
                        "rule_tod": hit["tod"],
                        "rule_family": hit["family"],
                        "rule_ret": float(hit["ret"]),
                    }
                )
                caught.append(row)
            else:
                row["caught"] = False
                missed.append(row)
        return {
            "n_oracle": int(len(pack)),
            "n_caught": int(len(caught)),
            "n_missed": int(len(missed)),
            "caught": caught,
            "missed": missed[:20],
            "catch_rate": float(len(caught) / len(pack)) if len(pack) else None,
        }

    vs = {k: match_pack(v) for k, v in packs.items()}

    # also: did we catch the open_cont style morning winner?
    morning = trdf[trdf["family"] == "A_am_continuation"] if not trdf.empty else trdf
    fade = trdf[trdf["family"] == "B_stretch_fade"] if not trdf.empty else trdf

    summary = {
        "date": date,
        "params": {
            "fo_min": args.fo_min,
            "fo_fade": args.fo_fade,
            "extend_bp": args.extend_bp,
            "fade_bp": args.fade_bp,
            "hold_sec": args.hold_sec,
            "trail": use_trail,
            "am": f"{args.am_start}-{args.am_end}",
            "fade": f"{args.fade_start}-{args.fade_end}",
            "max_trades": args.max_trades,
        },
        "n_trades": int(len(trdf)),
        "sum_ret": float(trdf["ret"].sum()) if len(trdf) else 0.0,
        "mean_ret": float(trdf["ret"].mean()) if len(trdf) else None,
        "trades": trades,
        "A": {
            "n": int(len(morning)),
            "sum_ret": float(morning["ret"].sum()) if len(morning) else 0.0,
            "rows": morning.to_dict(orient="records") if len(morning) else [],
        },
        "B": {
            "n": int(len(fade)),
            "sum_ret": float(fade["ret"].sum()) if len(fade) else 0.0,
            "rows": fade.to_dict(orient="records") if len(fade) else [],
        },
        "vs_oracle": vs,
        "verdict_notes": [],
    }
    notes = summary["verdict_notes"]
    notes.append(
        f"Causal book took {summary['n_trades']} trades, sum_ret={summary['sum_ret']:+.2%} "
        f"(A={summary['A']['sum_ret']:+.2%}, B={summary['B']['sum_ret']:+.2%})."
    )
    if "pack_oracle_ge_100" in vs:
        v = vs["pack_oracle_ge_100"]
        notes.append(
            f"vs oracle≥100% pack: caught {v['n_caught']}/{v['n_oracle']} "
            f"(rate={v['catch_rate']})."
        )
    if "pack_oracle_ge_50" in vs:
        v = vs["pack_oracle_ge_50"]
        notes.append(
            f"vs oracle≥50% pack: caught {v['n_caught']}/{v['n_oracle']} "
            f"(rate={v['catch_rate']}) — expected low; budget is 2–3 swings not 27."
        )
    notes.append(
        "Success criterion: catch the AM continuation (A) with large positive ret; "
        "optional B adds one fade without wrecking A."
    )

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
