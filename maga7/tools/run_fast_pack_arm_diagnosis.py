#!/usr/bin/env python3
"""Day-level diagnosis: when should path_fast_pack arm?

For each session day:
  - wash / opt_chop / AND / OR arm flags @10:30
  - lit vs wash_fast day_ret (does fast help or hurt?)
  - label miss (helps but not AND) / false_alarm (hurts but AND)

Default window: 2026-05-01 .. 2026-07-20 (QQQ bucketed through Jul).
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.option_surface import load_surface_range, opt_chop_score, surface_asof
from maga7.common.path_fast_pack import (
    PathFastPackConfig,
    _opt_chop_hit,
    _wash_hit,
    path_fast_pack_from_trade,
)

NY = "America/New_York"
STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
LIT_DAILY = Path(
    "/mnt/s990/data/maga7/results/path_hold_wash_fast_v1/may_jul__lit_always/daily.csv"
)
WASH_DAILY = Path(
    "/mnt/s990/data/maga7/results/path_hold_wash_fast_v1/may_jul__lit_wash_fast/daily.csv"
)
AND_DAILY = Path(
    "/mnt/s990/data/maga7/results/path_hold_opt_chop_may_jun_v1/wash_and_opt/daily.csv"
)
OUT = Path("/mnt/s990/data/maga7/results/fast_pack_arm_diagnosis_v1")


def _prep_stock(df: pd.DataFrame | None, date: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    try:
        ts = ts.dt.tz_convert(NY)
    except Exception:
        pass
    out["timestamp"] = ts
    out["date"] = ts.dt.strftime("%Y-%m-%d")
    return out


def _load_day_stocks(date: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        raw = load_stock_1s_day(STOCK_ROOT, sym, date)
        prep = _prep_stock(raw, date)
        if prep is not None and not prep.empty:
            stock_by[sym] = prep
    qqq = _prep_stock(load_stock_1s_day(STOCK_ROOT, "QQQ", date), date)
    return stock_by, qqq


def diagnose_day(
    date: str,
    cfg: PathFastPackConfig,
    *,
    lit_ret: float | None,
    wash_ret: float | None,
    and_ret: float | None,
    help_eps: float,
) -> dict:
    stock_by, qqq = _load_day_stocks(date)
    wash_kw = dict(
        date=date,
        stock_by=stock_by,
        qqq_df=qqq,
        symbols=SYMBOLS,
        asof="10:30",
        washout_breadth_min=3,
        wash_drop_min=0.008,
        frac_above_min=0.35,
        frac_above_max=0.70,
    )
    wash = bool(_wash_hit(cfg, **wash_kw)) if stock_by else False
    opt = bool(_opt_chop_hit(cfg, date=date, asof="10:30"))
    snap = None
    try:
        from maga7.common.path_fast_pack import _asof_snap

        snap = _asof_snap(cfg, date, "10:30")
    except Exception:
        snap = None
    imb = float(snap.get("options_vw_imbalance")) if snap else float("nan")
    chop = float(opt_chop_score(snap) or float("nan")) if snap else float("nan")

    help_delta = None
    if lit_ret is not None and wash_ret is not None:
        help_delta = float(wash_ret - lit_ret)
    label = "neutral"
    if help_delta is not None:
        if help_delta > help_eps:
            label = "fast_helps"
        elif help_delta < -help_eps:
            label = "fast_hurts"

    and_arm = wash and opt
    or_arm = wash or opt
    miss = label == "fast_helps" and not and_arm
    false_alarm = label == "fast_hurts" and and_arm
    wash_miss = label == "fast_helps" and not wash
    wash_fa = label == "fast_hurts" and wash

    return {
        "date": date,
        "wash": wash,
        "opt": opt,
        "and_arm": and_arm,
        "or_arm": or_arm,
        "imb": imb,
        "chop_score": chop,
        "lit_ret": lit_ret,
        "wash_ret": wash_ret,
        "and_ret": and_ret,
        "help_delta": help_delta,
        "label": label,
        "miss_and": miss,
        "false_alarm_and": false_alarm,
        "miss_wash": wash_miss,
        "false_alarm_wash": wash_fa,
        "n_stocks": len(stock_by),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-07-20")
    ap.add_argument("--help-eps", type=float, default=0.005)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--opt-imbalance-max", type=float, default=-0.05)
    ap.add_argument("--opt-chop-pctile-min", type=float, default=0.70)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lit = pd.read_csv(LIT_DAILY)
    wash = pd.read_csv(WASH_DAILY)
    lit["date"] = lit["date"].astype(str).str[:10]
    wash["date"] = wash["date"].astype(str).str[:10]
    lit_m = dict(zip(lit["date"], lit["day_ret"].astype(float)))
    wash_m = dict(zip(wash["date"], wash["day_ret"].astype(float)))
    and_m = {}
    if AND_DAILY.is_file():
        ad = pd.read_csv(AND_DAILY)
        ad["date"] = ad["date"].astype(str).str[:10]
        and_m = dict(zip(ad["date"], ad["day_ret"].astype(float)))

    dates = sorted(
        d
        for d in set(lit_m) | set(wash_m)
        if args.start <= d <= args.end
    )
    cfg = path_fast_pack_from_trade(
        {
            "path_fast_pack": {
                "enabled": True,
                "when": "wash_and_opt_chop",
                "opt_imbalance_max": args.opt_imbalance_max,
                "opt_chop_pctile_min": args.opt_chop_pctile_min,
                "opt_lookback_days": 40,
            }
        }
    )
    # warm surface cache
    _ = load_surface_range("QQQ", args.start, args.end)

    rows = []
    for i, date in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {date}", flush=True)
        rows.append(
            diagnose_day(
                date,
                cfg,
                lit_ret=lit_m.get(date),
                wash_ret=wash_m.get(date),
                and_ret=and_m.get(date),
                help_eps=float(args.help_eps),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "day_diagnosis.csv", index=False)

    def _summ(arm_col: str) -> dict:
        sub = df.dropna(subset=["help_delta"])
        helps = sub[sub["label"] == "fast_helps"]
        hurts = sub[sub["label"] == "fast_hurts"]
        armed = sub[sub[arm_col] == True]  # noqa: E712
        return {
            "n_days": int(len(sub)),
            "n_helps": int(len(helps)),
            "n_hurts": int(len(hurts)),
            "n_armed": int(len(armed)),
            "hit_helps": int(((helps[arm_col] == True)).sum()) if len(helps) else 0,  # noqa: E712
            "miss_helps": int(((helps[arm_col] == False)).sum()) if len(helps) else 0,  # noqa: E712
            "false_alarm_hurts": int(((hurts[arm_col] == True)).sum()) if len(hurts) else 0,  # noqa: E712
            "precision_on_labeled": (
                float(
                    ((armed["label"] == "fast_helps").sum())
                    / max(1, (armed["label"].isin(["fast_helps", "fast_hurts"]).sum()))
                )
                if len(armed)
                else None
            ),
            "sum_help_delta_if_armed": float(
                sub.loc[sub[arm_col] == True, "help_delta"].sum()  # noqa: E712
            ),
            "sum_help_delta_missed": float(
                helps.loc[helps[arm_col] == False, "help_delta"].sum()  # noqa: E712
            )
            if len(helps)
            else 0.0,
        }

    summary = {
        "params": {
            "opt_imbalance_max": args.opt_imbalance_max,
            "opt_chop_pctile_min": args.opt_chop_pctile_min,
            "help_eps": args.help_eps,
            "start": args.start,
            "end": args.end,
        },
        "wash": _summ("wash"),
        "opt": _summ("opt"),
        "and_arm": _summ("and_arm"),
        "or_arm": _summ("or_arm"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # highlight tables
    miss = df[df["miss_and"] == True][  # noqa: E712
        ["date", "wash", "opt", "lit_ret", "wash_ret", "help_delta", "imb", "chop_score"]
    ]
    fa = df[df["false_alarm_and"] == True][  # noqa: E712
        ["date", "wash", "opt", "lit_ret", "wash_ret", "help_delta", "imb", "chop_score"]
    ]
    miss.to_csv(out / "miss_and.csv", index=False)
    fa.to_csv(out / "false_alarm_and.csv", index=False)

    print("\n=== summary ===")
    for k in ["wash", "opt", "and_arm", "or_arm"]:
        s = summary[k]
        print(
            f"{k:8s} armed={s['n_armed']:2d}  "
            f"hit_helps={s['hit_helps']}/{s['n_helps']}  "
            f"miss={s['miss_helps']}  FA={s['false_alarm_hurts']}/{s['n_hurts']}  "
            f"Δarmed={s['sum_help_delta_if_armed']:+.3f}  "
            f"Δmiss={s['sum_help_delta_missed']:+.3f}"
        )
    print("\n漏网 AND (fast_helps but not armed):")
    print(miss.to_string(index=False) if len(miss) else "  (none)")
    print("\n误杀 AND (fast_hurts but armed):")
    print(fa.to_string(index=False) if len(fa) else "  (none)")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
