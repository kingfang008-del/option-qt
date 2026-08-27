#!/usr/bin/env python3
"""Ablate lit vs wash-fast vs wash+refine (path_fast_pack.wash_refine).

1) Day-blend from existing may_jul/jan_mar lit & wash_fast dailies (fast).
2) Full offline replay for wash_refine variant (authoritative).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.path_fast_pack import path_fast_pack_day_should_arm, path_fast_pack_from_trade
from maga7.common.replay import run_offline_replay

NY = "America/New_York"
STOCK_ROOT = Path("/mnt/s990/data/raw_1s/stocks")
SYMBOLS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WASH_ROOT = Path("/mnt/s990/data/maga7/results/path_hold_wash_fast_v1")
OUT = Path("/mnt/s990/data/maga7/results/wash_refine_ablation_v1")

LIT_TRADE = {
    "exit_mode": "mtm_trail",
    "hold_minutes": 45,
    "hold_extend_minutes": None,
    "trail_activate": 0.20,
    "trail_dd": 0.12,
    "stock_rev_exit": {
        "enabled": True,
        "when": "always",
        "min_hold_minutes": 10,
        "stock_max": 0.0,
        "opt_mtm_max": 0.10,
    },
}

FAST_BASE = {
    "enabled": True,
    "when": "mixed_wash_up",
    "hold_minutes": 20,
    "trail_activate": 0.15,
    "trail_dd": 0.08,
    "stock_rev_min_hold_minutes": 5,
    "stock_rev_stock_max": 0.0,
    "stock_rev_opt_mtm_max": 0.05,
    "washout_breadth_min": 3,
}

REFINE = {
    **FAST_BASE,
    "wash_refine": True,
    "wash_refine_chop_max": 1.85,
    "wash_refine_med_stock_ret_max": 0.003,
    "wash_refine_pcr_max": 2.0,
    "wash_refine_iv_mom_max": 0.03,
    "wash_refine_n_down_max": 4,
    "opt_symbol": "QQQ",
    "opt_gate": "imb_only",
}


def _prep(df: pd.DataFrame | None, date: str) -> pd.DataFrame | None:
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


def _arm_map(dates: list[str], refine: bool) -> dict[str, bool]:
    raw = dict(REFINE if refine else FAST_BASE)
    cfg = path_fast_pack_from_trade({"path_fast_pack": raw})
    out = {}
    for i, date in enumerate(dates):
        stock_by = {}
        for sym in SYMBOLS:
            stock_by[sym] = _prep(load_stock_1s_day(STOCK_ROOT, sym, date), date)
        stock_by = {k: v for k, v in stock_by.items() if v is not None and not v.empty}
        qqq = _prep(load_stock_1s_day(STOCK_ROOT, "QQQ", date), date)
        out[date] = bool(
            path_fast_pack_day_should_arm(
                cfg,
                date=date,
                stock_by=stock_by,
                qqq_df=qqq,
                symbols=SYMBOLS,
            )
        )
        if (i + 1) % 10 == 0:
            print(f"  arm {i+1}/{len(dates)}", flush=True)
    return out


def _equity_stats(day_ret: pd.Series) -> dict:
    r = pd.to_numeric(day_ret, errors="coerce").fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return {
        "total_ret": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
        "sum_day_ret": float(r.sum()),
        "maxdd": float(dd.min()) if len(dd) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
        "n_days": int(len(r)),
    }


def day_blend(window: str) -> pd.DataFrame:
    lit = pd.read_csv(WASH_ROOT / f"{window}__lit_always" / "daily.csv")
    wash = pd.read_csv(WASH_ROOT / f"{window}__lit_wash_fast" / "daily.csv")
    lit["date"] = lit["date"].astype(str).str[:10]
    wash["date"] = wash["date"].astype(str).str[:10]
    m = lit[["date", "day_ret"]].rename(columns={"day_ret": "lit_ret"}).merge(
        wash[["date", "day_ret"]].rename(columns={"day_ret": "wash_ret"}),
        on="date",
        how="inner",
    )
    dates = m["date"].tolist()
    print(f"[{window}] computing wash arms n={len(dates)}", flush=True)
    arm_wash = _arm_map(dates, refine=False)
    print(f"[{window}] computing refine arms", flush=True)
    arm_ref = _arm_map(dates, refine=True)
    m["arm_wash"] = m["date"].map(arm_wash)
    m["arm_refine"] = m["date"].map(arm_ref)
    m["blend_wash"] = np.where(m["arm_wash"], m["wash_ret"], m["lit_ret"])
    m["blend_refine"] = np.where(m["arm_refine"], m["wash_ret"], m["lit_ret"])
    return m


def run_replay(start: str, end: str, variant: str, pack: dict) -> dict:
    prof = deepcopy(load_profile(PEER3))
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    for k, v in LIT_TRADE.items():
        trade[k] = v
    trade["path_fast_pack"] = dict(pack)
    print(f"=== replay {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary, trades, daily = result["summary"], result["trades"], result.get("daily")
    out_dir = OUT / f"replay__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    trades.to_csv(out_dir / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(out_dir / "daily.csv", index=False)
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    return {
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "n_fast_pack_days": int(summary.get("n_fast_pack_days") or 0),
        "n_tp": int(reasons.get("TP", 0)),
        "n_trail": int(reasons.get("TRAIL", 0)),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument(
        "--windows",
        default="may_jul,jan_mar",
        help="day-blend windows with existing dailies",
    )
    ap.add_argument("--replay-start", default="2026-05-01")
    ap.add_argument("--replay-end", default="2026-07-20")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blend_rows = []
    for wname in [x.strip() for x in args.windows.split(",") if x.strip()]:
        m = day_blend(wname)
        m.to_csv(out / f"day_blend_{wname}.csv", index=False)
        for label, col in [
            ("lit", "lit_ret"),
            ("wash", "blend_wash"),
            ("wash_refine", "blend_refine"),
        ]:
            st = _equity_stats(m[col])
            st.update(
                {
                    "window": wname,
                    "variant": label,
                    "n_armed": int(
                        m["arm_wash"].sum()
                        if label == "wash"
                        else (m["arm_refine"].sum() if label == "wash_refine" else 0)
                    ),
                }
            )
            blend_rows.append(st)
            print(
                f"[blend {wname}] {label:12s} total={st['total_ret']:+.3f} "
                f"sum={st['sum_day_ret']:+.3f} dd={st['maxdd']:.3f} "
                f"worst={st['worst_day']:+.3f} armed={st['n_armed']}",
                flush=True,
            )
        # show refine vs wash arm disagreement
        diff = m[m["arm_wash"] != m["arm_refine"]][
            ["date", "arm_wash", "arm_refine", "lit_ret", "wash_ret"]
        ]
        diff.to_csv(out / f"arm_diff_{wname}.csv", index=False)
        print(f"[blend {wname}] arm diffs n={len(diff)}", flush=True)

    pd.DataFrame(blend_rows).to_csv(out / "blend_scoreboard.csv", index=False)

    replay_rows = []
    if not args.skip_replay:
        packs = [
            ("lit_always", {"enabled": False}),
            ("wash", FAST_BASE),
            ("wash_refine", REFINE),
        ]
        for variant, pack in packs:
            row = run_replay(args.replay_start, args.replay_end, variant, pack)
            replay_rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} dd={row['maxdd']:.3f} "
                f"fast_days={row['n_fast_pack_days']} "
                f"TP/TRAIL/REV={row['n_tp']}/{row['n_trail']}/{row['n_stock_rev']}",
                flush=True,
            )
        pd.DataFrame(replay_rows).to_csv(out / "replay_scoreboard.csv", index=False)
        (out / "replay_scoreboard.json").write_text(
            json.dumps(replay_rows, indent=2), encoding="utf-8"
        )

    (out / "summary.json").write_text(
        json.dumps(
            {"blend": blend_rows, "replay": replay_rows, "refine": REFINE},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
