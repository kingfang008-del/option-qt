#!/usr/bin/env python3
"""LEGACY: AM/PM overlay on peer3 Rule-A CORE (T+30). Do not use for session book.

Session H120 path (no peer3) → ``run_session_h120_portfolio_accept.py``.

Kept only for historical overlays that explicitly still want peer3 as base.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

CORE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _day_pnl_from_trades(tr: pd.DataFrame, *, scale: float) -> pd.Series:
    """Day sleeve PnL as sum(size*ret)*scale.

    Prefer ``pnl_frac`` (fill tools), then ``size`` / ``size_frac``.
    Opportunity sleeves recycle capital; day sums are additive book units,
    not a claim that equity compounds at sum(pnl) each day.
    """
    if tr is None or tr.empty:
        return pd.Series(dtype=float)
    t = tr.copy()
    t["date"] = t["date"].astype(str)
    if "pnl_frac" in t.columns:
        pnl = pd.to_numeric(t["pnl_frac"], errors="coerce").fillna(0.0) * float(scale)
    else:
        if "size" in t.columns:
            sf = pd.to_numeric(t["size"], errors="coerce").fillna(0.1)
        elif "size_frac" in t.columns:
            sf = pd.to_numeric(t["size_frac"], errors="coerce").fillna(0.1)
        else:
            sf = pd.Series(0.1, index=t.index)
        pnl = t["ret"].astype(float) * sf.astype(float) * float(scale)
    return pnl.groupby(t["date"]).sum()


def _combine(
    core_daily: pd.DataFrame,
    am_tr: pd.DataFrame | None,
    pm_tr: pd.DataFrame | None,
    *,
    am_budget: float,
    pm_budget: float,
    am_native_frac: float = 0.10,
    pm_native_frac: float = 0.10,
) -> pd.DataFrame:
    d = core_daily.copy()
    d["date"] = d["date"].astype(str)
    d["core_ret"] = pd.to_numeric(d["day_ret"], errors="coerce").fillna(0.0)
    # scale sleeve native position_frac up/down to target budget share of CORE unit.
    # CORE uses ~0.20; budgets are absolute risk units on the same equity curve.
    am_scale = (am_budget / am_native_frac) if am_native_frac > 0 else 0.0
    pm_scale = (pm_budget / pm_native_frac) if pm_native_frac > 0 else 0.0
    am = _day_pnl_from_trades(am_tr, scale=am_scale) if am_tr is not None else pd.Series(dtype=float)
    pm = _day_pnl_from_trades(pm_tr, scale=pm_scale) if pm_tr is not None else pd.Series(dtype=float)
    d["am_ret"] = d["date"].map(am).fillna(0.0)
    d["pm_ret"] = d["date"].map(pm).fillna(0.0)
    d["day_ret"] = d["core_ret"] + d["am_ret"] + d["pm_ret"]
    eq = (1.0 + d["day_ret"]).cumprod()
    d["eq"] = eq
    peak = eq.cummax()
    d["dd"] = eq / peak - 1.0
    return d


def _stats(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "total_ret": 0.0,
            "maxdd": 0.0,
            "n_days": 0,
            "sum_day_ret": 0.0,
            "sum_core": 0.0,
            "sum_am": 0.0,
            "sum_pm": 0.0,
        }
    total = float(daily["eq"].iloc[-1] - 1.0) if "eq" in daily.columns else float(
        (1 + daily["day_ret"]).prod() - 1
    )
    maxdd = float(daily["dd"].min()) if "dd" in daily.columns else 0.0
    return {
        "total_ret": total,
        "maxdd": maxdd,
        "n_days": int(len(daily)),
        "sum_day_ret": float(pd.to_numeric(daily["day_ret"], errors="coerce").fillna(0.0).sum()),
        "sum_core": float(pd.to_numeric(daily.get("core_ret", 0.0), errors="coerce").fillna(0.0).sum())
        if "core_ret" in daily.columns
        else float(pd.to_numeric(daily["day_ret"], errors="coerce").fillna(0.0).sum()),
        "sum_am": float(pd.to_numeric(daily.get("am_ret", 0.0), errors="coerce").fillna(0.0).sum())
        if "am_ret" in daily.columns
        else 0.0,
        "sum_pm": float(pd.to_numeric(daily.get("pm_ret", 0.0), errors="coerce").fillna(0.0).sum())
        if "pm_ret" in daily.columns
        else 0.0,
    }


def _load_trades(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return pd.read_csv(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/sleeve_portfolio_accept_v1")
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--am-budget", type=float, default=0.15)
    ap.add_argument("--pm-budget", type=float, default=0.15)
    ap.add_argument("--am-trades-strong", default="")
    ap.add_argument("--am-trades-weak", default="")
    ap.add_argument("--pm-trades-strong", default="")
    ap.add_argument("--pm-trades-weak", default="")
    ap.add_argument("--skip-core-replay", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = load_profile(CORE)
    rows = []
    for wname, start, end, am_p, pm_p in [
        ("strong", args.strong_start, args.strong_end, args.am_trades_strong, args.pm_trades_strong),
        ("weak", args.weak_start, args.weak_end, args.am_trades_weak, args.pm_trades_weak),
    ]:
        core_path = out / f"core_{wname}"
        daily_p = core_path / "daily.csv"
        if args.skip_core_replay and daily_p.is_file():
            daily = pd.read_csv(daily_p)
        else:
            print(f"[core] {wname} {start}→{end}", flush=True)
            p = copy.deepcopy(base)
            p["date_range"] = {"start": start, "end": end}
            res = run_offline_replay(p, scheme="single")
            core_path.mkdir(parents=True, exist_ok=True)
            res["daily"].to_csv(daily_p, index=False)
            res["trades"].to_csv(core_path / "trades.csv", index=False)
            (core_path / "summary.json").write_text(
                json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
            )
            daily = res["daily"]
        am_tr = _load_trades(am_p)
        pm_tr = _load_trades(pm_p)
        # filter trades into window
        if am_tr is not None and not am_tr.empty:
            am_tr = am_tr[(am_tr["date"].astype(str) >= start) & (am_tr["date"].astype(str) <= end)]
        if pm_tr is not None and not pm_tr.empty:
            pm_tr = pm_tr[(pm_tr["date"].astype(str) >= start) & (pm_tr["date"].astype(str) <= end)]
        core_only = _combine(daily, None, None, am_budget=0, pm_budget=0)
        combo = _combine(
            daily,
            am_tr,
            pm_tr,
            am_budget=float(args.am_budget),
            pm_budget=float(args.pm_budget),
        )
        combo.to_csv(out / f"combo_{wname}.csv", index=False)
        cs, cb = _stats(core_only), _stats(combo)
        keep_compound = (cb["total_ret"] / cs["total_ret"]) if cs["total_ret"] > 1e-9 else None
        # Additive keep: sum(day_ret). Prefer this for dense opportunity sleeves —
        # compound keep explodes when sleeve day sums are large vs CORE.
        keep_add = (cb["sum_day_ret"] / cs["sum_day_ret"]) if cs["sum_day_ret"] > 1e-9 else None
        row = {
            "window": wname,
            "core_ret": cs["total_ret"],
            "core_maxdd": cs["maxdd"],
            "combo_ret": cb["total_ret"],
            "combo_maxdd": cb["maxdd"],
            "keep": keep_add if keep_add is not None else keep_compound,
            "keep_compound": keep_compound,
            "keep_additive": keep_add,
            "core_sum_day": cs["sum_day_ret"],
            "combo_sum_day": cb["sum_day_ret"],
            "am_sum_day": cb["sum_am"],
            "pm_sum_day": cb["sum_pm"],
            "am_budget": args.am_budget,
            "pm_budget": args.pm_budget,
            "n_am": 0 if am_tr is None else int(len(am_tr)),
            "n_pm": 0 if pm_tr is None else int(len(pm_tr)),
        }
        rows.append(row)
        print(row, flush=True)

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    strong = rdf[rdf.window == "strong"].iloc[0]
    weak = rdf[rdf.window == "weak"].iloc[0]
    flags = []
    sk = strong["keep"]
    flags.append("strong_retain_ok" if sk is not None and float(sk) >= 0.85 else "strong_retain_fail")
    # Prefer additive ret compare when available (honest for opportunity sleeves).
    if float(weak["combo_sum_day"]) > float(weak["core_sum_day"]):
        flags.append("weak_ret_improved")
    elif float(weak["combo_ret"]) > float(weak["core_ret"]):
        flags.append("weak_ret_improved")
    if weak["combo_maxdd"] > weak["core_maxdd"]:
        flags.append("weak_dd_improved")
    if (
        "strong_retain_ok" in flags
        and ("weak_ret_improved" in flags or "weak_dd_improved" in flags)
    ):
        decision = "PROMOTE_PORTFOLIO_RESEARCH"
    elif "strong_retain_ok" in flags:
        decision = "OVERLAY_ONLY"
    else:
        decision = "REJECT_FOR_BASELINE"
    summary = {"decision": decision, "flags": flags, "scoreboard": rows}
    (out / "accept_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== portfolio {decision} {flags} ===", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
