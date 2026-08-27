#!/usr/bin/env python3
"""Session-only book: AM + MID (+ optional CORE_1030 H120). No peer3 / Rule-A.

Combines opportunity-fill trades by day using sum(pnl_frac)*budget_scale.
Gates use additive day sums and day-compound MaxDD — not peer3 keep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_trades(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _day_pnl(tr: pd.DataFrame | None, *, scale: float) -> pd.Series:
    if tr is None or tr.empty:
        return pd.Series(dtype=float)
    t = tr.copy()
    t["date"] = t["date"].astype(str)
    if "pnl_frac" in t.columns:
        pnl = pd.to_numeric(t["pnl_frac"], errors="coerce").fillna(0.0) * float(scale)
    else:
        size = (
            pd.to_numeric(t["size"], errors="coerce").fillna(0.1)
            if "size" in t.columns
            else (
                pd.to_numeric(t["size_frac"], errors="coerce").fillna(0.1)
                if "size_frac" in t.columns
                else pd.Series(0.1, index=t.index)
            )
        )
        pnl = t["ret"].astype(float) * size.astype(float) * float(scale)
    return pnl.groupby(t["date"]).sum()


def _filter(tr: pd.DataFrame | None, start: str, end: str) -> pd.DataFrame | None:
    if tr is None or tr.empty:
        return tr
    d = tr["date"].astype(str)
    return tr[(d >= start) & (d <= end)].copy()


def _book_stats(
    am: pd.Series,
    mid: pd.Series,
    core: pd.Series,
) -> dict[str, float]:
    idx = sorted(set(am.index) | set(mid.index) | set(core.index))
    if not idx:
        return {
            "n_days": 0,
            "am_sum": 0.0,
            "mid_sum": 0.0,
            "core1030_sum": 0.0,
            "sum_day": 0.0,
            "day_compound_ret": 0.0,
            "day_compound_maxdd": 0.0,
            "day_win": None,
            "worst_day": None,
            "red_days": 0,
        }
    d = pd.DataFrame(index=idx)
    d["am"] = am.reindex(idx).fillna(0.0)
    d["mid"] = mid.reindex(idx).fillna(0.0)
    d["core1030"] = core.reindex(idx).fillna(0.0)
    d["day"] = d["am"] + d["mid"] + d["core1030"]
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for x in d["day"]:
        eq *= 1.0 + float(x)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    return {
        "n_days": int(len(d)),
        "am_sum": float(d["am"].sum()),
        "mid_sum": float(d["mid"].sum()),
        "core1030_sum": float(d["core1030"].sum()),
        "sum_day": float(d["day"].sum()),
        "day_compound_ret": float(eq - 1.0),
        "day_compound_maxdd": float(mdd),
        "day_win": float((d["day"] > 0).mean()),
        "worst_day": float(d["day"].min()),
        "red_days": int((d["day"] <= 0).sum()),
        "_daily": d,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/session_h120_book_accept_v1")
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--am-budget", type=float, default=0.15, help="risk unit; native fill≈0.10")
    ap.add_argument("--mid-budget", type=float, default=0.10)
    ap.add_argument("--core1030-budget", type=float, default=0.0, help="0 = omit CORE_1030 sleeve")
    ap.add_argument("--native-frac", type=float, default=0.10)
    ap.add_argument("--am-trades-strong", required=True)
    ap.add_argument("--am-trades-weak", required=True)
    ap.add_argument("--mid-trades-strong", required=True)
    ap.add_argument("--mid-trades-weak", required=True)
    ap.add_argument("--core1030-trades-strong", default="")
    ap.add_argument("--core1030-trades-weak", default="")
    ap.add_argument("--maxdd-floor", type=float, default=-0.20)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nf = float(args.native_frac)
    am_scale = (float(args.am_budget) / nf) if nf > 0 else 0.0
    mid_scale = (float(args.mid_budget) / nf) if nf > 0 else 0.0
    c_scale = (float(args.core1030_budget) / nf) if nf > 0 and args.core1030_budget > 0 else 0.0

    rows = []
    for wname, start, end, am_p, mid_p, c_p in [
        (
            "strong",
            args.strong_start,
            args.strong_end,
            args.am_trades_strong,
            args.mid_trades_strong,
            args.core1030_trades_strong,
        ),
        (
            "weak",
            args.weak_start,
            args.weak_end,
            args.am_trades_weak,
            args.mid_trades_weak,
            args.core1030_trades_weak,
        ),
    ]:
        am_tr = _filter(_load_trades(am_p), start, end)
        mid_tr = _filter(_load_trades(mid_p), start, end)
        c_tr = _filter(_load_trades(c_p), start, end) if c_scale > 0 else None
        st = _book_stats(
            _day_pnl(am_tr, scale=am_scale),
            _day_pnl(mid_tr, scale=mid_scale),
            _day_pnl(c_tr, scale=c_scale),
        )
        daily = st.pop("_daily")
        daily.to_csv(out / f"daily_{wname}.csv")
        row = {
            "window": wname,
            "am_budget": args.am_budget,
            "mid_budget": args.mid_budget,
            "core1030_budget": args.core1030_budget,
            "n_am": 0 if am_tr is None else int(len(am_tr)),
            "n_mid": 0 if mid_tr is None else int(len(mid_tr)),
            "n_core1030": 0 if c_tr is None else int(len(c_tr)),
            **st,
        }
        rows.append(row)
        print(row, flush=True)

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    strong = rdf[rdf.window == "strong"].iloc[0]
    weak = rdf[rdf.window == "weak"].iloc[0]
    flags = []
    if float(strong["sum_day"]) > 0:
        flags.append("strong_add_ok")
    else:
        flags.append("strong_add_fail")
    if float(weak["sum_day"]) > 0:
        flags.append("weak_add_ok")
    else:
        flags.append("weak_add_fail")
    if float(strong["day_compound_maxdd"]) >= float(args.maxdd_floor):
        flags.append("strong_dd_ok")
    else:
        flags.append("strong_dd_fail")
    if float(weak["day_compound_maxdd"]) >= float(args.maxdd_floor):
        flags.append("weak_dd_ok")
    else:
        flags.append("weak_dd_fail")

    if (
        "strong_add_ok" in flags
        and "weak_add_ok" in flags
        and "strong_dd_ok" in flags
        and "weak_dd_ok" in flags
    ):
        decision = "PROMOTE_SESSION_BOOK"
    elif "strong_add_ok" in flags and "weak_add_ok" in flags:
        decision = "OVERLAY_DD_WATCH"
    else:
        decision = "REJECT_SESSION_BOOK"

    summary = {
        "decision": decision,
        "flags": flags,
        "note": "No peer3/Rule-A. Session H120 sleeves only.",
        "scoreboard": rows,
    }
    (out / "accept_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n=== session book {decision} {flags} ===", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
