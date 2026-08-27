#!/usr/bin/env python3
"""Path-adaptive hold: abolish fixed T30 as primary exit.

Thesis
------
Peer3 high total_ret is **not** explained by sitting 30 minutes. On May–Jul
control, trade-ret sum is dominated by TP; T+30 is a net drag (losers grind
the clock). Hold must answer *why still long* from live path features.

Design (Layer 5)
----------------
- Keep outer TP / SL / trade_toxic.
- Primary mid-hold: stock confirm stall (DELTA_STOP), adverse soft exit,
  option MTM trail after arming, optional ROI progress rails.
- Time is only a **safety max** (default 90m) or undeveloped stale cut —
  never the narrative of the edge.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

WINDOWS = [
    ("may_jul", "2026-05-01", "2026-07-17"),
    ("jan_mar", "2026-01-02", "2026-03-31"),
    ("d0720", "2026-07-20", "2026-07-20"),
]

DELTA_5 = {
    "enabled": True,
    "check_seconds": 300,
    "max_seconds": 900,
    "min_stock_move": 0.0015,
    "opt_mtm_max": 0.0,
}
ADV_SOFT = {
    "enabled": True,
    "mode": "soft_exit",
    "check_seconds": 300,
    "adverse_mae": 0.0015,
    "opt_mtm_max": 0.0,
    "require_still_adverse": True,
    "still_adverse_max": -0.0010,
    "max_opt_mfe": 0.15,
}
ROI_PROG = {
    "enabled": True,
    "rails": [
        {"mins": 5.0, "min_roi": -0.05},
        {"mins": 10.0, "min_roi": 0.0},
        {"mins": 20.0, "min_roi": 0.05},
    ],
}

# Shared: drop hold_extend; time = safety max only.
_PATH_BASE = {
    "exit_mode": "mtm_trail",
    "hold_minutes": 90,
    "hold_extend_minutes": None,
    "trail_activate": 0.20,
    "trail_dd": 0.15,
}

VARIANTS: dict[str, dict] = {
    "baseline_t30": {},
    # Path core: trail + delta stall; 90m hard cap only
    "path_trail_delta": {
        **_PATH_BASE,
        "delta_time_stop": dict(DELTA_5),
    },
    # + adverse soft (stock dug then still adverse)
    "path_trail_delta_adv": {
        **_PATH_BASE,
        "delta_time_stop": dict(DELTA_5),
        "adverse_soft": dict(ADV_SOFT),
    },
    # + ROI progress rails (undeveloped option mark)
    "path_full": {
        **_PATH_BASE,
        "delta_time_stop": dict(DELTA_5),
        "adverse_soft": dict(ADV_SOFT),
        "roi_time_stop": dict(ROI_PROG),
    },
    # Tighter trail arm (give winners less leash after +10%)
    "path_trail10_delta": {
        **_PATH_BASE,
        "trail_activate": 0.10,
        "trail_dd": 0.12,
        "delta_time_stop": dict(DELTA_5),
        "adverse_soft": dict(ADV_SOFT),
    },
    # Keep peer3 entries but no clock extend: pure T90 safety + delta only
    "no_extend_delta90": {
        "exit_mode": "none",
        "hold_minutes": 90,
        "hold_extend_minutes": None,
        "delta_time_stop": dict(DELTA_5),
    },
    # Asymmetric: keep T30→T45 + TP path; path only cuts undeveloped/adverse.
    # This does NOT claim T30 is the edge — it keeps the TP mass while replacing
    # clock-bleed losers with stock/option path evidence.
    "asymm_delta_on_t30": {
        "delta_time_stop": dict(DELTA_5),
    },
    "asymm_delta10_adv": {
        "delta_time_stop": {
            **DELTA_5,
            "check_seconds": 600,
            "max_seconds": 1200,
            "min_stock_move": 0.0010,
        },
        "adverse_soft": dict(ADV_SOFT),
    },
    "asymm_adv_only": {
        "adverse_soft": dict(ADV_SOFT),
    },
    # Replace clock for losers only: early mtm floor after path fails
    "asymm_floor_m10": {
        "early_exit_mode": "mtm_floor",
        "mtm_floor_ret": -0.10,
        "exit_min_hold_minutes": 10,
    },
    # Stock15bp extend gate (path condition to earn T45 — not a fixed gift)
    "asymm_ext_stock15": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
    },
    # Path reverse (May–Jul: @15m stock<0 separates T+30 losers from TP)
    "stock_rev15": {
        "stock_rev_exit": {
            "enabled": True,
            "min_hold_minutes": 15,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        }
    },
    "stock_rev15_tight": {
        "stock_rev_exit": {
            "enabled": True,
            "min_hold_minutes": 15,
            "stock_max": 0.0,
            "opt_mtm_max": 0.0,
        }
    },
    # Abolish T30 narrative: safety max 90m + stock_rev + trail; TP still primary
    "path_rev_trail90": {
        "exit_mode": "mtm_trail",
        "hold_minutes": 90,
        "hold_extend_minutes": None,
        "trail_activate": 0.20,
        "trail_dd": 0.15,
        "stock_rev_exit": {
            "enabled": True,
            "min_hold_minutes": 15,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        },
    },
    "stock_rev15_ext15": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
        "stock_rev_exit": {
            "enabled": True,
            "min_hold_minutes": 15,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        },
    },
    # Conditional: only arm STOCK_REV on mixed_wash_up mornings
    "cond_rev20_b3": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "mixed_wash_up",
            "min_hold_minutes": 20,
            "stock_max": -0.0015,
            "opt_mtm_max": 0.10,
            "washout_breadth_min": 3,
        }
    },
    "cond_rev20_b5": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "mixed_wash_up",
            "min_hold_minutes": 20,
            "stock_max": -0.0015,
            "opt_mtm_max": 0.10,
            "washout_breadth_min": 5,
        }
    },
    "cond_rev20_b5_ext15": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
        "stock_rev_exit": {
            "enabled": True,
            "when": "mixed_wash_up",
            "min_hold_minutes": 20,
            "stock_max": -0.0015,
            "opt_mtm_max": 0.10,
            "washout_breadth_min": 5,
        },
    },
    "cond_rev20_b3_ext15": {
        "hold_extend_require_stock": True,
        "hold_extend_stock_min": 0.0015,
        "stock_rev_exit": {
            "enabled": True,
            "when": "mixed_wash_up",
            "min_hold_minutes": 20,
            "stock_max": -0.0015,
            "opt_mtm_max": 0.10,
            "washout_breadth_min": 3,
        },
    },
    "always_rev20": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 20,
            "stock_max": -0.0015,
            "opt_mtm_max": 0.10,
        }
    },
}


def _hold_sec(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    et = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
    xt = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
    return (xt - et).dt.total_seconds()


def _reason_attrib(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {}
    g = trades.groupby("reason")["ret"].agg(["count", "sum", "mean"])
    out = {}
    for reason, row in g.iterrows():
        out[str(reason)] = {
            "n": int(row["count"]),
            "sum_ret": float(row["sum"]),
            "mean_ret": float(row["mean"]),
        }
    return out


def _hold_bucket_attrib(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    hs = _hold_sec(trades)
    bins = [-1, 300, 600, 900, 1800, 2700, 1e9]
    labels = ["<=5m", "5-10m", "10-15m", "15-30m", "30-45m", ">45m"]
    b = pd.cut(hs, bins=bins, labels=labels)
    tmp = trades.assign(_b=b, _hs=hs)
    out = {}
    for lab, sub in tmp.groupby("_b", observed=False):
        if sub.empty:
            continue
        out[str(lab)] = {
            "n": int(len(sub)),
            "sum_ret": float(sub["ret"].sum()),
            "mean_ret": float(sub["ret"].mean()),
            "med_hold_sec": float(sub["_hs"].median()),
        }
    return out


def _metrics(summary: dict, trades: pd.DataFrame) -> dict:
    hs = _hold_sec(trades)
    reasons = {}
    if trades is not None and not trades.empty and "reason" in trades.columns:
        reasons = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    return {
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary["trade_win"]),
        "trade_exp": float(summary.get("trade_exp") or 0),
        "med_hold_sec": float(hs.median()) if len(hs) else None,
        "mean_hold_sec": float(hs.mean()) if len(hs) else None,
        "n_t30": int(reasons.get("T+30", 0)),
        "n_t45": int(reasons.get("T+45", 0)),
        "n_t90": int(reasons.get("T+90", 0)),
        "n_tp": int(reasons.get("TP", 0)),
        "n_trail": int(reasons.get("TRAIL", 0)),
        "n_delta": int(reasons.get("DELTA_STOP", 0)),
        "n_stock_rev": int(reasons.get("STOCK_REV", 0)),
        "n_adv_soft": int(reasons.get("ADVERSE_SOFT", 0)),
        "n_trade_tox": int(
            reasons.get("TRADE_TOX", 0) + reasons.get("TRADE_TOX_RECONNECT", 0)
        ),
        "reason_top": dict(list(sorted(reasons.items(), key=lambda kv: -kv[1])[:10])),
        "attrib_reason": _reason_attrib(trades),
        "attrib_hold_bucket": _hold_bucket_attrib(trades),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--windows", default="may_jul,jan_mar,d0720")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/path_adaptive_hold_ablation_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want_w = {x.strip() for x in args.windows.split(",") if x.strip()}
    want_v = [x.strip() for x in args.variants.split(",") if x.strip()]

    rows = []
    for wname, start, end in WINDOWS:
        if wname not in want_w:
            continue
        for vname in want_v:
            if vname not in VARIANTS:
                raise SystemExit(f"unknown variant {vname}")
            prof = load_profile(args.profile)
            prof = deepcopy(prof)
            prof["date_range"] = {"start": start, "end": end}
            trade = prof.setdefault("trade", {})
            for k, v in VARIANTS[vname].items():
                trade[k] = v
            tag = f"{wname}__{vname}"
            print(f"=== {tag} ===", flush=True)
            result = run_offline_replay(prof, scheme="single")
            summary = result["summary"]
            trades = result["trades"]
            wdir = out / tag
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            trades.to_csv(wdir / "trades.csv", index=False)
            m = _metrics(summary, trades)
            m["window"] = wname
            m["variant"] = vname
            m["start"] = start
            m["end"] = end
            (wdir / "attrib.json").write_text(
                json.dumps(
                    {
                        "attrib_reason": m["attrib_reason"],
                        "attrib_hold_bucket": m["attrib_hold_bucket"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            rows.append(m)
            med_s = m["med_hold_sec"]
            med_txt = f"{med_s:.0f}s" if med_s is not None else "n/a"
            print(
                f"  ret={m['total_ret']:+.1%} dd={m['maxdd']:+.1%} "
                f"win={m['trade_win']:.1%} n={m['n_trades']} "
                f"med_hold={med_txt} "
                f"TP/TRAIL/REV/T30="
                f"{m['n_tp']}/{m['n_trail']}/{m.get('n_stock_rev', 0)}/{m['n_t30']}",
                flush=True,
            )

    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items() if not k.startswith("attrib_")})
    bdf = pd.DataFrame(flat)
    bdf.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )

    cmp_rows = []
    for wname, _, _ in WINDOWS:
        if wname not in want_w:
            continue
        base = bdf[(bdf.window == wname) & (bdf.variant == "baseline_t30")]
        if base.empty:
            continue
        br = float(base.iloc[0]["total_ret"])
        bd = float(base.iloc[0]["maxdd"])
        for r in bdf[bdf.window == wname].itertuples():
            cmp_rows.append(
                {
                    "window": wname,
                    "variant": r.variant,
                    "total_ret": r.total_ret,
                    "vs_base_ret": r.total_ret - br,
                    "maxdd": r.maxdd,
                    "vs_base_dd": r.maxdd - bd,
                    "trade_win": r.trade_win,
                    "med_hold_sec": r.med_hold_sec,
                    "n_tp": r.n_tp,
                    "n_t30": r.n_t30,
                    "n_delta": r.n_delta,
                    "n_trail": r.n_trail,
                }
            )
    cdf = pd.DataFrame(cmp_rows)
    cdf.to_csv(out / "vs_baseline.csv", index=False)

    # Promote: dual research windows (may_jul+jan_mar) keep ≥70% ret,
    # MaxDD not worse >3pp, and material drop in T+30 / med hold moves.
    research_w = [w for w in ("may_jul", "jan_mar") if w in want_w]
    promote = []
    for v in want_v:
        if v == "baseline_t30":
            continue
        sub = cdf[(cdf.variant == v) & (cdf.window.isin(research_w))]
        if len(research_w) and len(sub) < len(research_w):
            continue
        ok = True
        for w in research_w:
            row = sub[sub.window == w]
            brow = cdf[(cdf.variant == "baseline_t30") & (cdf.window == w)]
            if row.empty or brow.empty:
                ok = False
                break
            br = float(brow.iloc[0]["total_ret"])
            bd = float(brow.iloc[0]["maxdd"])
            if float(row.iloc[0]["total_ret"]) < 0.70 * br and br > 0:
                ok = False
                break
            if float(row.iloc[0]["maxdd"]) < bd - 0.03:
                ok = False
                break
        if ok:
            promote.append(v)

    best = None
    if promote and "jan_mar" in want_w:
        weak = cdf[(cdf.window == "jan_mar") & (cdf.variant.isin(promote))]
        if len(weak):
            best = weak.sort_values("vs_base_ret", ascending=False).iloc[0]["variant"]
    elif promote:
        best = promote[0]

    # Baseline attribution spotlight (may_jul if present)
    base_attr = next(
        (
            r
            for r in rows
            if r["window"] == "may_jul" and r["variant"] == "baseline_t30"
        ),
        None,
    )
    summary = {
        "verdict": "PROMOTE" if best else "RESEARCH",
        "best": best,
        "promote_candidates": promote,
        "thesis": (
            "Fixed T30 is not the edge. Attribute baseline by reason/hold-bucket; "
            "path exits must beat clock-bleed without gutting TP mass."
        ),
        "baseline_may_jul_attrib_reason": (base_attr or {}).get("attrib_reason"),
        "baseline_may_jul_attrib_hold_bucket": (base_attr or {}).get(
            "attrib_hold_bucket"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Path-Adaptive Hold Ablation",
        "",
        f"**Verdict: `{summary['verdict']}`** · best=`{best}`",
        "",
        "## Thesis",
        "",
        summary["thesis"],
        "",
        "## Baseline May–Jul attribution (trade-ret sum)",
        "",
        "```",
        json.dumps(summary.get("baseline_may_jul_attrib_reason"), indent=2),
        "```",
        "",
        "### By hold bucket",
        "",
        "```",
        json.dumps(summary.get("baseline_may_jul_attrib_hold_bucket"), indent=2),
        "```",
        "",
        "## Scoreboard",
        "",
        "```",
        bdf.to_string(index=False),
        "```",
        "",
        "## vs baseline",
        "",
        "```",
        cdf.to_string(index=False),
        "```",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
