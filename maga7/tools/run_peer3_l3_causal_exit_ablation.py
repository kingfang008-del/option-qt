#!/usr/bin/env python3
"""Peer3 baseline + L3 causal soft exits (defuse T30 time-bomb).

Keeps peer3 hold_extend T30→T45 + trade_toxic. Adds L3 arms:
  - stock_rev (underlying thesis fail)
  - hold_watchdog (QQQ adverse shock)
  - trail stacked on extend (path giveback)

Windows: May–Jul + Jan–Mar. Optional Jul20 fused for shortlist.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PEER3 = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
OUT = Path("/mnt/s990/data/maga7/results/peer3_l3_causal_exit_ablation_v1")
SESSION_JUL20 = Path(
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e"
)

WINDOWS = {
    "may_jul": ("2026-05-01", "2026-07-20"),
    "jan_mar": ("2026-01-02", "2026-03-31"),
}

# L3 overlays on peer3 trade block (baseline fields kept)
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "srev_10_10": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 10,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        }
    },
    "srev_5_05": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 5,
            "stock_max": 0.0,
            "opt_mtm_max": 0.05,
        }
    },
    "hwd_10": {
        "hold_watchdog": {
            "enabled": True,
            "qqq_adverse_from_entry": 0.010,
            "min_hold_seconds": 60,
            "require_option_mtm_max": None,
        }
    },
    "hwd_10_mtm0": {
        "hold_watchdog": {
            "enabled": True,
            "qqq_adverse_from_entry": 0.010,
            "min_hold_seconds": 60,
            "require_option_mtm_max": 0.0,
        }
    },
    "trail15": {
        # stack path trail on hold_extend (clock remains max hold)
        "early_exit_mode": "mtm_trail",
        "trail_activate": 0.15,
        "trail_dd": 0.08,
    },
    "srev_10_10__hwd_10": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 10,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        },
        "hold_watchdog": {
            "enabled": True,
            "qqq_adverse_from_entry": 0.010,
            "min_hold_seconds": 60,
            "require_option_mtm_max": None,
        },
    },
    "srev_10_10__trail15": {
        "stock_rev_exit": {
            "enabled": True,
            "when": "always",
            "min_hold_minutes": 10,
            "stock_max": 0.0,
            "opt_mtm_max": 0.10,
        },
        "early_exit_mode": "mtm_trail",
        "trail_activate": 0.15,
        "trail_dd": 0.08,
    },
}

CLOCK_REASONS = {"T+30", "T+45", "TIME", "HOLD", "HOLD_EXTEND", "EXTEND"}


def _srev(
    *,
    when: str = "always",
    min_hold: float = 10.0,
    stock_max: float,
    opt_mtm_max: float = 0.0,
    wash_breadth: int | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "when": when,
        "min_hold_minutes": float(min_hold),
        "stock_max": float(stock_max),
        "opt_mtm_max": float(opt_mtm_max),
    }
    if wash_breadth is not None:
        cfg["washout_breadth_min"] = int(wash_breadth)
    return {"stock_rev_exit": cfg}


def build_fine_grid() -> dict[str, dict[str, Any]]:
    """Around srev_uw_m3: deeper stock_max, longer min_hold, wash∧mX."""
    grid: dict[str, dict[str, Any]] = {}
    # always / underwater opt (opt_mtm_max=0) — stock_max depth × hold
    for sm_bp, sm in [("m3", -0.003), ("m4", -0.004), ("m5", -0.005), ("m6", -0.006), ("m8", -0.008), ("m10", -0.010)]:
        for hold in (10, 15, 20):
            name = f"uw_{sm_bp}_h{hold}"
            grid[name] = _srev(when="always", min_hold=hold, stock_max=sm, opt_mtm_max=0.0)
    # always / allow small green opt (opt_mtm 0.10) at deeper stock cuts
    for sm_bp, sm in [("m5", -0.005), ("m8", -0.008)]:
        for hold in (10, 15):
            name = f"stk_{sm_bp}_h{hold}"
            grid[name] = _srev(when="always", min_hold=hold, stock_max=sm, opt_mtm_max=0.10)
    # wash-gated ∧ deeper stock
    for sm_bp, sm in [("m3", -0.003), ("m5", -0.005), ("m8", -0.008)]:
        for opt_tag, opt in [("uw", 0.0), ("o10", 0.10)]:
            name = f"wash_{sm_bp}_{opt_tag}_h10"
            grid[name] = _srev(
                when="mixed_wash_up",
                min_hold=10,
                stock_max=sm,
                opt_mtm_max=opt,
                wash_breadth=3,
            )
    return grid


def _reason_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty or "reason" not in trades.columns:
        return {"reasons": {}, "n_clock": 0, "clock_share": None, "n_l3": 0}
    vc = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    n = int(sum(vc.values()))
    n_clock = sum(v for k, v in vc.items() if k in CLOCK_REASONS or k.startswith("T+"))
    l3_keys = ("STOCK_REV", "HOLD_SHOCK", "TRAIL", "PATH_GIVEBACK", "PATH_IV_SHOCK", "PATH_DELTA_FADE")
    n_l3 = sum(vc.get(k, 0) for k in l3_keys)
    return {
        "reasons": vc,
        "n_clock": n_clock,
        "clock_share": float(n_clock / n) if n else None,
        "n_l3": n_l3,
        "n_tp": int(vc.get("TP", 0)),
        "n_sl": int(vc.get("SL", 0)),
        "n_stock_rev": int(vc.get("STOCK_REV", 0)),
        "n_hold_shock": int(vc.get("HOLD_SHOCK", 0)),
        "n_trail": int(vc.get("TRAIL", 0)),
        "n_trade_tox": int(vc.get("TRADE_TOX", 0) + vc.get("TOXIC", 0)),
    }


def run_one(window: str, variant: str, overlay: dict) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(PEER3))
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v
    print(f"=== {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary, trades, daily = result["summary"], result["trades"], result.get("daily")
    tag = OUT / window / f"replay__{variant}"
    tag.mkdir(parents=True, exist_ok=True)
    (tag / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    trades.to_csv(tag / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(tag / "daily.csv", index=False)
    rs = _reason_stats(trades)
    row = {
        "window": window,
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary.get("trade_win") or 0),
        "end_equity": float(summary.get("end_equity") or 0),
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
    }
    # retention vs baseline filled later
    print(
        f"  ret={row['total_ret']:+.3f} (~{1+row['total_ret']:.1f}x) dd={row['maxdd']:.3f} "
        f"clock_share={row['clock_share']} l3={row['n_l3']} "
        f"REV/SHOCK/TRAIL={row['n_stock_rev']}/{row['n_hold_shock']}/{row['n_trail']}",
        flush=True,
    )
    return row


def run_jul20_fused(variants: list[str], catalog: dict[str, dict[str, Any]] | None = None) -> list[dict]:
    from maga7.tools.run_live_fused_replay import run_fused_replay

    catalog = catalog or VARIANTS
    rows = []
    raw = json.loads(Path(load_profile(PEER3)["_profile_path"]).read_text())
    for variant in variants:
        overlay = catalog[variant]
        p = deepcopy(raw)
        trade = p.setdefault("trade", {})
        for k, v in overlay.items():
            if isinstance(v, dict) and isinstance(trade.get(k), dict):
                trade[k] = {**trade[k], **v}
            else:
                trade[k] = v
        tmp = Path(f"/tmp/peer3_l3_{variant}.json")
        tmp.write_text(json.dumps(p, indent=2), encoding="utf-8")
        tag = f"l3_{variant}_m5_noprev"
        print(f"=== jul20 fused {variant} ===", flush=True)
        s = run_fused_replay(
            SESSION_JUL20,
            scheme="m5",
            disable_prevention=True,
            tag=tag,
            profile_path_override=tmp,
        )
        tpath = SESSION_JUL20 / f"fused_replay_{tag}" / "trades.csv"
        trades = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
        rs = _reason_stats(trades)
        row = {
            "window": "jul20_fused",
            "variant": variant,
            "total_ret": float(s.get("total_ret") or 0),
            "n_trades": int(s.get("n_trades") or 0),
            "sum_trade_ret": float(trades["ret"].sum()) if len(trades) and "ret" in trades.columns else None,
            **{k: v for k, v in rs.items() if k != "reasons"},
            "reasons": rs["reasons"],
            "trades": trades[["symbol", "reason", "ret"]].to_dict(orient="records")
            if len(trades)
            else [],
        }
        rows.append(row)
        print(
            f"  ret={row['total_ret']:+.4f} n={row['n_trades']} "
            f"REV/SHOCK/TRAIL/SL={row['n_stock_rev']}/{row['n_hold_shock']}/{row['n_trail']}/{rs['reasons'].get('SL', 0)}",
            flush=True,
        )
    return rows


def _load_baseline_rets(wins: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for w in wins:
        p = OUT / w / "replay__baseline" / "summary.json"
        if p.exists():
            out[w] = float(json.loads(p.read_text(encoding="utf-8"))["total_ret"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="may_jul,jan_mar")
    ap.add_argument(
        "--mode",
        choices=("v1", "fine"),
        default="v1",
        help="v1=coarse VARIANTS; fine=stock_max/min_hold/wash grid around uw_m3",
    )
    ap.add_argument(
        "--variants",
        default="",
        help="comma list or 'all' (default: all keys in selected catalog)",
    )
    ap.add_argument("--skip-fused", action="store_true")
    ap.add_argument(
        "--fused-variants",
        default="",
        help="comma list; empty → mode default / auto shortlist from fine scoreboard",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    catalog = build_fine_grid() if args.mode == "fine" else VARIANTS
    tag = "fine" if args.mode == "fine" else "v1"
    wins = [w.strip() for w in args.windows.split(",") if w.strip()]
    if not args.variants.strip() or args.variants.strip() == "all":
        variants = list(catalog.keys())
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Fine mode: reuse cached baseline; skip replaying it.
    base_rets = _load_baseline_rets(wins)
    if args.mode == "fine":
        missing = [w for w in wins if w not in base_rets]
        if missing:
            for w in missing:
                row = run_one(w, "baseline", {})
                base_rets[w] = row["total_ret"]

    scoreboard: list[dict] = []
    for window in wins:
        base_ret = base_rets.get(window)
        for variant in variants:
            if args.mode == "fine" and variant == "baseline":
                continue
            row = run_one(window, variant, catalog[variant])
            if variant == "baseline":
                base_ret = row["total_ret"]
                base_rets[window] = base_ret
            if base_ret is not None and (1 + base_ret) != 0:
                row["ret_vs_baseline"] = float(row["total_ret"] - base_ret)
                row["ret_retention"] = float((1 + row["total_ret"]) / (1 + base_ret))
            scoreboard.append(row)

    by_w = dict(base_rets)
    for r in scoreboard:
        if r["variant"] == "baseline":
            by_w[r["window"]] = r["total_ret"]
    for r in scoreboard:
        b = by_w.get(r["window"])
        if b is not None and (1 + b) != 0:
            r["ret_vs_baseline"] = float(r["total_ret"] - b)
            r["ret_retention"] = float((1 + r["total_ret"]) / (1 + b))

    slim = [{k: v for k, v in r.items() if k != "reasons"} for r in scoreboard]
    sb_csv = OUT / (f"scoreboard_{tag}.csv" if args.mode == "fine" else "scoreboard.csv")
    sb_json = OUT / (f"scoreboard_{tag}.json" if args.mode == "fine" else "scoreboard.json")
    pd.DataFrame(slim).to_csv(sb_csv, index=False)
    sb_json.write_text(json.dumps(scoreboard, indent=2, default=str), encoding="utf-8")

    fused_rows = []
    if not args.skip_fused:
        if args.fused_variants.strip():
            fv = [v.strip() for v in args.fused_variants.split(",") if v.strip()]
        elif args.mode == "fine":
            # shortlist: May–Jul retain>=0.80, plus best retain, plus wash near 0.75+
            mj = [r for r in scoreboard if r["window"] == "may_jul"]
            mj.sort(key=lambda r: -(r.get("ret_retention") or 0))
            fv = []
            for r in mj:
                ret_r = r.get("ret_retention") or 0
                if ret_r >= 0.80 or r["variant"] == mj[0]["variant"]:
                    fv.append(r["variant"])
            # always include a few structural anchors if present
            for anchor in ("uw_m5_h10", "uw_m8_h15", "uw_m10_h20", "wash_m5_uw_h10", "wash_m8_o10_h10"):
                if anchor in catalog and anchor not in fv:
                    fv.append(anchor)
            fv = fv[:8]
            print("auto fused shortlist:", fv, flush=True)
        else:
            fv = ["baseline", "srev_10_10", "hwd_10", "srev_10_10__trail15", "srev_10_10__hwd_10"]
        fused_rows = run_jul20_fused(fv, catalog=catalog)
        fused_path = OUT / (f"jul20_fused_{tag}.json" if args.mode == "fine" else "jul20_fused.json")
        fused_path.write_text(json.dumps(fused_rows, indent=2, default=str), encoding="utf-8")

    # short verdict
    verdict = {
        "mode": args.mode,
        "goal": "L3 causal soft exit on peer3; clock share down; May-Jul retain ≥~0.85; Jul20 less toxic",
    }
    for window in wins:
        rows = [r for r in scoreboard if r["window"] == window]
        base_ret = by_w.get(window)
        if base_ret is None:
            continue
        # synthetic baseline row for dd compare if not in scoreboard
        base_dd = None
        bp = OUT / window / "replay__baseline" / "summary.json"
        if bp.exists():
            base_dd = float(json.loads(bp.read_text(encoding="utf-8"))["maxdd"])
        cands = []
        for r in rows:
            if r["variant"] == "baseline":
                continue
            cands.append(
                {
                    "variant": r["variant"],
                    "total_ret": r["total_ret"],
                    "retention": r.get("ret_retention"),
                    "maxdd": r["maxdd"],
                    "clock_share": r.get("clock_share"),
                    "n_l3": r.get("n_l3"),
                    "n_stock_rev": r.get("n_stock_rev"),
                    "dd_delta": float(r["maxdd"] - base_dd) if base_dd is not None else None,
                }
            )
        ok = [c for c in cands if (c["retention"] or 0) >= 0.85]
        near = [c for c in cands if 0.80 <= (c["retention"] or 0) < 0.85]
        ok.sort(key=lambda c: (-(c["retention"] or 0), c["clock_share"] or 1, c["maxdd"]))
        near.sort(key=lambda c: (-(c["retention"] or 0), c["clock_share"] or 1))
        cands_sorted = sorted(cands, key=lambda c: -(c["retention"] or 0))
        verdict[window] = {
            "baseline_ret": base_ret,
            "candidates_retain_ge_85": ok[:8],
            "candidates_retain_80_85": near[:8],
            "top_by_retention": cands_sorted[:10],
        }
    if fused_rows:
        verdict["jul20_fused"] = [
            {
                "variant": r["variant"],
                "total_ret": r["total_ret"],
                "n_sl": r["reasons"].get("SL", 0),
                "n_stock_rev": r["n_stock_rev"],
                "n_hold_shock": r["n_hold_shock"],
                "n_trail": r["n_trail"],
                "trades": r["trades"],
            }
            for r in fused_rows
        ]
        # promotion: May-Jul>=0.85, clock down vs base, Jul20 better than baseline -8.3%
        mj_ok = {c["variant"] for c in verdict.get("may_jul", {}).get("candidates_retain_ge_85", [])}
        jul_base = -0.08285287776942352
        promote = []
        for fr in fused_rows:
            if fr["variant"] not in mj_ok:
                continue
            if float(fr["total_ret"]) > jul_base + 1e-9:
                promote.append(
                    {
                        "variant": fr["variant"],
                        "may_jul_retention": next(
                            c["retention"]
                            for c in verdict["may_jul"]["candidates_retain_ge_85"]
                            if c["variant"] == fr["variant"]
                        ),
                        "jul20_ret": fr["total_ret"],
                    }
                )
        verdict["promote_candidates"] = promote
    vpath = OUT / (f"verdict_{tag}.json" if args.mode == "fine" else "verdict.json")
    vpath.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(json.dumps(verdict, indent=2, default=str)[:6000])
    print("wrote", OUT, "tag=", tag)


if __name__ == "__main__":
    main()
