#!/usr/bin/env python3
"""July volatile window: post-entry failure exits on mf10 peer3 baseline.

Arms (state morning gate NOT re-tuned — focus hold-period):
  CTRL0          L0 baseline
  HWD_80         hold_watchdog QQQ adverse 0.8%
  HWD_100        hold_watchdog 1.0%
  HWD_80_MTM0    hold_watchdog 0.8% only if option MTM<=0
  WAVE_ABORT     post-fill wave confirm abort (profile defaults)
  S1_HWD_100     soft path confirm + hold_watchdog 1.0%
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

BASE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WAVE_PROF = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_wave_abort_v1.json"
)

PATH_SOFT = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "on_timeout": "allow",
    "delay_on_pos": False,
    "tod_start": "10:30",
    "tod_end": "14:00",
}


def _apply(base: dict[str, Any], arm: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    trade = p.setdefault("trade", {})
    trade["hold_watchdog"] = {"enabled": False}
    trade.pop("wave_abort", None)
    trade.pop("stock_path_confirm", None)
    p["state_gate"] = {"enabled": False}
    rr = dict(p.get("regime_router") or {})
    rr["enabled"] = False
    p["regime_router"] = rr

    if arm == "CTRL0":
        return p
    if arm == "HWD_80":
        trade["hold_watchdog"] = {
            "enabled": True,
            "qqq_adverse_from_entry": 0.008,
            "min_hold_seconds": 60,
            "require_option_mtm_max": None,
        }
        return p
    if arm == "HWD_100":
        trade["hold_watchdog"] = {
            "enabled": True,
            "qqq_adverse_from_entry": 0.010,
            "min_hold_seconds": 60,
            "require_option_mtm_max": None,
        }
        return p
    if arm == "HWD_80_MTM0":
        trade["hold_watchdog"] = {
            "enabled": True,
            "qqq_adverse_from_entry": 0.008,
            "min_hold_seconds": 60,
            "require_option_mtm_max": 0.0,
        }
        return p
    if arm == "WAVE_ABORT":
        w = load_profile(WAVE_PROF)
        trade["wave_abort"] = copy.deepcopy((w.get("trade") or {}).get("wave_abort") or {"enabled": True})
        return p
    if arm == "S1_HWD_100":
        trade["stock_path_confirm"] = dict(PATH_SOFT)
        trade["hold_watchdog"] = {
            "enabled": True,
            "qqq_adverse_from_entry": 0.010,
            "min_hold_seconds": 60,
            "require_option_mtm_max": None,
        }
        return p
    raise SystemExit(f"unknown arm {arm}")


def _summarize(tag: str, res: dict[str, Any]) -> dict[str, Any]:
    s = res["summary"]
    tr = res["trades"]
    daily = res["daily"]
    reasons = (
        tr["reason"].value_counts().to_dict()
        if tr is not None and not tr.empty and "reason" in tr.columns
        else {}
    )
    day0721 = None
    if daily is not None and not daily.empty:
        dcol = "date" if "date" in daily.columns else daily.columns[0]
        retc = "day_ret" if "day_ret" in daily.columns else None
        if retc is None:
            for c in daily.columns:
                if c != dcol and pd.api.types.is_numeric_dtype(daily[c]):
                    retc = c
                    break
        hit = daily[daily[dcol].astype(str).str[:10] == "2026-07-21"]
        if retc and not hit.empty:
            day0721 = float(hit.iloc[0][retc])
    t21 = None
    if tr is not None and not tr.empty:
        sub = tr[tr["date"].astype(str).str[:10] == "2026-07-21"]
        if not sub.empty:
            t21 = {
                "n": int(len(sub)),
                "mean_ret": float(sub["ret"].mean()),
                "reasons": sub["reason"].value_counts().to_dict(),
                "rows": sub[["symbol", "dir", "ret", "reason"]].to_dict(orient="records"),
            }
    return {
        "arm": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "n_HOLD_SHOCK": int(reasons.get("HOLD_SHOCK", 0)),
        "n_WAVE_ABORT": int(reasons.get("WAVE_ABORT", 0)),
        "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
        "day_2026_07_21": day0721,
        "trades_2026_07_21": t21,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-07-21")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/july_failure_exit_ablation_v1",
    )
    ap.add_argument(
        "--arms",
        default="CTRL0,HWD_80,HWD_100,HWD_80_MTM0,WAVE_ABORT,S1_HWD_100",
    )
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows: list[dict[str, Any]] = []

    for arm in arms:
        print(f"[run] {arm} {args.start}→{args.end}", flush=True)
        prof = _apply(base, arm)
        prof["date_range"] = {"start": args.start, "end": args.end}
        res = run_offline_replay(prof, scheme="single")
        sub = out / arm
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
        )
        res["daily"].to_csv(sub / "daily.csv", index=False)
        res["trades"].to_csv(sub / "trades.csv", index=False)
        row = _summarize(arm, res)
        rows.append(row)
        print(
            f"  ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']} "
            f"0721={None if row['day_2026_07_21'] is None else f'{row['day_2026_07_21']*100:+.2f}%'} "
            f"shock={row['n_HOLD_SHOCK']} wave={row['n_WAVE_ABORT']}",
            flush=True,
        )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    ctrl = next((r for r in rows if r["arm"] == "CTRL0"), rows[0])
    best = None
    for r in rows:
        if r["arm"] == "CTRL0":
            continue
        keep = r["total_ret"] / ctrl["total_ret"] if ctrl["total_ret"] > 1e-9 else None
        better_0721 = (
            r["day_2026_07_21"] is not None
            and ctrl["day_2026_07_21"] is not None
            and r["day_2026_07_21"] > ctrl["day_2026_07_21"] + 1e-6
        )
        better_dd = r["maxdd"] > ctrl["maxdd"]
        ok_keep = keep is not None and keep >= 0.95
        if ok_keep and (better_0721 or better_dd):
            if best is None or (r["day_2026_07_21"] or -9) > (best["day_2026_07_21"] or -9):
                best = {**r, "keep_vs_ctrl": keep}
    decision = "PROMOTE_FAILURE_EXIT" if best else "NO_JULY_FAILURE_LIFT"
    summary = {
        "decision": decision,
        "window": f"{args.start}..{args.end}",
        "protocol": "july_failure_exit_hold_wave",
        "ctrl": ctrl,
        "best": best,
        "rows": rows,
        "pass_rule": "keep>=0.95 of CTRL and (better 7/21 day or better MaxDD)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# July Failure / Deep-V Exit Ablation",
                "",
                f"**Decision: `{decision}`**",
                "",
                "## Best",
                "",
                "```json",
                json.dumps(best, indent=2, default=str),
                "```",
                "",
                "## Scoreboard",
                "",
                rdf.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "best": best}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
