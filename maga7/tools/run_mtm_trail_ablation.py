#!/usr/bin/env python3
"""Ablate option MTM trailing take-profit vs causal single+T+30 rails baseline.

Keeps TP/SL rails; replaces hard time-exit pressure with trail after peak MTM.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

BASE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_t30_rails_p20_v1.json"
)


def _run(profile: dict[str, Any], scheme: str = "single") -> dict[str, Any]:
    result = run_offline_replay(profile, scheme=scheme)
    s = result["summary"]
    reasons = (
        result["trades"]["reason"].value_counts().to_dict()
        if not result["trades"].empty and "reason" in result["trades"].columns
        else {}
    )
    return {
        "total_ret": float(s["total_ret"]),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "trade_win": float(s["trade_win"]),
        "trade_exp": float(s["trade_exp"]),
        "end_equity": float(s["end_equity"]),
        "exit_mode": s.get("exit_mode"),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
        "summary": s,
        "trades": result["trades"],
        "daily": result["daily"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(BASE))
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-13")
    p.add_argument("--tag", default="mtm_trail_ablation_may_jul")
    p.add_argument("--also-jan-jul", action="store_true")
    args = p.parse_args()

    periods = [(args.start_date, args.end_date, args.tag)]
    if args.also_jan_jul:
        periods.append(("2026-01-02", "2026-07-13", "mtm_trail_ablation_jan_jul"))

    activates = [0.15, 0.20, 0.30, 0.40]
    dds = [0.10, 0.15, 0.20, 0.25]
    holds = [30, 45, 60]

    for start, end, tag in periods:
        base = load_profile(args.profile)
        base["date_range"]["start"] = start
        base["date_range"]["end"] = end
        base.setdefault("trade", {})["bar_availability_delay_seconds"] = 60
        out = Path(base["_paths"]["results_dir"]) / tag
        out.mkdir(parents=True, exist_ok=True)

        scoreboard: list[dict[str, Any]] = []

        # baseline
        print(f"[{tag}] baseline_t30_rails ...", flush=True)
        got = _run(deepcopy(base), "single")
        row = {
            "name": "baseline_t30_rails",
            "exit_mode": "none",
            "hold_minutes": 30,
            "trail_activate": None,
            "trail_dd": None,
            **{k: got[k] for k in ("total_ret", "maxdd", "n_trades", "trade_win", "trade_exp", "end_equity")},
            "reasons": got["reasons"],
        }
        scoreboard.append(row)
        print(
            f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
            f"win={row['trade_win']:.1%} exp={row['trade_exp']:+.2%}",
            flush=True,
        )
        sub = out / row["name"]
        sub.mkdir(exist_ok=True)
        (sub / "summary.json").write_text(json.dumps(got["summary"], indent=2), encoding="utf-8")
        got["trades"].to_csv(sub / "trades.csv", index=False)
        got["daily"].to_csv(sub / "daily.csv", index=False)

        for hm in holds:
            for act in activates:
                for dd in dds:
                    name = f"trail_a{int(act*100)}_dd{int(dd*100)}_h{hm}"
                    prof = deepcopy(base)
                    prof["trade"]["exit_mode"] = "mtm_trail"
                    prof["trade"]["hold_minutes"] = hm
                    prof["trade"]["trail_activate"] = act
                    prof["trade"]["trail_dd"] = dd
                    # keep TP/SL rails
                    prof["trade"]["tp_mult"] = 1.6
                    prof["trade"]["sl_mult"] = 0.4
                    print(f"[{tag}] {name} ...", flush=True)
                    got = _run(prof, "single")
                    row = {
                        "name": name,
                        "exit_mode": "mtm_trail",
                        "hold_minutes": hm,
                        "trail_activate": act,
                        "trail_dd": dd,
                        **{
                            k: got[k]
                            for k in (
                                "total_ret",
                                "maxdd",
                                "n_trades",
                                "trade_win",
                                "trade_exp",
                                "end_equity",
                            )
                        },
                        "reasons": got["reasons"],
                    }
                    scoreboard.append(row)
                    print(
                        f"  ret={row['total_ret']:+.2%} dd={row['maxdd']:.2%} "
                        f"win={row['trade_win']:.1%} reasons={row['reasons']}",
                        flush=True,
                    )
                    sub = out / name
                    sub.mkdir(exist_ok=True)
                    (sub / "summary.json").write_text(
                        json.dumps(got["summary"], indent=2), encoding="utf-8"
                    )
                    got["trades"].to_csv(sub / "trades.csv", index=False)
                    got["daily"].to_csv(sub / "daily.csv", index=False)

        import pandas as pd

        sb = pd.DataFrame(scoreboard)
        sb.to_csv(out / "scoreboard.csv", index=False)
        baseline_ret = float(scoreboard[0]["total_ret"])
        baseline_dd = float(scoreboard[0]["maxdd"])
        # prefer higher ret with dd not worse than baseline-5pp, else best ret among dd>-0.40
        viable = [
            r
            for r in scoreboard[1:]
            if r["maxdd"] >= min(baseline_dd - 0.05, -0.40)
        ]
        pick = max(viable or scoreboard[1:], key=lambda r: r["total_ret"])
        best = max(scoreboard, key=lambda r: r["total_ret"])
        summary = {
            "period": f"{start}..{end}",
            "baseline": scoreboard[0],
            "best_by_ret": best,
            "pick_vs_baseline_dd": pick,
            "uplift_pp": {
                "best_ret": best["total_ret"] - baseline_ret,
                "pick": pick["total_ret"] - baseline_ret,
            },
            "note": (
                "mtm_trail: arm after peak MTM ret>=trail_activate; exit when "
                "ret <= peak - trail_dd. TP1.6/SL0.4 always on. scheme=single."
            ),
        }
        (out / "scoreboard.json").write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in summary if k != "note"}, indent=2))
        print(f"→ {out}")


if __name__ == "__main__":
    main()
