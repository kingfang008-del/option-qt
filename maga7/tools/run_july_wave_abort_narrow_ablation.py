#!/usr/bin/env python3
"""July: narrow WAVE_ABORT variants (avoid global false revoke on DN winners).

Baseline peer3 L0. Compare keep>=0.95 vs CTRL and 7/21 AMD toxic lift.
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

BASE_WA = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "revoke_seconds": 1800,
    "on_timeout": "allow",
}


def _arms() -> dict[str, dict[str, Any]]:
    return {
        "CTRL0": {},
        "WA_FULL": dict(BASE_WA),
        "WA_UP_ONLY": {**BASE_WA, "only_directions": ["UP"]},
        "WA_UP_REV_MTM0": {
            **BASE_WA,
            "only_directions": ["UP"],
            "revoke_opt_mtm_max": 0.0,
        },
        "WA_UP_DEEP_REV": {
            **BASE_WA,
            "only_directions": ["UP"],
            "thr_neg_revoke": -0.006,
        },
        "WA_UP_NO_REVOKE": {
            **BASE_WA,
            "only_directions": ["UP"],
            "allow_revoke": False,
        },
        "WA_UP_REV600": {
            **BASE_WA,
            "only_directions": ["UP"],
            "revoke_seconds": 600,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-07-21")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/july_wave_abort_narrow_v1",
    )
    args = ap.parse_args(argv)

    base = load_profile(BASE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for name, wa in _arms().items():
        print(f"[run] {name}", flush=True)
        p = copy.deepcopy(base)
        p["date_range"] = {"start": args.start, "end": args.end}
        rr = dict(p.get("regime_router") or {})
        rr["enabled"] = False
        p["regime_router"] = rr
        trade = p.setdefault("trade", {})
        trade["hold_watchdog"] = {"enabled": False}
        trade.pop("stock_path_confirm", None)
        p["state_gate"] = {"enabled": False}
        if wa:
            trade["wave_abort"] = copy.deepcopy(wa)
        else:
            trade.pop("wave_abort", None)

        res = run_offline_replay(p, scheme="single")
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "summary.json").write_text(
            json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
        )
        res["daily"].to_csv(sub / "daily.csv", index=False)
        res["trades"].to_csv(sub / "trades.csv", index=False)

        s = res["summary"]
        tr = res["trades"]
        daily = res["daily"]
        reasons = tr["reason"].value_counts().to_dict() if not tr.empty else {}
        d21 = daily[daily["date"].astype(str).str[:10] == "2026-07-21"]
        day21 = float(d21.iloc[0]["day_ret"]) if not d21.empty else None
        t21 = tr[tr["date"].astype(str).str[:10] == "2026-07-21"]
        amd = tr[
            (tr["date"].astype(str).str[:10] == "2026-07-21")
            & (tr["symbol"] == "AMD")
            & (tr["dir"] == "UP")
        ]
        t02 = tr[
            (tr["date"].astype(str).str[:10] == "2026-07-02")
            & (tr["symbol"] == "AMD")
            & (tr["dir"] == "DN")
        ]
        row = {
            "arm": name,
            "total_ret": float(s.get("total_ret") or 0.0),
            "maxdd": float(s.get("maxdd") or 0.0),
            "n_trades": int(s.get("n_trades") or 0),
            "n_WAVE_ABORT": int(reasons.get("WAVE_ABORT", 0)),
            "day_0721": day21,
            "amd_up_0721_ret": float(amd.iloc[0]["ret"]) if not amd.empty else None,
            "amd_up_0721_reason": str(amd.iloc[0]["reason"]) if not amd.empty else None,
            "amd_dn_0702_ret": float(t02.iloc[0]["ret"]) if not t02.empty else None,
            "amd_dn_0702_reason": str(t02.iloc[0]["reason"]) if not t02.empty else None,
        }
        rows.append(row)
        print(
            f"  ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} "
            f"0721={None if day21 is None else f'{day21*100:+.2f}%'} "
            f"AMD21={row['amd_up_0721_ret']}({row['amd_up_0721_reason']}) "
            f"AMD02={row['amd_dn_0702_ret']}({row['amd_dn_0702_reason']}) "
            f"wa={row['n_WAVE_ABORT']}",
            flush=True,
        )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    ctrl = next(r for r in rows if r["arm"] == "CTRL0")
    best = None
    for r in rows:
        if r["arm"] == "CTRL0":
            continue
        keep = r["total_ret"] / ctrl["total_ret"] if ctrl["total_ret"] > 1e-9 else None
        better_21 = (
            r["day_0721"] is not None
            and ctrl["day_0721"] is not None
            and r["day_0721"] > ctrl["day_0721"] + 1e-6
        )
        keep_dn_winner = (
            r["amd_dn_0702_ret"] is not None
            and ctrl["amd_dn_0702_ret"] is not None
            and r["amd_dn_0702_ret"] + 0.05 >= ctrl["amd_dn_0702_ret"]
        )
        ok = keep is not None and keep >= 0.95 and better_21 and keep_dn_winner
        r["keep_vs_ctrl"] = keep
        r["pass"] = ok
        if ok and (best is None or (r["day_0721"] or -9) > (best["day_0721"] or -9)):
            best = r

    decision = "PROMOTE_NARROW_WAVE_ABORT" if best else "NO_NARROW_PASS"
    summary = {
        "decision": decision,
        "window": f"{args.start}..{args.end}",
        "pass_rule": "keep>=0.95 AND better 7/21 day AND keep 7/02 AMD DN winner",
        "best": best,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# July Narrow WAVE_ABORT Ablation",
                "",
                f"**Decision: `{decision}`**",
                "",
                "```json",
                json.dumps(best, indent=2, default=str),
                "```",
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
