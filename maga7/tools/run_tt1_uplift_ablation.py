#!/usr/bin/env python3
"""Ablate Tue/Thu 1DTE uplift levers on research L2 baseline (one arm at a time).

Levers (vs frozen peer3 L2, May–Jul):
  01  symbol: prefer bigger |from_prev| (rank / displace / commit)
  02  hold:   longer extend / unconditional T45
  03  contract: tighter clear-OTM / max |otm| entry gate
  04  entry:  confirm bars after Rule-A

Reports whole-window metrics + Tue/Thu 1DTE slice.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

DEFAULT_PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _tt1_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "tt1_n": 0,
            "tt1_win": float("nan"),
            "tt1_exp": float("nan"),
            "tt1_pnl": 0.0,
            "tt1_hit_amd_share": float("nan"),
        }
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str))
    df["weekday"] = df["date"].dt.weekday
    df["dte"] = pd.to_numeric(df.get("sig_dte"), errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    sf = pd.to_numeric(df.get("size_frac"), errors="coerce").fillna(0.2)
    df["pnl"] = sf * df["ret"]
    tt1 = df[(df["weekday"].isin([1, 3])) & (df["dte"] == 1)]
    if tt1.empty:
        return {
            "tt1_n": 0,
            "tt1_win": float("nan"),
            "tt1_exp": float("nan"),
            "tt1_pnl": 0.0,
            "tt1_amd_n": 0,
        }
    return {
        "tt1_n": int(len(tt1)),
        "tt1_win": float((tt1["ret"] > 0).mean()),
        "tt1_exp": float(tt1["ret"].mean()),
        "tt1_pnl": float(tt1["pnl"].sum()),
        "tt1_amd_n": int((tt1["symbol"] == "AMD").sum()),
        "tt1_tue_exp": float(tt1.loc[tt1["weekday"] == 1, "ret"].mean())
        if (tt1["weekday"] == 1).any()
        else float("nan"),
        "tt1_thu_exp": float(tt1.loc[tt1["weekday"] == 3, "ret"].mean())
        if (tt1["weekday"] == 3).any()
        else float("nan"),
    }


def _row(name: str, lever: str, result: dict[str, Any]) -> dict[str, Any]:
    s = result["summary"]
    tt = _tt1_metrics(result["trades"])
    return {
        "arm": name,
        "lever": lever,
        "n_trades": s.get("n_trades"),
        "total_ret": s.get("total_ret"),
        "maxdd": s.get("maxdd"),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "day_win": s.get("day_win"),
        "end_equity": s.get("end_equity"),
        **tt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--start-date", default="2026-05-01")
    p.add_argument("--end-date", default="2026-07-17")
    p.add_argument("--tag", default="tt1_uplift_ablation_peer3_may_jul")
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated arm names to run (default: all)",
    )
    args = p.parse_args()

    base = load_profile(args.profile)
    base["date_range"]["start"] = args.start_date
    base["date_range"]["end"] = args.end_date

    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # (name, lever, trade_over, signal_over)
    variants: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
        ("00_baseline", "base", {}, {}),
        # 01 symbol → bigger mover
        ("01a_rank_fp", "symbol", {}, {"rank_by": "abs_from_prev"}),
        (
            "01b_displace_fp",
            "symbol",
            {
                "displace_on_later": True,
                "displace_universe": "all_first",
                "displace_score": "abs_from_prev",
                "displace_min_score_ratio": 1.0,
            },
            {},
        ),
        (
            "01c_commit_1100_fp",
            "symbol",
            {
                "topk_commit_tod": "11:00",
                "topk_rank": "abs_from_prev",
                "topk_post_commit_fill": True,
            },
            {},
        ),
        # 02 hold → less hard T+30 death
        (
            "02a_hold_ext60",
            "hold",
            {"hold_extend_minutes": 60},
            {},
        ),
        (
            "02b_hold45_hard",
            "hold",
            {
                "exit_mode": "none",
                "hold_minutes": 45,
                "hold_extend_minutes": 45,
            },
            {},
        ),
        (
            "02c_hold_ext60_mtm0",
            "hold",
            {
                "hold_minutes": 30,
                "hold_extend_minutes": 60,
                "hold_extend_mtm_min": 0.0,
                "hold_extend_require_mf": False,
            },
            {},
        ),
        # 03 contract → nearer ATM
        ("03a_clear_otm_50bps", "contract", {"clear_otm_ban_0dte_pct": 0.005}, {}),
        ("03b_max_otm_50bps", "contract", {"max_entry_abs_otm_pct": 0.005}, {}),
        ("03c_max_otm_100bps", "contract", {"max_entry_abs_otm_pct": 0.01}, {}),
        # 04 entry timing
        (
            "04a_confirm2_mf",
            "entry",
            {"entry_confirm_bars": 2, "entry_confirm_mode": "mf"},
            {},
        ),
        (
            "04b_confirm3_both",
            "entry",
            {"entry_confirm_bars": 3, "entry_confirm_mode": "both"},
            {},
        ),
        # Confirm only on Tue/Thu so MWF 0DTE core stays untouched
        (
            "04c_confirm2_mf_tt",
            "entry",
            {
                "entry_confirm_bars": 2,
                "entry_confirm_mode": "mf",
                "entry_confirm_weekdays": [1, 3],
            },
            {},
        ),
        (
            "04d_confirm3_both_tt",
            "entry",
            {
                "entry_confirm_bars": 3,
                "entry_confirm_mode": "both",
                "entry_confirm_weekdays": [1, 3],
            },
            {},
        ),
        # Best contract gate + TT-only confirm
        (
            "05_maxotm50_confirm2_tt",
            "combo",
            {
                "max_entry_abs_otm_pct": 0.005,
                "entry_confirm_bars": 2,
                "entry_confirm_mode": "mf",
                "entry_confirm_weekdays": [1, 3],
            },
            {},
        ),
    ]

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        variants = [v for v in variants if v[0] in only]

    scoreboard: list[dict[str, Any]] = []
    for name, lever, trade_over, sig_over in variants:
        prof = deepcopy(base)
        for k, v in trade_over.items():
            prof.setdefault("trade", {})[k] = v
        for k, v in sig_over.items():
            prof.setdefault("signal", {})[k] = v
        print(f"==> {name} [{lever}]", flush=True)
        result = run_offline_replay(prof, scheme="single")
        arm_dir = out / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "summary.json").write_text(
            json.dumps(result["summary"], indent=2, default=str), encoding="utf-8"
        )
        result["trades"].to_csv(arm_dir / "trades.csv", index=False)
        result["daily"].to_csv(arm_dir / "daily.csv", index=False)
        row = _row(name, lever, result)
        scoreboard.append(row)
        print(
            f"    total={row['total_ret']:+.1%} dd={row['maxdd']:.1%} n={row['n_trades']} "
            f"| tt1 n={row['tt1_n']} win={row['tt1_win']:.0%} exp={row['tt1_exp']:+.1%} "
            f"pnl={row['tt1_pnl']:+.3f}",
            flush=True,
        )
        pd.DataFrame(scoreboard).to_csv(out / "scoreboard.csv", index=False)

    sb = pd.DataFrame(scoreboard)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(
        sb.to_json(orient="records", indent=2), encoding="utf-8"
    )

    if not sb.empty and "00_baseline" in set(sb["arm"]):
        base_ret = float(sb.loc[sb["arm"] == "00_baseline", "total_ret"].iloc[0])
        base_tt = float(sb.loc[sb["arm"] == "00_baseline", "tt1_pnl"].iloc[0])
        sb = sb.copy()
        sb["d_total_ret"] = sb["total_ret"] - base_ret
        sb["d_tt1_pnl"] = sb["tt1_pnl"] - base_tt
        sb.to_csv(out / "scoreboard.csv", index=False)

    lines = [
        "# Tue/Thu 1DTE uplift ablation (L2 peer3)",
        "",
        f"Window: {args.start_date} → {args.end_date}",
        f"Profile: `{args.profile}`",
        "",
        sb.to_markdown(index=False, floatfmt=".4f") if hasattr(sb, "to_markdown") else sb.to_string(index=False),
        "",
        "## Readout",
        "",
        "- Keep arms that lift **tt1_pnl** without wrecking **total_ret**.",
        "- Symbol lever aims at AMD/biggest; hold lever at T+30 deaths; contract at far OTM; entry at 10:xx chase.",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "scoreboard.csv", flush=True)


if __name__ == "__main__":
    main()
