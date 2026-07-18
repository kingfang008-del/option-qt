#!/usr/bin/env python3
"""Dual-window scoreboard: freeze baseline vs LGBM bouncer scale/block.

Windows default: May–Jul (strong) + Feb–Apr (weak). Acceptance hint (TCN lesson):
May–Jul ≥95% of baseline ret AND weak window improved before promoting.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay


def _run(prof: dict, *, start: str, end: str, tag: str, out_root: Path) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    summary = res["summary"]
    sub = out_root / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    res["trades"].to_csv(sub / "trades.csv", index=False)
    res["daily"].to_csv(sub / "daily.csv", index=False)
    return {
        "tag": tag,
        "start": start,
        "end": end,
        "total_ret": float(summary.get("total_ret", summary.get("equity_end", 0)) or 0),
        "maxdd": float(summary.get("maxdd", summary.get("max_dd", 0)) or 0),
        "n_trades": int(summary.get("n_trades", 0) or 0),
        "win_rate": summary.get("win_rate"),
        "n_lgbm_block": summary.get("n_lgbm_block"),
        "n_lgbm_scale": summary.get("n_lgbm_scale"),
        "equity_end": summary.get("equity_end"),
        "raw": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--model", default="maga7/results/lgbm_bouncer/lgbm_bouncer_v1.txt")
    ap.add_argument("--out", default="maga7/results/lgbm_bouncer/scoreboard_dual_window")
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-17")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument("--p-min", type=float, nargs="+", default=[0.50, 0.55])
    ap.add_argument("--scale-when-low", type=float, default=0.5)
    ap.add_argument(
        "--actions",
        nargs="+",
        default=["scale"],
        choices=["scale", "block"],
        help="gate actions to sweep (default: scale only)",
    )
    ap.add_argument(
        "--only-directions",
        nargs="*",
        default=None,
        help="e.g. DN — only score these dirs (default: leave profile / model meta)",
    )
    args = ap.parse_args()

    base = load_profile(args.profile)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    model = str(Path(args.model).expanduser())

    windows = [
        ("strong_may_jul", args.strong_start, args.strong_end),
        ("weak_feb_apr", args.weak_start, args.weak_end),
    ]
    board = []

    for wname, start, end in windows:
        row_b = _run(base, start=start, end=end, tag=f"{wname}__baseline", out_root=out_root)
        row_b.pop("raw", None)
        board.append({**row_b, "variant": "baseline", "window": wname, "vs_baseline_ret": 1.0})
        base_ret = float(row_b["total_ret"])

        for thr in args.p_min:
            for action in args.actions:
                p = copy.deepcopy(base)
                p["lgbm_bouncer"] = {
                    "enabled": True,
                    "action": action,
                    "p_min": float(thr),
                    "scale_when_low": float(args.scale_when_low),
                    "model_path": model,
                    "block_on_missing": False,
                }
                if args.only_directions is not None:
                    p["lgbm_bouncer"]["only_directions"] = [
                        str(x).upper() for x in args.only_directions
                    ]
                tag = f"{wname}__lgbm_{action}_p{int(thr * 100):02d}"
                row = _run(p, start=start, end=end, tag=tag, out_root=out_root)
                row.pop("raw", None)
                ret = float(row["total_ret"])
                vs = (ret / base_ret) if base_ret != 0 else None
                board.append(
                    {
                        **row,
                        "variant": f"lgbm_{action}_p{thr:.2f}",
                        "window": wname,
                        "vs_baseline_ret": vs,
                    }
                )

    (out_root / "scoreboard.json").write_text(
        json.dumps(board, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    pd.DataFrame(board).to_csv(out_root / "scoreboard.csv", index=False)
    print(pd.DataFrame(board)[
        [c for c in ("window", "variant", "total_ret", "maxdd", "n_trades", "vs_baseline_ret", "n_lgbm_scale", "n_lgbm_block") if c in pd.DataFrame(board).columns]
    ].to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
