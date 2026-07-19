#!/usr/bin/env python3
"""Ablate open_ladder width / fixed rung: ATM .. OTM5 return gap.

Two modes (both run by default on peer3 May–Jul):
  - cap_k:  ladder_otm_rungs=k → nearest among ATM..OTMk (real knob)
  - fixed_k: lock map filtered to rung k only → force that strike rung

Usage:
  python -m maga7.tools.run_ladder_rung_ablation
  python -m maga7.tools.run_ladder_rung_ablation --mode cap --start 2026-05-01 --end 2026-07-17
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.open_lock import ladder_bucket_id
from maga7.common.replay import run_offline_replay

DEFAULT_PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _summary_row(arm: str, mode: str, k: int, result: dict) -> dict:
    s = result.get("summary") or result
    return {
        "arm": arm,
        "mode": mode,
        "rung_k": k,
        "label": "ATM" if k == 0 else f"OTM{k}",
        "n_trades": s.get("n_trades"),
        "total_ret": s.get("total_ret"),
        "maxdd": s.get("maxdd"),
        "trade_win": s.get("trade_win"),
        "trade_exp": s.get("trade_exp"),
        "day_win": s.get("day_win"),
        "end_equity": s.get("end_equity"),
        "n_signals_topk": s.get("n_signals_topk"),
    }


def _filter_lock_map(src: Path, dst: Path, rung: int) -> Path:
    df = pd.read_parquet(src)
    if "ladder_rung" not in df.columns:
        raise RuntimeError(f"lock map missing ladder_rung: {src}")
    sub = df[df["ladder_rung"].astype(int) == int(rung)].copy()
    # Keep only matching direction buckets for safety
    want = {
        ladder_bucket_id("p", rung),
        ladder_bucket_id("c", rung),
    }
    if "bucket_id" in sub.columns:
        sub = sub[sub["bucket_id"].astype(int).isin(want)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dst, index=False)
    return dst


def _run_one(profile: dict, *, tag_dir: Path, scheme: str) -> dict:
    tag_dir.mkdir(parents=True, exist_ok=True)
    result = run_offline_replay(profile, scheme=scheme)
    (tag_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    result["trades"].to_csv(tag_dir / "trades.csv", index=False)
    result["daily"].to_csv(tag_dir / "daily.csv", index=False)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="ATM..OTM5 ladder rung ablation")
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--start", default="2026-05-01")
    p.add_argument("--end", default="2026-07-17")
    p.add_argument("--scheme", default="single")
    p.add_argument(
        "--mode",
        default="both",
        choices=["cap", "fixed", "both"],
        help="cap=ladder width 0..5; fixed=force single rung; both=run all",
    )
    p.add_argument("--rungs", default="0,1,2,3,4,5", help="comma list of k")
    p.add_argument(
        "--tag",
        default="ladder_rung_ablation_peer3_may_jul",
        help="results subfolder under paths.results_dir",
    )
    args = p.parse_args()
    ks = [int(x) for x in str(args.rungs).split(",") if str(x).strip() != ""]

    base = load_profile(args.profile)
    base["date_range"] = {"start": args.start, "end": args.end}
    base.setdefault("trade", {})["contract_mode"] = "open_ladder"
    out_root = Path(base["_paths"]["results_dir"]) / args.tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lock_src = Path(base["_paths"]["open_locked_map"]).expanduser()
    rows: list[dict] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="ladder_rung_ab_"))

    try:
        modes = []
        if args.mode in {"cap", "both"}:
            modes.append("cap")
        if args.mode in {"fixed", "both"}:
            modes.append("fixed")

        for mode in modes:
            for k in ks:
                import copy

                prof = copy.deepcopy(base)
                arm = f"{mode}_rung{k}"
                label = "ATM" if k == 0 else f"OTM{k}"
                print(f"\n=== {arm} ({label}) ===", flush=True)
                if mode == "cap":
                    prof["trade"]["ladder_otm_rungs"] = int(k)
                    prof.setdefault("lock", {})["otm_rungs"] = int(k)
                else:
                    # Full resolve width but only one rung present in map.
                    prof["trade"]["ladder_otm_rungs"] = 5
                    prof.setdefault("lock", {})["otm_rungs"] = 5
                    filtered = tmp_root / f"lock_rung{k}.parquet"
                    _filter_lock_map(lock_src, filtered, k)
                    prof["_paths"]["open_locked_map"] = str(filtered)
                    # keep profile paths string for logging
                    prof.setdefault("paths", {})["open_locked_map"] = str(filtered)

                result = _run_one(prof, tag_dir=out_root / arm, scheme=args.scheme)
                row = _summary_row(arm, mode, k, result)
                rows.append(row)
                print(
                    f"  n={row['n_trades']} total_ret={row['total_ret']:.3f} "
                    f"maxdd={row['maxdd']:.3f} win={row['trade_win']:.3f} "
                    f"exp={row['trade_exp']:.3f}",
                    flush=True,
                )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    board = pd.DataFrame(rows)
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(
        board.to_json(orient="records", indent=2), encoding="utf-8"
    )

    # Markdown
    lines = [
        "# Ladder rung ablation（peer3）",
        "",
        f"- profile: `{args.profile}`",
        f"- window: {args.start} → {args.end}",
        f"- scheme: {args.scheme}",
        "",
        "## cap：`ladder_otm_rungs=k`（在 ATM..OTMk 里选最近）",
        "",
    ]
    cap = board[board["mode"] == "cap"].sort_values("rung_k")
    if not cap.empty:
        lines.append(
            cap[
                [
                    "label",
                    "rung_k",
                    "n_trades",
                    "total_ret",
                    "maxdd",
                    "trade_win",
                    "trade_exp",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    lines += ["", "## fixed：强制只用第 k 档行权价", ""]
    fix = board[board["mode"] == "fixed"].sort_values("rung_k")
    if not fix.empty:
        lines.append(
            fix[
                [
                    "label",
                    "rung_k",
                    "n_trades",
                    "total_ret",
                    "maxdd",
                    "trade_win",
                    "trade_exp",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    lines += [
        "",
        "## 怎么读",
        "",
        "- **cap**：真实旋钮。k=0 只 ATM；k=5 与现基线相同。",
        "- **fixed**：极端对照。强制买开盘锁的第 k 档（信号时可能已 ITM/深虚）。",
        "- 若 cap 在 k=2~3 已接近 k=5，说明再放宽到 OTM5 边际收益有限、风险主要在远档。",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== scoreboard ===")
    print(board.to_string(index=False))
    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
