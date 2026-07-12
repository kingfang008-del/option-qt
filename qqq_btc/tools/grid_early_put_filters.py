#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早盘 PUT 过滤网格 —— 针对 July1 HARD_STOP 型亏损。

在已有 test_infer.parquet 上扫 put_early_* 组合,对比 baseline @25%。

用法:
  python qqq_btc/tools/grid_early_put_filters.py \\
      --infer qqq_btc/results/ft56_julw1_with_vix/test_infer.parquet \\
      --output-dir qqq_btc/results/early_put_filter_grid_julw1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg


def _acct(trades: pd.DataFrame, pos: float) -> float:
    if trades is None or len(trades) == 0:
        return 0.0
    eq = 1.0
    for r in trades["net_return"].astype(float):
        eq *= 1.0 + pos * float(r)
    return float(eq - 1.0)


def _day_stats(trades: pd.DataFrame, pos: float) -> Dict[str, Any]:
    if trades is None or len(trades) == 0:
        return {}
    t = trades.copy()
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    t["date"] = t["entry_ts"].dt.strftime("%Y-%m-%d")
    t["sb"] = (t["entry_ts"].dt.hour * 60 + t["entry_ts"].dt.minute) - (9 * 60 + 30)
    out = {}
    for d, g in t.groupby("date"):
        out[d] = {
            "n": int(len(g)),
            "acct25": _acct(g, pos),
            "sum_ret": float(g["net_return"].sum()),
            "exits": g["exit_reason"].astype(str).tolist(),
            "n_early": int((g["sb"] < 30).sum()),
            "early_hardstop": int(((g["sb"] < 30) & (g["exit_reason"].astype(str) == "HARD_STOP")).sum()),
        }
    return out


def run_one(df: pd.DataFrame, cfg, pos: float) -> Dict[str, Any]:
    result = run_strict_replay(
        df,
        qcfg.FILL_MODEL,
        cfg,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=qcfg.EDGE_Q10_COL,
        call_edge_col=qcfg.CALL_EDGE_COL,
        put_edge_col=qcfg.PUT_EDGE_COL,
        put_gate_col=qcfg.PUT_GATE_COL,
    )
    trades = result.trades_frame()
    summary = result.summary(position_frac=pos)
    return {
        "acct25": float(summary.get("total_net_return", _acct(trades, pos))),
        "n_trades": int(len(trades)),
        "hit_rate": float(summary.get("hit_rate") or 0.0),
        "sum_net": float(summary.get("sum_net_return") or 0.0),
        "trades": trades,
        "by_day": _day_stats(trades, pos),
    }


def build_grid() -> List[Dict[str, Any]]:
    """候选组合:早盘窗口 × vix 门槛 × open30 × range。"""
    early_bars = [30]
    vix_mins: List[Optional[float]] = [None, 0.55, 0.60, 0.65, 0.70]
    open30_mins: List[Optional[float]] = [None, 1e-12]  # None=off; 1e-12 ≈ require >0
    range_mins: List[Optional[float]] = [None, 0.002, 0.003]
    combos = []
    for eb in early_bars:
        for vx in vix_mins:
            for o30 in open30_mins:
                for rng in range_mins:
                    if vx is None and o30 is None and rng is None:
                        continue  # baseline 单独跑
                    combos.append(
                        {
                            "name": (
                                f"early<{eb}"
                                f"|vix>={vx if vx is not None else '-'}"
                                f"|open30>={('>0' if o30 is not None else '-')}"
                                f"|range>={rng if rng is not None else '-'}"
                            ),
                            "put_early_session_bar": eb,
                            "put_early_vix_min": vx,
                            "put_early_open30_max_min": o30,
                            "put_early_range30_min": rng,
                        }
                    )
    return combos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--infer",
        type=Path,
        default=Path("qqq_btc/results/ft56_julw1_with_vix/test_infer.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qqq_btc/results/early_put_filter_grid_julw1"),
    )
    ap.add_argument("--pos-frac", type=float, default=None)
    args = ap.parse_args()

    pos = float(args.pos_frac if args.pos_frac is not None else qcfg.REPLAY.position_frac)
    df = pd.read_parquet(args.infer)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = replace(
        qcfg.REPLAY,
        put_early_session_bar=None,
        put_early_vix_min=None,
        put_early_open30_max_min=None,
        put_early_range30_min=None,
    )
    baseline = run_one(df, base_cfg, pos)
    rows = [
        {
            "name": "BASELINE",
            "put_early_session_bar": None,
            "put_early_vix_min": None,
            "put_early_open30_max_min": None,
            "put_early_range30_min": None,
            "acct25": baseline["acct25"],
            "delta_pp": 0.0,
            "n_trades": baseline["n_trades"],
            "hit_rate": baseline["hit_rate"],
            "sum_net": baseline["sum_net"],
            "july1_early_hardstop_killed": baseline["by_day"].get("2026-07-01", {}).get("early_hardstop", 0) == 0,
            "july1_day_acct25": baseline["by_day"].get("2026-07-01", {}).get("acct25"),
            "july7_kept": baseline["by_day"].get("2026-07-07", {}).get("n", 0) > 0,
            "killed_big_gt30": False,
        }
    ]

    for spec in build_grid():
        cfg = replace(
            base_cfg,
            put_early_session_bar=spec["put_early_session_bar"],
            put_early_vix_min=spec["put_early_vix_min"],
            put_early_open30_max_min=spec["put_early_open30_max_min"],
            put_early_range30_min=spec["put_early_range30_min"],
        )
        r = run_one(df, cfg, pos)
        base_big = set()
        if len(baseline["trades"]):
            bt = baseline["trades"].copy()
            bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True).dt.tz_convert("America/New_York")
            for _, t in bt[bt["net_return"] > 0.30].iterrows():
                base_big.add((str(t["entry_ts"].date()), t["entry_ts"].strftime("%H:%M")))
        cur_keys = set()
        if len(r["trades"]):
            ct = r["trades"].copy()
            ct["entry_ts"] = pd.to_datetime(ct["entry_ts"], utc=True).dt.tz_convert("America/New_York")
            for _, t in ct.iterrows():
                cur_keys.add((str(t["entry_ts"].date()), t["entry_ts"].strftime("%H:%M")))
        killed_big = sorted(base_big - cur_keys)

        rows.append(
            {
                **spec,
                "acct25": r["acct25"],
                "delta_pp": r["acct25"] - baseline["acct25"],
                "n_trades": r["n_trades"],
                "hit_rate": r["hit_rate"],
                "sum_net": r["sum_net"],
                "july1_early_hardstop_killed": r["by_day"].get("2026-07-01", {}).get("early_hardstop", 0) == 0,
                "july1_day_acct25": r["by_day"].get("2026-07-01", {}).get("acct25"),
                "july7_kept": r["by_day"].get("2026-07-07", {}).get("n", 0) > 0,
                "killed_big_gt30": bool(killed_big),
                "killed_big_list": killed_big,
                "by_day": r["by_day"],
            }
        )

    grid = pd.DataFrame(rows)
    score = (
        grid["july1_early_hardstop_killed"].astype(int) * 100
        + grid["july7_kept"].astype(int) * 50
        - grid["killed_big_gt30"].astype(int) * 200
        + grid["acct25"] * 100
        + grid["july1_day_acct25"].fillna(0) * 20
    )
    grid["score"] = score
    grid_sorted = grid.sort_values(["score", "acct25"], ascending=False)

    csv_path = out_dir / "grid.csv"
    flat = grid_sorted.drop(columns=[c for c in ("by_day", "killed_big_list") if c in grid_sorted.columns])
    flat.to_csv(csv_path, index=False)

    cand = grid_sorted[
        grid_sorted["july1_early_hardstop_killed"]
        & grid_sorted["july7_kept"]
        & ~grid_sorted["killed_big_gt30"]
    ]
    # 偏好更简单: vix-only 或 open30-only 优先于三重叠加
    def _simplicity(row) -> int:
        n = 0
        if pd.notna(row.get("put_early_vix_min")):
            n += 1
        if pd.notna(row.get("put_early_open30_max_min")):
            n += 1
        if pd.notna(row.get("put_early_range30_min")):
            n += 1
        return n

    if len(cand):
        cand = cand.copy()
        cand["n_rules"] = cand.apply(_simplicity, axis=1)
        # 同分下优先 vix-only(July1/July7 判别最清晰),再 open30,再 range
        cand["pref"] = cand.apply(
            lambda r: (
                0
                if pd.notna(r.get("put_early_vix_min"))
                and pd.isna(r.get("put_early_open30_max_min"))
                and pd.isna(r.get("put_early_range30_min"))
                else 1
                if pd.notna(r.get("put_early_open30_max_min"))
                and pd.isna(r.get("put_early_vix_min"))
                and pd.isna(r.get("put_early_range30_min"))
                else 2
                if pd.notna(r.get("put_early_range30_min"))
                and pd.isna(r.get("put_early_vix_min"))
                and pd.isna(r.get("put_early_open30_max_min"))
                else 3
            ),
            axis=1,
        )
        cand = cand.sort_values(
            ["acct25", "pref", "n_rules", "put_early_vix_min"],
            ascending=[False, True, True, True],
        )
        recommend = cand.iloc[0].to_dict()
    else:
        recommend = grid_sorted.iloc[0].to_dict()

    summary = {
        "infer": str(args.infer),
        "position_frac": pos,
        "baseline_acct25": baseline["acct25"],
        "baseline_n": baseline["n_trades"],
        "baseline_by_day": baseline["by_day"],
        "n_grid": len(rows) - 1,
        "recommend": {
            k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
            for k, v in recommend.items()
            if k not in ("by_day",)
        },
        "top5": flat.head(5).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    print(f"BASELINE @25%: {baseline['acct25']*100:+.2f}%  trades={baseline['n_trades']}")
    print(f"grid n={len(rows)-1} → {csv_path}")
    print("\nTOP (kill July1 early HARD_STOP, keep July7, no big-winner kill):")
    cols = [
        "name", "acct25", "delta_pp", "n_trades",
        "july1_early_hardstop_killed", "july1_day_acct25", "july7_kept", "killed_big_gt30",
    ]
    show = cand[cols].head(12) if len(cand) else grid_sorted[cols].head(12)
    with pd.option_context("display.max_colwidth", 80, "display.width", 180):
        print(show.to_string(index=False))
    print("\nRECOMMEND:")
    print(json.dumps(summary["recommend"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
