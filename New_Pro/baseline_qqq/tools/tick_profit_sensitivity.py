#!/usr/bin/env python3
"""0DTE tick 浮盈 trigger×keep 敏感性网格(oracle 入场 + 固定 rails)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.qqq import config as qcfg

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from raw1s_rule_validation import (  # noqa: E402
    build_minute_frame,
    compute_oracle_edge,
    discover_raw1s_days,
    load_raw1s_bucket_day,
)


def _rails_with_tick_profit(
    trigger: Optional[float],
    keep: float,
    *,
    ladder_on: bool = True,
) -> ExitRailsConfig:
    base = qcfg.EXIT_RAILS
    ladder = base.tick_profit_ladder if ladder_on else ()
    return ExitRailsConfig(
        **{
            **base.__dict__,
            "tick_profit_trigger_roi": trigger,
            "tick_profit_keep_ratio": keep,
            "tick_profit_ladder": ladder,
        }
    )


def _run_combo(
    days: Sequence[Tuple[Path, object, object]],
    rails: ExitRailsConfig,
    fill_model: OptionSpreadFillModel,
) -> dict:
    day_rois: List[float] = []
    worst_trades: List[float] = []
    hit_flags: List[float] = []
    tick_profit_exits = 0
    total_trades = 0

    for _fp, ticks, minute in days:
        if minute.empty or len(minute) < 60:
            continue
        minute_e = compute_oracle_edge(minute, fill_model, hold_bars=5)
        tick_df = ticks[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
        r = run_event_replay(
            minute_e,
            fill_model,
            qcfg.REPLAY,
            rails,
            tick_df=tick_df,
            edge_col="oracle_edge",
            event_cfg=EventReplayConfig(tick_disaster_stop=True),
        )
        if not r.trades:
            day_rois.append(0.0)
            continue
        rets = np.array([t.net_return for t in r.trades], dtype=np.float64)
        day_rois.append(float(np.prod(1.0 + rets) - 1.0))
        worst_trades.append(float(rets.min()))
        hit_flags.extend((rets > 0).astype(float).tolist())
        total_trades += len(r.trades)
        for t in r.trades:
            if t.exit_reason in ("TICK_PROFIT_TRAIL", "TICK_PROFIT_STEP"):
                tick_profit_exits += 1

    if not day_rois:
        return {"days": 0}

    dr = np.array(day_rois, dtype=np.float64)
    bad3 = float(np.sort(dr)[:3].sum()) if len(dr) >= 3 else float(dr.sum())
    big_win_days = ["2026-06-12", "2026-06-25", "2026-06-29"]
    # big_win proxy: top-3 day roi mean
    top3_mean = float(np.mean(np.sort(dr)[-3:])) if len(dr) >= 3 else float(dr.max())

    return {
        "days": len(day_rois),
        "trades": total_trades,
        "win_days": int((dr > 0).sum()),
        "hit_rate": float(np.mean(hit_flags)) if hit_flags else 0.0,
        "day_roi_mean": float(dr.mean()),
        "day_roi_median": float(np.median(dr)),
        "compound_day_roi": float(np.prod(1.0 + dr) - 1.0),
        "worst_trade_min": float(np.min(worst_trades)) if worst_trades else 0.0,
        "worst3_day_sum": bad3,
        "top3_day_mean": top3_mean,
        "tick_profit_exit_frac": float(tick_profit_exits / total_trades) if total_trades else 0.0,
    }


def load_days(raw_dir: Path, symbol: str, glob_pat: str, bucket: int) -> list:
    files = discover_raw1s_days(raw_dir, symbol, glob_pattern=glob_pat)
    out = []
    for fp in files:
        if fp.stem.endswith("2026-06-30"):
            continue
        ticks = load_raw1s_bucket_day(fp, bucket)
        minute = build_minute_frame(ticks)
        if ticks.empty:
            continue
        out.append((fp, ticks, minute))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="tick_profit trigger×keep sensitivity")
    ap.add_argument("--raw-1s-dir", default="/mnt/s990/data/raw_1s/options_databento")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--bucket", type=int, default=2)
    ap.add_argument("--glob", default="QQQ_2026-06-*.parquet")
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_0dte_tickprofit_sensitivity_2026m06.json",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_1s_dir).expanduser()
    days = load_days(raw_dir, args.symbol, args.glob, args.bucket)
    fm = qcfg.FILL_MODEL

    triggers = [None, 0.15, 0.20, 0.25]
    keeps = [0.45, 0.50, 0.55, 0.60]

    rows = []
    for trig in triggers:
        for keep in keeps:
            if trig is None and keep != 0.50:
                continue
            label = f"off" if trig is None else f"t{int(trig*100):02d}_k{int(keep*100):02d}"
            rails = _rails_with_tick_profit(trig, keep)
            stats = _run_combo(days, rails, fm)
            rows.append(
                {
                    "label": label,
                    "tick_profit_trigger_roi": trig,
                    "tick_profit_keep_ratio": keep,
                    **stats,
                }
            )

    # rank: maximize win_days, then minimize worst3_day_sum (less negative), then day_roi_mean
    ranked = sorted(
        [r for r in rows if r.get("days", 0) > 0],
        key=lambda r: (
            r["win_days"],
            r["worst3_day_sum"],
            r["day_roi_mean"],
        ),
        reverse=True,
    )

    result = {
        "meta": {
            "raw_1s_dir": str(raw_dir),
            "symbol": args.symbol,
            "bucket": args.bucket,
            "glob": args.glob,
            "days_loaded": len(days),
            "skip": "2026-06-30",
            "grid": {"triggers": [t for t in triggers if t is not None], "keeps": keeps},
        },
        "rows": rows,
        "ranked_top5": ranked[:5],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"days={len(days)} combos={len(rows)} -> {out}\n")
    print(f"{'label':<12} {'win':>4} {'hit':>6} {'dayμ':>7} {'worst3Σ':>8} {'top3μ':>7} {'tp%':>5}")
    for r in ranked:
        print(
            f"{r['label']:<12} {r['win_days']:>4}/{r['days']:<2} "
            f"{r['hit_rate']:>5.1%} {r['day_roi_mean']:>6.1%} "
            f"{r['worst3_day_sum']:>+7.1%} {r['top3_day_mean']:>6.1%} "
            f"{r['tick_profit_exit_frac']:>4.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
