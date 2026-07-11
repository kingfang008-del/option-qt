#!/usr/bin/env python3
"""
Oracle 噪声注入消融:量化「信号要多准才能赚钱」。

方法:
  oracle_edge 是完美信号(未来 5bar fill 净收益)。给它注入不同强度的
  高斯噪声 → 得到不同"有效 IC"的退化信号 → 用与生产完全相同的
  replay 策略栈(qcfg.REPLAY + qcfg.EXIT_RAILS + tick rails)回放,
  画出 IC–PnL 衰减曲线。

  另跑「仅排序」变体:丢掉 edge 数值,只保留日内 top-k% 的二值信息,
  回答"粗排序是否足够"(若足够,模型任务可从回归降级为日内排序)。

输出:
  - JSON 报告(逐 sigma/seed 明细 + 汇总)
  - 控制台表格:alpha | rank_IC | win_days | day_roi_mean | compound
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
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

ENTRY_START = 15   # 与 REPLAY.session_entry_start_bar 一致
ENTRY_END = 300    # 与 REPLAY.session_entry_end_bar 一致


def _round4(x: float) -> float:
    return round(float(x), 4)


def load_days(
    raw_dir: Path,
    symbol: str,
    glob_pat: str,
    bucket: int,
    hold_bars: int,
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """[(date, minute_with_oracle, tick_df)];minute 已含 oracle_edge。"""
    out = []
    for fp in discover_raw1s_days(raw_dir, symbol, glob_pattern=glob_pat):
        date_str = fp.stem.split("_", 1)[-1]
        ticks = load_raw1s_bucket_day(fp, bucket)
        if ticks.empty:
            continue
        minute = build_minute_frame(ticks)
        if minute.empty or len(minute) < 60:
            continue
        minute = compute_oracle_edge(minute, qcfg.FILL_MODEL, hold_bars=hold_bars)
        tick_df = ticks[
            ["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]
        ]
        out.append((date_str, minute, tick_df))
    return out


def entry_window_mask(minute: pd.DataFrame) -> pd.Series:
    sb = minute["session_bar"].astype(int)
    return (sb >= ENTRY_START) & (sb <= ENTRY_END)


def pooled_edge_std(days: Sequence[Tuple[str, pd.DataFrame, pd.DataFrame]]) -> float:
    vals = []
    for _d, minute, _t in days:
        m = entry_window_mask(minute)
        v = minute.loc[m, "oracle_edge"].to_numpy(dtype=np.float64)
        vals.append(v[np.isfinite(v)])
    allv = np.concatenate(vals)
    return float(np.std(allv))


def rank_ic(noisy: np.ndarray, true: np.ndarray) -> float:
    ok = np.isfinite(noisy) & np.isfinite(true)
    if ok.sum() < 20:
        return float("nan")
    rho, _ = spearmanr(noisy[ok], true[ok])
    return float(rho)


def replay_with_signal(
    days: Sequence[Tuple[str, pd.DataFrame, pd.DataFrame]],
    signal_maker,
) -> dict:
    """signal_maker(date, minute) -> np.ndarray 信号列;返回日度汇总。"""
    day_rois: List[float] = []
    ics: List[float] = []
    total_trades = 0
    hits: List[float] = []
    worst_trade = 0.0
    exit_counts: Dict[str, int] = {}

    for date_str, minute, tick_df in days:
        sig = signal_maker(date_str, minute)
        m = minute.copy()
        m["ablation_edge"] = sig

        w = entry_window_mask(m).to_numpy()
        ics.append(rank_ic(sig[w], m["oracle_edge"].to_numpy()[w]))

        r = run_event_replay(
            m,
            qcfg.FILL_MODEL,
            qcfg.REPLAY,
            qcfg.EXIT_RAILS,
            tick_df=tick_df,
            edge_col="ablation_edge",
            event_cfg=EventReplayConfig(tick_disaster_stop=True),
        )
        if not r.trades:
            day_rois.append(0.0)
            continue
        rets = np.array([t.net_return for t in r.trades], dtype=np.float64)
        day_rois.append(float(np.prod(1.0 + rets) - 1.0))
        total_trades += len(rets)
        hits.extend((rets > 0).astype(float).tolist())
        worst_trade = min(worst_trade, float(rets.min()))
        for t in r.trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    dr = np.array(day_rois, dtype=np.float64)
    active = int((dr != 0).sum())
    return {
        "days": len(day_rois),
        "active_days": active,
        "win_days": int((dr > 0).sum()),
        "trades": total_trades,
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())),
        "day_roi_median": _round4(float(np.median(dr))),
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
        "worst3_sum": _round4(float(np.sort(dr)[:3].sum())) if len(dr) >= 3 else _round4(float(dr.sum())),
        "worst_trade": _round4(worst_trade),
        "rank_ic_mean": _round4(float(np.nanmean(ics))),
        "exit_reasons": exit_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="oracle noise-injection ablation")
    ap.add_argument("--raw-1s-dir", default="/mnt/s990/data/raw_1s/dte1_options")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--bucket", type=int, default=2)
    ap.add_argument("--glob", default="QQQ_2025-06-*.parquet")
    ap.add_argument("--hold-bars", type=int, default=5)
    ap.add_argument("--alphas", default="0,0.5,1,2,3,5,8,12")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--top-pcts", default="0.02,0.05,0.10")
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_2025m06_oracle_noise_ablation.json",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_1s_dir).expanduser()
    alphas = [float(x) for x in args.alphas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    top_pcts = [float(x) for x in args.top_pcts.split(",")]

    print(f"loading days: {args.glob} bucket={args.bucket} hold={args.hold_bars} ...")
    days = load_days(raw_dir, args.symbol, args.glob, args.bucket, args.hold_bars)
    if not days:
        print("no days loaded")
        return 1
    sigma_base = pooled_edge_std(days)
    print(f"days={len(days)} edge_std={sigma_base:.4f}\n")

    rows: List[dict] = []

    # --- A: 高斯噪声退化 ---
    for alpha in alphas:
        seed_stats = []
        for seed in seeds:
            rng = np.random.default_rng(seed)

            def make(date_str, minute, _a=alpha, _rng=rng):
                true = minute["oracle_edge"].to_numpy(dtype=np.float64)
                noise = _rng.normal(0.0, _a * sigma_base, size=len(true))
                return true + noise

            stats = replay_with_signal(days, make)
            seed_stats.append(stats)
        agg = {
            "variant": "gaussian",
            "alpha": alpha,
            "seeds": len(seeds),
            "rank_ic_mean": _round4(float(np.mean([s["rank_ic_mean"] for s in seed_stats]))),
            "win_days_mean": _round4(float(np.mean([s["win_days"] for s in seed_stats]))),
            "day_roi_mean": _round4(float(np.mean([s["day_roi_mean"] for s in seed_stats]))),
            "compound_mean": _round4(float(np.mean([s["compound"] for s in seed_stats]))),
            "compound_min": _round4(float(np.min([s["compound"] for s in seed_stats]))),
            "trades_mean": _round4(float(np.mean([s["trades"] for s in seed_stats]))),
            "hit_rate_mean": _round4(float(np.mean([s["hit_rate"] for s in seed_stats]))),
            "worst3_mean": _round4(float(np.mean([s["worst3_sum"] for s in seed_stats]))),
            "per_seed": seed_stats,
        }
        rows.append(agg)
        print(
            f"gaussian a={alpha:<4} IC={agg['rank_ic_mean']:+.3f} "
            f"win={agg['win_days_mean']:.1f}/{len(days)} "
            f"dayROI={agg['day_roi_mean']:+.1%} comp={agg['compound_mean']:+.1%} "
            f"(min {agg['compound_min']:+.1%}) trades={agg['trades_mean']:.0f} "
            f"hit={agg['hit_rate_mean']:.0%}"
        )

    # --- B: 仅排序(日内 top-k%) ---
    for pct in top_pcts:
        def make_rank(date_str, minute, _p=pct):
            true = minute["oracle_edge"].to_numpy(dtype=np.float64)
            sig = np.full(len(true), -1.0)
            w = entry_window_mask(minute).to_numpy()
            vals = np.where(w & np.isfinite(true), true, -np.inf)
            n_pick = max(1, int(round(w.sum() * _p)))
            top_idx = np.argsort(vals)[-n_pick:]
            sig[top_idx] = 0.10  # 远超阈值 → 保证入场资格,数值信息被抹平
            return sig

        stats = replay_with_signal(days, make_rank)
        stats.update({"variant": "rank_only", "top_pct": pct})
        rows.append(stats)
        print(
            f"rank_only top{pct:.0%}  IC={stats['rank_ic_mean']:+.3f} "
            f"win={stats['win_days']}/{len(days)} dayROI={stats['day_roi_mean']:+.1%} "
            f"comp={stats['compound']:+.1%} trades={stats['trades']} hit={stats['hit_rate']:.0%}"
        )

    result = {
        "meta": {
            "raw_1s_dir": str(raw_dir),
            "symbol": args.symbol,
            "bucket": args.bucket,
            "glob": args.glob,
            "hold_bars": args.hold_bars,
            "days": len(days),
            "edge_std": _round4(sigma_base),
            "entry_window": [ENTRY_START, ENTRY_END],
            "alphas": alphas,
            "seeds": seeds,
            "top_pcts": top_pcts,
            "note": "oracle_edge 含未来函数;本实验只用于量化信号质量要求",
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
