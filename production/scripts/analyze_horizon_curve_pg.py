#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""理论持仓时长曲线分析: 在 alpha_logs 选出的候选基础上, 重建每分钟的累计 ROI,
回答 "持仓 N 分钟的平均 ROI 是多少, 何时见顶, 何时回吐".

直接对生产平仓策略给出锚点参考: 当前 30 分钟 horizon 是不是最优?
要不要 15m 提前止盈? 60m 长持有效吗?

不依赖真实订单 (订单数据只覆盖 04-24 一天), 复用 alpha_logs + option_snapshots_1m 跨 6 天.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_alpha_priority_pg import (  # noqa: E402
    _lookup_snapshot_near,
    _pick_atm_mid,
    _safe_float,
    load_alpha_rows,
    load_option_paths,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="理论持仓时长曲线 (跨多日)")
    p.add_argument("--dates", required=True, help="逗号分隔, 如 2026-04-17,2026-04-20,...")
    p.add_argument("--alpha-floor", type=float, default=0.6)
    p.add_argument("--selected-per-frame", type=int, default=3)
    p.add_argument("--horizon-mins", type=int, default=60)
    p.add_argument("--cooldown-mins", type=int, default=15)
    p.add_argument("--top-quantile", type=float, default=0.5,
                   help="只保留 |alpha| 排在前 quantile 的候选 (避免噪声)")
    return p.parse_args()


def _fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:+.2f}%"


def select_candidates(alpha_rows: list[dict[str, Any]], alpha_floor: float,
                      selected_per_frame: int) -> list[dict[str, Any]]:
    """直接复用现网逻辑: 每分钟按 |alpha| 选 top-K (基线公式).

    返回 list of {symbol, ts, datetime_ny, alpha, side(±1)}.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    out: list[dict[str, Any]] = []
    for ts in sorted(grouped.keys()):
        candidates = []
        for r in grouped[ts]:
            a = _safe_float(r["alpha"])
            if abs(a) < alpha_floor:
                continue
            candidates.append((abs(a), a, str(r["symbol"]), str(r["datetime_ny"])))
        candidates.sort(reverse=True)
        for _, a, sym, dt in candidates[:selected_per_frame]:
            out.append({
                "symbol": sym, "ts": ts, "datetime_ny": dt,
                "alpha": a, "side": 1 if a >= 0 else -1,
            })
    return out


def build_roi_paths(
    selected: list[dict[str, Any]],
    option_paths: dict[str, dict[int, str]],
    horizon_mins: int,
    cooldown_mins: int,
) -> list[list[float | None]]:
    """每个候选: 返回长度 horizon_mins 的 ROI 列表 (持仓 1, 2, ..., N 分钟).

    None 表示该分钟数据缺失.
    """
    horizon_sec = horizon_mins * 60
    cooldown_sec = cooldown_mins * 60
    last_ts: dict[tuple[str, int], int] = {}
    out: list[list[float | None]] = []

    for cand in sorted(selected, key=lambda x: x["ts"]):
        side = "CALL" if cand["alpha"] >= 0 else "PUT"
        key = (cand["symbol"], cand["side"])
        if cand["ts"] - last_ts.get(key, -10**12) < cooldown_sec:
            continue
        series = option_paths.get(cand["symbol"], {})
        entry_snap = _lookup_snapshot_near(series, cand["ts"])
        entry_mid = _pick_atm_mid(entry_snap, side)
        if entry_mid is None or entry_mid <= 0:
            continue

        roi_by_minute: list[float | None] = [None] * horizon_mins
        end_ts = cand["ts"] + horizon_sec
        for ts_key in sorted(series.keys()):
            if ts_key <= cand["ts"] or ts_key > end_ts:
                continue
            offset_min = int((ts_key - cand["ts"]) // 60) - 1
            if 0 <= offset_min < horizon_mins:
                mid = _pick_atm_mid(series[ts_key], side)
                if mid is not None and mid > 0:
                    roi_by_minute[offset_min] = (float(mid) - float(entry_mid)) / float(entry_mid)
        if any(v is not None for v in roi_by_minute):
            out.append(roi_by_minute)
            last_ts[key] = cand["ts"]
    return out


def collect_one_day(date_str: str, args: argparse.Namespace) -> list[list[float | None]]:
    conn = psycopg2.connect(PG_DB_URL)
    try:
        alpha_rows = load_alpha_rows(conn, date_str)
        if not alpha_rows:
            return []
        selected = select_candidates(alpha_rows, args.alpha_floor, args.selected_per_frame)
        if not selected:
            return []
        symbols = sorted({c["symbol"] for c in selected})
        min_ts = min(c["ts"] for c in selected) - 600
        max_ts = max(c["ts"] for c in selected) + args.horizon_mins * 60 + 600
        option_paths = load_option_paths(conn, min_ts, max_ts, symbols)
        return build_roi_paths(selected, option_paths, args.horizon_mins, args.cooldown_mins)
    finally:
        conn.close()


def _curve_stats(roi_paths: list[list[float | None]], horizon: int) -> list[dict[str, float | int]]:
    """每个分钟点: 求 mean / median / p25 / p75 / coverage / hit_rate (>0)."""
    out = []
    for m in range(horizon):
        vals = [p[m] for p in roi_paths if p[m] is not None]
        if not vals:
            out.append({"m": m + 1, "n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "hit": 0.0})
            continue
        sorted_v = sorted(vals)
        out.append({
            "m": m + 1, "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p25": sorted_v[int(len(sorted_v) * 0.25)],
            "p75": sorted_v[int(len(sorted_v) * 0.75)],
            "hit": sum(1 for v in vals if v > 0) / len(vals),
        })
    return out


def _running_max_curve(roi_paths: list[list[float | None]], horizon: int) -> list[float]:
    """每条路径累计 max(running max), 然后跨路径求均值: 体现 "若启用 trail-stop 能锁多少利润"."""
    series_means = []
    for m in range(horizon):
        running_maxes = []
        for path in roi_paths:
            seen = [path[i] for i in range(m + 1) if path[i] is not None]
            if not seen:
                continue
            running_maxes.append(max(seen))
        if running_maxes:
            series_means.append(statistics.mean(running_maxes))
        else:
            series_means.append(0.0)
    return series_means


def _stop_loss_simulation(roi_paths: list[list[float | None]], horizon: int,
                           stop_loss: float, take_profit: float) -> tuple[float, float, int]:
    """模拟简单止盈/止损: 命中即在该分钟的 ROI 锁定, 否则到 horizon 末尾的实际 ROI.

    返回 (mean_roi, median_roi, n).
    """
    final_rois = []
    for path in roi_paths:
        locked = None
        last_known = 0.0
        for v in path[:horizon]:
            if v is None:
                continue
            last_known = v
            if take_profit is not None and v >= take_profit:
                locked = v
                break
            if stop_loss is not None and v <= stop_loss:
                locked = v
                break
        final_rois.append(locked if locked is not None else last_known)
    if not final_rois:
        return 0.0, 0.0, 0
    return statistics.mean(final_rois), statistics.median(final_rois), len(final_rois)


def _trail_stop_simulation(roi_paths: list[list[float | None]], horizon: int,
                            arm_at: float, give_back: float) -> tuple[float, float, int]:
    """模拟移动止盈: ROI 升至 arm_at 后开启 trail; 若回撤超过 give_back 即平仓.

    给出"如果用 trail stop, 平均 ROI 是多少".
    """
    final_rois = []
    for path in roi_paths:
        peak = -math.inf
        armed = False
        locked = None
        last_known = 0.0
        for v in path[:horizon]:
            if v is None:
                continue
            last_known = v
            if v > peak:
                peak = v
            if not armed and peak >= arm_at:
                armed = True
            if armed and v <= peak - give_back:
                locked = v
                break
        final_rois.append(locked if locked is not None else last_known)
    if not final_rois:
        return 0.0, 0.0, 0
    return statistics.mean(final_rois), statistics.median(final_rois), len(final_rois)


def main() -> None:
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print(f"日期: {dates}")
    print(f"参数: alpha_floor={args.alpha_floor} selected_per_frame={args.selected_per_frame} "
          f"horizon={args.horizon_mins}m cooldown={args.cooldown_mins}m")

    all_paths: list[list[float | None]] = []
    by_day: dict[str, list[list[float | None]]] = {}
    for d in dates:
        print(f"  载入 {d} ...", flush=True)
        paths = collect_one_day(d, args)
        by_day[d] = paths
        all_paths.extend(paths)
        print(f"    候选数 = {len(paths)}")

    if not all_paths:
        print("无可分析数据.")
        return

    horizon = args.horizon_mins
    stats = _curve_stats(all_paths, horizon)
    running_max = _running_max_curve(all_paths, horizon)

    print(f"\n========= 持仓时长 ROI 曲线 (跨 {len(dates)} 天, n={len(all_paths)} 条理论持仓) =========")
    print(f"{'分钟':>4} {'n':>5} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'hit%':>6} {'runMax':>8}")
    keypoints = [1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for m in keypoints:
        if m > horizon:
            continue
        s = stats[m - 1]
        rm = running_max[m - 1]
        print(
            f"{m:>4} {int(s['n']):>5} {_fmt_pct(s['mean']):>8} {_fmt_pct(s['median']):>8} "
            f"{_fmt_pct(s['p25']):>8} {_fmt_pct(s['p75']):>8} {s['hit'] * 100:>5.1f}% {_fmt_pct(rm):>8}"
        )

    best_mean_idx = max(range(horizon), key=lambda i: stats[i]['mean'])
    best_median_idx = max(range(horizon), key=lambda i: stats[i]['median'])
    best_runmax_idx = max(range(horizon), key=lambda i: running_max[i])
    print(
        f"\n→ mean ROI 峰值 = 持仓 {best_mean_idx + 1} 分钟 ({_fmt_pct(stats[best_mean_idx]['mean'])})"
    )
    print(f"→ median ROI 峰值 = 持仓 {best_median_idx + 1} 分钟 ({_fmt_pct(stats[best_median_idx]['median'])})")
    print(f"→ running-max ROI 峰值 = 持仓 {best_runmax_idx + 1} 分钟 ({_fmt_pct(running_max[best_runmax_idx])})")

    print(f"\n========= 平仓策略对比 (每个策略给出最终 ROI 均值) =========")
    print(f"{'strategy':<32} {'mean':>9} {'median':>9} {'lift_vs_30m':>12}")
    baseline_30m = stats[min(29, horizon - 1)]['mean']
    print(f"{'固定 30m 平仓 (基线)':<32} {_fmt_pct(baseline_30m):>9} {_fmt_pct(stats[min(29, horizon-1)]['median']):>9} {'-':>12}")
    for cap_min in [15, 20, 25, 35, 45, 60]:
        if cap_min > horizon:
            continue
        s = stats[cap_min - 1]
        lift = s['mean'] - baseline_30m
        print(f"{'固定 ' + str(cap_min) + 'm 平仓':<32} {_fmt_pct(s['mean']):>9} {_fmt_pct(s['median']):>9} {_fmt_pct(lift):>12}")

    for tp in [0.05, 0.10, 0.15, 0.20, 0.30]:
        m, md, n = _stop_loss_simulation(all_paths, horizon, stop_loss=-0.30, take_profit=tp)
        print(f"{'TP +' + f'{tp*100:.0f}' + '% / SL -30%':<32} {_fmt_pct(m):>9} {_fmt_pct(md):>9} {_fmt_pct(m - baseline_30m):>12}")

    for arm, gb in [(0.05, 0.03), (0.08, 0.05), (0.10, 0.05), (0.10, 0.08), (0.15, 0.08), (0.15, 0.10)]:
        m, md, n = _trail_stop_simulation(all_paths, horizon, arm_at=arm, give_back=gb)
        label = f"trail arm={int(arm*100)}% give={int(gb*100)}%"
        print(f"{label:<32} {_fmt_pct(m):>9} {_fmt_pct(md):>9} {_fmt_pct(m - baseline_30m):>12}")

    print(f"\n========= 各日候选数检查 =========")
    for d in dates:
        n = len(by_day.get(d, []))
        print(f"  {d}: {n} 条理论持仓")


if __name__ == "__main__":
    main()
