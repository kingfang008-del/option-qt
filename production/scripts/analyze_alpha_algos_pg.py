#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次跑完所有候选 *alpha 排序算法*, 用同一份 alpha_logs/option_snapshots
评估理论 30m ROI, 输出统一对比表.

不调任何参数, 每个算法用一组合理的默认参数, 横向对比谁是源头最优.

算法清单 (族系/标签):
  A 单点强度
    A0 raw_abs           : |alpha|                    [现网基线]
    A1 alpha_pow_1.5     : |alpha|^1.5
    A2 alpha_pow_2       : |alpha|^2
    A3 frame_rank        : 同帧内按 |alpha| 排名 (1/N, 1-1/N...)
  B 波动归一化
    B1 alpha_over_iv     : |alpha| / iv                [可能是现网早期版本]
    B2 alpha_over_sqrt_iv: |alpha| / sqrt(iv)
  C 时间平滑
    C1 ema_short_6       : |EMA(alpha, 6)|
    C2 ema_long_12       : |EMA(alpha, 12)|
    C3 dema_diff         : |EMA_short - EMA_long|        (类 MACD)
  D 持续性
    D1 streak_mult       : |alpha| × (1 + streak/8 × 0.5)
    D2 consistency_mult  : |alpha| × (同方向占比 in last 6m)
    D3 flip_penalty      : |alpha| × (1 - flip_rate)
  E 信息论/夏普
    E1 short_sharpe      : mean(signed_alpha[-6:]) / max(0.1, std)
    E2 mean_signed_x_n   : |mean(signed_alpha[-6:])| × n_same_dir
  F 复合
    F1 blend_v3          : 0.4|alpha| + 0.6|EMA(alpha, 6)|       [前轮最优]
    F2 blend_v3_streak   : F1 × (1 + streak/6 × 0.5)            [本轮最优]

入选门槛: 统一用 |raw alpha| >= ALPHA_FLOOR (与现网一致, 公平比较).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psycopg2  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_alpha_priority_pg import (  # noqa: E402
    CandidateProfit,
    RankedCandidate,
    _safe_float,
    analyze_candidate_profits,
    load_alpha_rows,
    load_option_paths,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402


# ---------- 状态 ----------

@dataclass
class SymState:
    alpha_history: deque = field(default_factory=lambda: deque(maxlen=12))  # 带方向
    seen: int = 0
    streak: int = 0
    last_dir: int = 0
    ema_signed_6: float = 0.0
    ema_signed_12: float = 0.0


def _direction(v: float) -> int:
    return 1 if v >= 0 else -1


def update_state(st: SymState, alpha: float, *, streak_floor: float = 0.6) -> None:
    direction = _direction(alpha) if abs(alpha) >= 1e-9 else 0
    if direction != 0 and abs(alpha) >= streak_floor and direction == st.last_dir:
        st.streak += 1
    elif direction != 0 and abs(alpha) >= streak_floor:
        st.streak = 1
        st.last_dir = direction
    else:
        st.streak = 0
        st.last_dir = direction or st.last_dir

    if st.seen == 0:
        st.ema_signed_6 = float(alpha)
        st.ema_signed_12 = float(alpha)
    else:
        a6 = 2.0 / (6 + 1.0)
        a12 = 2.0 / (12 + 1.0)
        st.ema_signed_6 = float(a6 * alpha + (1 - a6) * st.ema_signed_6)
        st.ema_signed_12 = float(a12 * alpha + (1 - a12) * st.ema_signed_12)
    st.alpha_history.append(float(alpha))
    st.seen += 1


# ---------- 算法实现 ----------

# 每个算法签名: (alpha, iv, st, frame_alphas) -> (score, direction)
# frame_alphas 是当前帧所有候选 alpha (含全部 symbols, 用于 frame_rank)

AlgoFn = Callable[[float, float, SymState, list[float]], tuple[float, int]]


def algo_raw_abs(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(alpha), _direction(alpha)


def algo_alpha_pow_15(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(alpha) ** 1.5, _direction(alpha)


def algo_alpha_pow_2(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(alpha) ** 2.0, _direction(alpha)


def algo_frame_rank(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    abs_alpha = abs(alpha)
    rank_above = sum(1 for a in frame_alphas if abs(a) > abs_alpha)
    n = max(1, len(frame_alphas))
    score = 1.0 - rank_above / n  # 1.0 表示当前帧最强
    return score, _direction(alpha)


def algo_alpha_over_iv(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(alpha) / max(0.1, iv), _direction(alpha)


def algo_alpha_over_sqrt_iv(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(alpha) / math.sqrt(max(0.1, iv)), _direction(alpha)


def algo_ema_short_6(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(st.ema_signed_6), _direction(st.ema_signed_6)


def algo_ema_long_12(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    return abs(st.ema_signed_12), _direction(st.ema_signed_12)


def algo_dema_diff(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    diff = st.ema_signed_6 - st.ema_signed_12
    return abs(diff) * 2.0, _direction(diff) if abs(diff) > 1e-9 else _direction(alpha)


def algo_streak_mult(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    direction = _direction(alpha)
    valid_streak = st.streak if st.last_dir == direction else 0
    factor = 1.0 + min(valid_streak / 8.0, 1.0) * 0.5
    return abs(alpha) * factor, direction


def algo_consistency_mult(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    direction = _direction(alpha)
    if not st.alpha_history:
        return abs(alpha), direction
    recent = list(st.alpha_history)[-6:]
    same_dir = sum(1 for a in recent if (a >= 0) == (direction >= 0))
    consistency = same_dir / max(1, len(recent))
    return abs(alpha) * (0.5 + consistency), direction  # 0.5~1.5x


def algo_flip_penalty(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    direction = _direction(alpha)
    if len(st.alpha_history) < 2:
        return abs(alpha), direction
    recent = list(st.alpha_history)[-6:]
    flips = sum(1 for i in range(1, len(recent)) if (recent[i] >= 0) != (recent[i - 1] >= 0))
    flip_rate = flips / max(1, len(recent) - 1)
    return abs(alpha) * (1.0 - flip_rate), direction


def algo_short_sharpe(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    if len(st.alpha_history) < 3:
        return abs(alpha), _direction(alpha)
    recent = list(st.alpha_history)[-6:]
    n = len(recent)
    mean = sum(recent) / n
    var = sum((a - mean) ** 2 for a in recent) / max(1, n - 1)
    std = math.sqrt(var) if var > 1e-12 else 0.0
    direction = _direction(mean)
    score = abs(mean) / max(0.1, std)
    return score, direction


def algo_mean_signed_x_n(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    if len(st.alpha_history) < 2:
        return abs(alpha), _direction(alpha)
    recent = list(st.alpha_history)[-6:]
    n = len(recent)
    mean = sum(recent) / n
    direction = _direction(mean)
    same_dir = sum(1 for a in recent if (a >= 0) == (direction >= 0))
    return abs(mean) * (same_dir / max(1, n)), direction


def algo_blend_v3(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    smooth = st.ema_signed_6
    same = (alpha >= 0) == (smooth >= 0)
    keep = 1.0 if same else 0.4
    score = (0.4 * abs(alpha) + 0.6 * abs(smooth)) * keep
    return score, _direction(smooth)


def algo_blend_v3_streak(alpha: float, iv: float, st: SymState, frame_alphas: list[float]) -> tuple[float, int]:
    score, direction = algo_blend_v3(alpha, iv, st, frame_alphas)
    valid_streak = st.streak if st.last_dir == direction else 0
    factor = 1.0 + min(valid_streak / 6.0, 1.0) * 0.5
    return score * factor, direction


ALGOS: list[tuple[str, str, AlgoFn]] = [
    ("A", "A0_raw_abs",            algo_raw_abs),
    ("A", "A1_pow_1.5",            algo_alpha_pow_15),
    ("A", "A2_pow_2",              algo_alpha_pow_2),
    ("A", "A3_frame_rank",         algo_frame_rank),
    ("B", "B1_over_iv",            algo_alpha_over_iv),
    ("B", "B2_over_sqrt_iv",       algo_alpha_over_sqrt_iv),
    ("C", "C1_ema_short_6",        algo_ema_short_6),
    ("C", "C2_ema_long_12",        algo_ema_long_12),
    ("C", "C3_dema_diff",          algo_dema_diff),
    ("D", "D1_streak_mult",        algo_streak_mult),
    ("D", "D2_consistency_mult",   algo_consistency_mult),
    ("D", "D3_flip_penalty",       algo_flip_penalty),
    ("E", "E1_short_sharpe",       algo_short_sharpe),
    ("E", "E2_mean_signed_x_n",    algo_mean_signed_x_n),
    ("F", "F1_blend_v3",           algo_blend_v3),
    ("F", "F2_blend_v3_streak",    algo_blend_v3_streak),
]


# ---------- 回放 ----------

@dataclass
class RunResult:
    family: str
    label: str
    selected_rows: list[RankedCandidate]
    profits: list[CandidateProfit]
    avg_max: float
    avg_final: float
    hit10: float
    hit20: float
    pos: float
    avg_drawdown: float
    samples: int
    focus_counts: dict[str, int]


def replay(
    alpha_rows: list[dict[str, Any]],
    family: str,
    label: str,
    algo: AlgoFn,
    *,
    selected_per_frame: int,
    alpha_floor: float,
    option_paths: dict[str, dict[int, str]],
    horizon_mins: int,
    cooldown_mins: int,
    focus_symbols: list[str],
) -> RunResult:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    states: dict[str, SymState] = defaultdict(SymState)
    selected_rows: list[RankedCandidate] = []
    focus_counter: Counter = Counter()

    for ts in sorted(grouped.keys()):
        # 先全量推进 state, 再打分 (state 用本帧 alpha 更新)
        frame_alphas: list[float] = []
        rows_with_state: list[tuple[dict, SymState]] = []
        for row in grouped[ts]:
            sym = str(row["symbol"])
            alpha = _safe_float(row["alpha"])
            update_state(states[sym], alpha)
            frame_alphas.append(alpha)
            rows_with_state.append((row, states[sym]))

        frame_cands: list[tuple[float, RankedCandidate]] = []
        for row, st in rows_with_state:
            sym = str(row["symbol"])
            alpha = _safe_float(row["alpha"])
            iv = _safe_float(row.get("iv"), 0.5)
            price = _safe_float(row.get("price"))
            # 统一入选门槛: |raw alpha| >= floor (公平比较)
            if abs(alpha) < alpha_floor:
                continue
            sc, direction = algo(alpha, iv, st, frame_alphas)
            if sc <= 0:
                continue
            frame_cands.append((
                sc,
                RankedCandidate(
                    ts=ts,
                    datetime_ny=str(row["datetime_ny"]),
                    symbol=sym,
                    alpha=float(sc) * (1 if direction >= 0 else -1),
                    iv=iv, price=price,
                    roc_5m=0.0, snap_roc=0.0, macd_hist=0.0,
                    score=float(sc), is_priority_candidate=False,
                ),
            ))
        if not frame_cands:
            continue
        frame_cands.sort(key=lambda x: x[0], reverse=True)
        for _, cand in frame_cands[:selected_per_frame]:
            selected_rows.append(cand)
            if cand.symbol in focus_symbols:
                focus_counter[cand.symbol] += 1

    profits = analyze_candidate_profits(
        selected_rows, option_paths,
        horizon_mins=horizon_mins, cooldown_mins=cooldown_mins,
    )
    if profits:
        mx = [p.max_roi for p in profits]
        fn = [p.final_roi for p in profits]
        dd = [p.min_roi for p in profits]
        avg_max = sum(mx) / len(mx)
        avg_final = sum(fn) / len(fn)
        hit10 = sum(1 for x in mx if x >= 0.10) / len(mx)
        hit20 = sum(1 for x in mx if x >= 0.20) / len(mx)
        pos = sum(1 for x in fn if x > 0) / len(fn)
        avg_dd = sum(dd) / len(dd)
    else:
        avg_max = avg_final = hit10 = hit20 = pos = avg_dd = 0.0

    return RunResult(
        family=family, label=label, selected_rows=selected_rows, profits=profits,
        avg_max=avg_max, avg_final=avg_final, hit10=hit10, hit20=hit20,
        pos=pos, avg_drawdown=avg_dd, samples=len(profits),
        focus_counts={s: focus_counter.get(s, 0) for s in focus_symbols},
    )


# ---------- 输出 ----------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def composite(r: RunResult) -> float:
    """综合评分: avgMax + 0.5*avgFinal + 0.3*hit20 - 0.2*|avg_drawdown|"""
    return r.avg_max + 0.5 * r.avg_final + 0.3 * r.hit20 - 0.2 * abs(r.avg_drawdown)


def print_table(results: list[RunResult], focus_symbols: list[str], title: str) -> None:
    print(f"\n## {title}")
    header = (
        f"{'rank':>4} {'family':<3} {'label':<22} "
        f"{'comp':>6} {'avgMax':>7} {'avgFin':>7} {'hit10':>6} {'hit20':>6} "
        f"{'pos':>6} {'maxDD':>7} {'n':>4} "
        + " ".join(f"{s:>5}" for s in focus_symbols)
    )
    print(header)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>4} {r.family:<3} {r.label:<22} "
            f"{composite(r) * 100:>5.2f} "
            f"{_fmt_pct(r.avg_max):>7} {_fmt_pct(r.avg_final):>7} "
            f"{_fmt_pct(r.hit10):>6} {_fmt_pct(r.hit20):>6} "
            f"{_fmt_pct(r.pos):>6} {_fmt_pct(r.avg_drawdown):>7} "
            f"{r.samples:>4} "
            + " ".join(f"{r.focus_counts.get(s, 0):>5}" for s in focus_symbols)
        )


# ---------- 主 ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="所有 alpha 排序算法横向对比")
    p.add_argument("--date", required=True)
    p.add_argument("--focus-symbols", default="NVDA,MSFT,ADBE,COIN,MSTR,INTC")
    p.add_argument("--selected-per-frame", type=int, default=3)
    p.add_argument("--candidate-horizon-mins", type=int, default=30)
    p.add_argument("--candidate-cooldown-mins", type=int, default=15)
    p.add_argument("--alpha-floor", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    focus_symbols = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()]

    conn = psycopg2.connect(PG_DB_URL)
    try:
        alpha_rows = load_alpha_rows(conn, args.date)
        if not alpha_rows:
            print("当天没有 alpha_logs 数据.")
            return
        symbols_scope = sorted({str(r["symbol"]) for r in alpha_rows})
        min_ts = min(int(float(r["ts"])) for r in alpha_rows)
        max_ts = max(int(float(r["ts"])) for r in alpha_rows)
        option_paths = load_option_paths(
            conn, min_ts - 600, max_ts + args.candidate_horizon_mins * 60 + 600,
            focus_symbols + symbols_scope,
        )
    finally:
        conn.close()

    print(f"加载: alpha_rows={len(alpha_rows)} 条, symbols={len(symbols_scope)} | 算法数={len(ALGOS)}")
    print(f"参数: alpha_floor={args.alpha_floor} selected_per_frame={args.selected_per_frame}")
    print("综合评分 = avgMax + 0.5*avgFinal + 0.3*hit20 - 0.2*|maxDD|")

    common = dict(
        selected_per_frame=args.selected_per_frame,
        alpha_floor=args.alpha_floor,
        option_paths=option_paths,
        horizon_mins=args.candidate_horizon_mins,
        cooldown_mins=args.candidate_cooldown_mins,
        focus_symbols=focus_symbols,
    )

    results: list[RunResult] = []
    for family, label, fn in ALGOS:
        r = replay(alpha_rows, family, label, fn, **common)
        results.append(r)

    # 表 1: 按综合评分排序
    by_comp = sorted(results, key=composite, reverse=True)
    print_table(by_comp, focus_symbols, "全量横向对比 (按综合评分排序)")

    # 表 2: 按 avgMax 排序
    by_max = sorted(results, key=lambda r: r.avg_max, reverse=True)
    print_table(by_max, focus_symbols, "按 avgMax 排序")

    # 表 3: 按 hit10 排序
    by_hit = sorted(results, key=lambda r: r.hit10, reverse=True)
    print_table(by_hit, focus_symbols, "按 hit10 排序 (能跑到 10% 的概率)")

    print("\n>>> 综合最优:", by_comp[0].label)
    baseline = next(r for r in results if r.label == "A0_raw_abs")
    best = by_comp[0]
    if baseline.avg_max > 0:
        print(
            f"    vs 现网基线 A0_raw_abs: "
            f"avgMax {_fmt_pct(baseline.avg_max)} -> {_fmt_pct(best.avg_max)} "
            f"({(best.avg_max - baseline.avg_max) / max(0.001, abs(baseline.avg_max)) * 100:+.1f}%); "
            f"hit10 {_fmt_pct(baseline.hit10)} -> {_fmt_pct(best.hit10)}; "
            f"hit20 {_fmt_pct(baseline.hit20)} -> {_fmt_pct(best.hit20)}; "
            f"samples {baseline.samples} -> {best.samples}"
        )


if __name__ == "__main__":
    main()
