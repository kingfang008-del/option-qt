#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线验证：在 SignalEngine 已输出的 cross-sectional alpha 之上,
叠加 *方向持续性*(streak) 与 *时间维平滑*(EMA) 后, 是否能让
NVDA/MSFT/ADBE 这种长期强信号在排名里自然浮上来, 且 30 分钟理论
ROI 是否更优.

本脚本不依赖 ExecutionEngine 的 priority/iv/roc/macd 修补层, 完全围绕
alpha 自身做形状重塑, 用以验证“源头优化 alpha”这一方案的可行性.

公式变体:
  V0 raw                    : score = |alpha|
  V1 streak_mult            : score = |alpha| * streak_factor
  V2 ema_smooth             : score = |EMA_signed(alpha)|
  V3 blend                  : score = w_i * |alpha| + w_s * |EMA_signed(alpha)|
  V4 blend_x_streak         : score = V3 * streak_factor

streak_factor = 1 + min(streak / STREAK_NORM, 1.0) * STREAK_MAX_BOOST
streak        = 连续 |alpha| >= STREAK_FLOOR 且方向不变的分钟数

运行:
  python analyze_persistence_alpha_pg.py --date 2026-04-24 \
      --focus-symbols NVDA,MSFT,ADBE --selected-per-frame 3 \
      --candidate-horizon-mins 30 --candidate-cooldown-mins 15
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import psycopg2  # type: ignore

# 复用同目录脚本里已实现的加载/回放/打印函数
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
    print_candidate_profit_summary,
)

# config 常量
sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402


# ---------- persistence state ----------

@dataclass
class PersistenceState:
    streak: int = 0
    last_dir: int = 0
    ema_signed: float = 0.0
    seen: int = 0

    def update(self, alpha: float, *, ema_alpha: float, streak_floor: float) -> None:
        direction = 1 if alpha > 0 else (-1 if alpha < 0 else 0)
        # streak: 同方向 + |alpha| 达标
        if direction != 0 and abs(alpha) >= streak_floor and direction == self.last_dir:
            self.streak += 1
        elif direction != 0 and abs(alpha) >= streak_floor:
            self.streak = 1
            self.last_dir = direction
        else:
            self.streak = 0
            self.last_dir = direction or self.last_dir

        # 带方向 EMA: 直接对 signed alpha 做 EMA
        if self.seen == 0:
            self.ema_signed = float(alpha)
        else:
            self.ema_signed = float(ema_alpha * alpha + (1.0 - ema_alpha) * self.ema_signed)
        self.seen += 1


# ---------- 公式变体 ----------

@dataclass
class FormulaParams:
    streak_floor: float = 0.6
    streak_norm: float = 8.0
    streak_max_boost: float = 0.6
    smooth_span: int = 6
    blend_w_inst: float = 0.4
    blend_w_smooth: float = 0.6

    @property
    def ema_alpha(self) -> float:
        # 经典 EMA: alpha = 2 / (span + 1)
        span = max(2, int(self.smooth_span))
        return 2.0 / (span + 1.0)


def _streak_factor(streak: int, params: FormulaParams) -> float:
    if streak <= 0:
        return 1.0
    norm = max(1.0, float(params.streak_norm))
    return 1.0 + min(streak / norm, 1.0) * float(params.streak_max_boost)


ScoreFunc = Callable[[float, PersistenceState, FormulaParams], tuple[float, int]]


def score_v0_raw(alpha: float, st: PersistenceState, params: FormulaParams) -> tuple[float, int]:
    direction = 1 if alpha >= 0 else -1
    return abs(float(alpha)), direction


def score_v1_streak(alpha: float, st: PersistenceState, params: FormulaParams) -> tuple[float, int]:
    direction = 1 if alpha >= 0 else -1
    factor = _streak_factor(st.streak if st.last_dir == direction else 0, params)
    return abs(float(alpha)) * factor, direction


def score_v2_ema(alpha: float, st: PersistenceState, params: FormulaParams) -> tuple[float, int]:
    # 用平滑后的方向作为最终方向
    smooth = float(st.ema_signed)
    direction = 1 if smooth >= 0 else -1
    return abs(smooth), direction


def score_v3_blend(alpha: float, st: PersistenceState, params: FormulaParams) -> tuple[float, int]:
    smooth = float(st.ema_signed)
    # 当 raw 与 smooth 反向时, 视为信号正在转向, 用 min 做惩罚
    if (alpha >= 0) != (smooth >= 0):
        agreed = 0.0
    else:
        agreed = 1.0
    base = params.blend_w_inst * abs(float(alpha)) + params.blend_w_smooth * abs(smooth)
    direction = 1 if smooth >= 0 else -1
    return base * (0.4 + 0.6 * agreed), direction  # 反向时仍保留 40% 强度但降权


def score_v4_blend_streak(alpha: float, st: PersistenceState, params: FormulaParams) -> tuple[float, int]:
    base, direction = score_v3_blend(alpha, st, params)
    factor = _streak_factor(st.streak if st.last_dir == direction else 0, params)
    return base * factor, direction


FORMULAS: dict[str, ScoreFunc] = {
    "V0_raw":              score_v0_raw,
    "V1_streak":           score_v1_streak,
    "V2_ema_smooth":       score_v2_ema,
    "V3_blend":            score_v3_blend,
    "V4_blend_x_streak":   score_v4_blend_streak,
}


# ---------- 排序回放 ----------

@dataclass
class FormulaResult:
    name: str
    selected_rows: list[RankedCandidate]
    top1: Counter
    top3: Counter
    selected_counter: Counter
    frames: int
    avg_streak_at_select: float


def replay_formula(
    alpha_rows: list[dict[str, Any]],
    formula_name: str,
    score_func: ScoreFunc,
    params: FormulaParams,
    *,
    selected_per_frame: int,
    alpha_floor: float,
) -> FormulaResult:
    """对所有分钟帧使用同一个公式做横截面排序, 返回 selected 候选列表."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    states: dict[str, PersistenceState] = defaultdict(PersistenceState)
    top1: Counter = Counter()
    top3: Counter = Counter()
    selected_counter: Counter = Counter()
    selected_rows: list[RankedCandidate] = []
    frames = 0
    streak_acc: list[int] = []

    for ts in sorted(grouped.keys()):
        frame_candidates: list[tuple[float, int, RankedCandidate, int]] = []
        dt_ny = ""
        for row in grouped[ts]:
            sym = str(row["symbol"])
            alpha = _safe_float(row["alpha"])
            iv = _safe_float(row.get("iv"), 0.5)
            price = _safe_float(row.get("price"))
            dt_ny = str(row["datetime_ny"])

            # 1. 先用本分钟的原始 alpha 推进 state (注意: state 推进在打分之前)
            st = states[sym]
            st.update(alpha, ema_alpha=params.ema_alpha, streak_floor=params.streak_floor)

            # 2. 入选门槛 (沿用 ALPHA_ENTRY_THRESHOLD, 用 raw |alpha| 判定, 保证可比)
            if abs(alpha) < alpha_floor:
                continue

            # 3. 用所选公式打分
            score, direction = score_func(alpha, st, params)
            if score <= 0:
                continue

            cand = RankedCandidate(
                ts=ts,
                datetime_ny=dt_ny,
                symbol=sym,
                # alpha 字段保留 *公式建议的方向*, 用正负号承载, 配合 analyze_candidate_profits
                # 内部 (alpha >= 0 -> CALL, < 0 -> PUT) 的解析约定
                alpha=float(score) * (1 if direction >= 0 else -1),
                iv=iv,
                price=price,
                roc_5m=0.0,
                snap_roc=0.0,
                macd_hist=0.0,
                score=float(score),
                is_priority_candidate=False,
            )
            frame_candidates.append((float(score), st.streak, cand, direction))

        if not frame_candidates:
            continue
        frames += 1
        frame_candidates.sort(key=lambda x: x[0], reverse=True)

        top1[frame_candidates[0][2].symbol] += 1
        for _, _, cand, _ in frame_candidates[:3]:
            top3[cand.symbol] += 1

        for score, streak, cand, _ in frame_candidates[:selected_per_frame]:
            selected_counter[cand.symbol] += 1
            selected_rows.append(cand)
            streak_acc.append(int(streak))

    avg_streak = (sum(streak_acc) / len(streak_acc)) if streak_acc else 0.0
    return FormulaResult(
        name=formula_name,
        selected_rows=selected_rows,
        top1=top1,
        top3=top3,
        selected_counter=selected_counter,
        frames=frames,
        avg_streak_at_select=avg_streak,
    )


# ---------- 输出 ----------

def _avg(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def print_compact_compare(results: list[tuple[FormulaResult, list[CandidateProfit]]],
                          focus_symbols: list[str]) -> None:
    """同时把 *排名覆盖* 与 *理论收益* 摆在一张表里, 便于横向对比."""
    print("\n## 公式 A/B 对比汇总 (selected_per_frame 一致)")
    header = (
        f"{'formula':<22} {'frames':>6} "
        f"{'avgMax':>7} {'avgFinal':>8} {'hit10':>6} {'hit20':>6} {'pos':>6} "
        f"{'samples':>7} {'avgStreak':>10}"
    )
    print(header)
    for res, profits in results:
        if profits:
            mx = [p.max_roi for p in profits]
            fn = [p.final_roi for p in profits]
            avg_mx = _avg(mx)
            avg_fn = _avg(fn)
            hit10 = sum(1 for x in mx if x >= 0.10) / len(mx)
            hit20 = sum(1 for x in mx if x >= 0.20) / len(mx)
            pos = sum(1 for x in fn if x > 0) / len(fn)
        else:
            avg_mx = avg_fn = hit10 = hit20 = pos = 0.0
        print(
            f"{res.name:<22} {res.frames:>6} "
            f"{_fmt_pct(avg_mx):>7} {_fmt_pct(avg_fn):>8} "
            f"{_fmt_pct(hit10):>6} {_fmt_pct(hit20):>6} {_fmt_pct(pos):>6} "
            f"{len(profits):>7} {res.avg_streak_at_select:>10.2f}"
        )

    if not focus_symbols:
        return

    print("\n## 关注标的 (selected 次数 / top1 次数 / top3 次数)")
    sym_header = f"{'formula':<22} " + " ".join(
        f"{s:>20}" for s in focus_symbols
    )
    print(sym_header)
    for res, _ in results:
        cells = []
        for sym in focus_symbols:
            cells.append(
                f"{res.selected_counter.get(sym, 0):>6}/"
                f"{res.top1.get(sym, 0):>5}/"
                f"{res.top3.get(sym, 0):>5}"
            )
        joined = " ".join(f"{c:>20}" for c in cells)
        print(f"{res.name:<22} {joined}")

    print("\n## 关注标的: 各公式下的平均理论 30m ROI")
    sym_header2 = f"{'formula':<22} " + " ".join(
        f"{s:>26}" for s in focus_symbols
    )
    print(sym_header2)
    for res, profits in results:
        cells = []
        for sym in focus_symbols:
            sym_rows = [p for p in profits if p.symbol == sym]
            if sym_rows:
                avg_mx = _avg([p.max_roi for p in sym_rows])
                avg_fn = _avg([p.final_roi for p in sym_rows])
                cells.append(f"n={len(sym_rows):<3} max={_fmt_pct(avg_mx)} fin={_fmt_pct(avg_fn)}")
            else:
                cells.append("n/a")
        joined = " ".join(f"{c:>26}" for c in cells)
        print(f"{res.name:<22} {joined}")


# ---------- 主流程 ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线验证: 在源头 alpha 上叠加持续性, 看 NVDA/MSFT/ADBE 是否更易上榜")
    p.add_argument("--date", required=True, help="交易日, 格式 2026-04-24")
    p.add_argument("--focus-symbols", default="NVDA,MSFT,ADBE", help="关注标的, 逗号分隔")
    p.add_argument("--selected-per-frame", type=int, default=3, help="每分钟理论选出的候选数")
    p.add_argument("--candidate-horizon-mins", type=int, default=30, help="理论候选向后观察的分钟数")
    p.add_argument("--candidate-cooldown-mins", type=int, default=15, help="同一标的同方向理论候选去重冷却分钟数")
    p.add_argument("--alpha-floor", type=float, default=0.6, help="入选门槛 (raw |alpha| >= floor)")
    p.add_argument("--streak-floor", type=float, default=0.6, help="streak 计数的 |alpha| 阈值")
    p.add_argument("--streak-norm", type=float, default=8.0, help="streak 达到满激励所需分钟数")
    p.add_argument("--streak-max-boost", type=float, default=0.6, help="streak 满激励时的最大乘子增量")
    p.add_argument("--smooth-span", type=int, default=6, help="EMA 平滑窗口 (分钟)")
    p.add_argument("--blend-w-inst", type=float, default=0.4, help="blend 公式中 raw|alpha| 的权重")
    p.add_argument("--blend-w-smooth", type=float, default=0.6, help="blend 公式中 |EMA(alpha)| 的权重")
    p.add_argument("--limit", type=int, default=10, help="终端输出明细条数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    focus_symbols = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()]
    params = FormulaParams(
        streak_floor=args.streak_floor,
        streak_norm=args.streak_norm,
        streak_max_boost=args.streak_max_boost,
        smooth_span=args.smooth_span,
        blend_w_inst=args.blend_w_inst,
        blend_w_smooth=args.blend_w_smooth,
    )

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
            conn,
            min_ts - 600,
            max_ts + args.candidate_horizon_mins * 60 + 600,
            focus_symbols + symbols_scope,
        )
    finally:
        conn.close()

    print(
        f"加载: alpha_rows={len(alpha_rows)} 条, symbols={len(symbols_scope)}, "
        f"frames≈{len({int(float(r['ts'])) for r in alpha_rows})}"
    )
    print(
        f"参数: streak_floor={params.streak_floor} streak_norm={params.streak_norm} "
        f"streak_max_boost={params.streak_max_boost} smooth_span={params.smooth_span} "
        f"blend=({params.blend_w_inst},{params.blend_w_smooth})"
    )

    results: list[tuple[FormulaResult, list[CandidateProfit]]] = []
    for name, func in FORMULAS.items():
        res = replay_formula(
            alpha_rows,
            name,
            func,
            params,
            selected_per_frame=args.selected_per_frame,
            alpha_floor=args.alpha_floor,
        )
        profits = analyze_candidate_profits(
            res.selected_rows,
            option_paths,
            horizon_mins=args.candidate_horizon_mins,
            cooldown_mins=args.candidate_cooldown_mins,
        )
        results.append((res, profits))

    # 1. 全局对比表
    print_compact_compare(results, focus_symbols)

    # 2. 每个公式的明细 (沿用现有的统一打印格式)
    for res, profits in results:
        print_candidate_profit_summary(
            profits,
            focus_symbols,
            args.limit,
            title=f"[{res.name}] 全市场理论候选 30m 收益 (selected={len(res.selected_rows)})",
        )


if __name__ == "__main__":
    main()
