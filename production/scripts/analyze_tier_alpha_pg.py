#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线验证: 在 SignalEngine 已输出的 *全市场横截面 z (alpha)* 之上,
按市值/波动分层, 重新做 *组内横截面 z*, 看能否解决 MSFT/ADBE 这类
"alpha 数值天然不及阈值" 的问题, 让大票自然浮上来.

数学等价说明:
  alpha_logs.alpha = (raw - μ_全) / σ_全     (全市场 z)
  对它再做 tier 内 z 等价于:
      (alpha - μ_tier_alpha) / σ_tier_alpha
  ==  ((raw - μ_tier_raw) / σ_全) / (σ_tier_raw / σ_全)
  ==   (raw - μ_tier_raw) / σ_tier_raw       (raw 上的 tier 内 z)
  (clip 之外完全等价)

公式变体:
  T0_global              : 基线, score = |alpha|             (现网当前公式)
  T1_pure_tier           : score = |z_tier|
  T2_tier_with_prior     : score = |z_tier| * tier_prior
  T3_blend               : score = w_g * |alpha| + w_t * |z_tier|
  T4_blend_with_prior    : score = T3 * tier_prior

tier_prior 默认 A 组 1.10, B 组 1.00 (轻微偏好大票).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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
    print_candidate_profit_summary,
)

sys.path.insert(0, str(SCRIPT_DIR.parent / "baseline"))
from config import PG_DB_URL  # noqa: E402

# ---- tier 定义 ----
# A 组: MAG7 + 同等级大盘科技 (走势相对平滑, 波动率偏低, 期权 ROI 偏小)
TIER_A: set[str] = {
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
    "AVGO", "ORCL", "ADBE",
}
# B 组: 中小盘 + 高波动 (容易瞬时偏离均值, 拉高 std, 把 A 组 z 拍扁)
TIER_B: set[str] = {
    "MSTR", "COIN", "CRWV", "SMCI", "INTC", "DELL", "HOOD", "PLTR",
    "MU", "NKE", "XOM", "WMT", "NFLX", "AMD", "GS",
}


def get_tier(symbol: str) -> str:
    if symbol in TIER_A:
        return "A"
    if symbol in TIER_B:
        return "B"
    return "U"  # 未分类


# ---- 公式 ----

@dataclass
class Params:
    blend_w_global: float = 0.4
    blend_w_tier: float = 0.6
    tier_a_prior: float = 1.10
    tier_b_prior: float = 1.00
    tier_u_prior: float = 1.00
    min_tier_n: int = 3   # tier 内有效样本至少 N 个才算 z, 否则退回 alpha


@dataclass
class CandRow:
    sym: str
    alpha_global: float        # alpha_logs 中的 alpha (即全市场 z)
    z_tier: float              # 组内 z (基于本帧 tier 内的样本)
    tier: str
    iv: float
    price: float
    ts: int
    datetime_ny: str


ScoreFn = Callable[[CandRow, Params], tuple[float, int]]


def _direction(signed: float) -> int:
    return 1 if signed >= 0 else -1


def score_t0_global(c: CandRow, p: Params) -> tuple[float, int]:
    return abs(c.alpha_global), _direction(c.alpha_global)


def score_t1_pure_tier(c: CandRow, p: Params) -> tuple[float, int]:
    return abs(c.z_tier), _direction(c.z_tier)


def score_t2_tier_prior(c: CandRow, p: Params) -> tuple[float, int]:
    prior = {"A": p.tier_a_prior, "B": p.tier_b_prior}.get(c.tier, p.tier_u_prior)
    return abs(c.z_tier) * prior, _direction(c.z_tier)


def score_t3_blend(c: CandRow, p: Params) -> tuple[float, int]:
    score = p.blend_w_global * abs(c.alpha_global) + p.blend_w_tier * abs(c.z_tier)
    # 方向以两者中较强的为准 (一致就用一致, 不一致用 |z_tier| 的方向)
    if (c.alpha_global >= 0) == (c.z_tier >= 0):
        direction = _direction(c.alpha_global)
    else:
        direction = _direction(c.z_tier)
    return score, direction


def score_t4_blend_prior(c: CandRow, p: Params) -> tuple[float, int]:
    base, direction = score_t3_blend(c, p)
    prior = {"A": p.tier_a_prior, "B": p.tier_b_prior}.get(c.tier, p.tier_u_prior)
    return base * prior, direction


FORMULAS: dict[str, ScoreFn] = {
    "T0_global":           score_t0_global,
    "T1_pure_tier":        score_t1_pure_tier,
    "T2_tier_prior":       score_t2_tier_prior,
    "T3_blend":            score_t3_blend,
    "T4_blend_prior":      score_t4_blend_prior,
}


# ---- 回放 ----

@dataclass
class FormulaResult:
    name: str
    selected_rows: list[RankedCandidate]
    top1: Counter
    top3: Counter
    selected_counter: Counter
    frames: int
    tier_share_top1: dict[str, int]
    tier_share_selected: dict[str, int]


def _frame_tier_z(rows: list[dict[str, Any]], min_tier_n: int) -> dict[str, float]:
    """对一帧内的每个 symbol, 计算它在自己 tier 内的 z 值."""
    by_tier: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        sym = str(row["symbol"])
        tier = get_tier(sym)
        if tier == "U":
            continue  # 未分类的不参与 tier z
        alpha = _safe_float(row["alpha"])
        by_tier[tier].append((sym, alpha))

    z_by_sym: dict[str, float] = {}
    for tier, vals in by_tier.items():
        if len(vals) < min_tier_n:
            # 样本太少, 退回原 alpha (不做 tier z)
            for sym, alpha in vals:
                z_by_sym[sym] = alpha
            continue
        alphas = [v for _, v in vals]
        n = len(alphas)
        mean = sum(alphas) / n
        var = sum((a - mean) ** 2 for a in alphas) / max(1, n - 1)
        std = (var ** 0.5) if var > 1e-12 else 0.0
        for sym, alpha in vals:
            if std > 1e-9:
                z_by_sym[sym] = (alpha - mean) / std
            else:
                z_by_sym[sym] = alpha
    return z_by_sym


def replay_formula(
    alpha_rows: list[dict[str, Any]],
    name: str,
    func: ScoreFn,
    params: Params,
    *,
    selected_per_frame: int,
    alpha_floor: float,
    tier_floor: float,
    use_tier_floor: bool,
) -> FormulaResult:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    top1: Counter = Counter()
    top3: Counter = Counter()
    selected_counter: Counter = Counter()
    selected_rows: list[RankedCandidate] = []
    tier_share_top1: dict[str, int] = defaultdict(int)
    tier_share_selected: dict[str, int] = defaultdict(int)
    frames = 0

    for ts in sorted(grouped.keys()):
        rows = grouped[ts]
        z_by_sym = _frame_tier_z(rows, params.min_tier_n)
        if not z_by_sym:
            continue

        candidates: list[tuple[float, CandRow, int]] = []
        for row in rows:
            sym = str(row["symbol"])
            tier = get_tier(sym)
            if tier == "U":
                continue
            alpha = _safe_float(row["alpha"])
            z_tier_val = z_by_sym.get(sym, alpha)
            # 入选门槛: T0 用全市场 z, T1~T4 用 tier 内 z (这才是 tier 方案的核心 - 让 MSFT 这种
            # 全市场 z 不够但组内 z 很高的标的也能进候选池)
            if use_tier_floor:
                if abs(z_tier_val) < tier_floor:
                    continue
            else:
                if abs(alpha) < alpha_floor:
                    continue
            cand = CandRow(
                sym=sym,
                alpha_global=alpha,
                z_tier=z_tier_val,
                tier=tier,
                iv=_safe_float(row.get("iv"), 0.5),
                price=_safe_float(row.get("price")),
                ts=ts,
                datetime_ny=str(row["datetime_ny"]),
            )
            score, direction = func(cand, params)
            if score <= 0:
                continue
            candidates.append((float(score), cand, direction))

        if not candidates:
            continue
        frames += 1
        candidates.sort(key=lambda x: x[0], reverse=True)
        top1[candidates[0][1].sym] += 1
        tier_share_top1[candidates[0][1].tier] += 1
        for _, cand, _ in candidates[:3]:
            top3[cand.sym] += 1

        for score, cand, direction in candidates[:selected_per_frame]:
            selected_counter[cand.sym] += 1
            tier_share_selected[cand.tier] += 1
            selected_rows.append(
                RankedCandidate(
                    ts=cand.ts,
                    datetime_ny=cand.datetime_ny,
                    symbol=cand.sym,
                    alpha=float(score) * (1 if direction >= 0 else -1),
                    iv=cand.iv,
                    price=cand.price,
                    roc_5m=0.0,
                    snap_roc=0.0,
                    macd_hist=0.0,
                    score=float(score),
                    is_priority_candidate=False,
                )
            )

    return FormulaResult(
        name=name,
        selected_rows=selected_rows,
        top1=top1,
        top3=top3,
        selected_counter=selected_counter,
        frames=frames,
        tier_share_top1=dict(tier_share_top1),
        tier_share_selected=dict(tier_share_selected),
    )


# ---- 输出 ----

def _avg(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def print_compare(results: list[tuple[FormulaResult, list[CandidateProfit]]],
                  focus_symbols: list[str]) -> None:
    print("\n## 公式 A/B 对比 (selected_per_frame 一致)")
    print(
        f"{'formula':<22} {'frames':>6} {'avgMax':>7} {'avgFinal':>8} "
        f"{'hit10':>6} {'hit20':>6} {'pos':>6} {'samples':>7} "
        f"{'tierA_top1%':>11} {'tierA_sel%':>10}"
    )
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

        ttop = res.tier_share_top1
        tsel = res.tier_share_selected
        sum_top = sum(ttop.values()) or 1
        sum_sel = sum(tsel.values()) or 1
        a_top_pct = ttop.get("A", 0) / sum_top
        a_sel_pct = tsel.get("A", 0) / sum_sel

        print(
            f"{res.name:<22} {res.frames:>6} "
            f"{_fmt_pct(avg_mx):>7} {_fmt_pct(avg_fn):>8} "
            f"{_fmt_pct(hit10):>6} {_fmt_pct(hit20):>6} {_fmt_pct(pos):>6} "
            f"{len(profits):>7} "
            f"{_fmt_pct(a_top_pct):>11} {_fmt_pct(a_sel_pct):>10}"
        )

    if not focus_symbols:
        return

    print("\n## 关注标的覆盖 (selected / top1 / top3)")
    print(f"{'formula':<22} " + " ".join(f"{s:>20}" for s in focus_symbols))
    for res, _ in results:
        cells = []
        for sym in focus_symbols:
            cells.append(
                f"{res.selected_counter.get(sym, 0):>6}/"
                f"{res.top1.get(sym, 0):>5}/"
                f"{res.top3.get(sym, 0):>5}"
            )
        print(f"{res.name:<22} " + " ".join(f"{c:>20}" for c in cells))

    print("\n## 关注标的: 平均理论 30m ROI")
    print(f"{'formula':<22} " + " ".join(f"{s:>26}" for s in focus_symbols))
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
        print(f"{res.name:<22} " + " ".join(f"{c:>26}" for c in cells))


# ---- 主流程 ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线验证: 分层横截面 z 是否能让 MSFT/ADBE 浮上来")
    p.add_argument("--date", required=True, help="交易日, 格式 2026-04-24")
    p.add_argument("--focus-symbols", default="NVDA,MSFT,ADBE,COIN,MSTR,INTC", help="关注标的, 逗号分隔")
    p.add_argument("--selected-per-frame", type=int, default=3)
    p.add_argument("--candidate-horizon-mins", type=int, default=30)
    p.add_argument("--candidate-cooldown-mins", type=int, default=15)
    p.add_argument("--alpha-floor", type=float, default=0.6, help="T0 公式的入选门槛 |raw alpha| >= floor")
    p.add_argument("--tier-floor", type=float, default=0.8,
                   help="T1~T4 公式的入选门槛 |z_tier| >= floor (因为 tier z 标准差通常更小, 这里给得更紧)")
    p.add_argument("--blend-w-global", type=float, default=0.4)
    p.add_argument("--blend-w-tier", type=float, default=0.6)
    p.add_argument("--tier-a-prior", type=float, default=1.10)
    p.add_argument("--tier-b-prior", type=float, default=1.00)
    p.add_argument("--min-tier-n", type=int, default=3)
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    focus_symbols = [s.strip().upper() for s in args.focus_symbols.split(",") if s.strip()]
    params = Params(
        blend_w_global=args.blend_w_global,
        blend_w_tier=args.blend_w_tier,
        tier_a_prior=args.tier_a_prior,
        tier_b_prior=args.tier_b_prior,
        min_tier_n=args.min_tier_n,
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

    a_in_scope = sorted([s for s in symbols_scope if s in TIER_A])
    b_in_scope = sorted([s for s in symbols_scope if s in TIER_B])
    u_in_scope = sorted([s for s in symbols_scope if s not in TIER_A and s not in TIER_B])

    print(f"加载: alpha_rows={len(alpha_rows)} 条, symbols={len(symbols_scope)}")
    print(f"  tier A ({len(a_in_scope)}): {a_in_scope}")
    print(f"  tier B ({len(b_in_scope)}): {b_in_scope}")
    if u_in_scope:
        print(f"  tier U 未分类 ({len(u_in_scope)}): {u_in_scope}  -> 已跳过")
    print(
        f"参数: alpha_floor={args.alpha_floor} "
        f"blend=({params.blend_w_global},{params.blend_w_tier}) "
        f"prior=(A={params.tier_a_prior}, B={params.tier_b_prior}) min_tier_n={params.min_tier_n}"
    )

    results: list[tuple[FormulaResult, list[CandidateProfit]]] = []
    for name, func in FORMULAS.items():
        use_tier_floor = name != "T0_global"
        res = replay_formula(
            alpha_rows,
            name,
            func,
            params,
            selected_per_frame=args.selected_per_frame,
            alpha_floor=args.alpha_floor,
            tier_floor=args.tier_floor,
            use_tier_floor=use_tier_floor,
        )
        profits = analyze_candidate_profits(
            res.selected_rows,
            option_paths,
            horizon_mins=args.candidate_horizon_mins,
            cooldown_mins=args.candidate_cooldown_mins,
        )
        results.append((res, profits))

    print_compare(results, focus_symbols)

    for res, profits in results:
        print_candidate_profit_summary(
            profits,
            focus_symbols,
            args.limit,
            title=f"[{res.name}] 全市场理论候选 30m 收益 (selected={len(res.selected_rows)})",
        )


if __name__ == "__main__":
    main()
