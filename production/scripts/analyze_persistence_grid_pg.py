#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3_blend 公式的系统化网格搜索, 并验证两个新因子:
  * alpha_accel  : alpha 同方向加速时给加成
  * stock_consist: alpha 与正股 5m ROC 方向一致时给加成

流程:
  Stage A: 扫 blend(w_inst, smooth_span) - 25 组 (无 streak, 无新因子)
  Stage B: 在 A 的最优 blend 上扫 streak(max_boost, norm) - 9 组
  Stage C: 在 B 的最优配置上, 分别加 alpha_accel / stock_consist / 两者
           - 4 组对照
  最后输出 Pareto 最优集, 给出建议上线参数.

不依赖 ExecutionEngine, 只用 alpha_logs + option_snapshots_1m.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    streak: int = 0
    last_dir: int = 0
    ema_signed: float = 0.0
    seen: int = 0
    prev_alpha: float = 0.0
    prices: deque = field(default_factory=lambda: deque(maxlen=6))
    prev_price: float = 0.0


def _direction(v: float) -> int:
    return 1 if v >= 0 else -1


def update_state(st: SymState, alpha: float, price: float, *, ema_alpha: float, streak_floor: float) -> tuple[float, float]:
    """推进状态, 返回 (stock_roc_5m, alpha_change_signed)."""
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
        st.ema_signed = float(alpha)
    else:
        st.ema_signed = float(ema_alpha * alpha + (1.0 - ema_alpha) * st.ema_signed)

    alpha_change = float(alpha - st.prev_alpha) if st.seen > 0 else 0.0
    st.prev_alpha = float(alpha)
    st.seen += 1

    # 5m ROC (基于分钟价格 5 个间隔)
    if price > 0:
        st.prices.append(price)
    stock_roc = 0.0
    if len(st.prices) >= 6 and st.prices[0] > 0:
        stock_roc = (price - st.prices[0]) / st.prices[0]
    st.prev_price = price
    return stock_roc, alpha_change


# ---------- 公式参数 ----------

@dataclass
class Cfg:
    blend_w_inst: float = 0.4
    blend_w_smooth: float = 0.6
    smooth_span: int = 6
    # streak
    streak_floor: float = 0.6
    streak_norm: float = 8.0
    streak_max_boost: float = 0.0  # 0 = 不启用 streak
    # alpha 加速度
    accel_scale: float = 0.0       # 0 = 不启用
    accel_max: float = 0.4
    # alpha-stock 一致性
    consist_boost: float = 0.0     # 0 = 不启用
    # 一致性折扣 (raw 与 smooth 反向时降权)
    disagree_keep: float = 0.4

    @property
    def ema_alpha(self) -> float:
        span = max(2, int(self.smooth_span))
        return 2.0 / (span + 1.0)


def score(alpha: float, st: SymState, cfg: Cfg, stock_roc: float, alpha_change: float) -> tuple[float, int]:
    smooth = float(st.ema_signed)
    # 方向: 用平滑后的方向作为最终方向
    direction = _direction(smooth)

    # 一致性: raw 与 smooth 反向时降权
    same_dir = (alpha >= 0) == (smooth >= 0)
    agree_factor = 1.0 if same_dir else float(cfg.disagree_keep)

    base = (cfg.blend_w_inst * abs(alpha) + cfg.blend_w_smooth * abs(smooth)) * agree_factor

    # streak factor
    factor = 1.0
    if cfg.streak_max_boost > 0:
        valid_streak = st.streak if st.last_dir == direction else 0
        norm = max(1.0, float(cfg.streak_norm))
        factor *= 1.0 + min(valid_streak / norm, 1.0) * float(cfg.streak_max_boost)

    # alpha 加速度: 同方向且 |alpha| 在变大才加成
    if cfg.accel_scale > 0 and st.seen > 1:
        signed_change = alpha_change * direction
        if signed_change > 0:
            factor *= 1.0 + min(signed_change * float(cfg.accel_scale), float(cfg.accel_max))

    # alpha-stock 一致性: alpha 方向与正股 5m 方向一致时加成
    if cfg.consist_boost > 0 and abs(stock_roc) > 1e-9:
        if (stock_roc >= 0) == (direction >= 0):
            factor *= 1.0 + float(cfg.consist_boost)

    return base * factor, direction


# ---------- 回放 ----------

@dataclass
class RunResult:
    label: str
    cfg: Cfg
    selected_rows: list[RankedCandidate]
    profits: list[CandidateProfit]
    avg_max: float
    avg_final: float
    hit10: float
    hit20: float
    pos: float
    samples: int
    focus_counts: dict[str, int]


def replay(
    alpha_rows: list[dict[str, Any]],
    cfg: Cfg,
    *,
    selected_per_frame: int,
    alpha_floor: float,
    option_paths: dict[str, dict[int, str]],
    horizon_mins: int,
    cooldown_mins: int,
    focus_symbols: list[str],
    label: str,
) -> RunResult:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in alpha_rows:
        grouped[int(float(row["ts"]))].append(row)

    states: dict[str, SymState] = defaultdict(SymState)
    selected_rows: list[RankedCandidate] = []
    focus_counter: Counter = Counter()

    for ts in sorted(grouped.keys()):
        frame_cands: list[tuple[float, RankedCandidate]] = []
        for row in grouped[ts]:
            sym = str(row["symbol"])
            alpha = _safe_float(row["alpha"])
            iv = _safe_float(row.get("iv"), 0.5)
            price = _safe_float(row.get("price"))
            st = states[sym]
            stock_roc, alpha_change = update_state(
                st, alpha, price,
                ema_alpha=cfg.ema_alpha,
                streak_floor=cfg.streak_floor,
            )
            if abs(alpha) < alpha_floor:
                continue
            sc, direction = score(alpha, st, cfg, stock_roc, alpha_change)
            if sc <= 0:
                continue
            frame_cands.append((
                sc,
                RankedCandidate(
                    ts=ts,
                    datetime_ny=str(row["datetime_ny"]),
                    symbol=sym,
                    alpha=float(sc) * (1 if direction >= 0 else -1),
                    iv=iv,
                    price=price,
                    roc_5m=0.0,
                    snap_roc=0.0,
                    macd_hist=0.0,
                    score=float(sc),
                    is_priority_candidate=False,
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
        avg_max = sum(mx) / len(mx)
        avg_final = sum(fn) / len(fn)
        hit10 = sum(1 for x in mx if x >= 0.10) / len(mx)
        hit20 = sum(1 for x in mx if x >= 0.20) / len(mx)
        pos = sum(1 for x in fn if x > 0) / len(fn)
    else:
        avg_max = avg_final = hit10 = hit20 = pos = 0.0

    return RunResult(
        label=label, cfg=cfg, selected_rows=selected_rows, profits=profits,
        avg_max=avg_max, avg_final=avg_final, hit10=hit10, hit20=hit20,
        pos=pos, samples=len(profits),
        focus_counts={s: focus_counter.get(s, 0) for s in focus_symbols},
    )


# ---------- 输出 ----------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_result_table(results: list[RunResult], focus_symbols: list[str], title: str) -> None:
    print(f"\n## {title}")
    print(
        f"{'label':<28} {'avgMax':>7} {'avgFin':>7} {'hit10':>6} {'hit20':>6} {'pos':>6} {'n':>5} "
        + " ".join(f"{s:>6}" for s in focus_symbols)
    )
    for r in results:
        print(
            f"{r.label:<28} "
            f"{_fmt_pct(r.avg_max):>7} {_fmt_pct(r.avg_final):>7} "
            f"{_fmt_pct(r.hit10):>6} {_fmt_pct(r.hit20):>6} {_fmt_pct(r.pos):>6} "
            f"{r.samples:>5} "
            + " ".join(f"{r.focus_counts.get(s, 0):>6}" for s in focus_symbols)
        )


def pareto(results: list[RunResult], keys: tuple[str, str] = ("avg_max", "avg_final")) -> list[RunResult]:
    """简单 2D Pareto: 在 (avg_max, avg_final) 上, 没有其它点同时 >= 自己的就是 Pareto 最优."""
    out = []
    for r in results:
        v_a = getattr(r, keys[0])
        v_b = getattr(r, keys[1])
        dominated = False
        for o in results:
            if o is r:
                continue
            o_a = getattr(o, keys[0])
            o_b = getattr(o, keys[1])
            if o_a >= v_a and o_b >= v_b and (o_a > v_a or o_b > v_b):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


# ---------- 主流程 ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3_blend 网格搜索 + 新因子验证")
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

    print(f"加载: alpha_rows={len(alpha_rows)} 条, symbols={len(symbols_scope)}")

    common_kw = dict(
        selected_per_frame=args.selected_per_frame,
        alpha_floor=args.alpha_floor,
        option_paths=option_paths,
        horizon_mins=args.candidate_horizon_mins,
        cooldown_mins=args.candidate_cooldown_mins,
        focus_symbols=focus_symbols,
    )

    # ----- 基线: V0_raw (纯 |alpha|) -----
    baseline = replay(
        alpha_rows,
        Cfg(blend_w_inst=1.0, blend_w_smooth=0.0, smooth_span=2,
            streak_max_boost=0.0, accel_scale=0.0, consist_boost=0.0,
            disagree_keep=1.0),
        label="V0_raw_baseline", **common_kw,
    )
    print_result_table([baseline], focus_symbols, "基线 V0_raw")

    # ===== Stage A: blend(w_inst) x smooth_span =====
    stage_a: list[RunResult] = []
    for w_inst in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for span in [4, 6, 8, 10, 12]:
            cfg = Cfg(blend_w_inst=w_inst, blend_w_smooth=1.0 - w_inst, smooth_span=span,
                      streak_max_boost=0.0, accel_scale=0.0, consist_boost=0.0)
            r = replay(alpha_rows, cfg, label=f"A_blend_w{w_inst}_s{span}", **common_kw)
            stage_a.append(r)

    stage_a_sorted = sorted(stage_a, key=lambda r: (r.avg_max + r.avg_final), reverse=True)
    print_result_table(stage_a_sorted[:8], focus_symbols, "Stage A: blend 网格 Top-8 (按 avgMax+avgFinal)")
    pareto_a = pareto(stage_a)
    print_result_table(
        sorted(pareto_a, key=lambda r: r.avg_max, reverse=True),
        focus_symbols, "Stage A: Pareto 前沿 (avgMax/avgFinal)",
    )

    best_a = stage_a_sorted[0]
    print(
        f"\n>>> Stage A 最优: {best_a.label} "
        f"(w_inst={best_a.cfg.blend_w_inst}, span={best_a.cfg.smooth_span})"
    )

    # ===== Stage B: 在 A 最优上扫 streak =====
    stage_b: list[RunResult] = [best_a]  # 含 no-streak 基准
    for max_boost in [0.3, 0.5, 0.7]:
        for norm in [6, 8, 10]:
            cfg = Cfg(
                blend_w_inst=best_a.cfg.blend_w_inst,
                blend_w_smooth=best_a.cfg.blend_w_smooth,
                smooth_span=best_a.cfg.smooth_span,
                streak_max_boost=max_boost,
                streak_norm=norm,
            )
            r = replay(alpha_rows, cfg, label=f"B_streak_b{max_boost}_n{norm}", **common_kw)
            stage_b.append(r)

    stage_b_sorted = sorted(stage_b, key=lambda r: (r.avg_max + r.avg_final), reverse=True)
    print_result_table(stage_b_sorted[:6], focus_symbols, "Stage B: streak 扫描 Top-6")
    best_b = stage_b_sorted[0]
    print(
        f"\n>>> Stage B 最优: {best_b.label} "
        f"(boost={best_b.cfg.streak_max_boost}, norm={best_b.cfg.streak_norm})"
    )

    # ===== Stage C: 在 B 最优上加新因子 =====
    base_cfg = best_b.cfg
    stage_c: list[RunResult] = [best_b]
    accel_cfg = Cfg(**{**base_cfg.__dict__, "accel_scale": 1.5, "accel_max": 0.4})
    consist_cfg = Cfg(**{**base_cfg.__dict__, "consist_boost": 0.20})
    both_cfg = Cfg(**{**base_cfg.__dict__, "accel_scale": 1.5, "accel_max": 0.4, "consist_boost": 0.20})
    consist_strong_cfg = Cfg(**{**base_cfg.__dict__, "consist_boost": 0.40})

    for cfg, lbl in [
        (accel_cfg, "C_accel_only"),
        (consist_cfg, "C_consist0.20"),
        (consist_strong_cfg, "C_consist0.40"),
        (both_cfg, "C_accel+consist"),
    ]:
        r = replay(alpha_rows, cfg, label=lbl, **common_kw)
        stage_c.append(r)

    print_result_table(stage_c, focus_symbols, "Stage C: 在最优 V3 上叠加新因子")

    # ===== 汇总 Pareto + 推荐 =====
    all_results = [baseline] + stage_a + stage_b[1:] + stage_c[1:]
    pareto_all = pareto(all_results)
    print_result_table(
        sorted(pareto_all, key=lambda r: r.avg_max, reverse=True),
        focus_symbols, "全局 Pareto 前沿",
    )

    best_overall = max(all_results, key=lambda r: r.avg_max + r.avg_final * 0.5 + r.hit20 * 0.3)
    print(
        f"\n>>> 综合推荐 (avgMax + 0.5*avgFinal + 0.3*hit20 最大): {best_overall.label}"
    )
    cfg = best_overall.cfg
    print(
        f"    blend=({cfg.blend_w_inst:.2f}, {cfg.blend_w_smooth:.2f}) "
        f"smooth_span={cfg.smooth_span} "
        f"streak=(max_boost={cfg.streak_max_boost}, norm={cfg.streak_norm}) "
        f"accel_scale={cfg.accel_scale} consist_boost={cfg.consist_boost}"
    )
    print(
        f"    -> avgMax={_fmt_pct(best_overall.avg_max)} "
        f"avgFinal={_fmt_pct(best_overall.avg_final)} "
        f"hit10={_fmt_pct(best_overall.hit10)} hit20={_fmt_pct(best_overall.hit20)} "
        f"pos={_fmt_pct(best_overall.pos)} samples={best_overall.samples}"
    )


if __name__ == "__main__":
    main()
