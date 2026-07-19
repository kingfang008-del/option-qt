#!/usr/bin/env python3
"""本地三件事（不动 peer3 默认旋钮）：Hunt 大亏后停手 / 去一只票 / 滑点更贵更便宜。

白话：
  1) Hunt 亏狠了还要不要继续做当天后面的单？
  2) L2 是不是靠某一两只股票撑起来的？
  3) 成交价差一点，结论会不会翻脸？
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

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = {
    "strong": ("2026-05-01", "2026-07-17"),
    "weak": ("2026-02-01", "2026-04-30"),
}


def _tot(daily: pd.DataFrame) -> float:
    eq = 1.0
    for r in daily["day_ret"].astype(float):
        eq *= 1.0 + float(r)
    return eq - 1.0


def _run(prof: dict, *, start: str, end: str) -> dict:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    daily = res["daily"]
    trades = res["trades"]
    total = _tot(daily) if not daily.empty else float(s["total_ret"])
    hunt = trades[trades["event_source"].astype(str) == "hunt"] if (
        not trades.empty and "event_source" in trades.columns
    ) else trades.iloc[0:0]
    return {
        "total_ret": float(total),
        "maxdd": float(s["maxdd"]),
        "n_trades": int(s["n_trades"]),
        "n_hunt": int(s.get("n_hunt_trades") or 0),
        "n_hunt_day_circuit": int(s.get("n_hunt_day_circuit") or 0),
        "hunt_avg_ret": float(hunt["ret"].mean()) if len(hunt) else None,
        "hunt_worst": float(hunt["ret"].min()) if len(hunt) else None,
    }


def _drop_symbol(prof: dict, sym: str) -> dict:
    p = copy.deepcopy(prof)
    sym = sym.upper()
    p["symbols"] = [s for s in p.get("symbols") or [] if str(s).upper() != sym]
    sig = p.setdefault("signal", {})
    peers = list(sig.get("peer_symbols") or [])
    if peers:
        sig["peer_symbols"] = [s for s in peers if str(s).upper() != sym]
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default="maga7/results/watchdog/l2_local_checks")
    ap.add_argument(
        "--skip-leaveone",
        action="store_true",
        help="只跑熔断+滑点，跳过去票（更快）",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = load_profile(args.profile)
    rows: list[dict] = []

    def add(family: str, variant: str, window: str, metrics: dict, **extra):
        rows.append(
            {
                "family": family,
                "variant": variant,
                "window": window,
                **metrics,
                **extra,
            }
        )

    # --- 0) 基线对照 ---
    base_by_w: dict[str, dict] = {}
    for wname, (start, end) in WINDOWS.items():
        print(f"[base] {wname}...")
        m = _run(base, start=start, end=end)
        base_by_w[wname] = m
        add("base", "peer3_l2", wname, m)
        print(f"  ret={m['total_ret']:+.2%} dd={m['maxdd']:.2%} hunt={m['n_hunt']}")

    # --- 1) Hunt 大亏后停手 ---
    for thr in (-0.30, -0.50):
        tag = f"hunt_circuit_{thr:.2f}".replace("-", "m")
        for wname, (start, end) in WINDOWS.items():
            print(f"[{tag}] {wname}...")
            p = copy.deepcopy(base)
            p.setdefault("watchdog", {}).setdefault("hunter", {})["day_circuit_ret"] = thr
            m = _run(p, start=start, end=end)
            vs = m["total_ret"] / base_by_w[wname]["total_ret"] if base_by_w[wname]["total_ret"] else None
            add("hunt_circuit", tag, wname, m, vs_base=vs, threshold=thr)
            print(
                f"  ret={m['total_ret']:+.2%} vs_base={vs:.3f} "
                f"circuit_fires={m['n_hunt_day_circuit']}"
            )

    # --- 2) 滑点 ---
    for frac in (0.7, 0.8, 0.9):
        tag = f"fill_{frac:.1f}"
        for wname, (start, end) in WINDOWS.items():
            print(f"[{tag}] {wname}...")
            p = copy.deepcopy(base)
            # Fill knobs live under profile.fill, not trade.*
            p.setdefault("fill", {})["entry_frac"] = frac
            p["fill"]["exit_frac"] = frac
            m = _run(p, start=start, end=end)
            vs = m["total_ret"] / base_by_w[wname]["total_ret"] if base_by_w[wname]["total_ret"] else None
            add("fill", tag, wname, m, vs_base=vs, entry_frac=frac)
            print(f"  ret={m['total_ret']:+.2%} vs_base={vs:.3f}")

    # --- 3) 去一只票 ---
    if not args.skip_leaveone:
        for sym in list(base.get("symbols") or []):
            tag = f"drop_{sym}"
            p = _drop_symbol(base, sym)
            for wname, (start, end) in WINDOWS.items():
                print(f"[{tag}] {wname}...")
                m = _run(p, start=start, end=end)
                # vs same-window base (different universe — report absolute + note)
                add("leave_one_out", tag, wname, m, dropped=sym)
                print(f"  ret={m['total_ret']:+.2%} hunt={m['n_hunt']} dd={m['maxdd']:.2%}")

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # 白话结论
    def _vs(family: str, variant: str, window: str) -> float | None:
        hit = sb[(sb.family == family) & (sb.variant == variant) & (sb.window == window)]
        if hit.empty or "vs_base" not in hit.columns:
            return None
        v = hit.iloc[0].get("vs_base")
        return float(v) if v == v else None

    lines = []
    lines.append("# L2 本地三件事 — 白话结论\n")
    lines.append(f"**日期：** 2026-07-19  \n**Profile：** `{args.profile}`  \n")
    lines.append("基线未改参；下列均为对照实验。\n")

    lines.append("## 0. 当前基线（对照）\n")
    lines.append("| 窗 | 收益 | MaxDD | Hunt笔数 |\n|----|-----:|------:|---------:|\n")
    for w in WINDOWS:
        m = base_by_w[w]
        lines.append(
            f"| {w} | {m['total_ret']:+.1%} | {m['maxdd']:.1%} | {m['n_hunt']} |\n"
        )

    lines.append("\n## 1. Hunt 亏狠了 → 当天后面还做不做？\n")
    lines.append(
        "规则：Hunt 单笔收益 ≤ 阈值后，**停掉当天剩余开仓**（含基线）。"
        "说明：现在每天最多 1 笔 Hunt，「禁第二笔 Hunt」本身几乎无事可做；"
        "所以验的是更狠的版本——大亏后整日停手。\n\n"
    )
    lines.append("| 阈值 | 强窗 vs基线 | 弱窗 vs基线 | 强窗触发次数 |\n|------|------------:|------------:|-------------:|\n")
    for thr in (-0.30, -0.50):
        tag = f"hunt_circuit_{thr:.2f}".replace("-", "m")
        vs_s = _vs("hunt_circuit", tag, "strong")
        vs_w = _vs("hunt_circuit", tag, "weak")
        fire = sb[(sb.variant == tag) & (sb.window == "strong")]["n_hunt_day_circuit"]
        nf = int(fire.iloc[0]) if len(fire) else 0
        lines.append(
            f"| {thr:.0%} | {vs_s:.1%} | {vs_w:.1%} | {nf} |\n"
            if vs_s is not None and vs_w is not None
            else f"| {thr:.0%} | — | — | {nf} |\n"
        )
    # verdict
    vs30 = _vs("hunt_circuit", "hunt_circuit_m0.30", "strong")
    if vs30 is not None and vs30 < 0.95:
        lines.append(
            "\n**结论：不建议开。** 强窗掉到基线 95% 以下——"
            "典型是 Hunt 大亏日后基线反向单在救命（如 07-02），一停手把救命单也砍了。\n"
        )
    elif vs30 is not None:
        lines.append("\n**结论：可作研究候选**（强窗仍 ≥95% 基线），默认仍建议关着。\n")
    else:
        lines.append("\n**结论：缺数据。**\n")

    lines.append("\n## 2. 成交更贵/更便宜（滑点）\n")
    lines.append("默认 entry/exit_frac=0.8。扫 0.7 / 0.8 / 0.9。\n\n")
    lines.append("| frac | 强窗 vs基线 | 弱窗 vs基线 |\n|-----:|------------:|------------:|\n")
    for frac in (0.7, 0.8, 0.9):
        tag = f"fill_{frac:.1f}"
        vs_s = _vs("fill", tag, "strong")
        vs_w = _vs("fill", tag, "weak")
        if vs_s is not None and vs_w is not None:
            lines.append(f"| {frac:.1f} | {vs_s:.1%} | {vs_w:.1%} |\n")
    vs9s = _vs("fill", "fill_0.9", "strong")
    vs9w = _vs("fill", "fill_0.9", "weak")
    if (vs9s is not None and vs9s < 0.90) or (vs9w is not None and vs9w < 0.90):
        lines.append(
            "\n**结论：对坏成交很敏感。** frac=0.9 时弱窗/强窗相对默认掉较多——"
            "默认 0.8 可留，上线必须盯真实滑点。\n"
        )
    else:
        lines.append(
            "\n**结论：在 0.7–0.9 内相对默认不翻脸。**\n"
        )

    lines.append("\n## 3. 去掉一只股票重跑\n")
    if args.skip_leaveone:
        lines.append("（本次跳过）\n")
    else:
        lines.append(
            "每次从池子去掉一只，看 L2 还是否明显赚钱。"
            "注意：宇宙变了，数字不能直接当「vs 基线 %」，看绝对收益是否塌缩。\n\n"
        )
        lines.append("| 去掉 | 强窗收益 | 弱窗收益 | 强窗Hunt |\n|------|----------:|----------:|---------:|\n")
        lo = sb[sb.family == "leave_one_out"]
        for sym in list(base.get("symbols") or []):
            tag = f"drop_{sym}"
            srow = lo[(lo.variant == tag) & (lo.window == "strong")]
            wrow = lo[(lo.variant == tag) & (lo.window == "weak")]
            if srow.empty or wrow.empty:
                continue
            lines.append(
                f"| {sym} | {srow.iloc[0]['total_ret']:+.1%} | "
                f"{wrow.iloc[0]['total_ret']:+.1%} | {int(srow.iloc[0]['n_hunt'])} |\n"
            )
        # fragile if any strong goes negative or near zero
        strong_lo = lo[lo.window == "strong"]
        if len(strong_lo) and (strong_lo["total_ret"] < 0).any():
            lines.append(
                "\n**结论：去掉某只后强窗翻负 —— L2 对该票依赖重，不能当稳固通用边。**\n"
            )
        elif len(strong_lo):
            worst = strong_lo.loc[strong_lo["total_ret"].idxmin()]
            lines.append(
                f"\n**结论：去掉任一只强窗仍为正。"
                f"最差是去掉 `{worst['dropped']}`（{worst['total_ret']:+.1%}）。**\n"
            )

    lines.append("\n## 总判\n")
    lines.append(
        "- 本地能先做完的三件事已跑。\n"
        "- **进不进基线**：熔断默认保持关；滑点/去票结论见上。\n"
        "- **上线过渡**（Scanner/Shadow）仍要另做，本脚本不覆盖。\n"
    )
    md = "".join(lines)
    (out / "README.md").write_text(md)
    (out / "summary.json").write_text(
        json.dumps({"scoreboard": rows, "base": base_by_w}, indent=2, default=str) + "\n"
    )
    print("\n" + md)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
