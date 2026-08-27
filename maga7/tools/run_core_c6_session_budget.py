#!/usr/bin/env python3
"""CORE C6 — transferable session risk budget (realized equity DD).

C2 used 10:30 VIXY/breadth and failed: the same day-features print in both
climates. C6 asks whether a **path-accounting** size scale transfers:

    remaining size *= scale when current_dd = equity/peak-1 <= trigger

Calendar is eval-only. No Rule-A change, no morph BLOCK, no ML.

Corpus: C5 drop3 research_baseline book (current wired spine).
Action: post-hoc size rescale of the same fires (no skip / no backfill).

Negative controls (expect FAIL):
  after_day_loss  — cuts recovery days in BOTH windows
  second_seat     — naive daily unit cap; cuts strong-window fat

PASS:
  strong keep >= 0.95
  weak keep >= 0.95
  weak MaxDD Δ >= 50bp (shallower)
  n_scaled_weak >= 3
  n_scaled_strong >= 1   (must still be able to fire when strong path holes)

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c6_session_budget \\
    --tag research_core_c6_session_budget
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.session_risk_budget import (
    parse_session_risk_budget,
    resolve_session_risk_budget,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
C5_ROOT = Path("/mnt/s990/data/maga7/results/research_core_c5_morph_debt")
WINDOWS = (
    ("weak", "2026-01-02", "2026-03-31"),
    ("strong", "2026-04-01", "2026-07-21"),
)

VARIANTS: list[dict[str, Any]] = [
    {"name": "after_day_loss_x05", "kind": "after_day_loss", "scale": 0.5},
    {"name": "second_seat_x05", "kind": "second_seat", "scale": 0.5},
    {
        "name": "dd3_x05",
        "kind": "dd_step",
        "cfg": {"enabled": True, "mode": "dd_step", "dd_trigger": -0.03, "scale": 0.5},
    },
    {
        "name": "dd5_x05",
        "kind": "dd_step",
        "cfg": {"enabled": True, "mode": "dd_step", "dd_trigger": -0.05, "scale": 0.5},
    },
    {
        "name": "dd6_x05",
        "kind": "dd_step",
        "cfg": {"enabled": True, "mode": "dd_step", "dd_trigger": -0.06, "scale": 0.5},
    },
    {
        "name": "dd5_x07",
        "kind": "dd_step",
        "cfg": {"enabled": True, "mode": "dd_step", "dd_trigger": -0.05, "scale": 0.7},
    },
    {
        "name": "dd_linear10",
        "kind": "dd_linear",
        "cfg": {"enabled": True, "mode": "dd_linear", "dd_span": 0.10, "min_scale": 0.0},
    },
]


def load_c5_drop3_book(c5_root: Path = C5_ROOT) -> pd.DataFrame:
    parts = []
    for wname, _, _ in WINDOWS:
        p = c5_root / wname / "replay__drop3_deprecated" / "trades.csv"
        if not p.exists():
            raise SystemExit(f"missing C5 drop3 book {p} (run C5 first)")
        t = pd.read_csv(p)
        t["window"] = wname
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].astype(str)
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out["size_frac"] = pd.to_numeric(out.get("size_frac"), errors="coerce").fillna(0.2)
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], utc=True, errors="coerce")
    return out.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)


def annotate_path(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy().sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)
    t["seat"] = t.groupby("date", sort=False).cumcount() + 1
    eq = 100.0
    peak = 100.0
    loss_streak = 0
    prev = None
    dd_at: list[float] = []
    streak_at: list[int] = []
    after_loss: list[bool] = []
    for row in t.itertuples(index=False):
        d = str(row.date)
        if prev is not None and d != prev:
            day = t[t["date"] == prev]
            dret = float(np.prod(1.0 + day["size_frac"].to_numpy() * day["ret"].to_numpy()) - 1.0)
            loss_streak = loss_streak + 1 if dret < 0 else 0
        dd_at.append(eq / peak - 1.0 if peak > 0 else 0.0)
        streak_at.append(int(loss_streak))
        after_loss.append(bool(loss_streak >= 1))
        r = float(row.ret)
        sf = float(row.size_frac)
        if np.isfinite(r):
            eq *= 1.0 + sf * r
            peak = max(peak, eq)
        prev = d
    t["dd_at"] = dd_at
    t["streak_at"] = streak_at
    t["after_loss_day"] = after_loss
    return t


def equity(trades: pd.DataFrame, *, size_col: str = "size_frac") -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "n_scaled": 0}
    eq = 100.0
    peak = 100.0
    maxdd = 0.0
    rets: list[float] = []
    n_scaled = 0
    for row in trades.itertuples(index=False):
        r = float(row.ret)
        if not np.isfinite(r):
            continue
        sf = float(getattr(row, size_col))
        eq *= 1.0 + sf * r
        peak = max(peak, eq)
        if peak > 0:
            maxdd = min(maxdd, eq / peak - 1.0)
        rets.append(r)
        extra = getattr(row, "budget_scale", 1.0)
        if extra is not None and abs(float(extra) - 1.0) > 1e-12:
            n_scaled += 1
    rr = np.asarray(rets, dtype=float)
    return {
        "n": int(len(rr)),
        "total_ret": float(eq / 100.0 - 1.0),
        "maxdd": float(maxdd),
        "win": float((rr > 0).mean()) if len(rr) else None,
        "n_scaled": int(n_scaled),
    }


def keep_ratio(variant_ret: float, base_ret: float) -> float:
    den = 1.0 + float(base_ret)
    if den == 0:
        return 0.0
    return float((1.0 + float(variant_ret)) / den)


def apply_variant(trades: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    t = trades.copy()
    kind = str(spec.get("kind") or "")
    scales: list[float] = []
    reasons: list[str] = []
    if kind in {"dd_step", "dd_linear"}:
        cfg = parse_session_risk_budget(spec["cfg"])
        for dd in t["dd_at"].to_numpy():
            sc, reason = resolve_session_risk_budget(cfg, current_dd=float(dd))
            scales.append(float(sc))
            reasons.append(reason)
    elif kind == "after_day_loss":
        sc0 = float(spec.get("scale", 0.5))
        for hit in t["after_loss_day"].to_numpy():
            sc = sc0 if bool(hit) else 1.0
            scales.append(sc)
            reasons.append("after_day_loss" if sc < 1.0 else "ok")
    elif kind == "second_seat":
        sc0 = float(spec.get("scale", 0.5))
        for seat in t["seat"].to_numpy():
            sc = sc0 if int(seat) >= 2 else 1.0
            scales.append(sc)
            reasons.append("second_seat" if sc < 1.0 else "ok")
    else:
        raise ValueError(f"unknown variant kind {kind}")
    t["budget_scale"] = scales
    t["budget_reason"] = reasons
    t["size_frac"] = t["size_frac"].astype(float) * t["budget_scale"].astype(float)
    return t


def verdict_c6(
    *,
    strong_keep: float,
    weak_keep: float,
    weak_maxdd_delta: float,
    n_scaled_weak: int,
    n_scaled_strong: int,
    kind: str,
) -> dict[str, Any]:
    if kind not in {"dd_step", "dd_linear"}:
        return {"pass": False, "reason": "not_dd_budget"}
    if int(n_scaled_weak) < 3:
        return {"pass": False, "reason": "no_weak_fire"}
    if int(n_scaled_strong) < 1:
        return {"pass": False, "reason": "no_strong_fire"}
    if float(strong_keep) < 0.95 or float(weak_keep) < 0.95:
        return {"pass": False, "reason": "keep_below_bar"}
    if float(weak_maxdd_delta) < 0.005:
        return {"pass": False, "reason": "weak_dd_not_improved"}
    return {"pass": True, "reason": "pass"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c6_session_budget")
    ap.add_argument("--c5-root", default=str(C5_ROOT))
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    book = annotate_path(load_c5_drop3_book(Path(args.c5_root)))
    rows = []
    baseline_by_win: dict[str, dict[str, Any]] = {}
    for wname, w0, w1 in WINDOWS:
        sub = book[(book["date"] >= w0) & (book["date"] <= w1)].copy()
        st = equity(sub)
        baseline_by_win[wname] = st
        rows.append({"variant": "baseline_c5_drop3", "window": wname, **st, "keep": 1.0})

    for spec in VARIANTS:
        scaled = apply_variant(book, spec)
        for wname, w0, w1 in WINDOWS:
            sub = scaled[(scaled["date"] >= w0) & (scaled["date"] <= w1)].copy()
            st = equity(sub)
            base = baseline_by_win[wname]
            rows.append(
                {
                    "variant": spec["name"],
                    "kind": spec["kind"],
                    "window": wname,
                    **st,
                    "keep": keep_ratio(st["total_ret"], base["total_ret"]),
                    "maxdd_delta": float(st["maxdd"] - base["maxdd"]),
                }
            )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    def _row(variant: str, window: str) -> pd.Series | None:
        hit = sb[(sb.variant == variant) & (sb.window == window)]
        return hit.iloc[0] if len(hit) else None

    adopted = []
    for spec in VARIANTS:
        name = spec["name"]
        strong = _row(name, "strong")
        weak = _row(name, "weak")
        if strong is None or weak is None:
            continue
        v = verdict_c6(
            strong_keep=float(strong["keep"]),
            weak_keep=float(weak["keep"]),
            weak_maxdd_delta=float(weak["maxdd"] - baseline_by_win["weak"]["maxdd"]),
            n_scaled_weak=int(weak["n_scaled"]),
            n_scaled_strong=int(strong["n_scaled"]),
            kind=str(spec["kind"]),
        )
        adopted.append(
            {
                "variant": name,
                "kind": spec["kind"],
                "pass": bool(v["pass"]),
                "reason": v["reason"],
                "strong_keep": float(strong["keep"]),
                "weak_keep": float(weak["keep"]),
                "weak_maxdd": float(weak["maxdd"]),
                "weak_maxdd_delta": float(weak["maxdd"] - baseline_by_win["weak"]["maxdd"]),
                "strong_total_ret": float(strong["total_ret"]),
                "weak_total_ret": float(weak["total_ret"]),
                "n_scaled_strong": int(strong["n_scaled"]),
                "n_scaled_weak": int(weak["n_scaled"]),
                "cfg": spec.get("cfg"),
            }
        )
    passed = [a for a in adopted if a["pass"]]
    passed.sort(key=lambda x: (x["weak_maxdd_delta"], x["strong_keep"]), reverse=True)
    promote = f"C6_{passed[0]['variant']}" if passed else "NONE"

    summary = {
        "protocol": "core_c6_session_budget",
        "promotion_mark": "path_dd_soft_size_only",
        "corpus": "research_core_c5_morph_debt/replay__drop3_deprecated",
        "pass_rule": (
            "dd_budget AND strong keep>=0.95 AND weak keep>=0.95 "
            "AND weak MaxDD Δ>=50bp AND n_scaled_weak>=3 AND n_scaled_strong>=1"
        ),
        "baseline": {k: v for k, v in baseline_by_win.items()},
        "variants": adopted,
        "promote": promote,
        "pass": bool(promote != "NONE"),
        "next_step": (
            "wire_session_risk_budget_on_research_baseline"
            if promote != "NONE"
            else "do_not_wire_keep_morph_freeze"
        ),
        "note": (
            "Not a climate label. Live gate = realized equity DD at entry. "
            "Production freeze unchanged. No net-new morph BLOCK."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C6 — Session risk budget（可迁移状态预算）",
        "",
        "- action: **soft size scale from realized equity DD**（不改 Rule-A、不加 BLOCK）",
        "- 对照：C5 drop3 当前脊骨（弱 n=34 / 强 n=65）",
        "- 负对照：`after_day_loss` / `second_seat`（路径记账但不可迁移）",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}**",
        "",
        "## Scoreboard",
        "",
    ]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    if promote != "NONE":
        best = passed[0]
        lines += [
            "",
            "## 结论",
            "",
            f"**C6 PASS** → `{best['variant']}` "
            f"strong keep={best['strong_keep']:.3f} · weak keep={best['weak_keep']:.3f} · "
            f"weak MaxDD Δ={best['weak_maxdd_delta']:+.3f} · "
            f"n_scaled weak/strong={best['n_scaled_weak']}/{best['n_scaled_strong']}。",
            "可写入 research_baseline `trade.session_risk_budget`（默认 enabled）；生产 freeze 另闸。",
            "不是第七扇 morph 门。C2 气候标签仍然 FAIL。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**C6 FAIL** — 已实现回撤预算未能同时保住强窗并改善弱窗 MaxDD。不接线。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(sb.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
