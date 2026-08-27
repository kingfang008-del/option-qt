#!/usr/bin/env python3
"""CORE C5 — morph-debt audit (strip-one vs current research_baseline).

Inventory the Jul-26 entry-morph BLOCK stack, then dual-window replay with
each gate disabled. L1' climate prior and L4 STOCK_REV already FAIL — do not
pretend they cover these gates. A morph is DEPRECATED only if stripping it
keeps both windows (keep>=0.95) — then we can actually turn it off.

PASS: disable ≥1 redundant morph (net hard-BLOCK count down).
Else: freeze catalog — no net-new hard BLOCKs. Do not add replacements.

Corpus windows match C3:
  weak   2026-01-02 .. 2026-03-31
  strong 2026-04-01 .. 2026-07-21
Control: C3 current-profile baseline (not older S1).

Example:
  PYTHONPATH=. python -m maga7.tools.run_core_c5_morph_debt \\
    --tag research_core_c5_morph_debt
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.tools.run_core_c3_stock_rev import (
    WINDOWS,
    _disable_am,
    keep_ratio,
    reason_stats,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
C3_ROOT = Path("/mnt/s990/data/maga7/results/research_core_c3_stock_rev")

# Tape-pattern hard BLOCKs wired 2026-07-26. Not Watchdog / Hunt / event calendar.
MORPH_GATES: tuple[str, ...] = (
    "overnight_gap_gate",
    "peer_gap_gate",
    "range_stall_gate",
    "dn_gap_stall_gate",
    "up_gap_stall_gate",
    "fo_lod_chase_gate",
)
BLOCK_KEYS = {
    "overnight_gap_gate": "n_overnight_gap_block",
    "peer_gap_gate": "n_peer_gap_block",
    "range_stall_gate": "n_range_stall_block",
    "dn_gap_stall_gate": "n_dn_gap_stall_block",
    "up_gap_stall_gate": "n_up_gap_stall_block",
    "fo_lod_chase_gate": "n_fo_lod_chase_block",
}

SPINE_KEEP = (
    "Rule-A / peer3 / QQQ align",
    "Watchdog L1 degrade+halt",
    "Hunt L2 washout_reclaim",
    "event_calendar + company_news (exogenous)",
    "trade_toxic + hold_extend/giveback + S1 path_confirm",
    "dvol_size_scale (soft size, not BLOCK)",
)
OFF_DO_NOT_WIRE = (
    "climate_prior (C2 FAIL)",
    "stock_rev_exit (C3 FAIL)",
    "tcn_gate / lgbm_bouncer / seat_score_gate (C4 FAIL)",
    "hold_watchdog / chop_gate / corr_rewire / from_open_gate",
)


def _apply_overlay(trade: dict[str, Any], overlay: dict[str, Any]) -> None:
    for k, v in overlay.items():
        if v is None:
            trade[k] = None
        elif isinstance(v, dict) and isinstance(trade.get(k), dict):
            trade[k] = {**trade[k], **v}
        else:
            trade[k] = v


def strip_overlay(gates: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {g: {"enabled": False} for g in gates}


def verdict_strip(
    *,
    strong_keep: float,
    weak_keep: float,
    n_block_total: int,
    min_keep: float = 0.95,
) -> dict[str, Any]:
    keep_ok = float(strong_keep) >= float(min_keep) and float(weak_keep) >= float(min_keep)
    if int(n_block_total) <= 0:
        return {
            "status": "DEAD",
            "pass_deprecate": True,
            "reason": "zero_blocks_on_c3_book",
            "keep_ok": True,
        }
    if keep_ok:
        return {
            "status": "DEPRECATED",
            "pass_deprecate": True,
            "reason": "strip_keep_ok",
            "keep_ok": True,
        }
    return {
        "status": "KEEP",
        "pass_deprecate": False,
        "reason": "strip_hurts_keep",
        "keep_ok": False,
    }


def load_c3_baseline(window: str, c3_root: Path) -> dict[str, Any]:
    tag = c3_root / window / "replay__baseline"
    trades = pd.read_csv(tag / "trades.csv")
    summary = json.loads((tag / "summary.json").read_text(encoding="utf-8"))
    rs = reason_stats(trades)
    start, end = WINDOWS[window]
    blocks = {g: int(summary.get(BLOCK_KEYS[g], 0) or 0) for g in MORPH_GATES}
    return {
        "window": window,
        "variant": "baseline",
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary.get("trade_win") or 0.0),
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
        "keep": 1.0,
        "morph_blocks": blocks,
    }


def run_strip(
    window: str,
    variant: str,
    overlay: dict[str, Any],
    *,
    out_root: Path,
    profile_path: str,
) -> dict[str, Any]:
    start, end = WINDOWS[window]
    prof = deepcopy(load_profile(profile_path))
    _disable_am(prof)
    prof["date_range"] = {"start": start, "end": end}
    trade = prof.setdefault("trade", {})
    _apply_overlay(trade, overlay)
    print(f"=== C5 {window} / {variant} {start}..{end} ===", flush=True)
    result = run_offline_replay(prof, scheme="single")
    summary = result["summary"]
    trades = result["trades"]
    daily = result.get("daily")
    tag = out_root / window / f"replay__{variant}"
    tag.mkdir(parents=True, exist_ok=True)
    (tag / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    trades.to_csv(tag / "trades.csv", index=False)
    if daily is not None and len(daily):
        daily.to_csv(tag / "daily.csv", index=False)
    rs = reason_stats(trades)
    row = {
        "window": window,
        "variant": variant,
        "start": start,
        "end": end,
        "total_ret": float(summary["total_ret"]),
        "maxdd": float(summary["maxdd"]),
        "n_trades": int(summary["n_trades"]),
        "trade_win": float(summary.get("trade_win") or 0.0),
        **{k: v for k, v in rs.items() if k != "reasons"},
        "reasons": rs["reasons"],
    }
    print(
        f"  ret={row['total_ret']:+.3f} n={row['n_trades']} "
        f"dd={row['maxdd']:.3f} win={row['trade_win']:.3f}",
        flush=True,
    )
    return row


def inventory_from_profile(profile: dict[str, Any], c3_blocks: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    trade = profile.get("trade") or {}
    rows = []
    for g in MORPH_GATES:
        raw = trade.get(g) or {}
        rows.append(
            {
                "gate": g,
                "layer": "entry_morph_block",
                "enabled": bool(raw.get("enabled", False)),
                "mode": raw.get("mode"),
                "n_block_weak": int((c3_blocks.get("weak") or {}).get(g, 0)),
                "n_block_strong": int((c3_blocks.get("strong") or {}).get(g, 0)),
                "note": str(raw.get("note") or "")[:180],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_core_c5_morph_debt")
    ap.add_argument("--c3-root", default=str(C3_ROOT))
    ap.add_argument("--min-keep", type=float, default=0.95)
    ap.add_argument(
        "--variants",
        default="all",
        help="comma list of gate names, 'all', or 'all+bundle'",
    )
    args = ap.parse_args(argv)

    profile = load_profile(args.profile)
    out = Path(profile["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    c3_root = Path(args.c3_root)

    wins = list(WINDOWS.keys())
    baseline: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for w in wins:
        b = load_c3_baseline(w, c3_root)
        baseline[w] = b
        rows.append(b)
        print(
            f"=== C5 {w} / baseline (C3 reuse) ret={b['total_ret']:+.3f} "
            f"n={b['n_trades']} blocks={b['morph_blocks']} ===",
            flush=True,
        )

    c3_blocks = {w: baseline[w]["morph_blocks"] for w in wins}
    inv = inventory_from_profile(profile, c3_blocks)
    (out / "inventory.json").write_text(json.dumps(inv, indent=2, default=str))

    if args.variants.strip() == "all+bundle":
        variant_gates: list[tuple[str, tuple[str, ...]]] = [
            (g, (g,)) for g in MORPH_GATES
        ] + [("strip_all_morphs", MORPH_GATES)]
    elif args.variants.strip() == "all":
        variant_gates = [(g, (g,)) for g in MORPH_GATES]
    else:
        names = [x.strip() for x in args.variants.split(",") if x.strip()]
        variant_gates = []
        for n in names:
            if n == "strip_all_morphs":
                variant_gates.append((n, MORPH_GATES))
            elif n in MORPH_GATES:
                variant_gates.append((n, (n,)))
            else:
                raise SystemExit(f"unknown variant {n}")

    for variant, gates in variant_gates:
        overlay = strip_overlay(gates)
        for w in wins:
            row = run_strip(
                w, variant, overlay, out_root=out, profile_path=args.profile
            )
            b = baseline[w]
            row["keep"] = keep_ratio(row["total_ret"], b["total_ret"])
            row["maxdd_delta"] = float(row["maxdd"] - b["maxdd"])
            row["n_delta"] = int(row["n_trades"] - b["n_trades"])
            rows.append(row)

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    adopted: list[dict[str, Any]] = []
    for variant, gates in variant_gates:
        by_w = {r["window"]: r for r in rows if r["variant"] == variant}
        if any(w not in by_w for w in wins):
            continue
        n_block = sum(int(c3_blocks[w].get(g, 0)) for w in wins for g in gates)
        v = verdict_strip(
            strong_keep=float(by_w["strong"]["keep"]),
            weak_keep=float(by_w["weak"]["keep"]),
            n_block_total=n_block,
            min_keep=float(args.min_keep),
        )
        adopted.append(
            {
                "variant": variant,
                "gates": list(gates),
                **v,
                "strong_keep": float(by_w["strong"]["keep"]),
                "weak_keep": float(by_w["weak"]["keep"]),
                "strong_n": int(by_w["strong"]["n_trades"]),
                "weak_n": int(by_w["weak"]["n_trades"]),
                "n_delta_strong": int(by_w["strong"].get("n_delta") or 0),
                "n_delta_weak": int(by_w["weak"].get("n_delta") or 0),
                "n_block_c3": int(n_block),
            }
        )

    deprecate = [
        a
        for a in adopted
        if a["pass_deprecate"] and a["variant"] != "strip_all_morphs"
    ]
    n_on = sum(1 for r in inv if r["enabled"])
    n_drop = len(deprecate)
    passed = n_drop >= 1
    promote = "C5_DROP_" + ",".join(a["variant"] for a in deprecate) if passed else "NONE"
    summary = {
        "protocol": "core_c5_morph_debt",
        "control": "research_core_c3_stock_rev current-profile baseline",
        "morph_gates_on": n_on,
        "n_deprecate": n_drop,
        "pass_rule": "disable ≥1 morph with dual-window strip keep>=0.95 (or DEAD)",
        "spine_keep": list(SPINE_KEEP),
        "off_do_not_wire": list(OFF_DO_NOT_WIRE),
        "inventory": inv,
        "variants": adopted,
        "deprecate": [a["variant"] for a in deprecate],
        "promote": promote,
        "pass": bool(passed),
        "freeze_policy": "no_net_new_hard_block",
        "next_step": (
            "disable_deprecated_morphs_on_research_baseline"
            if passed
            else "freeze_morph_catalog_no_new_blocks"
        ),
        "note": (
            "C2/C3/C4 FAIL so L1'/L4 do not cover these gates. "
            "Event calendar / company_news stay exogenous (not morph). "
            "Do not add a replacement BLOCK in the same change."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# CORE C5 — Morph debt audit",
        "",
        f"- morph BLOCKs on: **{n_on}**",
        f"- deprecate candidates: **{deprecate and [a['variant'] for a in deprecate] or 'NONE'}**",
        f"- promote: **{promote}**",
        f"- pass (count down): **{passed}**",
        f"- freeze: **no_net_new_hard_block**",
        "",
        "## Inventory (C3 book fires)",
        "",
    ]
    try:
        lines.append(pd.DataFrame(inv).to_markdown(index=False))
    except Exception:
        lines.append(pd.DataFrame(inv).to_string(index=False))
    show = sb.drop(columns=["reasons", "morph_blocks"], errors="ignore")
    lines += ["", "## Strip-one scoreboard", ""]
    try:
        lines.append(show.to_markdown(index=False))
    except Exception:
        lines.append(show.to_string(index=False))
    if passed:
        lines += [
            "",
            "## 结论",
            "",
            f"**C5 PASS** — 可关 {n_drop} 门：`{', '.join(a['variant'] for a in deprecate)}`。",
            "关闸后禁止用另一门补上。生产 freeze 另闸。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**C5 未降门** — 六门 strip 都伤 keep，或未过 0.95。保持现状。",
            "**冻结：禁止再净增硬 BLOCK。** C2/C3/C4 已 FAIL，不能用 L1'/L4 覆盖当借口拆门。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(show.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": passed, "deprecate": summary["deprecate"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
