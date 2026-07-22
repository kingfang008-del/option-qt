#!/usr/bin/env python3
"""Ablation: demoted peer3 vs flow→response→state→event stack.

Layers (applied on peer3 research baseline):
  L0  demoted baseline (hold_extend, no path confirm, no state_gate)
  L1  + stock_path_confirm (full window)
  L2  + state_gate (mixed_wash breadth≥5 block UP)
  L3  full stack (path + state + delta/roi time stops + cond ladder)

Windows: strong May–Jul, weak Feb–Apr. Optional 07-20 fused freeze tag.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

BASE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
STACK = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_flow_state_event_v1.json"
)

PATH_CONFIRM_HARD = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "on_timeout": "block",
    "delay_on_pos": True,
    "tod_start": "10:30",
    "tod_end": "14:00",
}

# Soft path: adverse-first veto only; timeout allow (keeps strong edge).
PATH_CONFIRM_SOFT = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "on_timeout": "allow",
    "delay_on_pos": False,
    "tod_start": "10:30",
    "tod_end": "14:00",
}

STATE_GATE = {
    "enabled": True,
    "asof": "10:30",
    "mixed_wash": {
        "washout_breadth_min": 5,
        "wash_drop_min": 0.008,
        "frac_above_min": 0.35,
        "frac_above_max": 0.70,
        "action": "block_up",
    },
    "reclaim_trap": {
        "rule": "washout_and_reclaim",
        "action": "scale",
        "scale": 0.5,
        "wash_drop_min": 0.015,
        "washout_breadth_min": 5,
    },
}


def _apply_layer(base: dict[str, Any], layer: str) -> dict[str, Any]:
    p = copy.deepcopy(base)
    trade = p.setdefault("trade", {})
    # strip research knobs then re-add by layer
    trade.pop("stock_path_confirm", None)
    trade.pop("delta_time_stop", None)
    trade.pop("roi_time_stop", None)
    trade.pop("ladder_active", None)
    trade.pop("ladder_fallback_exit_mode", None)
    p["state_gate"] = {"enabled": False}

    if layer == "L0_baseline":
        return p
    if layer == "S0_state_only":
        # Day-state veto for chop / wash / deep-V reclaim trap (no path gate).
        p["state_gate"] = copy.deepcopy(STATE_GATE)
        return p
    if layer == "S1_path_soft":
        trade["stock_path_confirm"] = dict(PATH_CONFIRM_SOFT)
        return p
    if layer == "S01_state_path":
        # Agreed pack: mf10 ignition + state veto + soft path confirm.
        p["state_gate"] = copy.deepcopy(STATE_GATE)
        trade["stock_path_confirm"] = dict(PATH_CONFIRM_SOFT)
        return p
    if layer == "L1_path":
        trade["stock_path_confirm"] = dict(PATH_CONFIRM_HARD)
        return p
    if layer == "L2_path_state":
        trade["stock_path_confirm"] = dict(PATH_CONFIRM_HARD)
        p["state_gate"] = copy.deepcopy(STATE_GATE)
        return p
    if layer == "L3_full_stack":
        # use dedicated stack profile trade/state pieces
        stack = load_profile(STACK)
        p["state_gate"] = copy.deepcopy(stack.get("state_gate") or STATE_GATE)
        p["state_gate"]["enabled"] = True
        for k in (
            "stock_path_confirm",
            "delta_time_stop",
            "roi_time_stop",
            "ladder_active",
            "ladder_fallback_exit_mode",
        ):
            if k in (stack.get("trade") or {}):
                trade[k] = copy.deepcopy(stack["trade"][k])
        return p
    raise SystemExit(f"unknown layer: {layer}")


def _run(prof: dict[str, Any], *, start: str, end: str, tag: str, out: Path) -> dict[str, Any]:
    p = copy.deepcopy(prof)
    p["date_range"] = {"start": start, "end": end}
    rr = dict(p.get("regime_router") or {})
    rr["enabled"] = False
    p["regime_router"] = rr
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    return {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "n_stock_path_confirm_block": s.get("n_stock_path_confirm_block"),
        "n_state_gate_block": s.get("n_state_gate_block"),
        "state_gate_day_counts": s.get("state_gate_day_counts"),
        "n_ladder_days": s.get("n_ladder_days"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=BASE)
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/flow_state_window_validate_v1",
    )
    ap.add_argument(
        "--layers",
        default="L0_baseline,S0_state_only,S1_path_soft,S01_state_path",
        help="Default validates mf10 + state/path pack (not hard L3).",
    )
    ap.add_argument("--strong-start", default="2026-05-01")
    ap.add_argument("--strong-end", default="2026-07-21")
    ap.add_argument("--weak-start", default="2026-02-01")
    ap.add_argument("--weak-end", default="2026-04-30")
    ap.add_argument("--skip-weak", action="store_true")
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [x.strip() for x in str(args.layers).split(",") if x.strip()]
    windows = [("strong_may_jul", args.strong_start, args.strong_end)]
    if not args.skip_weak:
        windows.append(("weak_feb_apr", args.weak_start, args.weak_end))

    rows: list[dict[str, Any]] = []
    for wname, start, end in windows:
        for layer in layers:
            tag = f"{wname}_{layer}"
            print(f"[run] {tag} {start}→{end}", flush=True)
            prof = _apply_layer(base, layer)
            row = _run(prof, start=start, end=end, tag=tag, out=out)
            row["window"] = wname
            row["layer"] = layer
            rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} maxdd={row['maxdd']:+.3f} n={row['n_trades']} "
                f"path_blk={row.get('n_stock_path_confirm_block')} "
                f"state_blk={row.get('n_state_gate_block')}",
                flush=True,
            )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)

    # Verdict vs L0: prefer soft pack S01 / S0 / S1 (hard L3 only if present).
    def _cell(window: str, layer: str, col: str = "total_ret") -> float | None:
        hit = rdf[(rdf["window"] == window) & (rdf["layer"] == layer)]
        if hit.empty:
            return None
        val = hit.iloc[0][col]
        return None if val is None or (isinstance(val, float) and pd.isna(val)) else float(val)

    strong_l0 = _cell("strong_may_jul", "L0_baseline")
    weak_l0 = _cell("weak_feb_apr", "L0_baseline")
    cand_layers = [
        x
        for x in ("S01_state_path", "S0_state_only", "S1_path_soft", "L3_full_stack")
        if x in set(layers)
    ]
    cand_scores: list[dict[str, Any]] = []
    for layer in cand_layers:
        s_ret = _cell("strong_may_jul", layer)
        w_ret = _cell("weak_feb_apr", layer)
        s_dd = _cell("strong_may_jul", layer, "maxdd")
        w_dd = _cell("weak_feb_apr", layer, "maxdd")
        keep = (s_ret / strong_l0) if strong_l0 and strong_l0 > 1e-9 and s_ret is not None else None
        cand_scores.append(
            {
                "layer": layer,
                "strong_ret": s_ret,
                "weak_ret": w_ret,
                "strong_maxdd": s_dd,
                "weak_maxdd": w_dd,
                "strong_keep": keep,
                "weak_over_strong": (w_ret / s_ret) if s_ret and s_ret > 1e-9 and w_ret is not None else None,
            }
        )

    l0_ws = (weak_l0 / strong_l0) if strong_l0 and strong_l0 > 1e-9 and weak_l0 is not None else None
    flags: list[str] = []
    best = None
    for c in cand_scores:
        keep = c.get("strong_keep")
        if keep is not None and keep < 0.40:
            flags.append(f"{c['layer']}_strong_edge_destroyed")
            continue
        weak_ok = (
            c.get("weak_ret") is not None
            and weak_l0 is not None
            and c["weak_ret"] > weak_l0
        )
        frag_ok = (
            l0_ws is not None
            and c.get("weak_over_strong") is not None
            and c["weak_over_strong"] > l0_ws + 0.05
        )
        dd_ok = (
            c.get("weak_maxdd") is not None
            and _cell("weak_feb_apr", "L0_baseline", "maxdd") is not None
            and c["weak_maxdd"] > _cell("weak_feb_apr", "L0_baseline", "maxdd")
        )
        if keep is not None and keep >= 0.85 and (weak_ok or frag_ok or dd_ok):
            flags.append(f"{c['layer']}_pass")
            if best is None or (c.get("strong_keep") or 0) > (best.get("strong_keep") or 0):
                best = c
        elif keep is not None and keep >= 0.85 and c.get("strong_ret") is not None and strong_l0 is not None:
            # retain strong but no weak lift → still research-usable if not worse weak
            if c.get("weak_ret") is not None and weak_l0 is not None and c["weak_ret"] + 0.05 >= weak_l0:
                flags.append(f"{c['layer']}_retain_ok")
                if best is None:
                    best = c

    if any(f.endswith("_strong_edge_destroyed") for f in flags) and not any(
        f.endswith("_pass") or f.endswith("_retain_ok") for f in flags
    ):
        decision = "REJECT_STACK"
    elif any(f.endswith("_pass") for f in flags):
        decision = "PROMOTE_TO_RESEARCH_CANDIDATE"
    elif any(f.endswith("_retain_ok") for f in flags):
        decision = "PROMOTE_SOFT_PACK"
    else:
        decision = "INCONCLUSIVE_KEEP_ITERATING"

    summary = {
        "decision": decision,
        "flags": flags,
        "best_candidate": best,
        "candidates": cand_scores,
        "ratios": {"l0_weak_over_strong": l0_ws},
        "protocol": "mf10_ignition_plus_state_path_veto",
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "\n".join(
            [
                "# mf10 + State/Path Veto Validation",
                "",
                f"**Decision: `{decision}`**",
                "",
                "Protocol: sliding-window ignition stays; market state / soft path are veto-only.",
                "",
                "## Best candidate",
                "",
                "```json",
                json.dumps(best, indent=2, default=str),
                "```",
                "",
                "## Candidates",
                "",
                "```json",
                json.dumps(cand_scores, indent=2, default=str),
                "```",
                "",
                "## Scoreboard",
                "",
                rdf.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "flags": flags,
                "best_candidate": best,
                "ratios": summary["ratios"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
