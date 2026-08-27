#!/usr/bin/env python3
"""Grid: peer3 entries × relaxed / conditional ladder exits.

Runs offline May–Jul (quotes thru 07-17) and optional fused 07-20 stress.

Usage:
  python -m maga7.tools.run_sec_ladder_grid --leg offline
  python -m maga7.tools.run_sec_ladder_grid --leg fused
  python -m maga7.tools.run_sec_ladder_grid --leg both
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

PEER3 = (
    ROOT
    / "maga7/CONFIG/strategy_profiles/"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_SESSION = Path(
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e"
)


def _lac(
    *,
    max_hold: int,
    sl: list[float],
    tp_trail: float,
    trail_dd: float,
    tp_exit: float,
    stall_peak: float,
    stall_sec: int,
    mf_flip: bool = True,
    when: str = "always",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "when": when,
        "max_hold_seconds": int(max_hold),
        "keep_outer_rails": True,
        "sl_rails": [{"ret": float(x)} for x in sl],
        "tp_rails": [
            {"ret": float(tp_trail), "action": "trail", "trail_dd": float(trail_dd)},
            {"ret": float(tp_exit), "action": "exit"},
        ],
        "profit_stall": {"min_peak": float(stall_peak), "stall_seconds": int(stall_sec)},
        "mf_flip": bool(mf_flip),
        "mf_grace_seconds": 30 if mf_flip else 9999,
    }


# peer3 signal unchanged; exit overlays only. No delta/roi (those killed May–Jul).
VARIANTS: dict[str, dict[str, Any]] = {
    "peer3_extend": {
        # baseline control — no overlay
    },
    # Previous tight rails (reference)
    "ladder_tight_v1": {
        "exit_mode": "ladder_active",
        "hold_minutes": 5,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=300,
            sl=[-0.10, -0.18],
            tp_trail=0.12,
            trail_dd=0.04,
            tp_exit=0.20,
            stall_peak=0.08,
            stall_sec=20,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Relaxed: wider SL, later trail, longer cap, softer stall
    "ladder_loose_10m": {
        "exit_mode": "ladder_active",
        "hold_minutes": 10,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=600,
            sl=[-0.18, -0.28],
            tp_trail=0.20,
            trail_dd=0.08,
            tp_exit=0.35,
            stall_peak=0.15,
            stall_sec=60,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    "ladder_loose_15m": {
        "exit_mode": "ladder_active",
        "hold_minutes": 15,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=900,
            sl=[-0.20, -0.30],
            tp_trail=0.25,
            trail_dd=0.10,
            tp_exit=0.45,
            stall_peak=0.20,
            stall_sec=90,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Mid: protect without clipping peer3 TPs as hard
    "ladder_mid_8m": {
        "exit_mode": "ladder_active",
        "hold_minutes": 8,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=480,
            sl=[-0.15, -0.25],
            tp_trail=0.18,
            trail_dd=0.07,
            tp_exit=0.40,
            stall_peak=0.12,
            stall_sec=45,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # No stall — only SL ladder + trail + SEC_MAX + optional mf
    "ladder_mid_nostall": {
        "exit_mode": "ladder_active",
        "hold_minutes": 8,
        "hold_extend_minutes": None,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=480,
            sl=[-0.15, -0.25],
            tp_trail=0.18,
            trail_dd=0.07,
            tp_exit=0.40,
            stall_peak=0.99,
            stall_sec=9999,
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Conditional: mixed_wash_up days → mid ladder; else peer3 extend
    "ladder_cond_mid_wash": {
        "exit_mode": "hold_extend",
        "hold_minutes": 30,
        "hold_extend_minutes": 45,
        "hold_extend_mtm_min": 0.0,
        "hold_extend_require_mf": False,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=480,
            sl=[-0.15, -0.25],
            tp_trail=0.18,
            trail_dd=0.07,
            tp_exit=0.40,
            stall_peak=0.12,
            stall_sec=45,
            when="mixed_wash_up",
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
    # Conditional + looser toxic-day book
    "ladder_cond_loose_wash": {
        "exit_mode": "hold_extend",
        "hold_minutes": 30,
        "hold_extend_minutes": 45,
        "hold_extend_mtm_min": 0.0,
        "hold_extend_require_mf": False,
        "ladder_fallback_exit_mode": "hold_extend",
        "ladder_active": _lac(
            max_hold=600,
            sl=[-0.18, -0.28],
            tp_trail=0.20,
            trail_dd=0.08,
            tp_exit=0.35,
            stall_peak=0.15,
            stall_sec=60,
            when="mixed_wash_up",
        ),
        "delta_time_stop": {"enabled": False},
        "roi_time_stop": {"enabled": False},
    },
}


def _metrics(summary: dict, trades: pd.DataFrame) -> dict[str, Any]:
    ret = pd.to_numeric(trades["ret"], errors="coerce") if len(trades) else pd.Series(dtype=float)
    hold = None
    if len(trades) and "entry_ts" in trades.columns and "exit_ts" in trades.columns:
        et = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
        xt = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
        hold = (xt - et).dt.total_seconds()
    reasons = (
        trades["reason"].astype(str).value_counts().to_dict() if len(trades) and "reason" in trades.columns else {}
    )
    return {
        "n_trades": summary.get("n_trades"),
        "total_ret": summary.get("total_ret"),
        "maxdd": summary.get("maxdd"),
        "day_win": summary.get("day_win"),
        "trade_win": summary.get("trade_win"),
        "mean_ret": float(ret.mean()) if len(ret) else float("nan"),
        "mean_hold_sec": float(hold.mean()) if hold is not None and len(hold) else float("nan"),
        "median_hold_sec": float(hold.median()) if hold is not None and len(hold) else float("nan"),
        "n_ladder_days": summary.get("n_ladder_days"),
        "n_ladder_fallback_days": summary.get("n_ladder_fallback_days"),
        "reasons": reasons,
    }


def _run_offline(start: str, end: str, tag: str, names: list[str]) -> list[dict[str, Any]]:
    base = load_profile(PEER3)
    base["date_range"] = {"start": start, "end": end}
    out_root = Path(base["_paths"]["results_dir"]) / tag
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; choose {list(VARIANTS)}")
        ov = VARIANTS[name]
        prof = deepcopy(base)
        trade = prof.setdefault("trade", {})
        for k, v in ov.items():
            trade[k] = deepcopy(v)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== offline {name} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        s = result["summary"]
        trades = result["trades"]
        (out / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        trades.to_csv(out / "trades.csv", index=False)
        result["daily"].to_csv(out / "daily.csv", index=False)
        m = {"variant": name, "leg": "offline", "start": start, "end": end, "out": str(out)}
        m.update(_metrics(s, trades))
        rows.append(m)
        slim = {k: v for k, v in m.items() if k != "reasons"}
        print(json.dumps(slim, indent=2, default=str), flush=True)
        print(f"reasons={m['reasons']}", flush=True)
    (out_root / "grid_compare.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in rows]).to_csv(
        out_root / "grid_compare.csv", index=False
    )
    # rank by total_ret then maxdd
    ranked = sorted(rows, key=lambda r: (float(r.get("total_ret") or -999), float(r.get("maxdd") or -999)), reverse=True)
    (out_root / "grid_rank.json").write_text(json.dumps(ranked, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_root / 'grid_compare.json'}", flush=True)
    return rows


def _run_fused(session_dir: Path, tag: str, names: list[str]) -> list[dict[str, Any]]:
    from maga7.tools.run_live_fused_replay import run_fused_replay

    rows: list[dict[str, Any]] = []
    out_root = session_dir / tag
    out_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        ov = VARIANTS[name] or None
        # fused: empty overlay = pure peer3
        trade_ov = deepcopy(ov) if ov else None
        print(f"=== fused {name} ===", flush=True)
        summary = run_fused_replay(
            session_dir,
            scheme="single",
            disable_prevention=True,
            tag=f"{tag}_{name}",
            trade_overrides=trade_ov if trade_ov else None,
            profile_path_override=str(PEER3),
            redis_db=0,
        )
        trades_path = session_dir / f"fused_replay_{tag}_{name}" / "trades.csv"
        trades = pd.read_csv(trades_path) if trades_path.is_file() else pd.DataFrame()
        m = {
            "variant": name,
            "leg": "fused_20260720",
            "out": str(session_dir / f"fused_replay_{tag}_{name}"),
        }
        m.update(_metrics(summary, trades))
        rows.append(m)
        print(json.dumps({k: v for k, v in m.items() if k != "reasons"}, indent=2, default=str), flush=True)
        print(f"reasons={m['reasons']}", flush=True)
        if len(trades):
            print(trades.to_string(index=False), flush=True)
    (out_root / "fused_grid_compare.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", choices=("offline", "fused", "both"), default="both")
    ap.add_argument("--session-dir", default=str(DEFAULT_SESSION))
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--tag", default="sec_ladder_grid_v1")
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated subset of variants",
    )
    args = ap.parse_args()
    names = [x.strip() for x in args.variants.split(",") if x.strip()]

    all_rows: list[dict[str, Any]] = []
    if args.leg in ("offline", "both"):
        all_rows.extend(_run_offline(args.start_date, args.end_date, args.tag, names))
    if args.leg in ("fused", "both"):
        # fused stress: focus on extend / mid / conditional
        fused_names = [n for n in names if n in {
            "peer3_extend",
            "ladder_tight_v1",
            "ladder_mid_8m",
            "ladder_loose_10m",
            "ladder_cond_mid_wash",
            "ladder_cond_loose_wash",
        }]
        if not fused_names:
            fused_names = names
        all_rows.extend(_run_fused(Path(args.session_dir), args.tag, fused_names))

    print("=== GRID FINAL ===", flush=True)
    print(json.dumps(all_rows, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
