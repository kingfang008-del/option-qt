#!/usr/bin/env python3
"""Ablation: peer3 entries × ladder_active exits vs peer3 hold_extend.

Two legs:
  1) 2026-07-20 fused Redis session (no disk 1s quotes that day)
  2) offline May–Jul (quotes through 07-17) — same peer3 signal, exit swap only

Usage:
  python -m maga7.tools.run_sec_ladder_ablation --leg fused
  python -m maga7.tools.run_sec_ladder_ablation --leg offline
  python -m maga7.tools.run_sec_ladder_ablation --leg both
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
SEC_PROF = (
    ROOT
    / "maga7/CONFIG/strategy_profiles/"
    / "single_qqq_open_ladder_atm5otm_sec_ladder_active_v1.json"
)
DEFAULT_SESSION = Path(
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e"
)

# Exit-only overlay: keep peer3 signal / watchdog / hunter.
LADDER_EXIT_KEYS = (
    "exit_mode",
    "hold_minutes",
    "hold_extend_minutes",
    "hold_extend_mtm_min",
    "hold_extend_require_mf",
    "exit_mf_grace_seconds",
    "early_exit_mode",
    "ladder_active",
    "trade_toxic",
    "delta_time_stop",
    "roi_time_stop",
)


def _ladder_exit_overlay() -> dict[str, Any]:
    sec = load_profile(SEC_PROF)
    trade = sec.get("trade") or {}
    out: dict[str, Any] = {}
    for k in LADDER_EXIT_KEYS:
        if k in trade:
            out[k] = deepcopy(trade[k])
    # Explicit: no extend on this path.
    out["exit_mode"] = "ladder_active"
    out["hold_extend_minutes"] = None
    out["early_exit_mode"] = None
    return out


def _trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0,
            "sum_ret": 0.0,
            "mean_ret": float("nan"),
            "win_rate": float("nan"),
            "mean_hold_sec": float("nan"),
            "reasons": {},
        }
    ret = pd.to_numeric(trades["ret"], errors="coerce")
    hold_sec = None
    if "entry_ts" in trades.columns and "exit_ts" in trades.columns:
        et = pd.to_datetime(trades["entry_ts"], utc=True, errors="coerce")
        xt = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
        hold_sec = (xt - et).dt.total_seconds()
    reasons = (
        trades["reason"].astype(str).value_counts().to_dict()
        if "reason" in trades.columns
        else {}
    )
    return {
        "n_trades": int(len(trades)),
        "sum_ret": float(ret.sum()),
        "mean_ret": float(ret.mean()),
        "win_rate": float((ret > 0).mean()),
        "mean_hold_sec": float(hold_sec.mean()) if hold_sec is not None else float("nan"),
        "median_hold_sec": float(hold_sec.median()) if hold_sec is not None else float("nan"),
        "reasons": reasons,
    }


def _run_offline(start: str, end: str, tag: str) -> list[dict[str, Any]]:
    base = load_profile(PEER3)
    base["date_range"] = {"start": start, "end": end}
    results_dir = Path(base["_paths"]["results_dir"]) / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    overlay = _ladder_exit_overlay()
    variants = {
        "peer3_extend": {},
        "peer3_ladder_exit": overlay,
    }
    rows: list[dict[str, Any]] = []
    for name, ov in variants.items():
        prof = deepcopy(base)
        trade = prof.setdefault("trade", {})
        for k, v in ov.items():
            trade[k] = v
        out = results_dir / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== offline {name} {start}..{end} → {out} ===", flush=True)
        result = run_offline_replay(prof, scheme="single")
        summary = result["summary"]
        trades = result["trades"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        trades.to_csv(out / "trades.csv", index=False)
        result["daily"].to_csv(out / "daily.csv", index=False)
        m = {
            "variant": name,
            "leg": "offline",
            "start": start,
            "end": end,
            "total_ret": summary.get("total_ret"),
            "maxdd": summary.get("maxdd"),
            "day_win": summary.get("day_win"),
            "trade_win": summary.get("trade_win"),
            "n_trades": summary.get("n_trades"),
            "out": str(out),
            **{f"tm_{k}": v for k, v in _trade_metrics(trades).items() if k != "reasons"},
            "reasons": _trade_metrics(trades)["reasons"],
        }
        rows.append(m)
        print(json.dumps({k: v for k, v in m.items() if k != "reasons"}, indent=2, default=str), flush=True)
        print(f"reasons={m['reasons']}", flush=True)
    (results_dir / "compare.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in rows]).to_csv(
        results_dir / "compare.csv", index=False
    )
    print(f"wrote {results_dir / 'compare.json'}", flush=True)
    return rows


def _run_fused(session_dir: Path, tag: str) -> list[dict[str, Any]]:
    from maga7.tools.run_live_fused_replay import run_fused_replay

    overlay = _ladder_exit_overlay()
    variants = {
        "peer3_extend": None,
        "peer3_ladder_exit": overlay,
    }
    rows: list[dict[str, Any]] = []
    out_root = session_dir / tag
    out_root.mkdir(parents=True, exist_ok=True)
    for name, ov in variants.items():
        print(f"=== fused {name} → tag={tag}_{name} ===", flush=True)
        summary = run_fused_replay(
            session_dir,
            scheme="single",
            disable_prevention=True,
            tag=f"{tag}_{name}",
            trade_overrides=ov,
            profile_path_override=str(PEER3),
        )
        trades_path = session_dir / f"fused_replay_{tag}_{name}" / "trades.csv"
        trades = pd.read_csv(trades_path) if trades_path.is_file() else pd.DataFrame()
        tm = _trade_metrics(trades)
        m = {
            "variant": name,
            "leg": "fused_20260720",
            "total_ret": summary.get("total_ret"),
            "maxdd": summary.get("maxdd"),
            "n_trades": summary.get("n_trades"),
            "trade_win": summary.get("trade_win"),
            "n_scanner_signals": summary.get("n_scanner_signals"),
            "vs_live": summary.get("vs_live"),
            "out": str(session_dir / f"fused_replay_{tag}_{name}"),
            **{f"tm_{k}": v for k, v in tm.items() if k != "reasons"},
            "reasons": tm["reasons"],
        }
        rows.append(m)
        print(json.dumps({k: v for k, v in m.items() if k not in {"reasons", "vs_live"}}, indent=2, default=str), flush=True)
        print(f"reasons={m['reasons']}", flush=True)
        if trades_path.is_file():
            print(trades.to_string(index=False), flush=True)
    (out_root / "compare.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_root / 'compare.json'}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", choices=("fused", "offline", "both"), default="both")
    ap.add_argument("--session-dir", default=str(DEFAULT_SESSION))
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--tag", default="sec_ladder_ablation_v1")
    args = ap.parse_args()

    all_rows: list[dict[str, Any]] = []
    if args.leg in ("fused", "both"):
        all_rows.extend(_run_fused(Path(args.session_dir), args.tag))
    if args.leg in ("offline", "both"):
        all_rows.extend(_run_offline(args.start_date, args.end_date, args.tag))

    print("=== FINAL COMPARE ===", flush=True)
    print(json.dumps(all_rows, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
