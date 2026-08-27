#!/usr/bin/env python3
"""Dual-window accept for AM launch_slope sleeve v1 (open s3/H120 peer3).

Steps per window:
  1) scan_morning_launch_slope (open session only) if events missing
  2) run_morning_launch_option_fill for s3_r002_h120_fp0_p3 / horizon
  3) drop signals with sig_ts ≥ 10:25 (CORE mutex) and rebuild equity

Pass: strong total_ret>0 AND weak total_ret>0 AND strong maxdd > −0.20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile

CAND = "s3_r002_h120_fp0_p3"
EXIT_BOOK = "horizon"
AM_CUTOFF = "10:25"
CORE_PROF = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _results_dir() -> Path:
    return Path(load_profile(CORE_PROF)["_paths"]["results_dir"])


def _ensure_events(tag: str, start: str, end: str, py: str) -> None:
    out = _results_dir() / tag
    if (out / "events.parquet").is_file() or (out / "events.csv").is_file():
        print(f"[skip-scan] {tag}", flush=True)
        return
    _run(
        [
            py,
            "-m",
            "maga7.tools.scan_morning_launch_slope",
            "--start-date",
            start,
            "--end-date",
            end,
            "--tag",
            tag,
            "--sessions",
            "open_0930_1030",
        ]
    )


def _ensure_fill(events_tag: str, fill_tag: str, py: str) -> Path:
    out = _results_dir() / fill_tag
    summary = out / f"{CAND}__{EXIT_BOOK}" / "summary.json"
    if summary.is_file():
        print(f"[skip-fill] {fill_tag}", flush=True)
        return out
    _run(
        [
            py,
            "-m",
            "maga7.tools.run_morning_launch_option_fill",
            "--events-tag",
            events_tag,
            "--tag",
            fill_tag,
            "--candidates",
            CAND,
            "--exit-books",
            EXIT_BOOK,
            "--position-frac",
            "0.10",
        ]
    )
    return out


def _equity_tod_filtered(fill_root: Path) -> dict[str, Any]:
    tr_p = fill_root / f"{CAND}__{EXIT_BOOK}" / "trades.csv"
    raw_p = fill_root / f"{CAND}__{EXIT_BOOK}" / "summary.json"
    raw = json.loads(raw_p.read_text(encoding="utf-8")) if raw_p.is_file() else {}
    if not tr_p.is_file():
        return {
            "total_ret": float(raw.get("total_ret") or 0.0),
            "maxdd": float(raw.get("maxdd") or 0.0),
            "n_trades": int(raw.get("n_trades") or 0),
            "trade_win": raw.get("trade_win"),
            "n_dropped_after_1025": 0,
        }
    tr = pd.read_csv(tr_p)
    if tr.empty:
        return {"total_ret": 0.0, "maxdd": 0.0, "n_trades": 0, "trade_win": None, "n_dropped_after_1025": 0}
    # sig_ts may be tz-aware strings with mixed offsets → parse per-row.
    def _tod_hhmm(x: Any) -> str:
        t = pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_convert("America/New_York")
        return t.strftime("%H:%M")

    tod = tr["sig_ts"].map(_tod_hhmm)
    kept = tr[tod < AM_CUTOFF].copy()
    dropped = int(len(tr) - len(kept))
    if kept.empty:
        return {
            "total_ret": 0.0,
            "maxdd": 0.0,
            "n_trades": 0,
            "trade_win": None,
            "n_dropped_after_1025": dropped,
        }
    if "size_frac" in kept.columns:
        kept = kept.copy()
        kept["_sf"] = pd.to_numeric(kept["size_frac"], errors="coerce").fillna(0.1)
    else:
        kept = kept.copy()
        kept["_sf"] = 0.1
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for _, g in kept.groupby(kept["date"].astype(str), sort=True):
        r = float((g["ret"].astype(float) * g["_sf"].astype(float)).sum())
        eq *= 1.0 + r
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
    return {
        "total_ret": float(eq - 1.0),
        "maxdd": float(maxdd),
        "n_trades": int(len(kept)),
        "trade_win": float((kept["ret"].astype(float) > 0).mean()),
        "n_dropped_after_1025": dropped,
        "total_ret_raw": float(raw.get("total_ret") or 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/am_sleeve_accept_v1")
    ap.add_argument("--strong-start", default="2026-04-01")
    ap.add_argument("--strong-end", default="2026-07-22")
    ap.add_argument("--weak-start", default="2026-01-02")
    ap.add_argument("--weak-end", default="2026-03-31")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    py = args.python
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    windows = [
        ("strong", args.strong_start, args.strong_end, "research_launch_slope_apr_jul_am", "research_am_v1_fill_apr_jul"),
        ("weak", args.weak_start, args.weak_end, "research_launch_slope_jan_mar_am", "research_am_v1_fill_jan_mar"),
    ]
    rows: list[dict[str, Any]] = []
    for wname, start, end, ev_tag, fill_tag in windows:
        _ensure_events(ev_tag, start, end, py)
        fill_root = _ensure_fill(ev_tag, fill_tag, py)
        stats = _equity_tod_filtered(fill_root)
        row = {"window": wname, "start": start, "end": end, "fill_tag": fill_tag, **stats}
        rows.append(row)
        print(
            f"[{wname}] ret={row['total_ret']:+.3f} mdd={row['maxdd']:+.3f} n={row['n_trades']} "
            f"dropped≥10:25={row.get('n_dropped_after_1025')}",
            flush=True,
        )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    strong = rdf[rdf.window == "strong"].iloc[0]
    weak = rdf[rdf.window == "weak"].iloc[0]
    flags = []
    flags.append("strong_profit" if strong["total_ret"] > 0 else "strong_loss")
    if weak["total_ret"] > 0:
        flags.append("weak_profit")
    elif weak["total_ret"] > -0.05:
        flags.append("weak_flatish")
    else:
        flags.append("weak_loss")
    flags.append("strong_dd_ok" if strong["maxdd"] > -0.20 else "strong_dd_fail")
    if "strong_profit" in flags and "weak_profit" in flags and "strong_dd_ok" in flags:
        decision = "PROMOTE_AM_SLEEVE"
    elif "strong_profit" in flags and "strong_dd_ok" in flags and "weak_loss" not in flags:
        decision = "OVERLAY_ONLY"
    else:
        decision = "REJECT_FOR_AM_SLEEVE"
    summary = {
        "candidate": CAND,
        "exit_book": EXIT_BOOK,
        "signal_end_cutoff": AM_CUTOFF,
        "profile": "maga7/CONFIG/strategy_profiles/am_launch_slope_open_s3_h120_peer3_v1.json",
        "decision": decision,
        "flags": flags,
        "windows": rows,
    }
    (out / "accept_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== AM sleeve decision: {decision} flags={flags} ===", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
