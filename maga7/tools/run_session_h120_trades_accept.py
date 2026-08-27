#!/usr/bin/env python3
"""Dual-window accept for AM 09:30–10:00 / MID 12:30–13:30 H=120 trades sleeves.

Opportunity mode (default): every causal signal that clears seats/cooldown —
no fixed N-trades/day cap. Verdict uses **day_compound_ret** (and additive)
as primary; full compound equity is reported but not the promote gate.

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_trades_accept
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

CORE_PROF = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SESSIONS = ("AM_0930_1000", "MID_1230_1330")
H = 120


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _results_dir() -> Path:
    return Path(load_profile(CORE_PROF)["_paths"]["results_dir"])


def _ensure_foresight(tag: str, start: str, end: str, py: str) -> None:
    out = _results_dir() / tag
    if (out / "events.parquet").is_file() or (out / "events.csv").is_file():
        print(f"[skip-scan] {tag}", flush=True)
        return
    _run(
        [
            py,
            "-m",
            "maga7.tools.scan_session_horizon_foresight",
            "--start-date",
            start,
            "--end-date",
            end,
            "--tag",
            tag,
            "--stride-sec",
            "120",
            "--lookback-sec",
            "60",
            "--horizons",
            "30,60,90,120,180,300,450,600,900",
        ]
    )


def _ensure_fill(events_tag: str, fill_tag: str, session: str, py: str) -> dict[str, Any]:
    out = _results_dir() / fill_tag
    summary_p = out / "summary.json"
    if summary_p.is_file() and (out / "trades.csv").is_file():
        print(f"[skip-fill] {fill_tag}", flush=True)
        return json.loads(summary_p.read_text(encoding="utf-8"))
    _run(
        [
            py,
            "-m",
            "maga7.tools.run_session_h120_trades_fill",
            "--events-tag",
            events_tag,
            "--tag",
            fill_tag,
            "--session",
            session,
            "--horizon-sec",
            str(H),
            "--position-frac",
            "0.10",
            "--max-concurrent",
            "2",
            "--cooldown-minutes",
            "2",
            "--max-per-symbol-day",
            "0",
            "--max-per-session-day",
            "0",
            "--pick",
            "first",
        ]
    )
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def _primary_ret(row: dict[str, Any]) -> float:
    """Prefer day-compound; fall back to additive then full compound."""
    if row.get("day_compound_ret") is not None:
        return float(row["day_compound_ret"])
    if row.get("sum_pnl_frac_additive") is not None:
        return float(row["sum_pnl_frac_additive"])
    return float(row.get("total_ret") or 0.0)


def _decide(strong: dict[str, Any], weak: dict[str, Any], sleeve: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    sr, wr = _primary_ret(strong), _primary_ret(weak)
    flags.append("strong_profit" if sr > 0 else "strong_loss")
    if wr > 0:
        flags.append("weak_profit")
    elif wr > -0.05:
        flags.append("weak_flatish")
    else:
        flags.append("weak_loss")
    # prefer day_compound_maxdd; else trade-stream maxdd
    mdd = strong.get("day_compound_maxdd")
    if mdd is None:
        mdd = strong.get("maxdd") or 0.0
    flags.append("strong_dd_ok" if float(mdd) > -0.20 else "strong_dd_fail")
    if "strong_profit" in flags and "weak_profit" in flags and "strong_dd_ok" in flags:
        return f"PROMOTE_{sleeve}_SLEEVE", flags
    if "strong_profit" in flags and "strong_dd_ok" in flags and "weak_loss" not in flags:
        return "OVERLAY_ONLY", flags
    return f"REJECT_FOR_{sleeve}_SLEEVE", flags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/session_h120_trades_accept_opp_v1")
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
        (
            "strong",
            args.strong_start,
            args.strong_end,
            "research_session_horizon_foresight_apr_jul",
        ),
        (
            "weak",
            args.weak_start,
            args.weak_end,
            "research_session_horizon_foresight_jan_mar",
        ),
    ]

    all_rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}

    for wname, start, end, ev_tag in windows:
        _ensure_foresight(ev_tag, start, end, py)
        for sess in SESSIONS:
            sleeve = "AM" if sess.startswith("AM") else "MID"
            fill_tag = (
                f"research_session_{sleeve.lower()}_h120_opp_fill_"
                f"{'apr_jul' if wname == 'strong' else 'jan_mar'}"
            )
            stats = _ensure_fill(ev_tag, fill_tag, sess, py)
            row = {
                "window": wname,
                "session": sess,
                "sleeve": sleeve,
                "start": start,
                "end": end,
                "events_tag": ev_tag,
                "fill_tag": fill_tag,
                "mode": stats.get("mode"),
                "n_trades": int(stats.get("n_trades") or 0),
                "trades_per_day": stats.get("trades_per_day"),
                "trade_win": stats.get("trade_win"),
                "trade_mean": stats.get("trade_mean"),
                "sum_pnl_frac_additive": float(stats.get("sum_pnl_frac_additive") or 0.0),
                "day_compound_ret": float(stats.get("day_compound_ret") or 0.0),
                "day_compound_maxdd": float(stats.get("day_compound_maxdd") or 0.0),
                "day_win": stats.get("day_win"),
                "total_ret": float(stats.get("total_ret") or 0.0),
                "maxdd": float(stats.get("maxdd") or 0.0),
                "exp": stats.get("exp"),
            }
            all_rows.append(row)
            print(
                f"[{wname}/{sess}] tpd={row['trades_per_day']} mean={row['trade_mean']} "
                f"additive={row['sum_pnl_frac_additive']:+.3f} "
                f"day_comp={row['day_compound_ret']:+.3f} "
                f"compound={row['total_ret']:+.3f} n={row['n_trades']} win={row['trade_win']}",
                flush=True,
            )

    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)

    for sess in SESSIONS:
        sleeve = "AM" if sess.startswith("AM") else "MID"
        strong = rdf[(rdf.window == "strong") & (rdf.session == sess)].iloc[0].to_dict()
        weak = rdf[(rdf.window == "weak") & (rdf.session == sess)].iloc[0].to_dict()
        decision, flags = _decide(strong, weak, sleeve)
        decisions[sess] = {
            "decision": decision,
            "flags": flags,
            "strong": strong,
            "weak": weak,
            "profile": (
                "maga7/CONFIG/strategy_profiles/session_am_0930_1000_h120_trades_v1.json"
                if sleeve == "AM"
                else "maga7/CONFIG/strategy_profiles/session_mid_1230_1330_h120_trades_v1.json"
            ),
        }
        print(f"\n=== {sess} decision: {decision} flags={flags} ===", flush=True)

    summary = {
        "horizon_sec": H,
        "pricing": "new_option_data_s3_trades",
        "lookback_sec": 60,
        "stride_sec": 120,
        "decisions": decisions,
        "windows": all_rows,
        "note": (
            "Opportunity mode: no fixed N/day. Entry dir = causal 60s stock lookback; "
            "hold = H=120 clock on trade last ±slip; seats≤2, per-symbol cooldown=2m. "
            "Promote gate uses day_compound_ret (not full intra-day compound)."
        ),
    }
    (out / "accept_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
