#!/usr/bin/env python3
"""Causal portfolio fill from session foresight events (trade last, fixed H).

Reads ``events.parquet`` from ``scan_session_horizon_foresight`` (already priced
with ``/mnt/s990/new_option_data_s3_trades``). Keeps one horizon (default 120s),
uses **clock_ret** as fill PnL (causal clock exit — not oracle).

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_trades_fill \\
    --events-tag research_session_horizon_foresight_apr_jul \\
    --tag research_session_am_h120_fill_apr_jul \\
    --session AM_0930_1000 --horizon-sec 120
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
from maga7.common.replay import to_ny
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _load_events(results_dir: Path, tag: str) -> pd.DataFrame:
    p = results_dir / tag / "events.parquet"
    if p.is_file():
        return pd.read_parquet(p)
    c = results_dir / tag / "events.csv"
    if c.is_file():
        return pd.read_csv(c)
    raise SystemExit(f"missing events under {results_dir / tag}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--session",
        required=True,
        choices=["AM_0930_1000", "CORE_1030_1130", "MID_1230_1330"],
    )
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument(
        "--cooldown-minutes",
        type=float,
        default=2.0,
        help="Per-symbol re-entry cooldown after exit (default = hold ≈2m).",
    )
    ap.add_argument(
        "--max-per-symbol-day",
        type=int,
        default=0,
        help="Cap entries per symbol per day; 0 = unlimited (opportunity mode).",
    )
    ap.add_argument(
        "--max-per-session-day",
        type=int,
        default=0,
        help="Cap total entries per session day; 0 = unlimited (opportunity mode).",
    )
    ap.add_argument(
        "--pick",
        choices=["first", "strongest"],
        default="first",
        help="Order before portfolio gate: chronological first (causal) or strongest |stock_ret_lb|.",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    results_dir = Path(prof["_paths"]["results_dir"])
    events = _load_events(results_dir, args.events_tag)
    h = int(args.horizon_sec)
    sub = events[
        (events["session"].astype(str) == args.session)
        & (events["horizon_sec"].astype(int) == h)
    ].copy()
    if sub.empty:
        raise SystemExit(f"no events for {args.session} H={h}")

    raw: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        et = to_ny(r["entry_ts"])
        xt = et + pd.Timedelta(seconds=h)
        ret = float(r["clock_ret"])
        if not np.isfinite(ret):
            continue
        raw.append(
            {
                "date": str(r["date"]),
                "symbol": str(r["symbol"]),
                "dir": str(r["dir"]),
                "sig_ts": str(et),
                "entry_ts": str(et),
                "exit_ts": str(xt),
                "ticker": str(r.get("ticker") or ""),
                "dte": r.get("dte"),
                "ret": ret,
                "reason": f"clock_H{h}",
                "session": args.session,
                "horizon_sec": h,
                "stock_ret_lb": float(r["stock_ret_lb"]) if pd.notna(r.get("stock_ret_lb")) else None,
                "sleeve": (
                    "AM"
                    if args.session.startswith("AM")
                    else ("CORE" if args.session.startswith("CORE") else "MID")
                ),
            }
        )

    # Optional hard caps (0 = opportunity: take every signal that clears portfolio gates).
    max_sess = int(args.max_per_session_day)
    max_sym = int(args.max_per_symbol_day)
    picked: list[dict] = []
    by_date: dict[str, list[dict]] = {}
    for tr in raw:
        by_date.setdefault(str(tr["date"]), []).append(tr)
    for date in sorted(by_date):
        rows = by_date[date]
        if args.pick == "strongest":
            rows = sorted(
                rows,
                key=lambda r: (
                    -abs(float(r["stock_ret_lb"] or 0.0)),
                    str(r["entry_ts"]),
                    str(r["symbol"]),
                ),
            )
        else:
            rows = sorted(rows, key=lambda r: (str(r["entry_ts"]), str(r["symbol"])))
        n_sess = 0
        for tr in rows:
            if max_sess > 0 and n_sess >= max_sess:
                break
            sym = str(tr["symbol"])
            if max_sym > 0:
                sym_n = sum(1 for x in picked if x["date"] == date and x["symbol"] == sym)
                if sym_n >= max_sym:
                    continue
            picked.append(tr)
            n_sess += 1

    by_day: dict[str, list[dict]] = {}
    for tr in picked:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    sized: list[dict] = []
    for _, rows in sorted(by_day.items()):
        sized.extend(
            _portfolio_day(
                rows,
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=float(args.cooldown_minutes),
            )
        )
    trades_df = pd.DataFrame(sized)
    # _portfolio_day uses size/pnl_frac; equity helper expects pnl_frac
    if len(trades_df) and "pnl_frac" not in trades_df.columns and "size" in trades_df.columns:
        trades_df["pnl_frac"] = trades_df["ret"].astype(float) * trades_df["size"].astype(float)
    if len(trades_df) and "size_frac" not in trades_df.columns and "size" in trades_df.columns:
        trades_df["size_frac"] = trades_df["size"]

    stats = _equity_stats(trades_df)
    # Honest capacity metrics (compound alone can look absurd with dense recycling).
    if len(trades_df):
        pnl = trades_df["pnl_frac"].astype(float)
        day = trades_df.groupby(trades_df["date"].astype(str))["pnl_frac"].sum()
        additive = float(pnl.sum())
        day_eq = 1.0
        peak = 1.0
        day_mdd = 0.0
        for x in day.sort_index():
            day_eq *= 1.0 + float(x)
            peak = max(peak, day_eq)
            day_mdd = min(day_mdd, day_eq / peak - 1.0)
        honest = {
            "sum_pnl_frac_additive": additive,
            "day_compound_ret": float(day_eq - 1.0),
            "day_compound_maxdd": float(day_mdd),
            "day_win": float((day > 0).mean()),
            "day_mean_pnl": float(day.mean()),
            "trades_per_day": float(len(trades_df) / max(int(trades_df["date"].nunique()), 1)),
            "trade_mean": float(trades_df["ret"].astype(float).mean()),
            "trade_med": float(trades_df["ret"].astype(float).median()),
        }
    else:
        honest = {
            "sum_pnl_frac_additive": 0.0,
            "day_compound_ret": 0.0,
            "day_compound_maxdd": 0.0,
            "day_win": None,
            "day_mean_pnl": 0.0,
            "trades_per_day": 0.0,
            "trade_mean": None,
            "trade_med": None,
        }
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if len(trades_df):
        trades_df.to_csv(out / "trades.csv", index=False)
    summary = {
        "events_tag": args.events_tag,
        "session": args.session,
        "horizon_sec": h,
        "pricing": "option_trades_last_slip_from_foresight_clock",
        "mode": "opportunity" if max_sess <= 0 and max_sym <= 0 else "capped",
        "position_frac": float(args.position_frac),
        "max_concurrent": int(args.max_concurrent),
        "cooldown_minutes": float(args.cooldown_minutes),
        "n_signals": int(len(sub)),
        "n_raw": int(len(raw)),
        "n_picked": int(len(picked)),
        "n_trades": int(len(trades_df)),
        "pick": args.pick,
        "max_per_symbol_day": int(args.max_per_symbol_day),
        "max_per_session_day": int(args.max_per_session_day),
        **stats,
        **honest,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        f"[{args.session}] H={h} mode={summary['mode']} n={stats.get('n_trades')} "
        f"tpd={honest.get('trades_per_day'):.1f} trade_mean={honest.get('trade_mean')} "
        f"additive={honest.get('sum_pnl_frac_additive'):+.3f} "
        f"day_comp={honest.get('day_compound_ret'):+.3f} "
        f"compound={stats.get('total_ret', 0):+.3f} mdd={stats.get('maxdd', 0):+.3f} "
        f"win={stats.get('trade_win')}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
