#!/usr/bin/env python3
"""AM v2 Step2b: next signal family bakeoff — quote FillSpec only.

After Step2 FAIL (pulse fo08 / launch r002 cd120), try:
  1) launch_s3_r003_cd120       stricter |ret|≥0.3%
  2) launch_s3_r002_cd300_post10 longer cd + quote-healthy window
  3) pulse_fo12_causal_post10   higher FO band 1.2–2.0%, post-10:00

Same fixed exit/gates as Step2. trade-last diagnostic only.

PASS: dual-window quote econ → promote to Step3.
FAIL: stay on 2b/2c; do not loosen mark.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_step2b_signal_bakeoff \\
    --tag research_am_v2_step2b_signal_bakeoff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_otm_rungs
from maga7.common.stock_1s import load_symbol_1s_bars, session_dates
from maga7.tools.run_am_v2_step2_signal_bakeoff import (
    collect_launch,
    collect_pulse,
    score_signals,
)
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of

PROFILE = "maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json"
SPINE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_step2b_signal_bakeoff")
    ap.add_argument("--trades-root", default="/mnt/s990/new_option_data_s3_trades")
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.20)
    ap.add_argument("--max-hold-sec", type=int, default=900)
    ap.add_argument("--max-lag-sec", type=float, default=5.0)
    ap.add_argument("--max-spread-pct", type=float, default=0.15)
    ap.add_argument("--min-mid", type=float, default=0.05)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=1.0)
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args(argv)

    v2 = load_profile(args.profile)
    spine = load_profile(args.spine)
    paths = spine["_paths"]
    out = Path(paths["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    stock_1s = Path(paths["stock_1s_root"])
    quote_root = Path(paths["quote_1s_root"])
    trades_root = Path(args.trades_root)
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]).expanduser())
    otm = resolve_otm_rungs(spine, default=3)
    symbols = list(v2.get("symbols") or spine.get("symbols") or [])
    dates = [
        d
        for d in session_dates(args.start_date, args.end_date)
        if args.start_date <= d <= args.end_date and _window_of(d) is not None
    ]
    if int(args.max_days) > 0:
        dates = dates[: int(args.max_days)]

    print(f"am_v2 step2b bakeoff days={len(dates)} syms={len(symbols)}", flush=True)
    print("loading 1s→1m bars…", flush=True)
    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_symbol_1s_bars(stock_1s, sym, dates, bar_seconds=60)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf
            print(f"  {sym}: bars={len(sdf)}", flush=True)

    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    candidates: list[tuple[str, list[dict[str, Any]]]] = []

    print("collect launch_s3_r003_cd120…", flush=True)
    candidates.append(
        (
            "launch_s3_r003_cd120",
            collect_launch(
                name="launch_s3_r003_cd120",
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="11:30",
                abs_ret_min=0.003,
                cooldown_sec=120,
            ),
        )
    )
    print("collect launch_s3_r002_cd300_post10…", flush=True)
    candidates.append(
        (
            "launch_s3_r002_cd300_post10",
            collect_launch(
                name="launch_s3_r002_cd300_post10",
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                window_start="10:00",
                window_end="11:30",
                abs_ret_min=0.002,
                cooldown_sec=300,
            ),
        )
    )
    print("collect pulse_fo12_causal_post10…", flush=True)
    candidates.append(
        (
            "pulse_fo12_causal_post10",
            collect_pulse(
                name="pulse_fo12_causal_post10",
                dates=dates,
                symbols=symbols,
                stock_by_sym=stock_by_sym,
                stock_1s=stock_1s,
                window_start="10:00",
                window_end="11:30",
                min_fav_from_open=0.012,
                max_fav_from_open=0.020,
            ),
        )
    )

    rows = []
    for name, sigs in candidates:
        print(f"score {name}: signals={len(sigs)}", flush=True)
        st = score_signals(
            sigs,
            lock=lock,
            otm=otm,
            quote_root=quote_root,
            trades_root=trades_root,
            fill=fill,
            tp=float(args.tp),
            sl=float(args.sl),
            max_hold=int(args.max_hold_sec),
            max_lag=float(args.max_lag_sec),
            max_spread=float(args.max_spread_pct),
            min_mid=float(args.min_mid),
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=float(args.cooldown_minutes),
        )
        q = st["quote"]
        t = st["trade_diag"]
        row = {
            "signal": name,
            "n_signals": st["n_signals"],
            "fill_rate": st["fill_rate"],
            "n_gate_fail": st["n_gate_fail"],
            "n_no_quote": st["n_no_quote"],
            "quote_n": q["n"],
            "quote_tpd": q["tpd"],
            "quote_win": q["trade_win"],
            "quote_mean": q["mean_ret"],
            "quote_disc": q["disc_compound"],
            "quote_blind": q["blind_compound"],
            "quote_econ": q["econ_dual"],
            "trade_n": t["n"],
            "trade_mean": t["mean_ret"],
            "trade_disc": t["disc_compound"],
            "trade_blind": t["blind_compound"],
            "trade_econ": t["econ_dual"],
        }
        rows.append(row)
        print(
            f"  quote n={q['n']} tpd={q['tpd']:.2f} win={q['trade_win']} "
            f"mean={q['mean_ret']} disc={q['disc_compound']:+.3f} "
            f"blind={q['blind_compound']:+.3f} econ={q['econ_dual']}",
            flush=True,
        )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    econ = sb[sb.quote_econ == True].copy()  # noqa: E712
    econ = econ.sort_values(["quote_mean", "quote_disc"], ascending=[False, False])
    promote = "NONE"
    best = None
    if len(econ):
        best = econ.iloc[0].to_dict()
        promote = f"STEP2B_{best['signal']}"
    elif len(sb):
        soft = sb[(sb.quote_disc > 0) | (sb.quote_blind > 0)].copy()
        if len(soft):
            soft = soft.sort_values("quote_mean", ascending=False)
            best = soft.iloc[0].to_dict()

    summary = {
        "protocol": "am_v2_step2b_signal_bakeoff",
        "step": "2b",
        "promotion_mark": "quote_FillSpec",
        "exit": f"tp{args.tp}_sl{args.sl}_h{args.max_hold_sec}",
        "gate": {
            "max_lag_sec": float(args.max_lag_sec),
            "max_spread_pct": float(args.max_spread_pct),
            "min_mid": float(args.min_mid),
        },
        "candidates": [c[0] for c in candidates],
        "n_econ_quote": int(len(econ)),
        "promote": promote,
        "best_quote_econ": best if promote != "NONE" else None,
        "best_soft": best if promote == "NONE" else None,
        "scoreboard": sb.to_dict(orient="records"),
        "pass": bool(promote != "NONE"),
        "next_step": 3 if promote != "NONE" else "2c_try_other_families_or_stop",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# AM v2 Step2b — Signal bakeoff",
        "",
        f"- exit: TP{args.tp:g}/SL{args.sl:g}/h{args.max_hold_sec}",
        "- mark: **quote FillSpec** (trade-last diagnostic only)",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}** → next `{summary['next_step']}`",
        "",
        "## Candidates",
        "",
        "1. `launch_s3_r003_cd120` — |ret|≥0.3%, cd=120, 09:30–11:30",
        "2. `launch_s3_r002_cd300_post10` — |ret|≥0.2%, cd=300, 10:00–11:30",
        "3. `pulse_fo12_causal_post10` — FO 1.2–2.0%, 10:00–11:30",
        "",
        "## Scoreboard",
        "",
    ]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    if promote == "NONE":
        lines += [
            "",
            "## 结论",
            "",
            "**Step2b FAIL.** 不进 Step3。不放宽 mark。",
            "下一步：Step2c 换族（非 pulse/launch 变体），或暂停 AM 卫星信号搜索。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            f"**Step2b PASS** → promote `{promote}` → 进入 Step3 退出网格。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SCOREBOARD ===", flush=True)
    print(sb.to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"], "next": summary["next_step"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
