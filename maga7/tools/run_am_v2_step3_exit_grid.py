#!/usr/bin/env python3
"""AM v2 Step3: exit grid on frozen Step2b winner — quote FillSpec only.

Frozen signal: launch_s3_r002_cd300_post10
  |ret|≥0.2%, slope=3s, cd=300s, window 10:00–11:30

Grid (no ride / scaleout):
  TP ∈ {0.10, 0.15, 0.20}
  SL ∈ {0.15, 0.20, 0.25}
  hold ∈ {600, 900, 1200}

PASS: ≥1 cell with quote dual-window econ; promote best by quote mean then disc.
FAIL: keep Step2b signal, try alternate simple exits only — still no ride stack.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_step3_exit_grid \\
    --tag research_am_v2_step3_exit_grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_otm_rungs
from maga7.common.stock_1s import session_dates
from maga7.tools.run_am_v2_step2_signal_bakeoff import collect_launch, score_signals
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of

PROFILE = "maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json"
SPINE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
SIGNAL = "launch_s3_r002_cd300_post10"
TPS = (0.10, 0.15, 0.20)
SLS = (0.15, 0.20, 0.25)
HOLDS = (600, 900, 1200)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_step3_exit_grid")
    ap.add_argument("--trades-root", default="/mnt/s990/new_option_data_s3_trades")
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

    print(f"am_v2 step3 exit grid days={len(dates)} signal={SIGNAL}", flush=True)
    print("collect frozen launch signals once…", flush=True)
    sigs = collect_launch(
        name=SIGNAL,
        dates=dates,
        symbols=symbols,
        stock_1s=stock_1s,
        window_start="10:00",
        window_end="11:30",
        abs_ret_min=0.002,
        cooldown_sec=300,
    )
    print(f"  n_signals={len(sigs)}", flush=True)

    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    rows = []
    for tp in TPS:
        for sl in SLS:
            for h in HOLDS:
                name = f"tp{int(tp*100):02d}_sl{int(sl*100):02d}_h{h}"
                print(f"score {name}…", flush=True)
                st = score_signals(
                    sigs,
                    lock=lock,
                    otm=otm,
                    quote_root=quote_root,
                    trades_root=trades_root,
                    fill=fill,
                    tp=float(tp),
                    sl=float(sl),
                    max_hold=int(h),
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
                    "exit": name,
                    "tp": tp,
                    "sl": sl,
                    "max_hold_sec": h,
                    "fill_rate": st["fill_rate"],
                    "quote_n": q["n"],
                    "quote_tpd": q["tpd"],
                    "quote_win": q["trade_win"],
                    "quote_mean": q["mean_ret"],
                    "quote_disc": q["disc_compound"],
                    "quote_blind": q["blind_compound"],
                    "quote_econ": q["econ_dual"],
                    "quote_frac_tp": q.get("frac_tp"),
                    "quote_hold_p50": q.get("hold_p50"),
                    "trade_n": t["n"],
                    "trade_mean": t["mean_ret"],
                    "trade_disc": t["disc_compound"],
                    "trade_blind": t["blind_compound"],
                    "trade_econ": t["econ_dual"],
                }
                rows.append(row)
                print(
                    f"  quote n={q['n']} mean={q['mean_ret']} "
                    f"disc={q['disc_compound']:+.3f} blind={q['blind_compound']:+.3f} "
                    f"econ={q['econ_dual']}",
                    flush=True,
                )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    econ = sb[sb.quote_econ == True].copy()  # noqa: E712
    econ = econ.sort_values(
        ["quote_mean", "quote_disc", "quote_blind"], ascending=[False, False, False]
    )
    promote = "NONE"
    best = None
    if len(econ):
        best = econ.iloc[0].to_dict()
        promote = f"STEP3_{best['exit']}"

    baseline = sb[(sb.tp == 0.15) & (sb.sl == 0.20) & (sb.max_hold_sec == 900)]
    baseline_row = baseline.iloc[0].to_dict() if len(baseline) else None

    summary = {
        "protocol": "am_v2_step3_exit_grid",
        "step": 3,
        "promotion_mark": "quote_FillSpec",
        "frozen_signal": SIGNAL,
        "n_signals": len(sigs),
        "grid": {"tp": list(TPS), "sl": list(SLS), "hold": list(HOLDS)},
        "n_econ_quote": int(len(econ)),
        "promote": promote,
        "best_quote_econ": best,
        "baseline_tp15_sl20_h900": baseline_row,
        "scoreboard": sb.to_dict(orient="records"),
        "pass": bool(promote != "NONE"),
        "next_step": 4 if promote != "NONE" else "3b_alternate_simple_exits",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# AM v2 Step3 — Exit grid",
        "",
        f"- frozen signal: `{SIGNAL}`",
        f"- n_signals: {len(sigs)}",
        "- mark: **quote FillSpec**",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}** → next `{summary['next_step']}`",
        f"- cells with quote econ: {len(econ)} / {len(sb)}",
        "",
        "## Top quote-econ cells",
        "",
    ]
    top = econ.head(8) if len(econ) else sb.sort_values("quote_mean", ascending=False).head(8)
    try:
        lines.append(top.to_markdown(index=False))
    except Exception:
        lines.append(top.to_string(index=False))
    lines += ["", "## Full scoreboard", ""]
    try:
        lines.append(sb.to_markdown(index=False))
    except Exception:
        lines.append(sb.to_string(index=False))
    if promote != "NONE":
        lines += [
            "",
            "## 结论",
            "",
            f"**Step3 PASS** → freeze exit `{best['exit']}` → Step4 shadow 接线。",
            "不叠 ride/scaleout。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**Step3 FAIL** — 网格无 quote 双窗 econ。不叠 ride；可试 3b 简单替代退出。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== TOP ECON ===", flush=True)
    print((econ.head(10) if len(econ) else sb.head(10)).to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "pass": summary["pass"], "next": summary["next_step"]}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
