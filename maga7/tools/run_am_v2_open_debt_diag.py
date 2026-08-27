#!/usr/bin/env python3
"""AM v2 open-debt diagnostic — 09:30–10:00 only. NOT a promotion track.

Answers: how much upper-bound edge sits in the open half-hour that the
frozen post-10:00 sleeve intentionally skips?

Primary table: option trade-last (``/mnt/s990/new_option_data_s3_trades``)
Appendix:     quote FillSpec (usually sparse — Step1 lag debt)
Exit:         frozen Step3 TP15/SL25/h900 (apples-to-apples vs post10 sleeve)

Candidates:
  1) launch_s3_r002_cd300_open  — same signal as frozen sleeve, open window
  2) launch_s3_r002_cd120_open  — denser cooldown
  3) pulse_fo08_causal_open     — FO pulse in open half-hour

Verdicts are DIAG_* only — never promote / never wire from this tag.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_open_debt_diag \\
    --tag research_am_v2_open_debt_diag
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
# Frozen post10 sleeve reference (Step3)
POST10_REF = {
    "signal": "launch_s3_r002_cd300_post10",
    "exit": "tp15_sl25_h900",
    "quote_n": 286,
    "quote_tpd": 4.77,
    "quote_mean": 0.0106,
    "quote_disc": 0.132,
    "quote_blind": 0.021,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_open_debt_diag")
    ap.add_argument("--trades-root", default="/mnt/s990/new_option_data_s3_trades")
    ap.add_argument("--tp", type=float, default=0.15)
    ap.add_argument("--sl", type=float, default=0.25)
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

    print(
        f"am_v2 open-debt diag days={len(dates)} syms={len(symbols)} "
        f"window=09:30-10:00 exit=tp{args.tp:g}/sl{args.sl:g}/h{args.max_hold_sec}",
        flush=True,
    )
    print("loading 1s→1m bars…", flush=True)
    stock_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sdf = load_symbol_1s_bars(stock_1s, sym, dates, bar_seconds=60)
        if sdf is not None and not sdf.empty:
            stock_by_sym[sym] = sdf
            print(f"  {sym}: bars={len(sdf)}", flush=True)

    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    candidates: list[tuple[str, list[dict[str, Any]]]] = []

    print("collect launch_s3_r002_cd300_open…", flush=True)
    candidates.append(
        (
            "launch_s3_r002_cd300_open",
            collect_launch(
                name="launch_s3_r002_cd300_open",
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="10:00",
                abs_ret_min=0.002,
                cooldown_sec=300,
            ),
        )
    )
    print("collect launch_s3_r002_cd120_open…", flush=True)
    candidates.append(
        (
            "launch_s3_r002_cd120_open",
            collect_launch(
                name="launch_s3_r002_cd120_open",
                dates=dates,
                symbols=symbols,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="10:00",
                abs_ret_min=0.002,
                cooldown_sec=120,
            ),
        )
    )
    print("collect pulse_fo08_causal_open…", flush=True)
    candidates.append(
        (
            "pulse_fo08_causal_open",
            collect_pulse(
                name="pulse_fo08_causal_open",
                dates=dates,
                symbols=symbols,
                stock_by_sym=stock_by_sym,
                stock_1s=stock_1s,
                window_start="09:30",
                window_end="10:00",
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
            "quote_fill_rate": st["fill_rate"],
            "n_gate_fail": st["n_gate_fail"],
            "n_no_quote": st["n_no_quote"],
            # primary: trade-last
            "trade_n": t["n"],
            "trade_tpd": t["tpd"],
            "trade_win": t["trade_win"],
            "trade_mean": t["mean_ret"],
            "trade_disc": t["disc_compound"],
            "trade_blind": t["blind_compound"],
            "trade_econ": t["econ_dual"],
            # appendix: quote
            "quote_n": q["n"],
            "quote_tpd": q["tpd"],
            "quote_win": q["trade_win"],
            "quote_mean": q["mean_ret"],
            "quote_disc": q["disc_compound"],
            "quote_blind": q["blind_compound"],
            "quote_econ": q["econ_dual"],
        }
        rows.append(row)
        print(
            f"  TRADE n={t['n']} tpd={t['tpd']:.2f} win={t['trade_win']} "
            f"mean={t['mean_ret']} disc={t['disc_compound']:+.3f} "
            f"blind={t['blind_compound']:+.3f} econ={t['econ_dual']}",
            flush=True,
        )
        print(
            f"  QUOTE n={q['n']} tpd={q['tpd']:.2f} fill={st['fill_rate']:.2%} "
            f"mean={q['mean_ret']} disc={q['disc_compound']:+.3f} "
            f"blind={q['blind_compound']:+.3f} econ={q['econ_dual']}",
            flush=True,
        )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    # Best by trade mean among dual trade_econ; else soft trade disc|blind >0
    trade_econ = sb[sb.trade_econ == True].copy()  # noqa: E712
    trade_econ = trade_econ.sort_values(
        ["trade_mean", "trade_disc"], ascending=[False, False]
    )
    best_trade = trade_econ.iloc[0].to_dict() if len(trade_econ) else None
    if best_trade is None and len(sb):
        soft = sb[(sb.trade_disc > 0) | (sb.trade_blind > 0)].copy()
        if len(soft):
            best_trade = soft.sort_values("trade_mean", ascending=False).iloc[0].to_dict()

    quote_econ = sb[sb.quote_econ == True]  # noqa: E712
    verdict = "OPEN_DEBT_TRADE_EDGE" if best_trade and best_trade.get("trade_econ") else (
        "OPEN_DEBT_TRADE_SOFT" if best_trade else "OPEN_DEBT_NO_EDGE"
    )
    if len(quote_econ):
        verdict = "OPEN_QUOTE_ALSO_ECON_" + verdict  # unexpected — flag

    summary = {
        "protocol": "am_v2_open_debt_diag",
        "promotion": False,
        "promote": "NONE",
        "window": "09:30-10:00",
        "primary_mark": "option_trade_last",
        "appendix_mark": "quote_FillSpec",
        "exit": f"tp{args.tp}_sl{args.sl}_h{args.max_hold_sec}",
        "trades_root": str(trades_root),
        "post10_sleeve_ref": POST10_REF,
        "verdict": verdict,
        "best_trade_diag": best_trade,
        "n_trade_econ": int(len(trade_econ)),
        "n_quote_econ": int(len(quote_econ)),
        "scoreboard": sb.to_dict(orient="records"),
        "note": (
            "Diagnostic only. Does not change frozen post10 sleeve / Step4 shadow. "
            "If trade dual econ and quote empty/fail → open half-hour is upper-bound "
            "debt under quote-only north star."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# AM v2 — Open-debt diagnostic (09:30–10:00)",
        "",
        "- **NOT a promotion track** — `promote=NONE`",
        "- primary: **option trade-last**",
        "- appendix: quote FillSpec",
        f"- exit: TP{args.tp:g}/SL{args.sl:g}/h{args.max_hold_sec}（对齐冻结袖套）",
        f"- verdict: **{verdict}**",
        "",
        "## vs frozen post10 sleeve (quote)",
        "",
        f"- ref: `{POST10_REF['signal']}` / `{POST10_REF['exit']}`",
        f"- quote n≈{POST10_REF['quote_n']} tpd≈{POST10_REF['quote_tpd']} "
        f"mean≈{POST10_REF['quote_mean']:+.2%} "
        f"disc≈{POST10_REF['quote_disc']:+.1%} blind≈{POST10_REF['quote_blind']:+.1%}",
        "",
        "## Primary — trade-last",
        "",
    ]
    cols_t = [
        "signal",
        "n_signals",
        "trade_n",
        "trade_tpd",
        "trade_win",
        "trade_mean",
        "trade_disc",
        "trade_blind",
        "trade_econ",
    ]
    try:
        lines.append(sb[cols_t].to_markdown(index=False))
    except Exception:
        lines.append(sb[cols_t].to_string(index=False))
    lines += ["", "## Appendix — quote FillSpec", ""]
    cols_q = [
        "signal",
        "quote_fill_rate",
        "n_gate_fail",
        "quote_n",
        "quote_tpd",
        "quote_win",
        "quote_mean",
        "quote_disc",
        "quote_blind",
        "quote_econ",
    ]
    try:
        lines.append(sb[cols_q].to_markdown(index=False))
    except Exception:
        lines.append(sb[cols_q].to_string(index=False))
    lines += [
        "",
        "## 读法",
        "",
        "1. trade 双窗 econ → 开盘有**上界边**，但 quote 门挡住可执行路径。",
        "2. quote econ 仍无 → 北星下继续跳过开盘；不改冻结袖套。",
        "3. 若将来接开盘，必须先解决 lag 债（数据/锁约），不能靠放宽 mark。",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== TRADE PRIMARY ===", flush=True)
    print(sb[cols_t].to_string(index=False), flush=True)
    print("\n=== QUOTE APPENDIX ===", flush=True)
    print(sb[cols_q].to_string(index=False), flush=True)
    print(
        json.dumps(
            {"verdict": verdict, "promote": "NONE", "best_trade": best_trade and best_trade.get("signal")},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
