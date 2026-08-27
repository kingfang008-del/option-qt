#!/usr/bin/env python3
"""AM v2 blind-lift bakeoff — same frequency family, quality filters only.

Frozen baseline: launch |ret|≥0.2% k=3 cd=300 · 10:00–11:30 · TP15/SL25/h900
Goal: raise quote blind compound without densifying (no lower cd).

Variants (all cd=300):
  0) baseline_both_1000_1130
  1) up_only / dn_only
  2) win_1000_1030 / win_1030_1130
  3) r0025_cd300 / r003_cd300  (stricter ret — fewer fires, not denser)

PASS adopt: quote dual econ AND blind > baseline blind AND disc > 0.
Else: keep frozen baseline; do not densify.

Example:
  PYTHONPATH=. python -m maga7.tools.run_am_v2_blind_lift_bakeoff \\
    --tag research_am_v2_blind_lift
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
from maga7.common.replay import to_ny
from maga7.common.stock_1s import session_dates
from maga7.tools.run_am_v2_step2_signal_bakeoff import collect_launch, score_signals
from maga7.tools.scan_am_pocket_regime_ladder_v2 import _window_of

PROFILE = "maga7/CONFIG/strategy_profiles/am_v2_executable_path_v1.json"
SPINE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _in_hhmm(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = to_ny(ts)

    def _m(hhmm: str) -> int:
        a, b = hhmm.split(":")
        return int(a) * 60 + int(b)

    hm = t.hour * 60 + t.minute
    return _m(start) <= hm < _m(end)


def _filter_sigs(
    sigs: list[dict[str, Any]],
    *,
    name: str,
    dirs: tuple[str, ...] | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    want = {d.upper() for d in dirs} if dirs else None
    for s in sigs:
        if want is not None and str(s["dir"]).upper() not in want:
            continue
        if window_start and window_end and not _in_hhmm(s["decision_ts"], window_start, window_end):
            continue
        row = dict(s)
        row["signal"] = name
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--spine", default=SPINE)
    ap.add_argument("--tag", default="research_am_v2_blind_lift")
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
        f"am_v2 blind-lift days={len(dates)} "
        f"exit=tp{args.tp:g}/sl{args.sl:g}/h{args.max_hold_sec}",
        flush=True,
    )
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)

    print("collect base r002 cd300 10:00–11:30…", flush=True)
    base_sigs = collect_launch(
        name="baseline_both_1000_1130",
        dates=dates,
        symbols=symbols,
        stock_1s=stock_1s,
        window_start="10:00",
        window_end="11:30",
        abs_ret_min=0.002,
        cooldown_sec=300,
        dirs=("UP", "DN"),
    )
    print(f"  n={len(base_sigs)}", flush=True)

    print("collect r0025 cd300…", flush=True)
    r0025 = collect_launch(
        name="r0025_cd300_1000_1130",
        dates=dates,
        symbols=symbols,
        stock_1s=stock_1s,
        window_start="10:00",
        window_end="11:30",
        abs_ret_min=0.0025,
        cooldown_sec=300,
    )
    print(f"  n={len(r0025)}", flush=True)

    print("collect r003 cd300…", flush=True)
    r003 = collect_launch(
        name="r003_cd300_1000_1130",
        dates=dates,
        symbols=symbols,
        stock_1s=stock_1s,
        window_start="10:00",
        window_end="11:30",
        abs_ret_min=0.003,
        cooldown_sec=300,
    )
    print(f"  n={len(r003)}", flush=True)

    variants: list[tuple[str, list[dict[str, Any]]]] = [
        ("baseline_both_1000_1130", base_sigs),
        (
            "up_only_1000_1130",
            _filter_sigs(base_sigs, name="up_only_1000_1130", dirs=("UP",)),
        ),
        (
            "dn_only_1000_1130",
            _filter_sigs(base_sigs, name="dn_only_1000_1130", dirs=("DN",)),
        ),
        (
            "both_1000_1030",
            _filter_sigs(
                base_sigs,
                name="both_1000_1030",
                window_start="10:00",
                window_end="10:30",
            ),
        ),
        (
            "both_1030_1130",
            _filter_sigs(
                base_sigs,
                name="both_1030_1130",
                window_start="10:30",
                window_end="11:30",
            ),
        ),
        ("r0025_cd300_1000_1130", r0025),
        ("r003_cd300_1000_1130", r003),
    ]

    rows = []
    for name, sigs in variants:
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
            "variant": name,
            "n_signals": st["n_signals"],
            "fill_rate": st["fill_rate"],
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
            f"  quote n={q['n']} tpd={q['tpd']:.2f} mean={q['mean_ret']} "
            f"disc={q['disc_compound']:+.3f} blind={q['blind_compound']:+.3f} "
            f"econ={q['econ_dual']}",
            flush=True,
        )

    sb = pd.DataFrame(rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    base = sb[sb.variant == "baseline_both_1000_1130"]
    base_blind = float(base.iloc[0]["quote_blind"]) if len(base) else 0.0
    base_disc = float(base.iloc[0]["quote_disc"]) if len(base) else 0.0
    base_mean = float(base.iloc[0]["quote_mean"]) if len(base) else 0.0

    adoptable = sb[
        (sb.quote_econ == True)  # noqa: E712
        & (sb.quote_blind > base_blind)
        & (sb.quote_disc > 0)
        & (sb.variant != "baseline_both_1000_1130")
    ].copy()
    adoptable = adoptable.sort_values(
        ["quote_blind", "quote_mean", "quote_disc"], ascending=[False, False, False]
    )
    promote = "NONE"
    best = None
    if len(adoptable):
        best = adoptable.iloc[0].to_dict()
        promote = f"BLIND_LIFT_{best['variant']}"

    summary = {
        "protocol": "am_v2_blind_lift",
        "promotion_mark": "quote_FillSpec",
        "densify": False,
        "exit": f"tp{args.tp}_sl{args.sl}_h{args.max_hold_sec}",
        "baseline": {
            "variant": "baseline_both_1000_1130",
            "quote_mean": base_mean,
            "quote_disc": base_disc,
            "quote_blind": base_blind,
        },
        "n_adoptable": int(len(adoptable)),
        "promote": promote,
        "best": best,
        "pass": bool(promote != "NONE"),
        "scoreboard": sb.to_dict(orient="records"),
        "next": (
            "update_frozen_signal_filter"
            if promote != "NONE"
            else "keep_baseline_try_other_quality_or_step5"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# AM v2 — Blind-lift bakeoff（同频质量过滤）",
        "",
        "- densify: **forbidden** (cd stays 300)",
        f"- exit: TP{args.tp:g}/SL{args.sl:g}/h{args.max_hold_sec}",
        f"- baseline blind: {base_blind:+.3f} · disc: {base_disc:+.3f} · mean: {base_mean}",
        f"- promote: **{promote}**",
        f"- pass: **{summary['pass']}**",
        "",
        "## Scoreboard (quote)",
        "",
    ]
    cols = [
        "variant",
        "n_signals",
        "fill_rate",
        "quote_n",
        "quote_tpd",
        "quote_win",
        "quote_mean",
        "quote_disc",
        "quote_blind",
        "quote_econ",
    ]
    try:
        lines.append(sb[cols].to_markdown(index=False))
    except Exception:
        lines.append(sb[cols].to_string(index=False))
    if promote != "NONE":
        lines += [
            "",
            "## 结论",
            "",
            f"**采用** `{best['variant']}`：blind {best['quote_blind']:+.3f} > 基线 {base_blind:+.3f}，"
            "且 quote 双窗 econ。可写回冻结配方过滤。",
        ]
    else:
        lines += [
            "",
            "## 结论",
            "",
            "**无采用变体。** 方向/子窗/略严阈值未能在保持 quote econ 下抬高 blind。",
            "冻结基线不变；下一刀 Step5 dry，或另找非 densify 特征。",
        ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SCOREBOARD ===", flush=True)
    print(sb[cols].to_string(index=False), flush=True)
    print(
        json.dumps(
            {"promote": promote, "pass": summary["pass"], "baseline_blind": base_blind},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
