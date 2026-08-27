"""CORE peer3 × FRONTLOAD_CHOP: PRE vs A(scale) vs B(block) dual-window accept.

Arms:
  PRE   = research baseline (no frontload overlay)
  A     = frontload days size_scale (default 0.5)
  B     = frontload days block new CORE entries

Optional sub-state overlay (``--overlay weak``): only apply A/B when entry-time
regime is weak/choppy (``vixy_z`` / ``|qqq_from_prev|`` / flip).

Windows: strong Apr–Jul, weak Jan–Mar, week Jul20–24 slice.

Pass sketch (vs PRE):
  strong keep>=0.90 AND weak keep>=0.85 AND week_ret >= PRE week_ret

Example:
  PYTHONPATH=. python -m maga7.tools.run_frontload_core_ab_accept \\
    --tag research_frontload_core_ab_overlay \\
    --overlay weak --overlay-vixy-z-min 0.75 --overlay-max-abs-qqq-fp 0.008
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.frontload_chop import (
    FrontloadChopConfig,
    FrontloadChopGate,
    label_frontload_day,
)
from maga7.common.replay import run_offline_replay
from maga7.common.stock_1s import (
    build_stock_by_from_1s,
    regime_gate_from_1s,
    session_dates,
)

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

WINDOWS = (
    ("strong_apr_jul", "2026-04-01", "2026-07-24"),
    ("weak_jan_mar", "2026-01-02", "2026-03-31"),
    ("week_0720_24", "2026-07-20", "2026-07-24"),
)


def _build_day_flags(
    *,
    stock_1s_root: Path,
    symbols: list[str],
    dates: list[str],
    cfg: FrontloadChopConfig,
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for date in dates:
        by_sym: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            raw = load_stock_1s_day(stock_1s_root, sym, date)
            if raw is None or getattr(raw, "empty", True):
                continue
            by_sym[sym] = raw
        if len(by_sym) < 4:
            flags[date] = False
            continue
        lab = label_frontload_day(by_sym, symbols=list(by_sym.keys()), cfg=cfg)
        flags[date] = bool(lab["is_frontload"])
    return flags


def _wrap_gate(inner: Any, flags: dict[str, bool], fl_cfg: FrontloadChopConfig) -> Any:
    if not fl_cfg.enabled:
        return inner
    return FrontloadChopGate(inner=inner, day_flags=flags, fl_cfg=fl_cfg)


def _metrics(res: dict[str, Any]) -> dict[str, Any]:
    s = res["summary"]
    return {
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "n_regime_block": s.get("n_regime_block"),
        "n_regime_scale": s.get("n_regime_scale"),
    }


def _opt_float_arg(v: str | None) -> float | None:
    if v is None or str(v).strip().lower() in {"", "none", "null"}:
        return None
    return float(v)


def _fl_cfg_from_args(args: argparse.Namespace, *, mode: str, enabled: bool) -> FrontloadChopConfig:
    return FrontloadChopConfig(
        enabled=enabled,
        min_med_abs_h1=float(args.min_med_abs_h1),
        min_n_large=int(args.min_n_large),
        max_quiet_abs_1m=float(args.max_quiet_abs_1m),
        require_quiet=True,
        require_decel=True,
        min_decel_ratio=float(args.min_decel_ratio),
        min_med_abs_first=float(args.min_med_abs_first),
        mode=mode,
        size_scale=float(args.size_scale) if mode == "scale" else 0.0,
        overlay=str(args.overlay),
        overlay_combine=str(args.overlay_combine),
        overlay_vixy_z_min=_opt_float_arg(args.overlay_vixy_z_min),
        overlay_max_abs_qqq_fp=_opt_float_arg(args.overlay_max_abs_qqq_fp),
        overlay_on_flip=bool(args.overlay_on_flip),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default="research_frontload_core_ab_accept")
    ap.add_argument("--size-scale", type=float, default=0.5)
    ap.add_argument("--min-med-abs-h1", type=float, default=0.008)
    ap.add_argument("--min-n-large", type=int, default=4)
    ap.add_argument("--max-quiet-abs-1m", type=float, default=0.00085)
    ap.add_argument("--min-decel-ratio", type=float, default=1.85)
    ap.add_argument("--min-med-abs-first", type=float, default=0.006)
    ap.add_argument(
        "--overlay",
        choices=("always", "weak"),
        default="always",
        help="always=year-round AND; weak=only when sub-state predicates fire",
    )
    ap.add_argument("--overlay-combine", choices=("or", "and"), default="or")
    ap.add_argument("--overlay-vixy-z-min", default=None, help="float or none")
    ap.add_argument("--overlay-max-abs-qqq-fp", default=None, help="float or none")
    ap.add_argument("--overlay-on-flip", action="store_true")
    ap.add_argument(
        "--arms",
        default="PRE,A_scale,B_block",
        help="comma subset: PRE,A_scale,B_block",
    )
    ap.add_argument(
        "--windows",
        default="strong_apr_jul,weak_jan_mar,week_0720_24",
        help="comma subset of window names",
    )
    args = ap.parse_args(argv)

    base = load_profile(args.profile)
    out = Path(base["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    symbols = [str(s).upper() for s in (base.get("symbols") or [])]
    stock_1s = Path(base["_paths"]["stock_1s_root"])

    want = {x.strip() for x in str(args.windows).split(",") if x.strip()}
    wins = [w for w in WINDOWS if w[0] in want]
    if not wins:
        raise SystemExit(f"no windows selected from {want}")

    arm_want = {x.strip() for x in str(args.arms).split(",") if x.strip()}

    # Build stock frames once over union of dates
    start_all = min(w[1] for w in wins)
    end_all = max(w[2] for w in wins)
    dates_all = session_dates(start_all, end_all)
    print(f"building stock_by {start_all}..{end_all} n={len(dates_all)}", flush=True)
    prof_load = copy.deepcopy(base)
    prof_load["date_range"] = {"start": start_all, "end": end_all}
    stock_by = build_stock_by_from_1s(prof_load, dates=dates_all, include_refs=True)

    fl_label = _fl_cfg_from_args(args, mode="scale", enabled=True)
    print("labeling frontload days…", flush=True)
    flags = _build_day_flags(
        stock_1s_root=stock_1s,
        symbols=symbols,
        dates=dates_all,
        cfg=fl_label,
    )
    fl_dates = sorted(d for d, v in flags.items() if v)
    (out / "frontload_flags.json").write_text(
        json.dumps({"n": len(fl_dates), "dates": fl_dates, "cfg": fl_label.__dict__}, indent=2)
    )
    print(f"frontload days in union: {len(fl_dates)}", flush=True)

    arms: dict[str, FrontloadChopConfig] = {}
    if "PRE" in arm_want:
        arms["PRE"] = FrontloadChopConfig(enabled=False)
    if "A_scale" in arm_want:
        arms["A_scale"] = _fl_cfg_from_args(args, mode="scale", enabled=True)
    if "B_block" in arm_want:
        arms["B_block"] = _fl_cfg_from_args(args, mode="block", enabled=True)
    if not arms:
        raise SystemExit(f"no arms selected from {arm_want}")

    score_rows: list[dict[str, Any]] = []
    for wname, wstart, wend in wins:
        stock_w: dict[str, pd.DataFrame] = {}
        for sym, df in stock_by.items():
            if df is None or df.empty:
                continue
            sub = df[(df["date"].astype(str) >= wstart) & (df["date"].astype(str) <= wend)]
            if not sub.empty:
                stock_w[sym] = sub.reset_index(drop=True)
        for arm_name, fl_cfg in arms.items():
            print(f"run {wname} / {arm_name}…", flush=True)
            prof = copy.deepcopy(base)
            prof["date_range"] = {"start": wstart, "end": wend}
            inner = regime_gate_from_1s(prof, stock_w)
            gate = _wrap_gate(inner, flags, fl_cfg)
            res = run_offline_replay(
                prof,
                scheme="single",
                stock_by=stock_w,
                regime_gate=gate,
            )
            sub = out / f"{wname}__{arm_name}"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "summary.json").write_text(
                json.dumps(res["summary"], indent=2, default=str), encoding="utf-8"
            )
            res["trades"].to_csv(sub / "trades.csv", index=False)
            res["daily"].to_csv(sub / "daily.csv", index=False)
            m = _metrics(res)
            extra: dict[str, Any] = {}
            if isinstance(gate, FrontloadChopGate):
                extra = {
                    "n_frontload_scale": gate.n_scale,
                    "n_frontload_block": gate.n_block,
                    "n_overlay_skip": gate.n_overlay_skip,
                }
            row = {"window": wname, "arm": arm_name, **m, **extra}
            score_rows.append(row)
            print(
                f"  {arm_name}: n={m['n_trades']} ret={m['total_ret']:+.3f} "
                f"dd={m['maxdd']:.3f} fl_scale={extra.get('n_frontload_scale')} "
                f"fl_block={extra.get('n_frontload_block')} "
                f"ov_skip={extra.get('n_overlay_skip')}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)

    verdicts: dict[str, Any] = {}
    for wname, _, _ in wins:
        pre = sb[(sb.window == wname) & (sb.arm == "PRE")]
        if pre.empty:
            continue
        pre_ret = float(pre.iloc[0]["total_ret"])
        for arm in ("A_scale", "B_block"):
            sub = sb[(sb.window == wname) & (sb.arm == arm)]
            if sub.empty:
                continue
            ret = float(sub.iloc[0]["total_ret"])
            keep = (ret / pre_ret) if abs(pre_ret) > 1e-12 else None
            verdicts[f"{wname}__{arm}"] = {
                "pre_ret": pre_ret,
                "arm_ret": ret,
                "keep": keep,
                "maxdd_pre": float(pre.iloc[0]["maxdd"]),
                "maxdd_arm": float(sub.iloc[0]["maxdd"]),
                "n_pre": int(pre.iloc[0]["n_trades"]),
                "n_arm": int(sub.iloc[0]["n_trades"]),
            }

    def _keep(w: str, arm: str) -> float | None:
        v = verdicts.get(f"{w}__{arm}")
        return None if v is None else v.get("keep")

    pre_week = sb[(sb.window == "week_0720_24") & (sb.arm == "PRE")]
    week_pre_ret = float(pre_week.iloc[0]["total_ret"]) if len(pre_week) else None

    def arm_pass(arm: str, strong_thr: float, weak_thr: float) -> bool:
        ks = _keep("strong_apr_jul", arm)
        kw = _keep("weak_jan_mar", arm)
        row = sb[(sb.window == "week_0720_24") & (sb.arm == arm)]
        if ks is None or kw is None or week_pre_ret is None or row.empty:
            return False
        wr = float(row.iloc[0]["total_ret"])
        return bool(ks >= strong_thr and kw >= weak_thr and wr >= week_pre_ret - 1e-12)

    summary = {
        "protocol": "frontload_core_ab_accept",
        "overlay": {
            "mode": args.overlay,
            "combine": args.overlay_combine,
            "vixy_z_min": _opt_float_arg(args.overlay_vixy_z_min),
            "max_abs_qqq_fp": _opt_float_arg(args.overlay_max_abs_qqq_fp),
            "on_flip": bool(args.overlay_on_flip),
        },
        "n_frontload_days_union": len(fl_dates),
        "verdicts": verdicts,
        "week_pre_ret": week_pre_ret,
        "pass_A_scale": arm_pass("A_scale", 0.90, 0.85) if "A_scale" in arms else False,
        "pass_B_block": arm_pass("B_block", 0.90, 0.85) if "B_block" in arms else False,
        "promote": "NONE",
    }
    if summary["pass_B_block"]:
        summary["promote"] = "B_block_research"
    elif summary["pass_A_scale"]:
        summary["promote"] = "A_scale_research"

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
