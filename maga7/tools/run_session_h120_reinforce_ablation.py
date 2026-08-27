#!/usr/bin/env python3
"""Ablate entry-reinforce gates on session H120 opportunity books.

Base = causal 60s stock momentum (existing foresight events @ H=120).
Adds mf / streak / peer / vol_z / from_open AND-gates, then re-runs
opportunity portfolio sizing.

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_reinforce_ablation \\
    --events-tag research_session_horizon_foresight_apr_jul \\
    --session AM_0930_1000 \\
    --tag research_session_am_reinforce_apr_jul
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
from maga7.common.replay import month_list, to_ny
from maga7.common.session_entry_reinforce import (
    SessionReinforceConfig,
    cfg_to_dict,
    evaluate_reinforce,
)
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.stock_1s import load_symbol_1s_bars, shift_completed_1m
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"

VARIANTS: dict[str, SessionReinforceConfig] = {
    "BASE": SessionReinforceConfig(),
    "MF": SessionReinforceConfig(require_mf=True),
    "ST2": SessionReinforceConfig(streak_min=2),
    "P2": SessionReinforceConfig(peer_min=2),
    "P3": SessionReinforceConfig(peer_min=3),
    "VZ15": SessionReinforceConfig(vol_z_min=1.5),
    "FO035": SessionReinforceConfig(from_open_max=0.035),
    "MF_P2": SessionReinforceConfig(require_mf=True, peer_min=2),
    "MF_ST2": SessionReinforceConfig(require_mf=True, streak_min=2),
    "MF_P2_FO035": SessionReinforceConfig(
        require_mf=True, peer_min=2, from_open_max=0.035
    ),
    "MF_ST2_P2": SessionReinforceConfig(require_mf=True, streak_min=2, peer_min=2),
    "ALL_SOFT": SessionReinforceConfig(
        require_mf=True, streak_min=1, peer_min=2, vol_z_min=1.0, from_open_max=0.04
    ),
}


def _honest(trades_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {
            "sum_pnl_frac_additive": 0.0,
            "day_compound_ret": 0.0,
            "day_win": None,
            "trades_per_day": 0.0,
            "trade_mean": None,
            "worst_day": None,
            "red_days": 0,
        }
    pnl = trades_df["pnl_frac"].astype(float)
    day = trades_df.groupby(trades_df["date"].astype(str))["pnl_frac"].sum()
    day_eq = 1.0
    for x in day.sort_index():
        day_eq *= 1.0 + float(x)
    return {
        "sum_pnl_frac_additive": float(pnl.sum()),
        "day_compound_ret": float(day_eq - 1.0),
        "day_win": float((day > 0).mean()),
        "trades_per_day": float(len(trades_df) / max(int(trades_df["date"].nunique()), 1)),
        "trade_mean": float(trades_df["ret"].astype(float).mean()),
        "worst_day": float(day.min()),
        "red_days": int((day < 0).sum()),
    }


def _fill(
    raw: list[dict[str, Any]],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: float,
) -> pd.DataFrame:
    by_day: dict[str, list[dict]] = {}
    for tr in raw:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    sized: list[dict] = []
    for d in sorted(by_day):
        sized.extend(
            _portfolio_day(
                by_day[d],
                position_frac=position_frac,
                max_concurrent=max_concurrent,
                cooldown_minutes=cooldown_minutes,
            )
        )
    df = pd.DataFrame(sized)
    if len(df) and "pnl_frac" not in df.columns and "size" in df.columns:
        df["pnl_frac"] = df["ret"].astype(float) * df["size"].astype(float)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=2.0)
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma subset of variant names",
    )
    ap.add_argument(
        "--stock-source",
        choices=["1s", "1m"],
        default="1s",
        help="1s = aggregate from stock_1s + completed-bar shift (default).",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    symbols = list(prof.get("symbols") or [])
    want = {x.strip() for x in args.variants.split(",") if x.strip()}
    variants = {k: v for k, v in VARIANTS.items() if k in want}
    if not variants:
        raise SystemExit(f"no variants in {want}")

    ev_p = results_dir / args.events_tag / "events.parquet"
    if ev_p.is_file():
        events = pd.read_parquet(ev_p)
    else:
        events = pd.read_csv(results_dir / args.events_tag / "events.csv")
    h = int(args.horizon_sec)
    sub = events[
        (events["session"].astype(str) == args.session)
        & (events["horizon_sec"].astype(int) == h)
    ].copy()
    if sub.empty:
        raise SystemExit(f"no events for {args.session} H={h}")
    dates = sorted(sub["date"].astype(str).unique())
    start, end = dates[0], dates[-1]
    months = month_list(start, end)

    print(f"loading stock+mf source={args.stock_source} {start}..{end}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    if args.stock_source == "1s":
        stock_1s = Path(paths.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
        for sym in symbols:
            bars = load_symbol_1s_bars(stock_1s, sym, dates)
            if bars.empty:
                continue
            feat = attach_mf_features(bars)
            stock_by[sym] = shift_completed_1m(feat)
            print(f"  1s→1m-complete {sym} n={len(stock_by[sym])}", flush=True)
    else:
        print("WARNING: LEGACY 1m left-label mf (leaky)", flush=True)
        for sym in symbols:
            raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
            if raw.empty:
                continue
            sdf = attach_mf_features(raw)
            sdf = sdf[(sdf["date"] >= start) & (sdf["date"] <= end)].copy()
            sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
            if sdf["timestamp"].dt.tz is None:
                sdf["timestamp"] = sdf["timestamp"].dt.tz_localize(NY)
            else:
                sdf["timestamp"] = sdf["timestamp"].dt.tz_convert(NY)
            stock_by[sym] = sdf

    # unique signal rows (one per entry; events are per-horizon)
    sig = (
        sub.drop_duplicates(subset=["date", "symbol", "entry_ts", "dir"])
        .sort_values(["date", "entry_ts", "symbol"])
        .reset_index(drop=True)
    )
    print(f"signals={len(sig)} variants={list(variants)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)

    for vname, cfg in variants.items():
        raw_tr: list[dict[str, Any]] = []
        n_pass = n_block = 0
        reasons: dict[str, int] = {}
        for r in sig.itertuples():
            et = to_ny(r.entry_ts)
            ok, meta = evaluate_reinforce(
                stock_by=stock_by,
                symbol=str(r.symbol),
                date=str(r.date),
                entry_ts=et,
                direction=str(r.dir),
                cfg=cfg,
                peer_symbols=symbols,
            )
            if not ok:
                n_block += 1
                reasons[str(meta.get("reason"))] = reasons.get(str(meta.get("reason")), 0) + 1
                continue
            n_pass += 1
            ret = float(r.clock_ret)
            if not np.isfinite(ret):
                continue
            xt = et + pd.Timedelta(seconds=h)
            raw_tr.append(
                {
                    "date": str(r.date),
                    "symbol": str(r.symbol),
                    "dir": str(r.dir),
                    "entry_ts": str(et),
                    "exit_ts": str(xt),
                    "sig_ts": str(et),
                    "ticker": str(getattr(r, "ticker", "") or ""),
                    "ret": ret,
                    "reason": f"clock_H{h}",
                    "session": args.session,
                    "variant": vname,
                }
            )
        trades = _fill(
            raw_tr,
            position_frac=float(args.position_frac),
            max_concurrent=int(args.max_concurrent),
            cooldown_minutes=float(args.cooldown_minutes),
        )
        eq = _equity_stats(trades)
        hon = _honest(trades)
        row = {
            "variant": vname,
            "cfg": cfg_to_dict(cfg),
            "n_signals": int(len(sig)),
            "n_pass": int(n_pass),
            "n_block": int(n_block),
            "pass_rate": float(n_pass / max(len(sig), 1)),
            "block_reasons": reasons,
            "n_trades": int(len(trades)),
            "trade_win": eq.get("trade_win"),
            **hon,
            "compound_total_ret": eq.get("total_ret"),
            "maxdd": eq.get("maxdd"),
        }
        score_rows.append(row)
        if len(trades):
            trades.to_csv(out / f"trades_{vname}.csv", index=False)
        print(
            f"[{vname}] pass={n_pass}/{len(sig)} ({row['pass_rate']:.1%}) "
            f"n={row['n_trades']} mean={row['trade_mean']} "
            f"add={row['sum_pnl_frac_additive']:+.3f} day_win={row['day_win']} "
            f"worst_day={row['worst_day']}",
            flush=True,
        )

    score = pd.DataFrame(score_rows)
    # keep relative to BASE
    if "BASE" in score.variant.values:
        base_add = float(score.loc[score.variant == "BASE", "sum_pnl_frac_additive"].iloc[0])
        base_mean = float(score.loc[score.variant == "BASE", "trade_mean"].iloc[0])
        score["add_keep"] = score["sum_pnl_frac_additive"] / base_add if base_add else np.nan
        score["mean_keep"] = score["trade_mean"] / base_mean if base_mean else np.nan
    score.to_csv(out / "scoreboard.csv", index=False)
    # pick: higher trade_mean than BASE, day_win>=0.95, add_keep>=0.70, fewer or equal red days
    picks = []
    if "BASE" in score.variant.values:
        b = score[score.variant == "BASE"].iloc[0]
        for _, r in score.iterrows():
            if r["variant"] == "BASE":
                continue
            better_mean = (r["trade_mean"] or 0) > (b["trade_mean"] or 0)
            keep_ok = (r.get("add_keep") or 0) >= 0.70
            day_ok = (r["day_win"] or 0) >= 0.95
            red_ok = int(r["red_days"] or 0) <= int(b["red_days"] or 0)
            if better_mean and keep_ok and day_ok and red_ok:
                picks.append(r["variant"])
    summary = {
        "session": args.session,
        "events_tag": args.events_tag,
        "horizon_sec": h,
        "n_signals": int(len(sig)),
        "recommend": picks,
        "note": (
            "Reinforce is AND on top of 60s momentum. Prefer higher trade_mean "
            "with add_keep>=0.70 and day_win>=0.95."
        ),
        "scoreboard": score_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nrecommend={picks}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
