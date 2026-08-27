#!/usr/bin/env python3
"""Quote-fill ablation for session H120 (sessions with NBBO coverage).

Replays foresight H=120 signals with ``FillSpec(entry_frac=exit_frac=f)`` and
force clock exit at H. Skips AM (often no quotes).

Example:
  PYTHONPATH=. python -m maga7.tools.run_session_h120_quote_fill_frac \\
    --events-tag research_session_horizon_foresight_apr_jul \\
    --session MID_1230_1330 \\
    --fracs 0.8,0.85,0.9 \\
    --tag research_session_mid_quote_frac_apr_jul
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
from maga7.common.fills import FillSpec
from maga7.common.replay import load_quotes, path_for_ticker, simulate_trade, to_ny
from maga7.common.session_entry_reinforce import SessionReinforceConfig, evaluate_reinforce
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.replay import month_list
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day

FREEZE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"


def _honest(trades_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {
            "sum_pnl_frac_additive": 0.0,
            "day_win": None,
            "trades_per_day": 0.0,
            "trade_mean": None,
            "worst_day": None,
            "red_days": 0,
            "n_days": 0,
        }
    day = trades_df.groupby(trades_df["date"].astype(str))["pnl_frac"].sum()
    return {
        "sum_pnl_frac_additive": float(trades_df["pnl_frac"].astype(float).sum()),
        "day_win": float((day > 0).mean()),
        "trades_per_day": float(len(trades_df) / max(int(trades_df["date"].nunique()), 1)),
        "trade_mean": float(trades_df["ret"].astype(float).mean()),
        "worst_day": float(day.min()),
        "red_days": int((day < 0).sum()),
        "n_days": int(day.shape[0]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=FREEZE)
    ap.add_argument("--events-tag", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--fracs", default="0.8,0.85,0.9")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=float, default=2.0)
    ap.add_argument("--reinforce", choices=["none", "mf", "mf_p2_fo035"], default="none")
    args = ap.parse_args(argv)

    if str(args.session).startswith("AM"):
        print("AM often lacks quotes; refusing --session AM_*", flush=True)
        return 2

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    prof = load_profile(args.profile)
    paths = prof["_paths"]
    results_dir = Path(paths["results_dir"])
    symbols = list(prof.get("symbols") or [])
    h = int(args.horizon_sec)

    ev_p = results_dir / args.events_tag / "events.parquet"
    events = pd.read_parquet(ev_p) if ev_p.is_file() else pd.read_csv(
        results_dir / args.events_tag / "events.csv"
    )
    sub = events[
        (events["session"].astype(str) == args.session)
        & (events["horizon_sec"].astype(int) == h)
    ].copy()
    if sub.empty:
        raise SystemExit("no events")
    sig = (
        sub.drop_duplicates(subset=["date", "symbol", "entry_ts", "dir"])
        .sort_values(["date", "entry_ts", "symbol"])
        .reset_index(drop=True)
    )
    dates = sorted(sig["date"].astype(str).unique())
    start, end = dates[0], dates[-1]

    rein_cfg = SessionReinforceConfig()
    if args.reinforce == "mf":
        rein_cfg = SessionReinforceConfig(require_mf=True)
    elif args.reinforce == "mf_p2_fo035":
        rein_cfg = SessionReinforceConfig(
            require_mf=True, peer_min=2, from_open_max=0.035
        )

    stock_by: dict[str, pd.DataFrame] = {}
    if args.reinforce != "none":
        print(f"loading stock+mf for reinforce={args.reinforce}", flush=True)
        for sym in symbols:
            raw = load_stock_month_files(
                Path(paths["stock_root"]).expanduser(), sym, month_list(start, end)
            )
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

    # filter signals once
    kept: list[pd.Series] = []
    for _, r in sig.iterrows():
        if args.reinforce != "none":
            ok, _ = evaluate_reinforce(
                stock_by=stock_by,
                symbol=str(r["symbol"]),
                date=str(r["date"]),
                entry_ts=to_ny(r["entry_ts"]),
                direction=str(r["dir"]),
                cfg=rein_cfg,
                peer_symbols=symbols,
            )
            if not ok:
                continue
        kept.append(r)
    print(f"signals={len(sig)} kept={len(kept)} fracs={fracs}", flush=True)

    quote_cache: dict[tuple[str, str], Any] = {}
    score: list[dict[str, Any]] = []
    out = results_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)

    for frac in fracs:
        fill = FillSpec(entry_frac=float(frac), exit_frac=float(frac))
        raw_tr: list[dict[str, Any]] = []
        n_miss = 0
        for r in kept:
            date = str(r["date"])
            sym = str(r["symbol"])
            ticker = str(r.get("ticker") or "")
            et = to_ny(r["entry_ts"])
            xt = et + pd.Timedelta(seconds=h)
            qkey = (sym, date)
            if qkey not in quote_cache:
                quote_cache[qkey] = load_quotes(paths["quote_1s_root"], sym, date)
            path = path_for_ticker(quote_cache[qkey], ticker)
            if path is None or path.empty:
                n_miss += 1
                continue
            sim = simulate_trade(
                path,
                et,
                fill=fill,
                tp_mult=100.0,
                sl_mult=0.01,
                hold_minutes=max(1, int(np.ceil(h / 60))),
                direction=str(r["dir"]),
                force_exit_ts=xt,
                trade_toxic={"enabled": False},
            )
            if sim is None:
                n_miss += 1
                continue
            raw_tr.append(
                {
                    "date": date,
                    "symbol": sym,
                    "dir": str(r["dir"]),
                    "entry_ts": str(et),
                    "exit_ts": str(sim.exit_ts),
                    "sig_ts": str(et),
                    "ticker": ticker,
                    "ret": float(sim.ret),
                    "reason": str(sim.reason),
                    "session": args.session,
                    "fill_frac": float(frac),
                }
            )
        by_day: dict[str, list[dict]] = {}
        for tr in raw_tr:
            by_day.setdefault(str(tr["date"]), []).append(tr)
        sized: list[dict] = []
        for d in sorted(by_day):
            sized.extend(
                _portfolio_day(
                    by_day[d],
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown_minutes=float(args.cooldown_minutes),
                )
            )
        trades = pd.DataFrame(sized)
        if len(trades) and "pnl_frac" not in trades.columns:
            trades["pnl_frac"] = trades["ret"].astype(float) * trades["size"].astype(float)
        eq = _equity_stats(trades)
        hon = _honest(trades)
        row = {
            "fill_frac": float(frac),
            "n_signals_kept": int(len(kept)),
            "n_miss_quote": int(n_miss),
            "n_trades": int(len(trades)),
            "trade_win": eq.get("trade_win"),
            "compound_total_ret": eq.get("total_ret"),
            "maxdd": eq.get("maxdd"),
            **hon,
        }
        score.append(row)
        if len(trades):
            trades.to_csv(out / f"trades_frac{frac:.2f}.csv", index=False)
        print(
            f"[frac={frac:.2f}] n={row['n_trades']} miss={n_miss} "
            f"mean={row['trade_mean']} win={row['trade_win']} "
            f"add={row['sum_pnl_frac_additive']:+.3f} day_win={row['day_win']} "
            f"worst_day={row['worst_day']} red={row['red_days']}",
            flush=True,
        )

    sdf = pd.DataFrame(score)
    sdf.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "session": args.session,
        "events_tag": args.events_tag,
        "horizon_sec": h,
        "reinforce": args.reinforce,
        "pricing": "quote_1s FillSpec",
        "fracs": fracs,
        "scoreboard": score,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
