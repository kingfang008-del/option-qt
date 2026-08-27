#!/usr/bin/env python3
"""2DTE UP-only ablation: shorter hold / tighter trail / FD on 1s clocks.

Builds on Top2 detect (1m←1s). Only UP seats with 2DTE ATM call lock.
No fixed T30 profile hold — variants set explicit max_hold + trail/FD.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.config import load_profile
from maga7.common.decision_funnel import FROZEN_TRADE, FunnelConfig, day_decision_seats
from maga7.common.failure_detector import (
    FailureDetectorConfig,
    failure_cfg_for_sleeve,
    simulate_stock_with_failure,
)
from maga7.common.fills import FillSpec
from maga7.common.smooth_trend import SmoothStockTradeConfig
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity
from maga7.tools.run_top2_1s_dte_vehicle import _load_lock, _option_ret_from_day, _prep_quotes
from maga7.tools.run_top2_1s_parity import _prep_1m

WINDOWS = [
    {"name": "full_2026", "start": "2026-01-02", "end": "2026-07-17"},
    {"name": "weak_jan_mar", "start": "2026-01-02", "end": "2026-03-31"},
    {"name": "strong_may_jul", "start": "2026-05-01", "end": "2026-07-17"},
]


def _variants() -> dict[str, tuple[SmoothStockTradeConfig, FailureDetectorConfig]]:
    base_trade = SmoothStockTradeConfig(
        max_hold_minutes=int(FROZEN_TRADE.max_hold_minutes),
        break_max_adverse=float(FROZEN_TRADE.break_max_adverse),
        break_min_up_frac=float(FROZEN_TRADE.break_min_up_frac),
        break_lookback=10,
    )
    fd_off = FailureDetectorConfig(enabled=False, sleeve="impulse")
    fd_mae = replace(
        failure_cfg_for_sleeve("impulse"),
        early_giveback=9.0,
        path_min_up_frac=-1.0,
        structure_lookback=0,
        lose_open=False,
        lose_vwap=False,
    )
    fd_tight = replace(fd_mae, early_mae_cut=0.0025, max_eval_minutes=5.0, min_hold_minutes=0.5)

    out: dict[str, tuple[SmoothStockTradeConfig, FailureDetectorConfig]] = {
        "base_h180_trail120_fd": (base_trade, fd_mae),
        "h30_trail120": (replace(base_trade, max_hold_minutes=30), fd_off),
        "h45_trail120": (replace(base_trade, max_hold_minutes=45), fd_off),
        "h60_trail120": (replace(base_trade, max_hold_minutes=60), fd_off),
        "h30_trail80": (replace(base_trade, max_hold_minutes=30, break_max_adverse=0.008), fd_off),
        "h45_trail80": (replace(base_trade, max_hold_minutes=45, break_max_adverse=0.008), fd_off),
        "h30_trail60": (replace(base_trade, max_hold_minutes=30, break_max_adverse=0.006), fd_off),
        "h45_trail60_fd": (
            replace(base_trade, max_hold_minutes=45, break_max_adverse=0.006),
            fd_mae,
        ),
        "h20_trail60_fd_tight": (
            replace(base_trade, max_hold_minutes=20, break_max_adverse=0.006),
            fd_tight,
        ),
        "h15_trail80_fd_tight": (
            replace(base_trade, max_hold_minutes=15, break_max_adverse=0.008),
            fd_tight,
        ),
    }
    return out


def _summarize(df: pd.DataFrame, *, ret_col: str = "opt_ret") -> dict:
    ok = df[df["opt_ok"] == True].copy()  # noqa: E712
    if ok.empty:
        return {
            "n": 0,
            "n_missing": int(len(df)),
            "fill_rate": 0.0,
            "total_ret": 0.0,
            "maxdd": 0.0,
            "win": None,
            "avg": None,
            "median_hold": None,
        }
    x = ok.copy()
    x["ret"] = pd.to_numeric(x[ret_col], errors="coerce")
    eq = _equity(x, frac=0.5)
    return {
        "n": int(len(x)),
        "n_missing": int((df["opt_ok"] != True).sum()),  # noqa: E712
        "fill_rate": float(len(x) / max(len(df), 1)),
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "win": eq["trade_win"],
        "avg": float(x["ret"].mean()),
        "median_hold": float(pd.to_numeric(x["hold_minutes"], errors="coerce").median()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2026-01-02")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_2dte_short_hold_ablation_v1",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    stock_1s = Path(prof["_paths"]["stock_1s_root"]).expanduser()
    quote_root = Path(prof["_paths"]["quote_1s_root"])
    lock_path = Path(prof["_paths"]["open_locked_map"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = _load_lock(lock_path)
    fill_ask = FillSpec(1.0, 1.0)
    fill075 = FillSpec(0.75, 0.75)
    funnel = FunnelConfig()
    variants = _variants()

    cal = sorted(
        p.stem.split("_", 1)[1]
        for p in (stock_1s / "NVDA").glob("NVDA_*.parquet")
        if args.start_date <= p.stem.split("_", 1)[1] <= args.end_date
    )

    # Collect UP seats that have 2DTE ATM call lock
    seats: list[dict] = []
    for i, date in enumerate(cal):
        if i % 20 == 0:
            print(f"[seats] {date} ({i+1}/{len(cal)})", flush=True)
        day_1s_by: dict[str, pd.DataFrame] = {}
        day_1m_by: dict[str, pd.DataFrame] = {}
        for sym in SYMS:
            d1s = load_stock_1s_day(stock_1s, sym, date)
            if d1s.empty:
                continue
            d1s = d1s.copy()
            d1s["date"] = date
            hm = d1s["timestamp"].dt.hour * 60 + d1s["timestamp"].dt.minute
            d1s = d1s[(hm >= 9 * 60 + 30) & (hm < 16 * 60)].reset_index(drop=True)
            if d1s.empty:
                continue
            day_1s_by[sym] = d1s
            m1 = _prep_1m(d1s, symbol=sym, date=date)
            if not m1.empty:
                day_1m_by[sym] = m1
        if len(day_1m_by) < 2:
            continue
        day_seats, _ = day_decision_seats(day_1m_by, date=date, cfg=funnel)
        for seat in day_seats:
            if str(seat["direction"]).upper() != "UP":
                continue
            sym = str(seat["symbol"]).upper()
            if (sym, date, 2, "c") not in lock:
                continue
            if sym not in day_1s_by:
                continue
            seats.append(
                {
                    **seat,
                    "date": date,
                    "_day1s": day_1s_by[sym],
                    "contract": lock[(sym, date, 2, "c")],
                }
            )

    print(f"[seats] UP+2DTE-lock n={len(seats)}", flush=True)
    if not seats:
        raise SystemExit("no UP seats with 2DTE lock")

    qcache: dict[tuple[str, str], pd.DataFrame | None] = {}
    all_rows: list[dict] = []

    for vname, (trade_cfg, fd_cfg) in variants.items():
        print(f"[sim] {vname}", flush=True)
        for seat in seats:
            date = str(seat["date"])
            sym = str(seat["symbol"]).upper()
            sim = simulate_stock_with_failure(
                seat["_day1s"],
                entry_ts=seat["detect_ts"],
                direction="UP",
                trade_cfg=trade_cfg,
                fd_cfg=replace(fd_cfg, sleeve=str(seat["sleeve"])),
                date=date,
                sleeve=str(seat["sleeve"]),
                bar_seconds=1,
                trail_arm_minutes=float(trade_cfg.break_lookback),
            )
            if sim is None:
                continue
            qkey = (sym, date)
            if qkey not in qcache:
                qp = quote_root / sym / f"{sym}_{date}.parquet"
                if qp.exists():
                    qcache[qkey] = _prep_quotes(
                        pd.read_parquet(qp, columns=["timestamp", "ticker", "bid", "ask"])
                    )
                else:
                    qcache[qkey] = None
            base = {
                "variant": vname,
                "date": date,
                "symbol": sym,
                "sleeve": seat["sleeve"],
                "detect_ts": str(seat["detect_ts"]),
                "entry_ts": str(sim["entry_ts"]),
                "exit_ts": str(sim["exit_ts"]),
                "hold_minutes": float(sim["hold_minutes"]),
                "exit_reason": sim["exit_reason"],
                "stock_ret": float(sim["ret"]),
                "fd_fired": bool(sim["fd_fired"]),
                "contract": seat["contract"],
            }
            for fill_name, fill in (("askbid", fill_ask), ("fill075", fill075)):
                ores = _option_ret_from_day(
                    qcache[qkey],
                    contract=seat["contract"],
                    entry_ts=sim["entry_ts"],
                    exit_ts=sim["exit_ts"],
                    fill=fill,
                )
                if ores is None:
                    all_rows.append(
                        {**base, "fill": fill_name, "opt_ok": False, "opt_ret": None, "reason": "no_quote"}
                    )
                else:
                    all_rows.append(
                        {
                            **base,
                            "fill": fill_name,
                            "opt_ok": True,
                            "opt_ret": ores["ret"],
                            "entry_spread": ores["entry_spread"],
                            "reason": "ok",
                        }
                    )

    tdf = pd.DataFrame(all_rows)
    tdf.to_parquet(out / "trades.parquet", index=False)

    board = []
    for vname in variants:
        for w in WINDOWS:
            for fill_name in ("askbid", "fill075"):
                sub = tdf[
                    (tdf["variant"] == vname)
                    & (tdf["fill"] == fill_name)
                    & (tdf["date"] >= w["start"])
                    & (tdf["date"] <= w["end"])
                ]
                sm = _summarize(sub)
                # stock side on same rows
                ok = sub[sub["opt_ok"] == True].copy()  # noqa: E712
                if len(ok):
                    sx = ok.copy()
                    sx["ret"] = pd.to_numeric(sx["stock_ret"], errors="coerce")
                    seq = _equity(sx, frac=0.5)
                    stock_ret, stock_win = seq["total_ret"], seq["trade_win"]
                else:
                    stock_ret, stock_win = None, None
                board.append(
                    {
                        "variant": vname,
                        "window": w["name"],
                        "fill": fill_name,
                        **sm,
                        "stock_ret_paired": stock_ret,
                        "stock_win_paired": stock_win,
                    }
                )

    bdf = pd.DataFrame(board)
    bdf.to_csv(out / "scoreboard.csv", index=False)

    # Pick best askbid on may_jul with fill_rate>=0.8 and n>=20
    cand = bdf[
        (bdf["fill"] == "askbid")
        & (bdf["window"] == "strong_may_jul")
        & (bdf["n"] >= 20)
        & (bdf["fill_rate"] >= 0.8)
    ].copy()
    best = None
    if len(cand):
        best = cand.sort_values(["total_ret", "maxdd"], ascending=[False, False]).iloc[0].to_dict()

    # Consistency: best also not terrible on weak window
    weak_ok = False
    if best:
        wrow = bdf[
            (bdf["variant"] == best["variant"])
            & (bdf["fill"] == "askbid")
            & (bdf["window"] == "weak_jan_mar")
        ]
        if len(wrow):
            weak_ok = bool(wrow.iloc[0]["total_ret"] > -0.5 and wrow.iloc[0]["maxdd"] > -0.7)

    verdict = "PROMOTE" if best and best["total_ret"] > 0 and weak_ok else (
        "INTERESTING" if best and best["total_ret"] > -0.3 else "NOT_USEFUL"
    )

    summary = {
        "n_seats_up_2dte_lock": len(seats),
        "verdict": verdict,
        "best_may_jul_askbid": best,
        "weak_ok_for_best": weak_ok,
        "scoreboard": board,
        "note": "UP-only 2DTE ATM call; 1s exit variants; no profile T30.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    ask = bdf[bdf["fill"] == "askbid"].copy()
    pivot = ask.pivot_table(
        index="variant",
        columns="window",
        values=["total_ret", "maxdd", "win", "n", "median_hold"],
        aggfunc="first",
    )
    lines = [
        "# 2DTE UP-only Short-Hold Ablation",
        "",
        f"**Verdict: `{verdict}`** · seats with 2DTE lock: `{len(seats)}`",
        f"Best May–Jul ask/bid: `{best['variant'] if best else None}`",
        "",
        "## Ask/bid by variant × window",
        "",
        "```",
        ask.sort_values(["window", "total_ret"], ascending=[True, False]).to_string(index=False),
        "```",
        "",
        "## Notes",
        "",
        "- Only UP Top2 seats that have open_2dte ATM call lock (coverage-filtered).",
        "- Exit clock is 1s trail/FD/TIME — not research-baseline T30.",
        "- PROMOTE needs May–Jul ask/bid >0 and weak window not a blow-up.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(
        ask[ask.window == "strong_may_jul"][
            ["variant", "n", "total_ret", "maxdd", "win", "avg", "median_hold", "fill_rate"]
        ]
        .sort_values("total_ret", ascending=False)
        .to_string(index=False),
        flush=True,
    )
    print("verdict", verdict, "best", best["variant"] if best else None, flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
