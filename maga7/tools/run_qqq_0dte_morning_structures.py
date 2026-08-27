#!/usr/bin/env python3
"""QQQ 0DTE morning structure research (open continuation + gap fade).

Uses ``/mnt/s990/data/raw_1s/dte0_options/QQQ`` ATM buckets (UP=2, DN=0).
Independent of Mag7 freeze book. Research only.

Structures:
  - open_cont: at ``clock``, follow sign(close−open); optional |from_open| min
  - gap_fade: at ``clock``, fade overnight gap; optional |gap| min

Example:
  python -m maga7.tools.run_qqq_0dte_morning_structures \\
    --start-date 2026-02-01 --end-date 2026-06-30 \\
    --tag research_qqq_0dte_morn_struct_feb_jun
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

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.fills import FillSpec
from maga7.common.replay import simulate_trade, to_ny
from maga7.common.sec_mf import forward_returns
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day
from maga7.tools.run_morning_sec_qqq_dte1 import (
    BUCKET_ATM,
    _discover_option_dates,
    _load_atm_path,
)
from maga7.tools.scan_morning_sec_edge import _bdates, _morning_slice, _prior_close

NY = "America/New_York"
DEFAULT_OPT_ROOT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _rth_day(stock_1s: Path, date: str) -> pd.DataFrame:
    day = load_stock_1s_day(stock_1s, "QQQ", date)
    return _morning_slice(day, start="09:30", end="16:00")


def _sim(
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    *,
    direction: str,
    hold_sec: int,
    fill: FillSpec,
    tp_mult: float,
    sl_mult: float,
) -> Any | None:
    return simulate_trade(
        path,
        entry_ts,
        fill=fill,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        hold_minutes=max(1, int(np.ceil(hold_sec / 60.0))),
        direction=direction,
        exit_mode=None,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(hold_sec)),
        trade_toxic={"enabled": False},
        stock_bar_delay_seconds=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT_ROOT))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK_1S))
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--tag", default="research_qqq_0dte_morn_struct_feb_jun")
    ap.add_argument("--results-dir", default="maga7/results")
    ap.add_argument("--clocks-cont", default="09:45", help="open_cont clocks")
    ap.add_argument("--clocks-fade", default="10:00", help="gap_fade clocks")
    ap.add_argument("--horizons-cont", default="60,180,300")
    ap.add_argument("--horizons-fade", default="180,300,600")
    ap.add_argument("--from-open-mins", default="0,0.001,0.002,0.003")
    ap.add_argument("--gap-mins", default="0.002,0.003,0.005")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--tp-mult", type=float, default=1.6)
    ap.add_argument("--sl-mult", type=float, default=0.45)
    args = ap.parse_args()

    opt_root = Path(args.opt_root).expanduser()
    stock_1s = Path(args.stock_1s_root).expanduser()
    dates = [
        d
        for d in _discover_option_dates(opt_root, args.start_date, args.end_date)
        if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    if not dates:
        raise SystemExit(f"no overlapping days under {opt_root}")
    all_bd = _bdates(dates[0], dates[-1])
    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))

    clocks_cont = [c.strip() for c in args.clocks_cont.split(",") if c.strip()]
    clocks_fade = [c.strip() for c in args.clocks_fade.split(",") if c.strip()]
    h_cont = _parse_ints(args.horizons_cont)
    h_fade = _parse_ints(args.horizons_fade)
    fo_mins = _parse_floats(args.from_open_mins)
    gap_mins = _parse_floats(args.gap_mins)

    out_root = Path(args.results_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"0DTE days={len(dates)} {dates[0]}..{dates[-1]} root={opt_root}", flush=True)

    path_cache: dict[tuple[str, str], tuple[pd.DataFrame | None, str | None, float | None]] = {}

    def get_path(date: str, direction: str):
        key = (date, direction)
        if key not in path_cache:
            path_cache[key] = _load_atm_path(opt_root, date, direction)
        return path_cache[key]

    # Preload day frames once
    day_cache: dict[str, dict[str, Any]] = {}
    for date in dates:
        buf = _rth_day(stock_1s, date)
        if buf.empty:
            continue
        prev = _prior_close(stock_1s, "QQQ", date, all_bd)
        ts = pd.DatetimeIndex(pd.to_datetime(buf["timestamp"]))
        if ts.tz is None:
            ts = ts.tz_localize(NY)
        close = buf["close"].to_numpy(dtype=np.float64)
        open_px = float(close[0])
        gap = float((open_px - prev) / prev) if prev else float("nan")
        day_cache[date] = {"ts": ts, "close": close, "open": open_px, "gap": gap, "prev": prev}

    print(f"loaded stock days={len(day_cache)}", flush=True)

    raw_all: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    # ---- open_cont grid ----
    for clock in clocks_cont:
        for H in h_cont:
            for fo_min in fo_mins:
                variant = f"open_cont_{clock.replace(':', '')}_h{H}_fo{str(fo_min).replace('.', 'p')}"
                trades_raw: list[dict] = []
                for date, d in day_cache.items():
                    ts, close, open_px = d["ts"], d["close"], d["open"]
                    t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
                    i = int(ts.searchsorted(t0, side="left"))
                    if i >= len(close) - 1:
                        continue
                    from_open = float((close[i] - open_px) / open_px) if open_px else 0.0
                    if abs(from_open) < float(fo_min):
                        continue
                    direction = "UP" if from_open > 0 else "DN"
                    path, ticker, strike = get_path(date, direction)
                    if path is None or path.empty:
                        continue
                    entry_ts = to_ny(ts[i])
                    # require quote near entry
                    after = path[path["timestamp"] >= entry_ts]
                    if after.empty:
                        continue
                    lag = (to_ny(after.iloc[0]["timestamp"]) - entry_ts).total_seconds()
                    if lag > 5:
                        continue
                    sim = _sim(
                        path,
                        entry_ts,
                        direction=direction,
                        hold_sec=H,
                        fill=fill,
                        tp_mult=float(args.tp_mult),
                        sl_mult=float(args.sl_mult),
                    )
                    if sim is None:
                        continue
                    fr = forward_returns(close, H)
                    raw = float(fr[i]) if i < len(fr) and np.isfinite(fr[i]) else np.nan
                    stock = float(raw if direction == "UP" else -raw) if np.isfinite(raw) else np.nan
                    reason = str(sim.reason)
                    if reason == "DISPLACE":
                        reason = f"H{H}"
                    trades_raw.append(
                        {
                            "variant": variant,
                            "kind": "open_cont",
                            "date": date,
                            "month": date[:7],
                            "symbol": "QQQ",
                            "dir": direction,
                            "clock": clock,
                            "horizon_sec": H,
                            "from_open": from_open,
                            "gap": d["gap"],
                            "entry_ts": entry_ts,
                            "exit_ts": sim.exit_ts,
                            "sim_entry_ts": sim.entry_ts,
                            "ticker": ticker,
                            "strike": strike,
                            "bucket_id": BUCKET_ATM[direction],
                            "ret": float(sim.ret),
                            "stock_fwd": stock,
                            "reason": reason,
                            "entry": float(sim.entry),
                            "exit": float(sim.exit),
                            "entry_lag_sec": lag,
                        }
                    )

                by_day: dict[str, list[dict]] = {}
                for tr in trades_raw:
                    by_day.setdefault(str(tr["date"]), []).append(tr)
                sized: list[dict] = []
                for _, rows in sorted(by_day.items()):
                    sized.extend(
                        _portfolio_day(
                            rows,
                            position_frac=float(args.position_frac),
                            max_concurrent=1,
                            cooldown_minutes=0,
                        )
                    )
                trdf = pd.DataFrame(sized)
                stats = _equity_stats(trdf)
                stock_arr = np.asarray([t["stock_fwd"] for t in trades_raw if np.isfinite(t["stock_fwd"])], dtype=float)
                month_stats = {}
                if trades_raw:
                    tdf = pd.DataFrame(trades_raw)
                    for m, mg in tdf.groupby("month"):
                        month_stats[str(m)] = {
                            "n": int(len(mg)),
                            "opt_exp": float(mg["ret"].mean()),
                            "opt_win": float((mg["ret"] > 0).mean()),
                            "stock_exp": float(mg["stock_fwd"].mean()) if mg["stock_fwd"].notna().any() else None,
                        }
                row = {
                    "variant": variant,
                    "kind": "open_cont",
                    "clock": clock,
                    "horizon_sec": H,
                    "from_open_min": fo_min,
                    "gap_min": None,
                    "n_fills": int(len(trades_raw)),
                    "n_days": int(trdf["date"].nunique()) if not trdf.empty else 0,
                    "stock_fwd_mean": float(stock_arr.mean()) if len(stock_arr) else None,
                    "stock_fwd_win": float((stock_arr > 0).mean()) if len(stock_arr) else None,
                    "month_stats": month_stats,
                    **stats,
                }
                score_rows.append(row)
                raw_all.extend(trades_raw)
                sub = out_root / variant
                sub.mkdir(parents=True, exist_ok=True)
                if not trdf.empty:
                    trdf.to_csv(sub / "trades.csv", index=False)
                pd.DataFrame(trades_raw).to_csv(sub / "trades_raw.csv", index=False)
                (sub / "summary.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
                print(
                    f"{variant}: n={row['n_fills']} win={row.get('trade_win')} exp={row.get('exp')} "
                    f"total={row.get('total_ret')} stock_w={row.get('stock_fwd_win')}",
                    flush=True,
                )

    # ---- gap_fade grid ----
    for clock in clocks_fade:
        for H in h_fade:
            for g_min in gap_mins:
                variant = f"gap_fade_{clock.replace(':', '')}_h{H}_g{str(g_min).replace('.', 'p')}"
                trades_raw = []
                for date, d in day_cache.items():
                    gap = d["gap"]
                    if not np.isfinite(gap) or abs(gap) < float(g_min):
                        continue
                    ts, close = d["ts"], d["close"]
                    t0 = pd.Timestamp(f"{date} {clock}", tz=NY)
                    i = int(ts.searchsorted(t0, side="left"))
                    if i >= len(close) - 1:
                        continue
                    direction = "DN" if gap > 0 else "UP"
                    path, ticker, strike = get_path(date, direction)
                    if path is None or path.empty:
                        continue
                    entry_ts = to_ny(ts[i])
                    after = path[path["timestamp"] >= entry_ts]
                    if after.empty:
                        continue
                    lag = (to_ny(after.iloc[0]["timestamp"]) - entry_ts).total_seconds()
                    if lag > 5:
                        continue
                    sim = _sim(
                        path,
                        entry_ts,
                        direction=direction,
                        hold_sec=H,
                        fill=fill,
                        tp_mult=float(args.tp_mult),
                        sl_mult=float(args.sl_mult),
                    )
                    if sim is None:
                        continue
                    fr = forward_returns(close, H)
                    raw = float(fr[i]) if i < len(fr) and np.isfinite(fr[i]) else np.nan
                    stock = float(raw if direction == "UP" else -raw) if np.isfinite(raw) else np.nan
                    reason = str(sim.reason)
                    if reason == "DISPLACE":
                        reason = f"H{H}"
                    trades_raw.append(
                        {
                            "variant": variant,
                            "kind": "gap_fade",
                            "date": date,
                            "month": date[:7],
                            "symbol": "QQQ",
                            "dir": direction,
                            "clock": clock,
                            "horizon_sec": H,
                            "from_open": float((close[i] - d["open"]) / d["open"]),
                            "gap": gap,
                            "entry_ts": entry_ts,
                            "exit_ts": sim.exit_ts,
                            "sim_entry_ts": sim.entry_ts,
                            "ticker": ticker,
                            "strike": strike,
                            "bucket_id": BUCKET_ATM[direction],
                            "ret": float(sim.ret),
                            "stock_fwd": stock,
                            "reason": reason,
                            "entry": float(sim.entry),
                            "exit": float(sim.exit),
                            "entry_lag_sec": lag,
                        }
                    )

                by_day = {}
                for tr in trades_raw:
                    by_day.setdefault(str(tr["date"]), []).append(tr)
                sized = []
                for _, rows in sorted(by_day.items()):
                    sized.extend(
                        _portfolio_day(
                            rows,
                            position_frac=float(args.position_frac),
                            max_concurrent=1,
                            cooldown_minutes=0,
                        )
                    )
                trdf = pd.DataFrame(sized)
                stats = _equity_stats(trdf)
                stock_arr = np.asarray([t["stock_fwd"] for t in trades_raw if np.isfinite(t["stock_fwd"])], dtype=float)
                month_stats = {}
                if trades_raw:
                    tdf = pd.DataFrame(trades_raw)
                    for m, mg in tdf.groupby("month"):
                        month_stats[str(m)] = {
                            "n": int(len(mg)),
                            "opt_exp": float(mg["ret"].mean()),
                            "opt_win": float((mg["ret"] > 0).mean()),
                            "stock_exp": float(mg["stock_fwd"].mean()) if mg["stock_fwd"].notna().any() else None,
                        }
                row = {
                    "variant": variant,
                    "kind": "gap_fade",
                    "clock": clock,
                    "horizon_sec": H,
                    "from_open_min": None,
                    "gap_min": g_min,
                    "n_fills": int(len(trades_raw)),
                    "n_days": int(trdf["date"].nunique()) if not trdf.empty else 0,
                    "stock_fwd_mean": float(stock_arr.mean()) if len(stock_arr) else None,
                    "stock_fwd_win": float((stock_arr > 0).mean()) if len(stock_arr) else None,
                    "month_stats": month_stats,
                    **stats,
                }
                score_rows.append(row)
                raw_all.extend(trades_raw)
                sub = out_root / variant
                sub.mkdir(parents=True, exist_ok=True)
                if not trdf.empty:
                    trdf.to_csv(sub / "trades.csv", index=False)
                pd.DataFrame(trades_raw).to_csv(sub / "trades_raw.csv", index=False)
                (sub / "summary.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
                print(
                    f"{variant}: n={row['n_fills']} win={row.get('trade_win')} exp={row.get('exp')} "
                    f"total={row.get('total_ret')} stock_w={row.get('stock_fwd_win')}",
                    flush=True,
                )

    # leave-one-month for top variants by total_ret (n>=20)
    loo_rows = []
    cand = [r for r in score_rows if int(r.get("n_fills") or 0) >= 20]
    cand = sorted(cand, key=lambda r: float(r.get("total_ret") or -1e9), reverse=True)[:8]
    raw_df = pd.DataFrame(raw_all)
    for r in cand:
        v = r["variant"]
        sub = raw_df[raw_df["variant"] == v]
        months = sorted(sub["month"].astype(str).unique())
        for m in months:
            train = sub[sub["month"].astype(str) != m]
            if len(train) < 10:
                continue
            loo_rows.append(
                {
                    "variant": v,
                    "holdout_month": m,
                    "train_n": int(len(train)),
                    "train_exp": float(train["ret"].mean()),
                    "train_win": float((train["ret"] > 0).mean()),
                }
            )

    board = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "month_stats"}
            | {"month_stats": json.dumps(r.get("month_stats") or {})}
            for r in score_rows
        ]
    )
    board = board.sort_values(["total_ret", "n_fills"], ascending=[False, False])
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2, default=str), encoding="utf-8")
    if loo_rows:
        pd.DataFrame(loo_rows).to_csv(out_root / "leave_one_month.csv", index=False)
    if raw_all:
        pd.DataFrame(raw_all).to_csv(out_root / "trades_all_raw.csv", index=False)
    meta = {
        "opt_root": str(opt_root),
        "stock_1s_root": str(stock_1s),
        "start": args.start_date,
        "end": args.end_date,
        "n_days": len(dates),
        "dates_first": dates[0],
        "dates_last": dates[-1],
        "buckets": BUCKET_ATM,
        "note": "QQQ 0DTE ATM morning structures; research only",
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== 0DTE morning structure scoreboard (top) ===")
    cols = [
        "variant",
        "n_fills",
        "trade_win",
        "exp",
        "total_ret",
        "maxdd",
        "stock_fwd_win",
        "day_win",
    ]
    show = board[[c for c in cols if c in board.columns]].head(20)
    print(show.to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
