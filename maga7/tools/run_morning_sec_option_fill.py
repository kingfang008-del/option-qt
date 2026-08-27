#!/usr/bin/env python3
"""Price morning second-MF candidates with open_ladder 1s option fills.

Reads structural signals from ``research_morn_sec_edge_*`` (or regenerates),
then simulates ATM open_ladder paths. Independent of 10:30 freeze book.

Example:
  python -m maga7.tools.run_morning_sec_option_fill \\
    --events-tag research_morn_sec_edge_feb_jul \\
    --tag research_morn_sec_option_fill_feb_jul
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
from maga7.common.config import load_profile
from maga7.common.fills import FillSpec
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract, resolve_otm_rungs
from maga7.common.replay import load_quotes, month_list, path_for_ticker, simulate_trade, to_ny
from maga7.common.signals import attach_mf_features, load_stock_month_files

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Top stock-edge cells to validate with option fills.
DEFAULT_CANDIDATES = [
    # user-priority: 100s / streak20 / H180
    {"name": "w100_s20_h180_fp005_vz1_p0", "mf_window_sec": 100, "streak_min": 20, "horizon_sec": 180, "from_prev_min": 0.005, "vol_z_min": 1.0, "peer_min": 0},
    {"name": "w100_s20_h180_fp005_vz1_p3", "mf_window_sec": 100, "streak_min": 20, "horizon_sec": 180, "from_prev_min": 0.005, "vol_z_min": 1.0, "peer_min": 3},
    {"name": "w100_s20_h180_fp01_vz1_p0", "mf_window_sec": 100, "streak_min": 20, "horizon_sec": 180, "from_prev_min": 0.01, "vol_z_min": 1.0, "peer_min": 0},
    # robust sharpe leader
    {"name": "w300_s100_h120_fp005_vz15_p3", "mf_window_sec": 300, "streak_min": 100, "horizon_sec": 120, "from_prev_min": 0.005, "vol_z_min": 1.5, "peer_min": 3},
    {"name": "w300_s100_h120_fp003_vz1_p3", "mf_window_sec": 300, "streak_min": 100, "horizon_sec": 120, "from_prev_min": 0.003, "vol_z_min": 1.0, "peer_min": 3},
]


def _spot_at(sdf: pd.DataFrame | None, asof_ts) -> float | None:
    if sdf is None or sdf.empty:
        return None
    asof = to_ny(asof_ts)
    ts = pd.to_datetime(sdf["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    upto = sdf.loc[ts <= asof]
    if upto.empty:
        return None
    px = float(upto.iloc[-1]["close"])
    return px if np.isfinite(px) and px > 0 else None


def _portfolio_day(
    day_trades: list[dict],
    *,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: int,
) -> list[dict]:
    if not day_trades:
        return []
    rows = sorted(day_trades, key=lambda r: (r["entry_ts"], r["symbol"]))
    open_pos: list[tuple[pd.Timestamp, str]] = []
    last_exit: dict[str, pd.Timestamp] = {}
    out: list[dict] = []
    for tr in rows:
        et = to_ny(tr["entry_ts"])
        xt = to_ny(tr["exit_ts"])
        sym = str(tr["symbol"])
        open_pos = [(x, s) for x, s in open_pos if x > et]
        if any(s == sym for _, s in open_pos):
            continue
        if sym in last_exit and (et - last_exit[sym]).total_seconds() < cooldown_minutes * 60:
            continue
        if len(open_pos) >= int(max_concurrent):
            continue
        n_active = len(open_pos) + 1
        size = float(position_frac) / float(n_active)
        row = dict(tr)
        row["size"] = size
        row["pnl_frac"] = float(tr["ret"]) * size
        out.append(row)
        open_pos.append((xt, sym))
        last_exit[sym] = xt
    return out


def _equity_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "n_days": 0,
            "trade_win": None,
            "exp": None,
            "sum_pnl_frac": 0.0,
            "end_equity": 1.0,
            "maxdd": 0.0,
            "day_win": None,
        }
    tr = trades.sort_values(["date", "entry_ts"]).copy()
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    daily: dict[str, float] = {}
    for _, r in tr.iterrows():
        pnl = float(r["pnl_frac"])
        eq *= 1.0 + pnl
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0 if peak > 0 else 0.0)
        d = str(r["date"])
        daily[d] = daily.get(d, 0.0) + pnl
    day_rets = list(daily.values())
    return {
        "n_trades": int(len(tr)),
        "n_days": int(len(daily)),
        "trade_win": float((tr["ret"] > 0).mean()),
        "exp": float(tr["ret"].mean()),
        "sum_pnl_frac": float(tr["pnl_frac"].sum()),
        "end_equity": float(eq),
        "total_ret": float(eq - 1.0),
        "maxdd": float(maxdd),
        "day_win": float(np.mean([1.0 if x > 0 else 0.0 for x in day_rets])) if day_rets else None,
    }


def _filter_events(events: pd.DataFrame, cand: dict[str, Any]) -> pd.DataFrame:
    """Structural signals for one candidate (dedupe to first fire row)."""
    w = int(cand["mf_window_sec"])
    s = int(cand["streak_min"])
    h = int(cand["horizon_sec"])
    fp = float(cand["from_prev_min"])
    vz = float(cand["vol_z_min"])
    peer = int(cand["peer_min"])
    sub = events[
        (events["mf_window_sec"] == w)
        & (events["streak_min"] == s)
        & (events["horizon_sec"] == h)
    ].copy()
    if sub.empty:
        return sub
    # direction-aware from_prev + vol/peer gates
    up_ok = (sub["dir"] == "UP") & (sub["from_prev"] >= fp)
    dn_ok = (sub["dir"] == "DN") & (sub["from_prev"] <= -fp)
    sub = sub[up_ok | dn_ok]
    sub = sub[(sub["vol_z"].isna()) | (sub["vol_z"] >= vz)]
    sub = sub[sub["peer_n"] >= peer]
    # one row per structural event
    sub = sub.sort_values("ts").drop_duplicates(["date", "symbol", "dir", "mf_window_sec", "streak_min"], keep="first")
    return sub.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--events-tag", default="research_morn_sec_edge_feb_jul")
    ap.add_argument("--tag", default="research_morn_sec_option_fill_feb_jul")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--entry-delay-sec", type=int, default=0, help="sec signals: default 0")
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--candidates", default="", help="comma names; empty=all defaults")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    results_dir = Path(paths["results_dir"])
    events_path = results_dir / args.events_tag / "events.parquet"
    if not events_path.is_file():
        csv_path = results_dir / args.events_tag / "events.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing events: {events_path}")
        events = pd.read_csv(csv_path)
    else:
        events = pd.read_parquet(events_path)

    cands = DEFAULT_CANDIDATES
    if args.candidates.strip():
        want = {x.strip() for x in args.candidates.split(",") if x.strip()}
        cands = [c for c in DEFAULT_CANDIDATES if c["name"] in want]
        if not cands:
            raise SystemExit(f"no candidates matched {want}")

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    stock_root = Path(paths["stock_root"]).expanduser()
    otm_rungs = resolve_otm_rungs(prof, default=3)
    prefer_dte = int((prof.get("lock") or {}).get("prefer_dte", 0))
    allowed_dte = list((prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2])
    clear_otm = float(trade.get("clear_otm_ban_0dte_pct", 0.01) or 0.01)
    fill = FillSpec(
        entry_frac=float(fill_cfg.get("entry_frac", 0.75)),
        exit_frac=float(fill_cfg.get("exit_frac", 0.75)),
    )
    tp_mult = float(trade.get("tp_mult", 1.6))
    sl_mult = float(trade.get("sl_mult", 0.45))
    toxic_cfg = (trade.get("trade_toxic") or {}) if args.toxic else {"enabled": False}

    # preload 1m stock for mf-aware exits / spot (months from event dates)
    dates = sorted(events["date"].astype(str).unique())
    start, end = dates[0], dates[-1]
    months = month_list(start, end)
    symbols = sorted(events["symbol"].astype(str).unique())
    print(f"loading 1m stock {start}..{end} symbols={len(symbols)}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(stock_root, sym, months)
        if raw.empty:
            continue
        stock_by[sym] = attach_mf_features(raw)

    out_root = results_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []

    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    for cand in cands:
        name = cand["name"]
        print(f"=== {name} ===", flush=True)
        sigs = _filter_events(events, cand)
        print(f"  signals={len(sigs)}", flush=True)
        H = int(cand["horizon_sec"])
        hold_minutes = max(1, int(np.ceil(H / 60.0)))
        raw_trades: list[dict] = []
        n_opt = n_miss = 0

        for i, row in sigs.iterrows():
            if (len(raw_trades) + n_miss + 1) % 50 == 0:
                print(f"  priced {len(raw_trades)+n_miss}/{len(sigs)}", flush=True)
            sym = str(row["symbol"])
            date = str(row["date"])
            direction = str(row["dir"])
            sig_ts = to_ny(row["ts"])
            entry_ts = sig_ts + pd.Timedelta(seconds=int(args.entry_delay_sec))
            force_exit = entry_ts + pd.Timedelta(seconds=H)
            sdf = stock_by.get(sym)
            spot = _spot_at(sdf, entry_ts)
            by_dte = multi_idx.get((sym, date))
            ticker, dte, src = resolve_open_lock_contract(
                by_dte,
                direction=direction,
                moneyness="ATM",
                spot=spot,
                prefer_dte=prefer_dte,
                allowed_dte=allowed_dte,
                clear_otm_thresh=clear_otm,
                ladder=True,
                otm_rungs=otm_rungs,
            )
            if not ticker:
                n_miss += 1
                continue
            qkey = (sym, date)
            if qkey not in quote_cache:
                quote_cache[qkey] = load_quotes(quote_root, sym, date)
            path = path_for_ticker(quote_cache[qkey], ticker)
            stock_day = None
            if sdf is not None and "date" in sdf.columns:
                stock_day = sdf[sdf["date"].astype(str) == date]
            # optional 1s for toxic/adverse helpers
            stock_1s = load_stock_1s_day(paths["stock_1s_root"], sym, date)
            sim = simulate_trade(
                path,
                entry_ts,
                fill=fill,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                hold_minutes=hold_minutes,
                direction=direction,
                stock_day=stock_day,
                exit_mode=None,  # rails + force_exit horizon
                force_exit_ts=force_exit,
                stock_bar_delay_seconds=0,
                trade_toxic=toxic_cfg,
                stock_1s=stock_1s if not stock_1s.empty else None,
            )
            if sim is None:
                n_miss += 1
                continue
            n_opt += 1
            raw_trades.append(
                {
                    "date": date,
                    "symbol": sym,
                    "dir": direction,
                    "sig_ts": str(sig_ts),
                    "entry_ts": entry_ts,
                    "exit_ts": sim.exit_ts,
                    "ticker": ticker,
                    "dte": dte,
                    "lock_source": src,
                    "entry": float(sim.entry),
                    "exit": float(sim.exit),
                    "ret": float(sim.ret),
                    "reason": str(sim.reason),
                    "horizon_sec": H,
                    "mf_window_sec": int(cand["mf_window_sec"]),
                    "streak_min": int(cand["streak_min"]),
                    "from_prev": float(row["from_prev"]),
                    "vol_z": float(row["vol_z"]) if pd.notna(row["vol_z"]) else np.nan,
                    "peer_n": int(row["peer_n"]),
                    "stock_fwd": float(row["fwd_ret_signed"]) if "fwd_ret_signed" in row and pd.notna(row["fwd_ret_signed"]) else np.nan,
                }
            )

        # portfolio allocate per day
        by_day: dict[str, list[dict]] = {}
        for tr in raw_trades:
            by_day.setdefault(str(tr["date"]), []).append(tr)
        sized: list[dict] = []
        for d, rows in sorted(by_day.items()):
            sized.extend(
                _portfolio_day(
                    rows,
                    position_frac=float(args.position_frac),
                    max_concurrent=int(args.max_concurrent),
                    cooldown_minutes=int(args.cooldown_minutes),
                )
            )
        trades_df = pd.DataFrame(sized)
        sub = out_root / name
        sub.mkdir(parents=True, exist_ok=True)
        if not trades_df.empty:
            trades_df.to_csv(sub / "trades.csv", index=False)
        pd.DataFrame(raw_trades).to_csv(sub / "trades_raw.csv", index=False)
        stats = _equity_stats(trades_df)
        row = {
            "variant": name,
            **cand,
            "n_signals": int(len(sigs)),
            "n_opt_fills": int(n_opt),
            "n_miss": int(n_miss),
            "toxic": bool(args.toxic),
            "position_frac": float(args.position_frac),
            **stats,
        }
        (sub / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        score_rows.append(row)
        print(json.dumps({k: row[k] for k in ("variant", "n_opt_fills", "trade_win", "exp", "total_ret", "maxdd", "end_equity")}, indent=2), flush=True)

    board = pd.DataFrame(score_rows)
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2), encoding="utf-8")
    print("\n=== option-fill scoreboard ===")
    cols = ["variant", "n_opt_fills", "trade_win", "exp", "total_ret", "maxdd", "end_equity", "day_win"]
    print(board[[c for c in cols if c in board.columns]].to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
