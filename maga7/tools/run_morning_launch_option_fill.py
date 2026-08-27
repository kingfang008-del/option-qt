#!/usr/bin/env python3
"""Option-fill validation for launch-slope stock edges.

Reads ``research_launch_slope_*`` events, prices open_ladder ATM paths, and
compares exit books:
  - horizon: force flatten at H seconds (stock-edge label clock)
  - ladder:  ladder_active (second-level TP/SL/stall/mf) with SEC_MAX=H

Example:
  python -m maga7.tools.run_morning_launch_option_fill \\
    --events-tag research_launch_slope_may_jul \\
    --tag research_launch_slope_option_fill_may_jul
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
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day, _spot_at

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

# Robust open-session cells from research_launch_slope_may_jul (n≥20).
DEFAULT_CANDIDATES = [
    {
        "name": "s3_r003_h180_fp003_p2",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.003,
        "horizon_sec": 180,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "s3_r002_h120_fp0_p3",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "horizon_sec": 120,
        "from_prev_min": 0.0,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 0,
    },
    {
        "name": "s3_r002_h180_fp003_p2",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "horizon_sec": 180,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 0,
    },
    {
        "name": "s5_r002_h120_fp005_mf1",
        "session": "open_0930_1030",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "horizon_sec": 120,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 0,
        "mf_confirm": 1,
    },
    {
        "name": "s10_r003_h180_fp005_p3_mf1",
        "session": "open_0930_1030",
        "slope_sec": 10,
        "abs_ret_min": 0.003,
        "horizon_sec": 180,
        "from_prev_min": 0.005,
        "vol_z_min": 0.0,
        "peer_min": 3,
        "mf_confirm": 1,
    },
    {
        "name": "s3_r002_h180_fp003_p2_mf1",
        "session": "open_0930_1030",
        "slope_sec": 3,
        "abs_ret_min": 0.002,
        "horizon_sec": 180,
        "from_prev_min": 0.003,
        "vol_z_min": 0.0,
        "peer_min": 2,
        "mf_confirm": 1,
    },
    {
        "name": "mid_s5_r002_h180_fp005_vz1",
        "session": "mid_1030_1100",
        "slope_sec": 5,
        "abs_ret_min": 0.002,
        "horizon_sec": 180,
        "from_prev_min": 0.005,
        "vol_z_min": 1.0,
        "peer_min": 0,
        "mf_confirm": 0,
    },
]


def _ladder_for_horizon(h_sec: int) -> dict[str, Any]:
    """Impulse-scale ladder: hard cap = H, stepped TP/SL, stall, mf flip."""
    h = max(60, int(h_sec))
    return {
        "enabled": True,
        "when": "always",
        "max_hold_seconds": h,
        "keep_outer_rails": True,
        "sl_rails": [{"ret": -0.15}, {"ret": -0.25}],
        "tp_rails": [
            {"ret": 0.15, "action": "trail", "trail_dd": 0.06},
            {"ret": 0.30, "action": "exit"},
        ],
        "profit_stall": {"min_peak": 0.10, "stall_seconds": 30},
        "mf_flip": True,
        "mf_grace_seconds": 15,
    }


def _filter_events(events: pd.DataFrame, cand: dict[str, Any]) -> pd.DataFrame:
    sess = str(cand["session"])
    slope = int(cand["slope_sec"])
    thr = float(cand["abs_ret_min"])
    h = int(cand["horizon_sec"])
    fp = float(cand["from_prev_min"])
    vz = float(cand["vol_z_min"])
    peer = int(cand["peer_min"])
    mfc = int(cand.get("mf_confirm", 0) or 0)
    sub = events[
        (events["session"] == sess)
        & (events["slope_sec"] == slope)
        & (np.isclose(events["abs_ret_min"].astype(float), thr))
        & (events["horizon_sec"] == h)
    ].copy()
    if sub.empty:
        return sub
    up_ok = (sub["dir"] == "UP") & (sub["from_prev"] >= fp)
    dn_ok = (sub["dir"] == "DN") & (sub["from_prev"] <= -fp)
    sub = sub[up_ok | dn_ok]
    sub = sub[(sub["vol_z"].isna()) | (sub["vol_z"] >= vz)]
    sub = sub[sub["peer_n"] >= peer]
    if mfc:
        if "mf_ok" in sub.columns:
            sub = sub[sub["mf_ok"].astype(bool)]
        else:
            sub = sub.iloc[0:0]
    sub = sub.sort_values("ts").drop_duplicates(
        ["date", "symbol", "dir", "session", "slope_sec", "abs_ret_min"], keep="first"
    )
    return sub.reset_index(drop=True)


def _run_candidate(
    *,
    cand: dict[str, Any],
    exit_book: str,
    sigs: pd.DataFrame,
    multi_idx: dict,
    quote_cache: dict,
    stock_by: dict[str, pd.DataFrame],
    paths: dict[str, Any],
    fill: FillSpec,
    tp_mult: float,
    sl_mult: float,
    toxic_cfg: dict,
    otm_rungs: int,
    prefer_dte: int,
    allowed_dte: list[int],
    clear_otm: float,
    entry_delay_sec: int,
    position_frac: float,
    max_concurrent: int,
    cooldown_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    H = int(cand["horizon_sec"])
    hold_minutes = max(1, int(np.ceil(H / 60.0)))
    lac = _ladder_for_horizon(H) if exit_book == "ladder" else None
    raw_trades: list[dict] = []
    n_opt = n_miss = 0

    for _, row in sigs.iterrows():
        sym = str(row["symbol"])
        date = str(row["date"])
        direction = str(row["dir"])
        sig_ts = to_ny(row["ts"])
        want_entry = sig_ts + pd.Timedelta(seconds=int(entry_delay_sec))
        sdf = stock_by.get(sym)
        spot = _spot_at(sdf, want_entry)
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
            quote_cache[qkey] = load_quotes(paths["quote_1s_root"], sym, date)
        path = path_for_ticker(quote_cache[qkey], ticker)
        if path is None or path.empty:
            n_miss += 1
            continue
        # Anchor clocks to first usable quote (morning 0DTE books are often gappy).
        after = path[path["timestamp"] >= want_entry]
        if after.empty:
            n_miss += 1
            continue
        entry_ts = to_ny(after.iloc[0]["timestamp"])
        force_exit = entry_ts + pd.Timedelta(seconds=H)
        stock_day = None
        if sdf is not None and "date" in sdf.columns:
            stock_day = sdf[sdf["date"].astype(str) == date]
        stock_1s = load_stock_1s_day(paths["stock_1s_root"], sym, date)
        if exit_book == "ladder":
            sim = simulate_trade(
                path,
                entry_ts,
                fill=fill,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                hold_minutes=hold_minutes,
                direction=direction,
                stock_day=stock_day,
                exit_mode="ladder_active",
                ladder_active=lac,
                stock_bar_delay_seconds=0,
                trade_toxic=toxic_cfg,
                stock_1s=stock_1s if stock_1s is not None and not stock_1s.empty else None,
            )
        else:
            sim = simulate_trade(
                path,
                entry_ts,
                fill=fill,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                hold_minutes=hold_minutes,
                direction=direction,
                stock_day=stock_day,
                exit_mode=None,
                force_exit_ts=force_exit,
                stock_bar_delay_seconds=0,
                trade_toxic=toxic_cfg,
                stock_1s=stock_1s if stock_1s is not None and not stock_1s.empty else None,
            )
        if sim is None:
            n_miss += 1
            continue
        n_opt += 1
        held = (to_ny(sim.exit_ts) - to_ny(sim.entry_ts)).total_seconds()
        raw_trades.append(
            {
                "date": date,
                "symbol": sym,
                "dir": direction,
                "sig_ts": str(sig_ts),
                "entry_ts": sim.entry_ts,
                "exit_ts": sim.exit_ts,
                "ticker": ticker,
                "dte": dte,
                "lock_source": src,
                "entry": float(sim.entry),
                "exit": float(sim.exit),
                "ret": float(sim.ret),
                "reason": str(sim.reason),
                "held_sec": float(held),
                "exit_book": exit_book,
                "horizon_sec": H,
                "session": cand["session"],
                "slope_sec": int(cand["slope_sec"]),
                "abs_ret_min": float(cand["abs_ret_min"]),
                "from_prev": float(row["from_prev"]),
                "vol_z": float(row["vol_z"]) if pd.notna(row["vol_z"]) else np.nan,
                "peer_n": int(row["peer_n"]),
                "mf_ok": bool(row["mf_ok"]) if "mf_ok" in row and pd.notna(row["mf_ok"]) else False,
                "stock_fwd": float(row["fwd_ret_signed"])
                if "fwd_ret_signed" in row and pd.notna(row["fwd_ret_signed"])
                else np.nan,
            }
        )

    by_day: dict[str, list[dict]] = {}
    for tr in raw_trades:
        by_day.setdefault(str(tr["date"]), []).append(tr)
    sized: list[dict] = []
    for _, rows in sorted(by_day.items()):
        sized.extend(
            _portfolio_day(
                rows,
                position_frac=float(position_frac),
                max_concurrent=int(max_concurrent),
                cooldown_minutes=int(cooldown_minutes),
            )
        )
    trades_df = pd.DataFrame(sized)
    raw_df = pd.DataFrame(raw_trades)
    stats = _equity_stats(trades_df)
    reasons = (
        raw_df["reason"].astype(str).value_counts().to_dict() if len(raw_df) and "reason" in raw_df.columns else {}
    )
    meta = {
        "n_signals": int(len(sigs)),
        "n_opt_fills": int(n_opt),
        "n_miss": int(n_miss),
        "mean_held_sec": float(raw_df["held_sec"].mean()) if len(raw_df) else float("nan"),
        "reasons": reasons,
        **stats,
    }
    return trades_df, raw_df, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--events-tag", default="research_launch_slope_may_jul")
    ap.add_argument("--tag", default="research_launch_slope_option_fill_may_jul")
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--entry-delay-sec", type=int, default=0)
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exit-books", default="horizon,ladder", help="comma: horizon and/or ladder")
    ap.add_argument("--candidates", default="", help="comma names; empty=all defaults")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    trade = prof.get("trade") or {}
    fill_cfg = prof.get("fill") or {}
    results_dir = Path(paths["results_dir"])
    events_path = results_dir / args.events_tag / "events.parquet"
    if events_path.is_file():
        events = pd.read_parquet(events_path)
    else:
        csv_path = results_dir / args.events_tag / "events.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing events: {events_path}")
        events = pd.read_csv(csv_path)

    cands = list(DEFAULT_CANDIDATES)
    if args.candidates.strip():
        want = {x.strip() for x in args.candidates.split(",") if x.strip()}
        cands = [c for c in DEFAULT_CANDIDATES if c["name"] in want]
        if not cands:
            raise SystemExit(f"no candidates matched {want}")
    exit_books = [x.strip() for x in args.exit_books.split(",") if x.strip()]

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
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

    dates = sorted(events["date"].astype(str).unique())
    start, end = dates[0], dates[-1]
    months = month_list(start, end)
    symbols = sorted(events["symbol"].astype(str).unique())
    print(f"loading 1m stock {start}..{end} symbols={len(symbols)}", flush=True)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = load_stock_month_files(Path(paths["stock_root"]).expanduser(), sym, months)
        if raw.empty:
            continue
        stock_by[sym] = attach_mf_features(raw)

    out_root = results_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    for cand in cands:
        sigs = _filter_events(events, cand)
        print(f"=== {cand['name']} signals={len(sigs)} ===", flush=True)
        for book in exit_books:
            variant = f"{cand['name']}__{book}"
            print(f"  -> {variant}", flush=True)
            trades_df, raw_df, meta = _run_candidate(
                cand=cand,
                exit_book=book,
                sigs=sigs,
                multi_idx=multi_idx,
                quote_cache=quote_cache,
                stock_by=stock_by,
                paths=paths,
                fill=fill,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                toxic_cfg=toxic_cfg,
                otm_rungs=otm_rungs,
                prefer_dte=prefer_dte,
                allowed_dte=allowed_dte,
                clear_otm=clear_otm,
                entry_delay_sec=int(args.entry_delay_sec),
                position_frac=float(args.position_frac),
                max_concurrent=int(args.max_concurrent),
                cooldown_minutes=int(args.cooldown_minutes),
            )
            sub = out_root / variant
            sub.mkdir(parents=True, exist_ok=True)
            if not trades_df.empty:
                trades_df.to_csv(sub / "trades.csv", index=False)
            if not raw_df.empty:
                raw_df.to_csv(sub / "trades_raw.csv", index=False)
            row = {
                "variant": variant,
                "exit_book": book,
                **{k: cand[k] for k in cand},
                "toxic": bool(args.toxic),
                "position_frac": float(args.position_frac),
                **{k: v for k, v in meta.items() if k != "reasons"},
                "reasons": meta.get("reasons", {}),
            }
            (sub / "summary.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            score_rows.append(row)
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "variant",
                            "n_opt_fills",
                            "trade_win",
                            "exp",
                            "total_ret",
                            "maxdd",
                            "mean_held_sec",
                        )
                    },
                    indent=2,
                    default=str,
                ),
                flush=True,
            )
            print(f"     reasons={row['reasons']}", flush=True)

    board = pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in score_rows])
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2, default=str), encoding="utf-8")
    print("\n=== launch-slope option-fill scoreboard ===")
    cols = [
        "variant",
        "exit_book",
        "n_opt_fills",
        "trade_win",
        "exp",
        "total_ret",
        "maxdd",
        "mean_held_sec",
        "day_win",
    ]
    if len(board):
        print(board[[c for c in cols if c in board.columns]].to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
