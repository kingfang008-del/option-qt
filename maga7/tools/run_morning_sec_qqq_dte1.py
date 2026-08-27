#!/usr/bin/env python3
"""Morning sec-MF on QQQ priced with ``/mnt/s990/data/raw_1s/dte1_options/QQQ``.

Standalone from Mag7 open_ladder. Uses 1DTE ATM buckets:
  UP → bucket_id=2 (CALL_ATM), DN → bucket_id=0 (PUT_ATM).

These files already start at 09:30, so fills are not poisoned by a 10:00 window.
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
from maga7.common.sec_mf import attach_sec_mf_features, forward_returns
from maga7.tools.run_morning_sec_atm_dense import atm_morning_ok
from maga7.tools.run_morning_sec_option_fill import _equity_stats, _portfolio_day
from maga7.tools.run_morning_sec_trend_exit import _resolve_trend_exit
from maga7.tools.scan_morning_sec_edge import _bdates, _morning_slice, _prior_close, _rising_edges

NY = "America/New_York"
DEFAULT_OPT_ROOT = Path("/mnt/s990/data/raw_1s/dte1_options/QQQ")
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")

# Legacy 6-bucket: PUT_ATM=0, CALL_ATM=2
BUCKET_ATM = {"UP": 2, "DN": 0}


def _discover_option_dates(opt_root: Path, start: str, end: str) -> list[str]:
    out: list[str] = []
    for p in sorted(opt_root.glob("QQQ_*.parquet")):
        d = p.stem.split("_", 1)[1]
        if start <= d <= end:
            out.append(d)
    return out


def _load_atm_path(opt_root: Path, date: str, direction: str) -> tuple[pd.DataFrame | None, str | None, float | None]:
    path = opt_root / f"QQQ_{date}.parquet"
    if not path.is_file():
        return None, None, None
    df = pd.read_parquet(path)
    bid = int(BUCKET_ATM[direction])
    if "bucket_id" not in df.columns:
        return None, None, None
    sub = df[pd.to_numeric(df["bucket_id"], errors="coerce") == bid].copy()
    if sub.empty:
        return None, None, None
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    if getattr(sub["timestamp"].dt, "tz", None) is None:
        sub["timestamp"] = sub["timestamp"].dt.tz_localize(NY, ambiguous="infer")
    else:
        sub["timestamp"] = sub["timestamp"].dt.tz_convert(NY)
    sub = sub.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    keep = [c for c in ("timestamp", "bid", "ask", "bid_size", "ask_size", "ticker", "strike") if c in sub.columns]
    sub = sub[keep].reset_index(drop=True)
    ticker = str(sub["ticker"].iloc[0]) if "ticker" in sub.columns else None
    strike = float(sub["strike"].iloc[0]) if "strike" in sub.columns else None
    return sub, ticker, strike


def _collect_qqq_signals(
    *,
    stock_1s_root: Path,
    dates: list[str],
    windows: list[int],
    streak_mins: list[int],
    horizons: list[int],
    signal_end: str = "10:30",
    label_end: str = "11:00",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_bd = _bdates(dates[0], dates[-1]) if dates else []
    max_h = max(horizons)
    for date in dates:
        day = load_stock_1s_day(stock_1s_root, "QQQ", date)
        if day.empty:
            continue
        buf = _morning_slice(day, start="09:30", end=label_end)
        if buf.empty:
            continue
        prev = _prior_close(stock_1s_root, "QQQ", date, all_bd)
        for W in windows:
            f = attach_sec_mf_features(buf, mf_window_sec=W, vol_ma_sec=max(300, W * 2), prev_close=prev)
            if f.empty:
                continue
            ts = pd.DatetimeIndex(f["timestamp"])
            close = f["close"].to_numpy(dtype=np.float64)
            fwd = {h: forward_returns(close, h) for h in horizons}
            sig_mask = (ts.time < pd.Timestamp(signal_end).time()) & np.isfinite(f["mf"].to_numpy(dtype=np.float64))
            for smin in streak_mins:
                for direction, scol in (("UP", "streak_up"), ("DN", "streak_dn")):
                    edges = _rising_edges(f[scol].to_numpy(), smin)
                    edges = edges[sig_mask[edges]]
                    if len(edges) == 0:
                        continue
                    i = int(edges[0])
                    rets = {}
                    for h in horizons:
                        r = fwd[h][i]
                        if np.isfinite(r):
                            rets[h] = float(r if direction == "UP" else -r)
                    if not rets:
                        continue
                    rows.append(
                        {
                            "date": date,
                            "symbol": "QQQ",
                            "dir": direction,
                            "ts": ts[i],
                            "mf_window_sec": int(W),
                            "streak_min": int(smin),
                            "from_prev": float(f["from_prev"].iloc[i]),
                            "vol_z": float(f["vol_z"].iloc[i]) if np.isfinite(f["vol_z"].iloc[i]) else np.nan,
                            "entry_px": float(close[i]),
                            **{f"fwd_{h}": rets.get(h, np.nan) for h in horizons},
                        }
                    )
    return pd.DataFrame(rows)


def _filter_sigs(events: pd.DataFrame, *, W: int, S: int, fp: float, vz: float) -> pd.DataFrame:
    sub = events[(events["mf_window_sec"] == W) & (events["streak_min"] == S)].copy()
    if sub.empty:
        return sub
    up = (sub["dir"] == "UP") & (sub["from_prev"] >= fp)
    dn = (sub["dir"] == "DN") & (sub["from_prev"] <= -fp)
    sub = sub[up | dn]
    if vz > -1e8:
        sub = sub[(sub["vol_z"].isna()) | (sub["vol_z"] >= vz)]
    sub = sub.sort_values("ts").drop_duplicates(["date", "dir", "mf_window_sec", "streak_min"], keep="first")
    return sub.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opt-root", default=str(DEFAULT_OPT_ROOT))
    ap.add_argument("--stock-1s-root", default=str(DEFAULT_STOCK_1S))
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-10")
    ap.add_argument("--tag", default="research_morn_sec_qqq_dte1_feb_jul")
    ap.add_argument("--results-dir", default="maga7/results")
    ap.add_argument("--windows", default="100,300")
    ap.add_argument("--streaks", default="20,100")
    ap.add_argument("--horizons", default="180,300")
    ap.add_argument(
        "--candidates",
        default="ungated_w100_s20,w100_s20_fp005_vz1,w100_s20_fp01_vz1,w300_s100_fp005_vz15",
    )
    ap.add_argument("--exits", default="h180,h300,mf_flip_min30_max300,mf_flip_min60_max600")
    ap.add_argument("--min-morn-quotes", type=int, default=100)
    ap.add_argument("--max-entry-lag-sec", type=float, default=5.0)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--cooldown-minutes", type=int, default=5)
    ap.add_argument("--entry-frac", type=float, default=0.75)
    ap.add_argument("--exit-frac", type=float, default=0.75)
    ap.add_argument("--tp-mult", type=float, default=1.6)
    ap.add_argument("--sl-mult", type=float, default=0.45)
    ap.add_argument("--toxic", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    opt_root = Path(args.opt_root).expanduser()
    stock_1s = Path(args.stock_1s_root).expanduser()
    dates = _discover_option_dates(opt_root, args.start_date, args.end_date)
    # keep only days with stock too
    dates = [d for d in dates if (stock_1s / "QQQ" / f"QQQ_{d}.parquet").is_file()]
    if not dates:
        raise SystemExit("no overlapping QQQ option+stock days")

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    streaks = [int(x) for x in args.streaks.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"QQQ dte1 days={len(dates)} {dates[0]}..{dates[-1]}", flush=True)

    events = _collect_qqq_signals(
        stock_1s_root=stock_1s,
        dates=dates,
        windows=windows,
        streak_mins=streaks,
        horizons=horizons,
    )
    out_root = Path(args.results_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_root / "events.parquet", index=False)
    events.to_csv(out_root / "events.csv", index=False)
    print(f"signals rows={len(events)} wrote events", flush=True)

    # stock scoreboard quick view
    stock_rows = []
    for W in windows:
        for S in streaks:
            for fp in (0.0, 0.005, 0.01):
                for vz in (-1e9, 1.0, 1.5):
                    sigs = _filter_sigs(events, W=W, S=S, fp=fp, vz=vz)
                    for H in horizons:
                        col = f"fwd_{H}"
                        if col not in sigs.columns or sigs.empty:
                            continue
                        arr = sigs[col].dropna().to_numpy(dtype=np.float64)
                        if len(arr) < 5:
                            continue
                        stock_rows.append(
                            {
                                "mf_window_sec": W,
                                "streak_min": S,
                                "horizon_sec": H,
                                "from_prev_min": fp,
                                "vol_z_min": 0.0 if vz < 0 else vz,
                                "n": int(len(arr)),
                                "win": float((arr > 0).mean()),
                                "exp": float(arr.mean()),
                            }
                        )
    stock_board = pd.DataFrame(stock_rows).sort_values(["exp", "n"], ascending=[False, False])
    stock_board.to_csv(out_root / "stock_scoreboard.csv", index=False)
    print("top stock cells:\n", stock_board.head(10).to_string(index=False), flush=True)

    def _parse_fp(tok: str) -> float:
        # fp005 -> 0.005, fp01 -> 0.01, fp0 -> 0.0
        raw = tok[2:] if tok.startswith("fp") else tok
        if not raw:
            return 0.0
        return float("0." + raw)

    def _parse_vz(tok: str) -> float:
        # vz1 -> 1.0, vz15 -> 1.5, vz0 -> 0.0
        raw = tok[2:] if tok.startswith("vz") else tok
        if raw in {"", "0"}:
            return 0.0
        if "." in raw:
            return float(raw)
        if raw == "15":
            return 1.5
        return float(raw)

    # parse candidates
    cand_specs: list[dict[str, Any]] = []
    for name in [x.strip() for x in args.candidates.split(",") if x.strip()]:
        body = name.replace("ungated_", "") if name.startswith("ungated_") else name
        parts = body.split("_")
        W, S = int(parts[0][1:]), int(parts[1][1:])
        if name.startswith("ungated_"):
            cand_specs.append({"name": name, "W": W, "S": S, "fp": 0.0, "vz": -1e9})
            continue
        fp = _parse_fp(parts[2]) if len(parts) > 2 and parts[2].startswith("fp") else 0.0
        vz = _parse_vz(parts[3]) if len(parts) > 3 and parts[3].startswith("vz") else 0.0
        cand_specs.append({"name": name, "W": W, "S": S, "fp": fp, "vz": vz})

    exit_specs: list[dict[str, Any]] = []
    for tok in [x.strip() for x in args.exits.split(",") if x.strip()]:
        if tok.startswith("h") and tok[1:].isdigit():
            exit_specs.append({"name": tok, "kind": "horizon", "horizon_sec": int(tok[1:])})
        elif tok.startswith("mf_flip_"):
            body = tok.replace("mf_flip_", "")
            a, b = body.split("_")
            exit_specs.append(
                {
                    "name": tok,
                    "kind": "mf_flip",
                    "min_hold_sec": int(a.replace("min", "")),
                    "max_hold_sec": int(b.replace("max", "")),
                }
            )
        else:
            raise SystemExit(f"unknown exit {tok}")

    fill = FillSpec(entry_frac=float(args.entry_frac), exit_frac=float(args.exit_frac))
    toxic = {"enabled": False}
    feat_cache: dict[tuple[str, int], pd.DataFrame] = {}
    path_cache: dict[tuple[str, str], tuple[pd.DataFrame | None, str | None, float | None]] = {}
    score_rows: list[dict[str, Any]] = []
    all_bd = _bdates(dates[0], dates[-1])

    def get_feat(date: str, W: int) -> pd.DataFrame:
        key = (date, int(W))
        if key in feat_cache:
            return feat_cache[key]
        day = load_stock_1s_day(stock_1s, "QQQ", date)
        buf = _morning_slice(day, start="09:30", end="11:30")
        prev = _prior_close(stock_1s, "QQQ", date, all_bd)
        feat_cache[key] = attach_sec_mf_features(buf, mf_window_sec=W, vol_ma_sec=max(300, W * 2), prev_close=prev)
        return feat_cache[key]

    def get_path(date: str, direction: str):
        key = (date, direction)
        if key not in path_cache:
            path_cache[key] = _load_atm_path(opt_root, date, direction)
        return path_cache[key]

    for cand in cand_specs:
        sigs = _filter_sigs(events, W=cand["W"], S=cand["S"], fp=cand["fp"], vz=cand["vz"])
        print(f"\n=== {cand['name']} signals={len(sigs)} ===", flush=True)
        for ex in exit_specs:
            hold_probe = int(ex.get("horizon_sec") or ex.get("max_hold_sec") or 300)
            variant = f"{cand['name']}__{ex['name']}"
            raw_trades: list[dict] = []
            skip: dict[str, int] = {}
            hold_secs: list[float] = []
            entry_lags: list[float] = []
            exit_reasons: dict[str, int] = {}
            stock_fwds: list[float] = []
            H_stock = int(ex.get("horizon_sec") or 180)
            fwd_col = f"fwd_{H_stock}" if f"fwd_{H_stock}" in sigs.columns else f"fwd_{horizons[0]}"

            for _, row in sigs.iterrows():
                date = str(row["date"])
                direction = str(row["dir"])
                entry_ts = to_ny(row["ts"])
                path, ticker, strike = get_path(date, direction)
                ok, diag = atm_morning_ok(
                    path,
                    entry_ts,
                    hold_sec=hold_probe,
                    min_morn_quotes=int(args.min_morn_quotes),
                    max_entry_lag_sec=float(args.max_entry_lag_sec),
                )
                if not ok:
                    why = str(diag.get("why") or "reject")
                    skip[why] = skip.get(why, 0) + 1
                    continue

                if ex["kind"] == "horizon":
                    exit_ts = entry_ts + pd.Timedelta(seconds=int(ex["horizon_sec"]))
                    exit_why = f"H{int(ex['horizon_sec'])}"
                    hold_minutes = max(1, int(np.ceil(int(ex["horizon_sec"]) / 60.0)))
                else:
                    feat = get_feat(date, int(cand["W"]))
                    exit_ts, exit_why = _resolve_trend_exit(
                        feat,
                        entry_ts=entry_ts,
                        direction=direction,
                        mode="mf_flip",
                        min_hold_sec=int(ex["min_hold_sec"]),
                        max_hold_sec=int(ex["max_hold_sec"]),
                    )
                    if exit_ts is None:
                        skip["no_exit"] = skip.get("no_exit", 0) + 1
                        continue
                    hold_minutes = max(1, int(np.ceil(int(ex["max_hold_sec"]) / 60.0)))

                sim = simulate_trade(
                    path,
                    entry_ts,
                    fill=fill,
                    tp_mult=float(args.tp_mult),
                    sl_mult=float(args.sl_mult),
                    hold_minutes=hold_minutes,
                    direction=direction,
                    exit_mode=None,
                    force_exit_ts=exit_ts,
                    trade_toxic=toxic,
                    stock_bar_delay_seconds=0,
                )
                if sim is None:
                    skip["sim_none"] = skip.get("sim_none", 0) + 1
                    continue
                fill_lag = (to_ny(sim.entry_ts) - entry_ts).total_seconds()
                if fill_lag > float(args.max_entry_lag_sec) + 1.0:
                    skip["fill_lag"] = skip.get("fill_lag", 0) + 1
                    continue
                reason = str(sim.reason)
                if reason == "DISPLACE":
                    reason = exit_why
                held = (to_ny(sim.exit_ts) - to_ny(sim.entry_ts)).total_seconds()
                hold_secs.append(held)
                entry_lags.append(float(diag["entry_lag_sec"] or 0.0))
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                sf = float(row[fwd_col]) if fwd_col in row and pd.notna(row[fwd_col]) else np.nan
                if np.isfinite(sf):
                    stock_fwds.append(sf)
                raw_trades.append(
                    {
                        "date": date,
                        "symbol": "QQQ",
                        "dir": direction,
                        "entry_ts": entry_ts,
                        "sim_entry_ts": sim.entry_ts,
                        "exit_ts": sim.exit_ts,
                        "ticker": ticker,
                        "strike": strike,
                        "bucket_id": BUCKET_ATM[direction],
                        "ret": float(sim.ret),
                        "reason": reason,
                        "held_sec": held,
                        "entry_lag_sec": diag["entry_lag_sec"],
                        "morn_n": diag["morn_n"],
                        "entry": float(sim.entry),
                        "exit": float(sim.exit),
                        "stock_fwd": sf,
                    }
                )

            by_day: dict[str, list[dict]] = {}
            for tr in raw_trades:
                by_day.setdefault(str(tr["date"]), []).append(tr)
            sized: list[dict] = []
            for _, day_rows in sorted(by_day.items()):
                sized.extend(
                    _portfolio_day(
                        day_rows,
                        position_frac=float(args.position_frac),
                        max_concurrent=1,
                        cooldown_minutes=int(args.cooldown_minutes),
                    )
                )
            trdf = pd.DataFrame(sized)
            stats = _equity_stats(trdf)
            row_out = {
                "variant": variant,
                "candidate": cand["name"],
                "exit": ex["name"],
                "n_signals": int(len(sigs)),
                "n_fills": int(len(raw_trades)),
                "n_skip": int(sum(skip.values())),
                "skip": skip,
                "held_sec_p50": float(np.median(hold_secs)) if hold_secs else None,
                "entry_lag_p50": float(np.median(entry_lags)) if entry_lags else None,
                "stock_fwd_mean": float(np.mean(stock_fwds)) if stock_fwds else None,
                "stock_fwd_win": float(np.mean([1.0 if x > 0 else 0.0 for x in stock_fwds])) if stock_fwds else None,
                "exit_reasons": exit_reasons,
                "n_opt_days": int(len(dates)),
                **stats,
            }
            sub = out_root / variant
            sub.mkdir(parents=True, exist_ok=True)
            if not trdf.empty:
                trdf.to_csv(sub / "trades.csv", index=False)
            pd.DataFrame(raw_trades).to_csv(sub / "trades_raw.csv", index=False)
            (sub / "summary.json").write_text(json.dumps(row_out, indent=2, default=str), encoding="utf-8")
            score_rows.append(row_out)
            print(
                json.dumps(
                    {
                        k: row_out[k]
                        for k in (
                            "variant",
                            "n_fills",
                            "trade_win",
                            "exp",
                            "total_ret",
                            "maxdd",
                            "held_sec_p50",
                            "entry_lag_p50",
                            "stock_fwd_mean",
                            "skip",
                        )
                    },
                    indent=2,
                    default=str,
                ),
                flush=True,
            )

    board = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("skip", "exit_reasons")}
            | {"skip": json.dumps(r.get("skip") or {}), "exit_reasons": json.dumps(r.get("exit_reasons") or {})}
            for r in score_rows
        ]
    )
    board.to_csv(out_root / "scoreboard.csv", index=False)
    (out_root / "scoreboard.json").write_text(json.dumps(score_rows, indent=2, default=str), encoding="utf-8")
    meta = {
        "opt_root": str(opt_root),
        "stock_1s_root": str(stock_1s),
        "dates": dates,
        "n_days": len(dates),
        "buckets": BUCKET_ATM,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n=== QQQ dte1 morning scoreboard ===")
    cols = [
        "variant",
        "n_fills",
        "trade_win",
        "exp",
        "total_ret",
        "maxdd",
        "held_sec_p50",
        "entry_lag_p50",
        "stock_fwd_mean",
        "day_win",
    ]
    show = board[[c for c in cols if c in board.columns]].sort_values("total_ret", ascending=False)
    print(show.to_string(index=False))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
