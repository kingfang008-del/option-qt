#!/usr/bin/env python3
"""Scan big underlying moves vs freeze-baseline fills (missed / wrong-pick).

For each RTH day in [start,end]:
  - day_ret / max favorable excursion proxies from 1m stock
  - first Rule-A (+ peer/regime eligibility)
  - join actual trades from offline replay (or --trades csv)

Outputs under results/<tag>/:
  symbol_day.csv, daily_rank.csv, missed_big.csv, loss_vs_mover.csv, summary.json
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
from maga7.common.regime import Mag7RegimeGate
from maga7.common.replay import month_list, run_offline_replay
from maga7.common.signals import (
    attach_mf_features,
    build_all_first_rule_a_signals,
    count_peer_align,
    load_stock_month_files,
)

NY = "America/New_York"


def _rth_day_stats(df: pd.DataFrame, date: str) -> dict[str, float] | None:
    day = df[df["date"].astype(str) == date]
    if day.empty or "timestamp" not in day.columns:
        return None
    d = day.copy()
    ts = pd.to_datetime(d["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    d["_ts"] = ts
    t = d["_ts"].dt.time
    rth = d[(t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())]
    if len(rth) < 30:
        return None
    close = pd.to_numeric(rth["close"], errors="coerce")
    high = pd.to_numeric(rth["high"], errors="coerce") if "high" in rth.columns else close
    low = pd.to_numeric(rth["low"], errors="coerce") if "low" in rth.columns else close
    px0 = float(close.iloc[0])
    px1 = float(close.iloc[-1])
    if not np.isfinite(px0) or px0 <= 0:
        return None
    day_ret = px1 / px0 - 1.0
    max_up = float(high.max()) / px0 - 1.0
    max_dn = 1.0 - float(low.min()) / px0
    # morning / afternoon split at 12:00
    am = rth[rth["_ts"].dt.hour < 12]
    pm = rth[rth["_ts"].dt.hour >= 12]
    morn = float(am["close"].iloc[-1]) / px0 - 1.0 if len(am) else 0.0
    aft = (float(pm["close"].iloc[-1]) / float(am["close"].iloc[-1]) - 1.0) if (len(am) and len(pm)) else 0.0
    # max |from_prev| in signal window if present
    max_abs_fp = np.nan
    if "from_prev" in rth.columns:
        win = rth[(rth["_ts"].dt.time >= pd.Timestamp("10:30").time()) & (rth["_ts"].dt.time <= pd.Timestamp("14:00").time())]
        if len(win):
            fp = pd.to_numeric(win["from_prev"], errors="coerce").abs()
            max_abs_fp = float(fp.max()) if fp.notna().any() else np.nan
    return {
        "day_ret": float(day_ret),
        "max_up": float(max_up),
        "max_dn": float(max_dn),
        "morn_ret": float(morn),
        "aft_ret": float(aft),
        "max_abs_fp_win": float(max_abs_fp) if np.isfinite(max_abs_fp) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--start-date", default="2026-04-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument("--day-ret-min", type=float, default=0.02, help="|day_ret| threshold for big mover")
    ap.add_argument("--trades", default=None, help="optional trades.csv; else run freeze replay")
    ap.add_argument("--tag", default="missed_movers_apr_jul")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    prof["date_range"] = {"start": args.start_date, "end": args.end_date}
    # ensure TCN off
    if isinstance(prof.get("tcn_gate"), dict):
        prof["tcn_gate"]["enabled"] = False

    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    if args.trades:
        trades = pd.read_csv(args.trades)
    else:
        print("running freeze replay for trades...", flush=True)
        result = run_offline_replay(prof, scheme="single")
        trades = result["trades"]
        trades.to_csv(out / "baseline_trades.csv", index=False)
        result["daily"].to_csv(out / "baseline_daily.csv", index=False)
        print(
            {
                "n_trades": int(len(trades)),
                "total_ret": float(result["summary"]["total_ret"]),
                "maxdd": float(result["summary"]["maxdd"]),
            },
            flush=True,
        )

    trades = trades.copy()
    trades["date"] = trades["date"].astype(str)
    trade_keys = set(zip(trades["date"], trades["symbol"].astype(str).str.upper(), trades["dir"].astype(str).str.upper()))
    traded_syms = trades.groupby("date")["symbol"].apply(lambda s: set(s.astype(str).str.upper())).to_dict()
    day_pnl = (
        trades.assign(wret=trades["ret"].astype(float) * trades.get("size_frac", 0.2).astype(float))
        .groupby("date")
        .agg(n_trades=("ret", "size"), sum_ret=("ret", "sum"), mean_ret=("ret", "mean"), worst_ret=("ret", "min"))
        .reset_index()
    )

    paths = prof["_paths"]
    symbols = list(prof["symbols"])
    months = month_list(args.start_date, args.end_date)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in list(dict.fromkeys(symbols + ["QQQ"])):
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)]
        stock_by[sym] = attach_mf_features(
            raw,
            mf_window=int(prof["signal"].get("mf_window", 10)),
            vol_ma_window=int(prof["signal"].get("vol_ma_window", 20)),
        )

    trade_stock = {s: stock_by[s] for s in symbols if s in stock_by}
    sigs = build_all_first_rule_a_signals(trade_stock, prof["signal"])
    sigs["date"] = sigs["date"].astype(str)
    sig_map: dict[tuple[str, str], Any] = {}
    for r in sigs.itertuples(index=False):
        sig_map[(str(r.date), str(r.symbol).upper())] = r

    regime_gate = Mag7RegimeGate.from_profile(prof, months=months)
    peer_min = int(prof["signal"].get("peer_align_min") or 0)
    peer_mode = str(prof["signal"].get("peer_align_mode") or "mf10")
    peer_syms = list(prof["signal"].get("peer_symbols") or symbols)
    streak_min = int(prof["signal"].get("streak_min", 8))

    # event blackout days from daily (halt) — approximate via regime event calendar if present
    reg = prof.get("regime") or {}
    event_block = bool(reg.get("event_calendar_block"))
    # replay marks day_halt on daily; rebuild lightly from trades absence + known calendar
    # Prefer reading daily if we just ran replay
    daily_path = out / "baseline_daily.csv"
    if not args.trades:
        # re-run is expensive; daily already from last replay — save next time
        pass

    dates = sorted(
        {
            str(d)
            for sdf in trade_stock.values()
            for d in sdf["date"].astype(str).unique()
            if args.start_date <= str(d) <= args.end_date
        }
    )

    rows = []
    for date in dates:
        day_rows = []
        for sym in symbols:
            sdf = stock_by.get(sym)
            if sdf is None:
                continue
            st = _rth_day_stats(sdf, date)
            if st is None:
                continue
            sig = sig_map.get((date, sym))
            rule_a = sig is not None
            rule_dir = str(sig.dir).upper() if sig is not None else None
            rule_ts = pd.Timestamp(sig.sig_ts) if sig is not None else None
            peer_n = None
            qqq_ok = None
            eligible = False
            block_reason = None
            if rule_a and rule_ts is not None:
                if peer_min > 0:
                    peer_n = count_peer_align(
                        stock_by,
                        date=date,
                        asof_ts=rule_ts,
                        direction=str(rule_dir),
                        peer_symbols=peer_syms,
                        mode=peer_mode,
                        streak_min=streak_min,
                    )
                else:
                    peer_n = 0
                if regime_gate is not None:
                    dec = regime_gate.check(str(rule_dir), rule_ts)
                    qqq_ok = bool(dec.allow)
                    if not dec.allow:
                        block_reason = dec.reason
                else:
                    qqq_ok = True
                eligible = bool(
                    (peer_n is None or peer_n >= peer_min) and (qqq_ok is not False)
                )
                if peer_n is not None and peer_n < peer_min:
                    block_reason = block_reason or "peer"
            traded_dir = None
            for d in ("UP", "DN"):
                if (date, sym, d) in trade_keys:
                    traded_dir = d
                    break
            eod = "UP" if st["day_ret"] > 0.002 else ("DN" if st["day_ret"] < -0.002 else "FLAT")
            day_rows.append(
                {
                    "date": date,
                    "symbol": sym,
                    "day_ret": st["day_ret"],
                    "abs_day_ret": abs(st["day_ret"]),
                    "max_up": st["max_up"],
                    "max_dn": st["max_dn"],
                    "morn_ret": st["morn_ret"],
                    "aft_ret": st["aft_ret"],
                    "max_abs_fp_win": st["max_abs_fp_win"],
                    "rule_a": rule_a,
                    "rule_a_dir": rule_dir,
                    "rule_a_ts": rule_ts,
                    "peer_n": peer_n,
                    "qqq_ok": qqq_ok,
                    "eligible": eligible,
                    "block_reason": block_reason,
                    "traded_dir": traded_dir,
                    "eod": eod,
                    "dir_match_eod": bool(rule_dir == eod) if rule_dir and eod in {"UP", "DN"} else None,
                }
            )
        if not day_rows:
            continue
        dd = pd.DataFrame(day_rows).sort_values("abs_day_ret", ascending=False)
        # eligible order by signal time (TopK earliest)
        elig = dd[dd["eligible"]].copy()
        if len(elig) and elig["rule_a_ts"].notna().any():
            elig = elig.sort_values("rule_a_ts")
            elig["elig_rank"] = np.arange(1, len(elig) + 1)
            dd = dd.merge(elig[["symbol", "elig_rank"]], on="symbol", how="left")
        else:
            dd["elig_rank"] = np.nan
        topk = int(prof["signal"].get("top_k", 2))
        dd["would_topk"] = dd["elig_rank"].notna() & (dd["elig_rank"] <= topk)
        rows.append(dd)

    symbol_day = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    symbol_day.to_csv(out / "symbol_day.csv", index=False)

    # daily rank
    rank_rows = []
    for date, g in symbol_day.groupby("date"):
        g = g.sort_values("abs_day_ret", ascending=False)
        biggest = g.iloc[0]
        top2 = g.head(2)
        traded = traded_syms.get(date, set())
        elig = g[g["eligible"]].sort_values("rule_a_ts")
        rank_rows.append(
            {
                "date": date,
                "biggest": biggest["symbol"],
                "biggest_day_ret": float(biggest["day_ret"]),
                "biggest_rule_a": bool(biggest["rule_a"]),
                "biggest_eligible": bool(biggest["eligible"]),
                "biggest_would_topk": bool(biggest.get("would_topk", False)),
                "biggest_traded": biggest["symbol"] in traded,
                "topk_elig": ",".join(elig.head(2)["symbol"].tolist()),
                "n_eligible": int(len(elig)),
                "n_rule_a": int(g["rule_a"].sum()),
                "recall_top2_in_traded": float(
                    len(set(top2["symbol"]) & traded) / max(len(top2), 1)
                ),
                "traded": ",".join(sorted(traded)),
            }
        )
    daily_rank = pd.DataFrame(rank_rows)
    daily_rank.to_csv(out / "daily_rank.csv", index=False)

    thr = float(args.day_ret_min)
    big = symbol_day[symbol_day["abs_day_ret"] >= thr].copy()

    def _miss_reason(r: pd.Series) -> str:
        if r["traded_dir"] is not None:
            if r["eod"] in {"UP", "DN"} and r["traded_dir"] != r["eod"]:
                return "traded_wrong_dir"
            return "caught"
        if not r["rule_a"]:
            return "no_rule_a"
        if r.get("block_reason") == "peer" or (r["peer_n"] is not None and r["peer_n"] < peer_min):
            return "peer_fail"
        if r["qqq_ok"] is False:
            return f"regime:{r['block_reason'] or 'block'}"
        if pd.notna(r.get("elig_rank")) and float(r["elig_rank"]) > int(prof["signal"].get("top_k", 2)):
            return f"topk_full_rank{int(r['elig_rank'])}"
        if r["eligible"] and r.get("would_topk"):
            return "eligible_topk_but_no_fill"
        if r["eligible"]:
            return "eligible_not_topk"
        return "other"

    big["miss_reason"] = big.apply(_miss_reason, axis=1)
    big["month"] = big["date"].str[:7]
    big.to_csv(out / "missed_big.csv", index=False)

    missed = big[big["miss_reason"] != "caught"].copy()
    # loss days vs biggest mover
    loss_join = daily_rank.merge(day_pnl, on="date", how="left")
    loss_join["is_loss_day"] = loss_join["sum_ret"].fillna(0) < 0
    loss_join["biggest_missed"] = ~loss_join["biggest_traded"]
    loss_join.to_csv(out / "loss_vs_mover.csv", index=False)

    # summary
    reason_counts = missed["miss_reason"].value_counts().to_dict()
    by_month = (
        missed.groupby("month")
        .agg(n_miss=("symbol", "size"), median_abs=("abs_day_ret", "median"))
        .reset_index()
        .to_dict(orient="records")
    )
    # days where |biggest|>=thr and not traded
    day_miss = daily_rank[
        (daily_rank["biggest_day_ret"].abs() >= thr) & (~daily_rank["biggest_traded"])
    ]
    loss_and_miss = loss_join[loss_join["is_loss_day"] & loss_join["biggest_missed"]]

    # wrong-dir fills on big-move symbols
    wrong = big[big["miss_reason"] == "traded_wrong_dir"]

    summary = {
        "period": f"{args.start_date}..{args.end_date}",
        "day_ret_min": thr,
        "n_trading_days": int(daily_rank["date"].nunique()),
        "n_big_symbol_days": int(len(big)),
        "n_caught": int((big["miss_reason"] == "caught").sum()),
        "n_missed": int(len(missed)),
        "miss_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "n_days_biggest_missed": int(len(day_miss)),
        "n_loss_days_biggest_missed": int(len(loss_and_miss)),
        "n_traded_wrong_dir_on_big": int(len(wrong)),
        "by_month": by_month,
        "top_missed_days": day_miss.sort_values("biggest_day_ret", key=lambda s: s.abs(), ascending=False)
        .head(25)[
            [
                "date",
                "biggest",
                "biggest_day_ret",
                "biggest_rule_a",
                "biggest_eligible",
                "biggest_would_topk",
                "topk_elig",
                "traded",
            ]
        ]
        .to_dict(orient="records"),
        "loss_days_biggest_missed": loss_and_miss.sort_values("sum_ret")
        .head(20)[["date", "biggest", "biggest_day_ret", "sum_ret", "traded", "topk_elig"]]
        .to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in summary if k not in {"top_missed_days", "loss_days_biggest_missed", "by_month"}}, indent=2), flush=True)
    print("\nmiss_reason:", reason_counts, flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
