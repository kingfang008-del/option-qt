#!/usr/bin/env python3
"""Inventory MAG7 short-DTE expiries via Polygon (trading DTE).

Purpose
-------
Confirm when Mon/Wed expiries appear (expected ~2026-02) and, for each
trade_date, which trading-DTE∈{0,1,2} buckets are available for NVDA/TSLA.

Does not download quotes. Writes:
  - expiry_calendar.csv   (unique expiries + weekday)
  - day_dte_availability.csv
  - summary.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

from preprocess.download.dte_utils import trading_sessions_between

NY = "America/New_York"


def load_legacy_api_key() -> str:
    legacy_path = Path(__file__).resolve().parents[2] / "preprocess/download/step2_polygon_second_sniper_v1.py"
    try:
        tree = ast.parse(legacy_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets):
                continue
            val = node.value
            if isinstance(val, ast.Call) and len(val.args) >= 2 and isinstance(val.args[1], ast.Constant):
                return str(val.args[1].value)
            if isinstance(val, ast.Constant):
                return str(val.value)
    except Exception:
        return ""
    return ""


def trading_dates(symbol: str, start: str, end: str, client: RESTClient) -> list[str]:
    root = Path.home() / f"train_data/spnq_train_resampled/{symbol}/regular/09:30-16:00/1min"
    dates: set[str] = set()
    start_m, end_m = start[:7], end[:7]
    if root.exists():
        for f in sorted(root.glob("*.parquet")):
            mon = f.stem
            if mon < start_m or mon > end_m:
                continue
            df = pd.read_parquet(f, columns=["timestamp"])
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
            for d in ts.dt.date.astype(str).unique():
                if start <= d <= end:
                    dates.add(d)
    if dates:
        return sorted(dates)
    bars = list(
        client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start,
            to=end,
            limit=50000,
        )
    )
    for b in bars:
        d = pd.Timestamp(b.timestamp, unit="ms", tz="UTC").tz_convert(NY).strftime("%Y-%m-%d")
        if start <= d <= end:
            dates.add(d)
    return sorted(dates)


def list_expiries(client: RESTClient, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Probe Mon/Wed/Fri (and nearby) candidate dates for listed expirations.

    Faster than paging the full contract book: only check weekday candidates.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    candidates = pd.bdate_range(start_ts, end_ts, freq="C")  # weekdays
    rows = []
    for ts in tqdm(candidates, desc=f"{symbol}-expiry-probe"):
        exp = ts.strftime("%Y-%m-%d")
        # Prefer Mon/Wed/Fri; still probe Tue/Thu in case of holiday shifts.
        try:
            hit = next(
                client.list_options_contracts(
                    underlying_ticker=symbol,
                    expiration_date=exp,
                    expired="true",
                    limit=1,
                ),
                None,
            )
        except Exception:
            hit = None
        if hit is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "expiration": exp,
                "weekday": int(ts.weekday()),
                "weekday_name": ts.day_name(),
                "is_monday": int(ts.weekday() == 0),
                "is_wednesday": int(ts.weekday() == 2),
                "is_friday": int(ts.weekday() == 4),
            }
        )
    return pd.DataFrame(rows).sort_values("expiration").reset_index(drop=True)


def day_availability(trade_dates: list[str], expiries: pd.DataFrame, max_dte: int = 5) -> pd.DataFrame:
    if expiries.empty:
        return pd.DataFrame()
    exp_list = expiries["expiration"].tolist()
    rows = []
    for d in tqdm(trade_dates, desc="day-dte"):
        qd = pd.Timestamp(d)
        avail = {}
        for exp in exp_list:
            tdte = trading_sessions_between(qd, pd.Timestamp(exp))
            if 0 <= tdte <= max_dte:
                # keep nearest expiry for each dte bucket
                if tdte not in avail or exp < avail[tdte]:
                    avail[tdte] = exp
        row = {
            "date_str": d,
            "weekday_name": qd.day_name(),
            "has_dte0": int(0 in avail),
            "has_dte1": int(1 in avail),
            "has_dte2": int(2 in avail),
            "exp_dte0": avail.get(0),
            "exp_dte1": avail.get(1),
            "exp_dte2": avail.get(2),
            "n_short_expiries": len(avail),
        }
        # classify regime: friday-only vs mon/wed/fri short
        short_exps = [e for e in (avail.get(0), avail.get(1), avail.get(2)) if e]
        wds = {pd.Timestamp(e).weekday() for e in short_exps}
        row["short_has_mon"] = int(0 in wds)
        row["short_has_wed"] = int(2 in wds)
        row["short_has_fri"] = int(4 in wds)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(symbol: str, cal: pd.DataFrame, days: pd.DataFrame, mon_wed_hint: str) -> dict:
    mon = cal[cal["is_monday"] == 1]["expiration"].tolist() if not cal.empty else []
    wed = cal[cal["is_wednesday"] == 1]["expiration"].tolist() if not cal.empty else []
    fri = cal[cal["is_friday"] == 1]["expiration"].tolist() if not cal.empty else []
    first_mon = mon[0] if mon else None
    first_wed = wed[0] if wed else None
    # first day where short window includes Mon or Wed expiry
    first_mw_day = None
    if not days.empty:
        mw = days[(days["short_has_mon"] == 1) | (days["short_has_wed"] == 1)]
        if not mw.empty:
            first_mw_day = str(mw.iloc[0]["date_str"])
    post = days[days["date_str"] >= (first_mw_day or mon_wed_hint)] if not days.empty else days
    pre = days[days["date_str"] < (first_mw_day or mon_wed_hint)] if not days.empty else days
    def rate(df: pd.DataFrame, col: str) -> float | None:
        if df is None or df.empty:
            return None
        return float(df[col].mean())
    return {
        "symbol": symbol,
        "mon_wed_expected_from": mon_wed_hint,
        "n_expiries": int(len(cal)),
        "n_monday_expiries": int(len(mon)),
        "n_wednesday_expiries": int(len(wed)),
        "n_friday_expiries": int(len(fri)),
        "first_monday_expiry": first_mon,
        "first_wednesday_expiry": first_wed,
        "first_friday_expiry": fri[0] if fri else None,
        "first_trade_day_with_mon_or_wed_in_short_window": first_mw_day,
        "pre_mon_wed": {
            "n_days": int(len(pre)),
            "pct_dte0": rate(pre, "has_dte0"),
            "pct_dte1": rate(pre, "has_dte1"),
            "pct_dte2": rate(pre, "has_dte2"),
            "pct_short_has_mon": rate(pre, "short_has_mon"),
            "pct_short_has_wed": rate(pre, "short_has_wed"),
        },
        "post_mon_wed": {
            "n_days": int(len(post)),
            "pct_dte0": rate(post, "has_dte0"),
            "pct_dte1": rate(post, "has_dte1"),
            "pct_dte2": rate(post, "has_dte2"),
            "pct_short_has_mon": rate(post, "short_has_mon"),
            "pct_short_has_wed": rate(post, "short_has_wed"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="NVDA,TSLA")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-07-09")
    p.add_argument(
        "--expiry-end-date",
        default="2026-07-31",
        help="Upper bound for listed expirations (need a few weeks past end-date).",
    )
    p.add_argument(
        "--mon-wed-expected-from",
        default="2026-02-01",
        help="User prior: MAG7 Mon/Wed expiries roughly start here.",
    )
    p.add_argument(
        "--output-dir",
        default="stock_options/results/mag7_short_dte_expiry_inventory",
    )
    p.add_argument("--api-key", default=os.environ.get("POLYGON_API_KEY") or load_legacy_api_key())
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("POLYGON_API_KEY is not set and no legacy API key found")
    client = RESTClient(args.api_key)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_cal = []
    all_days = []
    summaries = {}
    for symbol in symbols:
        print(f"[inventory] {symbol}: list expiries", flush=True)
        cal = list_expiries(client, symbol, args.start_date, args.expiry_end_date)
        cal_path = out_dir / f"expiry_calendar_{symbol}.csv"
        cal.to_csv(cal_path, index=False)
        print(f"  expiries={len(cal)} -> {cal_path}", flush=True)

        dates = trading_dates(symbol, args.start_date, args.end_date, client)
        print(f"[inventory] {symbol}: {len(dates)} trade days", flush=True)
        days = day_availability(dates, cal, max_dte=5)
        days.insert(0, "symbol", symbol)
        day_path = out_dir / f"day_dte_availability_{symbol}.csv"
        days.to_csv(day_path, index=False)

        summary = summarize(symbol, cal, days, args.mon_wed_expected_from)
        summaries[symbol] = summary
        all_cal.append(cal)
        all_days.append(days)
        print(json.dumps(summary, indent=2), flush=True)

    if all_cal:
        pd.concat(all_cal, ignore_index=True).to_csv(out_dir / "expiry_calendar.csv", index=False)
    if all_days:
        pd.concat(all_days, ignore_index=True).to_csv(out_dir / "day_dte_availability.csv", index=False)

    blob = {
        "config": {k: v for k, v in vars(args).items() if k != "api_key"},
        "note": (
            "MAG7 Mon/Wed expiries are expected from ~2026-02; "
            "pre window is Friday-weekly oriented; post window should show Mon/Wed in short DTE."
        ),
        "symbols": summaries,
        "files": {
            "calendar": str(out_dir / "expiry_calendar.csv"),
            "day_availability": str(out_dir / "day_dte_availability.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
