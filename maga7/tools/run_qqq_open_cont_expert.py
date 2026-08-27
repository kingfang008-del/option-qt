#!/usr/bin/env python3
"""Replay QQQ open_cont executable expert (champion params).

Satellite sleeve — does not touch Mag7 research_baseline Rule-A.

Example:
  PYTHONPATH=. python -m maga7.tools.run_qqq_open_cont_expert \\
    --start-date 2026-07-10 --end-date 2026-07-23 \\
    --tag research_qqq_open_cont_expert_jul10_23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.qqq_open_cont import load_champion, signal_at_clock, simulate_day
from maga7.common.stock_1s import session_dates

PROFILE = "maga7/CONFIG/strategy_profiles/qqq_open_cont_0945_fo02_tp10_sl25_v1.json"


def _load_raw(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _window_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"n": 0, "mean": None, "win": None, "add": 0.0, "day_win": None}
    day = trades.groupby("date")["pnl_frac"].sum()
    return {
        "n": int(len(trades)),
        "mean": float(trades["ret"].mean()),
        "win": float((trades["ret"] > 0).mean()),
        "add": float(trades["pnl_frac"].sum()),
        "day_win": float((day > 0).mean()),
        "red_days": int((day < 0).sum()),
        "worst_day": float(day.min()),
        "n_quote": int((trades["book"] == "quote").sum()),
        "n_trades_book": int((trades["book"] == "trades").sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--book", default=None, help="auto|quote|trades")
    args = ap.parse_args(argv)

    raw = _load_raw(args.profile)
    paths_cfg = raw.get("paths") or {}
    stock_1s = Path(paths_cfg.get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks")
    quote_root = Path(paths_cfg.get("quote_1s_root") or "/mnt/s990/data/raw_1s/dte0_options/QQQ")
    trades_root = Path(
        paths_cfg.get("option_trades_root") or "/mnt/s990/new_option_data_s3_trades"
    )
    results = Path(paths_cfg.get("results_dir") or "/mnt/s990/data/maga7/results")
    champ = load_champion(raw)
    book = str(args.book or (raw.get("qqq_open_cont") or {}).get("book") or "auto")

    start = str(args.start_date or raw.get("date_range", {}).get("start") or "2026-05-01")
    end = str(args.end_date or raw.get("date_range", {}).get("end") or "2026-07-23")
    tag = args.tag or str(raw.get("result_tag") or "research_qqq_open_cont_expert_v1")
    out = results / tag
    out.mkdir(parents=True, exist_ok=True)

    dates = session_dates(start, end)
    print(
        f"qqq_open_cont expert {start}..{end} days={len(dates)} book={book} "
        f"fo≥{champ['from_open_min']} tp{champ['tp']}/sl{champ['sl']}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    n_sig = n_fill = 0
    for di, date in enumerate(dates):
        if di % 20 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)}) fills={n_fill}", flush=True)
        sig = signal_at_clock(
            stock_1s,
            date,
            clock=str(champ["clock"]),
            from_open_min=float(champ["from_open_min"]),
        )
        if sig is None:
            continue
        n_sig += 1
        trade = simulate_day(
            date=date,
            stock_1s_root=stock_1s,
            quote_root=quote_root,
            trades_root=trades_root,
            champion=champ,
            book=book,
        )
        if trade is None:
            continue
        n_fill += 1
        rows.append(trade)

    trades = pd.DataFrame(rows)
    if len(trades):
        trades.to_csv(out / "trades.csv", index=False)

    windows = (
        ("may_jul09", "2026-05-01", "2026-07-09"),
        ("jul10_23", "2026-07-10", "2026-07-23"),
        ("all", start, end),
    )
    win_stats: dict[str, Any] = {}
    for wname, a, b in windows:
        if trades.empty:
            win_stats[wname] = _window_stats(trades)
        else:
            sub = trades[(trades["date"] >= a) & (trades["date"] <= b)]
            win_stats[wname] = _window_stats(sub)

    summary = {
        "expert_id": "qqq_open_cont",
        "profile": raw.get("profile_id"),
        "champion": champ,
        "book": book,
        "start": start,
        "end": end,
        "n_signal_days": n_sig,
        "n_fills": n_fill,
        "windows": win_stats,
        "status": "ACCEPT_RESEARCH",
        "note": (
            "Satellite executable entry. Quote preferred; Jul uses trades fallback "
            "until dte0 quote catches up. Does not modify Mag7 research_baseline."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
