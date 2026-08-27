#!/usr/bin/env python3
"""Stock backtest: own-path smooth + impulse dual sleeve (May–Jul research).

Entry: first smooth/impulse launch per symbol/dir (morning-biased).
Exit: TRAIL_BREAK / SMOOTH_BREAK / TIME / EOD.
Portfolio: max 2 names/day, 1 symbol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import (
    ImpulseLaunchConfig,
    SmoothLaunchConfig,
    SmoothStockTradeConfig,
    apply_day_portfolio_cap,
    replay_smooth_impulse_stock_day,
)

SYMS = ["NVDA", "TSLA", "AAPL", "AMZN", "META", "MSFT", "AMD", "GOOGL"]
MONTHS = ["2026-05", "2026-06", "2026-07"]


def _equity(trades: pd.DataFrame, *, start: float = 100.0, frac: float = 0.5) -> dict:
    """Equal weight among same-day fills; compound day PnL."""
    if trades.empty:
        return {"total_ret": 0.0, "maxdd": 0.0, "equity_end": start, "n_trades": 0}
    t = trades.copy()
    t["ret"] = pd.to_numeric(t["ret"], errors="coerce").fillna(0.0)
    daily = []
    for date, g in t.groupby("date"):
        # equal split of frac across day's positions
        n = max(len(g), 1)
        day_ret = float((g["ret"] * (frac / n)).sum())
        daily.append({"date": date, "day_ret": day_ret, "n": len(g)})
    ddf = pd.DataFrame(daily).sort_values("date")
    eq = start
    peak = start
    maxdd = 0.0
    curve = []
    for r in ddf.itertuples():
        eq *= 1.0 + float(r.day_ret)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
        curve.append({"date": r.date, "equity": eq, "day_ret": r.day_ret, "n": r.n})
    return {
        "total_ret": eq / start - 1.0,
        "maxdd": maxdd,
        "equity_end": eq,
        "n_trades": int(len(t)),
        "n_days": int(len(ddf)),
        "trade_win": float((t["ret"] > 0).mean()),
        "avg_trade_ret": float(t["ret"].mean()),
        "daily": curve,
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
    ap.add_argument("--start-date", default="2026-05-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/research_smooth_impulse_stock_may_jul")
    ap.add_argument("--max-positions", type=int, default=2)
    ap.add_argument("--max-hold", type=int, default=120)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    root = Path(prof["_paths"]["stock_root"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    smooth_cfg = SmoothLaunchConfig(scan_end="11:30", min_look_ret=0.002, cooldown_minutes=60)
    impulse_cfg = ImpulseLaunchConfig(scan_end="11:30", min_look_ret=0.004)
    trade_cfg = SmoothStockTradeConfig(
        max_hold_minutes=int(args.max_hold),
        max_positions=int(args.max_positions),
        first_per_symbol_dir=True,
    )

    all_trades: list[dict] = []
    for sym in SYMS:
        print(f"[load] {sym}", flush=True)
        raw = load_stock_month_files(root, sym, MONTHS)
        if raw.empty:
            continue
        raw = attach_mf_features(raw)
        dates = sorted(
            d for d in raw["date"].astype(str).unique() if args.start_date <= d <= args.end_date
        )
        for date in dates:
            day = raw[raw["date"].astype(str) == date]
            rows = replay_smooth_impulse_stock_day(
                day,
                symbol=sym,
                date=date,
                smooth_cfg=smooth_cfg,
                impulse_cfg=impulse_cfg,
                trade_cfg=trade_cfg,
            )
            all_trades.extend(rows)

    capped = apply_day_portfolio_cap(all_trades, max_positions=int(args.max_positions))
    tdf = pd.DataFrame(capped)
    tdf.to_csv(out / "trades.csv", index=False)
    # uncapped for research
    pd.DataFrame(all_trades).to_csv(out / "trades_uncapped.csv", index=False)

    eq = _equity(tdf, frac=0.5)
    pd.DataFrame(eq["daily"]).to_csv(out / "daily.csv", index=False)

    by_sleeve = (
        tdf.groupby("sleeve")
        .agg(n=("ret", "size"), mean_ret=("ret", "mean"), win=("ret", lambda s: float((s > 0).mean())), sum_ret=("ret", "sum"))
        .reset_index()
        if not tdf.empty
        else pd.DataFrame()
    )
    by_exit = (
        tdf["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="n")
        if not tdf.empty
        else pd.DataFrame()
    )
    by_dir = (
        tdf.groupby("direction")
        .agg(n=("ret", "size"), mean_ret=("ret", "mean"), win=("ret", lambda s: float((s > 0).mean())))
        .reset_index()
        if not tdf.empty
        else pd.DataFrame()
    )

    summary = {
        "window": {"start": args.start_date, "end": args.end_date},
        "smooth_cfg": smooth_cfg.__dict__,
        "impulse_cfg": impulse_cfg.__dict__,
        "trade_cfg": {k: getattr(trade_cfg, k) for k in trade_cfg.__dataclass_fields__},
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "equity_end": eq["equity_end"],
        "n_trades": eq["n_trades"],
        "n_days": eq["n_days"],
        "trade_win": eq["trade_win"],
        "avg_trade_ret": eq["avg_trade_ret"],
        "n_uncapped": len(all_trades),
        "by_sleeve": by_sleeve.to_dict(orient="records") if len(by_sleeve) else [],
        "by_exit": by_exit.to_dict(orient="records") if len(by_exit) else [],
        "by_direction": by_dir.to_dict(orient="records") if len(by_dir) else [],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Smooth + Impulse Stock Replay — May–Jul",
        "",
        f"**Total ret: `{eq['total_ret']:+.1%}`** · MaxDD `{eq['maxdd']:+.1%}` · "
        f"trades `{eq['n_trades']}` · win `{eq['trade_win']:.1%}` · avg `{eq['avg_trade_ret']:+.2%}`",
        "",
        "Sizing: 50% day risk split equally across ≤2 names.",
        "",
        "## By sleeve",
        "",
        by_sleeve.to_markdown(index=False) if len(by_sleeve) else "(none)",
        "",
        "## By exit",
        "",
        by_exit.to_markdown(index=False) if len(by_exit) else "(none)",
        "",
        "## By direction",
        "",
        by_dir.to_markdown(index=False) if len(by_dir) else "(none)",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in summary if k not in {"smooth_cfg", "impulse_cfg", "trade_cfg"}}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
