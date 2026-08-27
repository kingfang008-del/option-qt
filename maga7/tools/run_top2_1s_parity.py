#!/usr/bin/env python3
"""Top2 + Impulse-MAE: 1m path vs 1s path parity.

Design (fair test of "1m map is expired"):
  - Detect seats on 1m bars aggregated from the same 1s feed (frozen funnel).
  - Simulate exits twice: on 1m bars vs native 1s bars.
  - Impulse-only early MAE FD; Smooth keeps baseline trail.

If 1s exits are not materially better, finer data alone does not fix the strategy.
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
from maga7.common.decision_funnel import (
    FUNNEL_VERSION,
    FROZEN_TRADE,
    FunnelConfig,
    day_decision_seats,
)
from maga7.common.failure_detector import (
    FailureDetectorConfig,
    failure_cfg_for_sleeve,
    simulate_stock_with_failure,
)
from maga7.common.replay import month_list
from maga7.common.signals import attach_mf_features, load_stock_month_files
from maga7.common.smooth_trend import SmoothStockTradeConfig
from maga7.tools.run_smooth_impulse_stock_replay import SYMS, _equity

FOLDS = [
    {"name": "fold_2025h1", "start": "2025-04-01", "end": "2025-06-30"},
    {"name": "fold_2025h2", "start": "2025-10-01", "end": "2025-12-31"},
    {"name": "fold_2026h1", "start": "2026-04-01", "end": "2026-07-17"},
]


def _trade_cfg() -> SmoothStockTradeConfig:
    return SmoothStockTradeConfig(
        max_hold_minutes=int(FROZEN_TRADE.max_hold_minutes),
        break_max_adverse=float(FROZEN_TRADE.break_max_adverse),
        break_min_up_frac=float(FROZEN_TRADE.break_min_up_frac),
        break_lookback=int(FROZEN_TRADE.break_lookback)
        if hasattr(FROZEN_TRADE, "break_lookback")
        else 10,
        max_positions=2,
        first_per_symbol_dir=True,
        prefer_smooth_over_impulse=True,
    )


def _fd_impulse_mae_only(sleeve: str) -> FailureDetectorConfig:
    """Promoted Phase3: Impulse MAE only; Smooth FD off."""
    s = str(sleeve).lower()
    if s != "impulse":
        return FailureDetectorConfig(enabled=False, sleeve=s)
    base = failure_cfg_for_sleeve("impulse")
    return replace(
        base,
        early_giveback=9.0,
        path_min_up_frac=-1.0,
        structure_lookback=0,
        lose_open=False,
        lose_vwap=False,
    )


def _prep_1m(day_1s: pd.DataFrame, *, symbol: str, date: str) -> pd.DataFrame:
    """Fast left-labeled 1m OHLCV (equivalent intent to aggregate_1s_to_1m)."""
    if day_1s.empty:
        return pd.DataFrame()
    x = day_1s.set_index("timestamp").sort_index()
    o = "open" if "open" in x.columns else "close"
    h = "high" if "high" in x.columns else "close"
    l = "low" if "low" in x.columns else "close"
    vcol = "volume" if "volume" in x.columns else None
    agg = {"open": (o, "first"), "high": (h, "max"), "low": (l, "min"), "close": ("close", "last")}
    if vcol:
        agg["volume"] = (vcol, "sum")
    m1 = x.resample("1min", label="left", closed="left").agg(**agg).dropna(subset=["close"]).reset_index()
    if m1.empty:
        return m1
    if "volume" not in m1.columns:
        m1["volume"] = 0.0
    m1["date"] = date
    m1["symbol"] = symbol
    return m1


def _simulate_seat(
    day: pd.DataFrame,
    seat: dict,
    *,
    trade_cfg: SmoothStockTradeConfig,
    date: str,
    resolution: str,
) -> dict | None:
    sleeve = str(seat["sleeve"])
    fd = _fd_impulse_mae_only(sleeve)
    bar_seconds = 1 if resolution == "1s" else 60
    sim = simulate_stock_with_failure(
        day,
        entry_ts=seat["detect_ts"],
        direction=seat["direction"],
        trade_cfg=trade_cfg,
        fd_cfg=fd,
        date=date,
        sleeve=sleeve,
        bar_seconds=bar_seconds,
        trail_arm_minutes=float(trade_cfg.break_lookback),
    )
    if sim is None:
        return None
    return {
        "date": date,
        "symbol": str(seat["symbol"]).upper(),
        "direction": seat["direction"],
        "sleeve": sleeve,
        "seat_rank": int(seat["seat_rank"]),
        "detect_ts": str(seat["detect_ts"]),
        "score": float(seat["score"]),
        "resolution": resolution,
        **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in sim.items()},
    }


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "total_ret": 0.0, "maxdd": 0.0, "win": None, "avg": None}
    eq = _equity(trades, frac=0.5)
    return {
        "n": int(len(trades)),
        "total_ret": eq["total_ret"],
        "maxdd": eq["maxdd"],
        "win": eq["trade_win"],
        "avg": eq["avg_trade_ret"],
        "fd_fire_rate": float(trades["fd_fired"].mean()) if "fd_fired" in trades.columns else 0.0,
        "by_sleeve_win": trades.groupby("sleeve")["ret"]
        .apply(lambda s: float((s > 0).mean()))
        .to_dict()
        if "sleeve" in trades.columns
        else {},
        "by_exit": trades["exit_reason"].value_counts().to_dict()
        if "exit_reason" in trades.columns
        else {},
    }


def _pairwise(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    keys = ["date", "symbol", "direction", "detect_ts"]
    if a.empty or b.empty:
        return {}
    aa = a.set_index(keys)
    bb = b.set_index(keys)
    common = aa.index.intersection(bb.index)
    if len(common) == 0:
        return {"n_paired": 0}
    ra = aa.loc[common, "ret"].astype(float)
    rb = bb.loc[common, "ret"].astype(float)
    ha = aa.loc[common, "hold_minutes"].astype(float)
    hb = bb.loc[common, "hold_minutes"].astype(float)
    delta = rb - ra
    return {
        "n_paired": int(len(common)),
        "mean_ret_1m": float(ra.mean()),
        "mean_ret_1s": float(rb.mean()),
        "mean_delta_1s_minus_1m": float(delta.mean()),
        "sum_delta": float(delta.sum()),
        "win_1m": float((ra > 0).mean()),
        "win_1s": float((rb > 0).mean()),
        "median_hold_1m": float(ha.median()),
        "median_hold_1s": float(hb.median()),
        "pct_1s_better": float((delta > 1e-6).mean()),
        "pct_1s_worse": float((delta < -1e-6).mean()),
        "same_exit_reason": float(
            (aa.loc[common, "exit_reason"].astype(str) == bb.loc[common, "exit_reason"].astype(str)).mean()
        ),
    }


def run_from_1s(
    stock_1s_root: Path,
    *,
    start: str,
    end: str,
    trade_cfg: SmoothStockTradeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Detect on 1m←1s; simulate on 1m and 1s."""
    funnel = FunnelConfig()
    # union of dates from NVDA as calendar proxy
    cal = sorted(
        p.stem.split("_", 1)[1]
        for p in (stock_1s_root / "NVDA").glob("NVDA_*.parquet")
        if start <= p.stem.split("_", 1)[1] <= end
    )
    rows_1m: list[dict] = []
    rows_1s: list[dict] = []
    n_missing = 0
    seat_n = 0
    for i, date in enumerate(cal):
        if i % 40 == 0:
            print(f"[1s-parity] {date} ({i+1}/{len(cal)})", flush=True)
        day_1s_by: dict[str, pd.DataFrame] = {}
        day_1m_by: dict[str, pd.DataFrame] = {}
        for sym in SYMS:
            d1s = load_stock_1s_day(stock_1s_root, sym, date)
            if d1s.empty:
                n_missing += 1
                continue
            # RTH slice for exit sim (keep premarket out of path)
            d1s = d1s.copy()
            d1s["date"] = date
            hm = d1s["timestamp"].dt.hour * 60 + d1s["timestamp"].dt.minute
            d1s_rth = d1s[(hm >= 9 * 60 + 30) & (hm < 16 * 60)].reset_index(drop=True)
            if d1s_rth.empty:
                continue
            day_1s_by[sym] = d1s_rth
            m1 = _prep_1m(d1s_rth, symbol=sym, date=date)
            if not m1.empty:
                day_1m_by[sym] = m1
        if len(day_1m_by) < 2:
            continue
        seats, _ = day_decision_seats(day_1m_by, date=date, cfg=funnel)
        seat_n += len(seats)
        for seat in seats:
            sym = str(seat["symbol"]).upper()
            if sym not in day_1m_by or sym not in day_1s_by:
                continue
            r1m = _simulate_seat(
                day_1m_by[sym], seat, trade_cfg=trade_cfg, date=date, resolution="1m"
            )
            r1s = _simulate_seat(
                day_1s_by[sym], seat, trade_cfg=trade_cfg, date=date, resolution="1s"
            )
            if r1m:
                rows_1m.append(r1m)
            if r1s:
                rows_1s.append(r1s)
    meta = {"n_calendar_days": len(cal), "n_seats": seat_n, "n_missing_sym_days": n_missing}
    return pd.DataFrame(rows_1m), pd.DataFrame(rows_1s), meta


def run_cache_1m_control(
    stock_root: Path,
    *,
    start: str,
    end: str,
    trade_cfg: SmoothStockTradeConfig,
) -> pd.DataFrame:
    """Optional: same logic on research 1m cache (spnq_train)."""
    months = month_list(start, end)
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMS:
        raw = load_stock_month_files(stock_root, sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"].astype(str) >= start) & (raw["date"].astype(str) <= end)]
        data[sym] = attach_mf_features(raw)
    funnel = FunnelConfig()
    dates: set[str] = set()
    for df in data.values():
        dates.update(df["date"].astype(str).unique().tolist())
    rows = []
    for date in sorted(d for d in dates if start <= d <= end):
        day_by = {s: df[df["date"].astype(str) == date] for s, df in data.items()}
        day_by = {s: d for s, d in day_by.items() if not d.empty}
        seats, _ = day_decision_seats(day_by, date=date, cfg=funnel)
        for seat in seats:
            sym = str(seat["symbol"]).upper()
            if sym not in day_by:
                continue
            r = _simulate_seat(
                day_by[sym], seat, trade_cfg=trade_cfg, date=date, resolution="1m_cache"
            )
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default=(
            "maga7/CONFIG/strategy_profiles/"
            "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
        ),
    )
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument(
        "--out",
        default="/mnt/s990/data/maga7/results/top2_1s_parity_v1",
    )
    ap.add_argument(
        "--skip-cache-control",
        action="store_true",
        help="Skip spnq_train 1m control replay",
    )
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    stock_1s = Path(prof["_paths"].get("stock_1s_root") or "/mnt/s990/data/raw_1s/stocks").expanduser()
    stock_1m = Path(prof["_paths"]["stock_root"]).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trade_cfg = _trade_cfg()

    print(f"[run] 1s root={stock_1s}", flush=True)
    t1m, t1s, meta = run_from_1s(
        stock_1s, start=args.start_date, end=args.end_date, trade_cfg=trade_cfg
    )
    t1m.to_parquet(out / "trades_1m_from_1s.parquet", index=False)
    t1s.to_parquet(out / "trades_1s.parquet", index=False)

    cache_df = pd.DataFrame()
    if not args.skip_cache_control:
        print("[run] 1m cache control", flush=True)
        cache_df = run_cache_1m_control(
            stock_1m, start=args.start_date, end=args.end_date, trade_cfg=trade_cfg
        )
        cache_df.to_parquet(out / "trades_1m_cache.parquet", index=False)

    sm_1m = _summarize(t1m)
    sm_1s = _summarize(t1s)
    sm_cache = _summarize(cache_df) if not cache_df.empty else {}
    pair = _pairwise(t1m, t1s)

    fold_rows = []
    for fold in FOLDS:
        a = t1m[(t1m["date"] >= fold["start"]) & (t1m["date"] <= fold["end"])]
        b = t1s[(t1s["date"] >= fold["start"]) & (t1s["date"] <= fold["end"])]
        sa, sb = _summarize(a), _summarize(b)
        fold_rows.append(
            {
                "fold": fold["name"],
                "ret_1m": sa["total_ret"],
                "ret_1s": sb["total_ret"],
                "win_1m": sa["win"],
                "win_1s": sb["win"],
                "maxdd_1m": sa["maxdd"],
                "maxdd_1s": sb["maxdd"],
                "1s_better_ret": bool(sb["total_ret"] > sa["total_ret"]),
                "pair": _pairwise(a, b),
            }
        )

    n_better = sum(1 for r in fold_rows if r["1s_better_ret"])
    # materiality: full-window win lift ≥2pp or ret lift ≥1pp
    win_lift = (sm_1s.get("win") or 0) - (sm_1m.get("win") or 0)
    ret_lift = (sm_1s.get("total_ret") or 0) - (sm_1m.get("total_ret") or 0)
    if ret_lift >= 0.01 and n_better >= 2:
        verdict = "1S_HELPS"
    elif abs(ret_lift) < 0.005 and abs(win_lift) < 0.02:
        verdict = "NO_MATERIAL_CHANGE"
    elif ret_lift > 0:
        verdict = "MARGINAL"
    else:
        verdict = "1S_NOT_BETTER"

    summary = {
        "funnel_version": FUNNEL_VERSION,
        "design": "detect_1m_from_1s__exit_compare_1m_vs_1s__fd_impulse_mae_only",
        "verdict": verdict,
        "meta": meta,
        "full_1m_from_1s": sm_1m,
        "full_1s": sm_1s,
        "full_1m_cache": sm_cache,
        "pairwise_1s_vs_1m": pair,
        "folds": fold_rows,
        "win_lift_pp": win_lift,
        "ret_lift": ret_lift,
        "n_folds_1s_better": n_better,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Top2 1s Parity (detect 1m ← 1s, exit 1m vs 1s)",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        f"- Seats: `{meta.get('n_seats')}` · calendar days `{meta.get('n_calendar_days')}`",
        f"- Full ret 1m / 1s: `{sm_1m.get('total_ret'):+.2%}` / `{sm_1s.get('total_ret'):+.2%}` "
        f"(lift `{ret_lift:+.2%}`)",
        f"- Full win 1m / 1s: `{sm_1m.get('win')}` / `{sm_1s.get('win')}` "
        f"(lift `{win_lift:+.1%}`)",
        f"- OOS folds where 1s ret better: `{n_better}/3`",
        "",
        "## Pairwise (same seats)",
        "",
        "```",
        json.dumps(pair, indent=2),
        "```",
        "",
        "## Folds",
        "",
        "```",
        pd.DataFrame(
            [
                {
                    "fold": r["fold"],
                    "ret_1m": r["ret_1m"],
                    "ret_1s": r["ret_1s"],
                    "win_1m": r["win_1m"],
                    "win_1s": r["win_1s"],
                    "1s_better": r["1s_better_ret"],
                }
                for r in fold_rows
            ]
        ).to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Detection clock stays 1m (frozen funnel lookbacks are bar-counts).",
        "- Exit/FD on 1s tests whether second-level path changes PnL.",
        "- `NO_MATERIAL_CHANGE` ⇒ need trade-tape/L2 alpha or logic change, not just 1s exits.",
        "- `1S_HELPS` ⇒ promote 1s exit clock; then research tape features.",
        "",
    ]
    if sm_cache:
        lines[5:5] = [
            f"- Control 1m cache ret/win: `{sm_cache.get('total_ret'):+.2%}` / `{sm_cache.get('win')}`",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"verdict": verdict, "ret_lift": ret_lift, "win_lift_pp": win_lift, "n_folds_1s_better": n_better, **{k: summary[k] for k in ("full_1m_from_1s", "full_1s")}}, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
