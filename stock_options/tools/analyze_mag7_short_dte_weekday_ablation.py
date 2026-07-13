#!/usr/bin/env python3
"""Weekday ablation on existing MAG7 short-DTE state-gate score panels.

Joins cached ``score_dataset_*.parquet`` with the weekday-enriched locked map
and reports mean label return / hit-rate by expiry_weekday (and trade_weekday).

Does not refit scorers. Useful after dte0 stability to check whether Mon/Wed/Fri
expiry buckets behave differently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stock_options.common.short_dte_config import DEFAULT_LOCKED_MAP_WEEKDAY, RESEARCH_START


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, choices=["NVDA", "TSLA"])
    p.add_argument("--selected-dte", type=int, default=0, choices=[0, 1, 2])
    p.add_argument(
        "--panel-dir",
        default="",
        help="default: stock_options/results/mag7_state_gate_{sym}_dte{dte}/cache",
    )
    p.add_argument("--locked-map-weekday", default=str(DEFAULT_LOCKED_MAP_WEEKDAY))
    p.add_argument("--target", default="target_exec_ret_30s")
    p.add_argument("--start-date", default=RESEARCH_START)
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument(
        "--output-dir",
        default="",
        help="default: stock_options/results/mag7_weekday_ablation_{sym}_dte{dte}",
    )
    return p.parse_args()


def load_panels(panel_dir: Path, start: str, end: str) -> pd.DataFrame:
    files = sorted(panel_dir.glob("score_dataset_*.parquet"))
    if not files:
        raise SystemExit(f"no score panels in {panel_dir}")
    frames = []
    for fp in files:
        df = pd.read_parquet(fp)
        if "date_str" not in df.columns:
            raise SystemExit(f"missing date_str in {fp}")
        df = df[(df["date_str"] >= start) & (df["date_str"] <= end)]
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit("no panel rows after date filter")
    return pd.concat(frames, ignore_index=True)


def day_weekday_lookup(locked: Path, symbol: str, dte: int) -> pd.DataFrame:
    m = pd.read_parquet(locked)
    m = m[(m["symbol"] == symbol) & (m["selected_dte"] == dte)].copy()
    if m.empty:
        raise SystemExit(f"no locked rows for {symbol} dte={dte}")
    if "expiry_weekday_name" not in m.columns:
        from stock_options.common.short_dte_config import enrich_locked_map_weekdays

        m = enrich_locked_map_weekdays(m)
    keys = [
        "date_str",
        "trade_weekday",
        "trade_weekday_name",
        "expiry_weekday",
        "expiry_weekday_name",
        "expiration",
        "is_mon_wed_fri_expiry",
    ]
    return m[keys].drop_duplicates("date_str")


def summarize(panel: pd.DataFrame, target: str, by: list[str]) -> pd.DataFrame:
    g = panel.groupby(by, dropna=False)
    out = g.agg(
        n_rows=(target, "size"),
        n_days=("date_str", "nunique"),
        mean_ret=(target, "mean"),
        median_ret=(target, "median"),
        hit_rate=(target, lambda s: float((s > 0).mean())),
        p10=(target, lambda s: float(s.quantile(0.1))),
        p90=(target, lambda s: float(s.quantile(0.9))),
    ).reset_index()
    return out.sort_values(["n_rows"], ascending=False)


def main() -> None:
    args = parse_args()
    sym = args.symbol.upper()
    dte = int(args.selected_dte)
    panel_dir = Path(
        args.panel_dir
        or f"stock_options/results/mag7_state_gate_{sym.lower()}_dte{dte}/cache"
    )
    out_dir = Path(
        args.output_dir
        or f"stock_options/results/mag7_weekday_ablation_{sym.lower()}_dte{dte}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panels(panel_dir, args.start_date, args.end_date)
    if "selected_dte" in panel.columns:
        panel = panel[panel["selected_dte"] == dte]
    lookup = day_weekday_lookup(Path(args.locked_map_weekday), sym, dte)
    panel = panel.merge(lookup, on="date_str", how="left")
    miss = float(panel["expiry_weekday_name"].isna().mean())
    if miss > 0.05:
        print(f"warning: {miss:.1%} rows missing expiry weekday join", flush=True)

    by_exp = summarize(panel, args.target, ["expiry_weekday_name"])
    by_trade = summarize(panel, args.target, ["trade_weekday_name"])
    by_both = summarize(panel, args.target, ["trade_weekday_name", "expiry_weekday_name"])
    by_exp.to_csv(out_dir / "by_expiry_weekday.csv", index=False)
    by_trade.to_csv(out_dir / "by_trade_weekday.csv", index=False)
    by_both.to_csv(out_dir / "by_trade_x_expiry_weekday.csv", index=False)

    summary = {
        "symbol": sym,
        "selected_dte": dte,
        "target": args.target,
        "rows": int(len(panel)),
        "days": int(panel["date_str"].nunique()),
        "join_missing_pct": miss,
        "by_expiry_weekday": by_exp.to_dict("records"),
        "by_trade_weekday": by_trade.to_dict("records"),
        "files": {
            "by_expiry_weekday": str(out_dir / "by_expiry_weekday.csv"),
            "by_trade_weekday": str(out_dir / "by_trade_weekday.csv"),
            "by_trade_x_expiry_weekday": str(out_dir / "by_trade_x_expiry_weekday.csv"),
            "summary": str(out_dir / "summary.json"),
        },
        "note": (
            "Unconditional panel means — not rule-gated. Use to check whether "
            "Mon/Wed/Fri expiry buckets need separate calibration before mixing."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("by expiry weekday:")
    print(by_exp.to_string(index=False))
    print("\nby trade weekday:")
    print(by_trade.to_string(index=False))
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
