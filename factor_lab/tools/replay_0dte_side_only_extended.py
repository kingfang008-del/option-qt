#!/usr/bin/env python3
"""Extended side-only 0DTE replay with longer train windows and option-native features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from factor_lab.tools.option_edge_routing_common import (
    build_routing_dataset,
    build_routing_dataset_option_native,
)
from factor_lab.tools.run_0dte_routing_ablations import SideOnlyBundle, grid_search_val, replay_side_only
from factor_lab.tools.replay_0dte_micro_action import ReplayParams, load_contract_minutes, summarize


def load_contract_minutes_1m(option_1m_root: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    import numpy as np

    out: dict[str, pd.DataFrame] = {}
    files = sorted((option_1m_root / "QQQ").glob("QQQ_*.parquet"))
    for f in files:
        date_str = f.stem.replace("QQQ_", "")
        if not (start <= date_str <= end):
            continue
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df["minute_ts"] = df["timestamp"].dt.floor("min")
        if "side" not in df.columns and "ticker" in df.columns:
            from factor_lab.tools.build_0dte_option_edge_labels import parse_side

            df["ticker"] = df["ticker"].astype(str).str.replace("O:", "", regex=False)
            df["side"] = df["ticker"].map(parse_side)
        if "spread_pct" not in df.columns:
            mid = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
            df["spread_pct"] = (pd.to_numeric(df["ask"], errors="coerce") - pd.to_numeric(df["bid"], errors="coerce")) / mid.replace(0, np.nan)
        keep = ["minute_ts", "ticker", "bucket_id", "side", "bid", "ask", "spread_pct"]
        for c in ("trade_volume", "quote_events", "buy_ratio"):
            if c in df.columns:
                keep.append(c)
        cols = [c for c in keep if c in df.columns]
        q = df.sort_values("timestamp").drop_duplicates(["ticker", "minute_ts"], keep="last")[cols].copy()
        for c in ["bid", "ask", "spread_pct"]:
            q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0.0)
        out[date_str] = q.sort_values(["minute_ts", "bucket_id", "ticker"]).reset_index(drop=True)
    return out


def run_experiment(
    *,
    name: str,
    feature_source: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    side_h: int,
    exec_h: int,
    label_dir: Path,
    micro_root: Path,
    option_1m_root: Path,
    quote_source: str,
    fixed_params: dict | None,
) -> dict:
    horizons = [1, 3, 5, 10]
    build = build_routing_dataset_option_native if feature_source == "option_native" else build_routing_dataset
    common = dict(label_dir=label_dir, symbol="QQQ", horizons=horizons)
    if feature_source == "micro":
        train = build(micro_root=micro_root, start=train_start, end=train_end, **common)
        val = build(micro_root=micro_root, start=val_start, end=val_end, **common)
        test = build(micro_root=micro_root, start=test_start, end=test_end, **common)
        train_full = build(micro_root=micro_root, start=train_start, end=val_end, **common)
    else:
        train = build(start=train_start, end=train_end, **common)
        val = build(start=val_start, end=val_end, **common)
        test = build(start=test_start, end=test_end, **common)
        train_full = build(start=train_start, end=val_end, **common)

    features = train.attrs["features"]
    if quote_source == "1m":
        quote_val = load_contract_minutes_1m(option_1m_root, val_start, val_end)
        quote_test = load_contract_minutes_1m(option_1m_root, test_start, test_end)
    else:
        quote_val = load_contract_minutes(micro_root, val_start, val_end)
        quote_test = load_contract_minutes(micro_root, test_start, test_end)

    bundle = SideOnlyBundle(side_h=side_h)
    bundle.fit(train, features)
    pred_val = bundle.predict_frame(val, features)

    if fixed_params:
        best = {**fixed_params, "side_h": side_h, "exec_h": exec_h, "mode": "side_only"}
    else:
        best, _ = grid_search_val(
            pred_val, quote_val, mode="side_only", exec_h=exec_h, side_h=side_h, use_bucket=False
        )
        if not best:
            return {"name": name, "error": "no val trades"}

    bundle.fit(train_full, features)
    pred_test = bundle.predict_frame(test, features)
    params = ReplayParams(
        horizon=exec_h,
        max_hold=exec_h,
        entry_quantile=float(best["entry_quantile"]),
        take_profit=float(best["take_profit"]),
        stop_loss=float(best["stop_loss"]),
        max_trades_per_day=8,
        cooldown=3,
    )
    test_summary, test_trades = replay_side_only(pred_test, quote_test, params)
    monthly = {}
    if not test_trades.empty:
        for mon, g in test_trades.groupby("month"):
            monthly[mon] = summarize(g, params)

    val_summary, _ = replay_side_only(pred_val, quote_val, params)
    return {
        "name": name,
        "feature_source": feature_source,
        "train_range": [train_start, train_end],
        "val_range": [val_start, val_end],
        "test_range": [test_start, test_end],
        "train_rows": int(len(train)),
        "train_full_rows": int(len(train_full)),
        "side_h": side_h,
        "exec_h": exec_h,
        "fixed_params": fixed_params is not None,
        "selected_val": val_summary if fixed_params else best,
        "val_replay": val_summary,
        "test": test_summary,
        "test_monthly": monthly,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--micro-root", default="/mnt/s990/data/microstructure/qqq_0dte_api_ladder")
    p.add_argument("--option-1m-root", default="/mnt/s990/data/raw_1m/dte0_options")
    p.add_argument("--label-dir", default=str(Path.home() / "train_data/option_edge_labels_0dte"))
    p.add_argument("--output", default="factor_lab/results/0dte_side_only_extended.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir).expanduser()
    micro_root = Path(args.micro_root)
    option_1m_root = Path(args.option_1m_root)
    fixed = {"entry_quantile": 0.975, "take_profit": 0.35, "stop_loss": -0.25}

    experiments = [
        dict(
            name="micro_train_q1_val_mar_test_q2",
            feature_source="micro",
            train_start="2026-01-01", train_end="2026-02-28",
            val_start="2026-03-01", val_end="2026-03-31",
            test_start="2026-04-01", test_end="2026-06-30",
            quote_source="micro",
            fixed_params=None,
        ),
        dict(
            name="micro_train_janmar_val_apr_test_mayjun",
            feature_source="micro",
            train_start="2026-01-01", train_end="2026-03-31",
            val_start="2026-04-01", val_end="2026-04-30",
            test_start="2026-05-01", test_end="2026-06-30",
            quote_source="micro",
            fixed_params=None,
        ),
        dict(
            name="micro_train_janmar_fixed_test_q2",
            feature_source="micro",
            train_start="2026-01-01", train_end="2026-03-31",
            val_start="2026-04-01", val_end="2026-04-30",
            test_start="2026-04-01", test_end="2026-06-30",
            quote_source="micro",
            fixed_params=fixed,
        ),
        dict(
            name="option_native_train_2025h2_2026q1_val_apr_test_mayjun",
            feature_source="option_native",
            train_start="2025-07-01", train_end="2026-03-31",
            val_start="2026-04-01", val_end="2026-04-30",
            test_start="2026-05-01", test_end="2026-06-30",
            quote_source="1m",
            fixed_params=None,
        ),
        dict(
            name="option_native_train_2025h2_2026q1_fixed_test_q2",
            feature_source="option_native",
            train_start="2025-07-01", train_end="2026-03-31",
            val_start="2026-04-01", val_end="2026-04-30",
            test_start="2026-04-01", test_end="2026-06-30",
            quote_source="1m",
            fixed_params=fixed,
        ),
    ]

    results = []
    for exp in experiments:
        print(f"=== {exp['name']} ===")
        try:
            out = run_experiment(
                side_h=1,
                exec_h=5,
                label_dir=label_dir,
                micro_root=micro_root,
                option_1m_root=option_1m_root,
                **exp,
            )
        except Exception as exc:
            out = {"name": exp["name"], "error": str(exc)}
        results.append(out)
        if "error" in out:
            print(f"  ERROR: {out['error']}")
            continue
        t = out["test"]
        print(
            f"  train_rows={out['train_rows']} val={out['val_replay']['total_net_return']:.3f} "
            f"test trades={t['trades']} total={t['total_net_return']:.3f} pf={t['profit_factor']:.3f}"
        )

    payload = {"side_h": 1, "exec_h": 5, "experiments": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
