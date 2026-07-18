#!/usr/bin/env python3
"""Build Rule-A-triggered TCN gate dataset (features + smooth-trend labels).

Only samples at Mag7 first Rule-A fires (matches live candidate set). Output parquet
for ``train_tcn_gate.py``. Does not mutate the frozen research baseline.
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
from maga7.common.replay import month_list
from maga7.common.signals import (
    attach_mf_features,
    build_all_first_rule_a_signals,
    load_stock_month_files,
)
from maga7.common.tcn_gate import (
    DEFAULT_CHANNELS,
    TcnGateConfig,
    build_feature_tensor,
    label_smooth_trend,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--start-date", default="2025-07-01")
    ap.add_argument("--end-date", default="2026-07-16")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--horizon-minutes", type=int, default=30)
    ap.add_argument("--breakout-pct", type=float, default=0.004)
    ap.add_argument("--max-adverse-pct", type=float, default=0.002)
    ap.add_argument(
        "--label-mode",
        default="mfe",
        choices=["smooth", "mfe", "soft"],
        help="smooth=MFE+MAE; mfe=MFE only (more learnable); soft=signed ret>0",
    )
    ap.add_argument(
        "--channels",
        default=None,
        help="comma list; default profile/DEFAULT + vol_z,tod for mfe research",
    )
    ap.add_argument(
        "--out",
        default="maga7/results/tcn_gate/dataset_rule_a.parquet",
    )
    args = ap.parse_args()

    prof = load_profile(args.profile)
    prof["date_range"] = {"start": args.start_date, "end": args.end_date}
    cfg = TcnGateConfig.from_profile(prof)
    if args.channels:
        channels = tuple(x.strip() for x in str(args.channels).split(",") if x.strip())
    else:
        base = tuple(cfg.channels or DEFAULT_CHANNELS)
        # research default: add static context channels when using mfe labels
        extra = ("vol_z", "tod") if args.label_mode == "mfe" else ()
        channels = tuple(dict.fromkeys(base + extra))
    window = int(args.window or cfg.window)

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
    if sigs.empty:
        raise SystemExit("no Rule-A signals in range")

    qqq = stock_by.get("QQQ")
    rows = []
    X_list = []
    for r in sigs.itertuples(index=False):
        sdf = stock_by.get(r.symbol)
        x = build_feature_tensor(
            sdf,
            qqq,
            asof_ts=r.sig_ts,
            window=window,
            channels=channels,
            direction=r.dir,
        )
        y = label_smooth_trend(
            sdf,
            asof_ts=r.sig_ts,
            direction=r.dir,
            horizon_minutes=args.horizon_minutes,
            breakout_pct=args.breakout_pct,
            max_adverse_pct=args.max_adverse_pct,
            label_mode=args.label_mode,
        )
        if x is None or y is None:
            continue
        X_list.append(x.reshape(-1))
        rows.append(
            {
                "date": str(r.date),
                "symbol": r.symbol,
                "dir": r.dir,
                "sig_ts": pd.Timestamp(r.sig_ts),
                "label": int(y),
            }
        )

    if not rows:
        raise SystemExit("no labeled samples")
    meta = pd.DataFrame(rows)
    X = np.stack(X_list, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # store flat features + reshape meta in sidecar json
    feat = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    ds = pd.concat([meta.reset_index(drop=True), feat], axis=1)
    ds.to_parquet(out, index=False)
    side = {
        "n": int(len(ds)),
        "pos_rate": float(ds["label"].mean()),
        "window": window,
        "channels": list(channels),
        "n_channels": len(channels),
        "horizon_minutes": args.horizon_minutes,
        "breakout_pct": args.breakout_pct,
        "max_adverse_pct": args.max_adverse_pct,
        "label_mode": args.label_mode,
        "start": args.start_date,
        "end": args.end_date,
        "feature_shape": [window, len(channels)],
    }
    out.with_suffix(".meta.json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(json.dumps(side, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
