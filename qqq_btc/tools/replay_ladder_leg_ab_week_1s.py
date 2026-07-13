#!/usr/bin/env python3
"""July W1 多日 1s ladder A/B 驱动（不改分钟版；复用 day_1s 单日逻辑）。

用法：
  python qqq_btc/tools/replay_ladder_leg_ab_week_1s.py
  python qqq_btc/tools/replay_ladder_leg_ab_week_1s.py --dates 2026-07-01,2026-07-02
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DAY_SCRIPT = REPO / "qqq_btc/tools/replay_ladder_leg_ab_day_1s.py"
DEFAULT_DATES = [
    "2026-07-01",
    "2026-07-02",
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
]


def summarize_frame(trades: pd.DataFrame, src: str) -> dict:
    sub = trades[trades["source"] == src]
    if sub.empty:
        return {"source": src, "n_signals": 0}
    piv = sub.pivot_table(
        index=["date", "entry_ts", "side"], columns="mode", values="net_return", aggfunc="first"
    )
    if "primary" not in piv.columns or "value_score" not in piv.columns:
        return {"source": src, "n_signals": int(len(piv)), "note": "missing mode"}
    piv = piv.dropna()
    if piv.empty:
        return {"source": src, "n_signals": 0}
    diff = piv["value_score"] - piv["primary"]
    vs = sub[sub["mode"] == "value_score"].set_index(["date", "entry_ts", "side"])
    pr = sub[sub["mode"] == "primary"].set_index(["date", "entry_ts", "side"])
    common = vs.index.intersection(pr.index)
    n_same = int((vs.loc[common, "ticker"].to_numpy() == pr.loc[common, "ticker"].to_numpy()).sum())
    return {
        "source": src,
        "n_signals": int(len(piv)),
        "primary_mean_net": float(piv["primary"].mean()),
        "value_score_mean_net": float(piv["value_score"].mean()),
        "uplift_mean": float(diff.mean()),
        "uplift_median": float(diff.median()),
        "pct_vs_better": float((diff > 1e-9).mean()),
        "pct_vs_worse": float((diff < -1e-9).mean()),
        "n_same_ticker": n_same,
        "n_switched": int(len(common) - n_same),
        "primary_sum": float(piv["primary"].sum()),
        "value_score_sum": float(piv["value_score"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default=",".join(DEFAULT_DATES))
    ap.add_argument(
        "--signals-dir",
        default=str(REPO / "qqq_btc/results/july_w1_ft56_honest_3gate_week"),
    )
    ap.add_argument(
        "--trades-root",
        default="/home/kingfang007/data/new_option_data_s3_trades/QQQ",
    )
    ap.add_argument(
        "--stock-1s-root",
        default="/mnt/s990/data/raw_1s/stocks/QQQ",
    )
    ap.add_argument(
        "--stock-1m",
        default=str(Path.home() / "train_data/spnq_train/QQQ/2026-07.parquet"),
    )
    ap.add_argument(
        "--primary-map",
        default=str(Path.home() / "train_data/locked_targets_map_1dte_jul2026_openwin.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "qqq_btc/results/july_w1_ladder_leg_ab_1s_w1"),
    )
    ap.add_argument("--edge-thresh", type=float, default=0.03)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = args.python

    day_summaries = []
    all_trades = []
    skipped = []

    for date in dates:
        trades_path = Path(args.trades_root) / f"QQQ_{date}.parquet"
        signals = Path(args.signals_dir) / f"signals_{date}.csv"
        stock_1s = Path(args.stock_1s_root) / f"QQQ_{date}.parquet"
        day_out = out_dir / f"day_{date}"
        if not trades_path.exists():
            skipped.append({"date": date, "reason": f"missing trades {trades_path}"})
            print(f"[skip] {date}: no trades")
            continue
        if not signals.exists():
            skipped.append({"date": date, "reason": f"missing signals {signals}"})
            print(f"[skip] {date}: no signals")
            continue

        cmd = [
            py,
            str(DAY_SCRIPT),
            "--date",
            date,
            "--signals",
            str(signals),
            "--trades-1s",
            str(trades_path),
            "--primary-map",
            str(args.primary_map),
            "--stock-1m",
            str(args.stock_1m),
            "--edge-thresh",
            str(args.edge_thresh),
            "--out-dir",
            str(day_out),
        ]
        if stock_1s.exists():
            cmd.extend(["--stock-1s", str(stock_1s)])
        else:
            # day script 需要存在的 path；给一个不存在的占位，load_spot 会 fallback 1m
            cmd.extend(["--stock-1s", str(stock_1s)])

        print(f"======== {date} ========")
        rc = subprocess.call(cmd, cwd=str(REPO))
        if rc != 0:
            skipped.append({"date": date, "reason": f"day script exit {rc}"})
            continue

        day_trades = pd.read_csv(day_out / "trades_ab_1s.csv")
        day_trades["date"] = date
        all_trades.append(day_trades)
        day_sum = json.loads((day_out / "summary_1s.json").read_text())
        day_summaries.append(
            {
                "date": date,
                "live_enter": day_sum.get("live_enter"),
                "edge_dense": day_sum.get("edge_dense"),
                "ladder_n": day_sum.get("ladder_n"),
                "spot_at_lock": day_sum.get("spot_at_lock"),
            }
        )

    if not all_trades:
        print("no trades produced")
        return 1

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(out_dir / "trades_ab_1s_all.csv", index=False)

    summary = {
        "dates": dates,
        "skipped": skipped,
        "by_day": day_summaries,
        "overall_live_enter": summarize_frame(trades, "live_enter"),
        "overall_edge_dense": summarize_frame(trades, "edge_dense"),
        "note": "week driver calls replay_ladder_leg_ab_day_1s.py per day; minute script untouched",
    }
    (out_dir / "summary_1s_week.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir / 'trades_ab_1s_all.csv'}")
    print(f"wrote {out_dir / 'summary_1s_week.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
