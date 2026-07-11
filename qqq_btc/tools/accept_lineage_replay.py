#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案 B 验收：新血缘 replay 后，按月账户收益符号检查（默认要求 6 月为正）。

用法:
  python qqq_btc/tools/accept_lineage_replay.py \\
      --trades qqq_btc/results/v4_lineage_b/replay_trades.parquet \\
      --require-month 2026-06 --min-acct 0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def month_acct(trades: pd.DataFrame, position_frac: float = 0.25) -> dict[str, dict]:
    tr = trades.copy()
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"])
    if tr["entry_ts"].dt.tz is None:
        tr["entry_ts"] = tr["entry_ts"].dt.tz_localize("America/New_York")
    else:
        tr["entry_ts"] = tr["entry_ts"].dt.tz_convert("America/New_York")
    tr["month"] = tr["entry_ts"].dt.strftime("%Y-%m")
    out: dict[str, dict] = {}
    for m, g in tr.groupby("month"):
        g = g.sort_values("entry_ts")
        eq = 1.0
        for r in g["net_return"].astype(float):
            eq *= 1.0 + float(position_frac) * float(r)
        out[str(m)] = {
            "trades": int(len(g)),
            "hit_rate": float((g["net_return"] > 0).mean()) if len(g) else 0.0,
            "sum_net": float(g["net_return"].sum()),
            "acct": float(eq - 1.0),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="lineage-B replay acceptance")
    p.add_argument("--trades", required=True)
    p.add_argument("--summary", default=None, help="可选 replay_summary.json，写入验收结果")
    p.add_argument("--position-frac", type=float, default=0.25)
    p.add_argument("--require-month", action="append", default=[], help="可重复，如 2026-06")
    p.add_argument("--min-acct", type=float, default=0.0, help="要求月份账户复利下限（默认 ≥0）")
    p.add_argument("--out", default=None, help="写 acceptance.json")
    args = p.parse_args()

    trades = pd.read_parquet(Path(args.trades).expanduser())
    monthly = month_acct(trades, position_frac=args.position_frac)
    require = args.require_month or ["2026-06"]

    failures = []
    for m in require:
        row = monthly.get(m)
        if row is None:
            failures.append({"month": m, "reason": "no_trades"})
            continue
        if row["acct"] < args.min_acct:
            failures.append(
                {
                    "month": m,
                    "reason": "acct_below_min",
                    "acct": row["acct"],
                    "min_acct": args.min_acct,
                }
            )

    report = {
        "ok": len(failures) == 0,
        "position_frac": args.position_frac,
        "require_month": require,
        "min_acct": args.min_acct,
        "monthly": monthly,
        "failures": failures,
    }
    if args.summary and Path(args.summary).expanduser().exists():
        report["replay_summary"] = json.loads(Path(args.summary).expanduser().read_text())

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        outp = Path(args.out).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        print(f"wrote {outp}")
    print(text)
    if failures:
        print("ACCEPTANCE FAILED", file=sys.stderr)
        sys.exit(1)
    print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
