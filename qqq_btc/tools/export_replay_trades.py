#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
strict replay 成交导出 —— 供 G3 exit_reason / fill 分布对拍。

用法:
  python qqq_btc/tools/export_replay_trades.py \\
    --parquet /tmp/qqq_infer.parquet \\
    --output /tmp/replay_trades.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--edge-col", default="net_edge")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    result = run_strict_replay(
        df,
        qcfg.FILL_MODEL,
        qcfg.REPLAY,
        qcfg.EXIT_RAILS,
        edge_col=args.edge_col,
        edge_q10_col=qcfg.EDGE_Q10_COL,
        call_edge_col=qcfg.CALL_EDGE_COL,
        put_edge_col=qcfg.PUT_EDGE_COL,
        put_gate_col=qcfg.PUT_GATE_COL,
    )
    rows = []
    for t in result.trades:
        rows.append(
            {
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "leg": t.leg,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "net_return": t.net_return,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
        )
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} trades → {args.output}")
    if len(out):
        print(out["exit_reason"].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    main()
