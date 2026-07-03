#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
strict replay CLI —— 推理产出 parquet → 验收指标。

(L1 分钟回放;L2 事件回放见 run_event_replay.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.run_replay")


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc strict replay (L1 minute)")
    parser.add_argument("--input", required=True, help="含 edge 列 + exec_* 报价的 parquet")
    parser.add_argument("--output", default=None, help="summary JSON 路径")
    parser.add_argument("--trades", default=None, help="成交明细 parquet")
    parser.add_argument("--long-only", action="store_true", default=None)
    parser.add_argument("--dual-leg", action="store_true", help="使用 call/put/straddle 双头")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(Path(args.input).expanduser())
    df = df.sort_values("timestamp").reset_index(drop=True)
    replay_cfg = qcfg.REPLAY
    if args.long_only is not None:
        from dataclasses import replace
        replay_cfg = replace(replay_cfg, long_only=args.long_only)

    kwargs = {}
    if args.dual_leg:
        kwargs.update(
            call_edge_col=qcfg.CALL_EDGE_COL,
            put_edge_col=qcfg.PUT_EDGE_COL,
            straddle_edge_col=qcfg.STRADDLE_EDGE_COL,
        )

    result = run_strict_replay(
        df,
        qcfg.FILL_MODEL,
        replay_cfg,
        qcfg.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=qcfg.EDGE_Q10_COL,
        **kwargs,
    )
    summary = result.summary()
    logger.info("replay summary: %s", summary)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.trades:
        result.trades_frame().to_parquet(Path(args.trades).expanduser(), index=False)


if __name__ == "__main__":
    main()
