#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件驱动回放 CLI —— L1 分钟 / L2 秒级验收。

用法:
  # L1: 与 run_replay.py 等价(统一走 ReplaySession)
  python qqq_btc/tools/run_event_replay.py \\
      --input /tmp/qqq_infer.parquet --output /tmp/replay.json

  # L2: 分钟信号 + 1s tick(首 tick 成交 + 灾难止损)
  python qqq_btc/tools/run_event_replay.py \\
      --input /tmp/qqq_infer.parquet \\
      --ticks ~/train_data/sniper_1s/QQQ_2026-01.parquet \\
      --fill-timing first_tick \\
      --output /tmp/replay_event.json \\
      --trades /tmp/replay_trades.parquet

  # S4 同源融合:alpha(+60s) + 1s option 一次刷入
  python qqq_btc/tools/run_event_replay.py \\
      --from-s4-sqlite preprocess/backtest/history_sqlite_1s/market_20260102.db \\
      --symbol QQQ --signal-at minute_open --fill-timing first_tick
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

from qqq_btc.common.event_replay import (
    EventReplayConfig,
    FillTiming,
    SignalTiming,
    compare_minute_vs_event,
    run_event_replay,
)
from qqq_btc.common.replay_io import build_s4_bundle_from_sqlite, load_ticks, load_ticks_sqlite
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.run_event_replay")


def _load_frame(path: Path) -> pd.DataFrame:
    path = path.expanduser()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _replay_kwargs(args) -> dict:
    kwargs = {"edge_col": "net_edge", "edge_q10_col": qcfg.EDGE_Q10_COL}
    if args.dual_leg:
        kwargs.update(
            call_edge_col=qcfg.CALL_EDGE_COL,
            put_edge_col=qcfg.PUT_EDGE_COL,
            straddle_edge_col=qcfg.STRADDLE_EDGE_COL,
        )
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="qqq_btc 事件驱动回放(L1/L2/S4)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", default=None, help="分钟 infer parquet(csv)")
    src.add_argument(
        "--from-s4-sqlite",
        default=None,
        help="S4 风格 SQLite:alpha+1s option 融合后回放(无需单独 --input)",
    )
    parser.add_argument("--ticks", default=None, help="可选 1s tick parquet(csv)")
    parser.add_argument(
        "--ticks-sqlite",
        default=None,
        help="从 S4 SQLite option_snapshots_1s 加载 tick(与 --ticks 二选一)",
    )
    parser.add_argument("--symbol", default="QQQ", help="--ticks-sqlite 时过滤标的")
    parser.add_argument("--output", default=None, help="summary JSON")
    parser.add_argument("--trades", default=None, help="成交明细 parquet")
    parser.add_argument(
        "--fill-timing",
        choices=[m.value for m in FillTiming],
        default=None,
    )
    parser.add_argument(
        "--signal-at",
        choices=[m.value for m in SignalTiming],
        default=None,
        help="minute_open=S4 分钟首 tick 发信号; minute_close=infer parquet 默认",
    )
    parser.add_argument("--no-tick-disaster", action="store_true", help="L2 关闭 tick 灾难止损")
    parser.add_argument("--tick-smooth", type=int, default=3, help="灾难止损 MTM 平滑窗口(秒)")
    parser.add_argument("--compare", action="store_true", help="L1 vs L2 对拍(需 --ticks)")
    parser.add_argument("--dual-leg", action="store_true")
    parser.add_argument("--long-only", action="store_true", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    signal_at = SignalTiming(args.signal_at) if args.signal_at else SignalTiming.MINUTE_CLOSE
    fill_timing = FillTiming(args.fill_timing) if args.fill_timing else FillTiming.MINUTE_CLOSE

    if args.from_s4_sqlite:
        bundle = build_s4_bundle_from_sqlite(args.from_s4_sqlite, symbol=args.symbol)
        minute_df = bundle.minute_df
        tick_df = bundle.tick_df
        if signal_at == SignalTiming.MINUTE_CLOSE:
            signal_at = SignalTiming.MINUTE_OPEN
        if fill_timing == FillTiming.MINUTE_CLOSE:
            fill_timing = FillTiming.FIRST_TICK
        logger.info(
            "S4 bundle: %d minutes, %d ticks, alpha_delay=%ss",
            len(minute_df), len(tick_df), bundle.alpha_delay_seconds,
        )
    else:
        minute_df = _load_frame(Path(args.input))
        if args.ticks_sqlite:
            tick_df = load_ticks_sqlite(Path(args.ticks_sqlite), symbol=args.symbol)
        elif args.ticks:
            tick_df = load_ticks(Path(args.ticks))
        else:
            tick_df = None

    replay_cfg = qcfg.REPLAY
    if args.long_only is not None:
        from dataclasses import replace

        replay_cfg = replace(replay_cfg, long_only=args.long_only)

    kw = _replay_kwargs(args)

    if args.compare:
        if tick_df is None:
            parser.error("--compare 需要 --ticks 或 --ticks-sqlite")
        report = compare_minute_vs_event(
            minute_df, tick_df, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS, **kw
        )
        logger.info("compare: %s", json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return

    event_cfg = EventReplayConfig(
        fill_timing=fill_timing,
        signal_timing=signal_at,
        tick_disaster_stop=not args.no_tick_disaster,
        tick_smooth_n=args.tick_smooth,
    )

    result = run_event_replay(
        minute_df,
        qcfg.FILL_MODEL,
        replay_cfg,
        qcfg.EXIT_RAILS,
        tick_df=tick_df,
        event_cfg=event_cfg,
        **kw,
    )
    summary = result.summary()
    summary["mode"] = "L2_event" if tick_df is not None else "L1_minute"
    summary["fill_timing"] = event_cfg.fill_timing.value
    summary["signal_timing"] = event_cfg.signal_timing.value
    logger.info("replay summary: %s", summary)

    if args.output:
        with open(Path(args.output).expanduser(), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.trades:
        result.trades_frame().to_parquet(Path(args.trades).expanduser(), index=False)
        logger.info("trades → %s", args.trades)


if __name__ == "__main__":
    main()
