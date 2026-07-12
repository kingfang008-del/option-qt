#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc OMS 启动入口 —— 在 legacy ExecutionEngineV8 前注入 fill_model / disaster tick exit。

    cd New_Pro/baseline_qqq
    QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_exec_qqq.py

环境变量:
    QQQ_BTC_LIVE=1              启用集成(必须)
    QQQ_BTC_TICK_EXITS=disaster_only | legacy | off
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "New_Pro" / "baseline_qqq"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

import baseline_paths  # noqa: E402,F401

from qqq_btc.live.bootstrap import bootstrap_qqq_btc_live

bootstrap_qqq_btc_live(patch_oms=True)

from execution_engine_v8 import ExecutionEngineV8  # noqa: E402

try:
    from config import LOG_DIR, TARGET_SYMBOLS, RUN_MODE, IS_SIMULATED, TRADING_ENABLED
except ImportError:
    LOG_DIR = Path.home() / "quant_project/logs"
    TARGET_SYMBOLS = ["QQQ"]
    RUN_MODE = "REALTIME_DRY"
    IS_SIMULATED = False
    TRADING_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [RUN_EXEC_QQQ] - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "LiveRunnerExecQqqBtc.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("LiveRunnerExecQqqBtc")


async def main() -> None:
    print("\n" + "=" * 60)
    print("🚀 qqq_btc Execution Engine (OMS + fill_model 0.775)")
    print("=" * 60 + "\n")
    logger.info(
        "RUN_MODE=%s | TRADING_ENABLED=%s | QQQ_BTC_TICK_EXITS=%s | OMS_MOCK_IBKR=%s",
        RUN_MODE,
        TRADING_ENABLED,
        os.environ.get("QQQ_BTC_TICK_EXITS", "disaster_only"),
        os.environ.get("OMS_MOCK_IBKR", "0"),
    )
    try:
        from startup_state_hygiene import run_startup_cleanup

        dry = os.environ.get("STARTUP_CLEANUP_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
        run_startup_cleanup(role="oms", dry_run=dry)
    except Exception as e:
        logger.warning("startup cleanup skipped: %s", e)

    engine = ExecutionEngineV8(symbols=TARGET_SYMBOLS, mode="realtime")
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("stopped by user")
