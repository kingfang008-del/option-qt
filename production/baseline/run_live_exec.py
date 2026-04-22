#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: run_live_exec.py
描述: 启动 Execution Engine (OMS) - 专做 IBKR 路由，无 PyTorch 依赖
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_engine_v8 import ExecutionEngineV8

try:
    from config import LOG_DIR
except ImportError:
    LOG_DIR = Path.home() / "quant_project/logs"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RUN_EXEC] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "LiveRunnerExec.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("LiveRunnerExec")

async def main():
    print("\n" + "="*60)
    print("🚀 Starting V8 Execution Engine (OMS)")
    print("="*60 + "\n")

    HOME = Path.home()
    PROJECT_ROOT = HOME / "quant_project"
    CONFIG_SRC_DIR = PROJECT_ROOT / "base"
    
    if str(CONFIG_SRC_DIR) not in sys.path:
        sys.path.append(str(CONFIG_SRC_DIR))

    try:
        from config import TARGET_SYMBOLS, RUN_MODE, IS_SIMULATED, TRADING_ENABLED
    except ImportError:
        logger.error(f"❌ Failed to import SYMBOLS from config")
        return
    logger.info(f"🧭 Runtime Mode: RUN_MODE={RUN_MODE} | IS_SIMULATED={IS_SIMULATED} | TRADING_ENABLED={TRADING_ENABLED}")

    # [🧹 Startup State Hygiene]
    # OMS 启动前无条件清掉 Redis oms:live_positions 里的上一轮残留快照。
    # 本进程随后会通过 _broadcast_state_to_redis 在自己就绪后重新 publish 真实账本，
    # 所以这里删除是安全的；好处是 SE 侧在 OMS 尚未 broadcast 完成前不会读到
    # 陈旧仓位快照。仿真模式 (IS_SIMULATED) 及 SKIP_STARTUP_CLEANUP=1 时自动跳过。
    try:
        from startup_state_hygiene import run_startup_cleanup
        _dry = os.environ.get("STARTUP_CLEANUP_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
        run_startup_cleanup(role="oms", dry_run=_dry)
    except Exception as e:
        logger.warning(f"⚠️ Startup cleanup skipped due to error: {e}")

    try:
        engine = ExecutionEngineV8(
            symbols=TARGET_SYMBOLS,
            mode='realtime'
        )
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Execution Engine: {e}")
        return

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
    except Exception as e:
        logger.critical(f"🔥 Crashed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
