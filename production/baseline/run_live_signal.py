#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: run_live_signal.py
描述: 启动 Signal Engine (无订单执行能力，专做 Alpha 计算)
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_engine_v8 import SignalEngineV8

try:
    from config import LOG_DIR
except ImportError:
    LOG_DIR = Path.home() / "quant_project/logs"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RUN_SIG] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "LiveRunnerSignal.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("LiveRunnerSignal")

async def main():
    print("\n" + "="*60)
    print("🚀 Starting V8 Signal Engine")
    print("="*60 + "\n")

    HOME = Path.home()
    PROJECT_ROOT = HOME / "quant_project"
    CONFIG_SRC_DIR = PROJECT_ROOT / "base"
    
    if str(CONFIG_SRC_DIR) not in sys.path:
        sys.path.append(str(CONFIG_SRC_DIR))
        
    config_paths = {
        'fast': str(HOME / "notebook/train/fast_feature.json"), 
        'slow': str(HOME / "notebook/train/slow_feature.json")
    }
    model_paths = {
        'slow': str(PROJECT_ROOT / "checkpoints_advanced_alpha/advanced_alpha_best.pth")
    }

    try:
        from config import TARGET_SYMBOLS, RUN_MODE, IS_SIMULATED, TRADING_ENABLED
    except ImportError:
        logger.error(f"❌ Failed to import SYMBOLS from config")
        return
    logger.info(f"🧭 Runtime Mode: RUN_MODE={RUN_MODE} | IS_SIMULATED={IS_SIMULATED} | TRADING_ENABLED={TRADING_ENABLED}")

    try:
        engine = SignalEngineV8(
            symbols=TARGET_SYMBOLS,
            mode='realtime',
            config_paths=config_paths,
            model_paths=model_paths
        )
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Signal Engine: {e}")
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
