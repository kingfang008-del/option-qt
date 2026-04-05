#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: run_live_orchestrator.py
描述: [Production] 启动实盘/模拟盘 Orchestrator (Client 999)
      - 连接 Redis DB 0 (与 Feature Service共享)
      - 连接 IBKR Gateway (Client 999)
      - 加载真实模型权重
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加当前目录到 sys.path 确保能 import 同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from system_orchestrator_v8 import V8Orchestrator
except ImportError:
    print("❌ Error: system_orchestrator_v8.py not found.")
    sys.exit(1)

# 导入 config
try:
    from config import LOG_DIR
except ImportError:
    # Fallback
    LOG_DIR = Path.home() / "quant_project/logs"

# 日志配置 (确保控制台也能看到启动信息)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RUNNER] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "LiveRunner.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("LiveRunner")

async def main():
    print("\n" + "="*60)
    print("🚀 Starting V8 Orchestrator in LIVE/PAPER Mode")
    print("="*60 + "\n")

    # 1. 准备模型配置路径
    HOME = Path.home()
    PROJECT_ROOT = HOME / "quant_project"
    
    CONFIG_SRC_DIR =PROJECT_ROOT / "base"
    
    # 将包含 config.py 的目录加入 sys.path
    if str(CONFIG_SRC_DIR) not in sys.path:
        sys.path.append(str(CONFIG_SRC_DIR))
        
    config_paths = {
        'fast': str( HOME / "notebook/train/fast_feature.json"), 
        'slow': str(  HOME/ "notebook/train/slow_feature.json")
    }
    
    # 2. 准备模型权重路径
    model_paths = {
        'slow': str(PROJECT_ROOT / "checkpoints_advanced_alpha/advanced_alpha_best.pth")
    }

    # 3. 从 config.py 加载标的
    try:
        from config import TARGET_SYMBOLS
        logger.info(f"Loaded {len(TARGET_SYMBOLS)} symbols from config: {TARGET_SYMBOLS}")
    except ImportError:
        logger.error(f"❌ Failed to import SYMBOLS from {CONFIG_SRC_DIR}/config.py")
        return

    # 4. 实例化 Orchestrator (Mode='realtime')
    try:
        orch = V8Orchestrator(
            symbols=TARGET_SYMBOLS,  # <--- [Fix] 使用 Config 中的统一列表
            mode='realtime',  # <--- Key for Client 999 & Live Trading
            config_paths=config_paths,
            model_paths=model_paths
        )
    except Exception as e:
        logger.critical(f"❌ Failed to initialize Orchestrator: {e}")
        return

    # 3. 运行
    try:
        await orch.run()
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user.")
    except Exception as e:
        logger.critical(f"🔥 Crashed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
