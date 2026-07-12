#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc Signal 启动入口 —— legacy SignalEngineV8 外壳 + qqq_btc v4 checkpoint。

    cd New_Pro/baseline_qqq
    QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_signal_qqq.py \\
      --checkpoint ../../checkpoints_qqq_v4/best.pth

仍走 Redis unified_inference_stream → ALPHA_FRAME → OMS;
仅替换 slow_model 与 net_edge 口径,保留 FCS batch 消费与 SYNC 屏障。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
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

bootstrap_qqq_btc_live(patch_oms=False)

from qqq_btc.live.signal_integration import create_qqq_btc_signal_engine  # noqa: E402
from signal_engine_v8 import SignalEngineV8  # noqa: E402

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
    format="%(asctime)s - [RUN_SIG_QQQ] - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "LiveRunnerSignalQqqBtc.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("LiveRunnerSignalQqqBtc")


def _default_paths():
    home = Path.home()
    repo_cfg = _REPO / "qqq_btc" / "CONFIG"
    # Prefer in-repo V4 layout; fall back to legacy flat / home paths.
    ckpt_candidates = [
        _REPO / "checkpoint" / "checkpoints_qqq_v4" / "best.pth",
        _REPO / "checkpoints_qqq_v4" / "best.pth",
        home / "quant_project/checkpoints_qqq_v4/best.pth",
        home / "quant_project/checkpoints_qqq_net_edge_v2/best.pth",
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), ckpt_candidates[0])
    v4_slow = repo_cfg / "slow_feature_qqq_v4.json"
    v2_slow = repo_cfg / "slow_feature_qqq_v2.json"
    if v4_slow.exists():
        slow_cfg = str(v4_slow)
    elif v2_slow.exists():
        slow_cfg = str(v2_slow)
    else:
        slow_cfg = str(_REPO / "New_Pro" / "CONFIG" / "slow_feature.json")
    fast_candidates = [
        repo_cfg / "fast_feature_qqq.json",
        home / "quant_project/config/fast_feature.json",
        _REPO / "New_Pro" / "CONFIG" / "fast_feature.json",
    ]
    fast_cfg = next((str(p) for p in fast_candidates if p.exists()), str(fast_candidates[1]))
    return {
        "fast": fast_cfg,
        "slow": slow_cfg,
    }, str(ckpt)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--feature-config",
        default=str(_REPO / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v4.json"),
    )
    args, _ = parser.parse_known_args()

    config_paths, default_ckpt = _default_paths()
    checkpoint = str(Path(args.checkpoint or default_ckpt).expanduser().resolve())

    print("\n" + "=" * 60)
    print("🚀 qqq_btc Signal Engine (SignalEngineV8 shell + v4 model)")
    print("=" * 60 + "\n")
    logger.info("RUN_MODE=%s | checkpoint=%s", RUN_MODE, checkpoint)

    try:
        from startup_state_hygiene import run_startup_cleanup

        dry = __import__("os").environ.get("STARTUP_CLEANUP_DRY_RUN", "").strip().lower() in (
            "1", "true", "yes",
        )
        run_startup_cleanup(role="se", dry_run=dry)
    except Exception as e:
        logger.warning("startup cleanup skipped: %s", e)

    EngineCls = create_qqq_btc_signal_engine(
        SignalEngineV8,
        checkpoint=checkpoint,
        config_path=args.feature_config,
    )
    engine = EngineCls(
        symbols=TARGET_SYMBOLS,
        config_paths=config_paths,
        model_paths={"slow": checkpoint},
        mode="realtime",
    )
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("stopped by user")
