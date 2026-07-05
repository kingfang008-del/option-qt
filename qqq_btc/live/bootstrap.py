#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc 实盘 bootstrap —— 在启动 legacy 双引擎前调用,注入 fill_model / tick-exit 契约。

用法:
    # Signal 进程
    QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_signal_qqq.py

    # OMS 进程
    QQQ_BTC_LIVE=1 python qqq_btc/tools/run_live_exec_qqq.py

或在任意入口最顶部:
    from qqq_btc.live.bootstrap import bootstrap_qqq_btc_live
    bootstrap_qqq_btc_live()
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("qqq_btc.live.bootstrap")


def is_qqq_btc_live() -> bool:
    return os.environ.get("QQQ_BTC_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def tick_exits_mode() -> str:
    """
    秒级 exit 模式(与 exit_rails MTM 契约对齐):
      disaster_only — 仅 check_disaster_stop(-25%),默认
      legacy        — 保留 OMS _evaluate_second_dynamic_exits 全量逻辑
      off           — 禁用秒级 exit
    """
    if not is_qqq_btc_live():
        return "legacy"
    mode = os.environ.get("QQQ_BTC_TICK_EXITS", "disaster_only").strip().lower()
    if mode in ("disaster_only", "legacy", "off"):
        return mode
    return "disaster_only"


def bootstrap_qqq_btc_live(*, patch_oms: bool = True) -> bool:
    """启用 qqq_btc 实盘集成。返回 True 表示已 bootstrap。"""
    if not is_qqq_btc_live():
        return False

    repo = Path(__file__).resolve().parents[2]
    v2_cfg = repo / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
    if v2_cfg.exists() and not os.environ.get("SLOW_FEATURE_CONFIG", "").strip():
        os.environ["SLOW_FEATURE_CONFIG"] = str(v2_cfg)
    os.environ.setdefault("ALPHA_ZSCORE_MODE", "absolute")
    os.environ.setdefault("USE_NET_EDGE_ALPHA", "1")
    # bar 收盘后立即下单:禁止 OMS 延迟队列与 Mock 回放延迟 bar
    os.environ.setdefault("EXECUTION_DELAY_BARS", "0")
    os.environ.setdefault("OMS_SIGNAL_DELAY_BARS", "0")
    # 门控收敛: spread/q10/阈值由 choose_entry 负责;FAST_GATE 与 replay 6% 重复
    os.environ.setdefault("FAST_GATE_ENABLED", "0")
    # 冷却与 replay cooldown_bars=5 对齐(分钟 bar)
    os.environ.setdefault("COOLDOWN_MINUTES", "5")

    if patch_oms:
        from qqq_btc.live.oms_integration import apply_oms_patches

        apply_oms_patches(tick_exits_mode=tick_exits_mode())
    logger.info(
        "qqq_btc live bootstrap OK | tick_exits=%s | fill=0.775 | immediate_entry=1",
        tick_exits_mode(),
    )
    return True
