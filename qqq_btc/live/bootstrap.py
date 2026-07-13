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
    v4_cfg = repo / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v4.json"
    v2_cfg = repo / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
    default_cfg = v4_cfg if v4_cfg.exists() else v2_cfg
    if default_cfg.exists() and not os.environ.get("SLOW_FEATURE_CONFIG", "").strip():
        os.environ["SLOW_FEATURE_CONFIG"] = str(default_cfg)
    os.environ.setdefault("ALPHA_ZSCORE_MODE", "absolute")
    os.environ.setdefault("USE_NET_EDGE_ALPHA", "1")
    os.environ.setdefault("BIDIRECTIONAL_ENABLED", "1")
    os.environ.setdefault("BIDIRECTIONAL_DUAL_EDGE_ENABLED", "1")
    # bar 收盘后立即下单:禁止 OMS 延迟队列与 Mock 回放延迟 bar
    os.environ.setdefault("EXECUTION_DELAY_BARS", "0")
    os.environ.setdefault("OMS_SIGNAL_DELAY_BARS", "0")
    # 门控收敛: spread/q10/阈值由 choose_entry 负责;FAST_GATE 与 replay 6% 重复
    os.environ.setdefault("FAST_GATE_ENABLED", "0")
    # 冷却与 replay cooldown_bars=10 对齐(分钟 bar)
    os.environ.setdefault("COOLDOWN_MINUTES", "10")
    # put_gate / regime：实盘默认因果自算，禁止默读 July 金标文件
    os.environ.setdefault("QQQ_BTC_PUT_GATE_MODE", "vixy_z")
    os.environ.setdefault("QQQ_BTC_REGIME_GOLD_1M", "0")
    # 与 deploy 同款冻结归一化（对拍开卷脚本须显式 export FCS_FROZEN_NORM_PATH=""）
    default_frozen = repo / "qqq_btc" / "CONFIG" / "frozen_norm_qqq_daily.npz"
    if default_frozen.exists():
        cur_frozen = os.environ.get("FCS_FROZEN_NORM_PATH", "").strip()
        if not cur_frozen:
            os.environ["FCS_FROZEN_NORM_PATH"] = str(default_frozen)
    default_sym_map = repo / "qqq_btc" / "CONFIG" / "symbol_map.json"
    if default_sym_map.exists():
        # FCS stock_id/sector_id 与 LMDB/infer 同源(QQQ→1),避免 enumerate→0 导致 edge 翻转
        os.environ.setdefault("FCS_SYMBOL_MAP", str(default_sym_map))

    if patch_oms:
        from qqq_btc.live.oms_integration import apply_oms_patches

        apply_oms_patches(tick_exits_mode=tick_exits_mode())
    logger.info(
        "qqq_btc live bootstrap OK | tick_exits=%s | fill=0.775 | immediate_entry=1",
        tick_exits_mode(),
    )
    return True
