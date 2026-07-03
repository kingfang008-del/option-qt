#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 qqq_btc/CONFIG/slow_feature_qqq_v2.json。

以 New_Pro/CONFIG/slow_feature.json 为基底(不修改原文件),做三处变更:

1. 追加 4 个日内时间特征(0DTE 缺口:theta/gamma 非平稳需要时间上下文)。
   calc='raw':列由 qqq_btc.common.time_features.add_time_features 在
   feature_merge 之后补算,rolling_norm 会自动跳过(sin/cos 天然有界)。
2. loss_weights:rank_net 显式置 0(单标的无截面语义),
   新增 net_edge_quantile=1.0(分布头 pinball loss,tft_qqq_v2 使用)。
3. 记录 fill 假设来源,提醒标签必须由 qqq_btc.common.labels 生成
   (fill 价精确口径,frac=0.775)。

用生成器而非手拷贝,保证 New_Pro 基底特征演进时可一键重新同步:
    python qqq_btc/tools/make_feature_config.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_CONFIG = REPO_ROOT / "New_Pro" / "CONFIG" / "slow_feature.json"
OUTPUT_CONFIG = REPO_ROOT / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"

TIME_FEATURES = [
    {"name": "time_session_sin", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话内位置 sin 编码,[-1,1],由 qqq_btc.common.time_features 补算"},
    {"name": "time_session_cos", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话内位置 cos 编码,[-1,1]"},
    {"name": "time_session_progress", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话进度 0->1 (09:30->16:00)"},
    {"name": "time_to_expiry_norm", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "0DTE 距 16:00 到期剩余时间归一化 1->0"},
]

# 日内趋势结构(超出 seq_len=30 视野的波段上下文),
# 由 qqq_btc.common.trend_features.add_trend_features 补算。
TREND_FEATURES = [
    {"name": "trend_fit_ret_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "过去 30 bar 对数价线性拟合总变化(拟合收益)"},
    {"name": "trend_fit_r2_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "30 bar 拟合优度 R^2,[0,1],趋势规则程度"},
    {"name": "trend_fit_ret_120m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "过去 120 bar 波段级拟合收益"},
    {"name": "trend_fit_r2_120m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "120 bar 波段级拟合优度 R^2"},
    {"name": "day_range_pos", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "当前价在当日已实现高低区间的位置,[0,1]"},
    {"name": "drawdown_from_day_high", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距当日已实现最高点的回撤,<=0"},
    {"name": "drawup_from_day_low", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距当日已实现最低点的反弹,>=0"},
]


def build_config() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    existing = {f["name"] for f in cfg["features"]}
    for feat in TIME_FEATURES + TREND_FEATURES:
        if feat["name"] not in existing:
            cfg["features"].append(feat)

    cfg["loss_weights"]["rank_net"] = 0.0
    cfg["loss_weights"]["net_edge_quantile"] = 1.0
    # call/put 双腿头:LMDB 含 label_call/put_return_fwd_net(双腿标签链)时生效,
    # 缺标签时 loss 自动跳过该项,权重常开无副作用
    cfg["loss_weights"]["call_put_edge"] = 0.5
    # 跨式头(有符号):LMDB 含 label_straddle_return_fwd_net 时生效
    cfg["loss_weights"]["straddle_edge"] = 0.5

    cfg["comment"] = (
        "qqq_btc v2:基于 New_Pro slow_feature.json 生成(make_feature_config.py),"
        "新增日内时间特征 + quantile loss。标签必须用 qqq_btc.common.labels 重建"
        "(fill 价口径,OptionSpreadFillModel frac=0.775),再跑 rolling_norm 与 LMDB。"
    )
    cfg.setdefault("parameters", {})["qqq_btc_v2"] = {
        "fill_model": {"entry_frac": 0.775, "exit_frac": 0.775, "commission_per_contract": 0.65},
        "label_source": "qqq_btc.common.labels.build_dual_leg_net_labels",
        "time_features": [f["name"] for f in TIME_FEATURES],
        "trend_features": [f["name"] for f in TREND_FEATURES],
        "quantiles": [0.1, 0.5, 0.9],
        # embedding 容量:不再查 Postgres(原 load_meta_info)。
        # 若 LMDB stock_id 使用 PG 大范围 id,保持默认;重建 id 映射后可调小。
        "embedding_caps": {"stock": 20000, "sector": 256},
    }
    return cfg


def main() -> None:
    cfg = build_config()
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    n_feats = len(cfg["features"])
    print(f"written: {OUTPUT_CONFIG} ({n_feats} features)")


if __name__ == "__main__":
    main()
