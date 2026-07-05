#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 qqq_btc/CONFIG/slow_feature_qqq_v2.json。

以 New_Pro/CONFIG/slow_feature.json 为基底(不修改原文件),做以下变更:

1. 追加 4 个日内时间特征 + 7 个趋势特征(calc=raw)。
   由 feature_merge_option_raw.FeatureEngineer 在 merge 阶段写入;
   label_pipeline / live 侧同口径补算作为兜底。
2. labels / labeling 参数与 process_labels_file 对齐(30min horizon + net 标签)。
3. loss_weights:rank_net=1.0(成对排序), net_edge_quantile=1.0;双腿/跨式权重保留,
   仅当 parquet 含对应列时 loss 生效。

用生成器而非手拷贝,保证 New_Pro 基底特征演进时可一键重新同步:
    python qqq_btc/tools/make_feature_config.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_CONFIG = REPO_ROOT / "New_Pro" / "CONFIG" / "slow_feature.json"
OUTPUT_CONFIG = REPO_ROOT / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"

# 与 preprocess/ask_bid/feature_merge_option_raw.py::process_labels_file 写入列一一对应。
# config key = parquet 列名去掉 label_ 前缀(训练 loss / LMDB 约定)。
PROCESS_LABELS = {
    "direction": {
        "type": "classification",
        "output_dim": 3,
        "column": "label_direction",
        "description": "三重障碍方向 0/1/2(先触下/盘整/先触上),horizon=k_slow",
    },
    "direction_net": {
        "type": "classification",
        "output_dim": 3,
        "column": "label_direction_net",
        "description": "扣执行成本后的净方向 0/1/2",
    },
    "volatility": {
        "type": "regression",
        "column": "label_volatility",
        "description": "未来 k_slow bar 已实现波动(log_ret std)",
    },
    "event": {
        "type": "classification",
        "output_dim": 1,
        "column": "label_event",
        "description": "未来窗口极端行情事件 0/1",
    },
    "return_fwd": {
        "type": "regression",
        "column": "label_return_fwd",
        "description": "未来 k_slow bar 毛收益(mid/close 口径)",
    },
    "return_fwd_gross": {
        "type": "regression",
        "column": "label_return_fwd_gross",
        "description": "毛收益副本,与 return_fwd 同口径",
    },
    "return_fwd_net": {
        "type": "regression",
        "column": "label_return_fwd_net",
        "description": "扣往返点差摩擦后的净收益(主训练目标)",
    },
    "execution_cost": {
        "type": "regression",
        "column": "label_execution_cost",
        "description": "估计往返执行成本(fill_frac 点差摩擦,含 floor)",
    },
}

# 可选双腿/跨式列:仅 label_pipeline(build_dual_leg_net_labels)写入;缺列时 loss 跳过。
OPTIONAL_DUAL_LEG_LABELS = {
    "call_return_fwd_net": {
        "type": "regression",
        "column": "label_call_return_fwd_net",
        "optional": True,
        "description": "CALL 腿 fill 价净收益",
    },
    "put_return_fwd_net": {
        "type": "regression",
        "column": "label_put_return_fwd_net",
        "optional": True,
        "description": "PUT 腿 fill 价净收益",
    },
    "straddle_return_fwd_net": {
        "type": "regression",
        "column": "label_straddle_return_fwd_net",
        "optional": True,
        "description": "跨式合成净收益(有符号)",
    },
    "net_valid": {
        "type": "classification",
        "output_dim": 1,
        "column": "label_net_valid",
        "optional": True,
        "description": "CALL 腿标签有效掩码",
    },
    "put_net_valid": {
        "type": "classification",
        "output_dim": 1,
        "column": "label_put_net_valid",
        "optional": True,
        "description": "PUT 腿标签有效掩码",
    },
    "straddle_valid": {
        "type": "classification",
        "output_dim": 1,
        "column": "label_straddle_valid",
        "optional": True,
        "description": "跨式标签有效掩码",
    },
}

# process_labels_file 默认超参(1min bar)
LABEL_HORIZON_K = 30
LABEL_VOL_WINDOW = 120
OPT_FILL_SPREAD_FRAC = 0.775

TIME_FEATURES = [
    {"name": "time_session_sin", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话内位置 sin 编码,[-1,1],feature_merge 写入"},
    {"name": "time_session_cos", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话内位置 cos 编码,[-1,1]"},
    {"name": "time_session_progress", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "会话进度 0->1 (09:30->16:00)"},
    {"name": "time_to_expiry_norm", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "0DTE 距 16:00 到期剩余时间归一化 1->0"},
]

TREND_FEATURES = [
    {"name": "trend_fit_ret_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "过去 30 bar 对数价线性拟合总变化(拟合收益)"},
    {"name": "trend_fit_r2_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "30 bar 拟合优度 R^2,[0,1],趋势规则程度"},
    {"name": "trend_fit_ret_120m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "过去 120 bar 波段级拟合收益"},
    {"name": "trend_fit_r2_120m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "120 bar 波段级拟合优度 R^2"},
    {"name": "spot_range_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "过去 30 bar 现货振幅 (high-low)/close"},
    {"name": "trend_strength_30m", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "|trend_fit_ret_30m| * trend_fit_r2_30m,方向确信度"},
    {"name": "day_range_pos", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "当前价在当日已实现高低区间的位置,[0,1]"},
    {"name": "drawdown_from_day_high", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距当日已实现最高点的回撤,<=0"},
    {"name": "drawup_from_day_low", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距当日已实现最低点的反弹,>=0"},
]

OPEN30_FEATURES = [
    {"name": "open30_ret", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "开盘至当前(或10:00冻结)的收益"},
    {"name": "open30_max_ret", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "开盘窗内最高收益(冲高幅度)"},
    {"name": "open30_peak_dd", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距开盘窗内最高点的回撤,<=0"},
    {"name": "open30_reversal", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "前15min收益减后15min收益,倒V为正"},
    {"name": "open30_range_pos", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "当前价在开盘窗高低区间位置,[0,1]"},
    {"name": "bars_since_open30_high_norm", "type": "real", "calc": "raw", "resolution": "1min",
     "description": "距开盘窗高点经过的bar数/30,归一化"},
]


def _sync_labeling_params(cfg: dict) -> None:
    """将 labeling / slow_channel 与 process_labels_file 对齐。"""
    params = cfg.setdefault("parameters", {})
    slow = params.setdefault("slow_channel", {})
    slow["label_horizon_k"] = LABEL_HORIZON_K

    labeling = params.setdefault("labeling", {})
    labeling["method"] = "multi_task"
    labeling["targets"] = list(PROCESS_LABELS.keys())
    labeling["k_slow"] = LABEL_HORIZON_K
    labeling["vol_window"] = LABEL_VOL_WINDOW
    labeling["volatility_k"] = LABEL_HORIZON_K
    labeling["event_k"] = LABEL_HORIZON_K
    labeling["opt_fill_spread_frac"] = OPT_FILL_SPREAD_FRAC
    labeling["executable_cost_margin"] = 0.0025
    labeling["executable_flat_margin"] = 0.0005

    resolutions = labeling.setdefault("resolutions", {})
    # 主训练分辨率 1min:horizon 与 k_slow 一致
    res_1m = resolutions.setdefault("1min", {})
    res_1m["horizon"] = LABEL_HORIZON_K
    res_1m.setdefault("entry_threshold", 0.002)
    res_1m.setdefault("upper_threshold", 0.002)
    res_1m.setdefault("lower_threshold", -0.002)
    res_1m.setdefault("vol_multiplier", 2.5)
    # 5min:按 bar 等比(30min / 5 = 6 bar)
    res_5m = resolutions.setdefault("5min", {})
    res_5m["horizon"] = max(1, LABEL_HORIZON_K // 5)
    res_5m.setdefault("entry_threshold", 0.0025)
    res_5m.setdefault("upper_threshold", 0.0025)
    res_5m.setdefault("lower_threshold", -0.0025)
    res_5m.setdefault("vol_multiplier", 2.0)


def build_config() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    existing = {f["name"] for f in cfg["features"]}
    for feat in TIME_FEATURES + TREND_FEATURES + OPEN30_FEATURES:
        if feat["name"] not in existing:
            cfg["features"].append(feat)

    # 标签契约:主路径 = process_labels_file;可选双腿列单独标注
    cfg["labels"] = {**PROCESS_LABELS, **OPTIONAL_DUAL_LEG_LABELS}
    _sync_labeling_params(cfg)

    cfg["loss_weights"]["rank_net"] = 1.0
    cfg["loss_weights"]["net_edge_quantile"] = 1.0
    # 双腿/跨式:仅当 LMDB 含对应列时生效,缺列 loss 自动跳过
    cfg["loss_weights"]["call_put_edge"] = 0.5
    cfg["loss_weights"]["straddle_edge"] = 0.5
    # 保证主标签权重键齐全
    for key, default in (
        ("direction", 4.0),
        ("direction_net", 4.0),
        ("return_fwd", 1.0),
        ("return_fwd_net", 3.0),
        ("return_fwd_gross", 0.5),
        ("execution_cost", 0.5),
        ("volatility", 0.1),
        ("event", 0.5),
    ):
        cfg["loss_weights"].setdefault(key, default)

    cfg["comment"] = (
        "qqq_btc v2:基于 New_Pro slow_feature.json 生成(make_feature_config.py)。"
        "特征含日内 time/trend(feature_merge 写入)。"
        "主标签由 feature_merge_option_raw.process_labels_file 写入"
        f"(k_slow={LABEL_HORIZON_K}, opt_fill_spread_frac={OPT_FILL_SPREAD_FRAC});"
        "可选双腿/跨式列由 label_pipeline(build_dual_leg_net_labels)补齐。"
        "再跑 rolling_norm 与 LMDB。"
    )
    cfg.setdefault("parameters", {})["qqq_btc_v2"] = {
        "fill_model": {
            "entry_frac": OPT_FILL_SPREAD_FRAC,
            "exit_frac": OPT_FILL_SPREAD_FRAC,
            "commission_per_contract": 0.65,
        },
        "label_source": "preprocess.ask_bid.feature_merge_option_raw.process_labels_file",
        "label_columns": [v["column"] for v in PROCESS_LABELS.values()],
        "optional_label_columns": [v["column"] for v in OPTIONAL_DUAL_LEG_LABELS.values()],
        "label_horizon_k": LABEL_HORIZON_K,
        "time_features": [f["name"] for f in TIME_FEATURES],
        "trend_features": [f["name"] for f in TREND_FEATURES],
        "open30_features": [f["name"] for f in OPEN30_FEATURES],
        "quantiles": [0.1, 0.5, 0.9],
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
