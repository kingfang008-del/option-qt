#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import bisect


import numpy as np
import bisect

# 👇 把这个函数重新加回 oms_replay_pitcher_core.py 的这个位置 👇
def label_ts_for_frame(ts_val: float) -> int:
    minute_start = int(ts_val // 60) * 60
    return minute_start - 60

def get_alpha_backward(alpha_map: dict[int, dict[str, dict]], sym: str, target_ts: float) -> dict:
    """[核心修复] 使用二分查找实现类似 merge_asof(direction='backward') 的向下兼容取值"""
    ts_keys = sorted(alpha_map.keys())
    if not ts_keys: return {}
    
    # 找到小于等于 target_ts 的最大时间戳
    idx = bisect.bisect_right(ts_keys, int(target_ts)) - 1
    if idx >= 0:
        return alpha_map[ts_keys[idx]].get(sym, {})
    return {}


def build_inference_payload(
    ts_val: float,
    batch_payloads: list[dict],
    alpha_map: dict[int, dict[str, dict]],  # 接收完整的 map
    roc_row: dict[str, float],
    target_symbols: list[str] | tuple[str, ...] | set[str],
    is_new_minute: bool = True  # 🚀 [核心修复 1] 增加这个参数
) -> dict | None:
    frame_id = str(int(ts_val))
    
    # 🚀 [核心对齐 1 & 2 & 3] 延迟 60 秒 + 向下兼容查表 (完美复刻 S4 merge_asof)
    target_alpha_ts = int(ts_val) - 60
    ts_keys = sorted(alpha_map.keys())
    idx = bisect.bisect_right(ts_keys, target_alpha_ts) - 1
    
    # 提取当前秒应该生效的 Alpha 行 (容忍 120 秒误差)
    alpha_rows = {}
    if idx >= 0 and (target_alpha_ts - ts_keys[idx] <= 120):
        alpha_rows = alpha_map[ts_keys[idx]]

    target_set = set(target_symbols)
    symbols, stock_price, stock_volume, fast_vol, precalc_alpha = [], [], [], [], []
    alpha_label_ts_arr, alpha_available_ts_arr, spy_rocs, qqq_rocs, stock_ids = [], [], [], [], []
    live_options, live_options_5m, vol_z_dict = {}, {}, {}

    for payload in batch_payloads:
        sym = str(payload.get("symbol", ""))
        if sym not in target_set: continue

        stock = payload.get("stock", {}) or {}
        opt_buckets = payload.get("option_buckets", []) or []
        opt_contracts = payload.get("option_contracts", []) or []
        
        # 🚀 [修复 1] 正确从内部字典提取 symbol
        alpha_row = alpha_rows.get(sym, {})

        symbols.append(sym)
        stock_price.append(float(stock.get("close", 0.0)))
        stock_volume.append(float(stock.get("volume", 0.0)))
        fast_vol.append(float(alpha_row.get("vol_z", 0.0)))
        precalc_alpha.append(float(alpha_row.get("alpha", 0.0)))
        
        # 记录真实使用的 Alpha 的物理产生时间
        extracted_label_ts = ts_keys[idx] if idx >= 0 else ts_val - 60
        alpha_label_ts_arr.append(float(extracted_label_ts))
        alpha_available_ts_arr.append(float(ts_val))
        
        spy_rocs.append(float(roc_row.get("spy_roc_5min", 0.0)))
        qqq_rocs.append(float(roc_row.get("qqq_roc_5min", 0.0)))
        stock_ids.append(0)
        live_options[sym] = {"buckets": opt_buckets, "contracts": opt_contracts}
        vol_z_dict[sym] = float(alpha_row.get("vol_z", 0.0))

    if not symbols: return None

    return {
        "ts": float(ts_val), "log_ts": float(ts_val), "source_ts": float(ts_val),
        "frame_id": frame_id, "symbols": symbols,
        "stock_price": np.asarray(stock_price, dtype=np.float32),
        "stock_volume": np.asarray(stock_volume, dtype=np.float32),
        "stock_id": np.asarray(stock_ids, dtype=np.int32),
        "sector_id": np.zeros(len(symbols), dtype=np.int32),
        "fast_vol": np.asarray(fast_vol, dtype=np.float32),
        "precalc_alpha": np.asarray(precalc_alpha, dtype=np.float32),
        "alpha_label_ts": np.asarray(alpha_label_ts_arr, dtype=np.float64),
        "alpha_available_ts": np.asarray(alpha_available_ts_arr, dtype=np.float64),
        "vol_z_dict": vol_z_dict,
        "spy_roc_5min": np.asarray(spy_rocs, dtype=np.float32),
        "qqq_roc_5min": np.asarray(qqq_rocs, dtype=np.float32),
        "live_options": live_options,
        "is_new_minute": is_new_minute,  # 🚀 [核心修复 2] 动态取值，而不是写死 True 
        "is_warmed_up": True,
        "real_history_len": 999, "total_history_len": 999,
        "real_norm_history_len": 999, "warmup_required_len": 31,
        "has_cross_day_warmup": True,
    }