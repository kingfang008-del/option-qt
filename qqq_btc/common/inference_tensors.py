#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理特征张量构造 —— 无 torch 依赖,供 run_inference 与单元测试共用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_LEN = 30


def build_feature_maps(config: dict) -> tuple[list[dict], list[dict], int, int]:
    static = {"stock_id", "sector_id", "day_of_week"}
    stock_map, option_map = [], []
    idx = {"stock": 0, "option": 0}
    for f in config["features"]:
        name = f["name"]
        if name in static:
            continue
        tower = "option" if name.startswith("options_") else "stock"
        entry = {"name": name, "source": f.get("resolution", "1min"), "target_idx": idx[tower]}
        (option_map if tower == "option" else stock_map).append(entry)
        idx[tower] += 1
    return stock_map, option_map, idx["stock"], idx["option"]


def row_to_tensors(
    df: pd.DataFrame,
    i: int,
    stock_map: list[dict],
    option_map: list[dict],
    n_stock: int,
    n_opt: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_stock = np.zeros((SEQ_LEN, n_stock), dtype=np.float32)
    x_option = np.zeros((SEQ_LEN, n_opt), dtype=np.float32)
    sl = max(0, i - SEQ_LEN + 1)
    chunk = df.iloc[sl : i + 1]

    def _fill(mat, fmap):
        for item in fmap:
            if item["name"] not in chunk.columns:
                continue
            v = pd.to_numeric(chunk[item["name"]], errors="coerce").fillna(0.0).values.astype(np.float32)
            l = len(v)
            if l > 0:
                mat[-l:, item["target_idx"]] = v

    _fill(x_stock, stock_map)
    _fill(x_option, option_map)
    return x_stock, x_option
