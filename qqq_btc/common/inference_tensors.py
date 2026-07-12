#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理特征张量构造 —— 无 torch 依赖,供 run_inference 与单元测试共用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_LEN = 30
WINDOW_5M_BARS = 6
FIVE_MIN_STRIDE = 5


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


def _fill_1m_column(mat: np.ndarray, col_idx: int, values: np.ndarray) -> None:
    l = min(len(values), SEQ_LEN)
    if l > 0:
        mat[-l:, col_idx] = values[-l:]


def apply_5m_stair_step(seq: np.ndarray) -> np.ndarray:
    """
    将已按 1min 对齐的 5min 特征序列重排为训练同款 stair-step:
    在序列上取 6 个 stride-5 锚点,各 repeat×5 → 长度 SEQ_LEN。

    接受 shape (T,) 或 (B, T)。
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 2:
        return np.stack([apply_5m_stair_step(row) for row in arr], axis=0)
    if arr.ndim != 1:
        raise ValueError(f"apply_5m_stair_step expects 1d/2d, got shape {arr.shape}")
    t = int(arr.shape[0])
    if t <= 0:
        return np.zeros(SEQ_LEN, dtype=np.float32)
    anchors = []
    for k in range(WINDOW_5M_BARS - 1, -1, -1):
        pos = t - 1 - FIVE_MIN_STRIDE * k
        pos = max(0, min(pos, t - 1))
        anchors.append(arr[pos])
    up = np.repeat(np.asarray(anchors, dtype=np.float32), FIVE_MIN_STRIDE)
    out = np.zeros(SEQ_LEN, dtype=np.float32)
    out[-min(len(up), SEQ_LEN) :] = up[-SEQ_LEN:]
    return out


def _fill_5m_column(mat: np.ndarray, col_idx: int, chunk: pd.DataFrame, col: str) -> None:
    """
    与 LMDBAlphaDataset._fill_matrix 一致:取 6 个 5min 锚点(在 1min 网格上每隔 5 bar),
    每个值 repeat 5 次填满 30×1min 窗口。
    """
    if col not in chunk.columns or chunk.empty:
        return
    n = len(chunk)
    anchors = []
    for k in range(WINDOW_5M_BARS - 1, -1, -1):
        pos = n - 1 - FIVE_MIN_STRIDE * k
        pos = max(0, min(pos, n - 1))
        anchors.append(float(pd.to_numeric(chunk[col].iloc[pos], errors="coerce") or 0.0))
    v = np.asarray(anchors, dtype=np.float32)
    l = min(len(v), SEQ_LEN // FIVE_MIN_STRIDE)
    if l <= 0:
        return
    up = np.repeat(v[-l:], FIVE_MIN_STRIDE)
    mat[-len(up):, col_idx] = up


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
            col = item["name"]
            if col not in chunk.columns:
                continue
            if item.get("source") == "5min":
                _fill_5m_column(mat, item["target_idx"], chunk, col)
            else:
                v = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).values.astype(np.float32)
                _fill_1m_column(mat, item["target_idx"], v)

    _fill(x_stock, stock_map)
    _fill(x_option, option_map)
    return x_stock, x_option
