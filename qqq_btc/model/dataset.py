#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMDB 数据集 —— 从原 UnifiedLMDBDataset 内化。

设计修正:
  - 训练期依赖(lmdb/msgpack/zstandard)在类内部 import,
    模块 import 本身不拉重依赖、无日志文件副作用
  - 标签严格读 process_labels_file / label_pipeline 列名,
    缺必需列直接报错,不再静默 fallback(原版 fallback 是 net 全 0 / 方向恒盘整的温床)
  - REQUIRED_LABELS 与 build_lmdb 对齐:回归三件套 + direction_net(方向 CE 权重最高)
  - sanity check 返回 dict 供训练脚本断言,而非只打日志
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .backbone import STATIC_FEATURE_NAMES

logger = logging.getLogger("qqq_btc.dataset")

# 与 tools/build_lmdb.REQUIRED_LABELS 一致;NetEdgeLoss 方向头依赖 direction_net。
REQUIRED_LABELS = (
    "label_return_fwd_net",
    "label_return_fwd_gross",
    "label_execution_cost",
    "label_direction_net",
)


class LMDBAlphaDataset(Dataset):
    def __init__(self, db_path: str, config: dict, seq_len: int = 30, strict_labels: bool = True):
        import lmdb
        import msgpack
        import msgpack_numpy
        import zstandard as zstd

        msgpack_numpy.patch()
        self._msgpack = msgpack
        self.db_path = db_path
        self.seq_len = seq_len
        self.strict_labels = strict_labels
        self.dctx = zstd.ZstdDecompressor()

        # 特征列 → (LMDB 源 key, 塔内列索引);与 backbone.analyze_features 同一约定
        self.stock_map, self.option_map = [], []
        idx = {"stock": 0, "option": 0}
        for f in config["features"]:
            name = f["name"]
            if name in STATIC_FEATURE_NAMES:
                continue
            tower = "option" if name.startswith("options_") else "stock"
            entry = {"name": name, "source": f.get("resolution", "1min"), "target_idx": idx[tower]}
            (self.option_map if tower == "option" else self.stock_map).append(entry)
            idx[tower] += 1
        self.n_stock_feats = idx["stock"]
        self.n_option_feats = idx["option"]
        logger.info(
            "Dataset %s | stock_feats=%d option_feats=%d",
            db_path, self.n_stock_feats, self.n_option_feats,
        )

        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            self.keys = msgpack.unpackb(self.dctx.decompress(txn.get(b"__keys__")), raw=False)
        env.close()
        self.txn = None
        self.env = None

    def _init_env(self):
        if self.txn is None:
            import lmdb

            self.env = lmdb.open(
                self.db_path, readonly=True, lock=False, readahead=True, meminit=False
            )
            self.txn = self.env.begin(write=False)

    def sanity_check(self, sample: int = 1000) -> dict:
        """返回标签质量统计;strict 模式下缺 net 标签直接抛错。"""
        self._init_env()
        n = min(sample, len(self.keys))
        rets, costs, missing = [], [], 0
        for i in range(n):
            val = self.txn.get(self.keys[i])
            if not val:
                continue
            data = self._msgpack.unpackb(self.dctx.decompress(val), raw=False)
            lbl = data.get("labels", {})
            if any(k not in lbl for k in REQUIRED_LABELS):
                missing += 1
                continue
            rets.append(float(lbl["label_return_fwd_net"]))
            costs.append(float(lbl["label_execution_cost"]))

        report = {
            "checked": n,
            "missing_labels": missing,
            "net_std": float(np.std(rets)) if rets else 0.0,
            "net_nonzero_pct": float(np.mean(np.abs(rets) > 1e-9) * 100) if rets else 0.0,
            "cost_mean": float(np.mean(costs)) if costs else 0.0,
        }
        if self.strict_labels:
            if missing > 0:
                raise ValueError(
                    f"{self.db_path}: {missing}/{n} 条缺必需标签 {REQUIRED_LABELS}。"
                    "LMDB 须由 process_labels_file(或 label_pipeline)生成,不做旧标签 fallback。"
                )
            if report["net_std"] < 1e-9:
                raise ValueError(f"{self.db_path}: net 标签方差为 0,检查标签管线。")
        return report

    def __len__(self):
        return len(self.keys)

    def _fill_matrix(self, mat, feat_map, src_1m, src_5m):
        for item in feat_map:
            source = src_5m if item["source"] == "5min" else src_1m
            arr = source.get(item["name"])
            if arr is None:
                continue
            v = np.asarray(arr, dtype=np.float32)
            if item["source"] == "5min":
                l = min(len(v), self.seq_len // 5)
                if l > 0:
                    up = np.repeat(v[-l:], 5)
                    mat[-len(up):, item["target_idx"]] = up
            else:
                l = min(len(v), self.seq_len)
                if l > 0:
                    mat[-l:, item["target_idx"]] = v[-l:]

    def __getitem__(self, idx):
        self._init_env()
        val = self.txn.get(self.keys[idx])
        if not val:
            return None
        try:
            data = self._msgpack.unpackb(self.dctx.decompress(val), raw=False)
        except Exception:
            return None

        x_stock = np.zeros((self.seq_len, self.n_stock_feats), dtype=np.float32)
        x_option = np.zeros((self.seq_len, self.n_option_feats), dtype=np.float32)
        src_1m = data.get("1min", {})
        src_5m = data.get("5min", {})
        self._fill_matrix(x_stock, self.stock_map, src_1m, src_5m)
        self._fill_matrix(x_option, self.option_map, src_1m, src_5m)
        x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=0.0, neginf=0.0)
        x_option = np.nan_to_num(x_option, nan=0.0, posinf=0.0, neginf=0.0)

        meta = data.get("metadata", {})
        ts = meta.get("timestamp", 0)
        try:
            dow = pd.to_datetime(ts, unit="ns").dayofweek
        except (ValueError, TypeError):
            dow = 0
        static = {
            "stock_id": int(meta.get("stock_id", 0)),
            "sector_id": int(meta.get("sector_id", 0)),
            "day_of_week": int(dow),
        }

        lbl = data.get("labels", {})
        if any(k not in lbl for k in REQUIRED_LABELS):
            if self.strict_labels:
                return None  # 缺标签样本剔除,不 fallback 到恒盘整/旧列
            net = lbl.get("label_return_fwd_net", lbl.get("label_return_fwd", 0.0)) or 0.0
            gross = lbl.get("label_return_fwd_gross", net) or 0.0
            cost = lbl.get("label_execution_cost", 0.0) or 0.0
            d_val = lbl.get("label_direction_net", lbl.get("label_direction", 1))
        else:
            # fill 标签无效行(无盘口/权利金过低)不进训练
            if self.strict_labels and "label_net_valid" in lbl and not bool(lbl["label_net_valid"]):
                return None
            net = lbl["label_return_fwd_net"]
            gross = lbl["label_return_fwd_gross"]
            cost = lbl["label_execution_cost"]
            d_val = lbl["label_direction_net"]

        target = {
            "direction": int(d_val),
            "return_fwd": float(np.nan_to_num(net, nan=0.0)),
            "return_fwd_gross": float(np.nan_to_num(gross, nan=0.0)),
            "execution_cost": float(np.nan_to_num(cost, nan=0.0)),
        }
        # 双腿标签(可选):存在时训练 call/put 双头,loss 侧按权重开关
        if "label_call_return_fwd_net" in lbl and "label_put_return_fwd_net" in lbl:
            target["call_return_fwd"] = float(np.nan_to_num(lbl["label_call_return_fwd_net"], nan=0.0))
            target["put_return_fwd"] = float(np.nan_to_num(lbl["label_put_return_fwd_net"], nan=0.0))
        if "label_straddle_return_fwd_net" in lbl:
            target["straddle_return_fwd"] = float(np.nan_to_num(lbl["label_straddle_return_fwd_net"], nan=0.0))
        if "label_spot_return_fwd_30m" in lbl and "label_spot_direction_30m" in lbl:
            target["spot_return_fwd"] = float(np.nan_to_num(lbl["label_spot_return_fwd_30m"], nan=0.0))
            target["spot_direction"] = int(np.nan_to_num(lbl["label_spot_direction_30m"], nan=1))
        if "label_best_bucket_id" in lbl:
            raw_bucket = int(np.nan_to_num(lbl["label_best_bucket_id"], nan=-1))
            best_bucket = raw_bucket if 0 <= raw_bucket <= 7 else 8
            if 0 <= raw_bucket <= 3:
                best_side = 0  # PUT
            elif 4 <= raw_bucket <= 7:
                best_side = 2  # CALL
            else:
                best_side = 1  # NONE/flat
            target["best_bucket"] = int(best_bucket)
            target["best_side"] = int(best_side)
        return x_stock, x_option, static, target, ts


def collate_fn(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    x_stk = torch.stack([torch.from_numpy(b[0]) for b in batch])
    x_opt = torch.stack([torch.from_numpy(b[1]) for b in batch])
    static = {
        k: torch.tensor([b[2][k] for b in batch]) for k in batch[0][2]
    }
    target = {
        k: torch.tensor(
            [b[3][k] for b in batch],
            dtype=torch.long
            if k in {"direction", "best_side", "best_bucket", "spot_direction"}
            else torch.float32,
        )
        for k in batch[0][3]
    }
    ts = [b[4] for b in batch]
    return x_stk, x_opt, static, target, ts
