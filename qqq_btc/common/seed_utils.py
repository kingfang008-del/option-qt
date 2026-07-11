#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局随机种子：训练 / 推理共用，保证可复现。"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """设置 python / numpy / torch / cudnn 种子。返回实际使用的 seed。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 可复现优先；部分 CUDA 算子可能稍慢
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 不强制 use_deterministic_algorithms：部分算子会直接报错
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def make_dataloader_generator(seed: int):
    """供 DataLoader(generator=...) 使用。"""
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn：每个 worker 独立但可复现。"""
    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_seed(cli_seed: Optional[int] = None, env_var: str = "QQQ_BTC_SEED") -> int:
    if cli_seed is not None:
        return int(cli_seed)
    raw = os.environ.get(env_var)
    if raw is not None and str(raw).strip() != "":
        return int(raw)
    return DEFAULT_SEED
