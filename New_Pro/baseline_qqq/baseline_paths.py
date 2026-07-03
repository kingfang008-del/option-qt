#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline_qqq 路径引导 —— 启动脚本首行 import 即可。

sys.path 顺序:
  1. repo 根 (qqq_btc)
  2. baseline_qqq (config / 分层包)
  3. compat (legacy 扁平 import shim)
  4. DAO (ibkr_connector_v8 等)
"""
from __future__ import annotations

import sys
from pathlib import Path

_BASELINE = Path(__file__).resolve().parent
_REPO = _BASELINE.parents[1]

for p in (_REPO, _BASELINE, _BASELINE / "compat", _BASELINE / "DAO"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
