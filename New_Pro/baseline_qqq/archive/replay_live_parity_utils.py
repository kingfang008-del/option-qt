"""
S4 回放 ↔ 实盘：阶段 1 时钟与指数 ROC 的定义（单一真相源说明 + 可测工具函数）。

Alpha 分钟标签（左对齐）与「+60s」含义
--------------------------------------
离线 ``alpha_logs`` / 分钟特征通常使用**左对齐**时间戳：标签 ``T`` 表示「以 T 为起点的这根分钟 K 的语义归属」；
该分钟的 alpha 在实际工程里只有到分钟走完之后才可被安全使用。

- **S4**（``s4_run_historical_replay_s2_1s.py``）：先把 alpha 行的 ``ts`` 加上
  ``ALPHA_AVAILABLE_DELAY_SECONDS``（默认 60），再与 1s 行情做 ``merge_asof(..., backward)``，
  避免在 T～T+60 这一分钟内提前看到仍在形成中的分钟 alpha（无 lookahead）。

- **实盘 FCS**（``FeatureComputeServiceV8._align_inference_timing``）：
  ``alpha_label_ts = current_minute_ts - 60``，payload 里的 ``ts`` 使用该标签；
  在**分钟 rollover** 上触发推理。效果与 S4 一致：进入新的一分钟后再发布「上一分钟」的 alpha。

- **指数 ROC 实盘补算**：FCS 在融合流中维护 SPY/QQQ 的 1s 收盘环形缓冲；若特征张量中
  ``spy_roc_5min`` / ``qqq_roc_5min`` 缺失、非有限或与缓冲按 S4 公式差异超过阈值，则用缓冲回算并打
  ``[FCS-IndexROC-S4]`` 日志（节流）。参见 ``record_index_1s_close_for_parity`` /
  ``_merge_tensor_and_s4_index_roc``。

若变更任一侧的时钟规则，须同步更新本说明与 ``ALPHA_FRAME`` 契约测试。

指数 5 分钟 ROC（SPY/QQQ，1s bar）
---------------------------------
S4 对全市场 1s ``close`` 做::

    roc_5m = pct_change(periods=300).fillna(0)

即当前秒相对 **300 根 1s 之前**（约 5 分钟）的涨跌幅。同一 ``ts`` 上的 SPY/QQQ ROC
通过 `merge` 广播到所有标的行（``spy_roc_5min`` / ``qqq_roc_5min``）。

实盘的 ``spy_roc_5min`` / ``qqq_roc_5min`` 目前来自特征张量槽位（``feature_compute_service_v8``）；
训练与离线打特征时必须以上述 **300×1s** 定义生成同名特征，否则指数护栏与 TREND/V0 的
``spy_roc``/``qqq_roc`` 与 S4 不一致。本模块提供 ``index_roc_5min_from_series`` 供单测与离线校验复用。

Spread divergence（空仓）
------------------------
``spread_divergence = 当前相对点差 − 开仓时记录的最后点差``（见 ``ExecutionEngineV8._build_strategy_ctx``）。
空仓时 ``last_spread_pct == 0``，故散度为 0；入场流动性门控主要依赖**绝对点差**与 bid/ask 有效性，
与 ``test_v0_ctx_contract`` 及回放路径一致，并非 bug。
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def index_roc_5min_from_series(closes: ArrayLike, *, periods: int = 300) -> float:
    """
    与 S4 ``pct_change(periods=300).fillna(0)`` 在**序列最后一个有效点**上的取值等价。

    Args:
        closes: 按时间升序的 1s 收盘价（与 S4 `df_idx[col]` 对齐后的索引顺序一致）。
        periods: 默认 300（1s × 300 ≈ 5 分钟）。

    Returns:
        若长度不足 ``periods + 1`` 或分母非正，返回 ``0.0``。
    """
    if closes is None:
        return 0.0
    arr = np.asarray(closes, dtype=float)
    n = int(arr.size)
    if n <= periods:
        return 0.0
    prev = float(arr[n - 1 - periods])
    cur = float(arr[n - 1])
    if not np.isfinite(prev) or not np.isfinite(cur) or prev <= 0.0:
        return 0.0
    return (cur - prev) / prev


def index_roc_5min_at_index(closes: ArrayLike, i: int, *, periods: int = 300) -> float:
    """序列位置 ``i`` 处的 5m ROC（用于与 pandas ``pct_change`` 按行对比）。"""
    arr = np.asarray(closes, dtype=float)
    n = int(arr.size)
    if i < 0 or i >= n or i < periods:
        return 0.0
    prev = float(arr[i - periods])
    cur = float(arr[i])
    if not np.isfinite(prev) or not np.isfinite(cur) or prev <= 0.0:
        return 0.0
    return (cur - prev) / prev
