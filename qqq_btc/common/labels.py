#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
net_edge 标签构建 —— fill 价精确口径。

与上一代 (New_Pro feature_merge_option_raw._apply_executable_net_labels) 的区别:
  旧: gross 用 mid 计算,cost 用 "2*(frac-0.5)*spread_pct" 近似,net = sign(g)*max(|g|-c, 0)
      —— 截断产生大量零标签,且 cost 与实际 fill 价存在二阶误差。
  新: 直接用 FillModel 的 entry/exit fill 价计算净收益,不截断:
      net = exit_fill/entry_fill - 1 - 佣金拖累
      gross 用同时刻 mid 计算,cost = gross - net 仅作诊断输出。

输出列(与 LMDB / TFT 三头模型的既有命名对齐,训练代码可直接复用):
  label_return_fwd_gross   mid 口径毛收益
  label_return_fwd_net     fill 口径净收益(含点差 + 佣金)
  label_execution_cost     gross - net(诊断)
  label_direction_net      0=做空侧净有利 / 1=盘整 / 2=做多侧净有利
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fill_model import OptionSpreadFillModel, PerpFillModel


@dataclass(frozen=True)
class LabelHorizon:
    """入场延迟 + 持有时间(与 New_Pro option_exec_label 语义一致)。"""
    entry_delay_bars: int = 1   # 60s @ 1min bar
    hold_bars: int = 5          # 300s @ 1min bar
    flat_margin: float = 0.0005  # |net| 低于此视为盘整(direction=1)

    @property
    def total_bars(self) -> int:
        return self.entry_delay_bars + self.hold_bars


def _leg_net_arrays(
    df: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    horizon: LabelHorizon,
    bid_col: str,
    ask_col: str,
    mid_col: str,
) -> tuple:
    """
    单腿(做多该腿视角)的 gross/net/valid/entry_premium 数组。
    net 用 fill 价,gross 用 mid;entry_premium = 信号 bar 对应的入场 fill 价
    (用于跨式的权利金加权)。
    """
    n = len(df)
    ed, hd = horizon.entry_delay_bars, horizon.hold_bars

    bid = pd.to_numeric(df.get(bid_col), errors="coerce").to_numpy(dtype=np.float64)
    ask = pd.to_numeric(df.get(ask_col), errors="coerce").to_numpy(dtype=np.float64)
    if mid_col in df.columns:
        mid = pd.to_numeric(df[mid_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        mid = (bid + ask) / 2.0

    entry_fill_series = fill_model.entry_fill(bid, ask)
    exit_fill_series = fill_model.exit_fill(bid, ask)

    gross = np.zeros(n, dtype=np.float64)
    net = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    entry_premium = np.zeros(n, dtype=np.float64)

    valid_len = max(0, n - horizon.total_bars)
    if valid_len > 0:
        idx = np.arange(valid_len)
        e_idx = idx + ed
        x_idx = idx + ed + hd

        entry_px = entry_fill_series[e_idx]
        exit_px = exit_fill_series[x_idx]
        entry_mid = mid[e_idx]
        exit_mid = mid[x_idx]

        ok = (
            np.isfinite(entry_px) & (entry_px > 0)
            & np.isfinite(exit_px) & (exit_px > 0)
            & np.isfinite(entry_mid) & (entry_mid > 0)
            & np.isfinite(exit_mid) & (exit_mid > 0)
        )
        commission_drag = fill_model.commission_return_drag(entry_px)

        g = np.where(ok, exit_mid / np.where(entry_mid > 0, entry_mid, np.nan) - 1.0, 0.0)
        v = np.where(
            ok,
            exit_px / np.where(entry_px > 0, entry_px, np.nan) - 1.0 - commission_drag,
            0.0,
        )
        gross[idx] = np.nan_to_num(g, nan=0.0)
        net[idx] = np.nan_to_num(v, nan=0.0)
        valid[idx] = ok
        entry_premium[idx] = np.where(ok, np.nan_to_num(entry_px, nan=0.0), 0.0)

    return gross, net, valid, entry_premium


def build_option_net_labels(
    df: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    horizon: LabelHorizon = LabelHorizon(),
    bid_col: str = "exec_call_bid",
    ask_col: str = "exec_call_ask",
    mid_col: str = "exec_call_mid",
) -> pd.DataFrame:
    """
    对含锚定合约分钟报价的特征表添加 net_edge 标签(做多 CALL 视角)。

    t 时刻信号 → t+entry_delay 按 entry_fill(买价偏对手侧)成交
    → t+entry_delay+hold 按 exit_fill(卖价偏对手侧)平仓。
    要求 df 已按 timestamp 升序。原地添加列并返回 df。
    """
    gross, net, valid, _ = _leg_net_arrays(df, fill_model, horizon, bid_col, ask_col, mid_col)

    direction = np.ones(len(df), dtype=np.int8)
    direction[net > horizon.flat_margin] = 2
    direction[net < -horizon.flat_margin] = 0

    df["label_return_fwd_gross"] = gross
    df["label_return_fwd_net"] = net
    df["label_execution_cost"] = np.where(valid, gross - net, 0.0)
    df["label_direction_net"] = direction
    df["label_net_valid"] = valid
    return df


def build_dual_leg_net_labels(
    df: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    horizon: LabelHorizon = LabelHorizon(),
) -> pd.DataFrame:
    """
    双腿标签:CALL ATM 与 PUT ATM 各自独立计算"买入该腿"的 fill 口径净收益。

    关键口径:买 PUT 不是"负的 CALL 收益" —— PUT 有自己的权利金基数、
    点差与 IV 路径,必须用 exec_put_* 报价单独算。产出:
      label_call_return_fwd_net / label_put_return_fwd_net   两腿净收益(有符号)
      label_straddle_return_fwd_net   跨式(同时买两腿)净收益,
        = (E_c·r_c + E_p·r_p) / (E_c + E_p),E 为各腿入场 fill 价(权利金加权);
        每腿的 r 已含各自佣金,故该式对双合约佣金也是精确的。
        大多数交易日为负(双份 theta)—— 这是跨式的真实成本结构,不截断。
      label_return_fwd_net 等主标签列                          沿用 CALL 腿(向后兼容)
      label_direction_net                                      改为双腿口径:
        2 = CALL 腿净有利, 0 = PUT 腿净有利, 1 = 两腿都无利可图
    训练时 call/put 双头的目标为 clamp(净收益, min=0)(在 loss 内处理),
    与 softplus 非负输出对齐;straddle 头为有符号回归。
    """
    call_gross, call_net, call_valid, call_prem = _leg_net_arrays(
        df, fill_model, horizon, "exec_call_bid", "exec_call_ask", "exec_call_mid"
    )
    put_gross, put_net, put_valid, put_prem = _leg_net_arrays(
        df, fill_model, horizon, "exec_put_bid", "exec_put_ask", "exec_put_mid"
    )

    df["label_return_fwd_gross"] = call_gross
    df["label_return_fwd_net"] = call_net
    df["label_execution_cost"] = np.where(call_valid, call_gross - call_net, 0.0)
    df["label_net_valid"] = call_valid

    df["label_call_return_fwd_net"] = call_net
    df["label_put_return_fwd_net"] = put_net
    df["label_put_net_valid"] = put_valid

    both_valid = call_valid & put_valid & (call_prem + put_prem > 0)
    total_prem = np.where(both_valid, call_prem + put_prem, np.nan)
    straddle = np.where(
        both_valid, (call_prem * call_net + put_prem * put_net) / total_prem, 0.0
    )
    df["label_straddle_return_fwd_net"] = np.nan_to_num(straddle, nan=0.0)
    df["label_straddle_valid"] = both_valid

    m = horizon.flat_margin
    direction = np.ones(len(df), dtype=np.int8)
    direction[(call_net > m) & (call_net >= put_net)] = 2
    direction[(put_net > m) & (put_net > call_net)] = 0
    df["label_direction_net"] = direction
    return df


def build_perp_net_labels(
    df: pd.DataFrame,
    fill_model: PerpFillModel,
    horizon: LabelHorizon = LabelHorizon(),
    price_col: str = "close",
    funding_col: str = "funding_rate_8h",
    bar_seconds: int = 60,
) -> pd.DataFrame:
    """
    BTC 永续 net 标签(做多视角):费率+滑点+持有期 funding。
    输出列命名与期权版一致,训练/回放代码无需分叉。
    """
    n = len(df)
    ed, hd = horizon.entry_delay_bars, horizon.hold_bars
    px = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=np.float64)
    if funding_col in df.columns:
        funding = pd.to_numeric(df[funding_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        funding = np.zeros(n, dtype=np.float64)

    gross = np.zeros(n, dtype=np.float64)
    net = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    valid_len = max(0, n - horizon.total_bars)
    if valid_len > 0:
        idx = np.arange(valid_len)
        e_idx = idx + ed
        x_idx = idx + ed + hd
        entry_mark = px[e_idx]
        exit_mark = px[x_idx]
        ok = np.isfinite(entry_mark) & (entry_mark > 0) & np.isfinite(exit_mark) & (exit_mark > 0)

        entry_px = fill_model.entry_fill(entry_mark)
        exit_px = fill_model.exit_fill(exit_mark)
        funding_drag = fill_model.funding_drag(funding[e_idx], holding_seconds=hd * bar_seconds)

        g = np.where(ok, exit_mark / np.where(entry_mark > 0, entry_mark, np.nan) - 1.0, 0.0)
        v = np.where(
            ok,
            exit_px / np.where(entry_px > 0, entry_px, np.nan) - 1.0 - funding_drag,
            0.0,
        )
        gross[idx] = np.nan_to_num(g, nan=0.0)
        net[idx] = np.nan_to_num(v, nan=0.0)
        valid[idx] = ok

    direction = np.ones(n, dtype=np.int8)
    direction[net > horizon.flat_margin] = 2
    direction[net < -horizon.flat_margin] = 0

    df["label_return_fwd_gross"] = gross
    df["label_return_fwd_net"] = net
    df["label_execution_cost"] = np.where(valid, gross - net, 0.0)
    df["label_direction_net"] = direction
    df["label_net_valid"] = valid
    return df


def label_quality_report(df: pd.DataFrame) -> dict:
    """
    P0 数据完整性检查(对应 ARCHITECTURE_NET_EDGE.md 的验收清单):
    net 方差非零、非零占比、cost 分布是否现实。
    """
    net = pd.to_numeric(df["label_return_fwd_net"], errors="coerce")
    cost = pd.to_numeric(df["label_execution_cost"], errors="coerce")
    valid = df.get("label_net_valid", pd.Series(True, index=df.index)).astype(bool)
    net_v = net[valid]
    cost_v = cost[valid]
    return {
        "rows": int(len(df)),
        "valid_rows": int(valid.sum()),
        "net_std": float(net_v.std()) if len(net_v) else 0.0,
        "net_nonzero_pct": float((net_v.abs() > 1e-9).mean() * 100) if len(net_v) else 0.0,
        "net_positive_pct": float((net_v > 0).mean() * 100) if len(net_v) else 0.0,
        "cost_mean_bps": float(cost_v.mean() * 1e4) if len(cost_v) else 0.0,
        "cost_p90_bps": float(cost_v.quantile(0.9) * 1e4) if len(cost_v) else 0.0,
    }
