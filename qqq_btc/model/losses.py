#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2 损失 —— 从原 StrategicAlphaLoss 内化,结构性移除 rank 项,pinball 一等公民。

组成:
  l_dir   方向 CE,按 |net| 与可执行 edge 加权(高价值样本权重大)
  l_net   net_edge 对 label_return_fwd_net 的 smooth_l1
  l_gross / l_cost   毛收益与执行成本回归(辅助头)
  l_raw   net_edge_raw 与 sign(gross)*max(|gross|-cost,0) 的一致性
  l_q     q10/q50/q90 pinball 分位损失
  l_cp    call/put 双腿头(softplus 非负)对 clamp(各腿净收益, min=0) 的回归;
          仅当 batch 提供 call_return_fwd/put_return_fwd(双腿标签管线)时生效
  l_strad 跨式头(有符号)对 label_straddle 的回归;负值区(theta 燃烧日)
          不截断 —— "今天不值得买波动"本身是要学的信号
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    err = target - pred
    return torch.maximum(quantile * err, (quantile - 1.0) * err).mean()


class NetEdgeLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        lw = (config or {}).get("loss_weights", {})
        self.w_dir = float(lw.get("direction_net", lw.get("direction", 4.0)))
        self.w_net = float(lw.get("return_fwd_net", 3.0))
        self.w_gross = float(lw.get("return_fwd_gross", 0.5))
        self.w_cost = float(lw.get("execution_cost", 0.5))
        self.w_quantile = float(lw.get("net_edge_quantile", 1.0))
        self.w_call_put = float(lw.get("call_put_edge", 0.0))
        self.w_straddle = float(lw.get("straddle_edge", 0.0))
        self.beta = float(
            (config or {}).get("parameters", {}).get("training", {}).get("return_huber_beta", 0.001)
        )

    def forward(self, out, target):
        net_r = torch.nan_to_num(target["return_fwd"], nan=0.0)
        gross_r = torch.nan_to_num(target.get("return_fwd_gross", net_r), nan=0.0)
        exec_cost = torch.nan_to_num(target.get("execution_cost", torch.zeros_like(net_r)), nan=0.0)

        ce = self.ce(out["logits_dir"], target["direction"])
        executable_edge = torch.clamp(torch.abs(gross_r) - exec_cost, min=0.0)
        w = torch.abs(net_r) * 12.0 + executable_edge * 8.0 + 1.0
        l_dir = (ce * torch.clamp(w, max=20.0)).mean()

        # 可交易量级(|net|>=1.5%)样本加权,避免被大量近零 theta 损耗主导
        trade_w = 1.0 + 3.0 * (torch.abs(net_r) >= 0.015).float()
        l_net = (
            F.smooth_l1_loss(out["net_edge"].squeeze(-1), net_r, beta=self.beta, reduction="none")
            * trade_w
        ).mean()
        l_gross = F.smooth_l1_loss(out["gross_return"].squeeze(-1), gross_r, beta=self.beta)
        l_cost = F.smooth_l1_loss(out["execution_cost"].squeeze(-1), exec_cost, beta=self.beta)
        l_raw = F.smooth_l1_loss(
            out["net_edge_raw"].squeeze(-1),
            torch.sign(gross_r) * executable_edge,
            beta=self.beta,
        )
        # 分位损失同样加重可交易样本,改善 q10 校准
        def _w_pinball(pred, target, q):
            err = target - pred
            pb = torch.maximum(q * err, (q - 1.0) * err)
            return (pb * trade_w).mean()

        l_q = (
            _w_pinball(out["net_edge_q10"].squeeze(-1), net_r, 0.1)
            + _w_pinball(out["net_edge_q50"].squeeze(-1), net_r, 0.5)
            + _w_pinball(out["net_edge_q90"].squeeze(-1), net_r, 0.9)
        )

        loss = (
            self.w_dir * l_dir
            + self.w_net * l_net
            + self.w_gross * l_gross
            + self.w_cost * l_cost
            + 0.5 * l_raw
            + self.w_quantile * l_q
        )

        if self.w_call_put > 0 and "call_return_fwd" in target and "put_return_fwd" in target:
            call_r = torch.clamp(torch.nan_to_num(target["call_return_fwd"], nan=0.0), min=0.0)
            put_r = torch.clamp(torch.nan_to_num(target["put_return_fwd"], nan=0.0), min=0.0)
            l_cp = (
                F.smooth_l1_loss(out["call_net_edge"].squeeze(-1), call_r, beta=self.beta)
                + F.smooth_l1_loss(out["put_net_edge"].squeeze(-1), put_r, beta=self.beta)
            )
            loss = loss + self.w_call_put * l_cp

        if self.w_straddle > 0 and "straddle_return_fwd" in target:
            strad_r = torch.nan_to_num(target["straddle_return_fwd"], nan=0.0)
            l_strad = F.smooth_l1_loss(out["straddle_net_edge"].squeeze(-1), strad_r, beta=self.beta)
            loss = loss + self.w_straddle * l_strad

        return loss, l_q.item()
