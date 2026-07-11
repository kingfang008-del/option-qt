#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2 损失 —— 从原 StrategicAlphaLoss 内化,pinball 一等公民。

组成:
  l_dir   方向 CE,按 |net| 与可执行 edge 加权(高价值样本权重大)
  l_net   net_edge 对 label_return_fwd_net 的 smooth_l1
  l_gross / l_cost   毛收益与执行成本回归(辅助头)
  l_raw   net_edge_raw 与 sign(gross)*max(|gross|-cost,0) 的一致性
  l_q     q10/q50/q90 pinball 分位损失
  l_rank  net_edge 成对排序损失(提升 bar 级区分度)
  l_cp    call/put 双腿头对有符号净收益的回归 + 各腿 rank 损失;
          不再 clamp(label, min=0) —— 负收益必须压低预测分
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


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    min_delta: float,
) -> torch.Tensor:
    """成对排序:label 更高者预测也应更高(随机配对,|diff|>min_delta 才计损失)。"""
    if pred.numel() <= 1:
        return pred.new_zeros(())
    idx = torch.randperm(pred.numel(), device=pred.device)
    diff = target - target[idx]
    mask = torch.abs(diff) > min_delta
    if not mask.any():
        return pred.new_zeros(())
    sign = torch.sign(diff[mask])
    pred_diff = pred[mask] - pred[idx][mask]
    weight = torch.clamp(torch.abs(diff[mask]) / min_delta, max=10.0)
    return (F.softplus(-sign * pred_diff / temperature) * weight).mean()


class NetEdgeLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.route_class_weights = ((config or {}).get("route_class_weights") or {})
        lw = (config or {}).get("loss_weights", {})
        train_cfg = (config or {}).get("parameters", {}).get("training", {})
        self.w_dir = float(lw.get("direction_net", lw.get("direction", 4.0)))
        self.w_net = float(lw.get("return_fwd_net", 3.0))
        self.w_gross = float(lw.get("return_fwd_gross", 0.5))
        self.w_cost = float(lw.get("execution_cost", 0.5))
        self.w_quantile = float(lw.get("net_edge_quantile", 1.0))
        self.w_rank = float(lw.get("rank_net", 0.0))
        self.w_call_put = float(lw.get("call_put_edge", 0.0))
        self.w_straddle = float(lw.get("straddle_edge", 0.0))
        self.w_best_side = float(lw.get("best_side", 0.0))
        self.w_best_bucket = float(lw.get("best_bucket", 0.0))
        self.w_spot_direction = float(lw.get("spot_direction", 0.0))
        self.w_spot_return = float(lw.get("spot_return", 0.0))
        self.beta = float(train_cfg.get("return_huber_beta", 0.001))
        self.rank_temp = float(train_cfg.get("rank_temperature", 0.002))
        self.rank_min_delta = float(train_cfg.get("rank_min_delta", 0.0002))

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
        net_pred = out["net_edge"].squeeze(-1)
        l_net = (
            F.smooth_l1_loss(net_pred, net_r, beta=self.beta, reduction="none")
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

        l_rank = net_pred.new_zeros(())
        if self.w_rank > 0:
            l_rank = pairwise_rank_loss(
                net_pred, net_r, temperature=self.rank_temp, min_delta=self.rank_min_delta
            )

        loss = (
            self.w_dir * l_dir
            + self.w_net * l_net
            + self.w_gross * l_gross
            + self.w_cost * l_cost
            + 0.5 * l_raw
            + self.w_quantile * l_q
            + self.w_rank * l_rank
        )

        if self.w_call_put > 0 and "call_return_fwd" in target and "put_return_fwd" in target:
            call_r = torch.nan_to_num(target["call_return_fwd"], nan=0.0)
            put_r = torch.nan_to_num(target["put_return_fwd"], nan=0.0)
            call_pred = out["call_net_edge"].squeeze(-1)
            put_pred = out["put_net_edge"].squeeze(-1)
            leg_gap = call_r - put_r
            leg_rank_mask = torch.abs(leg_gap) > self.rank_min_delta
            if leg_rank_mask.any():
                leg_rank = F.softplus(
                    -torch.sign(leg_gap[leg_rank_mask])
                    * (call_pred[leg_rank_mask] - put_pred[leg_rank_mask])
                    / self.rank_temp
                ).mean()
            else:
                leg_rank = call_pred.new_zeros(())
            call_w = (
                1.0
                + 5.0 * (call_r > 0.015).float()
                + 3.0 * ((call_r > put_r) & (call_r > 0.0)).float()
            )
            put_w = (
                1.0
                + 5.0 * (put_r > 0.015).float()
                + 3.0 * ((put_r > call_r) & (put_r > 0.0)).float()
            )
            l_cp = (
                (
                    F.smooth_l1_loss(call_pred, call_r, beta=self.beta, reduction="none")
                    * call_w
                ).mean()
                + (
                    F.smooth_l1_loss(put_pred, put_r, beta=self.beta, reduction="none")
                    * put_w
                ).mean()
                + leg_rank
            )
            loss = loss + self.w_call_put * l_cp

        if self.w_straddle > 0 and "straddle_return_fwd" in target:
            strad_r = torch.nan_to_num(target["straddle_return_fwd"], nan=0.0)
            l_strad = F.smooth_l1_loss(out["straddle_net_edge"].squeeze(-1), strad_r, beta=self.beta)
            loss = loss + self.w_straddle * l_strad

        if self.w_best_side > 0 and "best_side" in target and "logits_best_side" in out:
            route_w = torch.clamp(torch.abs(net_r) * 10.0 + 1.0, max=8.0)
            side_weights = self.route_class_weights.get("best_side")
            if side_weights is not None:
                side_weights = torch.as_tensor(
                    side_weights,
                    dtype=out["logits_best_side"].dtype,
                    device=out["logits_best_side"].device,
                )
                side_ce = F.cross_entropy(
                    out["logits_best_side"],
                    target["best_side"],
                    weight=side_weights,
                    reduction="none",
                )
            else:
                side_ce = self.ce(out["logits_best_side"], target["best_side"])
            l_side = (side_ce * route_w).mean()
            loss = loss + self.w_best_side * l_side

        if self.w_best_bucket > 0 and "best_bucket" in target and "logits_best_bucket" in out:
            route_w = torch.clamp(torch.abs(net_r) * 10.0 + 1.0, max=8.0)
            bucket_weights = self.route_class_weights.get("best_bucket")
            if bucket_weights is not None:
                bucket_weights = torch.as_tensor(
                    bucket_weights,
                    dtype=out["logits_best_bucket"].dtype,
                    device=out["logits_best_bucket"].device,
                )
            else:
                bucket_weights = None
            l_bucket = (
                F.cross_entropy(
                    out["logits_best_bucket"],
                    target["best_bucket"],
                    weight=bucket_weights,
                    reduction="none",
                )
                * route_w
            ).mean()
            loss = loss + self.w_best_bucket * l_bucket

        if self.w_spot_direction > 0 and "spot_direction" in target and "logits_spot_dir" in out:
            spot_r = torch.nan_to_num(target.get("spot_return_fwd", torch.zeros_like(net_r)), nan=0.0)
            spot_w = torch.clamp(torch.abs(spot_r) * 500.0 + 1.0, max=8.0)
            l_spot_dir = (
                self.ce(out["logits_spot_dir"], target["spot_direction"]) * spot_w
            ).mean()
            loss = loss + self.w_spot_direction * l_spot_dir

        if self.w_spot_return > 0 and "spot_return_fwd" in target and "spot_return" in out:
            spot_r = torch.nan_to_num(target["spot_return_fwd"], nan=0.0)
            l_spot_ret = F.smooth_l1_loss(
                out["spot_return"].squeeze(-1),
                spot_r,
                beta=max(self.beta, 0.0005),
            )
            loss = loss + self.w_spot_return * l_spot_ret

        return loss, l_q.item()
