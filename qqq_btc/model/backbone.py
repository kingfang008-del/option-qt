#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双流 TFT 骨架 —— 从 New_Pro/model/trading_tft_stock_embed.py(v11.0)内化,
qqq_btc 自此不再依赖 New_Pro 代码。

内化时修复的设计问题:
  - 纯网络层:本模块只依赖 torch,无 LMDB/msgpack/psycopg2/日志文件副作用
    (原文件 import 即写 train_advanced_alpha.log、改 sys.path、连 PG)
  - 删除 rank 流(单标的无截面语义),分位数头(q10/q50/q90)为一等公民
  - embedding 容量来自 config(原来查 Postgres 拿 max_stock_id=18000,
    单标的路径下浪费三张 18000 x hidden 的表)
  - 微调冻结白名单集中在本模块,与训练脚本解耦

网络结构与原版逐层一致(GLU/GRN/VSN/LSTM/可解释多头注意力/双塔/融合),
保证 SPY+QQQ 共训 checkpoint 可通过 load_pretrain_checkpoint 迁移。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)

# 静态特征(不进时序双塔)
STATIC_FEATURE_NAMES = ("stock_id", "sector_id", "day_of_week")

# embedding 容量默认值:不再查 PG。多标的共训时在 config
# parameters.qqq_btc_v2.embedding_caps 指定;若 LMDB 里的 stock_id 是
# 大范围的 PG id,保持默认大容量以免 clamp 碰撞。
DEFAULT_EMBEDDING_CAPS = {"stock": 20000, "sector": 256}


# ==============================================================================
# 基础组件(与原版逐层一致)
# ==============================================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, context_dim=None):
        super().__init__()
        self.context_dim = context_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu1 = nn.ELU()
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = GatedLinearUnit(hidden_dim, output_dim, dropout)
        self.ln = nn.LayerNorm(output_dim)
        self.res_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.res_proj(x) if self.res_proj else x
        x_enc = self.fc1(x)
        if context is not None and self.context_dim is not None:
            context_enc = self.context_proj(context)
            if x_enc.dim() == 3 and context_enc.dim() == 2:
                context_enc = context_enc.unsqueeze(1)
            x_enc = x_enc + context_enc
        x_enc = self.fc2(self.elu1(x_enc))
        return self.ln(self.gate(x_enc) + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim, dropout=0.3, context_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = len(input_dims)
        self.single_variable_grns = nn.ModuleList(
            GatedResidualNetwork(dim, hidden_dim, hidden_dim, dropout, context_dim=context_dim)
            for dim in input_dims
        )
        self.flattened_grn = GatedResidualNetwork(
            self.num_inputs * hidden_dim, hidden_dim, self.num_inputs, dropout, context_dim=context_dim
        )

    def forward(self, embedding_list, context=None):
        if self.num_inputs == 0 or not embedding_list:
            return None, None
        processed = [grn(emb, context) for grn, emb in zip(self.single_variable_grns, embedding_list)]
        is_temporal = processed[0].dim() == 3
        stack_dim = 2 if is_temporal else 1
        stacked = torch.stack(processed, dim=stack_dim)
        flattened = stacked.flatten(start_dim=stack_dim)
        weights = F.softmax(self.flattened_grn(flattened, context), dim=-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=stack_dim), weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.3):
        super().__init__()
        self.n_head = num_heads
        self.d_head = hidden_dim // num_heads
        self.qkv_linears = nn.Linear(hidden_dim, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split(
            (self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1
        )
        q = q.view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, T, self.d_head).unsqueeze(1)

        attn_score = torch.matmul(q, k) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))
        attn_score = torch.nan_to_num(attn_score, nan=-1e9)
        attn_prob = self.attn_dropout(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, v.repeat(1, self.n_head, 1, 1))
        attn_vec = attn_vec.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out_dropout(self.out_proj(attn_vec))


class GateAddNorm(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.glu = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.ln(self.glu(x) + residual)


class TFTEncoder(nn.Module):
    """单塔时序编码器(原 NvidiaTFTWrapper):VSN → LSTM → 静态富化 → 因果注意力。"""

    def __init__(self, hidden_dim, num_reals, num_cats, dropout=0.3, max_cat_cardinality=51):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_inputs = (num_reals + num_cats) > 0
        if not self.has_inputs:
            return

        self.real_projections = nn.ModuleList(nn.Linear(1, hidden_dim) for _ in range(num_reals))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(max_cat_cardinality, hidden_dim) for _ in range(num_cats)
        )
        self.max_cat_idx = max_cat_cardinality - 1

        input_dims = [hidden_dim] * (num_reals + num_cats)
        self.vsn = VariableSelectionNetwork(input_dims, hidden_dim, dropout, context_dim=hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.post_lstm_gate = GateAddNorm(hidden_dim, dropout)
        self.static_enrichment = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim
        )
        self.post_enrich_gate = GateAddNorm(hidden_dim, dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.post_attn_gate = GateAddNorm(hidden_dim, dropout)
        self.pos_wise_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_pos_gate = GateAddNorm(hidden_dim, dropout)

    def forward(self, x_reals, x_cats, c_s, c_h, c_c):
        B = c_s.shape[0]
        if not self.has_inputs:
            return torch.zeros(B, self.hidden_dim, device=c_s.device)

        embeddings = []
        for i, proj in enumerate(self.real_projections):
            real_val = torch.nan_to_num(x_reals[..., i], nan=0.0)
            embeddings.append(proj(real_val.unsqueeze(-1)))
        for i, embed in enumerate(self.cat_embeddings):
            idx = torch.clamp(x_cats[..., i].long(), 0, self.max_cat_idx)
            embeddings.append(embed(idx))

        T = embeddings[0].shape[1]
        vsn_out, _ = self.vsn(embeddings, context=c_s)
        lstm_out, _ = self.lstm(vsn_out, (c_h, c_c))
        lstm_out = self.post_lstm_gate(lstm_out, vsn_out)
        enriched = self.static_enrichment(lstm_out, context=c_s)
        enriched = self.post_enrich_gate(enriched, lstm_out)

        mask = torch.triu(torch.ones(T, T, device=c_s.device), diagonal=1).bool()
        attn_out = self.attention(enriched, mask)
        attn_out = self.post_attn_gate(attn_out, enriched)
        output = self.pos_wise_grn(attn_out)
        output = self.post_pos_gate(output, attn_out)
        return output[:, -1, :]


class PerSymbolCalibrator(nn.Module):
    """按标的仿射校准:共训时吸收各标的 edge 量级/门槛差(如 SPY vs QQQ)。"""

    def __init__(self, max_stock_id, hidden_dim):
        super().__init__()
        self.scale = nn.Embedding(max_stock_id + 1, 1)
        self.bias = nn.Embedding(max_stock_id + 1, 1)
        self.context = nn.Embedding(max_stock_id + 1, hidden_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.bias.weight)
        nn.init.normal_(self.context.weight, mean=0.0, std=0.02)

    def forward(self, stock_id, raw_edge):
        sid = torch.clamp(stock_id.long(), 0, self.scale.num_embeddings - 1)
        scale = F.softplus(self.scale(sid)) + 0.25
        return raw_edge * scale + self.bias(sid), self.context(sid)


class MonotoneQuantileHead(nn.Module):
    """q10 = q50 - softplus,q90 = q50 + softplus,结构上不交叉。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.q50 = nn.Linear(hidden_dim // 2, 1)
        self.down = nn.Linear(hidden_dim // 2, 1)
        self.up = nn.Linear(hidden_dim // 2, 1)

    def forward(self, fused: torch.Tensor):
        h = self.trunk(fused)
        q50 = self.q50(h)
        return q50 - F.softplus(self.down(h)), q50, q50 + F.softplus(self.up(h))


# ==============================================================================
# 双流主网络(v2:无 rank 流,分位数头内置)
# ==============================================================================

def resolve_embedding_caps(config: dict) -> dict:
    caps = dict(DEFAULT_EMBEDDING_CAPS)
    cfg_caps = (
        (config.get("parameters") or {}).get("qqq_btc_v2", {}).get("embedding_caps") or {}
    )
    caps.update({k: int(v) for k, v in cfg_caps.items()})
    return caps


def analyze_features(feature_config: list) -> dict:
    """按 options_ 前缀拆分 Stock/Option 双塔的列索引(与 LMDB 数据集约定一致)。"""
    info = {"stock": {"real": [], "cat": []}, "option": {"real": [], "cat": []}}
    idx = {"stock": 0, "option": 0}
    for f in feature_config:
        name = f["name"]
        if name in STATIC_FEATURE_NAMES:
            continue
        ftype = "cat" if f.get("type") == "categorical" else "real"
        tower = "option" if name.startswith("options_") else "stock"
        info[tower][ftype].append(idx[tower])
        idx[tower] += 1
    return info


class DualStreamAlphaNet(nn.Module):
    """
    Stock Tower + Option Tower → fusion → 多头输出。
    输出:logits_dir / gross_return / execution_cost / net_edge(_raw)
         / net_edge_q10/q50/q90 / call_net_edge / put_net_edge / straddle_net_edge。

    call_net_edge / put_net_edge 经 softplus 保证非负;训练用有符号标签 + 负样本压零,
    不再 clamp(label,min=0)。rank 损失提升 bar 级排序。
    straddle_net_edge 为有符号输出:跨式大多数交易日净收益为负(双份 theta),
    负值区分度("-2% 还是 -30%")本身就是信息,不能截断。
    """

    def __init__(self, config, caps=None, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.config = config
        caps = caps or resolve_embedding_caps(config)
        self.feat_info = analyze_features(config["features"])

        self.static_stock_embed = nn.Embedding(caps["stock"] + 1, hidden_dim)
        self.static_sector_embed = nn.Embedding(caps["sector"] + 1, hidden_dim)
        self.static_dow_embed = nn.Embedding(8, hidden_dim)
        self.static_vsn = VariableSelectionNetwork([hidden_dim] * 3, hidden_dim, dropout)
        self.grn_cs = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_ch = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_cc = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        self.tft_stock = TFTEncoder(
            hidden_dim,
            len(self.feat_info["stock"]["real"]),
            len(self.feat_info["stock"]["cat"]),
            dropout,
        )
        # 期权塔更高 dropout(原版设计:期权特征噪声更大)
        self.tft_option = TFTEncoder(
            hidden_dim,
            len(self.feat_info["option"]["real"]),
            len(self.feat_info["option"]["cat"]),
            dropout=0.45,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU(), nn.Dropout(dropout)
        )

        def head(out_dim=1):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, out_dim)
            )

        self.head_dir = head(3)
        self.head_gross_return = head()
        self.head_execution_cost = head()
        self.head_net_edge = head()
        self.head_call_net_edge = head()
        self.head_put_net_edge = head()
        self.head_straddle_net_edge = head()
        self.head_net_edge_quantile = MonotoneQuantileHead(hidden_dim)
        self.symbol_calibrator = PerSymbolCalibrator(caps["stock"], hidden_dim)

    def forward(self, x_stock, x_option, static_x):
        s_id = torch.clamp(static_x["stock_id"].long(), 0, self.static_stock_embed.num_embeddings - 1)
        sec_id = torch.clamp(static_x["sector_id"].long(), 0, self.static_sector_embed.num_embeddings - 1)
        d_id = torch.clamp(static_x["day_of_week"].long(), 0, 7)

        static_emb, _ = self.static_vsn(
            [self.static_stock_embed(s_id), self.static_sector_embed(sec_id), self.static_dow_embed(d_id)]
        )
        c_s = self.grn_cs(static_emb)
        c_h = self.grn_ch(static_emb).unsqueeze(0)
        c_c = self.grn_cc(static_emb).unsqueeze(0)

        emb_stock = self.tft_stock(
            x_stock[..., self.feat_info["stock"]["real"]],
            x_stock[..., self.feat_info["stock"]["cat"]],
            c_s, c_h, c_c,
        )
        emb_option = self.tft_option(
            x_option[..., self.feat_info["option"]["real"]],
            x_option[..., self.feat_info["option"]["cat"]],
            c_s, c_h, c_c,
        )

        fused = self.fusion(torch.cat([emb_stock, emb_option], dim=-1))
        raw_net_edge = self.head_net_edge(fused)
        calibrated_net_edge, _ = self.symbol_calibrator(static_x["stock_id"], raw_net_edge)
        q10, q50, q90 = self.head_net_edge_quantile(fused)

        return {
            "logits_dir": self.head_dir(fused),
            "gross_return": self.head_gross_return(fused),
            "execution_cost": F.softplus(self.head_execution_cost(fused)),
            "net_edge_raw": raw_net_edge,
            "net_edge": calibrated_net_edge,
            "net_edge_q10": q10,
            "net_edge_q50": q50,
            "net_edge_q90": q90,
            "call_net_edge": F.softplus(self.head_call_net_edge(fused)),
            "put_net_edge": F.softplus(self.head_put_net_edge(fused)),
            "straddle_net_edge": self.head_straddle_net_edge(fused),
        }


# ==============================================================================
# 微调冻结 / checkpoint 迁移
# ==============================================================================

FINETUNE_TRAINABLE_PREFIXES = (
    "fusion",
    "head_dir",
    "head_gross_return",
    "head_execution_cost",
    "head_net_edge",        # 含 head_net_edge_quantile
    "head_call_net_edge",
    "head_put_net_edge",
    "head_straddle_net_edge",
    "symbol_calibrator",
)


def freeze_for_finetune(model: DualStreamAlphaNet) -> tuple[int, int]:
    """P2 校准阶段:冻结双塔与静态嵌入,只训 fusion + heads + calibrator。"""
    trainable, total = 0, 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(FINETUNE_TRAINABLE_PREFIXES)
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def load_pretrain_checkpoint(
    model: DualStreamAlphaNet,
    ckpt_path: str,
    device: str = "cpu",
    *,
    allow_shape_mismatch: bool = True,
) -> list[str]:
    """
    加载共训/旧版 checkpoint:旧 head_rank 权重丢弃,
    新 quantile 头保持随机初始化,其余尽量完整匹配。
    allow_shape_mismatch=True 时跳过形状不兼容的层(如新增 chop 特征导致
    stock 塔输入维度变化),用于 v4→v5 迁移。
    返回被跳过的 key 列表。
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    state = {k: v for k, v in state.items() if not k.startswith("head_rank")}
    model_state = model.state_dict()
    skipped_shape: list[str] = []
    if allow_shape_mismatch:
        filtered = {}
        for k, v in state.items():
            if k not in model_state:
                continue
            if tuple(model_state[k].shape) != tuple(v.shape):
                skipped_shape.append(k)
                continue
            filtered[k] = v
        state = filtered
    missing, unexpected = model.load_state_dict(state, strict=False)
    _new_heads = ("head_net_edge_quantile", "head_straddle_net_edge")
    keep_missing = [m for m in missing if not m.startswith(_new_heads)]
    if keep_missing and not allow_shape_mismatch:
        raise RuntimeError(f"checkpoint 缺少非预期权重: {keep_missing[:8]}")
    if unexpected and not allow_shape_mismatch:
        raise RuntimeError(f"checkpoint 含未知权重: {unexpected[:8]}")
    return skipped_shape
