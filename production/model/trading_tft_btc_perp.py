#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC perpetual TFT model with delayed executable labels.

This keeps the two-tower idea from the stock-option model:
  - one continuous instrument stream
  - no option chain / strike / expiry selection
  - market/technical features tower + derivatives/microstructure features tower
  - direction is LONG / FLAT / SHORT
  - rank_score is a signed executable edge

Expected input from production/preprocess/BTC/build_btc_tft_features.py:
  output_dir/
    features_1min.parquet
    features_5min.parquet

If bid/ask columns are present, labels use:
  long_ret  = bid[t+d+h] / ask[t+d] - 1
  short_ret = bid[t+d] / ask[t+d+h] - 1

If bid/ask columns are absent, the script falls back to close-price proxy labels
and logs a warning. That proxy is useful for plumbing only, not live evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("train_btc_perp_tft.log", mode="a")],
)
logger = logging.getLogger("BTC_TFT")

EPS = 1e-9


@dataclass(frozen=True)
class LabelConfig:
    entry_delay_steps: int = 1
    holding_steps: int = 5
    min_edge: float = 0.0005
    cost_buffer: float = 0.0002
    require_bidask: bool = False


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.res = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        y = F.elu(self.fc1(x))
        y = self.fc2(y)
        y = self.glu(y)
        return self.norm(y + self.res(x))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int, dropout: float = 0.1, context_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = len(input_dims)
        self.single_variable_grns = nn.ModuleList(
            [GatedResidualNetwork(dim, hidden_dim, hidden_dim, dropout) for dim in input_dims]
        )
        self.flattened_grn = GatedResidualNetwork(
            self.num_inputs * hidden_dim,
            hidden_dim,
            self.num_inputs,
            dropout,
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding_list: list[torch.Tensor], context: torch.Tensor | None = None):
        if self.num_inputs == 0 or not embedding_list:
            return None, None

        processed_vars = []
        context_bias = self.context_proj(context) if context is not None and self.context_proj is not None else None
        if context_bias is not None and embedding_list[0].dim() == 3:
            context_bias = context_bias.unsqueeze(1)

        for i, emb in enumerate(embedding_list):
            processed = self.single_variable_grns[i](emb)
            if context_bias is not None:
                processed = processed + context_bias
            processed_vars.append(processed)

        is_temporal = processed_vars[0].dim() == 3
        if is_temporal:
            stacked = torch.stack(processed_vars, dim=2)
            flattened = stacked.flatten(start_dim=2)
        else:
            stacked = torch.stack(processed_vars, dim=1)
            flattened = stacked.flatten(start_dim=1)

        weights = self.softmax(self.flattened_grn(flattened)).unsqueeze(-1)
        sum_dim = 2 if is_temporal else 1
        return (stacked * weights).sum(dim=sum_dim), weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_head = num_heads
        self.d_head = hidden_dim // num_heads
        self.qkv_linears = nn.Linear(hidden_dim, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(b, t, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(b, t, self.d_head).unsqueeze(1)

        attn_score = torch.matmul(q, k) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))
        attn_prob = F.softmax(torch.nan_to_num(attn_score, nan=-1e9), dim=-1)
        attn_prob = self.attn_dropout(attn_prob)
        attn_vec = torch.matmul(attn_prob, v.repeat(1, self.n_head, 1, 1))
        attn_vec = attn_vec.permute(0, 2, 1, 3).contiguous().view(b, t, -1)
        return self.out_dropout(self.out_proj(attn_vec))


class GateAddNorm(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        return self.norm(self.glu(x) + residual)


class BtcTFTBackbone(nn.Module):
    def __init__(self, num_reals: int, cat_cardinalities: list[int], hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_inputs = (num_reals + len(cat_cardinalities)) > 0
        if not self.has_inputs:
            return

        self.real_projections = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_reals)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(max(2, int(card)), hidden_dim) for card in cat_cardinalities])
        self.vsn = VariableSelectionNetwork(
            [hidden_dim] * (num_reals + len(cat_cardinalities)),
            hidden_dim,
            dropout,
            context_dim=hidden_dim,
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.post_lstm_gate = GateAddNorm(hidden_dim, dropout)
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_enrich_gate = GateAddNorm(hidden_dim, dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.post_attn_gate = GateAddNorm(hidden_dim, dropout)
        self.pos_wise_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_pos_gate = GateAddNorm(hidden_dim, dropout)

    def forward(self, x_real, x_cat, c_s, c_h, c_c):
        b = c_s.shape[0]
        if not self.has_inputs:
            return torch.zeros(b, self.hidden_dim, device=c_s.device)

        embeddings = []
        for i, proj in enumerate(self.real_projections):
            real_val = torch.nan_to_num(x_real[..., i], nan=0.0)
            embeddings.append(proj(real_val.unsqueeze(-1)))

        for i, embed in enumerate(self.cat_embeddings):
            idx = torch.clamp(torch.nan_to_num(x_cat[..., i], nan=0.0).long(), 0, embed.num_embeddings - 1)
            embeddings.append(embed(idx))

        vsn_out, _ = self.vsn(embeddings, context=c_s)
        lstm_out, _ = self.lstm(vsn_out, (c_h, c_c))
        lstm_out = self.post_lstm_gate(lstm_out, vsn_out)
        enriched = self.static_enrichment(lstm_out)
        enriched = self.post_enrich_gate(enriched, lstm_out)
        t = enriched.shape[1]
        mask = torch.triu(torch.ones(t, t, device=c_s.device), diagonal=1).bool()
        attn_out = self.attention(enriched, mask)
        attn_out = self.post_attn_gate(attn_out, enriched)
        output = self.pos_wise_grn(attn_out)
        output = self.post_pos_gate(output, attn_out)
        return output[:, -1, :]


class BtcPerpTFT(nn.Module):
    def __init__(
        self,
        market_real_dim: int,
        market_cat_cards: list[int],
        deriv_real_dim: int,
        deriv_cat_cards: list[int],
        hidden_dim: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.static_context = nn.Parameter(torch.zeros(hidden_dim))
        self.grn_cs = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_ch = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_cc = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.market_backbone = BtcTFTBackbone(market_real_dim, market_cat_cards, hidden_dim, dropout)
        self.deriv_backbone = BtcTFTBackbone(deriv_real_dim, deriv_cat_cards, hidden_dim, dropout)
        self.fusion = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.head_dir = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.head_rank = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        b = x["market_real"].shape[0]
        static = self.static_context.unsqueeze(0).expand(b, -1)
        c_s = self.grn_cs(static)
        c_h = self.grn_ch(static).unsqueeze(0)
        c_c = self.grn_cc(static).unsqueeze(0)

        market_h = self.market_backbone(x["market_real"], x["market_cat"], c_s, c_h, c_c)
        deriv_h = self.deriv_backbone(x["deriv_real"], x["deriv_cat"], c_s, c_h, c_c)
        h = self.fusion(torch.cat([market_h, deriv_h], dim=-1))
        return {
            "logits_dir": self.head_dir(h),
            "rank_score": self.head_rank(h).squeeze(-1),
            "edge_score": self.head_edge(h).squeeze(-1),
        }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _feature_columns(config: dict) -> tuple[list[str], list[str]]:
    feats_1m, feats_5m = [], []
    for item in config.get("features", []):
        name = item["name"]
        res = item.get("resolution", "1min")
        if res == "1min":
            feats_1m.append(name)
        elif res == "5min":
            feats_5m.append(name)
    return feats_1m, feats_5m


def _split_btc_tower_features(features: Iterable[dict]) -> dict[str, list]:
    layout = {
        "market_real": [],
        "market_cat": [],
        "market_cat_cards": [],
        "deriv_real": [],
        "deriv_cat": [],
        "deriv_cat_cards": [],
    }
    for item in features:
        col = item["name"]
        is_cat = item.get("type") == "categorical"
        is_deriv = col.startswith("deriv_")
        bucket = "deriv" if is_deriv else "market"
        if is_cat:
            layout[f"{bucket}_cat"].append(col)
            layout[f"{bucket}_cat_cards"].append(int(item.get("cardinality", 64)))
        else:
            layout[f"{bucket}_real"].append(col)
    if not layout["market_real"] and not layout["market_cat"]:
        raise ValueError("BTC two-tower model requires at least one market feature.")
    return layout


def _coerce_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    return out


def _pick_price_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    bid_candidates = ["bid", "best_bid", "bid_price", "perp_bid", "close_bid"]
    ask_candidates = ["ask", "best_ask", "ask_price", "perp_ask", "close_ask"]
    bid_col = next((c for c in bid_candidates if c in df.columns), None)
    ask_col = next((c for c in ask_candidates if c in df.columns), None)
    price_col = "close" if "close" in df.columns else next((c for c in ["mark_price", "index_price"] if c in df.columns), "")
    if not price_col:
        raise ValueError("features_1min must contain close or mark/index price.")
    return bid_col, ask_col, price_col


def attach_delayed_executable_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    out = df.copy()
    bid_col, ask_col, price_col = _pick_price_columns(out)
    use_bid_ask = bid_col is not None and ask_col is not None
    if use_bid_ask:
        bid = pd.to_numeric(out[bid_col], errors="coerce")
        ask = pd.to_numeric(out[ask_col], errors="coerce")
    else:
        if cfg.require_bidask:
            raise ValueError("Bid/ask columns missing while require_bidask=True.")
        logger.warning("Bid/ask columns missing; BTC labels fall back to close-price proxy.")
        bid = pd.to_numeric(out[price_col], errors="coerce")
        ask = pd.to_numeric(out[price_col], errors="coerce")

    d = max(0, int(cfg.entry_delay_steps))
    h = max(1, int(cfg.holding_steps))
    entry_bid = bid.shift(-d)
    entry_ask = ask.shift(-d)
    exit_bid = bid.shift(-(d + h))
    exit_ask = ask.shift(-(d + h))

    long_ret = exit_bid / entry_ask.replace(0, np.nan) - 1.0
    short_ret = entry_bid / exit_ask.replace(0, np.nan) - 1.0
    long_net = long_ret - cfg.cost_buffer
    short_net = short_ret - cfg.cost_buffer
    best_net = np.maximum(long_net, short_net)

    direction = np.ones(len(out), dtype=np.int64)
    long_ok = (long_net >= short_net) & (long_ret > cfg.min_edge)
    short_ok = (short_net > long_net) & (short_ret > cfg.min_edge)
    direction[long_ok.fillna(False).to_numpy()] = 2
    direction[short_ok.fillna(False).to_numpy()] = 0

    signed_edge = np.zeros(len(out), dtype=np.float32)
    signed_edge[long_ok.fillna(False).to_numpy()] = long_net[long_ok].astype(np.float32)
    signed_edge[short_ok.fillna(False).to_numpy()] = -short_net[short_ok].astype(np.float32)

    out["label_long_exec_ret"] = long_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["label_short_exec_ret"] = short_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["label_best_net_ret"] = pd.Series(best_net).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    out["label_return_fwd"] = signed_edge
    out["label_direction"] = direction
    out["label_tradable"] = (best_net > 0).fillna(False).astype(np.float32)
    out["label_source_bidask"] = float(use_bid_ask)
    return out


class BtcPerpDataset(Dataset):
    def __init__(
        self,
        feature_dir: Path,
        config: dict,
        seq_len: int = 60,
        label_cfg: LabelConfig = LabelConfig(),
        stage: str = "train",
        split_date: str | None = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.config = config
        self.seq_len = int(seq_len)
        self.label_cfg = label_cfg
        self.feats_1m, self.feats_5m = _feature_columns(config)

        df_1m = _coerce_ts(pd.read_parquet(self.feature_dir / "features_1min.parquet"))
        df_5m = _coerce_ts(pd.read_parquet(self.feature_dir / "features_5min.parquet"))
        df_1m = attach_delayed_executable_labels(df_1m, label_cfg)

        for col in self.feats_1m:
            if col not in df_1m.columns:
                df_1m[col] = 0.0
        for col in self.feats_5m:
            if col not in df_5m.columns:
                df_5m[col] = 0.0

        merged = pd.merge_asof(
            df_1m.sort_values("timestamp"),
            df_5m[["timestamp"] + self.feats_5m].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("10min"),
            suffixes=("", "_5m"),
        )
        merged[self.feats_1m + self.feats_5m] = merged[self.feats_1m + self.feats_5m].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if split_date:
            split_ts = pd.Timestamp(split_date)
            if stage == "train":
                merged = merged[merged["timestamp"] < split_ts]
            else:
                merged = merged[merged["timestamp"] >= split_ts]

        self.df = merged.reset_index(drop=True)
        self.feature_cols = self.feats_1m + self.feats_5m
        self.layout = _split_btc_tower_features(config.get("features", []))
        self.market_real_cols = self.layout["market_real"]
        self.market_cat_cols = self.layout["market_cat"]
        self.market_cat_cards = self.layout["market_cat_cards"]
        self.deriv_real_cols = self.layout["deriv_real"]
        self.deriv_cat_cols = self.layout["deriv_cat"]
        self.deriv_cat_cards = self.layout["deriv_cat_cards"]
        self.x_market_real = self.df[self.market_real_cols].to_numpy(dtype=np.float32) if self.market_real_cols else np.zeros(
            (len(self.df), 0), dtype=np.float32
        )
        self.x_market_cat = self.df[self.market_cat_cols].to_numpy(dtype=np.float32) if self.market_cat_cols else np.zeros(
            (len(self.df), 0), dtype=np.float32
        )
        self.x_deriv_real = self.df[self.deriv_real_cols].to_numpy(dtype=np.float32) if self.deriv_real_cols else np.zeros(
            (len(self.df), 0), dtype=np.float32
        )
        self.x_deriv_cat = self.df[self.deriv_cat_cols].to_numpy(dtype=np.float32) if self.deriv_cat_cols else np.zeros(
            (len(self.df), 0), dtype=np.float32
        )
        self.labels = self.df[
            [
                "label_return_fwd",
                "label_direction",
                "label_tradable",
                "label_best_net_ret",
                "label_long_exec_ret",
                "label_short_exec_ret",
            ]
        ].copy()
        self.valid_indices = np.arange(self.seq_len - 1, len(self.df), dtype=np.int64)
        logger.info(
            "BTC dataset %s | rows=%d samples=%d market_real=%d market_cat=%d deriv_real=%d deriv_cat=%d tradable=%.2f%% bidask=%.0f%%",
            stage,
            len(self.df),
            len(self.valid_indices),
            len(self.market_real_cols),
            len(self.market_cat_cols),
            len(self.deriv_real_cols),
            len(self.deriv_cat_cols),
            float((self.labels["label_tradable"] > 0).mean() * 100.0),
            float(self.df["label_source_bidask"].mean() * 100.0),
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end = int(self.valid_indices[idx])
        start = end - self.seq_len + 1
        x = {
            "market_real": self.x_market_real[start : end + 1],
            "market_cat": self.x_market_cat[start : end + 1],
            "deriv_real": self.x_deriv_real[start : end + 1],
            "deriv_cat": self.x_deriv_cat[start : end + 1],
        }
        row = self.labels.iloc[end]
        target = {
            "direction": int(row["label_direction"]),
            "return_fwd": float(row["label_return_fwd"]),
            "tradable": float(row["label_tradable"]),
            "best_net_ret": float(row["label_best_net_ret"]),
            "long_exec_ret": float(row["label_long_exec_ret"]),
            "short_exec_ret": float(row["label_short_exec_ret"]),
        }
        return x, target, self.df.iloc[end]["timestamp"]


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    x = {
        "market_real": torch.stack([torch.from_numpy(b[0]["market_real"]) for b in batch]),
        "market_cat": torch.stack([torch.from_numpy(b[0]["market_cat"]) for b in batch]),
        "deriv_real": torch.stack([torch.from_numpy(b[0]["deriv_real"]) for b in batch]),
        "deriv_cat": torch.stack([torch.from_numpy(b[0]["deriv_cat"]) for b in batch]),
    }
    keys = batch[0][1].keys()
    target = {
        k: torch.tensor(
            [b[1][k] for b in batch],
            dtype=torch.long if k == "direction" else torch.float32,
        )
        for k in keys
    }
    ts = [b[2] for b in batch]
    return x, target, ts


class BtcExecutableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.huber = nn.SmoothL1Loss(reduction="none", beta=0.002)

    def forward(self, out, target):
        p = out["rank_score"]
        edge = out["edge_score"]
        r = torch.nan_to_num(target["return_fwd"], nan=0.0)
        tradable = torch.clamp(torch.nan_to_num(target["tradable"], nan=0.0), 0.0, 1.0)
        best_net = torch.nan_to_num(target["best_net_ret"], nan=0.0)

        mag = torch.clamp(torch.maximum(torch.abs(r), torch.relu(best_net)), max=0.05)
        w = torch.clamp(1.0 + 8.0 * tradable + 200.0 * mag, max=30.0)

        l_dir = (self.ce(out["logits_dir"], target["direction"]) * w).sum() / (w.sum() + EPS)

        if len(p) > 1:
            idx = torch.randperm(len(p), device=p.device)
            sign = torch.sign(r - r[idx])
            valid = sign.abs() > 0
            pair = F.margin_ranking_loss(p, p[idx], sign, margin=0.05, reduction="none")
            pair_w = torch.maximum(w, w[idx])
            l_rank = (pair[valid] * pair_w[valid]).sum() / (pair_w[valid].sum() + EPS) if valid.any() else p.sum() * 0.0
        else:
            l_rank = p.sum() * 0.0

        nonflat = r.abs() > 1e-12
        l_sign = (F.softplus(-torch.sign(r[nonflat]) * p[nonflat]) * w[nonflat]).sum() / (w[nonflat].sum() + EPS) if nonflat.any() else p.sum() * 0.0
        l_flat = (p[~nonflat] ** 2).mean() if (~nonflat).any() else p.sum() * 0.0
        target_scaled = torch.clamp(r * 100.0, -1.0, 1.0)
        l_point = (self.huber(torch.tanh(p), target_scaled) * w).sum() / (w.sum() + EPS)
        edge_target = torch.clamp(best_net * 100.0, -1.0, 1.0)
        l_edge = (self.huber(torch.tanh(edge), edge_target) * w).sum() / (w.sum() + EPS)

        return l_dir + 8.0 * l_rank + 0.5 * l_sign + 0.2 * l_flat + 0.5 * l_point + 0.5 * l_edge


def validate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            if not batch:
                continue
            x, target, ts = batch
            x = {k: v.to(device) for k, v in x.items()}
            out = model(x)
            pred = out["rank_score"].detach().cpu().numpy()
            pred_dir = torch.argmax(out["logits_dir"], dim=1).detach().cpu().numpy()
            ret = target["return_fwd"].numpy()
            direction = target["direction"].numpy()
            for i in range(len(pred)):
                rows.append(
                    {
                        "p": float(pred[i]),
                        "r": float(ret[i]),
                        "pred_dir": int(pred_dir[i]),
                        "direction": int(direction[i]),
                        "ts": ts[i],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"ic": 0.0, "spread": 0.0, "dir_acc": 0.0}

    def safe_ic(g):
        if len(g) < 20 or g["p"].std() < 1e-12 or g["r"].std() < 1e-12:
            return np.nan
        return g["p"].corr(g["r"], method="spearman")

    # Single BTC stream has one row per timestamp, so use rolling/global IC.
    ic = df["p"].corr(df["r"], method="spearman") if df["p"].std() > 0 and df["r"].std() > 0 else 0.0
    n = max(1, int(len(df) * 0.1))
    s = df.sort_values("p")
    spread = float(s.tail(n)["r"].mean() - s.head(n)["r"].mean())
    dir_acc = float((df["pred_dir"] == df["direction"]).mean())
    logger.info(
        "[Val] IC=%.4f Spread=%.6f DirAcc=%.2f%% RetMean=%.6f NonZero=%.2f%%",
        ic,
        spread,
        dir_acc * 100.0,
        df["r"].mean(),
        (df["r"].abs() > 1e-12).mean() * 100.0,
    )
    return {"ic": float(ic), "spread": spread, "dir_acc": dir_acc}


def train(args):
    config = _load_config(args.config)
    seq_len = int(args.seq_len or config.get("parameters", {}).get("1min", {}).get("sequence_length", 60))
    hidden_dim = int(args.hidden_dim or config.get("parameters", {}).get("1min", {}).get("hidden_dim", 96))
    label_cfg = LabelConfig(
        entry_delay_steps=args.entry_delay_steps,
        holding_steps=args.holding_steps,
        min_edge=args.min_edge,
        cost_buffer=args.cost_buffer,
        require_bidask=args.require_bidask,
    )
    train_ds = BtcPerpDataset(args.feature_dir, config, seq_len, label_cfg, stage="train", split_date=args.split_date)
    val_ds = BtcPerpDataset(args.feature_dir, config, seq_len, label_cfg, stage="val", split_date=args.split_date)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train/val split produced empty dataset. Check --split-date.")

    device = torch.device(args.device)
    model = BtcPerpTFT(
        len(train_ds.market_real_cols),
        train_ds.market_cat_cards,
        len(train_ds.deriv_real_cols),
        train_ds.deriv_cat_cards,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optim, max_lr=args.lr * 5, total_steps=max(1, args.epochs * len(train_dl)), pct_start=0.1)
    crit = BtcExecutableLoss().to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf
    for ep in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_dl, desc=f"BTC-Ep {ep}"):
            if not batch:
                continue
            x, target, _ = batch
            x = {k: v.to(device) for k, v in x.items()}
            target = {k: v.to(device) for k, v in target.items()}
            optim.zero_grad()
            out = model(x)
            loss = crit(out, target)
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        metrics = validate(model, val_dl, device)
        score = metrics["ic"] + 10.0 * metrics["spread"]
        logger.info("Ep %d Loss=%.5f Score=%.5f", ep, float(np.mean(losses)) if losses else np.nan, score)
        state = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "config": config,
            "feature_cols": train_ds.feature_cols,
            "market_real_cols": train_ds.market_real_cols,
            "market_cat_cols": train_ds.market_cat_cols,
            "deriv_real_cols": train_ds.deriv_real_cols,
            "deriv_cat_cols": train_ds.deriv_cat_cols,
            "label_cfg": label_cfg.__dict__,
            "metrics": metrics,
            "score": score,
        }
        torch.save(state, args.checkpoint_dir / "btc_perp_tft_latest.pth")
        if score > best_score:
            best_score = score
            torch.save(state, args.checkpoint_dir / "btc_perp_tft_best.pth")
            logger.info("🌟 Best BTC model score=%.5f", best_score)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True, help="Directory containing features_1min.parquet and features_5min.parquet")
    parser.add_argument("--config", type=Path, default=Path("production/CONFIG/btc_slow_feature.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_btc_perp_tft"))
    parser.add_argument("--split-date", type=str, required=True, help="Validation starts from this UTC timestamp/date, e.g. 2026-03-01")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--entry-delay-steps", type=int, default=1)
    parser.add_argument("--holding-steps", type=int, default=5)
    parser.add_argument("--min-edge", type=float, default=0.0005)
    parser.add_argument("--cost-buffer", type=float, default=0.0002)
    parser.add_argument("--require-bidask", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
