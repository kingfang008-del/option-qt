#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: trading_tft_stock_embed.py
描述: [Advanced Alpha Net v11.0 - Stock/Option Dual Stream]
架构升级:
    1. [Dual Stream]: 拆分为 Stock Tower (股票流) 和 Option Tower (期权流)。
    2. [Dataset]: 同步拆分 x_stock 和 x_option 两个输入张量。
    3. [Logic]: 期权特征自动识别 (startswith 'options_')。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import os
import sys
import lmdb
import msgpack
import msgpack_numpy
import zstandard as zstd
import traceback
import shutil
import psycopg2
from config import PG_DB_URL
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Patch msgpack
msgpack_numpy.patch()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_advanced_alpha.log", mode='a')
    ]
)
logger = logging.getLogger("Advanced_Alpha")

# ==============================================================================
# 1. 基础组件 (保持不变)
# ==============================================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout: x = self.dropout(x)
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

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
        if input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim)
        else:
            self.res_proj = None

    def forward(self, x, context=None):
        residual = self.res_proj(x) if self.res_proj else x
        x_enc = self.fc1(x)
        if context is not None and self.context_dim is not None:
            context_enc = self.context_proj(context)
            if x_enc.dim() == 3 and context_enc.dim() == 2:
                context_enc = context_enc.unsqueeze(1)
            x_enc = x_enc + context_enc
        x_enc = self.elu1(x_enc)
        x_enc = self.fc2(x_enc)
        x_enc = self.gate(x_enc)
        return self.ln(x_enc + residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim, dropout=0.3, context_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = len(input_dims)
        
        self.single_variable_grns = nn.ModuleList()
        for dim in input_dims:
            self.single_variable_grns.append(
                GatedResidualNetwork(dim, hidden_dim, hidden_dim, dropout, context_dim=context_dim)
            )
            
        self.flattened_grn = GatedResidualNetwork(
            self.num_inputs * hidden_dim, hidden_dim, self.num_inputs, dropout, context_dim=context_dim
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding_list, context=None):
        if self.num_inputs == 0 or not embedding_list:
            return None, None

        processed_vars = []
        for i, emb in enumerate(embedding_list):
            processed = self.single_variable_grns[i](emb, context)
            processed_vars.append(processed)
            
        ref_tensor = processed_vars[0]
        is_temporal = (ref_tensor.dim() == 3) 
        
        if is_temporal:
            stacked = torch.stack(processed_vars, dim=2) 
            flattened = stacked.flatten(start_dim=2)
        else:
            stacked = torch.stack(processed_vars, dim=1) 
            flattened = stacked.flatten(start_dim=1)
        
        weights = self.softmax(self.flattened_grn(flattened, context))
        weights = weights.unsqueeze(-1)
        
        sum_dim = 2 if is_temporal else 1
        combined = (stacked * weights).sum(dim=sum_dim)
        
        return combined, weights

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.3):
        super().__init__()
        self.n_head = num_heads
        self.d_head = hidden_dim // num_heads
        self.qkv_linears = nn.Linear(hidden_dim, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x, mask=None):
        B, T, H = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_head, self.d_head).permute(0, 2, 3, 1)
        v = v.view(B, T, self.d_head).unsqueeze(1)

        attn_score = torch.matmul(q, k) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float('-inf'))
        
        attn_score = torch.nan_to_num(attn_score, nan=-1e9)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.attn_dropout(attn_prob)

        attn_vec = torch.matmul(attn_prob, v.repeat(1, self.n_head, 1, 1))
        attn_vec = attn_vec.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out_dropout(self.out_proj(attn_vec))

class GateAddNorm(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.glu = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, residual=None):
        if residual is None: residual = x
        x = self.glu(x)
        return self.ln(x + residual)

class NvidiaTFTWrapper(nn.Module):
    def __init__(self, hidden_dim, num_reals, num_cats, static_embedding_dim, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.has_inputs = (num_reals + num_cats) > 0
        
        if not self.has_inputs:
            return 

        self.real_projections = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_reals)])
        self.cat_embeddings = nn.ModuleList([nn.Embedding(51, hidden_dim) for _ in range(num_cats)])
        #self.real_bns = nn.ModuleList([nn.BatchNorm1d(30) for _ in range(num_reals)])
        
        input_dims = [hidden_dim] * (num_reals + num_cats)
        self.vsn = VariableSelectionNetwork(input_dims, hidden_dim, dropout, context_dim=hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.post_lstm_gate = GateAddNorm(hidden_dim, dropout)
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim)
        self.post_enrich_gate = GateAddNorm(hidden_dim, dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.post_attn_gate = GateAddNorm(hidden_dim, dropout)
        self.pos_wise_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_pos_gate = GateAddNorm(hidden_dim, dropout)

    def forward(self, x_reals, x_cats, c_s, c_h, c_c):
        B = c_s.shape[0]
        if not self.has_inputs:
            return torch.zeros(B, self.hidden_dim, device=c_s.device)

        if x_reals.shape[-1] > 0: T = x_reals.shape[1]
        elif x_cats.shape[-1] > 0: T = x_cats.shape[1]
        else: T = 30
            
        embeddings = []
        for i, proj in enumerate(self.real_projections):
            real_val = x_reals[..., i] 
            real_val = torch.nan_to_num(real_val, nan=0.0)
            #if T == 30: real_val = self.real_bns[i](real_val)
            # ✅ 直接做映射
            embeddings.append(proj(real_val.unsqueeze(-1)))
            
        for i, embed in enumerate(self.cat_embeddings):
            idx = x_cats[..., i].long()
            idx = torch.clamp(idx, 0, 50)
            embeddings.append(embed(idx))
            
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

# ==============================================================================
# 2. Advanced Alpha Net (Dual Stream Update)
# ==============================================================================
class AdvancedAlphaNet(nn.Module):
    def __init__(self, config, caps, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.config = config
        
        # [修改] 按照 Stock vs Option 分类特征
        self.feat_info = self._analyze_features(config['features'])
        
        # 静态特征嵌入
        self.static_stock_embed = nn.Embedding(caps['stock'] + 1, hidden_dim)
        self.static_sector_embed = nn.Embedding(caps['sector'] + 1, hidden_dim)
        self.static_dow_embed = nn.Embedding(8, hidden_dim)
        
        self.static_vsn = VariableSelectionNetwork([hidden_dim]*3, hidden_dim, dropout)
        self.grn_cs = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_ch = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.grn_cc = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # [修改] 初始化双流 Tower
        self.tft_stock = NvidiaTFTWrapper(
            hidden_dim, 
            len(self.feat_info['stock']['real']), 
            len(self.feat_info['stock']['cat']), 
            hidden_dim, dropout
        )
        
        self.tft_option = NvidiaTFTWrapper(
            hidden_dim, 
            len(self.feat_info['option']['real']), 
            len(self.feat_info['option']['cat']), 
            hidden_dim, dropout=0.45  # 给期权塔更高的 Dropout
        )
        
        # Fusion 层
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU(), nn.Dropout(dropout))
        self.head_dir = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 3))
        self.head_rank = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def _analyze_features(self, feature_config):
        # [修改] 重构特征分析逻辑，分离 Stock 和 Option
        info = {
            'stock': {'real': [], 'cat': []}, 
            'option': {'real': [], 'cat': []}
        }
        
        # 注意：这里我们不再需要 "effective_index" 这种全局索引
        # 因为 Dataset 会分别返回 x_stock 和 x_option 两个矩阵
        # 这里的索引应该是相对于各自矩阵的列索引
        
        idx_stock = 0
        idx_option = 0

        for f in feature_config:
            name = f['name']
            
            # 跳过静态特征
            if name in ['stock_id', 'sector_id', 'day_of_week']: 
                continue 
            
            ftype = 'cat' if f.get('type') == 'categorical' else 'real'
            
            # 核心判断逻辑：是否为期权特征
            is_option = name.startswith('options_')
            
            if is_option:
                info['option'][ftype].append(idx_option)
                idx_option += 1
            else:
                info['stock'][ftype].append(idx_stock)
                idx_stock += 1
                
        logger.info(f"Feature Split: Stock={idx_stock}, Option={idx_option}")
        return info

    def forward(self, x_stock, x_option, static_x):
        # x_stock: (B, T, F_stock)
        # x_option: (B, T, F_option)
        
        B = static_x['stock_id'].shape[0]
        
        s_id = torch.clamp(static_x['stock_id'].long(), 0, self.static_stock_embed.num_embeddings-1)
        sec_id = torch.clamp(static_x['sector_id'].long(), 0, self.static_sector_embed.num_embeddings-1)
        d_id = torch.clamp(static_x['day_of_week'].long(), 0, 7)
        
        s_stock = self.static_stock_embed(s_id)
        s_sector = self.static_sector_embed(sec_id)
        s_dow = self.static_dow_embed(d_id)
        
        static_emb, _ = self.static_vsn([s_stock, s_sector, s_dow]) 
        c_s = self.grn_cs(static_emb)
        c_h = self.grn_ch(static_emb).unsqueeze(0)
        c_c = self.grn_cc(static_emb).unsqueeze(0)

        # 1. Stock Tower
        # Dataset 已经把特征分好，直接取用即可
        # 需要注意的是 cat/real 的切分
        # info 里的索引是相对于 x_stock/x_option 的列索引
        x_stk_r = x_stock[..., self.feat_info['stock']['real']]
        x_stk_c = x_stock[..., self.feat_info['stock']['cat']]
        emb_stock = self.tft_stock(x_stk_r, x_stk_c, c_s, c_h, c_c)
        
        # 2. Option Tower
        x_opt_r = x_option[..., self.feat_info['option']['real']]
        x_opt_c = x_option[..., self.feat_info['option']['cat']]
        emb_option = self.tft_option(x_opt_r, x_opt_c, c_s, c_h, c_c)
        
        # 3. Fusion
        combined = torch.cat([emb_stock, emb_option], dim=-1)
        fused = self.fusion(combined)
        
        return {'logits_dir': self.head_dir(fused), 'rank_score': self.head_rank(fused)}

# ==============================================================================
# 3. UnifiedLMDBDataset (同步重构)
# ==============================================================================
class UnifiedLMDBDataset(Dataset):
    def __init__(self, db_path, config, stage='train', seq_len=30):
        self.db_path = db_path
        self.seq_len = seq_len
        self.dctx = zstd.ZstdDecompressor()
        
        # [修改] 预处理特征映射
        # 我们需要构建两个映射表：
        # 1. Stock Feats -> (LMDB Source Key, Column Index in x_stock)
        # 2. Option Feats -> (LMDB Source Key, Column Index in x_option)
        
        self.stock_map = []
        self.option_map = []
        
        idx_stock = 0
        idx_option = 0
        
        for f in config['features']:
            name = f['name']
            if name in ['stock_id', 'sector_id', 'day_of_week']: continue
            
            # LMDB 中的源数据 key ('1min' 或 '5min')
            # 即使我们按照 Stock/Option 拆分，底层数据依然存储在 1min/5min 结构里
            # 需要根据 config 中的 resolution 来定位
            #res_key =  '1min'
            res_key = f.get('resolution', '1min')
            
            is_option = name.startswith('options_')
            
            if is_option:
                self.option_map.append({
                    'name': name, 'source': res_key, 'target_idx': idx_option
                })
                idx_option += 1
            else:
                self.stock_map.append({
                    'name': name, 'source': res_key, 'target_idx': idx_stock
                })
                idx_stock += 1
                
        self.n_stock_feats = idx_stock
        self.n_option_feats = idx_option
            
        logger.info(f"Dataset Init: {db_path} | Stock Feats: {self.n_stock_feats}, Option Feats: {self.n_option_feats}")
        
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.keys = msgpack.unpackb(self.dctx.decompress(txn.get(b'__keys__')), raw=False)
        self.txn = None
        
        if len(self.keys) > 0:
            self._sanity_check()

    def _init_env(self):
        if self.txn is None:
            self.env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=True, meminit=False)
            self.txn = self.env.begin(write=False)

    def _sanity_check(self):
        self._init_env()
        check_count = min(100, len(self.keys))
        zero_ret_count = 0
        for i in range(check_count):
            val = self.txn.get(self.keys[i])
            if not val: continue
            data = msgpack.unpackb(self.dctx.decompress(val), raw=False)
            lbl = data.get('labels', {})
            r_val = lbl.get('label_return_fwd', lbl.get('return_fwd', 0.0))
            if abs(r_val) < 1e-9: zero_ret_count += 1
        if zero_ret_count == check_count:
            logger.warning(f"🚨 [WARNING] 前 {check_count} 条数据的 return_fwd 全为 0！")

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        self._init_env()
        val = self.txn.get(self.keys[idx])
        if not val: return None
        try: data = msgpack.unpackb(self.dctx.decompress(val), raw=False)
        except: return None
        
        # [修改] 构建两个独立的矩阵
        x_stock = np.zeros((self.seq_len, self.n_stock_feats), dtype=np.float32)
        x_option = np.zeros((self.seq_len, self.n_option_feats), dtype=np.float32)
        
        # 从 LMDB 数据包中提取源数据
        # data 结构: {'1min': {...}, '5min': {...}, ...}
        src_1m = data.get('1min', {})
        src_5m = data.get('5min', {})
        
        # 填充 Stock 矩阵
        for item in self.stock_map:
            name = item['name']
            src_key = item['source']
            tgt_idx = item['target_idx']
            
            # 根据 resolution 选择源字典
            source_dict = src_5m if src_key == '5min' else src_1m
            arr = source_dict.get(name)
            
            if arr is not None:
                v = np.array(arr, dtype=np.float32)
                # 处理长度不一致 (Upsampling logic)
                # 如果源是 5min (长度6) 但 seq_len 是 30，则重复填充
                if src_key == '5min':
                    target_len_5m = self.seq_len // 5
                    l = min(len(v), target_len_5m)
                    if l > 0:
                        up = np.repeat(v[-l:], 5) # Upsample 5x
                        x_stock[-len(up):, tgt_idx] = up
                else:
                    l = min(len(v), self.seq_len)
                    if l > 0:
                        x_stock[-l:, tgt_idx] = v[-l:]

        # 填充 Option 矩阵 (逻辑同上)
        for item in self.option_map:
            name = item['name']
            src_key = item['source']
            tgt_idx = item['target_idx']
            
            source_dict = src_5m if src_key == '5min' else src_1m
            arr = source_dict.get(name)
            
            if arr is not None:
                v = np.array(arr, dtype=np.float32)
                if src_key == '5min':
                    target_len_5m = self.seq_len // 5
                    l = min(len(v), target_len_5m)
                    if l > 0:
                        up = np.repeat(v[-l:], 5)
                        x_option[-len(up):, tgt_idx] = up
                else:
                    l = min(len(v), self.seq_len)
                    if l > 0:
                        x_option[-l:, tgt_idx] = v[-l:]
        
        # NaN 处理
        x_stock = np.nan_to_num(x_stock, nan=0.0, posinf=0.0, neginf=0.0)
        x_option = np.nan_to_num(x_option, nan=0.0, posinf=0.0, neginf=0.0)

        # 元数据处理 (不变)
        meta = data.get('metadata', {})
        ts = meta.get('timestamp', 0)
        try: dow = pd.to_datetime(ts, unit='ns').dayofweek
        except: dow = 0
            
        static = {
            'stock_id': int(meta.get('stock_id', 0)),
            'sector_id': int(meta.get('sector_id', 0)),
            'day_of_week': int(dow)
        }
        
        lbl = data.get('labels', {})
        r_val = lbl.get('label_return_fwd', lbl.get('return_fwd', 0.0))
        d_val = lbl.get('label_direction', lbl.get('direction', 1))
        if r_val is None or np.isnan(r_val): r_val = 0.0
        
        tgt = {'direction': int(d_val), 'return_fwd': float(r_val)}
        
        # 返回 (Stock, Option, Static, Target, Timestamp)
        return x_stock, x_option, static, tgt, ts

# [修改] Collate Fn 适配 Tuple 结构
def collate_fn(batch):
    batch = [b for b in batch if b]
    if not batch: return None
    
    # b: (x_stock, x_option, static, tgt, ts)
    x_stk = torch.stack([torch.from_numpy(b[0]) for b in batch])
    x_opt = torch.stack([torch.from_numpy(b[1]) for b in batch])
    
    st_k = batch[0][2].keys()
    s = {k: torch.tensor([b[2][k] for b in batch]) for k in st_k}
    
    t_k = batch[0][3].keys()
    t = {k: torch.tensor([b[3][k] for b in batch], dtype=torch.long if k=='direction' else torch.float32) for k in t_k}
    
    ts = [b[4] for b in batch]
    return x_stk, x_opt, s, t, ts

class StrategicAlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, out, target):
        ce = self.ce(out['logits_dir'], target['direction'])
        w = torch.abs(target['return_fwd']) * 10.0 + 1.0
        l_dir = (ce * torch.clamp(w, max=20.0)).mean()
        
        p = out['rank_score'].squeeze()
        r = target['return_fwd']
        r = torch.nan_to_num(r, nan=0.0)
        
        if len(p) > 1:
            idx = torch.randperm(len(p))
            target_sign = torch.sign(r - r[idx])
            l_rank = F.margin_ranking_loss(p, p[idx], target_sign, margin=0.1)
        else: l_rank = torch.tensor(0.0, device=p.device)
        return l_dir + 10.0 * l_rank, l_rank.item()

def validate(model, loader, device):
    model.eval()
    preds, rets, tss = [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="Val", leave=False):
            if not b: continue
            x_stk, x_opt, s, t, ts = b
            x_stk, x_opt, s = x_stk.to(device), x_opt.to(device), {k:v.to(device) for k,v in s.items()}
            
            # [修改] 传入两个流
            o = model(x_stk, x_opt, s)
            
            preds.extend(o['rank_score'].cpu().numpy().flatten())
            rets.extend(t['return_fwd'].numpy().flatten())
            tss.extend(ts)
            
    df = pd.DataFrame({'p': preds, 'r': rets, 't': tss})
    logger.info(f"[Val] Ret Mean={df['r'].mean():.6f}, Std={df['r'].std():.6f}")

    # [调试代码] 检查是否有全0的情况
    zero_ret_groups = df.groupby('t')['r'].std()
    zero_pred_groups = df.groupby('t')['p'].std()
    
    cnt_zero_ret = (zero_ret_groups < 1e-9).sum()
    cnt_zero_pred = (zero_pred_groups < 1e-9).sum()
    
    if cnt_zero_ret > 0:
        logger.warning(f"⚠️ 发现 {cnt_zero_ret} 个时间点的真实收益率(Return)方差为0（可能是全0数据）")
    if cnt_zero_pred > 0:
        logger.warning(f"⚠️ 发现 {cnt_zero_pred} 个时间点的模型预测(Pred)方差为0（模型可能坍塌）")
    
    # ==========================================================================
    # [核心修复] 定义一个安全的计算函数，检查方差是否为 0
    # ==========================================================================
    def safe_spearman(x):
        if len(x) <= 10: 
            return np.nan
        # 检查预测值 p 和 真实值 r 的标准差
        if x['p'].std() < 1e-9 or x['r'].std() < 1e-9:
            return 0.0  # 如果任意一方没有波动，相关性定义为 0 (或者 np.nan)
        return x['p'].corr(x['r'], method='spearman')
    
    # 使用修复后的函数
    ic_series = df.groupby('t')[['p','r']].apply(safe_spearman)
    # ==========================================================================

    ic_mean = ic_series.mean()
    
    spread = 0.0
    if len(df) > 50:
        df_s = df.sort_values('p')
        n = int(len(df)*0.1)
        spread = df_s.tail(n)['r'].mean() - df_s.head(n)['r'].mean()
        
    return {'ic': ic_mean, 'spread': spread}


def load_meta_info():
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute("SELECT id, sector_id FROM stocks_us")
        rows = c.fetchall()
        conn.close()
        max_sid = max([r[0] for r in rows]) if rows else 15000
        sec_set = set([r[1] for r in rows if r[1]])
        return {'max_stock_id': max_sid + 100, 'max_sector_id': len(sec_set) + 10}

    except Exception as e:
        logger.error(f"Error loading meta info from PG: {e}")
        return {'max_stock_id': 18000, 'max_sector_id': 200}

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    latest_path = checkpoint_dir / "advanced_alpha_latest.pth"
    if not latest_path.exists():
        return 0, -1.0
    
    try:
        logger.info(f"🔄 Resuming from {latest_path}")
        
        # [核心修改] 添加 weights_only=False 以兼容 PyTorch 2.6+
        # 允许加载包含 numpy 标量等复杂对象的 checkpoint
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['epoch'] + 1, checkpoint.get('best_ic', -1.0)
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        # 如果加载失败，可以选择抛出异常或重新开始，这里保持原逻辑返回 0
        return 0, -1.0
    
    
def save_checkpoint(state, is_best, checkpoint_dir):
    latest_path = checkpoint_dir / "advanced_alpha_latest.pth"
    torch.save(state, latest_path)
    if is_best:
        shutil.copyfile(latest_path, checkpoint_dir / "advanced_alpha_best.pth")
        logger.info(f"🌟 Best IC: {state['best_ic']:.4f}")

def fine_tune():
    """
    [New] 工业级微调专用模式 (Fine-Tuning Mode)
    核心升级:
        1. 灾难性遗忘保护：冻结底层 Embedding 和 LSTM，只微调 Attention 和 Head。
        2. 学习率重置：仅对放开的参数进行优化。
    """
    # ================= 1. 配置与路径 =================
    config_path = Path.home() / 'notebook/train/slow_feature.json'
    with open(config_path) as f: config = json.load(f)
    
    h5_root = Path('/mnt/s990/data/h5_unified_overlap_id')
    ckpt_dir = Path("./checkpoints_advanced_alpha")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda')
    meta = load_meta_info()
    caps = {'stock': meta['max_stock_id'], 'sector': meta['max_sector_id'], 'dow': 7}
    
    # ================= 2. 准备数据 (统一使用 quote_alpha) =================
    logger.info("🎨 [Fine-Tune] Loading Datasets...")
    
    # 🚨 [修复] 保持与 main() 数据集名称严格一致
    val_filenames = ['val_quote_alpha.lmdb'] 
    
    val_datasets = []
    for fname in val_filenames:
        full_path = h5_root / fname
        if full_path.exists():
            ds = UnifiedLMDBDataset(str(full_path), config)
            val_datasets.append(ds)
        else:
            logger.warning(f"⚠️ Validation file not found: {full_path}")
            
    if len(val_datasets) > 0:
        val_ds = ConcatDataset(val_datasets)
    else:
        raise FileNotFoundError("没有找到任何验证集文件！")
    
    # 🚨 [修复] 保持与 main() 数据集名称严格一致
    train_ds = UnifiedLMDBDataset(str(h5_root/'train_quote_alpha.lmdb'), config)
     
    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # ================= 3. 初始化模型并加载基础权重 =================
    model = AdvancedAlphaNet(config, caps).to(device)
    best_ckpt_path = ckpt_dir / "advanced_alpha_best.pth"
    
    if best_ckpt_path.exists():
        logger.info(f"🔄 Loading Base Weights for Fine-Tuning: {best_ckpt_path}")
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        best_ic = checkpoint.get('best_ic', -1.0)
        logger.info(f"✅ Base Weights Loaded. Previous Best IC: {best_ic:.4f}")
    else:
        raise FileNotFoundError("微调必须基于已有的预训练权重！找不到 advanced_alpha_best.pth")

    # =====================================================================
    # 🛡️ 4. [核心修复] 参数冻结隔离术 (Parameter Freezing)
    # =====================================================================
    logger.info("❄️ Freezing base layers to prevent Catastrophic Forgetting...")
    
    # 步骤 A：先残忍地把所有参数全部冻结
    for param in model.parameters():
        param.requires_grad = False
        
    # 步骤 B：精准放开我们需要微调的顶层 (Attention, 融合层, 输出头)
    unfrozen_keywords = [
        'attention',       # TFT 的多头注意力机制 (捕捉最新时序规律)
        'post_attn_gate',  # Attention 后的门控
        'pos_wise_grn',    # Attention 后的前馈网络
        'post_pos_gate',   # 最后的残差门
        'fusion',          # 双塔融合层
        'head_dir',        # 方向预测头
        'head_rank'        # 排名预测头
    ]
    
    for name, param in model.named_parameters():
        if any(kw in name for kw in unfrozen_keywords):
            param.requires_grad = True
            
    # 统计参数量，确保冻结成功
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model Parameters: Trainable {trainable_params:,} / Total {total_params:,} ({(trainable_params/total_params)*100:.1f}%)")

    # ================= 5. 优化器与调度器设置 =================
    FT_EPOCHS = 10         
    FT_LR = 1e-5           
    
    # 🚨 [关键修复] 只把 required_grad=True 的参数传给优化器！否则会报错。
    trainable_params_list = filter(lambda p: p.requires_grad, model.parameters())
    optim = AdamW(trainable_params_list, lr=FT_LR, weight_decay=1e-3)
    
    crit = StrategicAlphaLoss().to(device)
    scheduler = OneCycleLR(
        optim, 
        max_lr=FT_LR * 5, 
        total_steps=FT_EPOCHS * len(train_dl), 
        pct_start=0.2,  # 微调不需要太长的 warmup，提早进入退火阶段
        div_factor=25
    )
    
    logger.info(f"🚀 Start Fine-Tuning for {FT_EPOCHS} Epochs...")
    
    # ================= 6. 微调训练循环 =================
    # 重置 best_ic，因为在新的微调验证集上，IC 基准不同
    current_ft_best_ic = -1.0 
    
    for ep in range(FT_EPOCHS):
        model.train()
        logs = []
        pbar = tqdm(train_dl, desc=f"FT-Ep {ep}")
        
        for b in pbar:
            if not b: continue
            
            x_stk, x_opt, s, t, _ = b
            x_stk, x_opt = x_stk.to(device), x_opt.to(device)
            s = {k:v.to(device) for k,v in s.items()}
            t = {k:v.to(device) for k,v in t.items()}
            
            optim.zero_grad()
            out = model(x_stk, x_opt, s)
            loss, lr_val = crit(out, t)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            
            logs.append(loss.item())
            pbar.set_postfix({'L': f"{loss.item():.4f}", 'lr': f"{lr_val:.2e}"})
            
        metrics = validate(model, val_dl, device)
        ic = metrics['ic']
        logger.info(f"FT-Ep {ep}: Loss={np.mean(logs):.4f}, IC={ic:.4f}, Spr={metrics['spread']:.6f}")
        
        is_best = ic > current_ft_best_ic
        if is_best: 
            current_ft_best_ic = ic
            logger.info("🔥 New Best IC during Fine-Tuning!")
            
        ft_state = {
            'epoch': ep, 
            'state_dict': model.state_dict(), 
            'best_ic': current_ft_best_ic, 
            'config': config
        }
        
        torch.save(ft_state, ckpt_dir / "advanced_alpha_ft_latest.pth")
        if is_best:
            torch.save(ft_state, ckpt_dir / "advanced_alpha_ft_best.pth")
            
    logger.info(f"✅ Fine-Tuning Complete. Best FT IC: {current_ft_best_ic:.4f}")

def main():
    config_path = Path.home() / 'notebook/train/slow_feature.json'
    with open(config_path) as f: config = json.load(f)
    h5 = Path('/mnt/s990/data/h5_unified_overlap_id')
    ckpt_dir = Path("./checkpoints_advanced_alpha")
    ckpt_dir.mkdir(exist_ok=True)
    
    meta = load_meta_info()
    caps = {'stock': meta['max_stock_id'], 'sector': meta['max_sector_id'], 'dow': 7}
    
    device = torch.device('cuda')
    # 2. [修改] 加载并合并多个验证集
    # 定义你的验证集文件名列表
    val_filenames = [
        # 'val_slow_channel_alpha.lmdb',
        'val_quote_alpha.lmdb' 
        # 你可以在这里添加更多
    ]
    
    val_datasets = []
    for fname in val_filenames:
        full_path = h5 / fname
        if full_path.exists():
            logger.info(f"Adding validation set: {fname}")
            ds = UnifiedLMDBDataset(str(full_path), config)
            val_datasets.append(ds)
        else:
            logger.warning(f"⚠️ Validation file not found: {full_path}")
            
    if len(val_datasets) > 0:
        # 使用 ConcatDataset 将它们“虚拟”拼接在一起
        val_ds = ConcatDataset(val_datasets)
        logger.info(f"Total Validation Samples: {len(val_ds)}")
    else:
        raise FileNotFoundError("没有找到任何验证集文件！")
    
    # [注意] UnifiedLMDBDataset 初始化时会自动区分 Stock/Option
    train_dl = DataLoader(UnifiedLMDBDataset(str(h5/'train_quote_alpha.lmdb'), config), batch_size=1024, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = AdvancedAlphaNet(config, caps).to(device)
    optim = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    crit = StrategicAlphaLoss().to(device)
    
    scheduler = OneCycleLR(optim, max_lr=5e-4, total_steps=20*len(train_dl), pct_start=0.1, div_factor=25)    
    start_ep, best_ic = load_checkpoint(model, optim, scheduler, ckpt_dir)
    
    logger.info("🚀 Training...")
    for ep in range(start_ep, 20):
        model.train()
        logs = []
        pbar = tqdm(train_dl, desc=f"Ep {ep}")
        for b in pbar:
            if not b: continue
            # [修改] 解包 tuple
            x_stk, x_opt, s, t, _ = b
            x_stk, x_opt, s, t = x_stk.to(device), x_opt.to(device), {k:v.to(device) for k,v in s.items()}, {k:v.to(device) for k,v in t.items()}
            
            optim.zero_grad()
            # [修改] 传入两个输入流
            out = model(x_stk, x_opt, s)
            loss, lr_val = crit(out, t)
            
            if torch.isnan(loss):
                logger.error("NaN Loss Detected! Skipping Batch.")
                continue
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            logs.append(loss.item())
            pbar.set_postfix({'L': f"{loss.item():.4f}", 'R': f"{lr_val:.4f}"})
            
        metrics = validate(model, val_dl, device)
        ic = metrics['ic']
        logger.info(f"Ep {ep}: Loss={np.mean(logs):.4f}, IC={ic:.4f}, Spr={metrics['spread']:.6f}")
        
        is_best = ic > best_ic
        if is_best: best_ic = ic
        save_checkpoint({
            'epoch': ep, 'state_dict': model.state_dict(), 'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(), 'best_ic': best_ic, 'config': config
        }, is_best, ckpt_dir)

if __name__ == "__main__":
    try: main()
    except Exception as e: traceback.print_exc()