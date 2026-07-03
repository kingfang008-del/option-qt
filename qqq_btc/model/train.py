#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一训练入口 —— 替代原文件 main()/fine_tune() 两套几乎重复的循环。

设计修正(相对 trading_tft_stock_embed.py):
  - 单一训练循环,--mode {pretrain, finetune} 只切换"权重加载 + 冻结策略",
    不再复制粘贴数据加载/循环/存档逻辑
  - 无 Postgres 依赖:embedding 容量从 config 读(resolve_embedding_caps)
  - 无模块级副作用:日志/随机种子在 main 里初始化
  - 数据集 strict:训练前强制 sanity_check,net 标签缺失/零方差直接失败

用法(训练机):
  # P1 共训(SPY+QQQ)
  python -m qqq_btc.model.train --mode pretrain \
      --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
      --data-root ~/train_data/lmdb --train-lmdb train_qqq_spy.lmdb \
      --val-lmdbs val_qqq.lmdb --checkpoint-dir checkpoints_index_etf_v2

  # P2 QQQ 校准(冻结双塔)
  python -m qqq_btc.model.train --mode finetune \
      --init-checkpoint checkpoints_index_etf_v2/best.pth \
      --train-lmdb train_qqq_only.lmdb --checkpoint-dir checkpoints_qqq_net_edge_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader

from .backbone import (
    DualStreamAlphaNet,
    freeze_for_finetune,
    load_pretrain_checkpoint,
    resolve_embedding_caps,
)
from .dataset import LMDBAlphaDataset, collate_fn
from .losses import NetEdgeLoss

logger = logging.getLogger("qqq_btc.train")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    """per-symbol IC + top 分位净收益 + q10 校准率。"""
    model.eval()
    preds, q10s, rets, sids = [], [], [], []
    for b in loader:
        if not b:
            continue
        x_stk, x_opt, s, t, _ = b
        out = model(
            x_stk.to(device), x_opt.to(device), {k: v.to(device) for k, v in s.items()}
        )
        preds.extend(out["net_edge"].cpu().numpy().flatten())
        q10s.extend(out["net_edge_q10"].cpu().numpy().flatten())
        rets.extend(t["return_fwd"].numpy().flatten())
        sids.extend(s["stock_id"].numpy().flatten())

    df = pd.DataFrame({"p": preds, "q10": q10s, "r": rets, "sid": sids})
    if len(df) < 50 or df["p"].std() < 1e-9 or df["r"].std() < 1e-9:
        logger.warning("验证样本不足或预测/标签方差为 0(可能输出坍塌)。")
        return {"ic": 0.0, "top_mean": 0.0, "top_hit": 0.0, "q10_coverage": 0.0}

    per_symbol_ic = (
        df.groupby("sid")[["p", "r"]]
        .apply(lambda x: x["p"].corr(x["r"], method="spearman") if len(x) > 20 else np.nan)
        .dropna()
    )
    ic = float(per_symbol_ic.mean()) if len(per_symbol_ic) else float(
        df["p"].corr(df["r"], method="spearman")
    )

    n_top = max(10, int(len(df) * 0.05))
    top = df.nlargest(n_top, "p")
    # q10 校准:真实收益低于预测 q10 的比例应接近 10%
    q10_coverage = float((df["r"] < df["q10"]).mean())

    metrics = {
        "ic": 0.0 if np.isnan(ic) else ic,
        "top_mean": float(top["r"].mean()),
        "top_hit": float((top["r"] > 0).mean()),
        "q10_coverage": q10_coverage,
    }
    logger.info(
        "[Val] IC=%.4f Top5Mean=%.6f Top5Hit=%.2f%% q10_cov=%.3f",
        metrics["ic"], metrics["top_mean"], metrics["top_hit"] * 100, q10_coverage,
    )
    return metrics


def build_loaders(args, config):
    train_ds = LMDBAlphaDataset(str(Path(args.data_root) / args.train_lmdb), config)
    report = train_ds.sanity_check()
    logger.info("Train label check: %s", report)

    val_sets = []
    for fname in [x.strip() for x in args.val_lmdbs.split(",") if x.strip()]:
        p = Path(args.data_root) / fname
        if p.exists():
            ds = LMDBAlphaDataset(str(p), config)
            ds.sanity_check()
            val_sets.append(ds)
        else:
            logger.warning("验证集不存在: %s", p)
    if not val_sets:
        raise FileNotFoundError("没有可用的验证 LMDB。")

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=pin,
    )
    val_dl = DataLoader(
        ConcatDataset(val_sets), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=max(1, args.num_workers // 2), pin_memory=pin,
    )
    return train_dl, val_dl


def run(args) -> None:
    set_seed(args.seed)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = build_loaders(args, config)

    caps = resolve_embedding_caps(config)
    model = DualStreamAlphaNet(config, caps).to(device)

    if args.mode == "finetune":
        if not args.init_checkpoint:
            raise ValueError("--mode finetune 需要 --init-checkpoint(共训权重)。")
        load_pretrain_checkpoint(model, args.init_checkpoint, device=str(device))
        trainable, total = freeze_for_finetune(model)
        logger.info("Finetune 冻结: trainable %s / total %s (%.1f%%)",
                    f"{trainable:,}", f"{total:,}", trainable / total * 100)
    elif args.init_checkpoint:
        load_pretrain_checkpoint(model, args.init_checkpoint, device=str(device))

    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=args.lr, weight_decay=1e-3)
    crit = NetEdgeLoss(config).to(device)
    scheduler = OneCycleLR(
        optim,
        max_lr=args.max_lr,
        total_steps=max(1, args.epochs * len(train_dl)),
        pct_start=0.2 if args.mode == "finetune" else 0.1,
        div_factor=25,
    )

    best_ic = -1.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for b in train_dl:
            if not b:
                continue
            x_stk, x_opt, s, t, _ = b
            x_stk, x_opt = x_stk.to(device), x_opt.to(device)
            s = {k: v.to(device) for k, v in s.items()}
            t = {k: v.to(device) for k, v in t.items()}

            optim.zero_grad()
            loss, _ = crit(model(x_stk, x_opt, s), t)
            if torch.isnan(loss):
                logger.error("NaN loss, batch skipped.")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            scheduler.step()
            losses.append(loss.item())

        metrics = validate(model, val_dl, device)
        logger.info("Ep %d: loss=%.4f ic=%.4f", ep, float(np.mean(losses) if losses else 0.0), metrics["ic"])

        state = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
            "best_ic": max(best_ic, metrics["ic"]),
            "config": config,
            "mode": args.mode,
        }
        torch.save(state, ckpt_dir / "latest.pth")
        if metrics["ic"] > best_ic:
            best_ic = metrics["ic"]
            shutil.copyfile(ckpt_dir / "latest.pth", ckpt_dir / "best.pth")
            logger.info("New best IC: %.4f", best_ic)


def parse_args():
    parser = argparse.ArgumentParser(description="qqq_btc v2 训练入口(pretrain/finetune 统一循环)")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--config", default="qqq_btc/CONFIG/slow_feature_qqq_v2.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-lmdb", required=True)
    parser.add_argument("--val-lmdbs", required=True, help="逗号分隔的验证 LMDB 文件名")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None, help="共训权重(finetune 必填)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
