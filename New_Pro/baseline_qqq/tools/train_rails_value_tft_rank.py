#!/usr/bin/env python3
"""
TFT + listwise 排序损失:直接优化日内尾部选择。

动机(step5/10):BCE 把全分布 IC 抬到 ~0.18,但 top2% 选中 bar 的真实
rails_value 经常为负(「中位排序对、极端头部错」)。ListNet / top-weighted
listwise 在「每个交易日」内对分数做 softmax 交叉熵,把梯度集中到头部。

协议与扩训基线对齐:
  train=2022-2024 / val=2025H1 / test=2025-08~2026-03
  同一 TFTEncoder + 54 特征缓存;早停看 val 的 top2% sel_rails_mean
  (比 IC 更贴近赚钱指标)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from train_rails_value_tft import (  # noqa: E402
    SEQ_LEN,
    _round4,
    build_sequences,
    cache_features,
    daily_ic,
    eval_segment,
    load_cached_days,
    make_model,
)


def listnet_loss(
    scores: torch.Tensor,
    rails: torch.Tensor,
    *,
    temperature: float = 0.1,
    top_weight_power: float = 0.0,
) -> torch.Tensor:
    """
    ListNet: KL(softmax(rails/τ) || softmax(scores/τ)).

    top_weight_power>0 时,用真实排序位次对目标分布再加权
    (位次越高权重越大),把梯度压向头部。
    """
    if scores.numel() < 8:
        return scores.new_zeros(())
    # 目标:rails 越高概率越大;用 softplus 避免负值主导
    target_logits = F.softplus(rails) / max(temperature, 1e-4)
    if top_weight_power > 0:
        # 日内百分位增益:rank_pct^power
        n = rails.numel()
        order = torch.argsort(rails)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(n, device=rails.device, dtype=torch.float32)
        pct = ranks / max(n - 1, 1)
        w = pct.pow(top_weight_power)
        # 加权后重新归一成目标分布
        log_p = F.log_softmax(target_logits, dim=0)
        # 用加权 CE: -sum (w_norm * log_softmax(scores))
        w = w / (w.sum() + 1e-8)
        log_q = F.log_softmax(scores / max(temperature, 1e-4), dim=0)
        return -(w * log_q).sum()
    p = F.softmax(target_logits, dim=0)
    log_q = F.log_softmax(scores / max(temperature, 1e-4), dim=0)
    return -(p * log_q).sum()


def approx_ndcg_loss(
    scores: torch.Tensor,
    rails: torch.Tensor,
    *,
    temperature: float = 0.1,
    k: int = 8,
) -> torch.Tensor:
    """
    可微 ApproxNDCG@k 的负值(作损失)。
    增益 = softplus(rails);折扣 = 1/log2(1+rank);rank 用 soft rank 近似。
    """
    n = scores.numel()
    if n < 4:
        return scores.new_zeros(())
    gains = F.softplus(rails)
    # soft rank_i ≈ (# of j with s_j > s_i) + 0.5 = sum_j sigmoid((s_j-s_i)/τ)
    diff = (scores.unsqueeze(0) - scores.unsqueeze(1)) / max(temperature, 1e-4)
    soft_rank = torch.sigmoid(diff).sum(dim=0)  # shape [n], 约 1-based 软排名
    discounts = 1.0 / torch.log2(soft_rank + 1.0)
    # 只强调前 k:用 soft top-k 掩码
    # 简化:全列表 NDCG,但对增益做 top 强调——真实 top-k 增益放大
    order = torch.argsort(rails, descending=True)
    top_mask = torch.zeros_like(gains)
    top_mask[order[: min(k, n)]] = 1.0
    gains = gains * (1.0 + 3.0 * top_mask)  # 头部增益 ×4
    dcg = (gains * discounts).sum()
    # ideal
    ideal_gains, _ = torch.sort(gains, descending=True)
    ideal_disc = 1.0 / torch.log2(
        torch.arange(n, device=scores.device, dtype=torch.float32) + 2.0
    )
    idcg = (ideal_gains * ideal_disc).sum().clamp_min(1e-6)
    ndcg = dcg / idcg
    return 1.0 - ndcg


def day_index_map(day_idx: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for d in np.unique(day_idx):
        out[int(d)] = np.where(day_idx == d)[0]
    return out


def selection_metric(
    pred: np.ndarray,
    rails: np.ndarray,
    day_idx: np.ndarray,
    top_pct: float = 0.02,
) -> Tuple[float, float]:
    """返回 (sel_rails_mean, oracle_top10_hit)。用日内全知分位近似 top-k(val 早停用)。"""
    sel_vals: List[float] = []
    hits: List[float] = []
    for d in np.unique(day_idx):
        s = day_idx == d
        if s.sum() < 30:
            continue
        p, r = pred[s], rails[s]
        ok = np.isfinite(p) & np.isfinite(r)
        if ok.sum() < 30:
            continue
        p, r = p[ok], r[ok]
        n_pick = max(1, int(round(len(p) * top_pct)))
        top = np.argsort(p)[-n_pick:]
        thr = np.quantile(r, 0.90)
        sel_vals.extend(r[top].tolist())
        hits.extend((r[top] >= thr).astype(float).tolist())
    if not sel_vals:
        return float("nan"), float("nan")
    return float(np.mean(sel_vals)), float(np.mean(hits))


def predict_all(model, X: np.ndarray, device, batch: int = 4096):
    preds, vetos = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            lt, lv = model(xb)
            preds.append(lt.cpu().numpy())  # logits, 排序用
            vetos.append(torch.sigmoid(lv).cpu().numpy())
    return np.concatenate(preds), np.concatenate(vetos)


def main() -> int:
    ap = argparse.ArgumentParser(description="TFT listwise ranking training")
    ap.add_argument("--train-globs", default="QQQ_2022-*.parquet,QQQ_2023-*.parquet,QQQ_2024-*.parquet")
    ap.add_argument(
        "--val-globs",
        default="QQQ_2025-01-*.parquet,QQQ_2025-02-*.parquet,QQQ_2025-03-*.parquet,QQQ_2025-04-*.parquet,QQQ_2025-05-*.parquet,QQQ_2025-06-*.parquet",
    )
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet", "QQQ_2026-03-*.parquet"]
        ),
    )
    ap.add_argument("--loss", choices=["listnet", "listnet_top", "approx_ndcg"], default="listnet_top")
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--top-weight-power", type=float, default=2.0, help="listnet_top 用")
    ap.add_argument("--ndcg-k", type=int, default=8)
    ap.add_argument("--veto-weight", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--days-per-step", type=int, default=8, help="每步打包多少个交易日")
    ap.add_argument(
        "--ckpt",
        default="/mnt/s990/data/cache/rails_value_tft_rank_2022_2024.pt",
    )
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_rails_value_tft_rank_2022_2024.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = cache_features()

    train_days = load_cached_days(args.train_globs.split(","))
    val_days = load_cached_days(args.val_globs.split(","))
    print(f"train days={len(train_days)} val days={len(val_days)} feats={len(features)}")

    Xtr, ytt, ytv, rtr, dtr, _btr, norm = build_sequences(train_days, features)
    Xva, yvt, yvv, rva, dva, _bva, _ = build_sequences(val_days, features, norm)
    print(f"seqs train={len(Xtr)} val={len(Xva)}")

    day_map = day_index_map(dtr)
    day_ids = list(day_map.keys())
    print(f"unique train days in seqs={len(day_ids)}")

    model = make_model(len(features), args.hidden, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,} loss={args.loss} device={device}")

    pos_w_veto = torch.tensor((1.0 - ytv.mean()) / max(ytv.mean(), 1e-6), device=device)
    bce_veto = nn.BCEWithLogitsLoss(pos_weight=pos_w_veto)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr_t = torch.from_numpy(Xtr)
    rtr_t = torch.from_numpy(rtr.astype(np.float32))
    ytv_t = torch.from_numpy(ytv)

    best_score, best_state, patience = -1e18, None, 0
    best_meta = {}
    rng = np.random.default_rng(42)

    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(day_ids)
        tot_loss, n_steps = 0.0, 0
        for i in range(0, len(day_ids), args.days_per_step):
            batch_days = day_ids[i : i + args.days_per_step]
            idx = np.concatenate([day_map[d] for d in batch_days])
            # 子采样过大日,控制显存/方差
            if len(idx) > 4096:
                idx = rng.choice(idx, size=4096, replace=False)

            xb = Xtr_t[idx].to(device, non_blocking=True)
            rb = rtr_t[idx].to(device, non_blocking=True)
            vb = ytv_t[idx].to(device, non_blocking=True)
            lt, lv = model(xb)

            # 按日拆开算 listwise,再平均
            loss_rank = xb.new_zeros(())
            n_lists = 0
            # 映射回原始 day
            local_days = dtr[idx]
            for d in np.unique(local_days):
                m = local_days == d
                if m.sum() < 16:
                    continue
                if args.loss == "listnet":
                    loss_rank = loss_rank + listnet_loss(
                        lt[m], rb[m], temperature=args.temperature
                    )
                elif args.loss == "listnet_top":
                    loss_rank = loss_rank + listnet_loss(
                        lt[m],
                        rb[m],
                        temperature=args.temperature,
                        top_weight_power=args.top_weight_power,
                    )
                else:
                    loss_rank = loss_rank + approx_ndcg_loss(
                        lt[m],
                        rb[m],
                        temperature=args.temperature,
                        k=args.ndcg_k,
                    )
                n_lists += 1
            if n_lists == 0:
                continue
            loss_rank = loss_rank / n_lists
            loss = loss_rank + args.veto_weight * bce_veto(lv, vb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
            n_steps += 1
        sched.step()

        # --- val ---
        pva_logits, _ = predict_all(model, Xva, device)
        ics = daily_ic(pva_logits, rva, dva)
        ic = float(np.mean(ics)) if ics else -1e9
        sel_mean, hit10 = selection_metric(pva_logits, rva, dva, top_pct=0.02)
        # 早停分数:优先选择质量,IC 作微弱加成
        score = (sel_mean if np.isfinite(sel_mean) else -1.0) + 0.01 * ic
        marker = ""
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {"val_ic": ic, "sel_rails_mean": sel_mean, "hit10": hit10, "epoch": ep}
            patience = 0
            marker = " *"
        else:
            patience += 1
        print(
            f"ep{ep:02d} loss={tot_loss / max(n_steps, 1):.4f} "
            f"val_IC={ic:+.4f} sel2%={sel_mean:+.4f} hit10={hit10:.1%}{marker}"
        )
        if patience >= args.patience:
            print("early stop")
            break

    model.load_state_dict(best_state)
    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "norm_mu": norm[0],
            "norm_sd": norm[1],
            "features": features,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "loss": args.loss,
            "val_ic": best_meta.get("val_ic"),
            "sel_rails_mean": best_meta.get("sel_rails_mean"),
            "hit10": best_meta.get("hit10"),
        },
        ckpt,
    )
    print(f"best {best_meta} saved -> {ckpt}")

    # --- 前向评估(复用 eval_segment;信号用 sigmoid 概率) ---
    model.eval()
    segments = []
    for seg in args.test_globs.split(";"):
        days = load_cached_days(seg.split(","))
        if not days:
            print(f"[{seg}] empty, skip")
            continue
        segments.append(eval_segment(seg, days, model, norm, features, device))

    result = {
        "meta": {
            "train_globs": args.train_globs,
            "val_globs": args.val_globs,
            "test_globs": args.test_globs,
            "loss": args.loss,
            "temperature": args.temperature,
            "top_weight_power": args.top_weight_power,
            "ndcg_k": args.ndcg_k,
            "veto_weight": args.veto_weight,
            "n_params": n_params,
            "best": {k: (_round4(v) if isinstance(v, float) else v) for k, v in best_meta.items()},
            "baseline_expanded_bce": {
                "val_ic": 0.1722,
                "forward_ic_mean": 0.181,
                "note": "step10 BCE 扩训",
            },
        },
        "test_segments": segments,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
