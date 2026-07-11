#!/usr/bin/env python3
"""
TFT 改造验证:rails_value top-k 标签 + veto 头(接续 LGBM v1/v2 实验)。

设计(全部来自前三步实验结论):
  - 标签   : binary「rails_value 属于当日 top-10%」(排序/分类优于回归)
  - veto 头: binary「入场后 15bar 最深回撤 <= -0.12」(early_stop_roi 口径)
  - 特征   : v2 的 54 个因果特征(现货微结构 + 跨bucket结构 + 报价强度)
  - 底座   : qqq_btc.model.backbone.TFTEncoder(VSN→LSTM→因果注意力),
             序列长度 30,单塔(特征已含双源信息)
  - 验收   : 测试月逐日 rank IC(P_top vs rails_value)≥0.20;
             top2% 选中 bar 真实 rails_value 均值 > +0.02;replay 转正

用法:
  # 1) 构建/更新缓存(逐日分钟帧:特征+标签,存数据盘)
  python train_rails_value_tft.py build-cache --globs "QQQ_2025-*.parquet,QQQ_2026-01-*.parquet,QQQ_2026-02-*.parquet"
  # 2) 训练 + walk-forward 评估
  python train_rails_value_tft.py train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

CACHE_DIR = Path("/mnt/s990/data/cache/rails_value_v2_minute")
RAW_DIR = Path("/mnt/s990/data/raw_1s/dte1_options")
SEQ_LEN = 30
LABEL_TOP = 0.10
VETO_DD = -0.12


def _round4(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# 缓存构建(依赖 v2 装载器,只跑一次)
# ---------------------------------------------------------------------------
def cmd_build_cache(args) -> int:
    import rails_value_lgbm_v2 as v2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    globs = args.globs.split(",")
    days = v2.load_month_days(RAW_DIR, "QQQ", globs, 2)
    meta = {"features": v2.FEATURES_V2, "seq_len": SEQ_LEN}
    (CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    keep_extra = [
        "timestamp", "session_bar", "rails_value", "veto_dd15", "oracle_edge",
        "exec_call_bid", "exec_call_ask", "exec_call_mid", "exec_call_spread_pct",
    ]
    n = 0
    for date_str, minute, _t in days:
        cols = list(dict.fromkeys(c for c in keep_extra + v2.FEATURES_V2 if c in minute.columns))
        minute[cols].to_parquet(CACHE_DIR / f"QQQ_{date_str}.parquet", index=False)
        n += 1
    print(f"cached {n} days -> {CACHE_DIR}")
    return 0


def load_cached_days(globs: Sequence[str]) -> List[Tuple[str, pd.DataFrame]]:
    out = []
    for g in globs:
        for fp in sorted(CACHE_DIR.glob(g)):
            date_str = fp.stem.split("_", 1)[-1]
            out.append((date_str, pd.read_parquet(fp)))
    return out


def cache_features() -> List[str]:
    return json.loads((CACHE_DIR / "meta.json").read_text())["features"]


# ---------------------------------------------------------------------------
# 数据集:滑窗序列
# ---------------------------------------------------------------------------
def build_sequences(
    days: List[Tuple[str, pd.DataFrame]],
    features: List[str],
    norm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """
    返回 X[N,L,F] float32, y_top[N], y_veto[N], rails[N], day_idx[N], bar_idx[N]。
    仅取入场窗内、标签有效、t>=L-1 的 bar。norm=None 时同时计算训练统计。
    """
    from rails_value_lgbm import ENTRY_END, ENTRY_START

    feats_all, rows = [], []
    for di, (_date, m) in enumerate(days):
        F = m[features].to_numpy(dtype=np.float32)
        feats_all.append(F)
        sb = m["session_bar"].to_numpy()
        rv = m["rails_value"].to_numpy()
        vd = m["veto_dd15"].to_numpy()
        ok = (
            (sb >= ENTRY_START) & (sb <= ENTRY_END)
            & np.isfinite(rv) & np.isfinite(vd)
            & (np.arange(len(m)) >= SEQ_LEN - 1)
        )
        ranks = pd.Series(np.where(ok, rv, np.nan)).rank(pct=True)
        for t in np.where(ok)[0]:
            rows.append((di, t, rv[t], float(ranks.iloc[t] >= 1.0 - LABEL_TOP), float(vd[t] <= VETO_DD)))

    if norm is None:
        cat = np.concatenate(feats_all)
        mu = np.nanmean(cat, axis=0)
        sd = np.nanstd(cat, axis=0)
        sd[sd < 1e-8] = 1.0
        norm = (mu.astype(np.float32), sd.astype(np.float32))
    mu, sd = norm

    N = len(rows)
    Fn = len(features)
    X = np.zeros((N, SEQ_LEN, Fn), dtype=np.float32)
    y_top = np.zeros(N, dtype=np.float32)
    y_veto = np.zeros(N, dtype=np.float32)
    rails = np.zeros(N, dtype=np.float32)
    day_idx = np.zeros(N, dtype=np.int64)
    bar_idx = np.zeros(N, dtype=np.int64)
    for i, (di, t, rv, yt, yv) in enumerate(rows):
        seq = feats_all[di][t - SEQ_LEN + 1 : t + 1]
        z = (seq - mu) / sd
        X[i] = np.clip(np.nan_to_num(z, nan=0.0), -10.0, 10.0)
        y_top[i], y_veto[i], rails[i] = yt, yv, rv
        day_idx[i], bar_idx[i] = di, t
    return X, y_top, y_veto, rails, day_idx, bar_idx, norm


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def make_model(num_reals: int, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    from qqq_btc.model.backbone import TFTEncoder

    class RailsValueTFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TFTEncoder(hidden, num_reals, 0, dropout=dropout)
            self.static = nn.Parameter(torch.zeros(hidden))
            self.head_top = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
            )
            self.head_veto = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
            )

        def forward(self, x):
            B = x.shape[0]
            c_s = self.static.unsqueeze(0).expand(B, -1)
            c_h = c_s.unsqueeze(0).contiguous()
            c_c = torch.zeros_like(c_h)
            emb = self.encoder(x, x[..., :0], c_s, c_h, c_c)
            return self.head_top(emb).squeeze(-1), self.head_veto(emb).squeeze(-1)

    return RailsValueTFT()


def daily_ic(pred: np.ndarray, rails: np.ndarray, day_idx: np.ndarray) -> List[float]:
    from scipy.stats import spearmanr

    out = []
    for d in np.unique(day_idx):
        s = day_idx == d
        if s.sum() < 30:
            continue
        rho, _ = spearmanr(pred[s], rails[s])
        if np.isfinite(rho):
            out.append(float(rho))
    return out


# ---------------------------------------------------------------------------
# 评估:IC + 选择质量 + 因果 top-k replay(带 veto 门)
# ---------------------------------------------------------------------------
def eval_segment(
    seg_name: str,
    days: List[Tuple[str, pd.DataFrame]],
    model,
    norm,
    features: List[str],
    device,
    top_pcts=(0.02, 0.05),
) -> dict:
    import torch

    import rails_value_lgbm_v2 as v2
    from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
    from qqq_btc.qqq import config as qcfg
    from rails_value_lgbm import causal_topk_signal, entry_mask

    X, _yt, _yv, rails, day_idx, bar_idx, _ = build_sequences(days, features, norm)
    with torch.no_grad():
        preds, vetos = [], []
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(device)
            lt, lv = model(xb)
            preds.append(torch.sigmoid(lt).cpu().numpy())
            vetos.append(torch.sigmoid(lv).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    veto = np.concatenate(vetos) if vetos else np.array([])

    ics = daily_ic(pred, rails, day_idx)
    ic_mean = float(np.mean(ics)) if ics else float("nan")

    # 逐日整列预测(评估窗外 bar 置 -inf / veto=1)
    pred_by_day: Dict[str, np.ndarray] = {}
    veto_by_day: Dict[str, np.ndarray] = {}
    for di, (date_str, m) in enumerate(days):
        p = np.full(len(m), -np.inf)
        v = np.ones(len(m))
        sel = day_idx == di
        p[bar_idx[sel]] = pred[sel]
        v[bar_idx[sel]] = veto[sel]
        pred_by_day[date_str] = p
        veto_by_day[date_str] = v

    # 选择质量(top2%,无 veto)
    sel_vals, hit10 = [], []
    for date_str, m in days:
        w = entry_mask(m).to_numpy()
        sig = causal_topk_signal(pred_by_day[date_str], w, 0.02)
        rvv = m["rails_value"].to_numpy()
        s = (sig > 0) & np.isfinite(rvv)
        ok = w & np.isfinite(rvv)
        if s.any() and ok.any():
            thr = np.nanquantile(rvv[ok], 0.90)
            sel_vals.extend(rvv[s].tolist())
            hit10.extend((rvv[s] >= thr).astype(float).tolist())
    diag = {
        "sel_rails_mean": _round4(float(np.mean(sel_vals))) if sel_vals else None,
        "sel_oracle_top10_hit": _round4(float(np.mean(hit10))) if hit10 else None,
        "n_selected": len(sel_vals),
    }

    # replay(tick 从原始 parquet 重读)
    replay_rows = []
    tick_cache: Dict[str, pd.DataFrame] = {}
    for date_str, _m in days:
        fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
        buckets = v2.load_day_buckets(fp)
        t = buckets.get(2)
        tick_cache[date_str] = (
            t[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
            if t is not None and not t.empty
            else pd.DataFrame()
        )

    for pct in top_pcts:
        for veto_thr in (None, 0.5):
            day_rois, hits = [], []
            n_trades = 0
            for date_str, m in days:
                w = entry_mask(m).to_numpy()
                gate = None if veto_thr is None else (veto_by_day[date_str] < veto_thr)
                mm = m.copy()
                mm["tft_signal"] = causal_topk_signal(
                    pred_by_day[date_str], w, pct, gate=gate
                )
                r = run_event_replay(
                    mm,
                    qcfg.FILL_MODEL,
                    qcfg.REPLAY,
                    qcfg.EXIT_RAILS,
                    tick_df=tick_cache[date_str] if not tick_cache[date_str].empty else None,
                    edge_col="tft_signal",
                    event_cfg=EventReplayConfig(tick_disaster_stop=True),
                )
                if not r.trades:
                    day_rois.append(0.0)
                    continue
                rets = np.array([t.net_return for t in r.trades])
                day_rois.append(float(np.prod(1.0 + rets) - 1.0))
                n_trades += len(rets)
                hits.extend((rets > 0).astype(float).tolist())
            dr = np.array(day_rois)
            row = {
                "top_pct": pct,
                "veto_thr": veto_thr,
                "win_days": int((dr > 0).sum()),
                "days": len(dr),
                "trades": n_trades,
                "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
                "day_roi_mean": _round4(float(dr.mean())),
                "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
                "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
            }
            replay_rows.append(row)
            tag = "no_veto" if veto_thr is None else f"veto<{veto_thr}"
            print(
                f"[{seg_name}] replay top{pct:.0%} [{tag}]: win={row['win_days']}/{row['days']} "
                f"dayROI={row['day_roi_mean']:+.1%} comp={row['compound']:+.1%} "
                f"trades={row['trades']} hit={row['hit_rate']:.0%} worst={row['worst_day']:+.1%}"
            )

    print(
        f"[{seg_name}] IC={ic_mean:+.3f} (day IC>0: "
        f"{float(np.mean(np.array(ics) > 0)):.0%})  diag={diag}"
    )
    return {
        "segment": seg_name,
        "rank_ic_mean": _round4(ic_mean),
        "pos_day_frac": _round4(float(np.mean(np.array(ics) > 0))) if ics else None,
        "daily_ic": [_round4(v) for v in ics],
        "selection_diag_top2pct": diag,
        "replay": replay_rows,
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = cache_features()

    train_days = load_cached_days(args.train_globs.split(","))
    val_days = load_cached_days(args.val_globs.split(","))
    print(f"train days={len(train_days)} val days={len(val_days)} feats={len(features)}")

    Xtr, ytt, ytv, rtr, dtr, _btr, norm = build_sequences(train_days, features)
    Xva, yvt, yvv, rva, dva, _bva, _ = build_sequences(val_days, features, norm)
    print(f"seqs train={len(Xtr)} val={len(Xva)}  top_frac={ytt.mean():.3f} veto_frac={ytv.mean():.3f}")

    model = make_model(len(features), args.hidden, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,} device={device}")

    pos_w_top = torch.tensor((1.0 - ytt.mean()) / max(ytt.mean(), 1e-6), device=device)
    pos_w_veto = torch.tensor((1.0 - ytv.mean()) / max(ytv.mean(), 1e-6), device=device)
    bce_top = nn.BCEWithLogitsLoss(pos_weight=pos_w_top)
    bce_veto = nn.BCEWithLogitsLoss(pos_weight=pos_w_veto)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr_t = torch.from_numpy(Xtr)
    ytt_t = torch.from_numpy(ytt)
    ytv_t = torch.from_numpy(ytv)
    n = len(Xtr)
    best_ic, best_state, patience = -1e9, None, 0
    rng = np.random.default_rng(42)

    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            xb = Xtr_t[idx].to(device, non_blocking=True)
            lt, lv = model(xb)
            loss = bce_top(lt, ytt_t[idx].to(device)) + args.veto_weight * bce_veto(
                lv, ytv_t[idx].to(device)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xva), 4096):
                xb = torch.from_numpy(Xva[i : i + 4096]).to(device)
                lt, _ = model(xb)
                preds.append(torch.sigmoid(lt).cpu().numpy())
            pva = np.concatenate(preds)
        ics = daily_ic(pva, rva, dva)
        ic = float(np.mean(ics)) if ics else -1e9
        marker = ""
        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            marker = " *"
        else:
            patience += 1
        print(f"ep{ep:02d} loss={tot / n:.4f} val_IC={ic:+.4f}{marker}")
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
            "val_ic": best_ic,
        },
        ckpt,
    )
    print(f"best val IC={best_ic:+.4f} saved -> {ckpt}")

    # --- walk-forward 评估 ---
    model.eval()
    segments = []
    for seg in args.test_globs.split(";"):
        days = load_cached_days(seg.split(","))
        if not days:
            print(f"[{seg}] no cached days, skip")
            continue
        segments.append(eval_segment(seg, days, model, norm, features, device))

    result = {
        "meta": {
            "train_globs": args.train_globs,
            "val_globs": args.val_globs,
            "test_globs": args.test_globs,
            "seq_len": SEQ_LEN,
            "label_top": LABEL_TOP,
            "veto_dd": VETO_DD,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_params": n_params,
            "best_val_ic": _round4(best_ic),
        },
        "test_segments": segments,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="rails_value TFT training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-cache")
    b.add_argument("--globs", default="QQQ_2025-*.parquet,QQQ_2026-01-*.parquet,QQQ_2026-02-*.parquet")

    t = sub.add_parser("train")
    t.add_argument("--train-globs", default=",".join(f"QQQ_2025-0{m}-*.parquet" for m in range(1, 7)))
    t.add_argument("--val-globs", default="QQQ_2025-07-*.parquet")
    t.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet"]
        ),
    )
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--veto-weight", type=float, default=0.5)
    t.add_argument("--ckpt", default="/tmp/rails_value_tft.pt")
    t.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_rails_value_tft_walkforward.json",
    )

    args = ap.parse_args()
    if args.cmd == "build-cache":
        return cmd_build_cache(args)
    return cmd_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
