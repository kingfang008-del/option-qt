#!/usr/bin/env python3
"""
前向复验:阈值选择段 vs 真正 holdout。

协议:
  - 模型:冻结 TFT(2025H1 训练,lr2e4 ckpt),不再重训
  - 选择段(cal):2025-08 ~ 2025-10 —— 只在这里网格搜门限
  - Holdout:2025-11 ~ 2026-03 —— 阈值冻结后原样评估
  - 特别关注 2026-03:此前所有实验从未见过

门变体(与 online_gate_replay 一致):
  A baseline veto
  E pick-quality(过去10日 top-k 选中 bar 真实 rails_value 均值)
  B 日内在线 IC
  F = E + B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.qqq import config as qcfg

import rails_value_lgbm_v2 as v2
from rails_value_lgbm import causal_topk_signal, entry_mask
from online_gate_replay import SETTLE_LAG
from train_rails_value_tft import (
    RAW_DIR,
    build_sequences,
    load_cached_days,
    make_model,
)


def _round4(x: float) -> float:
    return round(float(x), 4)


def predict_days(days, model, norm, features, device):
    X, _yt, _yv, rails, day_idx, bar_idx, _ = build_sequences(days, features, norm)
    preds, vetos = [], []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(device)
            lt, lv = model(xb)
            preds.append(torch.sigmoid(lt).cpu().numpy())
            vetos.append(torch.sigmoid(lv).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    veto = np.concatenate(vetos) if vetos else np.array([])
    out = {}
    for di, (date_str, m) in enumerate(days):
        p = np.full(len(m), -np.inf)
        vv = np.ones(len(m))
        sel = day_idx == di
        p[bar_idx[sel]] = pred[sel]
        vv[bar_idx[sel]] = veto[sel]
        out[date_str] = (p, vv, m["rails_value"].to_numpy())
    return out


def intraday_ic_series(pred, rails, w, min_obs=20):
    n = len(pred)
    valid = w & np.isfinite(rails) & np.isfinite(pred) & (pred > -np.inf)
    idx_valid = np.where(valid)[0]
    out = np.full(n, np.nan)
    for t in range(n):
        settled = idx_valid[idx_valid <= t - SETTLE_LAG]
        if len(settled) < min_obs:
            continue
        rho, _ = spearmanr(pred[settled], rails[settled])
        if np.isfinite(rho):
            out[t] = rho
    return out


def realized_day_ic(pred, rails, w):
    valid = w & np.isfinite(rails) & np.isfinite(pred) & (pred > -np.inf)
    if valid.sum() < 30:
        return float("nan")
    rho, _ = spearmanr(pred[valid], rails[valid])
    return float(rho)


def summarize(dr, n_trades, hits):
    dr = np.asarray(dr, dtype=np.float64)
    return {
        "days": len(dr),
        "active_days": int((dr != 0).sum()),
        "win_days": int((dr > 0).sum()),
        "trades": n_trades,
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())) if len(dr) else 0.0,
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)) if len(dr) else 0.0,
        "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
    }


def replay_days(days, signals, tick_cache):
    day_rois, hits = [], []
    n_trades = 0
    for date_str, m in days:
        mm = m.copy()
        mm["gate_signal"] = signals[date_str]
        r = run_event_replay(
            mm, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS,
            tick_df=tick_cache[date_str] if not tick_cache[date_str].empty else None,
            edge_col="gate_signal",
            event_cfg=EventReplayConfig(tick_disaster_stop=True),
        )
        if not r.trades:
            day_rois.append(0.0)
            continue
        rets = np.array([t.net_return for t in r.trades])
        day_rois.append(float(np.prod(1.0 + rets) - 1.0))
        n_trades += len(rets)
        hits.extend((rets > 0).astype(float).tolist())
    return np.array(day_rois), n_trades, hits


def month_key(date_str: str) -> str:
    return date_str[:7]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/s990/data/cache/rails_value_tft_2025H1_lr2e4.pt")
    ap.add_argument(
        "--cal-months", default="2025-08,2025-09,2025-10",
        help="阈值选择段(逗号分隔 YYYY-MM)",
    )
    ap.add_argument(
        "--holdout-months", default="2025-11,2025-12,2026-01,2026-02,2026-03",
        help="真正前向段",
    )
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument("--veto-thr", type=float, default=0.5)
    ap.add_argument("--pick-trailing-k", type=int, default=10)
    ap.add_argument("--pick-thrs", default="-0.02,0.0,0.02")
    ap.add_argument("--intraday-thrs", default="0.0,0.1")
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_tft_online_gate_forward.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    features = ck["features"]
    norm = (ck["norm_mu"], ck["norm_sd"])
    model = make_model(len(features), ck["hidden"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded ckpt val_ic={ck.get('val_ic'):.4f}")

    cal_months = args.cal_months.split(",")
    hold_months = args.holdout_months.split(",")
    all_months = cal_months + hold_months

    # 装载全部月份(按日期排序,trailing 门需要跨月连续)
    all_days: List[Tuple[str, pd.DataFrame]] = []
    for ym in all_months:
        y, m = ym.split("-")
        days = load_cached_days([f"QQQ_{y}-{m}-*.parquet"])
        print(f"  {ym}: {len(days)} days")
        all_days.extend(days)
    all_days.sort(key=lambda x: x[0])
    print(f"total days={len(all_days)}")

    pred_map = predict_days(all_days, model, norm, features, device)

    tick_cache: Dict[str, pd.DataFrame] = {}
    for date_str, _m in all_days:
        fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
        buckets = v2.load_day_buckets(fp)
        t = buckets.get(2)
        tick_cache[date_str] = (
            t[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
            if t is not None and not t.empty else pd.DataFrame()
        )

    online_ic: Dict[str, np.ndarray] = {}
    day_ic: Dict[str, float] = {}
    day_pick_vals: Dict[str, List[float]] = {}
    for date_str, m in all_days:
        p, _v, rv = pred_map[date_str]
        w = entry_mask(m).to_numpy()
        online_ic[date_str] = intraday_ic_series(p, rv, w)
        day_ic[date_str] = realized_day_ic(p, rv, w)
        s = causal_topk_signal(p, w, args.top_pct)
        sel = (s > 0) & np.isfinite(rv)
        day_pick_vals[date_str] = rv[sel].tolist()

    dates_sorted = [d for d, _ in all_days]
    trailing_pick_q: Dict[str, float] = {}
    pk = args.pick_trailing_k
    for i, d in enumerate(dates_sorted):
        vals: List[float] = []
        for x in dates_sorted[max(0, i - pk):i]:
            vals.extend(day_pick_vals[x])
        trailing_pick_q[d] = float(np.mean(vals)) if len(vals) >= 8 else float("nan")

    pick_thrs = [float(x) for x in args.pick_thrs.split(",")]
    intrad_thrs = [float(x) for x in args.intraday_thrs.split(",")]

    def build_signals(pick_thr: Optional[float], intrad_thr: Optional[float]):
        sig = {}
        for date_str, m in all_days:
            p, v, _rv = pred_map[date_str]
            w = entry_mask(m).to_numpy()
            gate = v < args.veto_thr
            if intrad_thr is not None:
                oi = online_ic[date_str]
                gate = gate & np.isfinite(oi) & (oi >= intrad_thr)
            if pick_thr is not None:
                pq = trailing_pick_q[date_str]
                if not (np.isfinite(pq) and pq >= pick_thr):
                    gate = np.zeros(len(m), dtype=bool)
            sig[date_str] = causal_topk_signal(p, w, args.top_pct, gate=gate)
        return sig

    def eval_period(signals, months: List[str]) -> Tuple[dict, Dict[str, dict]]:
        per = {}
        all_dr, all_tr, all_hits = [], 0, []
        for ym in months:
            seg_days = [(d, m) for d, m in all_days if month_key(d) == ym]
            if not seg_days:
                continue
            dr, ntr, hits = replay_days(seg_days, signals, tick_cache)
            s = summarize(dr, ntr, hits)
            per[ym] = s
            all_dr.extend(dr.tolist())
            all_tr += ntr
            all_hits.extend(hits)
        total = summarize(all_dr, all_tr, all_hits)
        comps = [per[ym]["compound"] for ym in months if ym in per]
        total["pos_months"] = int(sum(1 for c in comps if c > 0))
        total["n_months"] = len(comps)
        total["monthly_compounds"] = [_round4(c) for c in comps]
        return total, per

    # --- 网格:只在 cal 段选最优 F ---
    grid = [("A_baseline", None, None)]
    for pt in pick_thrs:
        grid.append((f"E_pickq>={pt}", pt, None))
    for it in intrad_thrs:
        grid.append((f"B_intraday>={it}", None, it))
    for pt in pick_thrs:
        for it in intrad_thrs:
            grid.append((f"F_pickq>={pt}+intraday>={it}", pt, it))

    print("\n=== CALIBRATION (threshold selection) ===")
    cal_rows = []
    best_f = None
    best_f_score = -1e18
    for name, pt, it in grid:
        signals = build_signals(pt, it)
        total, per = eval_period(signals, cal_months)
        # 选择目标:cal 段 compound,平局取交易更多
        score = total["compound"] + 1e-6 * total["trades"]
        cal_rows.append({"variant": name, "pick_thr": pt, "intraday_thr": it,
                         "cal_total": total, "cal_per_month": per, "score": score})
        tag = " *" if name.startswith("F_") and score > best_f_score else ""
        if name.startswith("F_") and score > best_f_score:
            best_f_score = score
            best_f = (name, pt, it)
        print(
            f"{name:<36} cal_comp={total['compound']:+.1%} "
            f"months+={total['pos_months']}/{total['n_months']} "
            f"trades={total['trades']}{tag}"
        )

    assert best_f is not None
    print(f"\nselected on cal: {best_f[0]}")

    # --- Holdout:冻结阈值 ---
    print("\n=== HOLDOUT (frozen thresholds) ===")
    holdout_rows = []
    # 始终报告 baseline + 选中的 F + 之前口头选的 F(pickq>=0,intraday>=0)
    report_variants = [
        ("A_baseline", None, None),
        best_f,
        ("F_prev_pickq>=0+intraday>=0", 0.0, 0.0),
    ]
    # 去重
    seen = set()
    uniq = []
    for v in report_variants:
        if v[0] not in seen:
            seen.add(v[0])
            uniq.append(v)

    for name, pt, it in uniq:
        signals = build_signals(pt, it)
        total, per = eval_period(signals, hold_months)
        holdout_rows.append({
            "variant": name, "pick_thr": pt, "intraday_thr": it,
            "holdout_total": total, "holdout_per_month": per,
        })
        comps = " ".join(f"{c:+.0%}" for c in total["monthly_compounds"])
        print(
            f"{name:<36} hold_comp={total['compound']:+.1%} "
            f"months+={total['pos_months']}/{total['n_months']} "
            f"trades={total['trades']} worst={total['worst_day']:+.1%} | {comps}"
        )

    # 2026-03 单独
    print("\n=== 2026-03 pure forward (never seen) ===")
    mar_rows = []
    for name, pt, it in uniq:
        signals = build_signals(pt, it)
        total, per = eval_period(signals, ["2026-03"])
        mar_rows.append({"variant": name, "march": total})
        print(
            f"{name:<36} mar_comp={total['compound']:+.1%} "
            f"trades={total['trades']} hit={total['hit_rate']:.0%} "
            f"active={total['active_days']}/{total['days']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "ckpt": args.ckpt,
            "cal_months": cal_months,
            "holdout_months": hold_months,
            "selected_on_cal": {"name": best_f[0], "pick_thr": best_f[1], "intraday_thr": best_f[2]},
            "top_pct": args.top_pct,
            "veto_thr": args.veto_thr,
        },
        "calibration_grid": [
            {k: v for k, v in r.items() if k != "score"} for r in cal_rows
        ],
        "holdout": holdout_rows,
        "march_2026": mar_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
