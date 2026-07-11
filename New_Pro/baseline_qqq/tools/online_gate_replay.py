#!/usr/bin/env python3
"""
日内在线放行门:用「当日已结算 bar 的预测-实际相关性」因果地决定是否放行交易。

动机(step5 诊断):TFT 的 IC 月均 0.176 已过盈亏平衡线,但"尾部选择"月度
不稳定(09月 IC 0.193 却在 top2% 全错)。与其猜哪个月灵,不如在日内实时
量测:模型今天的预测和已经跑完 rails 的 bar 的真实价值相关吗?

因果性:bar i 的 rails_value 最迟在 i+1+45(max_hold)+1 bar 结算完毕,
故在 bar t 可用样本为 i <= t-47。另提供跨日门:过去 K 个交易日的
全日已实现 IC 均值(完全因果,无日内延迟)。

变体:
  A baseline        : top-k + veto(step5 原样)
  B intraday gate   : additionally 要求 当日在线IC(>=20样本) >= thr
  C trailing gate   : additionally 要求 过去5日已实现日IC均值 >= thr
  D B+C 组合

输出:逐月 × 变体的 replay 汇总 + 7个月合计。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
from train_rails_value_tft import (
    RAW_DIR,
    build_sequences,
    load_cached_days,
    make_model,
)

SETTLE_LAG = 47  # bar i 的 rails_value 在 i+SETTLE_LAG 时保证已结算(1延迟+45 max_hold+1)


def _round4(x: float) -> float:
    return round(float(x), 4)


def predict_days(days, model, norm, features, device):
    """返回 {date: (pred_full, veto_full, rails_full)};窗外 bar pred=-inf。"""
    import torch

    X, _yt, _yv, rails, day_idx, bar_idx, _ = build_sequences(days, features, norm)
    preds, vetos = [], []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(device)
            lt, lv = model(xb)
            preds.append(torch.sigmoid(lt).cpu().numpy())
            vetos.append(torch.sigmoid(lv).cpu().numpy())
    pred = np.concatenate(preds)
    veto = np.concatenate(vetos)

    out = {}
    for di, (date_str, m) in enumerate(days):
        p = np.full(len(m), -np.inf)
        vv = np.ones(len(m))
        sel = day_idx == di
        p[bar_idx[sel]] = pred[sel]
        vv[bar_idx[sel]] = veto[sel]
        out[date_str] = (p, vv, m["rails_value"].to_numpy())
    return out


def intraday_ic_series(pred: np.ndarray, rails: np.ndarray, w: np.ndarray, min_obs: int = 20) -> np.ndarray:
    """online_ic[t] = spearman(pred[i], rails[i]) for 入场窗内 i <= t-SETTLE_LAG;样本不足为 NaN。"""
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


def realized_day_ic(pred: np.ndarray, rails: np.ndarray, w: np.ndarray) -> float:
    valid = w & np.isfinite(rails) & np.isfinite(pred) & (pred > -np.inf)
    if valid.sum() < 30:
        return float("nan")
    rho, _ = spearmanr(pred[valid], rails[valid])
    return float(rho)


def replay_days(days, signals_by_day, tick_cache) -> Tuple[np.ndarray, int, list]:
    day_rois, hits = [], []
    n_trades = 0
    for date_str, m in days:
        mm = m.copy()
        mm["gate_signal"] = signals_by_day[date_str]
        r = run_event_replay(
            mm,
            qcfg.FILL_MODEL,
            qcfg.REPLAY,
            qcfg.EXIT_RAILS,
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


def summarize(dr: np.ndarray, n_trades: int, hits: list) -> dict:
    return {
        "days": len(dr),
        "active_days": int((dr != 0).sum()),
        "win_days": int((dr > 0).sum()),
        "trades": n_trades,
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())),
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
        "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="online gating replay on TFT predictions")
    ap.add_argument("--ckpt", default="/mnt/s990/data/cache/rails_value_tft_2025H1_lr2e4.pt")
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet"]
        ),
    )
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument("--veto-thr", type=float, default=0.5)
    ap.add_argument("--intraday-thrs", default="0.0,0.1")
    ap.add_argument("--trailing-k", type=int, default=5)
    ap.add_argument("--trailing-thrs", default="0.05,0.10")
    ap.add_argument("--pick-trailing-k", type=int, default=10)
    ap.add_argument("--pick-thrs", default="-0.02,0.0,0.02")
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_tft_online_gate.json",
    )
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    features = ck["features"]
    norm = (ck["norm_mu"], ck["norm_sd"])
    model = make_model(len(features), ck["hidden"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded ckpt val_ic={ck.get('val_ic'):.4f} feats={len(features)}")

    intraday_thrs = [float(x) for x in args.intraday_thrs.split(",")]
    trailing_thrs = [float(x) for x in args.trailing_thrs.split(",")]

    # --- 逐月装载 + 预测(合并成按日期排序的全序列,便于跨日 trailing 门) ---
    all_days: List[Tuple[str, pd.DataFrame]] = []
    seg_of_day: Dict[str, str] = {}
    for seg in args.test_globs.split(";"):
        days = load_cached_days(seg.split(","))
        for d in days:
            seg_of_day[d[0]] = seg
        all_days.extend(days)
    all_days.sort(key=lambda x: x[0])
    print(f"total test days={len(all_days)}")

    pred_map = predict_days(all_days, model, norm, features, device)

    tick_cache: Dict[str, pd.DataFrame] = {}
    for date_str, _m in all_days:
        fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
        buckets = v2.load_day_buckets(fp)
        t = buckets.get(2)
        tick_cache[date_str] = (
            t[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
            if t is not None and not t.empty
            else pd.DataFrame()
        )

    # --- 预计算:日内在线IC序列、全日已实现IC ---
    online_ic: Dict[str, np.ndarray] = {}
    day_ic: Dict[str, float] = {}
    for date_str, m in all_days:
        p, _v, rv = pred_map[date_str]
        w = entry_mask(m).to_numpy()
        online_ic[date_str] = intraday_ic_series(p, rv, w)
        day_ic[date_str] = realized_day_ic(p, rv, w)

    dates_sorted = [d for d, _ in all_days]
    trailing_ic: Dict[str, float] = {}
    for i, d in enumerate(dates_sorted):
        prev = [day_ic[x] for x in dates_sorted[max(0, i - args.trailing_k):i]]
        prev = [x for x in prev if np.isfinite(x)]
        trailing_ic[d] = float(np.mean(prev)) if len(prev) >= 3 else float("nan")

    # --- pick-quality 门:模型无门控 top-k 选择的已实现 rails_value(逐日) ---
    day_pick_vals: Dict[str, List[float]] = {}
    for date_str, m in all_days:
        p, _v, rv = pred_map[date_str]
        w = entry_mask(m).to_numpy()
        s = causal_topk_signal(p, w, args.top_pct)
        sel = (s > 0) & np.isfinite(rv)
        day_pick_vals[date_str] = rv[sel].tolist()

    trailing_pick_q: Dict[str, float] = {}
    pk = args.pick_trailing_k
    for i, d in enumerate(dates_sorted):
        vals: List[float] = []
        for x in dates_sorted[max(0, i - pk):i]:
            vals.extend(day_pick_vals[x])
        trailing_pick_q[d] = float(np.mean(vals)) if len(vals) >= 8 else float("nan")

    # --- 构造各变体信号 ---
    def build_signals(
        intraday_thr: Optional[float],
        trailing_thr: Optional[float],
        pick_thr: Optional[float] = None,
    ):
        sig = {}
        for date_str, m in all_days:
            p, v, _rv = pred_map[date_str]
            w = entry_mask(m).to_numpy()
            gate = v < args.veto_thr
            if intraday_thr is not None:
                oi = online_ic[date_str]
                gate = gate & np.isfinite(oi) & (oi >= intraday_thr)
            if trailing_thr is not None:
                ti = trailing_ic[date_str]
                if not (np.isfinite(ti) and ti >= trailing_thr):
                    gate = np.zeros(len(m), dtype=bool)
            if pick_thr is not None:
                pq = trailing_pick_q[date_str]
                if not (np.isfinite(pq) and pq >= pick_thr):
                    gate = np.zeros(len(m), dtype=bool)
            sig[date_str] = causal_topk_signal(p, w, args.top_pct, gate=gate)
        return sig

    pick_thrs = [float(x) for x in args.pick_thrs.split(",")]
    variants = [("A_baseline_veto", None, None, None)]
    for it in intraday_thrs:
        variants.append((f"B_intraday_ic>={it}", it, None, None))
    for tt in trailing_thrs:
        variants.append((f"C_trailing5d_ic>={tt}", None, tt, None))
    variants.append((f"D_combo_intraday>={intraday_thrs[0]}_trailing>={trailing_thrs[0]}",
                     intraday_thrs[0], trailing_thrs[0], None))
    for pt in pick_thrs:
        variants.append((f"E_pickq10d>={pt}", None, None, pt))
    variants.append((f"F_pickq+intraday>={intraday_thrs[0]}", intraday_thrs[0], None, pick_thrs[1]))

    segs = list(dict.fromkeys(seg_of_day.values()))
    results = []
    for name, it, tt, pt in variants:
        signals = build_signals(it, tt, pt)
        per_month = {}
        month_comps = []
        all_dr, all_trades, all_hits = [], 0, []
        for seg in segs:
            seg_days = [(d, m) for d, m in all_days if seg_of_day[d] == seg]
            dr, ntr, hits = replay_days(seg_days, signals, tick_cache)
            s = summarize(dr, ntr, hits)
            per_month[seg.split("*")[0][-8:-1]] = s
            month_comps.append(s["compound"])
            all_dr.extend(dr.tolist())
            all_trades += ntr
            all_hits.extend(hits)
        total = summarize(np.array(all_dr), all_trades, all_hits)
        total["pos_months"] = int(np.sum(np.array(month_comps) > 0))
        total["n_months"] = len(month_comps)
        results.append({"variant": name, "total": total, "per_month": per_month})
        comps = " ".join(f"{c:+.0%}" for c in month_comps)
        print(
            f"{name:<42} months+ {total['pos_months']}/{total['n_months']} "
            f"total_comp={total['compound']:+.1%} trades={total['trades']} "
            f"hit={total['hit_rate']:.0%} worst={total['worst_day']:+.1%} | {comps}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "ckpt": args.ckpt,
                    "top_pct": args.top_pct,
                    "veto_thr": args.veto_thr,
                    "settle_lag": SETTLE_LAG,
                    "trailing_k": args.trailing_k,
                    "day_ic": {k: _round4(v) if np.isfinite(v) else None for k, v in day_ic.items()},
                },
                "variants": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
