#!/usr/bin/env python3
"""
在线校准:绝对分数 → 期望 rails_value + 半日观察门。

动机(step11):listwise 模型在「全日 oracle top2%」下选中 bar 真实价值
均值 +20%,但因果滚动分位过触发、选中价值为负。瓶颈是日内分数尺度
漂移,不是排序能力。

本脚本冻结 rank TFT,只改入场门:
  A 因果 top2%(基线,已知失败)
  B 绝对校准阈值:用 val 段拟合 score→E[rails] 的分位映射,
     仅当校准期望 >= thr 才入场(跨日绝对尺度)
  C 半日观察门:上午只观察;用已结算 bar 的在线 IC 决定下午是否放行;
     放行后用绝对校准阈值稀疏入场
  D B+C 组合
  E 上界对照:全日 oracle top2%(含未来分位,不可交易)

协议:模型=rails_value_tft_rank_2022_2024;前向 2025-08~2026-03。
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


def predict_logits(days, model, norm, features, device):
    """{date: (logit_full, rails_full)};窗外=-inf。"""
    X, _yt, _yv, rails, day_idx, bar_idx, _ = build_sequences(days, features, norm)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(device)
            lt, _ = model(xb)
            preds.append(lt.cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    out = {}
    for di, (date_str, m) in enumerate(days):
        p = np.full(len(m), -np.inf)
        sel = day_idx == di
        p[bar_idx[sel]] = pred[sel]
        out[date_str] = (p, m["rails_value"].to_numpy())
    return out


class ScoreCalibrator:
    """
    分位映射:把 score 映射到历史同百分位的平均 rails_value。
    用 val 段拟合,测试段冻结。
    """

    def __init__(self, n_bins: int = 40):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit(self, scores: np.ndarray, rails: np.ndarray) -> "ScoreCalibrator":
        ok = np.isfinite(scores) & np.isfinite(rails) & (scores > -np.inf)
        s, r = scores[ok], rails[ok]
        self.global_mean = float(r.mean())
        qs = np.linspace(0, 1, self.n_bins + 1)
        self.bin_edges = np.unique(np.quantile(s, qs))
        if len(self.bin_edges) < 3:
            self.bin_means = np.array([self.global_mean])
            return self
        idx = np.digitize(s, self.bin_edges[1:-1], right=False)
        means = []
        for b in range(len(self.bin_edges) - 1):
            m = r[idx == b]
            means.append(float(m.mean()) if len(m) >= 20 else self.global_mean)
        self.bin_means = np.array(means, dtype=np.float64)
        # 单调化(isotonic 简化:累积 max from left for upper tail emphasis)
        # 对交易更重要的是高分端单调,用 PAVA 近似:从左到右强制非降
        for i in range(1, len(self.bin_means)):
            if self.bin_means[i] < self.bin_means[i - 1]:
                self.bin_means[i] = self.bin_means[i - 1]
        return self

    def expect(self, scores: np.ndarray) -> np.ndarray:
        out = np.full(len(scores), np.nan)
        if self.bin_edges is None or self.bin_means is None:
            return out
        ok = np.isfinite(scores) & (scores > -np.inf)
        if not ok.any():
            return out
        if len(self.bin_edges) < 3:
            out[ok] = self.global_mean
            return out
        idx = np.digitize(scores[ok], self.bin_edges[1:-1], right=False)
        idx = np.clip(idx, 0, len(self.bin_means) - 1)
        out[ok] = self.bin_means[idx]
        return out


def morning_online_ic(
    pred: np.ndarray,
    rails: np.ndarray,
    w: np.ndarray,
    cutoff_bar: int,
    session_bar: np.ndarray,
    min_obs: int = 20,
) -> float:
    """截止 cutoff_bar 时,用已结算样本算在线 IC。"""
    # 可用结算样本: session_bar <= cutoff 且 i <= t_cut - SETTLE_LAG
    # 简化:取所有 session_bar <= cutoff - 对应 settle 的入场窗 bar
    valid = w & np.isfinite(rails) & np.isfinite(pred) & (pred > -np.inf)
    # 结算完成的 bar:其 session_bar + SETTLE_LAG 对应的时刻已过 cutoff
    # session_bar[i] + SETTLE_LAG <= cutoff_bar
    settled = valid & (session_bar + SETTLE_LAG <= cutoff_bar)
    if settled.sum() < min_obs:
        return float("nan")
    rho, _ = spearmanr(pred[settled], rails[settled])
    return float(rho) if np.isfinite(rho) else float("nan")


def summarize(day_rois, n_trades, hits, sel_vals=None):
    dr = np.asarray(day_rois, dtype=np.float64)
    out = {
        "days": len(dr),
        "active_days": int((dr != 0).sum()),
        "win_days": int((dr > 0).sum()),
        "trades": n_trades,
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())) if len(dr) else 0.0,
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)) if len(dr) else 0.0,
        "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
    }
    if sel_vals is not None and len(sel_vals):
        out["sel_rails_mean"] = _round4(float(np.mean(sel_vals)))
        out["n_selected_bars"] = len(sel_vals)
    return out


def replay_signals(days, signals, tick_cache):
    day_rois, hits, sel_vals = [], [], []
    n_trades = 0
    for date_str, m in days:
        mm = m.copy()
        sig = signals[date_str]
        mm["gate_signal"] = sig
        rv = m["rails_value"].to_numpy()
        s = (sig > 0) & np.isfinite(rv)
        if s.any():
            sel_vals.extend(rv[s].tolist())
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
    return summarize(day_rois, n_trades, hits, sel_vals)


def oracle_topk_signal(pred, w, rails, top_pct: float) -> np.ndarray:
    """全日可见分位(上界,不可交易)。"""
    sig = np.full(len(pred), -1.0)
    ok = w & np.isfinite(pred) & (pred > -np.inf)
    if ok.sum() < 10:
        return sig
    idx = np.where(ok)[0]
    n_pick = max(1, int(round(len(idx) * top_pct)))
    top = idx[np.argsort(pred[idx])[-n_pick:]]
    sig[top] = 0.10
    return sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/s990/data/cache/rails_value_tft_rank_2022_2024.pt")
    ap.add_argument(
        "--cal-globs",
        default=",".join(f"QQQ_2025-0{m}-*.parquet" for m in range(1, 7)),
        help="拟合 score→E[rails] 的校准段(模型未见的 val 段)",
    )
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet", "QQQ_2026-03-*.parquet"]
        ),
    )
    ap.add_argument("--abs-thrs", default="0.02,0.05,0.08,0.10")
    ap.add_argument("--morning-cutoff", type=int, default=150, help="半日分界 session_bar(~12:00)")
    ap.add_argument("--morning-ic-thrs", default="0.05,0.10,0.15")
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_tft_rank_online_calibration.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    features = ck["features"]
    norm = (ck["norm_mu"], ck["norm_sd"])
    model = make_model(len(features), ck["hidden"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded rank ckpt val_ic={ck.get('val_ic')} sel={ck.get('sel_rails_mean')}")

    # --- 校准段 ---
    cal_days = load_cached_days(args.cal_globs.split(","))
    print(f"cal days={len(cal_days)}")
    cal_pred = predict_logits(cal_days, model, norm, features, device)
    cal_scores, cal_rails = [], []
    for ds, m in cal_days:
        p, rv = cal_pred[ds]
        w = entry_mask(m).to_numpy()
        ok = w & np.isfinite(rv) & np.isfinite(p) & (p > -np.inf)
        cal_scores.append(p[ok])
        cal_rails.append(rv[ok])
    cal = ScoreCalibrator(n_bins=40).fit(np.concatenate(cal_scores), np.concatenate(cal_rails))
    print(
        f"calibrator bins={len(cal.bin_means)} "
        f"low={cal.bin_means[0]:+.3f} high={cal.bin_means[-1]:+.3f}"
    )

    abs_thrs = [float(x) for x in args.abs_thrs.split(",")]
    ic_thrs = [float(x) for x in args.morning_ic_thrs.split(",")]

    # --- 测试段 ---
    all_results = []
    for seg in args.test_globs.split(";"):
        days = load_cached_days(seg.split(","))
        if not days:
            continue
        pred_map = predict_logits(days, model, norm, features, device)
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

        # 预计算期望与上午 IC
        expect: Dict[str, np.ndarray] = {}
        am_ic: Dict[str, float] = {}
        for date_str, m in days:
            p, rv = pred_map[date_str]
            w = entry_mask(m).to_numpy()
            expect[date_str] = cal.expect(p)
            sb = m["session_bar"].to_numpy(dtype=np.int64)
            am_ic[date_str] = morning_online_ic(p, rv, w, args.morning_cutoff, sb)

        variants = []

        # A: 因果 top2%
        sig_a = {}
        for date_str, m in days:
            p, _ = pred_map[date_str]
            w = entry_mask(m).to_numpy()
            sig_a[date_str] = causal_topk_signal(p, w, args.top_pct)
        variants.append(("A_causal_top2", sig_a))

        # B: 绝对校准阈值
        for thr in abs_thrs:
            sig = {}
            for date_str, m in days:
                e = expect[date_str]
                w = entry_mask(m).to_numpy()
                s = np.full(len(m), -1.0)
                ok = w & np.isfinite(e) & (e >= thr)
                s[ok] = 0.10
                sig[date_str] = s
            variants.append((f"B_abs>={thr}", sig))

        # C: 半日观察 + 下午因果 top2%
        for ict in ic_thrs:
            sig = {}
            for date_str, m in days:
                p, _ = pred_map[date_str]
                w = entry_mask(m).to_numpy()
                sb = m["session_bar"].to_numpy(dtype=np.int64)
                base = causal_topk_signal(p, w, args.top_pct)
                s = np.full(len(m), -1.0)
                if np.isfinite(am_ic[date_str]) and am_ic[date_str] >= ict:
                    # 只保留下午
                    keep = (base > 0) & (sb >= args.morning_cutoff)
                    s[keep] = 0.10
                sig[date_str] = s
            variants.append((f"C_amIC>={ict}_pm_top2", sig))

        # D: 半日观察 + 下午绝对阈值
        for thr in abs_thrs:
            for ict in ic_thrs:
                sig = {}
                for date_str, m in days:
                    e = expect[date_str]
                    w = entry_mask(m).to_numpy()
                    sb = m["session_bar"].to_numpy(dtype=np.int64)
                    s = np.full(len(m), -1.0)
                    if np.isfinite(am_ic[date_str]) and am_ic[date_str] >= ict:
                        ok = w & (sb >= args.morning_cutoff) & np.isfinite(e) & (e >= thr)
                        s[ok] = 0.10
                    sig[date_str] = s
                variants.append((f"D_amIC>={ict}_abs>={thr}", sig))

        # E: oracle top2% 上界
        sig_e = {}
        for date_str, m in days:
            p, rv = pred_map[date_str]
            w = entry_mask(m).to_numpy()
            sig_e[date_str] = oracle_topk_signal(p, w, rv, args.top_pct)
        variants.append(("E_oracle_top2", sig_e))

        month = {"segment": seg, "am_ic_mean": _round4(float(np.nanmean(list(am_ic.values()))))}
        print(f"\n[{seg}] am_ic_mean={month['am_ic_mean']:+.3f}")
        for name, sig in variants:
            stats = replay_signals(days, sig, tick_cache)
            month[name] = stats
            print(
                f"  {name:<28} win={stats['win_days']}/{stats['days']} "
                f"comp={stats['compound']:+.1%} trades={stats['trades']} "
                f"sel={stats.get('sel_rails_mean')} active={stats['active_days']}"
            )
        all_results.append(month)

    # 合计
    # 收集所有变体名
    var_names = [k for k in all_results[0].keys() if k not in ("segment", "am_ic_mean")]
    totals = {}
    print("\n=== TOTALS ===")
    for name in var_names:
        comps = [m[name]["compound"] for m in all_results]
        trades = sum(m[name]["trades"] for m in all_results)
        sels = [m[name].get("sel_rails_mean") for m in all_results if m[name].get("sel_rails_mean") is not None]
        total_comp = float(np.prod([1.0 + c for c in comps]) - 1.0)
        totals[name] = {
            "compound": _round4(total_comp),
            "pos_months": int(sum(1 for c in comps if c > 0)),
            "n_months": len(comps),
            "trades": trades,
            "sel_rails_mean": _round4(float(np.mean(sels))) if sels else None,
            "monthly": [_round4(c) for c in comps],
        }
        print(
            f"{name:<28} total={totals[name]['compound']:+.1%} "
            f"pos={totals[name]['pos_months']}/{totals[name]['n_months']} "
            f"trades={trades} sel={totals[name]['sel_rails_mean']} "
            f"| {' '.join(f'{c:+.0%}' for c in comps)}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "ckpt": args.ckpt,
                    "cal_globs": args.cal_globs,
                    "morning_cutoff": args.morning_cutoff,
                    "calibrator": {
                        "n_bins": len(cal.bin_means) if cal.bin_means is not None else 0,
                        "bin_mean_low": _round4(float(cal.bin_means[0])) if cal.bin_means is not None else None,
                        "bin_mean_high": _round4(float(cal.bin_means[-1])) if cal.bin_means is not None else None,
                    },
                },
                "totals": totals,
                "months": all_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
