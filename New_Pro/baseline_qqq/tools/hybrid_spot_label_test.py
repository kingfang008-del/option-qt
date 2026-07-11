#!/usr/bin/env python3
"""
对照实验:三种"特征 × 标签"组合,定位 alpha 到底在哪。

  A(已测) 期权+现货特征 → 期权 rails_value 标签 : IC 0.13~0.18(TFT/LGBM v2)
  B(已测) 纯现货特征   → 现货 fut_ret 标签     : IC ≈ 0(spot_path_model)
  C(本脚本) 期权+现货特征 → 现货 fut_ret 标签  → 换算层 → 期权侧评估

若 C 恢复到接近 A:alpha 主要来自期权市场结构信息(skew/期限/量),
  两段式仍成立,但第一段的输入必须包含期权侧;
若 C 仍≈0:rails_value 的可预测成分不在方向,而在波动/权利金时机,
  「预测现货路径」这条路线整体不可行。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
from spot_path_model import HOLD, Translator, scenario_expectation
from train_rails_value_tft import RAW_DIR, cache_features, load_cached_days

ENTRY_START, ENTRY_END = 15, 300


def _round4(x: float) -> float:
    return round(float(x), 4)


def attach_spot_labels(days: List) -> List:
    """给缓存日帧加 fut_ret / fut_dn / premium_pct(现货 join)。"""
    out = []
    for date_str, m in days:
        sp = v2.load_spot_minute(date_str)
        if sp is None:
            continue
        sc_map = dict(zip(sp["timestamp"], sp["s_close"]))
        sc = m["timestamp"].map(sc_map).ffill().to_numpy(dtype=np.float64)
        n = len(m)
        fut_ret = np.full(n, np.nan)
        fut_dn = np.full(n, np.nan)
        for i in range(n):
            j0, j1 = i + 1, min(i + 1 + HOLD, n)
            if j1 - j0 < 10 or not (np.isfinite(sc[i]) and sc[i] > 0):
                continue
            seg = sc[j0:j1]
            if not np.all(np.isfinite(seg)):
                continue
            fut_ret[i] = seg[-1] / sc[i] - 1.0
            fut_dn[i] = seg.min() / sc[i] - 1.0
        m = m.copy()
        m["fut_ret"] = fut_ret
        m["fut_dn"] = fut_dn
        m["premium_pct"] = np.where(sc > 0, m["exec_call_mid"] / sc, np.nan)
        out.append((date_str, m))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-globs", default=",".join(f"QQQ_2025-0{m}-*.parquet" for m in range(1, 7)))
    ap.add_argument("--val-globs", default="QQQ_2025-07-*.parquet")
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet"]
        ),
    )
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument(
        "--out", default="New_Pro/baseline_qqq/reports/qqq_1dte_hybrid_spot_label.json"
    )
    args = ap.parse_args()

    import lightgbm as lgb

    features = cache_features()

    train_days = attach_spot_labels(load_cached_days(args.train_globs.split(",")))
    val_days = attach_spot_labels(load_cached_days(args.val_globs.split(",")))
    print(f"train days={len(train_days)} val days={len(val_days)} feats={len(features)}")

    def to_frame(days):
        subs = []
        for _d, m in days:
            sb = m["session_bar"]
            ok = (sb >= ENTRY_START) & (sb <= ENTRY_END) & m["fut_ret"].notna()
            cols = list(dict.fromkeys(
                features + ["fut_ret", "fut_dn", "rails_value", "session_bar", "premium_pct"]
            ))
            subs.append(m.loc[ok, cols].assign(date=_d))
        return pd.concat(subs, ignore_index=True)

    tr_df, va_df = to_frame(train_days), to_frame(val_days)
    print(f"rows train={len(tr_df)} val={len(va_df)}")

    def fit(objective, alpha, target):
        params = dict(
            n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
            random_state=42, verbose=-1,
        )
        if objective == "quantile":
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        else:
            mdl = lgb.LGBMRegressor(objective="regression", **params)
        mdl.fit(tr_df[features], tr_df[target],
                eval_set=[(va_df[features], va_df[target])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        return mdl

    m_mean = fit("regression", None, "fut_ret")
    m_q10 = fit("quantile", 0.10, "fut_ret")
    m_q90 = fit("quantile", 0.90, "fut_ret")
    m_dn = fit("regression", None, "fut_dn")

    # 现货标签的 val IC(这是关键读数:期权特征能否预测现货路径)
    pv = m_mean.predict(va_df[features])
    ics = [spearmanr(g["p"], g["fut_ret"])[0]
           for _d, g in va_df.assign(p=pv).groupby("date") if len(g) >= 30]
    print(f"hybrid: val daily IC (pred vs fut_ret spot label) = {np.mean(ics):+.3f}")
    imp = sorted(zip(features, m_mean.feature_importances_), key=lambda x: -x[1])
    print("top features:", [k for k, _ in imp[:12]])

    translator = Translator(tr_df[["fut_ret", "session_bar", "premium_pct", "rails_value"]].dropna())

    results = []
    for seg_name in args.test_globs.split(";"):
        days = attach_spot_labels(load_cached_days(seg_name.split(",")))
        if not days:
            continue
        tick_cache: Dict[str, pd.DataFrame] = {}
        for date_str, _m in days:
            fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
            buckets = v2.load_day_buckets(fp)
            t = buckets.get(2)
            tick_cache[date_str] = (
                t[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
                if t is not None and not t.empty else pd.DataFrame()
            )

        ic_spot, ic_rails = [], []
        scores: Dict[str, np.ndarray] = {}
        gates: Dict[str, np.ndarray] = {}
        for date_str, m in days:
            F = m[features].to_numpy(dtype=np.float64)
            pm = m_mean.predict(F)
            pq10 = m_q10.predict(F)
            pq90 = m_q90.predict(F)
            pdn = m_dn.predict(F)
            sb = m["session_bar"].to_numpy()
            prem = m["premium_pct"].to_numpy()
            s2 = np.array([
                scenario_expectation(translator, a, b, c,
                                     sb[i], prem[i] if np.isfinite(prem[i]) else 0.005)
                for i, (a, b, c) in enumerate(zip(pq10, pm, pq90))
            ])
            scores[date_str] = s2
            gates[date_str] = pdn > -0.0018

            rv = m["rails_value"].to_numpy()
            fr = m["fut_ret"].to_numpy()
            w = entry_mask(m).to_numpy() & np.isfinite(rv) & np.isfinite(fr)
            if w.sum() >= 30:
                ic_spot.append(float(spearmanr(pm[w], fr[w])[0]))
                ic_rails.append(float(spearmanr(s2[w], rv[w])[0]))

        month_out = {
            "segment": seg_name,
            "ic_spot_label": _round4(float(np.mean(ic_spot))),
            "ic_rails_translated": _round4(float(np.mean(ic_rails))),
        }
        for tag, gated in (("no_gate", False), ("dn_gate", True)):
            day_rois, hits, sel_vals = [], [], []
            n_trades = 0
            for date_str, m in days:
                w = entry_mask(m).to_numpy()
                sig = causal_topk_signal(scores[date_str], w, args.top_pct,
                                         gate=gates[date_str] if gated else None)
                mm = m.copy()
                mm["hybrid_signal"] = sig
                rvv = m["rails_value"].to_numpy()
                s = (sig > 0) & np.isfinite(rvv)
                if s.any():
                    sel_vals.extend(rvv[s].tolist())
                r = run_event_replay(
                    mm, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS,
                    tick_df=tick_cache[date_str] if not tick_cache[date_str].empty else None,
                    edge_col="hybrid_signal",
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
            month_out[f"replay_{tag}"] = {
                "win_days": int((dr > 0).sum()), "days": len(dr), "trades": n_trades,
                "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
                "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
                "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
                "sel_rails_mean": _round4(float(np.mean(sel_vals))) if sel_vals else None,
            }
        r0, r1 = month_out["replay_no_gate"], month_out["replay_dn_gate"]
        print(
            f"[{seg_name}] IC_spot={month_out['ic_spot_label']:+.3f} "
            f"IC_rails={month_out['ic_rails_translated']:+.3f} | "
            f"no_gate comp={r0['compound']:+.1%} sel={r0['sel_rails_mean']} | "
            f"dn_gate comp={r1['compound']:+.1%} trades={r1['trades']} sel={r1['sel_rails_mean']}"
        )
        results.append(month_out)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"val_ic_spot_label": _round4(float(np.mean(ics))),
                 "top_features": [k for k, _ in imp[:15]]},
        "months": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
