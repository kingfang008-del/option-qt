#!/usr/bin/env python3
"""
两段式架构验证:现货路径模型 + 确定性期权价值换算层。

依据 spot_path_decomposition.py 的实证:rails_value 与未来30min现货路径
rank 相关 0.84,先知路径即可达日 IC 0.83 —— alpha 全部在『预测现货路径』。

第一段(学习):LGBM 在纯现货 1s 数据(2022-03~2025-06,约800天)上
  预测未来30min现货收益的 mean / q10 / q90 与最大回撤;
  特征只用现货(动量/波动/量能/订单流/VWAP/日内位置/距日低反弹结构)。
第二段(不学习):经验换算表 T[fut_ret桶, 时段桶, 权利金桶] → rails_value
  中位数,在 2025-01~07 期权数据上校准;
  期望价值 = Σ 情景权重 × T(情景) ,情景取预测分布的 5 个分位点。
评估:2025-08 ~ 2026-02 七个月,期权侧逐日 rank IC(vs rails_value)、
  top2% 选中质量、生产栈 replay(与 TFT/LGBM 完全同协议)。
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
from train_rails_value_tft import CACHE_DIR, RAW_DIR, load_cached_days

SPOT_DIR = Path("/mnt/s990/data/raw_1s/stocks/QQQ")
HOLD = 30
ENTRY_START, ENTRY_END = 15, 300

SPOT_FEATURES: List[str] = []


def _round4(x: float) -> float:
    return round(float(x), 4)


def session_bar_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour - 9) * 60 + ts.dt.minute - 30


# ---------------------------------------------------------------------------
# 现货特征(纯因果,含 V 形反转结构特征)
# ---------------------------------------------------------------------------
def build_spot_features(g: pd.DataFrame) -> pd.DataFrame:
    """g: load_spot_minute 输出(s_open/high/low/close/volume/upvol/dnvol/rv1s)。"""
    m = g.copy().reset_index(drop=True)
    sc = m["s_close"].ffill()
    logs = np.log(sc.where(sc > 0))
    f: Dict[str, pd.Series] = {}

    for w in (1, 3, 5, 15, 30, 60):
        f[f"ret_{w}"] = logs.diff(w)
    f["vol_10"] = logs.diff().rolling(10, min_periods=4).std()
    f["vol_30"] = logs.diff().rolling(30, min_periods=10).std()
    f["vol_ratio"] = f["vol_10"] / (f["vol_30"] + 1e-12)
    f["rv1s"] = m["s_rv1s"]
    f["rv1s_z30"] = (m["s_rv1s"] - m["s_rv1s"].rolling(30, min_periods=5).mean()) / (
        m["s_rv1s"].rolling(30, min_periods=5).std() + 1e-12
    )

    # 日内位置 / 结构(V 形反转的载体)
    cmin = sc.cummin()
    cmax = sc.cummax()
    f["dist_daylow"] = sc / cmin - 1.0
    f["dist_dayhigh"] = sc / cmax - 1.0
    f["day_range_pos"] = (sc - cmin) / (cmax - cmin + 1e-12)
    is_new_low = sc <= cmin + 1e-9
    grp = is_new_low.cumsum()
    f["bars_since_low"] = grp.groupby(grp).cumcount().astype(float)
    rmin30 = sc.rolling(30, min_periods=5).min()
    f["rebound_30"] = sc / rmin30 - 1.0           # 近30bar低点反弹幅度
    rmax30 = sc.rolling(30, min_periods=5).max()
    f["pullback_30"] = sc / rmax30 - 1.0
    f["ret_from_open"] = logs - logs.iloc[0]

    # 量能 / 订单流
    vol = m["s_volume"].astype(float)
    f["volume_z30"] = (vol - vol.rolling(30, min_periods=5).mean()) / (
        vol.rolling(30, min_periods=5).std() + 1e-9
    )
    flow = (m["s_upvol"] - m["s_dnvol"]) / (m["s_upvol"] + m["s_dnvol"] + 1e-9)
    f["flow_imb"] = flow
    f["flow_imb_ma10"] = flow.rolling(10, min_periods=3).mean()
    f["flow_imb_sum30"] = flow.rolling(30, min_periods=10).sum()
    cum_pv = (sc * vol).cumsum()
    f["vwap_dist"] = sc / (cum_pv / (vol.cumsum() + 1e-9)) - 1.0
    hl = (m["s_high"].ffill() - m["s_low"].ffill()) / sc
    f["hl_range_ma15"] = hl.rolling(15, min_periods=5).mean()
    f["close_pos_hl"] = (sc - m["s_low"].ffill()) / (m["s_high"].ffill() - m["s_low"].ffill() + 1e-9)

    sb = session_bar_of(m["timestamp"]).astype(float)
    f["session_bar"] = sb
    f["tod_sin"] = np.sin(2 * np.pi * sb / 390.0)
    f["tod_cos"] = np.cos(2 * np.pi * sb / 390.0)

    for k, val in f.items():
        m[k] = val
    global SPOT_FEATURES
    SPOT_FEATURES = sorted(f.keys())
    return m


def build_spot_day(date_str: str) -> Optional[pd.DataFrame]:
    g = v2.load_spot_minute(date_str)
    if g is None or len(g) < 200:
        return None
    m = build_spot_features(g)
    sc = m["s_close"].ffill().to_numpy(dtype=np.float64)
    n = len(m)
    fut_ret = np.full(n, np.nan)
    fut_dn = np.full(n, np.nan)
    for i in range(n):
        j0, j1 = i + 1, min(i + 1 + HOLD, n)
        if j1 - j0 < 10 or not (np.isfinite(sc[i]) and sc[i] > 0):
            continue
        seg = sc[j0:j1]
        fut_ret[i] = seg[-1] / sc[i] - 1.0
        fut_dn[i] = seg.min() / sc[i] - 1.0
    m["fut_ret"] = fut_ret
    m["fut_dn"] = fut_dn
    m["date"] = date_str
    return m


def spot_dates(start: str, end: str) -> List[str]:
    out = []
    for fp in sorted(SPOT_DIR.glob("QQQ_*.parquet")):
        d = fp.stem.split("_", 1)[-1]
        if start <= d <= end:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# 第二段:经验换算表
# ---------------------------------------------------------------------------
RET_EDGES = [-1, -0.004, -0.002, -0.001, -0.0003, 0.0003, 0.001, 0.002, 0.004, 1]
SB_EDGES = [0, 90, 180, 270, 400]
PREM_EDGES = [0, 0.004, 0.007, 1]  # 权利金/现货


class Translator:
    """T[ret_bucket, sb_bucket, prem_bucket] -> median rails_value,带边际回退。"""

    def __init__(self, cal: pd.DataFrame):
        cal = cal.copy()
        cal["rb"] = pd.cut(cal["fut_ret"], RET_EDGES, labels=False)
        cal["sb_b"] = pd.cut(cal["session_bar"], SB_EDGES, labels=False)
        cal["pb"] = pd.cut(cal["premium_pct"], PREM_EDGES, labels=False)
        self.full = cal.groupby(["rb", "sb_b", "pb"])["rails_value"].median()
        self.marg = cal.groupby("rb")["rails_value"].median()

    def value(self, fut_ret: float, session_bar: float, premium_pct: float) -> float:
        rb = int(np.digitize(fut_ret, RET_EDGES) - 1)
        rb = min(max(rb, 0), len(RET_EDGES) - 2)
        sb_b = int(np.digitize(session_bar, SB_EDGES) - 1)
        sb_b = min(max(sb_b, 0), len(SB_EDGES) - 2)
        pb = int(np.digitize(premium_pct, PREM_EDGES) - 1)
        pb = min(max(pb, 0), len(PREM_EDGES) - 2)
        v = self.full.get((rb, sb_b, pb))
        if v is None or not np.isfinite(v):
            v = self.marg.get(rb, 0.0)
        return float(v)


def scenario_expectation(
    tr: Translator, q10: float, q50: float, q90: float, sb: float, prem: float
) -> float:
    """5 情景近似(q10/q30/q50/q70/q90 各权重0.2,q30/q70 线性插值)。"""
    q30 = q50 - 0.5 * (q50 - q10)
    q70 = q50 + 0.5 * (q90 - q50)
    return float(np.mean([tr.value(q, sb, prem) for q in (q10, q30, q50, q70, q90)]))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="spot path model + deterministic translator")
    ap.add_argument("--train-start", default="2022-03-01")
    ap.add_argument("--train-end", default="2025-05-31")
    ap.add_argument("--valid-start", default="2025-06-01")
    ap.add_argument("--valid-end", default="2025-07-31")
    ap.add_argument("--cal-globs", default=",".join(f"QQQ_2025-0{m}-*.parquet" for m in range(1, 8)))
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet"]
        ),
    )
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument(
        "--out", default="New_Pro/baseline_qqq/reports/qqq_1dte_spot_path_model.json"
    )
    args = ap.parse_args()

    import lightgbm as lgb

    # ---------- 第一段:训练现货路径模型 ----------
    print("building spot training frames ...")
    tr_days = [build_spot_day(d) for d in spot_dates(args.train_start, args.train_end)]
    va_days = [build_spot_day(d) for d in spot_dates(args.valid_start, args.valid_end)]
    tr_df = pd.concat([d for d in tr_days if d is not None], ignore_index=True)
    va_df = pd.concat([d for d in va_days if d is not None], ignore_index=True)

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        sb = df["session_bar"]
        ok = (sb >= ENTRY_START) & (sb <= ENTRY_END) & df["fut_ret"].notna()
        return df[ok]

    tr_df, va_df = prep(tr_df), prep(va_df)
    print(f"train rows={len(tr_df)} ({tr_df['date'].nunique()}d)  "
          f"val rows={len(va_df)} ({va_df['date'].nunique()}d)  feats={len(SPOT_FEATURES)}")

    def fit(objective: str, alpha: Optional[float], target: str):
        params = dict(
            n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=42, verbose=-1,
        )
        if objective == "quantile":
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        else:
            mdl = lgb.LGBMRegressor(objective="regression", **params)
        mdl.fit(
            tr_df[SPOT_FEATURES], tr_df[target],
            eval_set=[(va_df[SPOT_FEATURES], va_df[target])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return mdl

    print("fitting mean/q10/q90/dn models ...")
    m_mean = fit("regression", None, "fut_ret")
    m_q10 = fit("quantile", 0.10, "fut_ret")
    m_q90 = fit("quantile", 0.90, "fut_ret")
    m_dn = fit("regression", None, "fut_dn")

    # 现货侧验证 IC
    pv = m_mean.predict(va_df[SPOT_FEATURES])
    ics = []
    for d, g in va_df.assign(p=pv).groupby("date"):
        if len(g) >= 30:
            r, _ = spearmanr(g["p"], g["fut_ret"])
            ics.append(r)
    spot_val_ic = float(np.mean(ics))
    print(f"spot-side val daily IC (pred vs fut_ret) = {spot_val_ic:+.3f}")

    imp = sorted(zip(SPOT_FEATURES, m_mean.feature_importances_), key=lambda x: -x[1])
    print("top spot features:", [k for k, _ in imp[:10]])

    # ---------- 校准换算表(期权训练月) ----------
    print("calibrating translator ...")
    cal_days = load_cached_days(args.cal_globs.split(","))
    cal_rows = []
    for date_str, m in cal_days:
        sp = v2.load_spot_minute(date_str)
        if sp is None:
            continue
        sc_map = dict(zip(sp["timestamp"], sp["s_close"]))
        ts = m["timestamp"]
        sc = ts.map(sc_map).ffill().to_numpy(dtype=np.float64)
        n = len(m)
        rv = m["rails_value"].to_numpy()
        mid = m["exec_call_mid"].to_numpy()
        sb = m["session_bar"].to_numpy()
        for i in range(n):
            j0, j1 = i + 1, min(i + 1 + HOLD, n)
            if (
                not (ENTRY_START <= sb[i] <= ENTRY_END)
                or not np.isfinite(rv[i])
                or j1 - j0 < 10
                or not (np.isfinite(sc[i]) and sc[i] > 0)
            ):
                continue
            seg = sc[j0:j1]
            if not np.all(np.isfinite(seg)):
                continue
            cal_rows.append({
                "fut_ret": seg[-1] / sc[i] - 1.0,
                "session_bar": sb[i],
                "premium_pct": mid[i] / sc[i],
                "rails_value": rv[i],
            })
    cal_df = pd.DataFrame(cal_rows)
    translator = Translator(cal_df)
    print(f"translator calibrated on {len(cal_df)} rows")

    # ---------- 评估:7 个测试月 ----------
    results = []
    for seg_name in args.test_globs.split(";"):
        days = load_cached_days(seg_name.split(","))
        if not days:
            continue

        # 逐日打分
        score_s1: Dict[str, np.ndarray] = {}   # 纯现货排序
        score_s2: Dict[str, np.ndarray] = {}   # 换算层期望
        gate_dn: Dict[str, np.ndarray] = {}    # 预测回撤门
        ics_s1, ics_s2 = [], []
        for date_str, m in days:
            sp = v2.load_spot_minute(date_str)
            if sp is None:
                score_s1[date_str] = np.full(len(m), -np.inf)
                score_s2[date_str] = np.full(len(m), -np.inf)
                gate_dn[date_str] = np.zeros(len(m), dtype=bool)
                continue
            spf = build_spot_features(sp)
            F = spf[SPOT_FEATURES].to_numpy(dtype=np.float64)
            p_mean = m_mean.predict(F)
            p_q10 = m_q10.predict(F)
            p_q90 = m_q90.predict(F)
            p_dn = m_dn.predict(F)
            pred_map = {
                t: (a, b, c, d)
                for t, a, b, c, d in zip(spf["timestamp"], p_mean, p_q10, p_q90, p_dn)
            }
            mid = m["exec_call_mid"].to_numpy()
            sc_map = dict(zip(sp["timestamp"], sp["s_close"]))
            sc = m["timestamp"].map(sc_map).ffill().to_numpy(dtype=np.float64)
            sb = m["session_bar"].to_numpy()

            s1 = np.full(len(m), -np.inf)
            s2 = np.full(len(m), -np.inf)
            gd = np.zeros(len(m), dtype=bool)
            for i, t in enumerate(m["timestamp"]):
                pr = pred_map.get(t)
                if pr is None:
                    continue
                pm, pq10, pq90, pdn = pr
                s1[i] = pm
                prem = mid[i] / sc[i] if np.isfinite(sc[i]) and sc[i] > 0 else 0.005
                s2[i] = scenario_expectation(translator, pq10, pm, pq90, sb[i], prem)
                gd[i] = pdn > -0.0018  # 预测30min最大回撤不超过 ~0.18%
            score_s1[date_str], score_s2[date_str], gate_dn[date_str] = s1, s2, gd

            rv = m["rails_value"].to_numpy()
            w = entry_mask(m).to_numpy() & np.isfinite(rv) & (s1 > -np.inf)
            if w.sum() >= 30:
                r1, _ = spearmanr(s1[w], rv[w])
                r2, _ = spearmanr(s2[w], rv[w])
                ics_s1.append(float(r1))
                ics_s2.append(float(r2))

        # 选择质量 + replay(S2)
        tick_cache: Dict[str, pd.DataFrame] = {}
        for date_str, _m in days:
            fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
            buckets = v2.load_day_buckets(fp)
            t = buckets.get(2)
            tick_cache[date_str] = (
                t[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
                if t is not None and not t.empty else pd.DataFrame()
            )

        month_out = {"segment": seg_name,
                     "ic_s1_spot_rank": _round4(float(np.mean(ics_s1))),
                     "ic_s2_translated": _round4(float(np.mean(ics_s2))),
                     "pos_day_frac_s2": _round4(float(np.mean(np.array(ics_s2) > 0)))}

        for tag, gated in (("no_gate", False), ("dn_gate", True)):
            day_rois, hits, sel_vals, hit10 = [], [], [], []
            n_trades = 0
            for date_str, m in days:
                w = entry_mask(m).to_numpy()
                gate = gate_dn[date_str] if gated else None
                mm = m.copy()
                sig = causal_topk_signal(score_s2[date_str], w, args.top_pct, gate=gate)
                mm["spot_signal"] = sig
                rvv = m["rails_value"].to_numpy()
                s = (sig > 0) & np.isfinite(rvv)
                ok = w & np.isfinite(rvv)
                if s.any() and ok.any():
                    sel_vals.extend(rvv[s].tolist())
                    hit10.extend((rvv[s] >= np.nanquantile(rvv[ok], 0.90)).astype(float).tolist())
                r = run_event_replay(
                    mm, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS,
                    tick_df=tick_cache[date_str] if not tick_cache[date_str].empty else None,
                    edge_col="spot_signal",
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
                "win_days": int((dr > 0).sum()),
                "days": len(dr),
                "trades": n_trades,
                "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
                "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
                "worst_day": _round4(float(dr.min())) if len(dr) else 0.0,
                "sel_rails_mean": _round4(float(np.mean(sel_vals))) if sel_vals else None,
                "sel_top10_hit": _round4(float(np.mean(hit10))) if hit10 else None,
            }

        rr0, rr1 = month_out["replay_no_gate"], month_out["replay_dn_gate"]
        print(
            f"[{seg_name}] IC_s1={month_out['ic_s1_spot_rank']:+.3f} "
            f"IC_s2={month_out['ic_s2_translated']:+.3f} | "
            f"no_gate comp={rr0['compound']:+.1%} sel={rr0['sel_rails_mean']} | "
            f"dn_gate comp={rr1['compound']:+.1%} trades={rr1['trades']} sel={rr1['sel_rails_mean']}"
        )
        results.append(month_out)

    summary = {
        "meta": {
            "train": [args.train_start, args.train_end],
            "valid": [args.valid_start, args.valid_end],
            "spot_val_ic": _round4(spot_val_ic),
            "n_spot_features": len(SPOT_FEATURES),
            "top_features": [k for k, _ in imp[:12]],
            "translator_rows": len(cal_df),
            "top_pct": args.top_pct,
        },
        "months": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
