#!/usr/bin/env python3
"""
验证「期权 rails_value ≈ 未来现货路径 × 确定性时间损耗」的分解假设。

若给定未来现货路径后 rails_value 几乎可确定(高 R²),则:
  - 全部可学的 alpha 都在『预测现货路径』这一步
  - 期权侧只是一个无需学习的确定性换算层(delta/gamma/theta)
  - 建模对象应从期权报价序列换成现货路径

输出:
  1) rails_value 与未来现货路径统计量的 rank 相关
  2) 「先知现货路径」LGBM(只喂未来现货路径+时间+入场时权利金水平)
     对 rails_value 的解释力(rank IC / R²)—— 这就是换算层的可确定性上限
  3) 按时段分桶:同样的现货涨幅,早盘 vs 尾盘的期权 ROI 差(theta 效应)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
_REPO = _TOOLS.parent.parent.parent
sys.path.insert(0, str(_REPO))

import rails_value_lgbm_v2 as v2
from rails_value_lgbm import entry_mask

RAW_DIR = Path("/mnt/s990/data/raw_1s/dte1_options")
HOLD = 30  # 对齐 rails 的典型持有窗


def main() -> int:
    import lightgbm as lgb

    globs = [f"QQQ_2025-{m:02d}-*.parquet" for m in (5, 6, 8, 9)]
    days = v2.load_month_days(RAW_DIR, "QQQ", globs, 2)
    print(f"days={len(days)}")

    rows = []
    for date_str, m, _t in days:
        if "s_close" not in m.columns:
            continue
        sc = m["s_close"].ffill().to_numpy(dtype=np.float64)
        mid = m["exec_call_mid"].to_numpy(dtype=np.float64)
        rv = m["rails_value"].to_numpy(dtype=np.float64)
        sb = m["session_bar"].to_numpy(dtype=np.int64)
        w = entry_mask(m).to_numpy()
        n = len(m)
        for i in range(n):
            if not w[i] or not np.isfinite(rv[i]):
                continue
            j0, j1 = i + 1, min(i + 1 + HOLD, n)
            if j1 - j0 < 10 or not np.isfinite(sc[i]) or sc[i] <= 0:
                continue
            seg = sc[j0:j1]
            if not np.all(np.isfinite(seg)):
                continue
            base = sc[i]
            rows.append({
                "date": date_str,
                "rails_value": rv[i],
                "session_bar": sb[i],
                "premium_pct": mid[i] / base if base > 0 else np.nan,  # 权利金/现货
                "fut_ret": seg[-1] / base - 1.0,
                "fut_max_up": seg.max() / base - 1.0,
                "fut_max_dn": seg.min() / base - 1.0,
                "fut_path_pos": float(np.mean(seg / base - 1.0 > 0)),  # 在水上的时间占比
                "fut_t_maxup": int(np.argmax(seg)),  # 高点出现的早晚
            })
    df = pd.DataFrame(rows).dropna()
    print(f"samples={len(df)}")

    # 1) 单变量 rank 相关
    print("\n=== rails_value 与未来现货路径的 spearman ===")
    for c in ("fut_ret", "fut_max_up", "fut_max_dn", "fut_path_pos", "fut_t_maxup"):
        rho, _ = spearmanr(df[c], df["rails_value"])
        print(f"  {c:<14} {rho:+.3f}")

    # 2) 先知路径 → rails_value 的可确定性
    feat_sets = {
        "path_only": ["fut_ret", "fut_max_up", "fut_max_dn", "fut_path_pos", "fut_t_maxup"],
        "path+time+premium": [
            "fut_ret", "fut_max_up", "fut_max_dn", "fut_path_pos", "fut_t_maxup",
            "session_bar", "premium_pct",
        ],
    }
    dates = sorted(df["date"].unique())
    split = dates[int(len(dates) * 0.7)]
    tr, te = df[df["date"] <= split], df[df["date"] > split]
    print(f"\n=== 先知现货路径 LGBM(train {len(tr)} / test {len(te)}) ===")
    out_stats = {}
    for name, cols in feat_sets.items():
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                min_child_samples=50, random_state=0, verbose=-1)
        mdl.fit(tr[cols], tr["rails_value"])
        pred = mdl.predict(te[cols])
        rho, _ = spearmanr(pred, te["rails_value"])
        ss_res = float(np.sum((te["rails_value"] - pred) ** 2))
        ss_tot = float(np.sum((te["rails_value"] - te["rails_value"].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot
        # 逐日 rank IC
        ics = []
        for d, g in te.assign(pred=pred).groupby("date"):
            if len(g) >= 30:
                r, _ = spearmanr(g["pred"], g["rails_value"])
                ics.append(r)
        print(f"  {name:<20} rank_corr={rho:+.3f}  R2={r2:.3f}  daily_IC_mean={np.mean(ics):+.3f}")
        out_stats[name] = {"rank_corr": round(rho, 4), "r2": round(r2, 4),
                           "daily_ic": round(float(np.mean(ics)), 4)}

    # 3) theta 效应:同样现货涨幅,不同时段的期权 ROI
    print("\n=== theta 效应:fut_ret 分桶 × 时段分桶 的 rails_value 中位数 ===")
    df["ret_bucket"] = pd.cut(df["fut_ret"], [-1, -0.002, -0.0005, 0.0005, 0.002, 1],
                              labels=["<-0.2%", "-0.2~-0.05%", "flat", "+0.05~0.2%", ">+0.2%"])
    df["tod_bucket"] = pd.cut(df["session_bar"], [0, 90, 180, 270, 400],
                              labels=["09:45-11", "11-12:30", "12:30-14", "14-15:30"])
    piv = df.pivot_table(values="rails_value", index="ret_bucket",
                         columns="tod_bucket", aggfunc="median", observed=True)
    print(piv.round(4).to_string())

    out = Path("New_Pro/baseline_qqq/reports/spot_path_decomposition.json")
    out.write_text(json.dumps({
        "samples": len(df),
        "oracle_path_models": out_stats,
        "theta_table": json.loads(piv.round(4).to_json()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
