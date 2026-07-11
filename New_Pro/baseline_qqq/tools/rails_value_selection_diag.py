#!/usr/bin/env python3
"""
诊断:LGBM 高分 bar 的真实 rails_value 条件分布。

回答「IC 达标但 replay 亏钱」的原因:
  - 若选中 bar 的平均 rails_value 为正 → 执行/门控层损耗问题
  - 若为负 → 选择本身错了(结构性误差:模型追高/追波动),IC 均值是误导指标
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

import rails_value_lgbm as rv


def main() -> int:
    import lightgbm as lgb

    raw_dir = Path("/mnt/s990/data/raw_1s/dte1_options")
    train_globs = ["QQQ_2025-01-*.parquet", "QQQ_2025-02-*.parquet", "QQQ_2025-03-*.parquet", "QQQ_2025-04-*.parquet"]
    val_globs = ["QQQ_2025-05-*.parquet"]
    test_globs = ["QQQ_2025-06-*.parquet"]

    train_days = rv.load_month_days(raw_dir, "QQQ", train_globs, 2)
    val_days = rv.load_month_days(raw_dir, "QQQ", val_globs, 2)
    test_days = rv.load_month_days(raw_dir, "QQQ", test_globs, 2)

    Xtr, ytr, _ = rv.to_xy(train_days, label_top=0.10)
    Xva, yva, _ = rv.to_xy(val_days, label_top=0.10)

    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=600, learning_rate=0.03,
        num_leaves=63, min_child_samples=80, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    rows = []
    for date_str, minute, _t in test_days:
        w = rv.entry_mask(minute).to_numpy()
        pred = model.predict(minute[rv.FEATURES].to_numpy(dtype=np.float64))
        sig = rv.causal_topk_signal(pred, w, 0.02)
        sel = sig > 0
        rvv = minute["rails_value"].to_numpy()
        ok = w & np.isfinite(rvv)
        day_mean = float(np.nanmean(rvv[ok])) if ok.any() else np.nan
        sel_ok = sel & np.isfinite(rvv)
        # oracle top10% bars(全知)作对照
        thr = np.nanquantile(rvv[ok], 0.90) if ok.any() else np.nan
        rows.append({
            "date": date_str,
            "n_sel": int(sel_ok.sum()),
            "sel_rails_mean": float(np.nanmean(rvv[sel_ok])) if sel_ok.any() else np.nan,
            "sel_in_oracle_top10_frac": float(np.nanmean(rvv[sel_ok] >= thr)) if sel_ok.any() else np.nan,
            "day_rails_mean": day_mean,
            "sel_sbar_mean": float(minute.loc[sel_ok, "session_bar"].mean()) if sel_ok.any() else np.nan,
            "sel_vol10_mean": float(minute.loc[sel_ok, "vol_10m"].mean()) if sel_ok.any() else np.nan,
            "day_vol10_mean": float(minute.loc[ok, "vol_10m"].mean()),
            "sel_ret15_mean": float(minute.loc[sel_ok, "ret_15m"].mean()) if sel_ok.any() else np.nan,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== 汇总 ===")
    print(f"选中bar真实rails_value均值 : {df['sel_rails_mean'].mean():+.4f}")
    print(f"全日rails_value均值        : {df['day_rails_mean'].mean():+.4f}")
    print(f"选中bar命中oracle top10率  : {df['sel_in_oracle_top10_frac'].mean():.1%} (随机=10%)")
    print(f"选中bar vol10 / 全日 vol10 : {df['sel_vol10_mean'].mean() / df['day_vol10_mean'].mean():.2f}x")
    print(f"选中bar 前15m动量均值      : {df['sel_ret15_mean'].mean():+.4f}")
    out = Path("New_Pro/baseline_qqq/reports/qqq_1dte_rails_value_selection_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
