"""独立验证:QQQ 现货未来方向(15/30/45/60 分钟)到底可不可预测。

不使用任何期权标签,只用 t 时刻可得的特征(现货技术特征 + 期权流特征 + 时间特征)。
训练 HistGradientBoostingRegressor 预测同 session 未来 h 分钟收益,评估:
  - Spearman IC(pred vs 真实前向收益)
  - 方向准确率(整体 / 按 |pred| 置信度 top20%/top10%)
  - val/test 逐月稳定性

硬门槛:高置信区间方向准确率稳定 >= 53% 才认为 2DTE 方向路由成立。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONS = [15, 30, 45, 60]

# t 时刻可得的因果特征(排除 label_*、exec_* 报价与原始 OHLC 水平值)
BASE_FEATURES = [
    "close_log_return", "vwap_log_return", "volume_log", "volume_ratio",
    "vwap_diff", "return_divergence", "garman_klass_vol", "bb_width",
    "adx_smooth_10", "poc_deviation",
    "options_vw_spread", "options_vw_imbalance", "options_vw_iv",
    "options_vw_delta", "options_vw_gamma", "options_vw_vega",
    "options_vw_theta", "options_pcr_volume", "options_iv_momentum",
    "options_iv_divergence", "options_gamma_accel", "options_flow_skew",
    "options_struc_atm_iv", "options_struc_skew",
    "time_session_sin", "time_session_cos", "time_session_progress",
    "trend_fit_ret_30m", "trend_fit_r2_30m",
    "trend_fit_ret_120m", "trend_fit_r2_120m",
    "day_range_pos", "drawdown_from_day_high", "drawup_from_day_low",
    "open30_ret", "open30_max_ret", "open30_peak_dd", "open30_reversal",
    "open30_range_pos", "bars_since_open30_high_norm",
    "vix_level", "vix_proxy_close",
]

DERIVED_RET_WINDOWS = [5, 15, 30, 60]


def load_stage(root: Path) -> pd.DataFrame:
    files = sorted((root / "QQQ/regular/09:30-16:00/1min").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["_day"] = ts.dt.date
    df["_month"] = ts.dt.strftime("%Y-%m")
    return df


def add_derived(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    px = pd.to_numeric(df["close"], errors="coerce")
    grp = df["_day"]
    extra = []
    for w in DERIVED_RET_WINDOWS:
        col = f"past_ret_{w}m"
        df[col] = px / px.groupby(grp).shift(w) - 1.0
        extra.append(col)
    ret1 = px.groupby(grp).pct_change()
    for w in [15, 60]:
        col = f"rvol_{w}m"
        df[col] = ret1.groupby(grp).rolling(w).std().reset_index(level=0, drop=True)
        extra.append(col)
    return df, extra


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    px = pd.to_numeric(df["close"], errors="coerce")
    for h in HORIZONS:
        df[f"fwd_ret_{h}m"] = px.groupby(df["_day"]).shift(-h) / px - 1.0
    return df


def eval_block(pred: np.ndarray, y: np.ndarray, months: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(y)
    pred, y, months = pred[m], y[m], months[m]
    out = {"n": int(len(y))}
    out["ic"] = float(spearmanr(pred, y).statistic)
    nz = y != 0
    out["acc_all"] = float(((pred > 0) == (y > 0))[nz].mean())
    for frac, tag in [(0.2, "top20"), (0.1, "top10")]:
        thr = np.quantile(np.abs(pred), 1 - frac)
        sel = (np.abs(pred) >= thr) & nz
        out[f"acc_{tag}"] = float(((pred[sel] > 0) == (y[sel] > 0)).mean())
        out[f"mean_fwd_{tag}"] = float(np.abs(y[sel]).mean())
        # 方向对时的平均收益 - 方向错时的平均损失(现货口径)
        signed = np.where(pred[sel] > 0, y[sel], -y[sel])
        out[f"signed_ret_{tag}_bp"] = float(signed.mean() * 1e4)
    per_month = {}
    for mon in np.unique(months):
        mm = (months == mon) & nz
        thr = np.quantile(np.abs(pred[months == mon]), 0.8)
        sel = mm & (np.abs(pred) >= thr)
        per_month[str(mon)] = {
            "n": int(mm.sum()),
            "acc_all": float(((pred[mm] > 0) == (y[mm] > 0)).mean()),
            "acc_top20": float(((pred[sel] > 0) == (y[sel] > 0)).mean()) if sel.sum() else None,
            "signed_ret_top20_bp": float(np.where(pred[sel] > 0, y[sel], -y[sel]).mean() * 1e4) if sel.sum() else None,
        }
    out["per_month"] = per_month
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default="/home/kingfang007/train_data/quote_features_train_dte2_ladder_spotroute_202606")
    ap.add_argument("--val-root", default="/home/kingfang007/train_data/quote_features_val_dte2_ladder_spotroute_202606")
    ap.add_argument("--test-root", default="/home/kingfang007/train_data/quote_features_test_dte2_ladder_spotroute_202606")
    ap.add_argument("--output", default="qqq_btc/results/spot_direction_validation.json")
    args = ap.parse_args()

    stages = {}
    for name, root in [("train", args.train_root), ("val", args.val_root), ("test", args.test_root)]:
        df = load_stage(Path(root))
        df, extra = add_derived(df)
        df = add_labels(df)
        stages[name] = df
        print(f"[{name}] rows={len(df)} days={df['_day'].nunique()} months={df['_month'].nunique()}")

    feats = [c for c in BASE_FEATURES if c in stages["train"].columns] + extra
    print(f"features used: {len(feats)}")

    results = {}
    for h in HORIZONS:
        ycol = f"fwd_ret_{h}m"
        tr = stages["train"].dropna(subset=[ycol])
        X_tr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        y_tr = tr[ycol].values
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        model.fit(X_tr, y_tr)
        res_h = {"train_rows": int(len(tr))}
        for name in ["val", "test"]:
            df = stages[name].dropna(subset=[ycol])
            X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            pred = model.predict(X)
            res_h[name] = eval_block(pred, df[ycol].values, df["_month"].values)
        results[f"h{h}"] = res_h
        v, t = res_h["val"], res_h["test"]
        print(f"\n=== horizon {h}m (train n={len(tr)}) ===")
        print(f"  val : IC={v['ic']:.4f} acc={v['acc_all']:.3f} top20={v['acc_top20']:.3f} top10={v['acc_top10']:.3f} signed_top20={v['signed_ret_top20_bp']:.1f}bp")
        print(f"  test: IC={t['ic']:.4f} acc={t['acc_all']:.3f} top20={t['acc_top20']:.3f} top10={t['acc_top10']:.3f} signed_top20={t['signed_ret_top20_bp']:.1f}bp")
        for name in ["val", "test"]:
            for mon, r in res_h[name]["per_month"].items():
                a20 = f"{r['acc_top20']:.3f}" if r["acc_top20"] is not None else "n/a"
                s20 = f"{r['signed_ret_top20_bp']:.1f}" if r["signed_ret_top20_bp"] is not None else "n/a"
                print(f"    {name} {mon}: acc={r['acc_all']:.3f} top20_acc={a20} top20_signed={s20}bp")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
