#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 2026-07 VIXY 5min close 全常数 → vix_level 塌缩。

根因: spnq_train_resampled/.../5min/2026-07.parquet 的 close 仅 1 个唯一值,
导致 generate_vix_level_global(5min) 的 rolling z 在数根 bar 后钉死。

做法:
  1) 用同月 1min OHLCV 按 5min resample 重写 VIXY 5min
  2) 用 2026-01..07 的 5min 序列重算 vix_level(与 feature_merge 同公式)
  3) 回写诚实 / databento quote_features_{raw,test} 的 5min vix 列
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().roots[2] if False else Path(__file__).resolve().parents[2]
VIX_BASE = Path.home() / "train_data/spnq_train_resampled/VIXY/regular/09:30-16:00"
YM = "2026-07"
FROZEN = REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"


def _resample_1m_to_5m(df1: pd.DataFrame) -> pd.DataFrame:
    d = df1.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is None:
        d["timestamp"] = d["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        d["timestamp"] = d["timestamp"].dt.tz_convert("America/New_York")
    d = d.sort_values("timestamp").set_index("timestamp")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "vwap" in d.columns:
        # volume-weighted if possible; else last
        if "volume" in d.columns:
            pv = (d["vwap"] * d["volume"]).resample("5min", label="left", closed="left").sum()
            vol = d["volume"].resample("5min", label="left", closed="left").sum()
            vwap = pv / vol.replace(0, np.nan)
        else:
            vwap = d["vwap"].resample("5min", label="left", closed="left").last()
    else:
        vwap = None
    out = d.resample("5min", label="left", closed="left").agg(
        {k: v for k, v in agg.items() if k in d.columns}
    )
    out = out.dropna(subset=["close"])
    if vwap is not None:
        out["vwap"] = vwap.reindex(out.index)
    if "transactions" in d.columns:
        out["transactions"] = d["transactions"].resample("5min", label="left", closed="left").sum()
    out = out.reset_index()
    return out


def _recompute_vix_level_5min(months: list[str]) -> pd.DataFrame:
    """与 feature_merge_option_raw.generate_vix_level_global 的 5min 分支同公式。"""
    frames = []
    for ym in months:
        fp = VIX_BASE / "5min" / f"{ym}.parquet"
        if not fp.exists():
            continue
        frames.append(pd.read_parquet(fp))
    if not frames:
        raise SystemExit("no 5min VIXY months to recompute")
    df_all = pd.concat(frames, ignore_index=True)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    if df_all["timestamp"].dt.tz is None:
        df_all["timestamp"] = df_all["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        df_all["timestamp"] = df_all["timestamp"].dt.tz_convert("America/New_York")
    df_all = df_all.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df_all = df_all.set_index("timestamp")
    df_all["vix_proxy_close"] = df_all["close"].ffill()
    res_factor = 5
    adj_long_term = int(63 * 390 / res_factor)
    long_term_ma = df_all["vix_proxy_close"].rolling(
        window=adj_long_term, min_periods=int(adj_long_term / 3)
    ).mean()
    df_all["vixy_detrended_level"] = df_all["vix_proxy_close"] / (long_term_ma + 1e-9)
    adj_macro = int(63 * 390 / res_factor)
    adj_macro_min = int(21 * 390 / res_factor)
    vixy_mean_macro = df_all["vix_proxy_close"].rolling(window=adj_macro, min_periods=adj_macro_min).mean()
    vixy_std_macro = df_all["vix_proxy_close"].rolling(window=adj_macro, min_periods=adj_macro_min).std()
    df_all["vix_z"] = (df_all["vix_proxy_close"] - vixy_mean_macro) / (vixy_std_macro + 1e-9)
    adj_intraday = int(60 / res_factor)
    adj_intraday_min = max(2, int(adj_intraday / 3))
    vixy_mean_intraday = df_all["vix_proxy_close"].rolling(
        window=adj_intraday, min_periods=adj_intraday_min
    ).mean()
    vixy_std_intraday = df_all["vix_proxy_close"].rolling(
        window=adj_intraday, min_periods=adj_intraday_min
    ).std()
    df_all["vix_level"] = (df_all["vix_proxy_close"] - vixy_mean_intraday) / (
        vixy_std_intraday + 1e-9
    )
    vixy_returns = df_all["vix_proxy_close"].pct_change()
    vixy_rolling_std_jump = vixy_returns.rolling(window=21).std()
    df_all["is_vix_jump"] = (vixy_returns.abs() > 4 * vixy_rolling_std_jump).astype(int)
    for col in ("vixy_detrended_level", "vix_z", "vix_level"):
        df_all[col] = df_all[col].ffill().fillna(0.0)
    df_all["vix_level"] = df_all["vix_level"].replace(0, np.nan).ffill().fillna(0.0)
    return df_all.reset_index()


def _patch_feature_5min(feat_path: Path, vix_july: pd.DataFrame, *, apply_frozen: bool) -> dict:
    if not feat_path.exists():
        return {"path": str(feat_path), "skipped": True}
    df = pd.read_parquet(feat_path)
    before = {
        "nunique": int(df["vix_level"].nunique()) if "vix_level" in df.columns else None,
        "std": float(df["vix_level"].std()) if "vix_level" in df.columns else None,
    }
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    v = vix_july[["timestamp", "vix_level", "vix_proxy_close", "vix_z"]].copy()
    v["timestamp"] = pd.to_datetime(v["timestamp"])
    if v["timestamp"].dt.tz is None:
        v["timestamp"] = v["timestamp"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        v["timestamp"] = v["timestamp"].dt.tz_convert("America/New_York")
    for c in ("vix_level", "vix_proxy_close", "vix_z"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        v.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    if apply_frozen and FROZEN.exists():
        from qqq_btc.common.frozen_norm import apply_frozen_norm_df

        # 只对 vix_level 做列变换时仍走整表 apply(其它列已是 raw/test 原样)
        # test 路径: 输入应为 raw patched 后再 frozen
        df = apply_frozen_norm_df(df, str(FROZEN), feature_names=["vix_level"])
    bak = feat_path.with_suffix(".parquet.bak_vixflat")
    if not bak.exists():
        shutil.copy2(feat_path, bak)
    df.to_parquet(feat_path, index=False)
    after = {"nunique": int(df["vix_level"].nunique()), "std": float(df["vix_level"].std())}
    return {"path": str(feat_path), "before": before, "after": after, "backup": str(bak)}


def main() -> None:
    p1 = VIX_BASE / "1min" / f"{YM}.parquet"
    p5 = VIX_BASE / "5min" / f"{YM}.parquet"
    if not p1.exists():
        raise SystemExit(f"missing 1min VIXY {p1}")
    df1 = pd.read_parquet(p1)
    fixed5 = _resample_1m_to_5m(df1)
    print(
        f"resampled 5min n={len(fixed5)} close_nunique={fixed5['close'].nunique()} "
        f"std={fixed5['close'].std():.4f}"
    )
    bak5 = p5.with_suffix(".parquet.bak_constclose")
    if p5.exists() and not bak5.exists():
        shutil.copy2(p5, bak5)
        print("backed up", bak5)
    # 先写 OHLCV,再全局重算 vix 列
    fixed5.to_parquet(p5, index=False)

    months = [f"2026-{m:02d}" for m in range(1, 8)]
    all5 = _recompute_vix_level_5min(months)
    july = all5[all5["timestamp"].dt.strftime("%Y-%m") == "2026-07"].copy()
    # 写回含 vix 的 5min 月文件
    cols = [
        c
        for c in [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions",
            "vix_proxy_close",
            "vix_z",
            "vix_level",
            "is_vix_jump",
            "vixy_detrended_level",
        ]
        if c in july.columns
    ]
    july[cols].to_parquet(p5, index=False)
    print(
        f"rewrote {p5} vix_nunique={july['vix_level'].nunique()} "
        f"vix_std={july['vix_level'].std():.4f}"
    )

    report = {"vixy_5min": str(p5), "vix_std": float(july["vix_level"].std()), "patches": []}
    targets = [
        (
            Path.home()
            / "train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
            False,
        ),
        (
            Path.home()
            / "train_data/july_w1_v4_databento/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
            False,
        ),
    ]
    for path, _ in targets:
        report["patches"].append(_patch_feature_5min(path, july, apply_frozen=False))

    # test = frozen(raw); 对诚实/db 从刚修好的 raw 再写 test 5min
    from qqq_btc.common.frozen_norm import apply_frozen_norm_df

    for raw_p, test_p in [
        (
            Path.home()
            / "train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
            Path.home()
            / "train_data/july_w1_v4_honest_openwin/quote_features_test/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
        ),
        (
            Path.home()
            / "train_data/july_w1_v4_databento/quote_features_raw/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
            Path.home()
            / "train_data/july_w1_v4_databento/quote_features_test/QQQ/regular/09:30-16:00/5min/2026-07.parquet",
        ),
    ]:
        if not raw_p.exists():
            continue
        test_p.parent.mkdir(parents=True, exist_ok=True)
        bak = test_p.with_suffix(".parquet.bak_vixflat")
        if test_p.exists() and not bak.exists():
            shutil.copy2(test_p, bak)
        raw_df = pd.read_parquet(raw_p)
        # test 5min: 与 rebuild 脚本一致,对整表 frozen(至少 vix_level)
        normed = apply_frozen_norm_df(raw_df, str(FROZEN), feature_names=None)
        normed.to_parquet(test_p, index=False)
        report["patches"].append(
            {
                "path": str(test_p),
                "after": {
                    "nunique": int(normed["vix_level"].nunique()),
                    "std": float(normed["vix_level"].std()),
                },
            }
        )
        print(
            f"wrote test {test_p.name} parent={test_p.parents[4].name} "
            f"vix_std={normed['vix_level'].std():.4f} nunique={normed['vix_level'].nunique()}"
        )

    out = REPO / "qqq_btc/results/july_w1_ft56_honest_signal_diff/fix_vixy_5min_july2026.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("report", out)


if __name__ == "__main__":
    main()
