#!/usr/bin/env python3
"""补写缺失/塌缩的 VIXY 波动特征到 quote_features。

典型场景（2026-07-13）：
  1) resampled VIXY 只有 OHLCV，未重算 vix_level → put_gate 全 0/NaN
  2) 实验目录 quote_features_{raw,test} 继承了坏的 vix 列

本脚本：
  1. 用足够历史窗口重算 VIXY resampled 的 vix_proxy_close / vix_z / vix_level
  2. 按 timestamp merge_asof 回写目标特征 parquet（默认备份 .bak_vixpatch）
  3. 可选对 test 特征只做 vix_level frozen_norm

用法:
  # 重算 2026-07，并补到 jul13 实验 raw/test
  python qqq_btc/tools/patch_vixy_features.py \\
    --ym 2026-07 \\
    --feature-root ~/train_data/jul13_v4_old_lock_massive/quote_features_raw \\
    --feature-root ~/train_data/jul13_v4_old_lock_massive/quote_features_test \\
    --apply-frozen-on-test

  # 仅检查是否塌缩
  python qqq_btc/tools/patch_vixy_features.py --ym 2026-07 --check-only \\
    --feature-root ~/train_data/jul13_v4_old_lock_massive/quote_features_raw

  # 只补指定日
  python qqq_btc/tools/patch_vixy_features.py --ym 2026-07 --day 2026-07-13 \\
    --feature-root ~/train_data/.../quote_features_raw
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_VIX_BASE = Path.home() / "train_data/spnq_train_resampled/VIXY/regular/09:30-16:00"
DEFAULT_FROZEN = REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"
DEFAULT_CFG = REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"
VIX_COLS = ("vix_proxy_close", "vix_z", "vix_level", "vixy_detrended_level", "is_vix_jump")


def _ny_ts(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("America/New_York", ambiguous="infer")
    return ts.dt.tz_convert("America/New_York")


def _res_factor(res: str) -> int:
    if "min" in res:
        try:
            return max(1, int(res.replace("min", "")))
        except ValueError:
            return 1
    return 1


def recompute_vixy_month(
    *,
    ym: str,
    res: str = "1min",
    vix_base: Path = DEFAULT_VIX_BASE,
    history_months: int = 7,
) -> pd.DataFrame:
    """重算含 ym 的全局滚动窗口，回写该月 VIXY resampled，并返回该月帧。"""
    res_dir = vix_base / res
    if not res_dir.is_dir():
        raise FileNotFoundError(f"missing VIXY dir: {res_dir}")

    y, m = [int(x) for x in ym.split("-")]
    months: list[str] = []
    yy, mm = y, m
    for _ in range(max(1, history_months)):
        months.append(f"{yy:04d}-{mm:02d}")
        mm -= 1
        if mm <= 0:
            mm = 12
            yy -= 1
    months = list(reversed(months))

    frames = []
    for month in months:
        fp = res_dir / f"{month}.parquet"
        if not fp.exists():
            continue
        frames.append(pd.read_parquet(fp))
    if not frames:
        raise FileNotFoundError(f"no VIXY {res} months around {ym} under {res_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["timestamp"] = _ny_ts(df_all["timestamp"])
    df_all = df_all.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df_all = df_all.set_index("timestamp")
    if "close" not in df_all.columns:
        raise ValueError(f"{res} VIXY missing close")
    df_all["vix_proxy_close"] = pd.to_numeric(df_all["close"], errors="coerce").ffill()

    rf = _res_factor(res)
    adj_long = int(63 * 390 / rf)
    long_ma = df_all["vix_proxy_close"].rolling(
        window=adj_long, min_periods=max(1, int(adj_long / 3))
    ).mean()
    df_all["vixy_detrended_level"] = df_all["vix_proxy_close"] / (long_ma + 1e-9)

    adj_macro = int(63 * 390 / rf)
    adj_macro_min = int(21 * 390 / rf)
    mu = df_all["vix_proxy_close"].rolling(window=adj_macro, min_periods=adj_macro_min).mean()
    sd = df_all["vix_proxy_close"].rolling(window=adj_macro, min_periods=adj_macro_min).std()
    df_all["vix_z"] = (df_all["vix_proxy_close"] - mu) / (sd + 1e-9)

    adj_intra = int(60 / rf) if rf < 60 else 3
    adj_intra_min = max(2, int(adj_intra / 3))
    mu_i = df_all["vix_proxy_close"].rolling(
        window=adj_intra, min_periods=adj_intra_min
    ).mean()
    sd_i = df_all["vix_proxy_close"].rolling(
        window=adj_intra, min_periods=adj_intra_min
    ).std()
    df_all["vix_level"] = (df_all["vix_proxy_close"] - mu_i) / (sd_i + 1e-9)

    rets = df_all["vix_proxy_close"].pct_change()
    jump_std = rets.rolling(window=21).std()
    df_all["is_vix_jump"] = (rets.abs() > 4 * jump_std).astype(int)

    for col in ("vixy_detrended_level", "vix_z", "vix_level"):
        df_all[col] = df_all[col].ffill().fillna(0.0)
    df_all["vix_level"] = df_all["vix_level"].replace(0, np.nan).ffill().fillna(0.0)

    out = df_all.reset_index()
    out["year_month"] = out["timestamp"].dt.strftime("%Y-%m")
    month_df = out.loc[out["year_month"] == ym].copy()
    if month_df.empty:
        raise RuntimeError(f"recomputed VIXY has no rows for {ym}")

    keep = [
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
            *VIX_COLS,
        ]
        if c in month_df.columns
    ]
    target = res_dir / f"{ym}.parquet"
    bak = target.with_suffix(".parquet.bak_vixpatch")
    if target.exists() and not bak.exists():
        shutil.copy2(target, bak)
    month_df[keep].to_parquet(target, index=False)
    return month_df


def diagnose_vix(df: pd.DataFrame, *, day: Optional[str] = None) -> dict[str, Any]:
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = _ny_ts(work["timestamp"])
        if day:
            work = work.loc[work["timestamp"].dt.strftime("%Y-%m-%d") == day]
    out: dict[str, Any] = {"rows": int(len(work))}
    for col in ("vix_level", "vix_proxy_close", "vix_z"):
        if col not in work.columns:
            out[col] = {"missing": True}
            continue
        s = pd.to_numeric(work[col], errors="coerce")
        out[col] = {
            "missing": False,
            "nunique": int(s.nunique(dropna=True)),
            "nan_frac": float(s.isna().mean()) if len(s) else 1.0,
            "zero_frac": float((s.fillna(0) == 0).mean()) if len(s) else 1.0,
            "std": float(s.std(ddof=0)) if len(s) else 0.0,
            "mean": float(s.mean()) if len(s) else 0.0,
        }
    return out


def patch_feature_file(
    path: Path,
    vix_month: pd.DataFrame,
    *,
    day: Optional[str] = None,
    apply_frozen: bool = False,
    frozen_path: Path = DEFAULT_FROZEN,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "skipped": True, "reason": "missing"}
    df = pd.read_parquet(path)
    before = diagnose_vix(df, day=day)
    df["timestamp"] = _ny_ts(df["timestamp"])
    v = vix_month[["timestamp", *[c for c in VIX_COLS if c in vix_month.columns]]].copy()
    v["timestamp"] = _ny_ts(v["timestamp"])
    v = v.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    drop_cols = [c for c in VIX_COLS if c in df.columns]
    base = df.drop(columns=drop_cols, errors="ignore").sort_values("timestamp")
    merged = pd.merge_asof(base, v, on="timestamp", direction="backward")
    if day:
        mask = merged["timestamp"].dt.strftime("%Y-%m-%d") == day
        # 仅替换指定日；其余日保留原列（若原先不存在则用 merge 结果）
        for c in VIX_COLS:
            if c not in merged.columns:
                continue
            if c in df.columns:
                merged.loc[~mask, c] = df.set_index(
                    _ny_ts(df["timestamp"])
                ).reindex(merged.loc[~mask, "timestamp"])[c].to_numpy()

    if apply_frozen and frozen_path.exists() and "vix_level" in merged.columns:
        from qqq_btc.common.frozen_norm import apply_frozen_norm_df

        merged = apply_frozen_norm_df(
            merged, str(frozen_path), feature_names=["vix_level"]
        )

    after = diagnose_vix(merged, day=day)
    report = {
        "path": str(path),
        "before": before,
        "after": after,
        "apply_frozen": bool(apply_frozen and frozen_path.exists()),
        "day": day,
    }
    if dry_run:
        report["dry_run"] = True
        return report

    bak = path.with_suffix(path.suffix + ".bak_vixpatch")
    if not bak.exists():
        shutil.copy2(path, bak)
        report["backup"] = str(bak)
    merged.to_parquet(path, index=False)
    return report


def _iter_month_parquets(root: Path, ym: str, res: str) -> Iterable[Path]:
    # 兼容 .../QQQ/regular/09:30-16:00/{res}/{ym}.parquet 与直接给到该文件
    if root.is_file():
        yield root
        return
    direct = root / res / f"{ym}.parquet"
    if direct.exists():
        yield direct
        return
    for p in sorted(root.rglob(f"{ym}.parquet")):
        if f"/{res}/" in str(p).replace("\\", "/") or p.parent.name == res:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ym", required=True, help="YYYY-MM")
    ap.add_argument("--day", default=None, help="可选 YYYY-MM-DD，只补该日")
    ap.add_argument("--res", default="1min", choices=("1min", "5min"))
    ap.add_argument("--vix-base", type=Path, default=DEFAULT_VIX_BASE)
    ap.add_argument("--history-months", type=int, default=7)
    ap.add_argument(
        "--feature-root",
        action="append",
        default=[],
        help="可重复；特征根目录或具体 parquet",
    )
    ap.add_argument(
        "--apply-frozen-on-test",
        action="store_true",
        help="路径名含 quote_features_test 时对 vix_level 应用 frozen_norm",
    )
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {"ym": args.ym, "res": args.res, "day": args.day, "patches": []}

    if args.check_only:
        for root in args.feature_root:
            for path in _iter_month_parquets(Path(root).expanduser(), args.ym, args.res):
                df = pd.read_parquet(path)
                report["patches"].append(
                    {"path": str(path), "diag": diagnose_vix(df, day=args.day)}
                )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.out_json:
            args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return

    month_df = recompute_vixy_month(
        ym=args.ym,
        res=args.res,
        vix_base=args.vix_base.expanduser(),
        history_months=args.history_months,
    )
    report["vixy"] = {
        "rows": int(len(month_df)),
        "vix_level_nunique": int(month_df["vix_level"].nunique()),
        "vix_level_std": float(month_df["vix_level"].std()),
        "vix_proxy_mean": float(month_df["vix_proxy_close"].mean()),
    }
    print(
        f"[vixy] {args.ym} {args.res}: rows={report['vixy']['rows']} "
        f"vix_nunique={report['vixy']['vix_level_nunique']} "
        f"std={report['vixy']['vix_level_std']:.4f}"
    )

    for root in args.feature_root:
        root_p = Path(root).expanduser()
        for path in _iter_month_parquets(root_p, args.ym, args.res):
            use_frozen = args.apply_frozen_on_test and "quote_features_test" in str(path)
            one = patch_feature_file(
                path,
                month_df,
                day=args.day,
                apply_frozen=use_frozen,
                frozen_path=args.frozen.expanduser(),
                dry_run=args.dry_run,
            )
            report["patches"].append(one)
            b = one.get("before", {}).get("vix_level", {})
            a = one.get("after", {}).get("vix_level", {})
            print(
                f"[patch] {path}: vix_nunique {b.get('nunique')}→{a.get('nunique')} "
                f"std {b.get('std')}→{a.get('std')}"
            )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
        print("wrote", args.out_json)
    else:
        print(text)


if __name__ == "__main__":
    main()
