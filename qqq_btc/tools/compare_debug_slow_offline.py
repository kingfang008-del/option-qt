#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step-1 特征门控：PG debug_slow vs 离线 quote_features（归一化后）。

实盘/流式 FCS 把归一化 slow 特征写入 debug_slow；离线 replay 用
quote_features_test（rolling_norm 后）。开仓/平仓对拍之前，先过本门控。

用法:
  python qqq_btc/tools/compare_debug_slow_offline.py \\
      --dates 2026-07-01,2026-07-02 \\
      --offline ~/train_data/july_w1_v4_experiment/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet \\
      --out qqq_btc/results/july_w1_ft56_4c_stream_rolling/feat_parity_step1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

NY = "America/New_York"
DEFAULT_PG = "dbname=quant_trade user=postgres password=postgres host=localhost port=5432"

# FCS debug_slow 常缺 / SE 本地补算的列，不纳入硬门禁
SKIP_FEATURES = {
    "trend_fit_ret_30m",
    "trend_fit_r2_30m",
    "trend_fit_ret_120m",
    "trend_fit_r2_120m",
    "trend_strength_30m",
    "spot_range_30m",
    "open30_ret",
    "open30_high",
    "open30_low",
    "bars_since_open30_high_norm",
    "bars_since_open30_low_norm",
    "time_session_sin",
    "time_session_cos",
    "hour",
    "minute",
    "day_of_week",
    # 离线 July raw 全日恒 0，非流式对齐问题
    "vix_level",
    # 离线 poc 偶发 1e11 量级脏值，5m 路径另案
    "poc_deviation",
    # Deep Warmup / 开盘前几根 volume 基线与离线月冷启动不一致，非 options stamp 问题
    "volume_ratio",
}


def _load_offline(path: Path, dates: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    day_set = {pd.Timestamp(d).date() for d in dates}
    df = df[df["timestamp"].dt.date.isin(day_set)].copy()
    df["ts"] = df["timestamp"].map(lambda t: float(pd.Timestamp(t).timestamp()))
    return df.sort_values("ts")


def _load_debug_slow(dates: list[str], symbol: str, pg_url: str) -> pd.DataFrame:
    import psycopg2

    frames = []
    conn = psycopg2.connect(pg_url)
    try:
        for d in dates:
            ymd = d.replace("-", "")
            part = f"debug_slow_{ymd}"
            with conn.cursor() as c:
                c.execute("SELECT to_regclass(%s)", (f"public.{part}",))
                if c.fetchone()[0] is None:
                    print(f"[warn] missing partition {part}")
                    continue
                c.execute(
                    f"SELECT * FROM {part} WHERE symbol=%s ORDER BY ts",
                    (symbol,),
                )
                cols = [d[0] for d in c.description]
                rows = c.fetchall()
            if not rows:
                print(f"[warn] empty {part} symbol={symbol}")
                continue
            frames.append(pd.DataFrame(rows, columns=cols))
    finally:
        conn.close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _common_feat_cols(offline: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    skip = {"ts", "symbol", "timestamp", "created_at", "write_wall_ts", "write_wall_at", "source_ts"}
    skip |= SKIP_FEATURES
    cols = []
    for c in offline.columns:
        if c in skip or c.startswith("label_") or c.startswith("exec_"):
            continue
        if c not in live.columns:
            continue
        if not np.issubdtype(offline[c].dtype, np.number) and offline[c].dtype != object:
            continue
        cols.append(c)
    return sorted(cols)


def compare_day(
    offline: pd.DataFrame,
    live: pd.DataFrame,
    *,
    date: str,
    feats: list[str],
    med_tol: float,
    corr_min: float,
    ts_shift_sec: int = 0,
) -> dict:
    day = pd.Timestamp(date).date()
    off = offline[offline["timestamp"].dt.date == day].copy()
    # live ts → NY date
    live = live.copy()
    live["ny_date"] = pd.to_datetime(live["ts"], unit="s", utc=True).dt.tz_convert(NY).dt.date
    lv = live[live["ny_date"] == day].copy()
    if off.empty or lv.empty:
        return {
            "date": date,
            "pass": False,
            "reason": f"empty offline={len(off)} live={len(lv)}",
            "n_matched": 0,
        }

    # align on rounded second ts
    # FCS debug_slow.ts = alpha_label_ts（分钟起点）；离线 feature.timestamp = 结束标签
    # → 同一根 bar: live.ts + 60 == offline.ts。stock/options 都应在此口径下对齐；
    # 若 options 系统性偏一分钟，优先查 pitcher day_iv 是否误用 floor 而非 end-label。
    off["ts_i"] = off["ts"].round().astype(np.int64)
    lv["ts_i"] = (pd.to_numeric(lv["ts"], errors="coerce") + float(ts_shift_sec)).round().astype(np.int64)
    m = off.merge(lv, on="ts_i", suffixes=("_off", "_live"), how="inner")
    if m.empty:
        return {
            "date": date,
            "pass": False,
            "reason": f"no timestamp overlap (ts_shift_sec={ts_shift_sec})",
            "n_matched": 0,
            "ts_shift_sec": ts_shift_sec,
        }

    col_reports = []
    hard_fail = []
    for feat in feats:
        a = pd.to_numeric(m[f"{feat}_off"], errors="coerce")
        b = pd.to_numeric(m[f"{feat}_live"], errors="coerce")
        mask = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n < 10:
            col_reports.append(
                {"feature": feat, "n": n, "pass": False, "note": "too_few_rows"}
            )
            hard_fail.append(feat)
            continue
        aa, bb = a[mask].to_numpy(dtype=float), b[mask].to_numpy(dtype=float)
        err = np.abs(aa - bb)
        med = float(np.median(err))
        p90 = float(np.quantile(err, 0.90))
        p99 = float(np.quantile(err, 0.99))
        mx = float(np.max(err))
        corr = float(np.corrcoef(aa, bb)[0, 1]) if np.std(aa) > 1e-12 and np.std(bb) > 1e-12 else None
        # 主判据：中位误差。corr 对 1～3 根尖峰极敏感；若主体已贴合(p90<=tol)
        # 则用去掉 top1% 误差后的稳健 corr 再判一次。
        ok = med <= med_tol
        robust_corr = corr
        if ok and corr is not None and corr < corr_min:
            if p90 <= med_tol:
                keep = err <= np.quantile(err, 0.99)
                if int(keep.sum()) >= 10 and np.std(aa[keep]) > 1e-12 and np.std(bb[keep]) > 1e-12:
                    robust_corr = float(np.corrcoef(aa[keep], bb[keep])[0, 1])
                ok = robust_corr is None or robust_corr >= corr_min
            else:
                ok = False
        elif corr is not None and corr < corr_min:
            ok = False
        if not ok:
            hard_fail.append(feat)
        col_reports.append(
            {
                "feature": feat,
                "n": n,
                "med_abs_err": med,
                "p90_abs_err": p90,
                "p99_abs_err": p99,
                "max_abs_err": mx,
                "corr": corr,
                "robust_corr": robust_corr,
                "pass": ok,
            }
        )

    col_reports.sort(key=lambda r: (r.get("pass", False), -float(r.get("med_abs_err") or 0)))
    n_pass = sum(1 for r in col_reports if r.get("pass"))
    return {
        "date": date,
        "ts_shift_sec": int(ts_shift_sec),
        "n_matched": int(len(m)),
        "n_feats": len(col_reports),
        "n_pass": n_pass,
        "pass_rate": float(n_pass / len(col_reports)) if col_reports else 0.0,
        "pass": len(hard_fail) == 0,
        "failed_features": hard_fail[:30],
        "worst": col_reports[:15],
        "columns": col_reports,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Step-1: debug_slow vs offline feature gate")
    ap.add_argument("--dates", required=True, help="comma YYYY-MM-DD")
    ap.add_argument(
        "--offline",
        default=str(
            Path.home()
            / "train_data/july_w1_v4_experiment/quote_features_test/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
        ),
    )
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--pg-url", default=os.environ.get("PG_DB_URL", DEFAULT_PG))
    ap.add_argument("--med-tol", type=float, default=0.05, help="median |err| hard gate")
    ap.add_argument("--corr-min", type=float, default=0.95)
    ap.add_argument(
        "--ts-shift-sec",
        type=int,
        default=60,
        help="live.ts + shift 再与 offline timestamp 对齐(FCS 分钟起点 vs 离线常 +60s)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    offline = _load_offline(Path(args.offline).expanduser(), dates)
    live = _load_debug_slow(dates, args.symbol, args.pg_url)
    if offline.empty:
        print("ERROR: offline empty")
        return 2
    if live.empty:
        print("ERROR: debug_slow empty — run FCS stream first")
        return 2

    feats = _common_feat_cols(offline, live)
    print(
        f"offline rows={len(offline)} live rows={len(live)} common_feats={len(feats)} "
        f"ts_shift_sec={args.ts_shift_sec}"
    )

    by_day = []
    for d in dates:
        rep = compare_day(
            offline,
            live,
            date=d,
            feats=feats,
            med_tol=args.med_tol,
            corr_min=args.corr_min,
            ts_shift_sec=args.ts_shift_sec,
        )
        by_day.append(rep)
        status = "PASS" if rep.get("pass") else "FAIL"
        print(
            f"\n=== {d} [{status}] matched={rep.get('n_matched')} "
            f"pass_rate={rep.get('pass_rate', 0):.1%} shift={rep.get('ts_shift_sec')} ==="
        )
        if rep.get("reason"):
            print(f"  reason: {rep['reason']}")
        for w in rep.get("worst") or []:
            if w.get("pass"):
                continue
            print(
                f"  FAIL {w['feature']:28s} med={w.get('med_abs_err')} "
                f"max={w.get('max_abs_err')} corr={w.get('corr')}"
            )
        fails = rep.get("failed_features") or []
        if fails:
            print(f"  failed({len(fails)}): {fails[:12]}")

    overall = all(r.get("pass") for r in by_day)
    summary = {
        "step": 1,
        "gate": "debug_slow_vs_offline_normed_features",
        "offline": str(Path(args.offline).expanduser()),
        "symbol": args.symbol,
        "med_tol": args.med_tol,
        "corr_min": args.corr_min,
        "ts_shift_sec": args.ts_shift_sec,
        "n_feats": len(feats),
        "overall_pass": overall,
        "by_day": [{k: v for k, v in r.items() if k != "columns"} for r in by_day],
        "note": "Step-2 entry/exit 仅在 overall_pass=true 后进行",
    }
    print(f"\n=== STEP-1 OVERALL: {'PASS' if overall else 'FAIL'} ===")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        # full detail
        full = dict(summary)
        full["by_day_full"] = by_day
        out.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str))
        print(f"wrote {out}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
