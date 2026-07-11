#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fixed-8 vs V4 标签/fill 对齐诊断。

重点验证 B 场景假说:
  - 标签: label_pipeline → day_iv + rank=0 主合约
  - 回放: eval attach_exec_quotes → 1m parquet, keep=last (无 rank 过滤)
  - fixed-8 1m 每 bucket 每分钟最多 2 合约 → 口径分叉

用法:
  python qqq_btc/tools/fixed8_label_fill_diagnostic.py
  python qqq_btc/tools/fixed8_label_fill_diagnostic.py --dates 2026-06-02,2026-06-17
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.qqq import anchor
from qqq_btc.tools.eval_test_set import attach_exec_quotes, drop_embedded_exec_columns

OUT_DIR = _REPO / "qqq_btc/results"

V4_ANCHOR = _REPO / "qqq_btc/CONFIG/anchor_qqq_0dte.json"
V8_ANCHOR = _REPO / "qqq_btc/CONFIG/anchor_qqq_0dte_v8_fixed8.json"
V4_FEAT_TEST = Path.home() / "train_data/quote_features_test/QQQ/regular/09:30-16:00/1min"
V8_FEAT_TEST = Path.home() / "train_data/quote_features_test_fixed8_v8/QQQ/regular/09:30-16:00/1min"
DYN_1M = Path("/mnt/s990/data/raw_1m/options_databento")
FIX8_1M = Path("/mnt/s990/data/raw_1m/options_databento_fixed8_corrected")

OPTION_FEATS = [
    "options_vw_iv", "options_struc_skew", "options_vw_imbalance",
    "options_vw_spread", "options_flow_skew", "options_vw_delta",
]


def _ny_ts(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("America/New_York", ambiguous="infer")
    return ts.dt.tz_convert("America/New_York")


def _quote_diff_stats(
    label_bid: pd.Series,
    label_ask: pd.Series,
    replay_bid: pd.Series,
    replay_ask: pd.Series,
) -> dict:
    m = label_bid.notna() & replay_bid.notna() & (label_bid > 0) & (replay_bid > 0)
    if not m.any():
        return {"n": 0}
    lb, la, rb, ra = label_bid[m], label_ask[m], replay_bid[m], replay_ask[m]
    mid_l = (lb + la) / 2
    mid_r = (rb + ra) / 2
    bid_rel = ((rb - lb) / lb.replace(0, np.nan)).abs()
    mid_rel = ((mid_r - mid_l) / mid_l.replace(0, np.nan)).abs()
    flip = (lb != rb) | (la != ra)
    return {
        "n": int(m.sum()),
        "bid_mismatch_pct": float(flip.mean() * 100),
        "bid_rel_mean": float(bid_rel.mean()),
        "bid_rel_p95": float(bid_rel.quantile(0.95)),
        "mid_rel_mean": float(mid_rel.mean()),
        "mid_rel_p95": float(mid_rel.quantile(0.95)),
    }


def diagnose_fill_gap(dates: list[str]) -> dict:
    """标签 day_iv 报价 vs replay 1m attach 报价 (fixed-8)。"""
    v8_cfg = anchor.load_anchor_config(V8_ANCHOR)
    rows = []
    for day in dates:
        ts_grid = pd.date_range(
            f"{day} 09:30", f"{day} 16:00", freq="1min", tz="America/New_York"
        )
        base = pd.DataFrame({"timestamp": ts_grid})
        # 标签侧: load_bucket_minute_quotes (rank=0)
        call_l = anchor.load_bucket_minute_quotes("QQQ", base["timestamp"], 2, v8_cfg, "exec_call")
        put_l = anchor.load_bucket_minute_quotes("QQQ", base["timestamp"], 0, v8_cfg, "exec_put")
        merged = base.copy()
        for q, prefix in ((call_l, "exec_call"), (put_l, "exec_put")):
            if q.empty:
                continue
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                q.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("5min"),
            )
        # 回放侧: attach_exec_quotes (keep=last, 无 rank)
        replay = attach_exec_quotes(base, FIX8_1M, "QQQ", call_bucket=2, put_bucket=0)
        for leg in ("call", "put"):
            cs = _quote_diff_stats(
                merged.get(f"exec_{leg}_bid", pd.Series(dtype=float)),
                merged.get(f"exec_{leg}_ask", pd.Series(dtype=float)),
                replay.get(f"exec_{leg}_bid", pd.Series(dtype=float)),
                replay.get(f"exec_{leg}_ask", pd.Series(dtype=float)),
            )
            cs["date"] = day
            cs["leg"] = leg
            rows.append(cs)
    df = pd.DataFrame(rows)
    summary = {}
    for leg in ("call", "put"):
        sub = df[df["leg"] == leg]
        if sub.empty:
            continue
        summary[leg] = {
            "days": len(sub),
            "avg_bid_mismatch_pct": float(sub["bid_mismatch_pct"].mean()),
            "avg_mid_rel_mean": float(sub["mid_rel_mean"].mean()),
            "max_mid_rel_p95": float(sub["mid_rel_p95"].max()),
        }
    return {"per_day": rows, "summary": summary}


def diagnose_1m_structure(dates: list[str]) -> list[dict]:
    """fixed-8 vs dynamic 1m: 每 bucket 每分钟合约数。"""
    out = []
    for day in dates:
        for tag, root in (("dynamic", DYN_1M), ("fixed8", FIX8_1M)):
            fp = root / "QQQ" / f"QQQ_{day}.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp)
            for b in (0, 2):
                sub = df[df["bucket_id"] == b]
                if sub.empty:
                    continue
                per_min = sub.groupby("timestamp").size()
                out.append({
                    "date": day,
                    "source": tag,
                    "bucket": int(b),
                    "max_contracts_per_min": int(per_min.max()),
                    "pct_minutes_multi_contract": float((per_min > 1).mean() * 100),
                    "n_minutes": int(per_min.size),
                })
    return out


def diagnose_parquet_labels(months: list[str]) -> dict:
    """V4 test parquet vs V8 test parquet 标签列 (同 timestamp 对齐)。"""
    parts = []
    for month in months:
        f4 = V4_FEAT_TEST / f"{month}.parquet"
        f8 = V8_FEAT_TEST / f"{month}.parquet"
        if not f4.exists() or not f8.exists():
            continue
        d4 = pd.read_parquet(f4, columns=["timestamp", "label_return_fwd_net"])
        d8 = pd.read_parquet(f8, columns=["timestamp", "label_return_fwd_net"])
        d4["timestamp"] = _ny_ts(d4["timestamp"])
        d8["timestamp"] = _ny_ts(d8["timestamp"])
        m = d4.merge(d8, on="timestamp", suffixes=("_v4", "_v8"))
        m["month"] = month
        parts.append(m)
    if not parts:
        return {}
    all_df = pd.concat(parts, ignore_index=True)
    v4 = all_df["label_return_fwd_net_v4"]
    v8 = all_df["label_return_fwd_net_v8"]
    both = v4.notna() & v8.notna()
    corr = float(v4[both].corr(v8[both])) if both.sum() > 50 else float("nan")
    return {
        "n_rows": int(len(all_df)),
        "v4_label_std": float(v4.std()),
        "v8_label_std": float(v8.std()),
        "v4_abs_ge_0015_pct": float((v4.abs() >= 0.015).mean() * 100),
        "v8_abs_ge_0015_pct": float((v8.abs() >= 0.015).mean() * 100),
        "corr_same_ts": corr,
        "mean_abs_diff": float((v4 - v8).abs().mean()),
    }


def diagnose_feature_drift(months: list[str]) -> list[dict]:
    rows = []
    for month in months:
        f4 = V4_FEAT_TEST / f"{month}.parquet"
        f8 = V8_FEAT_TEST / f"{month}.parquet"
        if not f4.exists() or not f8.exists():
            continue
        d4 = pd.read_parquet(f4)
        d8 = pd.read_parquet(f8)
        for c in OPTION_FEATS:
            if c not in d4.columns or c not in d8.columns:
                continue
            a = pd.to_numeric(d4[c], errors="coerce").dropna()
            b = pd.to_numeric(d8[c], errors="coerce").dropna()
            n = min(len(a), len(b))
            if n < 100:
                continue
            corr = float(np.corrcoef(a.values[:n], b.values[:n])[0, 1])
            rows.append({
                "month": month,
                "feature": c,
                "v4_mean": float(a.mean()),
                "v8_mean": float(b.mean()),
                "mean_diff": float(b.mean() - a.mean()),
                "corr": corr,
            })
    return rows


def diagnose_b_trades() -> dict:
    """B 场景 infer 交易: edge 与 label 方向。"""
    infer_path = Path("/tmp/qqq_btc_abcd_B_v4_on_v8/test_infer.parquet")
    trades_path = Path("/tmp/qqq_btc_abcd_B_v4_on_v8/replay_trades.parquet")
    if not infer_path.exists():
        return {"status": "missing_infer"}
    df = pd.read_parquet(infer_path)
    ne = pd.to_numeric(df["net_edge"], errors="coerce")
    lbl = pd.to_numeric(df.get("label_return_fwd_net"), errors="coerce")
    out = {
        "n_bars": len(df),
        "edge_ge_003_pct": float((ne >= 0.03).mean() * 100),
        "edge_mean": float(ne.mean()),
        "label_std": float(lbl.std()) if lbl.notna().any() else None,
        "ic_edge_vs_label": float(ne.corr(lbl, method="spearman")) if lbl.notna().sum() > 50 else None,
    }
    if trades_path.exists():
        tr = pd.read_parquet(trades_path)
        out["n_trades"] = len(tr)
        if len(tr) and "net_return" in tr.columns:
            out["trade_hit_rate"] = float((tr["net_return"] > 0).mean())
            out["trade_avg_return"] = float(tr["net_return"].mean())
            if "leg" in tr.columns:
                out["trades_by_leg"] = tr["leg"].value_counts().to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="fixed-8 label/fill diagnostic")
    parser.add_argument("--dates", default="2026-06-02,2026-06-17,2026-04-15")
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    args = parser.parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    report = {
        "tag": "fixed8_label_fill_diagnostic",
        "hypothesis": "replay attach_exec_quotes 未做 rank=0 主合约过滤, fixed-8 1m 每 bucket 双合约导致 fill 与标签分叉",
    }

    print("=== [1] 1m 结构: dynamic vs fixed-8 ===")
    struct = diagnose_1m_structure(dates)
    report["1m_structure"] = struct
    for r in struct:
        print(f"  {r['date']} {r['source']} bucket{r['bucket']}: "
              f"max={r['max_contracts_per_min']}/min multi={r['pct_minutes_multi_contract']:.1f}%")

    print("\n=== [2] 标签 day_iv vs replay 1m 报价差 (fixed-8) ===")
    fill = diagnose_fill_gap(dates)
    report["fill_gap"] = fill
    for leg, s in fill.get("summary", {}).items():
        print(f"  {leg}: bid mismatch={s['avg_bid_mismatch_pct']:.1f}% "
              f"mid_rel_mean={s['avg_mid_rel_mean']:.4f} mid_rel_p95_max={s['max_mid_rel_p95']:.4f}")

    print("\n=== [3] test parquet 标签: V4 vs V8 ===")
    lbl = diagnose_parquet_labels(months)
    report["parquet_labels"] = lbl
    if lbl:
        print(f"  v4 std={lbl['v4_label_std']:.4f} v8 std={lbl['v8_label_std']:.4f} "
              f"corr={lbl['corr_same_ts']:.4f} |net|>=1.5%: v4={lbl['v4_abs_ge_0015_pct']:.1f}% v8={lbl['v8_abs_ge_0015_pct']:.1f}%")

    print("\n=== [4] test 特征漂移 (V4 vs V8 parquet) ===")
    drift = diagnose_feature_drift(months)
    report["feature_drift"] = drift
    for r in drift:
        print(f"  {r['month']} {r['feature']}: corr={r['corr']:.3f} diff={r['mean_diff']:+.4f}")

    print("\n=== [5] B 场景 (V4 ckpt + V8 特征) ===")
    b = diagnose_b_trades()
    report["b_scenario"] = b
    print(f"  {b}")

    # verdict
    verdicts = []
    multi = [r for r in struct if r["source"] == "fixed8" and r["max_contracts_per_min"] > 1]
    if multi:
        verdicts.append("CONFIRMED: fixed-8 1m 存在同 bucket 多合约,dynamic 为 1 合约/min")
    if fill.get("summary"):
        call_mm = fill["summary"].get("call", {}).get("avg_bid_mismatch_pct", 0)
        if call_mm > 10:
            verdicts.append(f"CONFIRMED: 标签 vs replay CALL 报价不一致 {call_mm:.0f}% 分钟")
    if lbl.get("v4_label_std", 1) < 0.01 and lbl.get("v8_label_std", 0) > 0.1:
        verdicts.append("CONFIRMED: V4 test parquet 标签已失效(std~0), V8 标签正常 — LMDB/训练应用 V8 标签链")
    if lbl.get("corr_same_ts") is not None and not np.isnan(lbl["corr_same_ts"]) and abs(lbl["corr_same_ts"]) < 0.1:
        verdicts.append("CONFIRMED: 同 timestamp 标签 V4/V8 几乎无关 — 非同一 fill 口径")
    report["verdicts"] = verdicts

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "fixed8_label_fill_diagnostic.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    drift_csv = OUT_DIR / "fixed8_label_fill_feature_drift.csv"
    pd.DataFrame(drift).to_csv(drift_csv, index=False)
    fill_csv = OUT_DIR / "fixed8_label_fill_gap.csv"
    pd.DataFrame(fill.get("per_day", [])).to_csv(fill_csv, index=False)

    print("\n=== VERDICTS ===")
    for v in verdicts:
        print(f"  • {v}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
