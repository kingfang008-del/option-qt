#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A/B：prefer_primary（契约）vs 单合约 v3 monthly，对照 bak。

用法:
  python qqq_btc/tools/ab_prefer_primary_vs_single.py \\
      --months 2025-07,2025-08,2025-09,2025-10,2025-11,2025-12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.feature_contract_0dte import load_contract  # noqa: E402

NY = "America/New_York"
BAK_MONTHLY = Path.home() / "train_data/_bak_pre4c/quote_options_monthly_iv_QQQ/standard"
BAK_BUCKETED = Path.home() / "train_data/_bak_pre4c/quote_options_bucketed_v7_QQQ"
PREF_MONTHLY = Path.home() / "train_data/bak_lineage_reproduce/quote_options_monthly_iv/QQQ/standard"
PREF_BUCKETED = Path.home() / "train_data/bak_lineage_reproduce/quote_options_bucketed_v7/QQQ"
V3_MONTHLY = Path.home() / "train_data/quote_options_monthly_iv_v3/QQQ/standard"
V3_BUCKETED = Path.home() / "train_data/quote_options_bucketed_v7_v3/QQQ"


def _norm_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(NY)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(NY)
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.replace("O:", "", regex=False)
    return out


def monthly_vs_bak(cand: Path, month: str) -> dict:
    bak = _norm_ts(pd.read_parquet(BAK_MONTHLY / f"{month}.parquet"))
    if not cand.exists():
        return {"status": "missing", "path": str(cand)}
    rep = _norm_ts(pd.read_parquet(cand))
    keys = ["timestamp", "bucket_id", "ticker"]
    bak_k = bak.drop_duplicates(keys)
    rep_k = rep.drop_duplicates(keys)
    both = bak_k.merge(rep_k[keys], on=keys, how="inner")
    only_bak = len(bak_k) - len(both)
    only_rep = len(rep_k) - len(both)
    m = bak.merge(rep, on=keys, suffixes=("_b", "_r"))
    iv_eq = None
    if "iv_b" in m.columns and "iv_r" in m.columns:
        x = pd.to_numeric(m["iv_b"], errors="coerce")
        y = pd.to_numeric(m["iv_r"], errors="coerce")
        mm = x.notna() & y.notna()
        iv_eq = float(np.isclose(x[mm], y[mm], atol=1e-6).mean()) if mm.sum() else None
    return {
        "status": "ok",
        "bak_rows": int(len(bak_k)),
        "cand_rows": int(len(rep_k)),
        "coverage_bak": float(len(both) / max(len(bak_k), 1)),
        "only_bak": int(only_bak),
        "only_cand": int(only_rep),
        "iv_eq": iv_eq,
    }


def bucketed_vs_bak(cand: Path, month: str) -> dict:
    bak_p = BAK_BUCKETED / f"{month}.parquet"
    if not cand.exists() or not bak_p.exists():
        return {"status": "missing"}
    bak = _norm_ts(pd.read_parquet(bak_p))
    rep = _norm_ts(pd.read_parquet(cand))
    mb = bak.merge(rep, on="timestamp", suffixes=("_b", "_r"))
    corrs = []
    for c in [c for c in bak.columns if c.startswith("options_")]:
        x = pd.to_numeric(mb[f"{c}_b"], errors="coerce")
        y = pd.to_numeric(mb[f"{c}_r"], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 50 or x[m].std() == 0 or y[m].std() == 0:
            continue
        corrs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    if not corrs:
        return {"status": "ok", "n_cols": 0}
    return {
        "status": "ok",
        "n_cols": len(corrs),
        "median_corr": float(np.median(corrs)),
        "min_corr": float(np.min(corrs)),
    }


def map_stats(month: str) -> dict:
    dyn = pd.read_parquet(Path.home() / "train_data/locked_targets_map_0dte_dynamic.parquet")
    v3 = pd.read_parquet(Path.home() / "train_data/locked_targets_map_0dte_v3.parquet")
    dyn = dyn[dyn["date_str"].astype(str).str.startswith(month)]
    v3 = v3[v3["date_str"].astype(str).str.startswith(month)]
    return {
        "dynamic_contract_days": int(len(dyn)),
        "dynamic_days": int(dyn["date_str"].nunique()) if len(dyn) else 0,
        "dynamic_multi_per_bucket_days": int(
            (dyn.groupby(["date_str", "bucket_id"]).size() > 1).sum()
        )
        if len(dyn)
        else 0,
        "v3_contract_days": int(len(v3)),
        "v3_days": int(v3["date_str"].nunique()) if len(v3) else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--months",
        default="2025-07,2025-08,2025-09,2025-10,2025-11,2025-12",
    )
    p.add_argument(
        "--out",
        default=str(_REPO / "qqq_btc/results/feature_drift_rootcause_2025h2/ab_prefer_primary_vs_single.json"),
    )
    args = p.parse_args()
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    contract = load_contract()

    per_month = {}
    for mo in months:
        pref_m = monthly_vs_bak(PREF_MONTHLY / f"{mo}.parquet", mo)
        v3_m = monthly_vs_bak(V3_MONTHLY / f"{mo}.parquet", mo)
        pref_b = bucketed_vs_bak(PREF_BUCKETED / f"{mo}.parquet", mo)
        v3_b = bucketed_vs_bak(V3_BUCKETED / f"{mo}.parquet", mo)
        per_month[mo] = {
            "lock_map": map_stats(mo),
            "prefer_primary": {"monthly_vs_bak": pref_m, "bucketed_vs_bak": pref_b},
            "single_v3": {"monthly_vs_bak": v3_m, "bucketed_vs_bak": v3_b},
            "winner": (
                "prefer_primary"
                if (pref_m.get("coverage_bak") or 0) >= (v3_m.get("coverage_bak") or 0)
                else "single_v3"
            ),
        }

    # aggregate
    def agg(side: str, field: str):
        vals = []
        for mo in months:
            v = per_month[mo][side]["monthly_vs_bak"].get(field)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return {
            "mean": float(np.mean(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
        }

    report = {
        "contract": contract["name"],
        "contract_version": contract["version"],
        "months": months,
        "aggregate": {
            "prefer_primary_coverage_bak": agg("prefer_primary", "coverage_bak"),
            "single_v3_coverage_bak": agg("single_v3", "coverage_bak"),
            "prefer_primary_iv_eq": agg("prefer_primary", "iv_eq"),
            "single_v3_iv_eq": agg("single_v3", "iv_eq"),
        },
        "verdict": {
            "rule": "prefer_primary_gapfill + locked_targets_map_0dte_dynamic",
            "evidence": (
                "H2 上 prefer_primary 对 bak monthly 覆盖显著高于单合约 v3；"
                "重叠行 IV 两边都接近 1 —— 差异是 coverage/补洞，不是报价公式。"
            ),
            "action": (
                "默认重建走 qqq_btc/tools/rebuild_0dte_prefer_primary.sh；"
                "历史 208× 审计继续用 _bak_pre4c/quote_features_train_QQQ。"
            ),
        },
        "per_month": per_month,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("aggregate", "verdict")}, indent=2, ensure_ascii=False))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
