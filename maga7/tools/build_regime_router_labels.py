#!/usr/bin/env python3
"""Build oracle day_type labels from baseline hard-day scan features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _day_type(r: pd.Series) -> str:
    if r.qqq_above_open_1030 and float(r.qqq_bounce_lod_1030) >= 0.005 and int(r.n_dn) > 0 and float(r.sum_ret_dn) < 0:
        return "rebound_trap_dn"
    if int(r.n_dn) > 0 and float(r.sum_ret_dn) < -0.2 and float(r.sum_ret_up) > -0.05:
        return "dn_toxic"
    if int(r.n_up) > 0 and float(r.sum_ret_up) < -0.2 and float(r.sum_ret_dn) >= -0.05:
        return "up_toxic"
    if pd.notna(r.qqq_range_1030) and float(r.qqq_range_1030) >= 0.015:
        return "wide_chop"
    return "other_loss"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="maga7/results/regime_router/bad_day_features.csv")
    ap.add_argument("--out", default="maga7/results/regime_router/day_type_labels.csv")
    ap.add_argument("--rescan", action="store_true", help="re-run scan_baseline_hard_days first")
    args = ap.parse_args()

    if args.rescan:
        # invoke scanner as module
        from maga7.tools import scan_baseline_hard_days as scan

        sys.argv = ["scan_baseline_hard_days"]
        scan.main()

    feat_path = Path(args.features)
    if not feat_path.is_file():
        from maga7.tools import scan_baseline_hard_days as scan

        sys.argv = ["scan_baseline_hard_days"]
        scan.main()
    feat = pd.read_csv(feat_path)
    bad = feat[feat["label"] == "bad"].copy()
    if "day_type" not in bad.columns or bad["day_type"].isna().all():
        bad["day_type"] = bad.apply(_day_type, axis=1)
    labels = bad[["date", "day_type", "day_ret", "n_up", "n_dn", "sum_ret_up", "sum_ret_dn"]].copy()
    labels["date"] = labels["date"].astype(str)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    # also json map for quick load
    mp = {str(r.date): str(r.day_type) for r in labels.itertuples(index=False)}
    out.with_suffix(".json").write_text(json.dumps(mp, indent=2), encoding="utf-8")
    summary = {
        "n_labeled": int(len(labels)),
        "counts": labels["day_type"].value_counts().to_dict(),
        "path": str(out),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
