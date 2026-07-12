#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单日特征 parity CLI —— 秒级离线 vs 在线重算,无需 dry_sim。

示例:
  python qqq_btc/tools/feature_parity_day.py --day 2026-06-26
  python qqq_btc/tools/feature_parity_day.py --day 2026-06-26 --json /tmp/feat_parity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.feature_parity import (
    DEFAULT_OFFLINE_PARQUET,
    audit_offline_parquet_day,
    format_report_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="离线 parquet vs 在线重算特征 parity(秒级)")
    parser.add_argument("--day", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_OFFLINE_PARQUET),
        help="离线 quote_features_raw parquet",
    )
    parser.add_argument("--det-tol", type=float, default=1e-5)
    parser.add_argument("--price-tol", type=float, default=1e-3)
    parser.add_argument("--vix-min-corr", type=float, default=0.5)
    parser.add_argument(
        "--carryover",
        action="store_true",
        help="拼接上一交易日 tail(默认无预热,对齐 --warmup-from same-day)",
    )
    parser.add_argument("--stream", action="store_true", help="逐 bar 流式 enrich")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    report = audit_offline_parquet_day(
        Path(args.parquet),
        args.day,
        det_tol=args.det_tol,
        price_tol=args.price_tol,
        vix_min_corr=args.vix_min_corr,
        use_carryover=args.carryover,
        stream_mode=args.stream,
    )
    print(format_report_summary(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json_out}")

    failed = [c.feature for c in report.columns if not c.pass_]
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
