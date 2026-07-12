#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FCS 特征对拍 CLI —— 无预热(cold) vs 有预热(warm) vs 流式逐 bar。

与 compare_stream_replay_day 解耦: 不跑 Redis 多进程,秒级定位 FCS/离线偏差。

示例:
  python qqq_btc/tools/fcs_parity_day.py --day 2026-06-26
  python qqq_btc/tools/fcs_parity_day.py --day 2026-06-26 --json /tmp/fcs_parity.json
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
    FeatureColumnReport,
    FeatureParityReport,
    audit_fcs_parity_day,
    audit_offline_parquet_day,
    format_report_summary,
)


def _dict_to_report(block: dict) -> FeatureParityReport:
    return FeatureParityReport(
        day=block["day"],
        rows=block["rows"],
        columns=[
            FeatureColumnReport(
                feature=c["feature"],
                tier=c["tier"],
                rows=c["rows"],
                med_abs_err=c["med_abs_err"],
                max_abs_err=c["max_abs_err"],
                corr=c.get("corr"),
                pass_=c["pass"],
                note=c.get("note", ""),
            )
            for c in block["columns"]
        ],
    )


def _summarize_mode(name: str, block: dict) -> str:
    cols = block.get("columns", [])
    failed = [c["feature"] for c in cols if not c.get("pass")]
    det = [c for c in cols if c.get("tier") == "deterministic"]
    det_fail = [c["feature"] for c in det if not c.get("pass")]
    return (
        f"[{name}] pass_rate={block.get('pass_rate', 0):.1%} "
        f"rows={block.get('rows')} failed={len(failed)} "
        f"det_fail={det_fail[:8]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="FCS cold/warm/stream feature parity")
    parser.add_argument("--day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--parquet", default=str(DEFAULT_OFFLINE_PARQUET))
    parser.add_argument("--det-tol", type=float, default=1e-5)
    parser.add_argument("--price-tol", type=float, default=1e-3)
    parser.add_argument("--vix-min-corr", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=("all", "cold", "warm", "stream"),
        default="all",
        help="cold=无预热(默认对齐 --warmup-from same-day)",
    )
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    if args.mode == "all":
        payload = audit_fcs_parity_day(
            Path(args.parquet),
            args.day,
            det_tol=args.det_tol,
            price_tol=args.price_tol,
            vix_min_corr=args.vix_min_corr,
        )
        print(f"day={args.day} parquet={args.parquet}\n")
        for key, label in (
            ("cold_no_carryover", "cold 无预热"),
            ("warm_with_carryover", "warm 有预热"),
            ("stream_incremental", "stream 逐bar无预热"),
        ):
            print(_summarize_mode(label, payload[key]))
        for key, label in (
            ("cold_no_carryover", "cold 无预热"),
            ("warm_with_carryover", "warm 有预热"),
            ("stream_incremental", "stream 逐bar无预热"),
        ):
            print(f"\n--- {label} ---")
            print(format_report_summary(_dict_to_report(payload[key])))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nWrote {args.json_out}")
        failed = [
            c["feature"]
            for k in ("cold_no_carryover", "stream_incremental")
            for c in payload[k]["columns"]
            if not c["pass"] and c["tier"] == "deterministic"
        ]
        return 0 if not failed else 2

    use_carryover = args.mode == "warm"
    stream_mode = args.mode == "stream"
    report = audit_offline_parquet_day(
        Path(args.parquet),
        args.day,
        det_tol=args.det_tol,
        price_tol=args.price_tol,
        vix_min_corr=args.vix_min_corr,
        use_carryover=use_carryover,
        stream_mode=stream_mode,
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
