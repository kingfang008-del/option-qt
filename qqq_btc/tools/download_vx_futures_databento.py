#!/usr/bin/env python3
"""从 Databento XCBF.PITCH 下载标准月度 VIX futures（VX）日线。

目标是构造 VX1/VX2/恒定期限与 term-structure selector，不下载昂贵的全历史
1 分钟数据。Databento 的 end 为开区间。

示例:
  python qqq_btc/tools/download_vx_futures_databento.py
  python qqq_btc/tools/download_vx_futures_databento.py \
    --start 2024-01-01 --end 2026-07-13T22:00:00Z --max-cost-usd 10
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import databento as db
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 databento 包；请在 ibkr 环境安装 databento") from exc


DATASET = "XCBF.PITCH"
SCHEMA = "ohlcv-1d"
CFE_PUBLISHER_ID = 105  # 排除 publisher 106 的 off-market trades
MONTH_CODES = "FGHJKMNQUVXZ"
DEFAULT_OUT = Path("/mnt/s990/data/raw_1m/vix_futures_databento")


def standard_vx_symbols(start: date, end: date) -> list[str]:
    """生成覆盖区间的标准月度 VX raw symbols，并包含 end 年后续月份。"""
    return [
        f"VX/{code}{str(year)[-1]}"
        for year in range(start.year, end.year + 1)
        for code in MONTH_CODES
    ]


def load_api_key(explicit: str | None, key_file: Path) -> str:
    key = explicit or os.environ.get("DATABENTO_API_KEY")
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(
            "未找到 Databento API key：设置 DATABENTO_API_KEY 或提供 --key-file"
        )
    return key


def estimate(
    client: db.Historical,
    *,
    start: str,
    end: str,
    symbols: list[str],
    schema: str,
) -> dict[str, Any]:
    kwargs = {
        "dataset": DATASET,
        "start": start,
        "end": end,
        "symbols": symbols,
        "schema": schema,
        "stype_in": "raw_symbol",
    }
    return {
        "schema": schema,
        "cost_usd": float(client.metadata.get_cost(**kwargs)),
        "records": int(client.metadata.get_record_count(**kwargs)),
        "billable_size": int(client.metadata.get_billable_size(**kwargs)),
    }


def download_frame(
    client: db.Historical,
    *,
    start: str,
    end: str,
    symbols: list[str],
    schema: str,
) -> pd.DataFrame:
    store = client.timeseries.get_range(
        dataset=DATASET,
        start=start,
        end=end,
        symbols=symbols,
        schema=schema,
        stype_in="raw_symbol",
        stype_out="instrument_id",
    )
    frame = store.to_df(map_symbols=True).reset_index()
    if "publisher_id" in frame.columns:
        frame = frame.loc[frame["publisher_id"] == CFE_PUBLISHER_ID].copy()
    return frame.sort_values(["ts_event", "symbol"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download standard VX futures from Databento")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument(
        "--end",
        default="2026-07-13T22:00:00Z",
        help="开区间；默认钉住当前 XCBF 历史许可边界",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--api-key", default=None)
    p.add_argument("--key-file", type=Path, default=Path.home() / "api_key.txt")
    p.add_argument("--max-cost-usd", type=float, default=10.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_d = pd.Timestamp(args.start).date()
    end_d = pd.Timestamp(args.end).date()
    if end_d <= start_d:
        raise SystemExit("--end 必须晚于 --start")

    symbols = standard_vx_symbols(start_d, end_d)
    key = load_api_key(args.api_key, args.key_file)
    client = db.Historical(key)

    bars_est = estimate(
        client,
        start=args.start,
        end=args.end,
        symbols=symbols,
        schema=SCHEMA,
    )
    defs_est = estimate(
        client,
        start=args.start,
        end=args.end,
        symbols=symbols,
        schema="definition",
    )
    total_cost = bars_est["cost_usd"] + defs_est["cost_usd"]
    print(
        f"Databento estimate: bars=${bars_est['cost_usd']:.4f} "
        f"definitions=${defs_est['cost_usd']:.4f} total=${total_cost:.4f}"
    )
    print(
        f"records: bars={bars_est['records']:,} definitions={defs_est['records']:,}"
    )
    if args.estimate_only:
        return
    if total_cost > args.max_cost_usd:
        raise SystemExit(
            f"预计费用 ${total_cost:.2f} 超过 --max-cost-usd ${args.max_cost_usd:.2f}"
        )

    out = args.output_dir.expanduser()
    bars_path = out / "vx_standard_ohlcv_1d.parquet"
    defs_path = out / "vx_standard_definitions.parquet"
    conditions_path = out / "dataset_conditions.json"
    manifest_path = out / "manifest.json"
    existing = [
        p
        for p in (bars_path, defs_path, conditions_path, manifest_path)
        if p.exists()
    ]
    if existing and not args.force:
        raise SystemExit(
            "输出已存在，拒绝重复付费下载；如确需覆盖请传 --force："
            + ", ".join(str(p) for p in existing)
        )
    out.mkdir(parents=True, exist_ok=True)

    bars = download_frame(
        client,
        start=args.start,
        end=args.end,
        symbols=symbols,
        schema=SCHEMA,
    )
    definitions = download_frame(
        client,
        start=args.start,
        end=args.end,
        symbols=symbols,
        schema="definition",
    )

    bars.to_parquet(bars_path, index=False)
    definitions.to_parquet(defs_path, index=False)
    conditions = client.metadata.get_dataset_condition(
        dataset=DATASET,
        start_date=args.start,
        end_date=pd.Timestamp(args.end).date().isoformat(),
    )
    conditions_path.write_text(
        json.dumps(conditions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    non_ok_conditions = [
        row
        for row in conditions
        if str(row.get("condition") or "").lower() not in ("", "available", "ok")
    ]
    manifest = {
        "dataset": DATASET,
        "schema": SCHEMA,
        "publisher_id": CFE_PUBLISHER_ID,
        "start": args.start,
        "end_exclusive": args.end,
        "symbols": symbols,
        "estimated_cost_usd": total_cost,
        "bars_rows": len(bars),
        "definitions_rows": len(definitions),
        "bars_path": str(bars_path),
        "definitions_path": str(defs_path),
        "dataset_conditions_path": str(conditions_path),
        "non_ok_conditions": non_ok_conditions,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {bars_path} rows={len(bars):,}")
    print(f"wrote {defs_path} rows={len(definitions):,}")
    print(
        f"wrote {conditions_path} degraded_or_missing={len(non_ok_conditions)}"
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
