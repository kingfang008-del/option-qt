#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bak monthly 盘中「换约」逻辑：prefer_primary_gapfill。

结论（对归档 `_bak_pre4c/quote_options_monthly_iv` / `locked_targets_map_0dte_dynamic`）
================================================================

这不是按盘中 delta 主动换仓，而是：

1. 每个交易日、每个 bucket 有 **primary**（该 bucket 当日 bar 数最多的合约）
   和可选的 **secondary**（同 bucket 的另一合约）。
2. 分钟特征序列 **优先用 primary 报价**。
3. **仅当该分钟 primary 无报价时**，才用同 bucket 的 secondary 补洞。
4. primary 与 secondary 在 bak monthly 中 **同分钟不重叠**（overlap=0）；
   secondary 出现的分钟集合 == fixed8 1m 中「secondary 有报价且 primary 无报价」的分钟。

证据
----
- `locked_targets_map_0dte_dynamic.parquet` 与 bak monthly 合约日集合 382/382 完全一致。
- 当前 `options_databento` 1m 只有 primary（约 220/390 contract-days）；
  secondary 170 条在 `options_databento_fixed8_corrected` 中已齐全。
- prefer-primary 重建 vs bak 活跃 ticker 加权匹配约 **98.7%**。

与现有代码的差异
----------------
`options_locked_feature.calculate_locked_features` 使用
`drop_duplicates(['timestamp','bucket_id'], keep='last')`。
若同分钟双合约都在，会取 last，**不等于** bak 的 prefer-primary。
复现 bak 特征时应先调用本模块的 `prefer_primary_gapfill`。

用法
----
  python qqq_btc/tools/bak_monthly_switch_logic.py report
  python qqq_btc/tools/bak_monthly_switch_logic.py assemble-1m \\
      --src-1m /mnt/s990/data/raw_1m/options_databento_fixed8_corrected \\
      --out-1m /mnt/s990/data/v4_original_jul5/bak_monthly_redownload/raw_1m_prefer_primary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

NY = "America/New_York"
DEFAULT_DYNAMIC_MAP = Path.home() / "train_data/locked_targets_map_0dte_dynamic.parquet"
DEFAULT_BAK_MAP = Path.home() / "train_data/locked_targets_map_bak_monthly_apr_jun_dynamic.parquet"
DEFAULT_MANIFEST = Path("/mnt/s990/data/v4_original_jul5/manifest/bak_monthly_switch_logic.json")


def load_role_map(
    dynamic_map: Path = DEFAULT_DYNAMIC_MAP,
    date_from: str = "2026-04-01",
    date_to: str = "2026-06-30",
) -> pd.DataFrame:
    """从 dynamic map 生成 primary/secondary 角色表。"""
    dyn = pd.read_parquet(dynamic_map)
    dyn = dyn[dyn["date_str"].between(date_from, date_to)].copy()
    dyn["contract_symbol"] = dyn["contract_symbol"].astype(str).str.replace("O:", "")
    prim = (
        dyn.sort_values(["date_str", "bucket_id", "n_rows"], ascending=[True, True, False])
        .drop_duplicates(["date_str", "bucket_id"], keep="first")
        .assign(role="primary", rank=0)
    )
    sec = dyn.merge(
        prim[["date_str", "bucket_id", "contract_symbol"]],
        on=["date_str", "bucket_id", "contract_symbol"],
        how="left",
        indicator=True,
    )
    sec = sec[sec["_merge"] == "left_only"].drop(columns="_merge").assign(role="secondary", rank=1)
    out = pd.concat([prim, sec], ignore_index=True)
    return out.sort_values(["date_str", "bucket_id", "rank"]).reset_index(drop=True)


def primary_lookup(role_map: pd.DataFrame) -> dict[tuple[str, int], str]:
    prim = role_map[role_map["role"] == "primary"]
    return {
        (str(r.date_str), int(r.bucket_id)): str(r.contract_symbol)
        for r in prim.itertuples()
    }


def prefer_primary_gapfill(
    quotes: pd.DataFrame,
    date_str: str,
    primary_by_bucket: dict[tuple[str, int], str],
    ticker_col: str = "ticker",
    ts_col: str = "timestamp",
    bucket_col: str = "bucket_id",
) -> pd.DataFrame:
    """同分钟同 bucket：优先保留 primary；primary 缺失时才保留 secondary。"""
    df = quotes.copy()
    if ts_col not in df.columns and "ts" in df.columns:
        df[ts_col] = df["ts"]
    df[ts_col] = pd.to_datetime(df[ts_col])
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize(NY)
    else:
        df[ts_col] = df[ts_col].dt.tz_convert(NY)

    df["_is_primary"] = [
        str(t) == primary_by_bucket.get((date_str, int(b)))
        for t, b in zip(df[ticker_col], df[bucket_col])
    ]
    # primary first
    df = df.sort_values([ts_col, bucket_col, "_is_primary"], ascending=[True, True, False])
    out = df.drop_duplicates([ts_col, bucket_col], keep="first").drop(columns="_is_primary")
    return out.reset_index(drop=True)


def assemble_prefer_primary_1m(
    src_1m_root: Path,
    out_1m_root: Path,
    role_map: pd.DataFrame,
    symbol: str = "QQQ",
    dates: Iterable[str] | None = None,
) -> dict:
    """从含双合约的 1m 目录写出 prefer-primary 后的每日 parquet。"""
    prim = primary_lookup(role_map)
    src = Path(src_1m_root) / symbol
    out = Path(out_1m_root) / symbol
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(role_map["date_str"].unique()) if dates is None else list(dates)
    stats = {"days": 0, "rows_in": 0, "rows_out": 0, "missing_files": []}
    for day in days:
        fp = src / f"{symbol}_{day}.parquet"
        if not fp.exists():
            stats["missing_files"].append(str(fp))
            continue
        raw = pd.read_parquet(fp)
        stats["rows_in"] += len(raw)
        filled = prefer_primary_gapfill(raw, day, prim)
        # 仍保留当日全部合约行（含 secondary 仅出现在 gap 分钟），便于 day_iv；
        # 若只要活跃序列，用 filled。这里写「全量但已按 bak 规则过滤同分钟冲突」：
        # 实际 bak monthly 保留了 primary 全部分钟 + secondary 仅 gap 分钟的行。
        # 重建全量行：primary 所有行 ∪ secondary 仅 gap 行
        raw = raw.copy()
        ts_col = "timestamp" if "timestamp" in raw.columns else "ts"
        raw[ts_col] = pd.to_datetime(raw[ts_col])
        if raw[ts_col].dt.tz is None:
            raw[ts_col] = raw[ts_col].dt.tz_localize(NY)
        else:
            raw[ts_col] = raw[ts_col].dt.tz_convert(NY)
        parts = []
        for b in sorted(raw["bucket_id"].dropna().unique()):
            b = int(b)
            p_sym = prim.get((day, b))
            sub = raw[raw["bucket_id"].astype(int) == b]
            if p_sym is None:
                parts.append(sub)
                continue
            prim_rows = sub[sub["ticker"].astype(str) == p_sym]
            prim_ts = set(prim_rows[ts_col])
            sec_rows = sub[sub["ticker"].astype(str) != p_sym]
            sec_gap = sec_rows[~sec_rows[ts_col].isin(prim_ts)]
            parts.append(pd.concat([prim_rows, sec_gap], ignore_index=True))
        rebuilt = pd.concat(parts, ignore_index=True).sort_values([ts_col, "bucket_id", "ticker"])
        rebuilt.to_parquet(out / f"{symbol}_{day}.parquet", index=False)
        stats["rows_out"] += len(rebuilt)
        stats["days"] += 1
        _ = filled  # active series available if needed later
    return stats


def cmd_report(args: argparse.Namespace) -> None:
    role = load_role_map(Path(args.dynamic_map), args.date_from, args.date_to)
    report = {
        "logic": "prefer_primary_gapfill",
        "summary": (
            "每个 bucket 的 primary=当日最多 bar 的合约；secondary 仅在 primary 缺分钟时补洞。"
            "不是盘中按 delta 主动换约。"
        ),
        "contract_days": int(len(role)),
        "days": int(role["date_str"].nunique()),
        "primary": int((role["role"] == "primary").sum()),
        "secondary": int((role["role"] == "secondary").sum()),
        "map_out": str(DEFAULT_BAK_MAP),
        "manifest": str(DEFAULT_MANIFEST),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_assemble(args: argparse.Namespace) -> None:
    role = load_role_map(Path(args.dynamic_map), args.date_from, args.date_to)
    stats = assemble_prefer_primary_1m(
        Path(args.src_1m), Path(args.out_1m), role, symbol=args.symbol
    )
    print(json.dumps(stats, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Bak monthly prefer-primary gapfill logic")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="打印逻辑摘要")
    r.add_argument("--dynamic-map", default=str(DEFAULT_DYNAMIC_MAP))
    r.add_argument("--date-from", default="2026-04-01")
    r.add_argument("--date-to", default="2026-06-30")
    r.set_defaults(func=cmd_report)

    a = sub.add_parser("assemble-1m", help="从双合约 1m 组装 prefer-primary 1m")
    a.add_argument("--dynamic-map", default=str(DEFAULT_DYNAMIC_MAP))
    a.add_argument("--date-from", default="2026-04-01")
    a.add_argument("--date-to", default="2026-06-30")
    a.add_argument("--symbol", default="QQQ")
    a.add_argument(
        "--src-1m",
        default="/mnt/s990/data/raw_1m/options_databento_fixed8_corrected",
    )
    a.add_argument(
        "--out-1m",
        default="/mnt/s990/data/v4_original_jul5/bak_monthly_redownload/raw_1m_prefer_primary",
    )
    a.set_defaults(func=cmd_assemble)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
