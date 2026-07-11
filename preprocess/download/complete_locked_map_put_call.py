#!/usr/bin/env python3
"""补全 locked_targets_map 中缺失的 PUT/CALL 腿。

根因：step1 依赖 day_iv（由交易分钟 aggs 衍生），无成交合约进不了 IV →
锁约时整条腿缺失。Quote 侧往往仍有 bid/ask，只需用对侧同 strike/到期合成 OCC。

规则（同 expiry + same strike）：
  CALL ATM (bucket 2) ↔ PUT ATM (bucket 0)
  CALL OTM (bucket 3) ↔ PUT OTM (bucket 1)

用法:
  python complete_locked_map_put_call.py \\
      --input ~/train_data/locked_targets_map_0dte.parquet \\
      --inplace

  python complete_locked_map_put_call.py \\
      --input ~/train_data/locked_targets_map_1dte.parquet \\
      --inplace
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# incomplete side → (source_bucket, target_bucket)
# 缺 PUT 时从 CALL 翻；缺 CALL 时从 PUT 翻
FILL_RULES = {
    "put": [(2, 0), (3, 1)],   # fill missing 0/1 from 2/3
    "call": [(0, 2), (1, 3)],  # fill missing 2/3 from 0/1
}

REQUIRED_BUCKETS = {0, 1, 2, 3}
FLIP_RE = re.compile(r"([CP])(\d{8})$")


def flip_contract(symbol: str) -> str | None:
    """O:QQQ220302C00343000 ↔ O:QQQ220302P00343000"""
    s = str(symbol)
    prefix = "O:" if s.startswith("O:") else ""
    body = s[2:] if prefix else s
    m = FLIP_RE.search(body)
    if not m:
        return None
    side = "P" if m.group(1) == "C" else "C"
    flipped = FLIP_RE.sub(f"{side}\\2", body, count=1)
    return prefix + flipped


def complete_day(group: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """补全单日缺失 bucket；返回 (rows_df, added_records)。"""
    have = {int(b): row for b, row in zip(group["bucket_id"].astype(int), group.to_dict("records"))}
    buckets = set(have)
    added: list[dict] = []

    if buckets >= REQUIRED_BUCKETS:
        return group.copy(), added

    missing = REQUIRED_BUCKETS - buckets
    need_put = missing & {0, 1}
    need_call = missing & {2, 3}

    rules: list[tuple[int, int]] = []
    if need_put:
        rules.extend(FILL_RULES["put"])
    if need_call:
        rules.extend(FILL_RULES["call"])

    for src_b, dst_b in rules:
        if dst_b not in missing:
            continue
        if src_b not in have:
            continue
        src = have[src_b]
        flipped = flip_contract(src["contract_symbol"])
        if flipped is None:
            continue
        new_row = {
            "date_str": src["date_str"],
            "contract_symbol": flipped,
            "bucket_id": dst_b,
            "front_dte": src["front_dte"],
            "symbol": src["symbol"],
        }
        have[dst_b] = new_row
        added.append(
            {
                **new_row,
                "source_bucket": src_b,
                "source_contract": src["contract_symbol"],
                "fill_side": "put" if dst_b in (0, 1) else "call",
            }
        )
        missing.discard(dst_b)

    out = pd.DataFrame([have[b] for b in sorted(have)])
    return out, added


def complete_map(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts: list[pd.DataFrame] = []
    all_added: list[dict] = []
    for _, g in df.groupby("date_str", sort=True):
        completed, added = complete_day(g)
        parts.append(completed)
        all_added.extend(added)

    out = pd.concat(parts, ignore_index=True)
    # 保持列顺序
    out = out[df.columns.tolist()]
    added_df = pd.DataFrame(all_added)

    before_pat = (
        df.groupby("date_str")["bucket_id"]
        .apply(lambda s: tuple(sorted(s.astype(int).unique())))
        .value_counts()
        .to_dict()
    )
    after_pat = (
        out.groupby("date_str")["bucket_id"]
        .apply(lambda s: tuple(sorted(s.astype(int).unique())))
        .value_counts()
        .to_dict()
    )
    still_incomplete = [
        d
        for d, g in out.groupby("date_str")
        if set(g["bucket_id"].astype(int)) != REQUIRED_BUCKETS
    ]
    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "days": int(out["date_str"].nunique()),
        "rows_added": int(len(added_df)),
        "days_filled": int(added_df["date_str"].nunique()) if not added_df.empty else 0,
        "before_bucket_patterns": {str(k): int(v) for k, v in before_pat.items()},
        "after_bucket_patterns": {str(k): int(v) for k, v in after_pat.items()},
        "still_incomplete_days": still_incomplete,
        "still_incomplete_count": len(still_incomplete),
    }
    return out, added_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete missing PUT/CALL legs in locked map")
    parser.add_argument("--input", required=True, help="locked_targets_map_*.parquet")
    parser.add_argument("--output", default=None, help="输出路径；默认加 _completed 后缀")
    parser.add_argument("--inplace", action="store_true", help="原地覆盖 --input（先写 .bak）")
    parser.add_argument("--report", default=None, help="JSON 报告路径")
    args = parser.parse_args()

    inp = Path(args.input).expanduser()
    df = pd.read_parquet(inp)
    required = {"date_str", "contract_symbol", "bucket_id", "front_dte", "symbol"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise SystemExit(f"missing columns: {sorted(missing_cols)}")

    out, added_df, summary = complete_map(df)
    summary["input"] = str(inp)

    if args.inplace:
        bak = inp.with_suffix(inp.suffix + ".bak")
        if not bak.exists():
            df.to_parquet(bak, index=False, compression="zstd")
            summary["backup"] = str(bak)
        out_path = inp
    else:
        out_path = Path(args.output).expanduser() if args.output else inp.with_name(
            inp.stem + "_completed.parquet"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    summary["output"] = str(out_path)

    if not added_df.empty:
        added_path = out_path.with_name(out_path.stem + "_added.parquet")
        added_df.to_parquet(added_path, index=False, compression="zstd")
        summary["added_path"] = str(added_path)
        # 可供 step2 下载的 map（去重后的新合约行）
        dl_cols = ["date_str", "contract_symbol", "bucket_id", "front_dte", "symbol"]
        dl_map = added_df[dl_cols].drop_duplicates()
        dl_path = out_path.with_name(out_path.stem + "_to_download.parquet")
        dl_map.to_parquet(dl_path, index=False, compression="zstd")
        summary["to_download_path"] = str(dl_path)
        summary["to_download_rows"] = int(len(dl_map))

    report_path = Path(args.report) if args.report else out_path.with_suffix(".complete_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary,
        "added_sample": added_df.head(20).to_dict(orient="records") if not added_df.empty else [],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
