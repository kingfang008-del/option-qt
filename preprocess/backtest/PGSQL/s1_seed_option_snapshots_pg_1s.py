#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="PostgreSQL 1s seeder wrapper. Delegates to the canonical SQLite seeder and mirrors writes into PostgreSQL."
    )
    parser.add_argument("--start_date", type=str, default="2026-01-02", help="只处理该日期及之后的数据")
    parser.add_argument("--end_date", type=str, default="2099-12-31", help="只处理该日期及之前的数据")
    parser.add_argument(
        "--stock-root",
        type=str,
        default="/mnt/s990/data/raw_1s/stocks",
        help="1s 股票原始 parquet 根目录",
    )
    parser.add_argument(
        "--option-root",
        type=str,
        default="/mnt/s990/data/raw_1s/options",
        help="1s 期权原始 parquet 根目录",
    )
    parser.add_argument(
        "--skip-pg-clean",
        action="store_true",
        help="默认会清理目标 PG 日分区；如需保留已有 PG 数据，则显式跳过清理。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sqlite_script = (
        Path(__file__).resolve().parents[1]
        / "second"
        / "s1_seed_option_snapshots_sqlite_1s.py"
    )

    cmd = [
        sys.executable,
        str(sqlite_script),
        "--start_date", args.start_date,
        "--end_date", args.end_date,
        "--stock-root", args.stock_root,
        "--option-root", args.option_root,
        "--mirror-pg",
    ]
    if args.skip_pg_clean:
        cmd.append("--skip-pg-clean")

    print("🪞 [PG Wrapper] Delegating to canonical SQLite seeder with PostgreSQL mirroring...")
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd).returncode)
