#!/usr/bin/env python3
"""
清空 history_sqlite_1s 目录下每个 *.db 中存放 alpha 信号数据的表。

本仓库 1s 历史库中为 alpha_logs（非表名 alpha）。如需其它表名，使用 --table。
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _default_db_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "history_sqlite_1s"


def _resolve_db_dir(cli_path: Path) -> Path:
    p = cli_path.expanduser().resolve()
    if p.exists():
        return p
    # 若在仓库 production/scripts 下运行，上一级起拼相对路径常见失败，再试相对于 cwd
    cwd = Path.cwd()
    cand = cwd / cli_path
    if cand.exists():
        return cand.resolve()
    # 兼容从仓库根运行时的相对路径
    root_candidate = cwd / cli_path
    if root_candidate.exists():
        return root_candidate.resolve()
    return p


def _iter_sqlite(db_dir: Path) -> list[Path]:
    return sorted(p for p in db_dir.glob("*.db") if p.is_file())


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _validate_table(name: str) -> None:
    if not _TABLE_RE.match(name):
        sys.stderr.write(
            f"非法表名: {name!r}（只允许字母数字下划线，且不能以数字开头）\n"
        )
        sys.exit(2)


def main() -> None:
    repo_root = Path.home() / "quant_project"

    parser = argparse.ArgumentParser(
        description="清空 history_sqlite_1s 下各 SQLite 文件中指定表的行。"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_default_db_dir(repo_root),
        help="含 market_YYYYMMDD.db 的目录（默认: production/preprocess/backtest/history_sqlite_1s）",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="alpha_logs",
        help="要清空的表名（默认: alpha_logs；若仅有 alpha 则传 --table alpha）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要处理的文件与行数预估，不写库",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="每个库 DELETE 后执行 VACUUM（慢，可回收磁盘空间）",
    )
    args = parser.parse_args()

    _validate_table(args.table)
    db_dir = _resolve_db_dir(args.dir)
    if not db_dir.is_dir():
        sys.stderr.write(f"目录不存在或不是文件夹: {db_dir}\n")
        sys.exit(1)

    files = _iter_sqlite(db_dir)
    if not files:
        sys.stderr.write(f"{db_dir} 下未发现 *.db\n")
        sys.exit(1)

    for db_path in files:
        uri = db_path.resolve().as_uri()
        uri_ro = uri + "?mode=ro"

        try:
            with sqlite3.connect(uri_ro, uri=True) as probe:
                if not _table_exists(probe, args.table):
                    print(f"[SKIP] {db_path.name}: 无表 {args.table}")
                    continue
                n_before = probe.execute(
                    f"SELECT COUNT(*) FROM {args.table}"
                ).fetchone()[0]
        except sqlite3.Error as e:
            print(f"[ERR ] {db_path.name}: 只读探测失败 ({e})")
            continue

        if args.dry_run:
            print(f"[DRY ] {db_path.name}: would DELETE FROM {args.table} ({n_before} rows)")
            continue

        try:
            with sqlite3.connect(str(db_path), timeout=600.0) as conn:
                if not _table_exists(conn, args.table):
                    print(f"[SKIP] {db_path.name}: 无表 {args.table}")
                    continue
                conn.execute(f"DELETE FROM {args.table}")
                conn.commit()
                if args.vacuum:
                    conn.execute("VACUUM")
            print(f"[OK  ] {db_path.name}: cleared {args.table} ({n_before} rows deleted)")
        except sqlite3.Error as e:
            print(f"[ERR ] {db_path.name}: ({e})")


if __name__ == "__main__":
    main()
