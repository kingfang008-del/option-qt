#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: migrate_alpha_sqlite_to_pg.py
描述: 将 SQLite (market_YYYYMMDD.db) 中的 alpha_logs 表数据批量迁移至 PostgreSQL。
      迁移前会自动清空目标日期的 PG 分区表 (TRUNCATE TABLE alpha_logs_YYYYMMDD)。
"""

import sys
import sqlite3
import argparse
import logging
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

# 确保能导入 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "DB"))

from config import DB_DIR, PG_DB_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MIGRATE] - %(message)s')
logger = logging.getLogger("AlphaMigrator")


def create_pg_partition_if_not_exists(c_pg, date_str):
    """确保目标日期的 PG 分区表存在，如果不存在则采用 PARTITION OF 模式创建"""
    table_name = f"alpha_logs_{date_str}"
    from config import NY_TZ
    from datetime import datetime, timedelta
    
    # 1. 计算该日期的 TS 范围 (NY 时间 00:00:00)
    dt_start = NY_TZ.localize(datetime.strptime(date_str, "%Y%m%d"))
    dt_end = dt_start + timedelta(days=1)
    
    start_ts = dt_start.timestamp()
    end_ts = dt_end.timestamp()
    
    # 2. 检查分区是否存在
    c_pg.execute(f"SELECT to_regclass('public.{table_name}')")
    if not c_pg.fetchone()[0]:
        logger.info(f"➕ Creating missing PG partition: {table_name}")
        c_pg.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} PARTITION OF alpha_logs 
            FOR VALUES FROM ({start_ts}) TO ({end_ts});
        """)
        # 创建索引加速
        c_pg.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_ts_idx ON {table_name} (ts);")
        c_pg.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_sym_idx ON {table_name} (symbol);")


def migrate_database(db_path: Path, conn_pg):
    date_str = db_path.stem.split('_')[1]
    pg_table = f"alpha_logs_{date_str}"
    c_pg = conn_pg.cursor()
    
    # 1. 连接 SQLite
    try:
        conn_sq = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        c_sq = conn_sq.cursor()
    except Exception as e:
        logger.error(f"❌ Failed to connect to SQLite DB {db_path.name}: {e}")
        return

    # 2. 检查 SQLite 中是否有 alpha_logs 表及其数据量
    c_sq.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
    if not c_sq.fetchone():
        logger.info(f"⏭️ Skipping {db_path.name}, no 'alpha_logs' table found.")
        conn_sq.close()
        return

    c_sq.execute("SELECT COUNT(*) FROM alpha_logs")
    total_rows = c_sq.fetchone()[0]
    
    if total_rows == 0:
        logger.info(f"⏭️ Skipping {db_path.name}, 'alpha_logs' table is empty.")
        conn_sq.close()
        return

    logger.info(f"🚀 Migrating {total_rows} rows from {db_path.name} -> PostgreSQL '{pg_table}'")

    # 3. 准备 PostgreSQL 目标
    create_pg_partition_if_not_exists(c_pg, date_str)
    
    # 4. [关键要求] 清空 PG 该日期原有的数据
    logger.info(f"🧹 Truncating existing PostgreSQL partition: {pg_table}")
    try:
        c_pg.execute(f"TRUNCATE TABLE {pg_table}")
    except psycopg2.errors.UndefinedTable:
        # 如果表刚刚被创建，Truncate 一般不会报错，但也做个兜底
        pass

    # 5. 批量读取与写入
    batch_size = 5000
    # [修正] SQLite alpha_logs 实际并无 payload_json，而是具体的 7 个列
    c_sq.execute("SELECT ts, datetime_ny, symbol, alpha, iv, price, vol_z FROM alpha_logs ORDER BY ts ASC")
    
    insert_query = f"""
        INSERT INTO {pg_table} (ts, datetime_ny, symbol, alpha, iv, price, vol_z) 
        VALUES %s
    """
    
    rows_processed = 0
    while True:
        rows = c_sq.fetchmany(batch_size)
        if not rows:
            break
            
        # 执行批量插入
        try:
            execute_values(c_pg, insert_query, rows)
            conn_pg.commit()
            rows_processed += len(rows)
            if rows_processed % 50000 == 0:
                logger.info(f"  ... inserted {rows_processed}/{total_rows} rows")
        except Exception as e:
            logger.error(f"❌ Failed to insert batch into PG {pg_table}: {e}")
            conn_pg.rollback()
            break
            
    conn_sq.close()
    c_pg.close()
    logger.info(f"✅ Migration for {date_str} completed successfully! Total rows: {rows_processed}\n")

def main():
    parser = argparse.ArgumentParser(description="Migrate alpha_logs from SQLite to PostgreSQL")
    parser.add_argument('--start-date', type=str, default="20260101", help="Start processing from this date (YYYYMMDD)")
    parser.add_argument('--end-date', type=str, default="20991231", help="Process up to this date (YYYYMMDD)")
    args = parser.parse_args()

    # ================= 1. 获取需要处理的数据库列表 =================
    all_dbs = sorted([f for f in DB_DIR.glob("market_*.db") if f.stem.startswith("market_") and len(f.stem) == 15])
    target_dbs = []
    for db in all_dbs:
        date_str = db.stem.split('_')[1]
        if args.start_date <= date_str <= args.end_date:
            target_dbs.append(db)
            
    if not target_dbs:
        logger.error(f"❌ No databases found between {args.start_date} and {args.end_date} in {DB_DIR}.")
        return
        
    logger.info(f"🔍 Found {len(target_dbs)} databases to process.")

    # ================= 2. 建立 PostgreSQL 连接 =================
    try:
        conn_pg = psycopg2.connect(PG_DB_URL)
        # 不使用 autocommit，方便基于批次 commit
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL at {PG_DB_URL}: {e}")
        return

    # ================= 3. 逐个迁移 =================
    for db_path in target_dbs:
        migrate_database(db_path, conn_pg)

    conn_pg.close()
    logger.info("🎉 All migrations finished!")

if __name__ == "__main__":
    main()
