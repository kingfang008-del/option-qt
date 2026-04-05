#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: db_cleanup_mar_2026.py
描述: 删除 PostgreSQL 中 2026年3月份的所有历史数据 (1m, 5m, snapshots, alpha_logs)
"""

import psycopg2
import sys
from datetime import datetime
import pytz

# 尝试引入配置
try:
    from production.baseline.config import PG_DB_URL, NY_TZ
except ImportError:
    # 基础硬编码兜底 (与 config.py 一致)
    PG_DB_URL = "dbname=quant_trade user=postgres password=postgres host=192.168.50.116 port=5432"
    NY_TZ = pytz.timezone('America/New_York')

TABLES = [
    'market_bars_1m',
    'option_snapshots_1m',
    'market_bars_5m',
    'option_snapshots_5m',
    'alpha_logs'
]

# 2026年3月的时间范围 (NY 时区)
start_date = datetime(2026, 3, 1)
end_date = datetime(2026, 3, 31)

def cleanup():
    print(f"🚀 开始清理 2026-03 数据 (分区表物理删除模式)...")
    print(f"🕒 日期范围: {start_date.strftime('%Y%m%d')} -> {end_date.strftime('%Y%m%d')}")
    
    confirm = input("\n🚨 警告: 此操作不可逆！确认要直接 DROP 掉这些表的三月份分区吗？(y/n): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消。")
        return

    # 生成 20260301 到 20260331 的日期后缀
    date_suffixes = [(start_date + timedelta(days=x)).strftime("%Y%m%d") for x in range((end_date - start_date).days + 1)]

    try:
        conn = psycopg2.connect(PG_DB_URL)
        cur = conn.cursor()
        conn.autocommit = True # DROP TABLE 最好开启自动提交
        
        for table in TABLES:
            print(f"\n🧹 正在处理基础表: {table} 的分区 ...")
            dropped_count = 0
            
            for date_suffix in date_suffixes:
                partition_name = f"{table}_{date_suffix}"
                
                # PostgreSQL 中可以安全的 DROP TABLE IF EXISTS
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {partition_name} CASCADE;")
                    dropped_count += 1
                except Exception as e:
                    print(f"   ❌ 删除分区 {partition_name} 失败: {e}")
                    
            print(f"   ✅ 完成，成功检查并尝试删除了该表下的 {dropped_count} 个分区。")
            
        print("\n✨ 2026年3月的所有分区表清理完毕！主表依然保留。")
        
    except Exception as e:
        print(f"❌ 数据库连接/操作致命错误: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    cleanup()
