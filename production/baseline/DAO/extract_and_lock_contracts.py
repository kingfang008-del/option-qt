#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: extract_and_lock_contracts.py
描述: 从上一个交易日的 option_snapshots_1m 提取合约并覆写到今天的 contract_locks。
      用于在系统重启时，快速恢复上个交易日正在跟踪的合约，确保数据连续性。
"""

import asyncio
import logging
import datetime
import json
import psycopg2

# Fix for Python 3.14+ where ib_insync/eventkit might fail without a loop
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Option
from config import PG_DB_URL, NY_TZ, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, BUCKET_SPECS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExtractAndLock")

async def extract_and_lock():
    # 1. 初始化 IB 连接 (用于补全合约详情)
    ib = IB()
    try:
        logger.info(f"🔌 Connecting to IBKR ({IBKR_HOST}:{IBKR_PORT})...")
        await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID + 99)
    except Exception as e:
        logger.error(f"❌ Failed to connect to IBKR: {e}")
        return

    try:
        today_obj = datetime.datetime.now(NY_TZ).date()
        today_str = str(today_obj)
        
        conn = psycopg2.connect(PG_DB_URL)
        cursor = conn.cursor()

        # 2. 找到上一个交易日 (最大的 ts，且日期早于今天)
        # 获取所有分区的最大时间戳
        cursor.execute("SELECT MAX(ts) FROM option_snapshots_1m WHERE ts < %s", (datetime.datetime.combine(today_obj, datetime.time.min).timestamp(),))
        last_ts = cursor.fetchone()[0]
        
        if not last_ts:
            logger.warning("⚠️ No snapshots found before today. Nothing to extract.")
            return

        last_date_str = datetime.datetime.fromtimestamp(last_ts, NY_TZ).strftime('%Y-%m-%d')
        logger.info(f"🔍 Latest snapshot found on: {last_date_str} (ts: {last_ts})")

        # 3. 提取上个交易日最后一刻的所有合约
        # 注意：这里我们取每个 symbol 在那个日期的最后一条记录
        cursor.execute("""
            WITH last_snaps AS (
                SELECT symbol, buckets_json, ts,
                       ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY ts DESC) as rn
                FROM option_snapshots_1m
                WHERE ts >= %s AND ts <= %s
            )
            SELECT symbol, buckets_json FROM last_snaps WHERE rn = 1
        """, (last_ts - 60, last_ts)) # 取最后一分钟内的
        
        rows = cursor.fetchall()
        if not rows:
            logger.warning(f"⚠️ No snapshot records found for {last_date_str}")
            return

        logger.info(f"📦 Found snapshots for {len(rows)} symbols. Resolving contracts...")

        # 4. 解析合约详情并准备写入
        all_locks = []
        for symbol, buckets_json in rows:
            if isinstance(buckets_json, str):
                data = json.loads(buckets_json)
            else:
                data = buckets_json
                
            contracts_list = data.get('contracts', [])
            if not contracts_list:
                continue

            # data['buckets'] 对应 TAG_TO_INDEX
            # 我们需要把 index 转回 Tag
            idx_to_tag = {v['bucket_idx']: k for k, v in BUCKET_SPECS.items()}
            
            for idx, local_symbol in enumerate(contracts_list):
                if not local_symbol:
                    continue
                
                tag = idx_to_tag.get(idx)
                if not tag:
                    continue

                # 通过 localSymbol 获取完整 Contract 详情
                logger.info(f"🔎 Resolving {symbol} {tag}: {local_symbol}")
                # 注意：IB.qualifyContracts 比较慢，批量处理时可以优化
                # 简单起见，这里直接用 Option(localSymbol=...)
                try:
                    # 尝试从本地解析出部分信息，或者直接去 IB 查
                    # 生产环境中最稳妥的是 reqContractDetails
                    temp_opt = Option(localSymbol=local_symbol, exchange='SMART', currency='USD')
                    details = await ib.reqContractDetailsAsync(temp_opt)
                    
                    if not details:
                        logger.warning(f"❌ Could not resolve details for {local_symbol}")
                        continue
                    
                    c = details[0].contract
                    # row format for contract_locks:
                    # (date, symbol, tag, conId, expiry, strike, p_right, multiplier, localSymbol, tradingClass)
                    all_locks.append((
                        today_str, symbol, tag,
                        int(c.conId), c.lastTradeDateOrContractMonth,
                        float(c.strike), c.right, str(c.multiplier),
                        c.localSymbol, c.tradingClass
                    ))
                except Exception as e:
                    logger.error(f"❌ Error resolving {local_symbol}: {e}")

        # 5. 覆写到今天的 contract_locks
        if all_locks:
            logger.info(f"💾 Overwriting {len(all_locks)} locks for {today_str}...")
            # 删除今天的旧锁 (Overwrite)
            cursor.execute("DELETE FROM contract_locks WHERE date = %s", (today_str,))
            
            # 批量插入
            insert_query = """
                INSERT INTO contract_locks (date, symbol, tag, conId, expiry, strike, p_right, multiplier, localSymbol, tradingClass)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, symbol, tag) DO UPDATE 
                SET conId=EXCLUDED.conId, expiry=EXCLUDED.expiry, strike=EXCLUDED.strike, p_right=EXCLUDED.p_right, 
                    multiplier=EXCLUDED.multiplier, localSymbol=EXCLUDED.localSymbol, tradingClass=EXCLUDED.tradingClass
            """
            cursor.executemany(insert_query, all_locks)
            
            conn.commit()
            logger.info("✅ Successfully restored locks to PostgreSQL.")
        else:
            logger.warning("⚠️ No valid contracts resolved to lock.")

        cursor.close()
        conn.close()

    finally:
        if ib.isConnected():
            ib.disconnect()
        else:
            logger.warning("⚠️ IBKR was not connected, some details might be missing.")

if __name__ == "__main__":
    asyncio.run(extract_and_lock())
