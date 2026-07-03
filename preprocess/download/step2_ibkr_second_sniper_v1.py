#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: step2_ibkr_second_sniper_v1.py
描述: [IBKR 秒级历史快下] 
      通过 IBKR API (ib_insync) 获取指定股票的 1s K线数据，并保存为 Parquet。
      支持“按天”分多次 Request 拼接，解决 IBKR 对 1s 数据的时长限制 (通常为 1800s/30min)。
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time as dt_time, timedelta
import pytz
from ib_insync import IB, Stock, util
from tqdm.asyncio import tqdm
from pathlib import Path

# 引入配置
try:
    from config import TARGET_SYMBOLS, DATA_DIR, IBKR_HOST, IBKR_PORT, NY_TZ
except ImportError:
    # 路径探测
    sys.path.append(str(Path(__file__).resolve().parents[1] / "baseline"))
    from config import TARGET_SYMBOLS, DATA_DIR, IBKR_HOST, IBKR_PORT, NY_TZ

# ================= 全局配置 =================
OUTPUT_BASE_DIR = Path("/mnt/s990/data/raw_1s/stocks")
IBKR_CLIENT_ID = 88  # 专用下载 ID
MAX_CONCURRENT_SYMBOLS = 1 # IBKR 对 1s 数据的 Pacing Violation 非常严，必须降低并发
CHUNK_DURATION_S = 1800 # 30分钟一个 Chunk，防止触发 IBKR "Duration too long for bar size"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [IB_1s] - %(levelname)s - %(message)s')
logger = logging.getLogger("IBKR_1s_Sniper")

# ================= 核心下载逻辑 =================

async def download_symbol_day(ib, symbol, date_str):
    """
    下载单个股票某一天的全天 (09:30 - 16:00) 1s 数据
    """
    out_dir = OUTPUT_BASE_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{date_str}.parquet"
    
    if out_path.exists():
        return f"⏩ {symbol} {date_str} exists, skipped."

    contract = Stock(symbol, 'SMART', 'USD')
    # 验证合约 (获取 ConId)
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        return f"❌ {symbol}: Contract not found on IBKR."
    contract = qualified[0]

    # 定义 RTH 时间范围
    target_day = datetime.strptime(date_str, "%Y%m%d")
    start_dt = NY_TZ.localize(datetime.combine(target_day, dt_time(9, 30)))
    end_dt = NY_TZ.localize(datetime.combine(target_day, dt_time(16, 0)))
    
    # 因为 IBKR reqHistoricalData 是从 endDateTime 向前追溯，所以我们需要分段请求
    all_bars = []
    curr_end = end_dt
    
    logger.info(f"📥 Starting download for {symbol} on {date_str}...")
    
    while curr_end > start_dt:
        # 格式化日期: '20260319 16:00:00 US/Eastern'
        end_str = curr_end.strftime("%Y%m%d %H:%M:%S US/Eastern")
        
        try:
            # 请求 1800s 的 1 sec Bars
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_str,
                durationStr=f'{CHUNK_DURATION_S} S',
                barSizeSetting='1 secs',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False
            )
            
            if not bars:
                # 可能是没开盘或没数据，尝试向前跳
                curr_end -= timedelta(seconds=CHUNK_DURATION_S)
                continue
                
            df = util.df(bars)
            # 过滤掉超出 09:30 的部分
            df['date'] = df['date'].dt.tz_convert(NY_TZ)
            df = df[df['date'] >= start_dt]
            
            if not df.empty:
                all_bars.append(df)
            
            # 更新下一次请求的结束时间 (减去一秒防止重叠，或者依靠索引去重)
            curr_end = df['date'].min()
            
            # [🔥 重要修复] 针对 1s 数据，必须在请求之间增加强制睡眠，否则会触发 Pacing Violation (Error 162)
            await asyncio.sleep(8) 
            
        except Exception as e:
            logger.error(f"❌ {symbol} download error at {end_str}: {e}")
            break
            
    if not all_bars:
        return f"⚠️ {symbol} {date_str}: No data fetched."

    # 合并、去重、排序
    final_df = pd.concat(all_bars).drop_duplicates(subset=['date']).sort_values('date')
    
    # 标准化字段 (与 Polygon 版对齐)
    final_df['timestamp'] = final_df['date']
    final_df['ts'] = final_df['timestamp'].view('int64') // 10**9
    
    # 格式清理
    final_df = final_df[['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # 保存
    final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    return f"🎯 {symbol} {date_str}: Success! {len(final_df)} seconds of data."

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help="Comma separated symbols, if empty use TARGET_SYMBOLS")
    parser.add_argument('--start-date', type=str, required=True, help="YYYYMMDD")
    parser.add_argument('--end-date', type=str, required=True, help="YYYYMMDD")
    args = parser.parse_args()
    
    symbols = args.symbols.split(',') if args.symbols else TARGET_SYMBOLS
    start_dt = datetime.strptime(args.start_date, "%Y%m%d")
    end_dt = datetime.strptime(args.end_date, "%Y%m%d")
    
    date_list = []
    curr = start_dt
    while curr <= end_dt:
        if curr.weekday() < 5: # 仅周一至周五
            date_list.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)

    if not date_list:
        logger.error("No valid trading days in range.")
        return

    ib = IB()
    try:
        logger.info(f"🔌 Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT}...")
        await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
    except Exception as e:
        logger.error(f"❌ IBKR Connection Failed: {e}")
        return

    # 任务队列
    tasks = []
    for date_str in date_list:
        for sym in symbols:
            tasks.append((sym, date_str))

    logger.info(f"🚀 Total Tasks: {len(tasks)} (Symbols: {len(symbols)} x Days: {len(date_list)})")
    
    # 分批并发执行，防止 IBKR 熔断
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)
    
    async def sem_task(sym, d):
        async with semaphore:
            res = await download_symbol_day(ib, sym, d)
            logger.info(res)
            # 每一天/标的完成后歇久一点
            await asyncio.sleep(15) 

    await tqdm.gather(*[sem_task(s, d) for s, d in tasks], desc="Total Progress")

    ib.disconnect()
    logger.info("🏁 All tasks completed.")

if __name__ == "__main__":
    util.patchAsyncio()
    asyncio.run(main())
