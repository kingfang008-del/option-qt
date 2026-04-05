import sys
import os
import time
import argparse
import datetime
import pandas as pd
import psycopg2
import psycopg2.extras
from collections import deque
from pathlib import Path
from pytz import timezone
from polygon import RESTClient
from polygon.rest.models import Agg

# 增加路径，确保能找到 baseline/config.py
sys.path.append(str(Path(__file__).resolve().parent.parent / "baseline"))
from config import PG_DB_URL, TARGET_SYMBOLS

# ================= 配置区域 =================
api_key = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
if not api_key:
    raise ValueError("请设置 POLYGON_API_KEY")

EASTERN = timezone('America/New_York')
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)
# ===========================================

class RateLimiter:
    """速率限制器：60秒内最多 max_calls 次"""
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = deque()

    def wait(self):
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_calls:
            wait_time = self.period - (now - self.timestamps[0])
            if wait_time > 0:
                print(f"    [限流] 触发频率限制，主动等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time + 1.0)
            
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > self.period:
                self.timestamps.popleft()
        
        self.timestamps.append(now)

# 初始化客户端和限流器 (Polygon 免费版 5 次/分钟, 商业版根据订阅定)
client = RESTClient(api_key=api_key)
limiter = RateLimiter(max_calls=4, period=62) 

def sync_stock_to_pg(symbol, start_date, end_date):
    """从 Polygon 下载并 UPSERT 到 PostgreSQL"""
    print(f"\n>>> 正在同步 {symbol} ({start_date} ~ {end_date})")
    
    # 限流等待
    limiter.wait()
    
    try:
        aggs = []
        for a in client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="minute",
            from_=start_date,
            to=end_date,
            limit=50000, 
            sort="asc",
            adjusted=True
        ):
            if isinstance(a, Agg):
                # 转换时区并过滤盘中
                dt_ny = datetime.datetime.fromtimestamp(a.timestamp / 1000, tz=EASTERN)
                if MARKET_OPEN <= dt_ny.time() < MARKET_CLOSE:
                    aggs.append((
                        symbol,
                        dt_ny.timestamp(),
                        float(a.open),
                        float(a.high),
                        float(a.low),
                        float(a.close),
                        float(a.volume)
                    ))
        
        if not aggs:
            print(f"    [无数据] {symbol} 在该区间无有效 Bar。")
            return

        # 执行 UPSERT
        conn = psycopg2.connect(PG_DB_URL)
        cur = conn.cursor()
        
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO market_bars_1m (symbol, ts, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, ts) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """, aggs)
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"    [成功] 已同步 {len(aggs)} 条 Bar 数据。")

    except Exception as e:
        print(f"    [错误] {symbol} 同步失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="Polygon to PostgreSQL Stock Data Sync")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, help="Comma separated symbols (default all in config)")
    args = parser.parse_args()

    target_symbols = TARGET_SYMBOLS
    if args.symbols:
        target_symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"🚀 启动股票同步任务，目标标的: {len(target_symbols)} 只")
    
    for i, symbol in enumerate(target_symbols, 1):
        print(f"[{i}/{len(target_symbols)}]", end="")
        sync_stock_to_pg(symbol, args.start_date, args.end_date)
    
    print("\n✨ 所有同步任务已完成。")

if __name__ == "__main__":
    main()