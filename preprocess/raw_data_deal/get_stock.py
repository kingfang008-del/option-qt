import argparse
import datetime
import pandas as pd
import sqlite3
import os
import time
from collections import deque
from polygon import RESTClient
from polygon.rest.models import Agg
from pytz import timezone

# ================= 配置区域 =================
api_key = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
if not api_key:
    raise ValueError("请设置 POLYGON_API_KEY")

# 必须设置时区以正确过滤盘中时间
eastern = timezone('America/New_York')

# 只要盘中数据，定义时间范围
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)

# 目标日期范围
start_date = datetime.date(2026, 3, 1)
end_date = datetime.date(2026,  3, 31)

# 是否覆盖已有按月 parquet（YYYY-MM.parquet）
# False: 已存在且非空的月份跳过写入；若区间内月份都齐则整票跳过
# True:  强制重下并覆盖已有月份文件
overwrite = False

data_root = "/home/kingfang007/train_data/spnq_train"
db_path = "/home/kingfang007/notebook/stocks.db"
# ===========================================

class RateLimiter:
    """速率限制器：60秒内最多 max_calls 次"""
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = deque()

    def wait(self):
        now = time.time()
        # 清理过期时间戳
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()

        # 检查是否达到上限
        if len(self.timestamps) >= self.max_calls:
            wait_time = self.period - (now - self.timestamps[0])
            if wait_time > 0:
                print(f"    [限流] 触发频率限制，主动等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time + 1.5) # 多睡1.5秒缓冲，确保不误触
            
            # 醒来后再次清理
            now = time.time()
            while self.timestamps and now - self.timestamps[0] > self.period:
                self.timestamps.popleft()
        
        self.timestamps.append(now)

# 初始化客户端和限流器
client = RESTClient(api_key=api_key)
# 保守设置：每分钟只允许 4 次请求 (留1次作为容错余量)
limiter = RateLimiter(max_calls=4, period=62) 

def get_target_stocks():
    #from config import TARGET_SYMBOLS
    #TARGET_SYMBOLS=['VIXY','QQQ','NVDA', 'TSLA', 'AMD', 'INTC', 'MSFT', 'AMZN', 'GOOG', 'META', 'AAPL' ]
    TARGET_SYMBOLS=[ 'GOOGL']
     
    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()t
    # query = """
    #     SELECT symbol, level
    #     FROM stocks_us 
    #     WHERE level IN ('spnq') 
    #       AND sector IS NOT NULL 
    #       AND sector != 'Unknown'
    # """
    # query = """
    #     SELECT symbol, level
    #     FROM stocks_us 
    #     WHERE  symbol= 'VIXY'
    # """
    
     
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 建议使用的 Top 50 训练白名单 (按流动性降序)
    
     
    
     
     # 动态生成占位符并执行查询
    placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
    query = f"SELECT symbol, level FROM stocks_us WHERE symbol IN ({placeholders})"
    
    cursor.execute(query, TARGET_SYMBOLS)

    stocks = cursor.fetchall()
     
    conn.close()
    return stocks

def generate_month_ranges(start, end):
    """生成区间内涉及的所有月份字符串 (YYYY-MM)"""
    months = []
    curr = start
    while curr <= end:
        months.append(curr.strftime("%Y-%m"))
        if curr.month == 12:
            curr = datetime.date(curr.year + 1, 1, 1)
        else:
            curr = datetime.date(curr.year, curr.month + 1, 1)
    return months

def month_file_ready(f_path):
    """按月文件是否已存在且非空"""
    return os.path.exists(f_path) and os.path.getsize(f_path) > 0

def check_is_completed(symbol, start_dt, end_dt):
    """
    检查某股票是否已经全部完成下载
    返回: True (已完成，可跳过), False (未完成)
    """
    symbol_dir = os.path.join(data_root, symbol)
    if not os.path.exists(symbol_dir):
        return False
        
    target_months = generate_month_ranges(start_dt, end_dt)
    for m in target_months:
        f_path = os.path.join(symbol_dir, f"{m}.parquet")
        # 只要有一个月不存在或为空，就视为未完成
        if not month_file_ready(f_path):
            return False
    return True

def download_stock_bulk_safe(symbol, level, start_dt, end_dt, overwrite=False):
    """带重试机制的下载函数。overwrite=True 时覆盖已有按月 parquet。"""
    symbol_dir = os.path.join(data_root, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    target_months = generate_month_ranges(start_dt, end_dt)

    # 非覆盖模式：只补缺月；若全部已齐则直接返回
    if not overwrite:
        missing_months = [
            m for m in target_months
            if not month_file_ready(os.path.join(symbol_dir, f"{m}.parquet"))
        ]
        if not missing_months:
            print(f"\n>>> 跳过 {symbol}（按月文件已齐全）")
            return
        print(f"\n>>> 开始处理 {symbol}（补缺月份: {', '.join(missing_months)}）")
    else:
        print(f"\n>>> 开始处理 {symbol}（覆盖模式，将重写月份文件）")

    # === 重试循环：处理网络错误或429 ===
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 每次尝试前先过限流器
            limiter.wait()
            
            print(f"    [请求] {symbol} ({start_dt} ~ {end_dt}) 第 {attempt+1} 次尝试...")
            
            # --- API 请求 ---
            aggs = []
            for a in client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_=start_dt.strftime("%Y-%m-%d"),
                to=end_dt.strftime("%Y-%m-%d"),
                limit=50000, 
                sort="asc",
                adjusted=True
            ):
                if isinstance(a, Agg):
                    ts = datetime.datetime.fromtimestamp(a.timestamp / 1000, tz=eastern)
                    t_time = ts.time()
                    # 内存过滤：只保留盘中
                    if t_time >= MARKET_OPEN and t_time < MARKET_CLOSE:
                        aggs.append({
                            "timestamp": ts,
                            "open": a.open,
                            "high": a.high,
                            "low": a.low,
                            "close": a.close,
                            "volume": a.volume,
                            "vwap": a.vwap,
                            "transactions": a.transactions
                        })
            
            # --- 数据处理与保存 ---
            if not aggs:
                print(f"    [无数据] API返回为空，生成空文件占位")
                for m in target_months:
                    f_p = os.path.join(symbol_dir, f"{m}.parquet")
                    # 覆盖模式：已有文件也重写成空占位；否则仅补缺
                    if overwrite or not os.path.exists(f_p):
                        open(f_p, 'w').close()
            else:
                df_all = pd.DataFrame(aggs)
                df_all['month_grp'] = df_all['timestamp'].dt.strftime('%Y-%m')
                grouped = df_all.groupby('month_grp')
                
                saved_months = set()
                for month_str, group_df in grouped:
                    output_file = os.path.join(symbol_dir, f"{month_str}.parquet")
                    existed = month_file_ready(output_file)
                    if (not overwrite) and existed:
                        print(f"    [跳过] {month_str} 已存在，不覆盖")
                        saved_months.add(month_str)
                        continue
                    group_df.drop(columns=['month_grp']).to_parquet(output_file, index=False)
                    saved_months.add(month_str)
                    action = "覆盖" if existed else "保存"
                    print(f"    [{action}] {month_str} -> {len(group_df)} 条")
                
                # 补全缺失月份为空文件
                for m in target_months:
                    if m not in saved_months:
                        f_p = os.path.join(symbol_dir, f"{m}.parquet")
                        if overwrite or not os.path.exists(f_p):
                            open(f_p, 'w').close()
                            print(f"    [补全] {m} 无盘中数据")

            # 成功执行完逻辑，跳出重试循环
            return 

        except Exception as e:
            err_msg = str(e)
            print(f"    [异常] {err_msg}")
            
            # 检查是否是 429 Too Many Requests
            if "429" in err_msg or "Too Many Requests" in err_msg:
                print("    !!! 触发严重限流，强制休眠 65 秒后重试 !!!")
                time.sleep(65)
                continue # 进入下一次 for 循环重试
            else:
                # 如果是其他严重错误（如鉴权失败），可能重试也没用，但为了稳健还是等一会
                if attempt < max_retries - 1:
                    print("    等待 5 秒后重试...")
                    time.sleep(5)
                else:
                    print(f"    [放弃] {symbol} 多次重试失败")

def parse_args():
    parser = argparse.ArgumentParser(description="按月下载股票分钟数据")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有按月 parquet（YYYY-MM.parquet），强制重新下载",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    do_overwrite = overwrite or args.overwrite

    # 1. 获取所有待处理股票
    all_stocks = get_target_stocks()
    print(f"数据库中符合条件的股票共: {len(all_stocks)} 只")
    print(f"覆盖模式: {'开启' if do_overwrite else '关闭'}")

    # 2. 【预检查】过滤掉已经下载完成的股票（覆盖模式不跳过）
    if do_overwrite:
        pending_stocks = list(all_stocks)
        print("覆盖模式：不跳过已完成股票，将重写按月文件。")
    else:
        print("正在检查本地文件，过滤已完成的任务...")
        pending_stocks = []
        for sym, lvl in all_stocks:
            if not check_is_completed(sym, start_date, end_date):
                pending_stocks.append((sym, lvl))
        
        skip_count = len(all_stocks) - len(pending_stocks)
        print(f"已跳过 {skip_count} 只已完成的股票。")

    print(f"剩余 {len(pending_stocks)} 只股票需要下载。\n")

    # 3. 处理剩余股票
    for i, (symbol, level) in enumerate(pending_stocks, 1):
        print(f"总进度 {i}/{len(pending_stocks)}", end=" ")
        download_stock_bulk_safe(symbol, level, start_date, end_date, overwrite=do_overwrite)
    
    print("\n所有任务全部完成。")
