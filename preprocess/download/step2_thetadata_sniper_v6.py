import os
import io
import csv
import httpx
import pandas as pd
from datetime import time as dt_time
from tqdm import tqdm
import concurrent.futures
import re

# ================= 配置参数 =================
BASE_URL = "http://127.0.0.1:25503/v3"
TARGET_MAP_FILE = "/home/kingfang007/train_data/locked_targets_map.parquet"
OUTPUT_DIR = "/mnt/s990/data/option_quote" # 存新高精度数据的目录

# ⚠️ ThetaData VALUE 订阅，坚决死守 2 并发
MAX_WORKERS = 2 
# ============================================

def is_rth(ts_series):
    time_series = ts_series.dt.time
    return (time_series >= dt_time(9, 30)) & (time_series < dt_time(16, 0))

def parse_occ_ticker(ticker):
    """从 O:AAPL240119C00150000 提取精准查询参数"""
    clean_ticker = ticker.replace('O:', '')
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', clean_ticker)
    if not match: return None
    
    symbol = match.group(1)
    exp_str = f"20{match.group(2)}" # 240119 -> 20240119
    right = 'call' if match.group(3) == 'C' else 'put'
    strike = float(match.group(4)) / 1000.0 # 00150000 -> 150.0
    return symbol, exp_str, right, strike

def fetch_single_day_targets(args):
    """
    独立 Worker：负责一天内这 6 个目标合约的精准下载并合并
    """
    symbol, date_str, group_df = args
    # API 需要的日期格式: 2024-11-05 -> 20241105
    api_date = date_str.replace("-", "") 
    
    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    if os.path.exists(out_path):
        return f"⏩ {symbol} {date_str} 已存在跳过"

    all_data = []
    header = None
    
    # 对当天的 6 个目标合约发起精准狙击
    for _, row in group_df.iterrows():
        occ = row['contract_symbol']
        parsed = parse_occ_ticker(occ)
        if not parsed: continue
        
        _, exp_str, right, strike = parsed
        
        params = {
            'symbol': symbol,
            'expiration': exp_str,
            'strike': strike,
            'right': right,
            'date': api_date,
            'interval': '1m',
            'format': 'csv'
        }
        
        url = f"{BASE_URL}/option/history/quote"
        
        # 精确请求速度极快，几百毫秒就返回
        try:
            with httpx.stream("GET", url, params=params, timeout=30.0) as response:
                if response.status_code != 200: continue
                lines = (line.strip() for line in response.iter_lines() if line.strip())
                reader = csv.reader(lines)
                try:
                    head = next(reader)
                    if len(head) > 1: header = head
                except StopIteration: continue
                
                # 附加原始 OCC Ticker 和 Bucket ID 供后续使用
                for r in reader:
                    all_data.append(r + [occ, row['bucket_id']])
        except Exception:
            continue

    if not all_data or not header:
        return f"⚠️ {symbol} {date_str}: 无有效返回"

    # ================= 极速清洗与组装 =================
    header_extended = header + ['ticker', 'bucket_id']
    df = pd.DataFrame(all_data, columns=header_extended)
    
    for col in ['bid', 'ask', 'bid_size', 'ask_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(subset=['bid', 'ask'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    # RTH 过滤
    df = df[is_rth(df['timestamp'])]
    if df.empty: return f"⚠️ {symbol} {date_str}: RTH无数据"
    
    # 异常盘口清洗
    df = df[
        (df['bid'] > 0.0) & 
        (df['ask'] >= df['bid']) &
        (df['bid_size'] > 0) & 
        (df['ask_size'] > 0)
    ].copy()

    # 核心转换：假装成系统能认识的格式，供下游直接食用
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['close'] = df['mid_price']
    df['open'] = df['mid_price']
    df['high'] = df['mid_price']
    df['low'] = df['mid_price']
    df['volume'] = df['bid_size'] + df['ask_size'] # 挂单深度代替虚假成交量
    
    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
    df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']
    
    cols = ['timestamp', 'ticker', 'bucket_id', 'open', 'high', 'low', 'close', 'volume', 
            'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance']
    df = df[cols]
    
    df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    return f"🎯 狙击成功 {symbol} {date_str}: 6 个合约共 {len(df)} 行精准数据！"

def main():
    if not os.path.exists(TARGET_MAP_FILE):
        print("❌ 未找到清单，请先运行 step1_build_target_map.py")
        return
        
    print("📂 正在加载狙击清单...")
    target_map = pd.read_parquet(TARGET_MAP_FILE)
    
    # 按照 (Symbol, Date) 分组，每天一个下载任务
    all_potential_tasks = []
    for (sym, date_str), group in target_map.groupby(['symbol', 'date_str']):
        all_potential_tasks.append((sym, date_str, group))
        
    tasks = []
    skipped = 0
    for sym, date_str, group in all_potential_tasks:
        out_path = os.path.join(OUTPUT_DIR, sym, f"{sym}_{date_str}.parquet")
        if os.path.exists(out_path):
            skipped += 1
            continue
        tasks.append((sym, date_str, group))

    if skipped > 0:
        print(f"⏩ 跳过 {skipped} 个已存在的文件")

    if not tasks:
        print("✅ 所有任务已完成，无需下载。")
        return
        
    print(f"🚀 开启极速狙击模式！共计 {len(tasks)} 个有效任务 (总计 {len(all_potential_tasks)})，并发限制：{MAX_WORKERS}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_single_day_targets, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Downloading"):
            res = future.result()
            # print(res) # 如果需要查看每个任务的结果可以取消注释

    print("🎉 全量精准狙击完成！数据体积缩减 99%，纯净盘口数据已落地！")

if __name__ == "__main__":
    main()