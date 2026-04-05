import os
import csv
import httpx
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from tqdm import tqdm
import concurrent.futures

# ================= 配置参数 =================
BASE_URL = "http://127.0.0.1:25503/v3"

# ⚠️ 诊断建议: 先用 AAPL 测 1 天的数据！
from config import TARGET_SYMBOLS   

# ⚠️ 诊断建议: 如果你是免费账户，务必把这里改成最近的几天，否则 2023 年必为空！
START_DATE = datetime(2026, 3, 1) 
END_DATE = datetime(2026, 3, 7)

DTE_MIN = 3     
DTE_MAX = 65    

OUTPUT_DIR = "/mnt/s990/data/quote_option_data_2603"
MAX_WORKERS = 4 
# ============================================

def is_rth(ts_series):
    time_series = ts_series.dt.time
    return (time_series >= dt_time(9, 30)) & (time_series < dt_time(16, 0))

def fetch_and_process_daily(args):
    symbol, target_date = args
    # 官方要求 YYYYMMDD 格式
    date_str = target_date.strftime("%Y%m%d")
    out_date_str = target_date.strftime("%Y-%m-%d")

    sym_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(sym_dir, exist_ok=True)
    out_path = os.path.join(sym_dir, f"{symbol}_{out_date_str}.parquet")

    if os.path.exists(out_path):
        return f"⏩ {symbol} {out_date_str}: 已存在，跳过。"

    # 完全严格对齐官方文档参数
    params = {
        'symbol': symbol,
        'expiration': '*',    # 请求所有到期日
        'right': 'both',      # 请求 Call 和 Put
        'date': date_str,     # 精确指定单日
        'interval': '1m',     # 让服务器聚合 1 分钟快照
        'format': 'csv'
    }
    
    url = f"{BASE_URL}/option/history/quote"

    try:
        data = []
        header = None
        
        # 增加 Timeout 到 120 秒，防止 SPY 等巨无霸超时
        with httpx.stream("GET", url, params=params, timeout=120.0) as response:
            
            # 🚨 强力诊断 1：拦截非 200 状态码
            if response.status_code != 200:
                raw_err = response.read().decode('utf-8', errors='ignore')
                return f"❌ {symbol} {out_date_str}: 请求失败 (HTTP {response.status_code}) -> {raw_err[:100]}"
                
            # 逐行读取，忽略空行
            lines = (line.strip() for line in response.iter_lines() if line.strip())
            reader = csv.reader(lines)

            try:
                # 尝试获取 CSV 表头
                header = next(reader)
                
                # 🚨 强力诊断 2：服务器可能返回了一条文字报错信息而不是 CSV 表头
                if len(header) == 1 and "error" in header[0].lower():
                    return f"❌ {symbol} {out_date_str}: 服务器业务报错 -> {header[0]}"
                    
            except StopIteration:
                return f"⚠️ {symbol} {out_date_str}: 服务器返回了 200 OK，但内容完全为空！(可能是该日期没开盘，或无权限)"

            # 读取剩余数据
            for row in reader:
                data.append(row)

        if not data:
            return f"⚠️ {symbol} {out_date_str}: 仅返回了表头 {header}，无实际期权数据。"

        # =======================================================
        # 此时已经拿到了数据，进入 Pandas 处理环节
        # =======================================================
        df = pd.DataFrame(data, columns=header)
        
        # 🚨 强力诊断 3：检查表头是否包含我们需要的基础字段
        required_cols = ['strike', 'bid', 'ask', 'bid_size', 'ask_size', 'timestamp', 'expiration', 'right']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return f"❌ {symbol} {out_date_str}: CSV 格式改变！缺少列: {missing_cols}。当前表头: {header}"

        # 强制转换为数值类型 (忽略解析错误的垃圾行)
        for col in ['strike', 'bid', 'ask', 'bid_size', 'ask_size']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=['strike', 'bid', 'ask'], inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
        df.dropna(subset=['timestamp', 'expiration'], inplace=True)

        # A. 时间裁剪 (09:30 - 16:00)
        df = df[is_rth(df['timestamp'])]
        if df.empty: return f"⚠️ {symbol} {out_date_str}: RTH (交易时段) 期间无有效数据。"

        # B. 期限裁剪
        df['dte'] = (df['expiration'].dt.date - df['timestamp'].dt.date).dt.days
        df = df[(df['dte'] >= DTE_MIN) & (df['dte'] <= DTE_MAX)]
        if df.empty: return f"⚠️ {symbol} {out_date_str}: DTE 裁剪后为空。"

        # C. 异常盘口清洗
        df = df[
            (df['bid'] > 0.0) & 
            (df['ask'] >= df['bid']) &
            (df['bid_size'] > 0) & 
            (df['ask_size'] > 0)
        ].copy()
        if df.empty: return f"⚠️ {symbol} {out_date_str}: 盘口清洗后为空。"

        # D. 衍生微观特征
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
        df['close'] = df['mid_price']
        df['open'] = df['mid_price']
        df['high'] = df['mid_price']
        df['low'] = df['mid_price']
        df['volume'] = df['bid_size'] + df['ask_size']

        exp_str = df['expiration'].dt.strftime('%y%m%d')
        strike_str = (df['strike'] * 1000).astype(int).astype(str).str.zfill(8)
        right_occ = df['right'].str[0].str.upper()
        df['ticker'] = symbol + exp_str + right_occ + strike_str

        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']

        cols_to_keep = [
            'timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume',
            'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance'
        ]
        df = df[cols_to_keep]

        df.to_parquet(out_path, engine='pyarrow', index=False)
        return f"✅ {symbol} {out_date_str}: 成功! 写入 {len(df):,} 行"

    except httpx.ReadTimeout:
        return f"🚨 {symbol} {out_date_str}: 请求超时 (可能是 expiration='*' 数据量过大，尝试调大 timeout)"
    except Exception as e:
        import traceback
        return f"🚨 {symbol} {out_date_str}: 崩溃 - {str(e)} \n {traceback.format_exc()}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dates_to_run = []
    curr = START_DATE
    while curr <= END_DATE:
        if curr.weekday() < 5:
            dates_to_run.append(curr)
        curr += timedelta(days=1)
    
    tasks = [(sym, d) for sym in TARGET_SYMBOLS for d in dates_to_run]
    
    print(f"🚀 开始 HTTP 并发下载测试，任务数: {len(tasks)}")
    
    results = []
    # 如果总是失败，可以先把 max_workers 设为 1，逐个排查
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_process_daily, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            results.append(future.result())

    print("\n" + "="*50)
    print("📊 执行报告")
    for res in results:
        print(res)
    print("==================================================")

if __name__ == "__main__":
    main()