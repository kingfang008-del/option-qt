import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import sqlite3
import concurrent.futures
from tqdm import tqdm
import traceback
from datetime import datetime
from collections import defaultdict

# ================= 配置区域 =================
# 🚀 你的 1s 数据源目录
#INPUT_BASE_DIR = Path("/mnt/s990/data/raw_1s/stocks")
INPUT_BASE_DIR = Path.home() / "train_data/spnq_train"

# 降频后的输出目录
OUTPUT_BASE_DIR = Path.home() / "train_data/spnq_train_resampled"

resample_freq = {
    "pre_market": {
        "04:00-07:00": ["10s", "1min"],
        "07:00-09:30": ["1min", "5min"]
    },
    "after_hours": {
        "16:00-18:00": ["5s", "30s"],
        "18:00-20:00": ["15s", "1min"]
    },
    "regular": {
        "09:30-16:00": ["1min", "5min"]
    }
}

# 阈值设置：针对日内滞后数据的判定阈值 (建议大盘股 0.18，若包含高波妖股建议调高至 0.30)
ANTI_LAG_THRESHOLD = 0.30

# 错误日志路径
ERROR_LOG_PATH = Path(__file__).parent / "resample_errors.log"

def log_error(msg):
    """记录错误信息到专门的日志文件，便于追查。"""
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ERROR_LOG_PATH, "a") as f:
            f.write(f"[{now_str}] {msg}\n")
    except:
        pass # 静默降级

def apply_anti_lag_fix(df, threshold=ANTI_LAG_THRESHOLD):
    """反滞后修正逻辑 (Post-Resample Anti-Lag Fix)"""
    if df.empty or 'close' not in df.columns:
        return df
    
    price_series = df['close'].bfill()
    fwd_pct_change = price_series.pct_change(fill_method=None).shift(-1).abs()
    
    lag_mask = fwd_pct_change > threshold
    
    if lag_mask.any():
        cols_to_clean = ['open', 'high', 'low', 'close']
        df.loc[lag_mask, cols_to_clean] = np.nan
        
        if 'volume' in df.columns:
            df.loc[lag_mask, 'volume'] = 0
        if 'total_value' in df.columns:
            df.loc[lag_mask, 'total_value'] = 0
            
    return df

def resample_data(df, freq, time_range, symbol=None, file_name=None):
    """重采样数据，并内嵌 Anti-Lag 修复。已修复缺失列、维度一致性和时间重叠问题。"""
    # 1. 检查必备列
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        log_error(f"Missing core columns in {symbol} {file_name}. Expected: {required_cols}, Got: {list(df.columns)}")
        return pd.DataFrame()

    # 安全地补齐非强制列
    if 'vwap' not in df.columns:
        df['vwap'] = df['close']
    if 'transactions' not in df.columns:
        df['transactions'] = np.where(df['volume'] > 0, 1, 0)
        
    df = df.dropna(subset=['close'])
    if df.empty:
        return pd.DataFrame()

    df = df.set_index('timestamp').sort_index()

    try:
        # 注意：此处传入的 df 已经被提前处理成了 America/New_York 时区
        df = df.between_time(time_range[0], time_range[1], inclusive='left')
        if df.empty:
            return pd.DataFrame()
    except Exception as e:
        log_error(f"Time filter failed for {symbol} {file_name} [{time_range[0]}-{time_range[1]}]: {e}")
        return pd.DataFrame()

    # 🚀 [终极对齐] 使用 label='right' 标记结束点，使用 closed='left' 包含 00s-59s。
    df['total_value'] = df['vwap'] * df['volume']
    resampled_df = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'transactions': 'sum',
        'total_value': 'sum'
    })

    if resampled_df.index.empty:
        return pd.DataFrame()
    
    date = resampled_df.index[0].date()
    # 🚀 [核心对齐] 由于使用 label='right', closed='left'，时间序列整体右移了一个 freq
    full_time_index = pd.date_range(
        start=pd.Timestamp(f"{date} {time_range[0]}", tz=resampled_df.index.tz) + pd.Timedelta(freq),
        end=pd.Timestamp(f"{date} {time_range[1]}", tz=resampled_df.index.tz),
        freq=freq
    )
    resampled_df = resampled_df.reindex(full_time_index)
    
    # 3. 修复与填充
    resampled_df = apply_anti_lag_fix(resampled_df)

    resampled_df['total_value'] = resampled_df['total_value'].fillna(0)
    resampled_df['volume'] = resampled_df['volume'].fillna(0)
    resampled_df['transactions'] = resampled_df['transactions'].fillna(0)
    
    resampled_df['vwap'] = np.nan
    valid_volume_mask = resampled_df['volume'] > 0
    resampled_df.loc[valid_volume_mask, 'vwap'] = resampled_df.loc[valid_volume_mask, 'total_value'] / resampled_df.loc[valid_volume_mask, 'volume']
    
    resampled_df['close'] = resampled_df['close'].ffill()
    resampled_df['vwap'] = resampled_df['vwap'].ffill()
    resampled_df['open'] = resampled_df['open'].fillna(resampled_df['close'])
    resampled_df['high'] = resampled_df['high'].fillna(resampled_df['close'])
    resampled_df['low'] = resampled_df['low'].fillna(resampled_df['close'])
    resampled_df['vwap'] = resampled_df['vwap'].fillna(resampled_df['close'])
    
    resampled_df.bfill(inplace=True)

    # 4. 清理并返回
    resampled_df = resampled_df.drop(columns=['total_value'], errors='ignore')
    resampled_df = resampled_df.reset_index().rename(columns={'index': 'timestamp'})
    
    final_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
    
    # 强制转换数据格式
    for col in ['open', 'high', 'low', 'close', 'vwap']:
        resampled_df[col] = resampled_df[col].astype('float64')
    for col in ['volume', 'transactions']:
        resampled_df[col] = resampled_df[col].astype('int64')

    return resampled_df[final_cols]

def process_symbol(symbol):
    input_dir = INPUT_BASE_DIR / symbol

    if not input_dir.exists():
        log_error(f"Skipped {symbol}: Directory {input_dir} not found.")
        return f"Skipped {symbol}"
    
    # 🚀 [智能归集] 将同一年的某月的数据聚拢在一起处理，极大提升吞吐并兼容下游
    monthly_files = defaultdict(list)
    for file_path in input_dir.glob("*.parquet"):
        # 匹配形如 NVDA_2022-03-01.parquet 的文件
        match = re.match(rf"^{symbol}_(\d{{4}})-(\d{{2}})-\d{{2}}\.parquet$", file_path.name)
        if match:
            year, month = match.groups()
            monthly_files[f"{year}-{month}"].append(file_path)

    if not monthly_files:
        return f"Skipped {symbol} - No valid daily files found."

    for year_month, files in monthly_files.items():
        year, month = year_month.split('-')
        
        # 1. 读取并合并当月的所有日间文件
        dfs = []
        for f in files:
            try:
                daily_df = pd.read_parquet(f)
                dfs.append(daily_df)
            except Exception as e:
                log_error(f"Read PARQUET error for {symbol} - {f.name}: {e}")
        
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)

        # 2. 时间列鲁棒解析与时区强转 (极其重要！)
        time_col = 'timestamp' if 'timestamp' in df.columns else 'ts' if 'ts' in df.columns else 'dt'
        if time_col not in df.columns:
            log_error(f"No timestamp column in {symbol} for month {year_month}")
            continue
            
        try:
            if pd.api.types.is_numeric_dtype(df[time_col]):
                # 处理纯数字 Unix 时间戳
                df['timestamp'] = pd.to_datetime(df[time_col], unit='s', utc=True).dt.tz_convert('America/New_York')
            else:
                # 处理字符串或Datetime，补齐 UTC 并转换为美东
                df['timestamp'] = pd.to_datetime(df[time_col])
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                elif str(df['timestamp'].dt.tz) != 'America/New_York':
                    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
        except Exception as e:
            log_error(f"Time conversion error for {symbol} month {year_month}: {e}")
            continue

        # 因为源头全是 1s 数据，我们需要生成全时段的降频表
        sessions_to_process = ["pre_market", "regular", "after_hours"]

        # 3. 执行降频并按月落盘
        for session in sessions_to_process:
            if session in resample_freq:
                for time_range, freqs in resample_freq[session].items():
                    start_time, end_time = time_range.split('-')
                    for freq in freqs:
                        try:
                            # 按照美东当地日历日进行切分执行（防止跨日污染）
                            daily_groups = df.groupby(df['timestamp'].dt.date)
                            all_resampled_dfs = []

                            for d, daily_df in daily_groups:
                                resampled_day_df = resample_data(
                                    daily_df, 
                                    freq, 
                                    (start_time, end_time),
                                    symbol=symbol, 
                                    file_name=f"{year_month}_{d}"
                                )
                                # 修复维度错误: 只收集标准的返回 DataFrame
                                if not resampled_day_df.empty and len(resampled_day_df.columns) == 8:
                                    all_resampled_dfs.append(resampled_day_df)
                                elif not resampled_day_df.empty:
                                    log_error(f"Dimension Mismatch {symbol} {year_month} {d}: Columns {list(resampled_day_df.columns)}")
                            
                            if not all_resampled_dfs:
                                continue
                            
                            # 将全月的每日降频结果拼起来
                            resampled_df = pd.concat(all_resampled_dfs, ignore_index=True)

                            if not resampled_df.empty:
                                output_dir = OUTPUT_BASE_DIR / symbol / session / time_range / freq
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                # 🚀 最终输出与下游完全兼容的按月 Parquet 文件！
                                output_path = output_dir / f"{year}-{month}.parquet"
                                resampled_df.to_parquet(output_path, index=False)
                        except Exception as e:
                            log_error(f"Concat/Save error for {symbol} {year_month} {session} {freq}: {e}\n{traceback.format_exc()}")

    return f"Processed {symbol}"

def main():
    Path("../data").mkdir(exist_ok=True)
    
    # 确保存放错误日志的父目录存在
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ERROR_LOG_PATH.exists():
        ERROR_LOG_PATH.write_text("")
    
    try:
        db_path = Path.home() / "notebook/stocks.db"
        if not db_path.exists():
             raise FileNotFoundError("Database not found")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 使用动态导入 Target List
        try:
            from config import TARGET_SYMBOLS
            symbols_list = TARGET_SYMBOLS
        except ImportError:
            symbols_list = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META', 'GOOGL', 'SMCI', 'AMD']
            
        placeholders = ','.join(['?'] * len(symbols_list))
        query = f"SELECT symbol, level FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, symbols_list)
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"无法从数据库加载股票列表 ({e})，将尝试扫描源目录提取。")
        if INPUT_BASE_DIR.exists():
            symbols = [d.name for d in INPUT_BASE_DIR.iterdir() if d.is_dir()]
        else:
            symbols = []

    if not symbols:
        print("未获取到任何股票代码，程序退出。")
        return

    max_workers = os.cpu_count()
    print(f"共找到 {len(symbols)} 个股票代码。使用 {max_workers} 个进程开始并行归集处理...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_symbol, symbols), total=len(symbols)))

    processed_count = sum(1 for r in results if r.startswith("Processed"))
    skipped_count = len(results) - processed_count
    print(f"\n所有股票数据处理完成！成功处理: {processed_count}, 跳过/失败: {skipped_count}")
    print(f"详细错误日志已记录至: {ERROR_LOG_PATH}")

if __name__ == "__main__":
    main()