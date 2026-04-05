import os
import glob
import pandas as pd
from pathlib import Path
import sqlite3
from collections import defaultdict
from tqdm import tqdm 
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# [配置] 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_target_symbols(db_path: str) -> set:
    """从数据库获取目标股票列表，返回集合以提高查找速度"""
    expanded_db_path = os.path.expanduser(db_path)
    if not os.path.exists(expanded_db_path):
        print(f"錯誤：資料庫文件不存在於 {expanded_db_path}")
        return set()
        
    try:
        conn = sqlite3.connect(expanded_db_path)
         
        cursor = conn.cursor()
        # 建议使用的 Top 50 训练白名单 (按流动性降序)
        from config import TARGET_SYMBOLS  
         # 动态生成占位符并执行查询
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT  symbol  FROM stocks_us WHERE symbol IN ({placeholders})"
        
        cursor.execute(query, TARGET_SYMBOLS)
     
        symbols = [row[0] for row in cursor.fetchall()]

        conn.close()
        print(f"從資料庫找到 {len(symbols)} 支目標股票。")
        return symbols
    except sqlite3.Error as e:
        print(f"讀取資料庫時發生錯誤: {e}")
        return set()

def process_single_symbol(args):
    """
    [Worker 函数] 处理单个股票的所有月份合并任务
    """
    symbol_dir, output_base_path, target_symbols = args
    symbol = os.path.basename(symbol_dir)

    # 过滤非目标股票
    if symbol not in target_symbols:
        return None

    try:
        # 查找所有相关文件
        all_daily_files = glob.glob(os.path.join(symbol_dir, f"{symbol}_*.parquet"))
        if not all_daily_files:
            return f"[Skip] {symbol}: 无每日文件"

        standard_files = []
        high_features_files = []
        
        for file in all_daily_files:
            if file.endswith("_high_features.parquet"):
                high_features_files.append(file)
            else:
                standard_files.append(file)
        
        file_groups = {
            "standard": {"files": standard_files, "output_subdir": "standard"},
            "high_features": {"files": high_features_files, "output_subdir": "high_features"}
        }

        results_info = []

        for file_type, group_config in file_groups.items():
            daily_files = group_config["files"]
            if not daily_files:
                continue
            
            # 按月份分组
            monthly_groups = defaultdict(list)
            for file in daily_files:
                filename = os.path.basename(file)
                try:
                    # 解析日期: SYMBOL_YYYYMMDD.parquet
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        date_part = parts[1].split('.')[0]
                        if len(date_part) == 8 and date_part.isdigit():
                            year_month = date_part[:6] # YYYYMM
                            year_month_fmt = f"{year_month[:4]}-{year_month[4:]}"
                            monthly_groups[year_month_fmt].append(file)
                except Exception:
                    continue
            
            output_dir = os.path.join(output_base_path, symbol, group_config['output_subdir'])
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理每个月
            for year_month, files in monthly_groups.items():
                output_path = os.path.join(output_dir, f"{year_month}.parquet")
                
                # 如果需要跳过已存在的文件，可以在这里加判断
                # if os.path.exists(output_path): continue

                dfs = []
                for file in files:
                    try:
                        df = pd.read_parquet(file)
                        if not df.empty:
                            dfs.append(df)
                    except Exception:
                        pass
                
                if not dfs:
                    continue
                
                try:
                    merged_df = pd.concat(dfs, ignore_index=True)
                except Exception as e:
                    print(f"Concat error for {symbol} {year_month}: {e}")
                    continue
                
                # ================= [核心修复: 统一到 New York 时区] =================
                if 'timestamp' in merged_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(merged_df['timestamp']):
                        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True)
                    
                    if merged_df['timestamp'].dt.tz is None:
                         merged_df['timestamp'] = merged_df['timestamp'].dt.tz_localize('UTC')
                    else:
                         merged_df['timestamp'] = merged_df['timestamp'].dt.tz_convert('UTC')
                    
                    merged_df['timestamp'] = merged_df['timestamp'].dt.tz_convert('America/New_York')
                    merged_df.sort_values(by='timestamp', inplace=True)

                if 'expiration_date' in merged_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(merged_df['expiration_date']):
                        merged_df['expiration_date'] = pd.to_datetime(merged_df['expiration_date'], utc=True)
                    
                    if merged_df['expiration_date'].dt.tz is None:
                        merged_df['expiration_date'] = merged_df['expiration_date'].dt.tz_localize('UTC')
                    else:
                        merged_df['expiration_date'] = merged_df['expiration_date'].dt.tz_convert('UTC')
                        
                    merged_df['expiration_date'] = merged_df['expiration_date'].dt.tz_convert('America/New_York')
                # ==================================================================

                try:
                    merged_df.to_parquet(output_path, index=False, compression='zstd', engine='pyarrow')
                except Exception as e:
                    print(f"Error saving {output_path}: {e}")
        
        return f"[Success] {symbol}"

    except Exception as e:
        import traceback
        return f"[Error] {symbol}: {e}\n{traceback.format_exc()}"

def merge_monthly_options_data_multiprocess(base_path="~/train_data/quote_options_day_iv", 
                                            output_base_path="~/train_data/quote_options_monthly_iv",  
                                            db_path="~/notebook/stocks.db",
                                            max_workers=10):
    """
    [多进程版] 并行执行月度合并任务
    """
    base_path = os.path.expanduser(base_path)
    output_base_path = os.path.expanduser(output_base_path)

    # 1. 获取目标股票
    target_symbols = get_target_symbols(db_path)
    if not target_symbols:
        print("未能獲取目標股票列表，程序終止。")
        return
    
    # 2. 扫描所有股票目录
    all_symbol_dirs = [d for d in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(d)]
    
    # 3. 准备任务列表
    # 仅保留在 target_symbols 中的目录
    tasks = []
    for d in all_symbol_dirs:
        symbol = os.path.basename(d)
        if symbol in target_symbols:
            tasks.append((d, output_base_path, target_symbols))
            
    print(f"准备处理 {len(tasks)} 个股票目录，使用 {max_workers} 个进程并发...")

    # 4. 多进程执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示总体进度
        futures = {executor.submit(process_single_symbol, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Total Progress"):
            result = future.result()
            # 可以选择打印结果或记录日志
            # if result and "[Error]" in result:
            #     print(result)

if __name__ == "__main__":
    # 必须加这个，防止多进程递归启动
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass

    database_path = "~/notebook/stocks.db"
    
    # 根据机器配置调整 max_workers，建议设置为 CPU 核数 - 2
    # 对于 IO 密集型任务，可以设得稍大一些，例如 10-20
    merge_monthly_options_data_multiprocess(
        db_path=database_path, 
        max_workers=10
    )
    
    print("\n所有股票的月度数据合并完成。")