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

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_target_symbols(db_path: str) -> set:
    """从数据库获取目标股票列表"""
    expanded_db_path = os.path.expanduser(db_path)
    if not os.path.exists(expanded_db_path):
        print(f"错误：数据库文件不存在于 {expanded_db_path}")
        return set()
        
    try:
        conn = sqlite3.connect(expanded_db_path)
        cursor = conn.cursor()
        from config import TARGET_SYMBOLS
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT symbol FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, TARGET_SYMBOLS)
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return set(symbols)
    except Exception as e:
        print(f"读取数据库时发生错误: {e}")
        return set()

def process_single_symbol(args):
    """
    [纯粹版 Worker] 合并单日的 Greeks 文件为月度文件
    """
    symbol, all_daily_files, output_base_path = args

    try:
        if not all_daily_files:
            return f"[Skip] {symbol}: 无每日文件"

        # 1. 解析文件名，按年月进行分组
        monthly_groups = defaultdict(list)
        for file in all_daily_files:
            filename = os.path.basename(file)
            try:
                # 例如: AMD_2026-02-26.parquet -> 提取 2026-02
                # 假设格式是 SYMBOL_YYYY-MM-DD.parquet，最后一部分通常是日期
                date_part = filename.replace('.parquet', '').split('_')[-1]
                if len(date_part) >= 7:
                    year_month = date_part[:7] 
                    monthly_groups[year_month].append(file)
            except Exception:
                continue
        
        # 2. 目标输出目录 (保持原来的输出结构: <symbol>/standard/)
        output_dir = os.path.join(output_base_path, symbol, "standard")
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        
        # 3. 按月合并并保存
        for year_month, files in monthly_groups.items():
            output_path = os.path.join(output_dir, f"{year_month}.parquet")
            
            dfs = []
            for file in files:
                try:
                    df = pd.read_parquet(file)
                    if not df.empty:
                        dfs.append(df)
                except Exception:
                    pass
            
            if not dfs: continue
            
            # 垂直拼接该月所有天数的数据
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # ================= [核心时区与排序清洗] =================
            if 'timestamp' in merged_df.columns:
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
                
                if merged_df['timestamp'].dt.tz is None:
                     merged_df['timestamp'] = merged_df['timestamp'].dt.tz_localize('America/New_York', ambiguous='infer')
                else:
                     merged_df['timestamp'] = merged_df['timestamp'].dt.tz_convert('America/New_York')
                
                # 排序并去重（防止极其偶发的重复）
                # 注意：如果 df 特别大，这里可能会占用较多内存
                merged_df.sort_values(by=['timestamp', 'bucket_id'], inplace=True)
                merged_df.drop_duplicates(subset=['timestamp', 'bucket_id'], keep='last', inplace=True)
                
            if 'expiration_date' in merged_df.columns:
                merged_df['expiration_date'] = pd.to_datetime(merged_df['expiration_date'])
                if merged_df['expiration_date'].dt.tz is None:
                     merged_df['expiration_date'] = merged_df['expiration_date'].dt.tz_localize('America/New_York', ambiguous='infer')
                else:
                     merged_df['expiration_date'] = merged_df['expiration_date'].dt.tz_convert('America/New_York')
            # ==================================================================

            try:
                merged_df.to_parquet(output_path, index=False, compression='zstd')
                success_count += 1
            except Exception as e:
                print(f"Error saving {output_path}: {e}")
        
        return f"[Success] {symbol}: 成功合并了 {success_count} 个月份"

    except Exception as e:
        import traceback
        return f"[Error] {symbol}: {e}\n{traceback.format_exc()}"


def main():
    # ---------- 路径配置 ----------
    # 输入: option_cac_day_vectorized 输出的带 Greeks 的日级 Parquet 文件夹
    INPUT_BASE_PATH = "/home/kingfang007/train_data/quote_options_day_iv"
    
    # 输出: 按月合并好的文件夹
    OUTPUT_BASE_PATH = "/home/kingfang007/train_data/quote_options_monthly_iv"
    
    DB_PATH = "/home/kingfang007/notebook/stocks.db"
    MAX_WORKERS = 16
    # -----------------------------

    target_symbols = get_target_symbols(DB_PATH)
    if not target_symbols: return
    
    # 1. 查找所有 Parquet 文件 (修改为递归查找，以适配 symbol/standard/ 子目录结构)
    # 路径通常为: INPUT_BASE_PATH/<symbol>/standard/<symbol>_<date>.parquet
    all_files = glob.glob(os.path.join(INPUT_BASE_PATH, "**", "*.parquet"), recursive=True)
    
    # 2. 按 Symbol 分组
    symbol_to_files = defaultdict(list)
    for f in all_files:
        filename = os.path.basename(f)
        try:
            # 格式: SYMBOL_YYYY-MM-DD.parquet -> 提取 SYMBOL
            parts = filename.replace('.parquet', '').split('_')
            if len(parts) >= 2:
                symbol = "_".join(parts[:-1])
                # 校验是否在目标列表中
                if symbol in target_symbols:
                    symbol_to_files[symbol].append(f)
        except Exception:
            continue
            
    tasks = []
    for symbol, files in symbol_to_files.items():
        tasks.append((symbol, files, OUTPUT_BASE_PATH))
            
    print(f"🚀 准备处理 {len(tasks)} 个股票的日转月合并 (纯净版)，并发数: {MAX_WORKERS}...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_symbol, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="合并月度文件进度"):
            result = future.result()

    print("\n✅ 日转月合并完成！现在你可以运行 options_locked_feature.py 了。")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()