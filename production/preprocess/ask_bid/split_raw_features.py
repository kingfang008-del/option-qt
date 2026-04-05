import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
import sqlite3  # [新增] 导入 sqlite3

# --- 1. 全局配置 (请根据您的实际情况修改) ---

# 【必须修改】指定您存放未标准化特征的源目录
SOURCE_DIR = Path.home() / "train_data/quote_features_raw" 

# 【必须修改】定义新的、分离的训练集和验证集输出目录
TRAIN_DIR = Path.home() / "train_data/quote_features_train"
VAL_DIR = Path.home() / "train_data/quote_features_val"
TEST_DIR = Path.home() / "train_data/quote_features_test"

# 定义训练集、验证集和测试集的时间范围 (左闭右闭)
TRAIN_DATE_RANGE = ("2022-03-01", "2025-06-30") 
VAL_DATE_RANGE = ("2025-07-01", "2025-12-31")
TEST_DATE_RANGE = ("2026-01-01", "2026-03-18")

# 使用的CPU核心数
MAX_WORKERS = 32

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_destination_path(file_path: Path, source_base: Path, train_base: Path, val_base: Path, test_base: Path,
                         train_range: Tuple[pd.Timestamp, pd.Timestamp], 
                         val_range: Tuple[pd.Timestamp, pd.Timestamp],
                         test_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Path | None:
    """
    根据文件路径中的年月判断其应被分发到哪个目录。
    """
    try:
        # 从文件名 'YYYY-MM.parquet' 中提取年月
        year_month_str = file_path.stem
        file_date = pd.to_datetime(year_month_str + "-01")

        # 判断文件日期属于哪个范围
        dest_base = None
        if train_range[0] <= file_date <= train_range[1]:
            dest_base = train_base
        elif val_range[0] <= file_date <= val_range[1]:
            dest_base = val_base
        elif test_range[0] <= file_date <= test_range[1]:
            dest_base = test_base
        
        if dest_base:
            # 获取文件相对于源目录的路径
            relative_path = file_path.relative_to(source_base)
            return dest_base / relative_path
        
        return None 

    except ValueError:
        logging.warning(f"无法从文件名解析日期: {file_path.name}，将跳过此文件。")
        return None

def process_and_copy_file(file_path: Path, source_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path,
                          train_range_ts: Tuple, val_range_ts: Tuple, test_range_ts: Tuple) -> str:
    """
    单个文件处理的工作函数
    """
    dest_path = get_destination_path(file_path, source_dir, train_dir, val_dir, test_dir, 
                                     train_range_ts, val_range_ts, test_range_ts)
    
    if dest_path:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            return f"[成功] {file_path.name}"
        except Exception as e:
            return f"[错误] {e}"
    else:
        return f"[跳过]"

def get_valid_symbols() -> set:
    """
    [新增] 从数据库获取有效的股票代码列表
    """
    db_path = Path.home() / "notebook/stocks.db"
    
    if not db_path.exists():
        logging.error(f"数据库文件未找到: {db_path}")
        return set()

    logging.info(f"正在读取股票白名单: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor() #level IN ('sp500', 'nq100', 'spnq', 'nq')
        from config import TARGET_SYMBOLS
             # 动态生成占位符并执行查询
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT distinct symbol  FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, TARGET_SYMBOLS)
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        logging.info(f"✅ 成功加载 {len(symbols)} 只股票作为白名单。")
        return symbols
    except Exception as e:
        logging.error(f"读取数据库失败: {e}")
        return set()

def main():
    logging.info("--- 开始执行数据分割流程 (带股票过滤) ---")
    
    # 1. 检查源目录
    if not SOURCE_DIR.is_dir():
        logging.error(f"错误：源目录 '{SOURCE_DIR}' 不存在！")
        return
        
    # 2. [新增] 获取股票白名单
    valid_symbols = get_valid_symbols()
    if not valid_symbols:
        logging.error("未获取到有效的股票列表，程序终止。")
        return

    # 3. [优化] 扫描文件逻辑
    # 不再盲目扫描所有目录，而是遍历白名单，只去查找存在的文件夹
    logging.info("正在扫描源目录 (仅限白名单股票)...")
    tasks = []
    
    # 进度条显示扫描过程
    for symbol in tqdm(valid_symbols, desc="扫描股票目录"):
        symbol_path = SOURCE_DIR / symbol
        
        # 检查该股票的目录是否存在
        if symbol_path.exists() and symbol_path.is_dir():
            # 查找该股票目录下的所有 .parquet 文件 (递归)
            # 假设结构: symbol/session/time/res/file.parquet
            files = list(symbol_path.glob('**/*.parquet'))
            tasks.extend(files)
    
    if not tasks:
        logging.error(f"未找到属于白名单股票的任何 .parquet 文件。")
        return
    
    logging.info(f"共找到 {len(tasks)} 个符合条件的文件待处理。")
    
    # 4. 准备并行处理
    train_range_ts = (pd.to_datetime(TRAIN_DATE_RANGE[0]), pd.to_datetime(TRAIN_DATE_RANGE[1]))
    val_range_ts = (pd.to_datetime(VAL_DATE_RANGE[0]), pd.to_datetime(VAL_DATE_RANGE[1]))
    test_range_ts = (pd.to_datetime(TEST_DATE_RANGE[0]), pd.to_datetime(TEST_DATE_RANGE[1]))

    worker_func = partial(process_and_copy_file, 
                          source_dir=SOURCE_DIR, 
                          train_dir=TRAIN_DIR, 
                          val_dir=VAL_DIR,
                          test_dir=TEST_DIR,
                          train_range_ts=train_range_ts,
                          val_range_ts=val_range_ts,
                          test_range_ts=test_range_ts)

    # 5. 执行并行复制
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks), desc="分割和复制文件"))

    # 6. 统计结果
    success_count = sum(1 for r in results if r.startswith("[成功]"))
    skipped_count = sum(1 for r in results if r.startswith("[跳过]"))
    error_count = sum(1 for r in results if r.startswith("[错误]"))
    
    print("\n" + "="*80)
    print("--- 流程完成 ---")
    print(f"  - 目标股票数 (Whitelist): {len(valid_symbols)}")
    print(f"  - 扫描到的文件总数: {len(tasks)}")
    print(f"  - 成功复制: {success_count}")
    print(f"  - 日期跳过: {skipped_count}")
    print(f"  - 错误: {error_count}")
    print("="*80)

if __name__ == "__main__":
    main()