import os
import glob
import logging
import pandas as pd
import concurrent.futures
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 配置区域 =================
SRC_DIR = "/home/kingfang007/data/new_option_data_s3"
DEST_DIR = "/home/kingfang007/data/new_option_data_s3_clean"
MAX_WORKERS = 8  # 并发线程数，可根据机器配置调整

def process_file(src_path):
    """读取单个 Parquet 文件，过滤 volume <= 0 的行，并保存到新目录"""
    try:
        # 提取相对路径结构，例如: AAPL/AAPL_2022-03-01.parquet
        rel_path = os.path.relpath(src_path, SRC_DIR)
        dest_path = os.path.join(DEST_DIR, rel_path)
        
        # 确保目标文件的父目录存在
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 检查是否已经处理过
        if os.path.exists(dest_path):
            return 0  # 已存在，跳过

        # 读取原始数据
        # 即使文件损坏，也用 try-except 捕获
        df = pd.read_parquet(src_path)
        
        # 如果文件为空或者没有 'v' 列，直接忽略
        if df.empty or 'v' not in df.columns:
            logger.warning(f"⚠️ 文件为空或缺失 'v' 列: {src_path}")
            return 0
            
        initial_len = len(df)
        
        # 执行过滤逻辑：仅保留有真实成交量的数据
        df_clean = df[df['v'] > 0].copy()
        clean_len = len(df_clean)
        
        # 如果过滤后还有数据，保存到新目录
        if clean_len > 0:
            df_clean.to_parquet(dest_path, index=False)
            
        return initial_len - clean_len

    except Exception as e:
        logger.error(f"❌ 处理文件失败 {src_path}: {e}")
        return 0

def main():
    if not os.path.exists(SRC_DIR):
        logger.error(f"❌ 源目录不存在: {SRC_DIR}")
        return

    logger.info(f"🔍 扫描源目录: {SRC_DIR}")
    parquet_files = glob.glob(os.path.join(SRC_DIR, "**", "*.parquet"), recursive=True)
    
    if not parquet_files:
        logger.warning("⚠️ 未找到任何 Parquet 文件。")
        return
        
    logger.info(f"📂 共找到 {len(parquet_files)} 个 .parquet 文件，开始清洗...")
    
    total_removed = 0
    
    # 使用线程池并发加速处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 显示进度条
        futures = {executor.submit(process_file, fp): fp for fp in parquet_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="清洗进度"):
            try:
                removed_count = future.result()
                total_removed += removed_count
            except Exception as e:
                logger.error(f"Task generated an exception: {e}")

    logger.info("=========================================")
    logger.info(f"✅ 清洗完成！所有有效数据已保存至: {DEST_DIR}")
    logger.info(f"🗑️ 共计剔除了 {total_removed:,} 行零成交量 (幽灵刻度) 数据。")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
