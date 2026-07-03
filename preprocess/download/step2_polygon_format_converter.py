import os
import glob
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# ================= 配置参数 =================
# 你存放 step2_polygon_min_sniper_v1 下载的原始数据目录
INPUT_DIR = "/mnt/s990/data/massive_options_1m"
# 要完整转换为 ThetaData 格式并输出的新目录
OUTPUT_DIR = "/mnt/s990/data/massive_options_1m_formatted"

MAX_WORKERS = 16
# ============================================

def process_file(file_path):
    try:
        # 获取股票代码，用于创建子目录
        # 文件路径一般为: INPUT_DIR/<symbol>/<symbol>_<date>.parquet
        parts = file_path.split(os.sep)
        symbol = parts[-2]
        filename = parts[-1] 
        
        out_dir = os.path.join(OUTPUT_DIR, symbol)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        if os.path.exists(out_path):
            return f"⏩ {filename} 已存在跳过"
            
        df = pd.read_parquet(file_path)
        if df.empty:
            return f"⚠️ {filename}: 空文件"

        # -------------------------------------------------------------
        # 核心转换逻辑：将 Polygon 的数据形态对齐为 ThetaData 格式
        # -------------------------------------------------------------
        
        # 1. Ticker 还原: ThetaData 格式中包含 O: 前缀
        if 'ticker' in df.columns:
            # 如果原数据中没有 O: 就加上
            df['ticker'] = df['ticker'].apply(lambda x: x if str(x).startswith('O:') else f"O:{x}")
        
        # 如果原来没有计算 mid_price，重新计算
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid'] + df['ask']) / 2.0
            
        # 2. 伪造 K 线 (OHLC): 使用 mid_price 填充
        df['open'] = df['mid_price']
        df['high'] = df['mid_price']
        df['low'] = df['mid_price']
        df['close'] = df['mid_price']
        
        # 3. 计算 volume 和相关的微观结构因子
        df['volume'] = df['bid_size'] + df['ask_size']
        
        # 4. 计算买卖价差与订单不平衡度
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        
        # volume_imbalance 逻辑与 ThetaData 中一致
        df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume'].replace(0, 1) # 防除 0
        
        # 5. 时间戳处理
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
            # 去除时区信息以匹配截图中的 naive datetime 格式显示 (可选) 
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # 6. 选择特定的列输出，保持严密对齐
        cols = [
            'timestamp', 'ticker', 'bucket_id', 
            'open', 'high', 'low', 'close', 'volume', 
            'bid', 'ask', 'bid_size', 'ask_size', 
            'spread_pct', 'volume_imbalance'
        ]
        
        # 容错：如果某些列确实没有，用 0 填充
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in ['ticker', 'timestamp'] else None
                
        final_df = df[cols].copy()
        
        # 保存压缩格式
        final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
        
        return f"🎯 转换成功 {filename}: {len(final_df)} 行"
        
    except Exception as e:
        return f"❌ {file_path} 处理报错: {e}"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录 {INPUT_DIR}")
        return

    # 递归搜索所有的 parquet 文件
    print(f"📂 正在搜索 {INPUT_DIR} 下的 Parquet 文件...")
    all_files = glob.glob(os.path.join(INPUT_DIR, "*", "*.parquet"))
    
    if not all_files:
        print("⚠️ 输入目录中没有发现有效的 Parquet 文件")
        return
        
    print(f"🚀 开启格式转换模式！共计 {len(all_files)} 个文件。并发限制：{MAX_WORKERS}...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in all_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_files), desc="Converting Format"):
            res = future.result()
            # print(res)

    print(f"🎉 格式转换完成！你可以将下游流程的数据源指向: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
