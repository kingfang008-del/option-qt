import os
import datetime
import pandas as pd
from polygon import RESTClient
import concurrent.futures
from tqdm import tqdm
import logging
from pytz import timezone

# ================= 全局配置 =================
API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"  # 你的 Polygon API Key
TARGET_MAP_FILE = "/home/kingfang007/data/locked_targets_map.parquet" # Step 1 生成的名单
OUTPUT_DIR = "/mnt/s990/data/option_quote_sniper" # 建议新建一个纯净的输出目录

# Polygon 并发极高，可以大胆开！(依你的套餐和机器性能而定，通常 20-50 没问题)
MAX_WORKERS = 50 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
eastern = timezone('America/New_York')
# ============================================

def process_single_day_polygon(args):
    """
    【独立 Worker】负责向 Polygon 索要 1 天内这 6 個目標合約的盤口，並壓縮成 1 分鐘線
    """
    symbol, date_str, group_df = args
    client = RESTClient(API_KEY)
    
    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{date_str}.parquet")

    # 斷點續傳
    if os.path.exists(out_path):
        return f"⏩ {symbol} {date_str} 已存在，跳過。"

    all_contracts_data = []

    # 遍歷當天鎖定的 6 個合約
    for _, row in group_df.iterrows():
        occ_ticker = row['contract_symbol']
        bucket_id = row['bucket_id']
        
        # 確保 Polygon 格式 (需要 O: 前綴)
        poly_ticker = occ_ticker if occ_ticker.startswith("O:") else f"O:{occ_ticker}"
        
        quotes_list = []
        try:
            # 🎯 終極狙擊：嚴格限制時間視窗為「這一天」
            # 這是提速 100 倍的核心，Polygon 不需要翻幾十天的歷史分頁了
            quotes_generator = client.list_quotes(
                ticker=poly_ticker, 
                timestamp_gte=date_str, 
                timestamp_lte=date_str,
                limit=50000 
            )
            
            for q in quotes_generator:
                ts = getattr(q, 'sip_timestamp', getattr(q, 'participant_timestamp', None))
                if not ts: continue
                quotes_list.append({
                    'timestamp': ts,
                    'bid': getattr(q, 'bid_price', 0.0),
                    'ask': getattr(q, 'ask_price', 0.0),
                    'bid_size': getattr(q, 'bid_size', 0),
                    'ask_size': getattr(q, 'ask_size', 0)
                })
        except Exception as e:
            # 如果某個合約當天沒數據/報錯，跳過即可
            continue
            
        if not quotes_list:
            continue

        # -----------------------------------------------------
        # 在內存中將 Tick 降採樣為 1 分鐘級別 (完全復刻之前的邏輯)
        # -----------------------------------------------------
        df = pd.DataFrame(quotes_list)
        df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
        df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
        df['bid_size'] = pd.to_numeric(df['bid_size'], errors='coerce')
        df['ask_size'] = pd.to_numeric(df['ask_size'], errors='coerce')
        df.dropna(subset=['timestamp', 'bid', 'ask'], inplace=True)
        
        # 盤口清洗
        df = df[
            (df['bid'] > 0.0) & 
            (df['ask'] >= df['bid']) &
            (df['bid_size'] > 0) & 
            (df['ask_size'] > 0)
        ].copy()
        
        if df.empty: continue

        # 轉換時區
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert(eastern)
        
        # 分鐘級壓縮
        df['minute_ts'] = df['timestamp'].dt.floor('min')
        
        # 過濾 RTH (交易時段)
        time_series = df['minute_ts'].dt.time
        df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
        if df.empty: continue
            
        # 每一分鐘保留最後一筆快照
        df = df.sort_values('timestamp').drop_duplicates(subset=['minute_ts'], keep='last').copy()
        df['timestamp'] = df['minute_ts']
        
        # 特徵工程
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
        df['open'] = df['mid_price']
        df['high'] = df['mid_price']
        df['low'] = df['mid_price']
        df['close'] = df['mid_price']
        df['volume'] = df['bid_size'] + df['ask_size']
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']
        
        # 綁定身份標籤，供下游 options_locked_feature 直接使用
        df['ticker'] = occ_ticker.replace('O:', '')
        df['bucket_id'] = bucket_id
        
        cols_to_keep = [
            'timestamp', 'ticker', 'bucket_id', 'open', 'high', 'low', 'close', 'volume',
            'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance'
        ]
        all_contracts_data.append(df[cols_to_keep])

    # -----------------------------------------------------
    # 合併當天 6 個合約並落盤
    # -----------------------------------------------------
    if not all_contracts_data:
        return f"⚠️ {symbol} {date_str}: 該日所選合約均無有效盤口。"
        
    final_df = pd.concat(all_contracts_data, ignore_index=True)
    # 按時間和 bucket 排序，讓數據更整潔
    final_df = final_df.sort_values(['timestamp', 'bucket_id'])
    
    # 使用最高壓縮比 zstd 存儲
    final_df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd')
    
    return f"🎯 {symbol} {date_str}: 成功！下載並壓縮了 {len(final_df)} 行極品數據。"


def main():
    if not os.path.exists(TARGET_MAP_FILE):
        logger.error(f"❌ 找不到目標清單 {TARGET_MAP_FILE}，請先運行 step1 雷達腳本！")
        return
        
    logger.info("📂 正在加載狙擊清單...")
    target_map = pd.read_parquet(TARGET_MAP_FILE)
    
    tasks = []
    # 將任務按 (股票, 日期) 打包
    for (sym, date_str), group in target_map.groupby(['symbol', 'date_str']):
        tasks.append((sym, date_str, group))
        
    logger.info(f"🚀 啟動 Polygon 狂暴狙擊模式！共計 {len(tasks)} 個單日精確任務。")
    logger.info(f"⚡ 當前並發線程數: {MAX_WORKERS}")
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任務
        futures = {executor.submit(process_single_day_polygon, task): task for task in tasks}
        
        # 進度條
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="下載進度"):
            res = future.result()
            if "成功" in res:
                success_count += 1
                
    logger.info("-" * 50)
    logger.info(f"🎉 狙擊完成！成功獲取了 {success_count} / {len(tasks)} 天的高精度盤口數據。")
    logger.info(f"📁 數據已無縫對齊 options_locked_feature 的輸入格式，準備進入特徵生成環節！")

if __name__ == "__main__":
    main()