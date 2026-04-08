import boto3
import os
import datetime
from datetime import time as dt_time
import pandas as pd
import glob
import logging
from botocore.config import Config
import concurrent.futures
from tqdm import tqdm
import shutil
import time

# ================= 配置区域 =================

# [S3 凭证] 请从 Polygon Dashboard -> Flat Files 获取
ACCESS_KEY = "8634ce52-719b-4bdb-83d5-ada3aa31601d"
SECRET_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
ENDPOINT_URL = "https://files.polygon.io"
BUCKET_NAME = "flatfiles" 

# [下载设置] - 改为 Quotes 目录
S3_PREFIX = "us_options_opra/quotes_v1"
START_DATE = datetime.date(2026, 1, 1)
END_DATE = datetime.date(2026, 2, 28)

# [时间与期限约束] (继承自 thetadata 逻辑)
DTE_MIN = 3
DTE_MAX = 65

# [路径]
RAW_DIR = "./data/temp_s3_raw"
PROCESSED_DIR = "/mnt/s990/data/option_quote"

try:
    from config import TARGET_SYMBOLS
except ImportError:
    # 默认目标
    TARGET_SYMBOLS =   TARGET_SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 核心逻辑 =================

class S3QuotePipeline:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(max_pool_connections=50)
        )
        self.contract_cache = {sym: set() for sym in TARGET_SYMBOLS}
        
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        self.progress_file = os.path.join(PROCESSED_DIR, "processed_dates.json")
        self.processed_dates = self._load_progress()
        self._load_existing_contracts()

    def _load_progress(self):
        import json
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f))
            except: pass
        return set()

    def _mark_date_completed(self, date_str):
        import json
        self.processed_dates.add(date_str)
        with open(self.progress_file, 'w') as f:
            json.dump(sorted(list(self.processed_dates)), f)

    def _load_existing_contracts(self):
        for sym in TARGET_SYMBOLS:
            path = os.path.join(PROCESSED_DIR, sym, "contracts_list.txt")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            self.contract_cache[sym].add(line.strip())
                except: pass

    def save_contracts_list(self):
        logger.info("💾 正在更新所有股票的 contracts_list.txt ...")
        for sym in TARGET_SYMBOLS:
            if not self.contract_cache[sym]: continue
            sym_dir = os.path.join(PROCESSED_DIR, sym)
            os.makedirs(sym_dir, exist_ok=True)
            path = os.path.join(sym_dir, "contracts_list.txt")
            sorted_contracts = sorted(list(self.contract_cache[sym]))
            with open(path, 'w') as f:
                f.write('\n'.join(sorted_contracts) + '\n')

    def download_day(self, date_obj):
        date_month_str = date_obj.strftime('%Y/%m')
        date_file_str = date_obj.strftime('%Y-%m-%d')
        prefix = f"{S3_PREFIX}/{date_month_str}/{date_file_str}"
        
        try:
            resp = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
            if 'Contents' not in resp:
                logger.warning(f"📅 {date_file_str}: S3 无数据 (可能休市)")
                return []
            
            downloaded_files = []
            for obj in resp['Contents']:
                key = obj['Key']
                fname = key.replace('/', '_')
                local_path = os.path.join(RAW_DIR, fname)
                
                if not os.path.exists(local_path) or os.path.getsize(local_path) != obj['Size']:
                    logger.info(f"⬇️ 开始下载: {key} ...")
                    self.s3.download_file(BUCKET_NAME, key, local_path)
                
                downloaded_files.append(local_path)
            
            return downloaded_files
        except Exception as e:
            logger.error(f"❌ 下载失败 {date_file_str}: {e}")
            return []

    def process_and_filter(self, file_paths, date_obj):
        if not file_paths: return

        date_str = date_obj.strftime('%Y-%m-%d')
        logger.info(f"⚙️ 正在清洗并聚合 {date_str} 的 Quote 数据为 1分钟快照...")

        daily_data_chunks = {sym: [] for sym in TARGET_SYMBOLS}
        
        # Quote flat files are massive. We chunk by 1,000,000 rows.
        for fp in file_paths:
            try:
                for chunk in pd.read_csv(fp, chunksize=1000000, usecols=['ticker', 'sip_timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size']):
                    # 1. 提取资产符号做初步过滤
                    chunk['clean_ticker'] = chunk['ticker'].str.replace('O:', '', regex=False)
                    chunk['underlying'] = chunk['clean_ticker'].str[:-15]
                    
                    df = chunk[chunk['underlying'].isin(TARGET_SYMBOLS)].copy()
                    if df.empty: continue
                    
                    # 2. 时间解析 (ns -> UTC -> US/Eastern)
                    df['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns', utc=True).dt.tz_convert('America/New_York')
                    df['minute_ts'] = df['timestamp'].dt.floor('min')
                    
                    # 3. 交易时段 (09:30 - 16:00)
                    time_series = df['minute_ts'].dt.time
                    df = df[(time_series >= dt_time(9, 30)) & (time_series < dt_time(16, 0))]
                    if df.empty: continue
                    
                    # 4. 按分钟采样: 取该分钟内部最后一笔 Quote
                    # 因为我们在 chunk 级别处理，可能会存在跨 chunk 的情况，最后统一还要在符号维度的外层再做一次 drop_duplicates
                    df = df.sort_values(['clean_ticker', 'timestamp']).drop_duplicates(subset=['clean_ticker', 'minute_ts'], keep='last')
                    df['timestamp'] = df['minute_ts']
                    
                    # 5. 计算衍生数据
                    df['expiration'] = pd.to_datetime(df['clean_ticker'].str[-15:-9], format='%y%m%d', utc=True).dt.tz_convert('America/New_York')
                    df['dte'] = (df['expiration'].dt.normalize() - df['timestamp'].dt.normalize()).dt.days
                    
                    # DTE 裁剪
                    df = df[(df['dte'] >= DTE_MIN) & (df['dte'] <= DTE_MAX)]
                    if df.empty: continue
                    
                    # 重命名以对齐原引擎的列
                    df.rename(columns={
                        'bid_price': 'bid',
                        'ask_price': 'ask'
                    }, inplace=True)
                    
                    # 异常盘口清洗
                    df = df[
                        (df['bid'] > 0.0) & 
                        (df['ask'] >= df['bid']) &
                        (df['bid_size'] > 0) & 
                        (df['ask_size'] > 0)
                    ]
                    if df.empty: continue
                    
                    # 微观特征提取
                    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
                    df['close'] = df['mid_price']
                    df['open'] = df['mid_price']
                    df['high'] = df['mid_price']
                    df['low'] = df['mid_price']
                    df['volume'] = df['bid_size'] + df['ask_size']
                    df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
                    df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']
                    
                    df.rename(columns={'clean_ticker': 'ticker'}, inplace=True)
                    
                    cols_to_keep = [
                        'underlying', 'timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                        'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance'
                    ]
                    
                    # 分发到 daily_data_chunks
                    for sym, group in df[cols_to_keep].groupby('underlying'):
                        daily_data_chunks[sym].append(group.drop(columns=['underlying']))
                        
            except Exception as e:
                logger.error(f"❌ 处理文件失败 {fp}: {e}")
                import traceback
                traceback.print_exc()

        # 每日跑完所有 Part 之后，把各个 chunk 的 1-min aggregated merge 在一起并存盘
        for sym in TARGET_SYMBOLS:
            if not daily_data_chunks[sym]: continue
            
            sym_df = pd.concat(daily_data_chunks[sym])
            # 解决同一分钟垮文件引发的多条纪录
            sym_df = sym_df.sort_values(['ticker', 'timestamp']).drop_duplicates(subset=['ticker', 'timestamp'], keep='last')
            
            # 更新 memory list
            unique_contracts = sym_df['ticker'].unique()
            self.contract_cache[sym].update(unique_contracts)
            
            # 存盘
            sym_dir = os.path.join(PROCESSED_DIR, sym)
            os.makedirs(sym_dir, exist_ok=True)
            save_path = os.path.join(sym_dir, f"{sym}_{date_str}.parquet")
            
            if os.path.exists(save_path):
                df_exist = pd.read_parquet(save_path)
                sym_df = pd.concat([df_exist, sym_df]).sort_values(['ticker', 'timestamp']).drop_duplicates(subset=['ticker', 'timestamp'], keep='last')
            
            sym_df.to_parquet(save_path, index=False, engine='pyarrow', compression='zstd')

    def cleanup_raw(self, file_paths):
        for fp in file_paths:
            try: os.remove(fp)
            except: pass

    def run(self):
        current = START_DATE
        total_days = (END_DATE - START_DATE).days + 1
        
        with tqdm(total=total_days, desc="Quote S3 Pipeline") as pbar:
            while current <= END_DATE:
                if current.weekday() >= 5: # 跳过周末
                    current += datetime.timedelta(days=1)
                    pbar.update(1)
                    continue

                current_date_str = current.strftime('%Y-%m-%d')
                if current_date_str in self.processed_dates:
                    current += datetime.timedelta(days=1)
                    pbar.update(1)
                    continue

                raw_files = self.download_day(current)
                if raw_files:
                    self.process_and_filter(raw_files, current)
                    self.cleanup_raw(raw_files)
                    
                self._mark_date_completed(current_date_str)
                
                if current.day % 5 == 0:
                    self.save_contracts_list()

                current += datetime.timedelta(days=1)
                pbar.update(1)
        
        self.save_contracts_list()
        logger.info("✅ 所有任务完成！Quote 级别数据已压缩洗净。")

if __name__ == "__main__":
    pipeline = S3QuotePipeline()
    pipeline.run()
