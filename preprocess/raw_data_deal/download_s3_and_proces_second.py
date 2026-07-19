# 秒级成交（默认）
#python preprocess/raw_data_deal/download_s3_and_process.py
# 仍下分钟
#S3_DATA_KIND=minute python preprocess/raw_data_deal/download_s3_and_process.py
# 盘口（慎用，体积大）
#S3_DATA_KIND=quotes python preprocess/raw_data_deal/download_s3_and_process.py


import boto3
import os
import datetime
import pandas as pd
import glob
import logging
from botocore.config import Config
import concurrent.futures
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import shutil
import time
from polygon import RESTClient


# ================= 配置区域 =================

# [S3 凭证] 请从 Polygon Dashboard -> Flat Files 获取
ACCESS_KEY = "8634ce52-719b-4bdb-83d5-ada3aa31601d"
SECRET_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
ENDPOINT_URL = "https://files.polygon.io"
BUCKET_NAME = "flatfiles" 

# [下载设置]
# Polygon OPRA flat files 无独立 second_aggs；秒级用 trades（tick）再可选聚合成 1s。
#   minute  → us_options_opra/minute_aggs_v1   （分钟 OHLCV，已有）
#   trades  → us_options_opra/trades_v1        （成交 tick / 秒级，体积适中）
#   quotes  → us_options_opra/quotes_v1        （盘口 tick，单日可达 100GB+，需分块）
DATA_KIND = os.environ.get("S3_DATA_KIND", "trades").strip().lower()  # minute | trades | quotes
S3_PREFIX_BY_KIND = {
    "minute": "us_options_opra/minute_aggs_v1",
    "trades": "us_options_opra/trades_v1",
    "quotes": "us_options_opra/quotes_v1",
}
if DATA_KIND not in S3_PREFIX_BY_KIND:
    raise ValueError(f"DATA_KIND must be one of {sorted(S3_PREFIX_BY_KIND)}, got {DATA_KIND!r}")
S3_PREFIX = S3_PREFIX_BY_KIND[DATA_KIND]
# trades → 聚合成 1s OHLCV（v/o/c/h/l/t/n）；False 则保留 tick
AGGREGATE_TRADES_TO_1S = os.environ.get("S3_AGG_1S", "1").strip() not in {"0", "false", "False"}
# quotes → 每合约每秒最后一档盘口
AGGREGATE_QUOTES_TO_1S = os.environ.get("S3_QUOTE_AGG_1S", "1").strip() not in {"0", "false", "False"}

START_DATE = datetime.date(2026, 7, 17)
END_DATE = datetime.date(2026, 7, 18)  # 先下 Jul1 烟雾；需要多天再改 END_DATE

# [路径]
# RAW_DIR: 临时存放 S3 原始大文件的目录 (处理完会删除)
RAW_DIR = f"./data/temp_s3_raw_{DATA_KIND}"
# PROCESSED_DIR: 最终存放清洗后数据（按 kind 分目录，避免覆盖分钟 parquet）
_PROCESSED_ROOT = "/mnt/s990/new_option_data_s3"
PROCESSED_DIR = (
    _PROCESSED_ROOT if DATA_KIND == "minute" else f"{_PROCESSED_ROOT}_{DATA_KIND}"
)
#from config import TARGET_SYMBOLS
# [目标股票列表] (Tier 1 ~ Tier 5 + Macro)
TARGET_SYMBOLS =   ['QQQ','NVDA', 'TSLA', 'AMD', 'INTC', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AAPL' ,'MU','AVGO']
#TARGET_SYMBOLS =   ['GOOGL']
# [Polygon API Token] 用于预检合约
API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def retry_api_call(max_retries=5, initial_wait=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    if "connection" in error_msg or "429" in error_msg or "ssl" in error_msg:
                        logger.warning(f"⚠️ 网络波动 (重试 {attempt+1}): {e}. 等待 {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time *= 2 
                    else:
                        raise e 
            logger.error(f"❌ 重试耗尽: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

# ================= 核心逻辑 =================

class S3Pipeline:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(max_pool_connections=50)
        )
        self.contract_cache = {sym: set() for sym in TARGET_SYMBOLS}
        
        # 初始化目录
        os.makedirs(RAW_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # 初始化进度追踪文件
        self.progress_file = os.path.join(PROCESSED_DIR, "processed_dates.json")
        self.processed_dates = self._load_progress()
        
        # 预加载本地已有的合约列表，确保后续保存时是完全的并集
        self._load_existing_contracts()
        
        # 多线程预检完整度
       # self.pre_check_completeness()

    def _load_progress(self):
        """读取已完成下载并清洗完毕的日期列表"""
        import json
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f))
            except: pass
        return set()

    def _mark_date_completed(self, date_str):
        """将日期标记为已完成并立即存盘"""
        import json
        self.processed_dates.add(date_str)
        with open(self.progress_file, 'w') as f:
            json.dump(sorted(list(self.processed_dates)), f)

    def _load_existing_contracts(self):
        """预读取本地已有的合约列表，避免覆盖"""
        for sym in TARGET_SYMBOLS:
            path = os.path.join(PROCESSED_DIR, sym, "contracts_list.txt")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            self.contract_cache[sym].add(line.strip())
                except: pass

    def save_contracts_list(self):
        """将内存中的合约列表写入磁盘"""
        logger.info("💾 正在更新所有股票的 contracts_list.txt ...")
        for sym in TARGET_SYMBOLS:
            if not self.contract_cache[sym]: continue
            
            sym_dir = os.path.join(PROCESSED_DIR, sym)
            os.makedirs(sym_dir, exist_ok=True)
            path = os.path.join(sym_dir, "contracts_list.txt")
            
            sorted_contracts = sorted(list(self.contract_cache[sym]))
            with open(path, 'w') as f:
                f.write('\n'.join(sorted_contracts) + '\n')

    def pre_check_completeness(self):
        logger.info("🔍 [预检] 正在使用多线程扫描本地已下载文件和合约完整度...")
        
        all_found_dates = set()
        
        def check_symbol(symbol):
            symbol_dir = os.path.join(PROCESSED_DIR, symbol)
            
            # 1. 扫描本地 parquet 的日期
            local_dates_for_sym = set()
            if os.path.exists(symbol_dir):
                for f in os.listdir(symbol_dir):
                    if f.endswith('.parquet') and f.startswith(f"{symbol}_"):
                        # extact date: "AAPL_2022-03-01.parquet"
                        date_str = f.replace(f"{symbol}_", "").replace(".parquet", "")
                        local_dates_for_sym.add(date_str)
                        
            # 2. 获取本地缓存合约数
            local_count = len(self.contract_cache[symbol])
            
            # 3. 拉取 API (为了统计完整度)
            client = RESTClient(API_KEY)
            api_tickers_map = {}
            
            @retry_api_call(max_retries=3, initial_wait=2)
            def fetch_list():
                return list(client.list_options_contracts(
                    underlying_ticker=symbol, 
                    expiration_date_gte=START_DATE.strftime('%Y-%m-%d'),
                    expiration_date_lte=END_DATE.strftime('%Y-%m-%d'),
                    limit=1000, expired="true", sort="expiration_date", order="asc"
                ))

            try:
                contracts = fetch_list()
                for c in contracts: api_tickers_map[c.ticker] = c
            except Exception:
                pass
                
            api_count = len(api_tickers_map)
            all_tickers = set(self.contract_cache[symbol]).union(api_tickers_map.keys())
            total_count = len(all_tickers)
            
            # 4. 显示统计面板
            ratio_contract = (local_count / api_count * 100) if api_count > 0 else 0
            
            msg = (
                f"\n{'='*20} 📊 [{symbol}] 数据预检统计 {'='*20}\n"
                f"1. 合约完整度(S3挖掘 vs API应有):\n"
                f"   - 历史文件中挖掘出: {local_count}\n"
                f"   - Polygon API 最新返回: {api_count}\n"
                f"   - 约当覆盖比例: {ratio_contract:.1f}% (当前合并总量: {total_count})\n"
                f"2. 数据下载进度:\n"
                f"   - 本地有效交易日文件: {len(local_dates_for_sym)} 个\n"
                f"{'='*58}"
            )
            print(msg)
            
            return local_dates_for_sym
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_symbol, sym) for sym in TARGET_SYMBOLS]
            for future in concurrent.futures.as_completed(futures):
                try: 
                    dates = future.result()
                    all_found_dates.update(dates)
                except Exception as e: 
                    logger.error(f"Symbol Check Failed: {e}")
                    
        # 更新到 processed_dates (如果任意股票有这天的数据，说明S3的这一天已经被处理过了，因为S3是按天全量打包的)
        new_dates_count = 0
        for d in all_found_dates:
            if d not in self.processed_dates:
                self.processed_dates.add(d)
                new_dates_count += 1
                
        if new_dates_count > 0:
            logger.info(f"✨ 发现 {new_dates_count} 个已下载的日期并未记录在 json 中，已自动合并入跳过名单。")
            self._mark_date_completed("DUMMY_JUST_SAVE") # force a save to disk
            self.processed_dates.discard("DUMMY_JUST_SAVE")

    def download_day(self, date_obj):
        """下载某一天的 S3 文件到临时目录"""
        # 目录结构: {S3_PREFIX}/YYYY/MM/YYYY-MM-DD.csv.gz
        date_month_str = date_obj.strftime('%Y/%m')
        date_file_str = date_obj.strftime('%Y-%m-%d')
        prefix = f"{S3_PREFIX}/{date_month_str}/{date_file_str}"
        
        try:
            # 列出文件
            resp = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
            if 'Contents' not in resp:
                logger.warning(f"📅 {date_file_str}: S3 无数据 (可能休市) prefix={prefix}")
                return []
            
            downloaded_files = []
            for obj in resp['Contents']:
                key = obj['Key']
                size_gb = obj['Size'] / 1e9
                # 保持文件名唯一，防止覆盖
                fname = key.replace('/', '_')
                local_path = os.path.join(RAW_DIR, fname)
                
                # 下载
                if not os.path.exists(local_path) or os.path.getsize(local_path) != obj['Size']:
                    logger.info(f"⬇️ 下载 {key} ({size_gb:.2f} GB) → {local_path}")
                    self.s3.download_file(BUCKET_NAME, key, local_path)
                
                downloaded_files.append(local_path)
            
            return downloaded_files
        except Exception as e:
            logger.error(f"❌ 下载失败 {date_file_str}: {e}")
            return []

    @staticmethod
    def _add_underlying(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["clean_ticker"] = df["ticker"].astype(str).str.replace("O:", "", regex=False)
        df["underlying"] = df["clean_ticker"].str.extract(r"^([A-Z]+)\d")
        return df

    def _save_symbol_day(self, sym: str, date_str: str, group: pd.DataFrame):
        unique_contracts = group["ticker"].unique()
        self.contract_cache[sym].update(unique_contracts)
        sym_dir = os.path.join(PROCESSED_DIR, sym)
        os.makedirs(sym_dir, exist_ok=True)
        save_path = os.path.join(sym_dir, f"{sym}_{date_str}.parquet")
        drop_cols = [c for c in ("clean_ticker", "underlying") if c in group.columns]
        group_to_save = group.drop(columns=drop_cols)
        if os.path.exists(save_path):
            df_exist = pd.read_parquet(save_path)
            df_merged = pd.concat([df_exist, group_to_save], ignore_index=True)
            # 1s 聚合跨 chunk 可能重复同一 (ticker,timestamp)，再压一遍
            if (
                DATA_KIND == "trades"
                and AGGREGATE_TRADES_TO_1S
                and {"ticker", "timestamp", "o", "h", "l", "c", "v"}.issubset(df_merged.columns)
            ):
                agg_map = {
                    "o": ("o", "first"),
                    "h": ("h", "max"),
                    "l": ("l", "min"),
                    "c": ("c", "last"),
                    "v": ("v", "sum"),
                }
                if "n" in df_merged.columns:
                    agg_map["n"] = ("n", "sum")
                if "t" in df_merged.columns:
                    agg_map["t"] = ("t", "first")
                df_merged = (
                    df_merged.sort_values("timestamp")
                    .groupby(["ticker", "timestamp"], as_index=False)
                    .agg(**agg_map)
                )
            elif DATA_KIND == "quotes" and AGGREGATE_QUOTES_TO_1S and {"ticker", "timestamp"}.issubset(
                df_merged.columns
            ):
                df_merged = (
                    df_merged.sort_values("timestamp")
                    .groupby(["ticker", "timestamp"], as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
                )
            else:
                df_merged = df_merged.drop_duplicates()
            df_merged.to_parquet(save_path, index=False)
        else:
            group_to_save.to_parquet(save_path, index=False)

    def _process_minute_chunk(self, df: pd.DataFrame, date_str: str):
        df = df.rename(
            columns={
                "volume": "v",
                "open": "o",
                "close": "c",
                "high": "h",
                "low": "l",
                "window_start": "t",
                "transactions": "n",
            },
            errors="ignore",
        )
        if "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ns", utc=True).dt.tz_convert("America/New_York")
        df = self._add_underlying(df)
        df_filtered = df[df["underlying"].isin(TARGET_SYMBOLS)]
        if df_filtered.empty:
            return
        for sym, group in df_filtered.groupby("underlying"):
            self._save_symbol_day(sym, date_str, group)

    def _process_trades_chunk(self, df: pd.DataFrame, date_str: str):
        # trades_v1: ticker, price, size, sip_timestamp, exchange, conditions, ...
        need = {"ticker", "price", "size", "sip_timestamp"}
        if not need.issubset(df.columns):
            raise ValueError(f"trades chunk missing cols: {need - set(df.columns)}")
        df = self._add_underlying(df)
        df = df[df["underlying"].isin(TARGET_SYMBOLS)].copy()
        if df.empty:
            return
        df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True).dt.tz_convert(
            "America/New_York"
        )
        if AGGREGATE_TRADES_TO_1S:
            df["sec"] = df["timestamp"].dt.floor("s")
            rows = []
            for (sym, ticker), g in df.groupby(["underlying", "ticker"], sort=False):
                g = g.sort_values("timestamp")
                agg = (
                    g.groupby("sec", sort=True)
                    .agg(
                        o=("price", "first"),
                        h=("price", "max"),
                        l=("price", "min"),
                        c=("price", "last"),
                        v=("size", "sum"),
                        n=("price", "count"),
                    )
                    .reset_index()
                )
                agg["ticker"] = ticker
                agg["underlying"] = sym
                agg["timestamp"] = agg["sec"]
                agg["t"] = agg["sec"].astype("int64")  # ns since epoch (tz-aware → int64)
                rows.append(agg.drop(columns=["sec"]))
            if not rows:
                return
            out = pd.concat(rows, ignore_index=True)
            for sym, group in out.groupby("underlying"):
                self._save_symbol_day(sym, date_str, group)
        else:
            keep = [
                c
                for c in (
                    "ticker",
                    "underlying",
                    "clean_ticker",
                    "timestamp",
                    "sip_timestamp",
                    "price",
                    "size",
                    "exchange",
                    "conditions",
                    "correction",
                    "participant_timestamp",
                )
                if c in df.columns
            ]
            for sym, group in df[keep].groupby("underlying"):
                self._save_symbol_day(sym, date_str, group)

    def _process_quotes_chunk(self, df: pd.DataFrame, date_str: str):
        need = {"ticker", "sip_timestamp", "bid_price", "ask_price"}
        if not need.issubset(df.columns):
            raise ValueError(f"quotes chunk missing cols: {need - set(df.columns)}")
        df = self._add_underlying(df)
        df = df[df["underlying"].isin(TARGET_SYMBOLS)].copy()
        if df.empty:
            return
        df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True).dt.tz_convert(
            "America/New_York"
        )
        if AGGREGATE_QUOTES_TO_1S:
            df["sec"] = df["timestamp"].dt.floor("s")
            # 每秒最后一档
            df = df.sort_values("timestamp").groupby(["underlying", "ticker", "sec"], as_index=False).tail(1)
            df["timestamp"] = df["sec"]
            df = df.drop(columns=["sec"])
        keep = [
            c
            for c in (
                "ticker",
                "underlying",
                "clean_ticker",
                "timestamp",
                "sip_timestamp",
                "bid_price",
                "bid_size",
                "ask_price",
                "ask_size",
            )
            if c in df.columns
        ]
        for sym, group in df[keep].groupby("underlying"):
            self._save_symbol_day(sym, date_str, group)

    def process_and_filter(self, file_paths, date_obj):
        """读取下载的 CSV，筛选目标股票，分发存储，提取合约，统一格式"""
        if not file_paths:
            return

        date_str = date_obj.strftime("%Y-%m-%d")
        logger.info(f"⚙️ 正在处理清洗 {date_str} | kind={DATA_KIND} | prefix={S3_PREFIX}")

        # quotes / trades 可能很大：分块读
        chunksize = 1_000_000 if DATA_KIND in {"trades", "quotes"} else None

        for fp in file_paths:
            try:
                if DATA_KIND == "minute":
                    df = pd.read_csv(fp)
                    self._process_minute_chunk(df, date_str)
                elif DATA_KIND == "trades":
                    for chunk in pd.read_csv(fp, chunksize=chunksize):
                        self._process_trades_chunk(chunk, date_str)
                elif DATA_KIND == "quotes":
                    usecols = [
                        "ticker",
                        "sip_timestamp",
                        "bid_price",
                        "bid_size",
                        "ask_price",
                        "ask_size",
                    ]
                    for chunk in pd.read_csv(fp, chunksize=chunksize, usecols=usecols):
                        self._process_quotes_chunk(chunk, date_str)
                else:
                    raise ValueError(DATA_KIND)
            except Exception as e:
                logger.error(f"❌ 处理文件失败 {fp}: {e}")

    def cleanup_raw(self, file_paths):
        """删除临时文件以释放空间"""
        for fp in file_paths:
            try:
                os.remove(fp)
            except: pass

    def run(self):
        current = START_DATE
        total_days = (END_DATE - START_DATE).days + 1
        
        with tqdm(total=total_days, desc="S3 Pipeline Progress") as pbar:
            while current <= END_DATE:
                current_date_str = current.strftime('%Y-%m-%d')
                
                # --- 断点续传逻辑核心 ---
                if current_date_str in self.processed_dates:
                    # logger.info(f"⏭️ 跳过已完成的日期: {current_date_str}")
                    current += datetime.timedelta(days=1)
                    pbar.update(1)
                    continue

                # 1. 下载当天的所有 Part 文件
                raw_files = self.download_day(current)
                
                if raw_files:
                    # 2. 清洗、分发、提取合约
                    self.process_and_filter(raw_files, current)
                    
                    # 3. 删除巨大的原始文件 (重要！否则硬盘会爆)
                    self.cleanup_raw(raw_files)
                    
                # 标记该日期已彻底处理完毕 (不管是有数据还是一日休市)
                self._mark_date_completed(current_date_str)
                
                # 4. 实时保存一次合约列表 (防止中途崩溃数据丢失)
                if current.day % 5 == 0: # 每5天保存一次索引
                    self.save_contracts_list()

                current += datetime.timedelta(days=1)
                pbar.update(1)
        
        # 最后再一次保存所有合约列表
        self.save_contracts_list()
        logger.info("✅ 所有任务完成！数据已清洗并归档。")

if __name__ == "__main__":
    print(f"DATA_KIND: {DATA_KIND}")
    print(f"S3_PREFIX: {S3_PREFIX}")
    print(f"目标股票池: {len(TARGET_SYMBOLS)} 支 → {TARGET_SYMBOLS}")
    print(f"日期: {START_DATE} → {END_DATE}")
    print(f"数据存放于: {PROCESSED_DIR}")
    print(f"临时缓存于: {RAW_DIR} (处理后自动删除)")
    if DATA_KIND == "trades":
        print(f"AGGREGATE_TRADES_TO_1S: {AGGREGATE_TRADES_TO_1S}")
    if DATA_KIND == "quotes":
        print(f"AGGREGATE_QUOTES_TO_1S: {AGGREGATE_QUOTES_TO_1S}")
        print("⚠️ quotes 单日可达 ~100GB+，确认磁盘空间后再跑")

    pipeline = S3Pipeline()
    pipeline.run()
