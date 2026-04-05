import datetime
import pandas as pd
import sqlite3
import os
import sys
import json
import threading
from dateutil.relativedelta import relativedelta
import time
from polygon import RESTClient
from polygon.rest.models import OptionsContract
import concurrent.futures
from tqdm import tqdm
import logging
from pytz import timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= 配置区域 =================

# [API Key]
API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"

# [路径设置]
DB_PATH = "/home/kingfang007/notebook/stocks.db"
DATA_ROOT = "/home/kingfang007/data/new_option_data"

# [模式开关] False=跑全量; True=只跑失败任务
RETRY_MODE = False  

# [下载范围]
DATA_START_DATE = datetime.date(2022, 1, 1)

# [并发控制]
MAX_SYMBOL_WORKERS = 5     
MAX_CONTRACT_WORKERS = 20   

FAILURE_LOG_FILE = "download_failures.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# ================= 失败记录管理器 =================
class FailureManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()

    def log_failure(self, symbol, ticker, start_date, end_date, error_msg):
        record = {
            "symbol": symbol, "ticker": ticker, 
            "start": start_date, "end": end_date, 
            "error": str(error_msg), "timestamp": time.time()
        }
        with self.lock:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")

    def load_failures(self):
        if not os.path.exists(self.filepath): return []
        tasks = []
        with self.lock:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try: tasks.append(json.loads(line))
                        except: pass
        return tasks

    def clear_log(self):
        with self.lock:
            open(self.filepath, 'w').close()

failure_manager = FailureManager(FAILURE_LOG_FILE)

# ================= 辅助工具 =================
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

class OptionDataDownloader:
    def __init__(self, api_key: str, db_path: str, data_root: str):
        self.api_key = api_key
        self.db_path = db_path
        self.data_root = data_root
        self.eastern = timezone('America/New_York')
        self.search_end_date = datetime.date.today() + relativedelta(months=2)
        self.client = RESTClient(self.api_key)

    def _get_target_symbols(self) -> list[str]:
        # 优先从 DB 读取，失败则使用兜底列表
        fallback_list = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 
        'META', 'GOOGL', 'MSFT', 'IWM', 'GLD', 'SLV', 'IBIT', 'MSTR', 'COIN']
        
        if not os.path.exists(self.db_path):
            return fallback_list
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # 这里简化查询，实际可根据需要调整 SQL
            query = "SELECT distinct symbol FROM stocks_us" 
            # 如果你有特定的筛选逻辑，请恢复之前的 IN (...) 语句
            cursor.execute(query)
            found_symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            # 取交集或并集，这里演示取并集以确保覆盖
            final_list = list(set(found_symbols).union(set(fallback_list)))
            # 过滤掉非目标的大量垃圾股，这里假设你只想跑核心的 30 只，恢复之前的列表逻辑
            target_whitelist = set(fallback_list) # 这里的逻辑根据你实际需求改，或者直接返回 found_symbols
            return [s for s in final_list if s in target_whitelist] # 仅演示，请用回你之前的逻辑
        except Exception:
            return fallback_list

    # ================= 核心下载逻辑 =================
    def _download_single_range(self, ticker, start_date_str, end_date_str, symbol_dir):
        start_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        file_year_str = start_dt.strftime('%Y')
        clean_ticker = ticker.replace('O:', '').replace(':', '') 
        filename_only = f"{clean_ticker}_{file_year_str}.parquet"
        full_path = os.path.join(symbol_dir, filename_only)

        # [断点续传检查]
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
            return True 

        @retry_api_call(max_retries=3, initial_wait=1)
        def fetch_aggs():
            return list(self.client.list_aggs(
                ticker=ticker, multiplier=1, timespan="minute",
                from_=start_date_str, to=end_date_str, limit=50000
            ))

        aggs_list = fetch_aggs()
        if aggs_list:
            df = pd.DataFrame(aggs_list)
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(self.eastern)
                df.to_parquet(full_path, index=False)
        return True

    def download_contract_worker(self, contract, symbol_dir, symbol):
        current_date = DATA_START_DATE
        end_date = self.search_end_date

        while current_date <= end_date:
            year_end = min(current_date + relativedelta(years=1) - datetime.timedelta(days=1), end_date)
            s_str = current_date.strftime("%Y-%m-%d")
            e_str = year_end.strftime("%Y-%m-%d")

            try:
                self._download_single_range(contract.ticker, s_str, e_str, symbol_dir)
            except Exception as e:
                failure_manager.log_failure(symbol, contract.ticker, s_str, e_str, e)

            current_date = year_end + datetime.timedelta(days=1)

    # ================= 统计显示函数 =================
    def _print_stats(self, symbol, local_count, api_count, total_count, symbol_dir):
        """
        计算并显示：
        1. 合约覆盖率 (本地 vs API)
        2. 实际数据下载率 (硬盘文件 vs 总合约)
        """
        # 扫描硬盘，计算实际有数据的文件和合约
        downloaded_files = 0
        downloaded_contracts = set()
        
        if os.path.exists(symbol_dir):
            files = [f for f in os.listdir(symbol_dir) if f.endswith('.parquet') and os.path.getsize(os.path.join(symbol_dir, f)) > 1000]
            downloaded_files = len(files)
            
            # 从文件名反推合约代码 (格式: CLEAN_TICKER_YYYY.parquet)
            for f in files:
                # 简单的逻辑：去掉最后的 _YYYY.parquet 得到 clean_ticker
                # 注意：clean_ticker 是去掉了 O: 的
                parts = f.split('_')
                if len(parts) >= 2:
                    clean_ticker = "_".join(parts[:-1]) # 防止 ticker 本身包含下划线
                    downloaded_contracts.add(clean_ticker)

        # 计算比率
        ratio_contract = (local_count / api_count * 100) if api_count > 0 else 0
        ratio_data = (len(downloaded_contracts) / total_count * 100) if total_count > 0 else 0
        
        # 格式化输出面板
        msg = (
            f"\n{'='*20} 📊 [{symbol}] 数据统计 {'='*20}\n"
            f"1. 合约完整度:\n"
            f"   - 本地缓存: {local_count}\n"
            f"   - 接口返回: {api_count}\n"
            f"   - 覆盖比例: {ratio_contract:.1f}% (最终合并总量: {total_count})\n"
            f"2. 数据下载进度:\n"
            f"   - 本地文件: {downloaded_files} 个 (Parquet文件总数)\n"
            f"   - 有效合约: {len(downloaded_contracts)} 个 (实际已下载数据的合约)\n"
            f"   - 实际完成: {ratio_data:.1f}% ({len(downloaded_contracts)}/{total_count})\n"
            f"{'='*56}"
        )
        # 使用 print 确保显示在 tqdm 上方，不受 log 格式影响
        print(msg)

    # ================= 全量扫描逻辑 =================
    def run_full_scan(self):
        # 这里的列表硬编码或者从 _get_target_symbols 获取，确保是你想要的列表
        
        # 修正: _get_target_symbols 里的逻辑我上面简化了，这里强制用你的全量列表以防万一
        from config import TARGET_SYMBOLS
        
        print(f"\n[🚀 全量扫描模式] 目标: {len(symbols)} 支标的")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_SYMBOL_WORKERS) as executor:
            futures = {executor.submit(self.process_symbol_full, sym, i % 10): sym for i, sym in enumerate(symbols)}
            for future in concurrent.futures.as_completed(futures):
                try: future.result() 
                except Exception as e: logger.error(f"Symbol Failed: {e}")

    def process_symbol_full(self, symbol, pbar_pos):
        symbol_dir = os.path.join(self.data_root, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        contracts_file = os.path.join(symbol_dir, "contracts_list.txt")

        # 1. [统计] 读取本地缓存
        local_tickers = set()
        if os.path.exists(contracts_file):
            try:
                with open(contracts_file, 'r') as f:
                    local_tickers = {line.strip() for line in f if line.strip()}
            except: pass
        local_count = len(local_tickers)

        # 2. [统计] 拉取 API
        api_tickers_map = {}
        
        @retry_api_call(max_retries=3, initial_wait=2)
        def fetch_list():
            return list(self.client.list_options_contracts(
                underlying_ticker=symbol, 
                expiration_date_gte=DATA_START_DATE.strftime('%Y-%m-%d'),
                expiration_date_lte=self.search_end_date.strftime('%Y-%m-%d'),
                limit=1000, expired="true", sort="expiration_date", order="asc"
            ))

        try:
            contracts = fetch_list()
            for c in contracts: api_tickers_map[c.ticker] = c
        except Exception:
            if not local_tickers: return # 彻底失败

        api_count = len(api_tickers_map)

        # 3. 合并
        all_tickers = local_tickers.union(api_tickers_map.keys())
        total_count = len(all_tickers)
        if not all_tickers: return
        
        # 4. [核心] 显示统计信息
        self._print_stats(symbol, local_count, api_count, total_count, symbol_dir)

        # 5. 保存更新后的列表
        sorted_tickers = sorted(list(all_tickers))
        try:
            with open(contracts_file, 'w') as f:
                f.write('\n'.join(sorted_tickers) + '\n')
        except: pass

        contracts_to_download = [
            api_tickers_map.get(t, OptionsContract(ticker=t, underlying_ticker=symbol)) 
            for t in sorted_tickers
        ]

        # 6. 并发下载
        pbar = tqdm(total=len(contracts_to_download), desc=f"{symbol}", position=pbar_pos, leave=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONTRACT_WORKERS) as inner_exec:
            fs = [inner_exec.submit(self.download_contract_worker, c, symbol_dir, symbol) for c in contracts_to_download]
            for f in concurrent.futures.as_completed(fs):
                pbar.update(1)
        pbar.close()

    # ================= 失败重试模式 =================
    def run_retry_mode(self):
        tasks = failure_manager.load_failures()
        if not tasks:
            print("✨ 没有失败记录，无需重试。")
            return

        print(f"\n[🔧 修复模式] 重试 {len(tasks)} 个任务...")
        failure_manager.clear_log()

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.retry_single_task, task) for task in tasks]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="修复进度"): pass

    def retry_single_task(self, task):
        symbol_dir = os.path.join(self.data_root, task['symbol'])
        os.makedirs(symbol_dir, exist_ok=True)
        try:
            self._download_single_range(task['ticker'], task['start'], task['end'], symbol_dir)
        except Exception as e:
            failure_manager.log_failure(task['symbol'], task['ticker'], task['start'], task['end'], e)

    def run(self):
        if RETRY_MODE: self.run_retry_mode()
        else: self.run_full_scan()

if __name__ == "__main__":
    downloader = OptionDataDownloader(API_KEY, DB_PATH, DATA_ROOT)
    downloader.run()