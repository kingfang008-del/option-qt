import datetime
import pandas as pd
import sqlite3
import os
from dateutil.relativedelta import relativedelta
import time
from polygon import RESTClient
from polygon.rest.models import OptionsContract, Agg
import concurrent.futures
from tqdm import tqdm
import logging
from pytz import timezone
import threading
import random

# --- 全局設定 ---
DATA_START_DATE = datetime.date(2022, 1, 1)

def process_and_save_quotes(full_path: str, quotes_list: list, ticker: str, eastern_str: str = 'America/New_York'):
    """
    在獨立的多進程中執行 CPU 密集型的特徵工程和 ZSTD 壓縮
    獨立的模組級函數，確保在 macOS/Windows 的 Spawn 進程模式下可以被跨進程 Pickling 傳遞
    """
    eastern = timezone(eastern_str)
    try:
        df = pd.DataFrame(quotes_list)
        df.dropna(subset=['timestamp'], inplace=True)
        
        # 盤口過濾：剔除異常和無效報價
        df = df[
            (df['bid'] > 0.0) & 
            (df['ask'] >= df['bid']) &
            (df['bid_size'] > 0) & 
            (df['ask_size'] > 0)
        ].copy()
        
        if not df.empty:
            # 時間轉換 (納秒 ns -> datetime) 並轉換至美東時區
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert(eastern)
            
            # 取出分鐘級時間槽
            df['minute_ts'] = df['timestamp'].dt.floor('min')
            # 交易時段過濾 (09:30 - 16:00)
            time_series = df['minute_ts'].dt.time
            df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
            
            if not df.empty:
                # 在每一分鐘內，保留最後一筆（最新）的盤口快照
                df = df.sort_values('timestamp').drop_duplicates(subset=['minute_ts'], keep='last').copy()
                df['timestamp'] = df['minute_ts']
                
                df['mid_price'] = (df['bid'] + df['ask']) / 2.0
                df['open'] = df['mid_price']
                df['high'] = df['mid_price']
                df['low'] = df['mid_price']
                df['close'] = df['mid_price']
                df['volume'] = df['bid_size'] + df['ask_size']
                df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
                df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']
                df['ticker'] = ticker
                
                cols_to_keep = [
                    'timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                    'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance'
                ]
                df = df[cols_to_keep]
                
                # ZSTD 壓縮高度消耗 CPU，丟失 GIL 鎖以發揮多核算力
                df.to_parquet(full_path, index=False, compression='zstd' )
                
                return os.path.getsize(full_path)
    except Exception as e:
        pass
    return 0

# 日誌設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class OptionDataDownloader:
    """
    Polygon.io 選擇權數據下載器 (高並發與斷點續傳優化版)
    """
    def __init__(self, api_key: str, db_path: str, data_root: str, force_fetch: bool = True):
        self.api_key = api_key
        self.db_path = db_path
        self.data_root = data_root
        self.force_fetch = force_fetch
        self.eastern = timezone('America/New_York')
        
        # 用於解耦下載(I/O)和壓縮寫入(CPU) - 已改為多進程架構
        self.lock = threading.Lock()
        self.accumulated_bytes = 0
        self.valid_files_count = 0
        self.total_contracts = 0
        self.pending_tasks = 0
        self.pbar = None
        self.process_pool = None
        
        # 動態設定結束日期為今天 + 2個月
        self.search_end_date = datetime.date.today() + relativedelta(months=2)
        
        logger.info(f"下載策略配置: 日期範圍 {DATA_START_DATE} 至 {self.search_end_date} (未來2個月)")

    def _get_target_symbols(self) -> list[str]:
        """從資料庫獲取目標股票代號列表"""
        if not os.path.exists(self.db_path):
            logger.error(f"資料庫文件不存在: {self.db_path}")
            return []
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        from config import TARGET_SYMBOLS
        
         
        
        placeholders = ','.join(['?'] * len(TARGET_SYMBOLS))
        query = f"SELECT distinct symbol FROM stocks_us WHERE symbol IN ({placeholders})"
        cursor.execute(query, TARGET_SYMBOLS)
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
            
        logger.info(f"從資料庫找到 {len(symbols)} 支目標股票。")
        return symbols

    def prepare_symbol_tasks(self, symbol: str):
        """
        準備該 symbol 所有的下載任務清單，包含從 API 獲取最新合約。
        返回: list of tuples (client, contract, existing_files, symbol_dir)
        """
        client = RESTClient(self.api_key)
        symbol_dir = os.path.join(self.data_root, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        contracts_file = os.path.join(symbol_dir, "contracts_list.txt")

        # 1. 讀取本地已有的合約清單
        local_tickers = set()
        if os.path.exists(contracts_file):
            try:
                with open(contracts_file, 'r') as f:
                    local_tickers = {line.strip() for line in f if line.strip()}
            except Exception:
                pass

        # 2. 判斷是否需要從 API 拉取完整合約清單
        # 使用 self.force_fetch 變數，由使用者手動控制
        need_api_fetch = self.force_fetch
        api_tickers_map = {}
        
        if need_api_fetch:
            try:
                # 建立基本的過濾條件
                api_params = {
                    "underlying_ticker": symbol,
                    "expiration_date_gte": DATA_START_DATE.strftime('%Y-%m-%d'),
                    "expiration_date_lte": self.search_end_date.strftime('%Y-%m-%d'),
                    "expired": "true"
                }

                # 【終極優化方案】：只拉取「活著的/未來即將過期的合約」，永遠不再重複拉取歷史過期合約
                # 如果本地已經有緩存，歷史肯定已經掃過了，我們只關心最近新出的合約。
                if local_tickers:
                    fetch_start_date = datetime.date.today() - datetime.timedelta(days=7) # 往前多看7天防漏
                    api_params["expiration_date_gte"] = fetch_start_date.strftime('%Y-%m-%d')
                    logger.info(f"[{symbol}] API 同步已開啟。已有本地緩存({len(local_tickers)}個)，僅增量拉取 {fetch_start_date} 之後的合約。")
                else:
                    logger.info(f"[{symbol}] API 同步已開啟。無本地緩存，正在執行全量合約掃描 ({DATA_START_DATE} ~ )...")

                for contract in client.list_options_contracts(
                    limit=1000,
                    sort="expiration_date", 
                    order="asc",
                    **api_params
                ):
                    api_tickers_map[contract.ticker] = contract

            except Exception as e:
                logger.debug(f"[{symbol}] API 獲取合約列表失敗: {e}")
                if not local_tickers:
                    return []
        else:
            if not local_tickers:
                logger.warning(f"[{symbol}] 未開啟 API 合約同步，且本地無緩存合約，將跳過此股票。")
                return []
            else:
                logger.info(f"[{symbol}] 手動跳過 API 合約同步，直接使用本地緩存的 {len(local_tickers)} 個合約。")

        # 3. 合併清單並保留最全局
        all_tickers = local_tickers.union(api_tickers_map.keys())
        if not all_tickers:
            return []
            
        sorted_tickers = sorted(list(all_tickers))

        # 4. 覆寫回本地 cache (僅在需要 Fetch 時更新，避免多餘 I/O)
        if need_api_fetch:
            try:
                with open(contracts_file, 'w') as f:
                    f.write('\n'.join(sorted_tickers) + '\n')
            except Exception:
                pass

        # 將 Ticker 轉回 OptionsContract 物件
        # 對於純本地緩存的，我們人造一個 OptionsContract 物件，確保後續方法能執行
        contracts_to_download = [
            api_tickers_map.get(ticker, OptionsContract(ticker=ticker, underlying_ticker=symbol)) 
            for ticker in sorted_tickers
        ]

        # 5. 掃描本地已存在的 Parquet 檔案
        existing_files = set()
        try:
            with os.scandir(symbol_dir) as it:
                for entry in it:
                    if entry.name.endswith('.parquet') and entry.is_file():
                        if entry.stat().st_size > 100:
                            existing_files.add(entry.name)
        except Exception:
            pass

        # 6. 返回任務打包清單 (提早過濾，極速跳過)
        tasks = []
        for contract in contracts_to_download:
            try:
                ticker_str = contract.ticker.replace('O:', '')
                exp_date_str = ticker_str[-15:-9]
                exp_date = datetime.datetime.strptime(exp_date_str, "%y%m%d").date()
                file_year_str = exp_date.strftime('%Y')
                filename_only = f"{ticker_str}_quotes_{file_year_str}.parquet"
                
                # 如果本地已經存在且大小合法，第一階段直接剔除，絕不加入第二階段的線程池
                if filename_only in existing_files:
                    continue
            except Exception:
                pass
            
            # 不傳遞共享的 client 以避免 urllib3 連接池鎖定，讓每個下載任務自建 Client
            tasks.append((self.api_key, contract, symbol_dir))
            
        return tasks

    def _processing_done_callback(self, future):
        """
        ProcessPoolExecutor 任務完成後的回調函數。
        用於更新進度條和統計信息。
        """
        with self.lock:
            self.pending_tasks -= 1
            self.pbar.update(1) # 無論成功失敗，都更新總進度條

            try:
                file_bytes = future.result()
                if file_bytes > 0:
                    self.accumulated_bytes += file_bytes
                    self.valid_files_count += 1
            except Exception as e:
                logger.error(f"後台處理任務失敗: {e}")
            
            if self.pbar and self.valid_files_count > 0:
                avg_bytes = self.accumulated_bytes / self.valid_files_count
                est_gb = (avg_bytes * self.total_contracts) / (1024 ** 3)
                self.pbar.set_postfix_str(f"排隊處理: {self.pending_tasks} | 預估: {est_gb:.2f} GB")


    def download_contract_data_optimized(self, api_key: str, contract: OptionsContract, symbol_dir: str):
        """
        下載單個合約的 Quote (買賣盤口) 數據，並壓縮為 1分鐘 K 線格式
        【核心優化】：根據期權代碼解析到期日，精準限制只下載合約存續期間(DTE)的數據。
        """
        try:
            ticker_str = contract.ticker.replace('O:', '')
            # O:AAPL251219C00150000 -> 後15位是 251219C00150000 -> 到期日為前6位
            exp_date_str = ticker_str[-15:-9]
            exp_date = datetime.datetime.strptime(exp_date_str, "%y%m%d").date()
        except Exception as e:
            logger.warning(f"無法解析合約過期日: {contract.ticker}")
            return

        # 這裡加入 DTE 參數 (最遠取前 65 天)，進一步大幅減少 API 請求跨度
        DTE_MAX_DAYS = 65 
        
        # 精準鎖定下載日期：結束=到期日，開始=到期日前 65天
        real_end_date = min(self.search_end_date, exp_date)
        ideal_start_date = exp_date - datetime.timedelta(days=DTE_MAX_DAYS)
        real_start_date = max(DATA_START_DATE, ideal_start_date)

        # 如果開始日期 > 結束日期，說明該合約在這個搜尋視窗內不存在/不符合要求，直接跳過
        if real_start_date > real_end_date:
            return "SKIPPED"

        file_year_str = exp_date.strftime('%Y')
        filename_only = f"{ticker_str}_quotes_{file_year_str}.parquet"
        full_path = os.path.join(symbol_dir, filename_only)
        
        # 二次防禦：如果在排隊期間被其他進程下載了，直接跳過
        if os.path.exists(full_path) and os.path.getsize(full_path) > 100:
            return "SKIPPED"

        try:
            # 讓每個線程擁有獨立的 RESTClient，徹底避開 urllib3 預設 maxsize=10 的連接池瓶頸
            client = RESTClient(api_key)
            
            quotes_list = []
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    quotes_list.clear() # 每次重試前清空可能殘留的半成品數據
                    # 使用 list_quotes 下載逐筆 Quotes 數據 (純網絡 I/O)
                    quotes_generator = client.list_quotes(
                        ticker=contract.ticker, 
                        timestamp_gte=real_start_date.strftime("%Y-%m-%d"), 
                        timestamp_lte=real_end_date.strftime("%Y-%m-%d"),
                        limit=50000
                    )
                    
                    for q in quotes_generator:
                        ts = getattr(q, 'sip_timestamp', getattr(q, 'participant_timestamp', None))
                        quotes_list.append({
                            'timestamp': ts,
                            'bid': getattr(q, 'bid_price', 0.0),
                            'ask': getattr(q, 'ask_price', 0.0),
                            'bid_size': getattr(q, 'bid_size', 0),
                            'ask_size': getattr(q, 'ask_size', 0)
                        })
                    
                    success = True
                    break # 成功獲取完整數據，跳出重試迴圈
                    
                except Exception as api_err:
                    if attempt < max_retries - 1:
                        # 指數退避策略: 加入隨機抖動 (Jitter)，避免所有被踢掉的線程在同一毫秒發起重連 (Thundering Herd)
                        sleep_time = (2 ** attempt) + random.uniform(0.1, 1.5)
                        logger.warning(f"[{contract.ticker}] 連線中斷 (第 {attempt+1} 次重試)，等待 {sleep_time:.2f} 秒後重連... 錯誤: {api_err}")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"[{contract.ticker}] 重試 {max_retries} 次依然失敗，放棄此合約。錯誤: {api_err}")
                        raise api_err # 重試全部失敗，拋出異常交給外層統一處理
            
            if success and quotes_list:
                # 反壓控制: 若進程池排隊任務 > 1500，稍微暫停網絡下載，防止記憶體被撐爆
                while self.pending_tasks > 1500:
                    time.sleep(0.1)

                with self.lock:
                    self.pending_tasks += 1
                    
                # 提交到多進程池
                future = self.process_pool.submit(process_and_save_quotes, full_path, quotes_list, contract.ticker, 'America/New_York')
                # 綁定回調，任務結束後由進程池的線程更新進度條
                future.add_done_callback(self._processing_done_callback)
                return "ENQUEUED"
                
        except Exception as e:
            logger.error(f"[{contract.ticker}] 下載或分發任務失敗: {e}")
            pass
        return "SKIPPED"

    def run(self):
        symbols = self._get_target_symbols()
        if not symbols:
            logger.warning("沒有目標股票。")
            return
            
        print("\n[階段 1/2] 正在掃描並統計所有股票的期權合約數 (本地緩存 + API同步)...")
        all_download_tasks = []
        
        # 提取任務階段不需要太多線程，主要受限於 API I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(symbols), 20)) as executor:
            futures = {executor.submit(self.prepare_symbol_tasks, symbol): symbol for symbol in symbols}
            # 為階段 1 加上簡單進度條
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="合約提取進度", unit="支股票"):
                try:
                    tasks = future.result()
                    all_download_tasks.extend(tasks)
                except Exception as exc:
                    symbol = futures[future]
                    logger.error(f"[{symbol}] 合約準備任務失敗: {exc}")
        
        total_contracts = len(all_download_tasks)
        if total_contracts == 0:
            print("沒有需要下載的合約任務。")
            return
            
        self.total_contracts = total_contracts
        
        # ⚠️ 服務器端主動踢人的原因：併發太高被 Polygon 防火牆攔截 (SSLEOFError)
        # 降速求穩：將極限網絡線程數從 120 降回 40 左右，這是大部份 API 服務器比較能容忍的長連接數量
        workers_count = min(total_contracts, 30) 
        # ZSTD 壓縮吃 CPU，我們派出所有的實體/邏輯核心進行運算，繞開 GIL
        writer_count = 20
        
        print(f"\n[階段 2/2] 🚀 準備高速並發下載 {total_contracts} 個合約的 Quotes 特徵數據")
        print(f"網絡下載線程數: {workers_count}")
        print(f"後台壓縮寫入進程 (多核利用): {writer_count}")
        print("-" * 50)

        self.accumulated_bytes = 0
        self.valid_files_count = 0
        self.pending_tasks = 0

        # 將後台專用的 CPU 壓縮進程池點火 ! 使用真正的進程 ProcessPool 撕裂 GIL !
        with concurrent.futures.ProcessPoolExecutor(max_workers=writer_count) as process_pool:
            self.process_pool = process_pool
            
            # 全局進度條：可以直觀看到還剩多少時間！
            with tqdm(total=total_contracts, desc="總下載並寫入進度", unit="合約", smoothing=0.1) as pbar:
                self.pbar = pbar
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers_count) as executor:
                    # 提交所有合約（攤平跨所有股票）到全局線程池
                    futures = [
                        executor.submit(self.download_contract_data_optimized, *task) 
                        for task in all_download_tasks
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            status = future.result()
                            # 若被跳過（本地有檔案或是沒有報價等），網絡線程直接在此更新進度條
                            # 若成功觸發寫入 (ENQUEUED)，則全權交給 writer_worker 回調負責更新進度條
                            if status == "SKIPPED":
                                with self.lock:
                                    self.pbar.update(1)
                        except Exception as exc:
                            with self.lock:
                                self.pbar.update(1)
                                
            self.pbar = None
            print("\n\n所有網絡下載請求已分發完畢，等待餘下佇列中的進程完成 ZSTD 壓縮落盤 (可見上方進度條)...")
        
        print("-" * 50)
        print("🎉 所有數據同步並壓縮落盤完成！")

if __name__ == "__main__":
    API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
    DB_PATH = "/home/kingfang007/notebook/stocks.db"
    DATA_ROOT = "/mnt/s990/data/option_quote"
    
    # 手動開關: 是否要從 API 拉取最新合約名單？ 
    # False = 絕對不連 Polygon API 抓合約，純用本地現有的緩存 contracts_list.txt 直接開跑第二階段
    FORCE_FETCH_CONTRACTS = False
    
    downloader = OptionDataDownloader(API_KEY, DB_PATH, DATA_ROOT, force_fetch=FORCE_FETCH_CONTRACTS)
    downloader.run()