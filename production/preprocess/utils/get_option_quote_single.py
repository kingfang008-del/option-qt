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

# --- 全局設定 ---
DATA_START_DATE = datetime.date(2022, 1, 1)

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
        existing_files = {}
        try:
            for f in os.listdir(symbol_dir):
                if f.endswith('.parquet'):
                    full_path = os.path.join(symbol_dir, f)
                    existing_files[f] = os.path.getsize(full_path)
        except Exception:
            existing_files = {}

        # 6. 返回任務打包清單
        return [(client, contract, existing_files, symbol_dir) for contract in contracts_to_download]

    def download_contract_data_optimized(self, client: RESTClient, contract: OptionsContract, existing_files: dict, symbol_dir: str):
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
            return

        file_year_str = exp_date.strftime('%Y')
        filename_only = f"{ticker_str}_quotes_{file_year_str}.parquet"
        full_path = os.path.join(symbol_dir, filename_only)
        
        # 斷點續傳
        if filename_only in existing_files and existing_files[filename_only] > 100:
            return

        try:
            # 使用 list_quotes 下載逐筆 Quotes 數據
            quotes_generator = client.list_quotes(
                ticker=contract.ticker, 
                timestamp_gte=real_start_date.strftime("%Y-%m-%d"), 
                timestamp_lte=real_end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            quotes_list = []
            for q in quotes_generator:
                # 解析 Polygon SDK 的 Quote 物件，如果沒有 sip_timestamp 就嘗試 participant_timestamp
                ts = getattr(q, 'sip_timestamp', getattr(q, 'participant_timestamp', None))
                quotes_list.append({
                    'timestamp': ts,
                    'bid': getattr(q, 'bid_price', 0.0),
                    'ask': getattr(q, 'ask_price', 0.0),
                    'bid_size': getattr(q, 'bid_size', 0),
                    'ask_size': getattr(q, 'ask_size', 0)
                })
            
            if quotes_list:
                df = pd.DataFrame(quotes_list)
                df.dropna(subset=['timestamp'], inplace=True)
                
                # 盤口過濾：剔除異常和無效報價
                df = df[
                    (df['bid'] > 0.0) & 
                    (df['ask'] >= df['bid']) &
                    (df['bid_size'] > 0) & 
                    (df['ask_size'] > 0)
                ].copy()
                
                if df.empty:
                    return

                # 時間轉換 (納秒 ns -> datetime) 並轉換至美東時區
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert(self.eastern)
                
                # ====================================================
                # 將幾萬條 Tick 數據，降採樣壓縮成 1 分鐘級別的快照
                # ====================================================
                # 取出分鐘級時間槽
                df['minute_ts'] = df['timestamp'].dt.floor('min')
                # 交易時段過濾 (09:30 - 16:00)
                time_series = df['minute_ts'].dt.time
                df = df[(time_series >= datetime.time(9, 30)) & (time_series < datetime.time(16, 0))]
                
                if df.empty:
                    return
                    
                # 在每一分鐘內，保留最後一筆（最新）的盤口快照
                df = df.sort_values('timestamp').drop_duplicates(subset=['minute_ts'], keep='last').copy()
                df['timestamp'] = df['minute_ts']
                
                # ====================================================
                # 特徵工程: 計算 Mid-Price 和其他特徵 (和 S3 下載版對齊)
                # ====================================================
                df['mid_price'] = (df['bid'] + df['ask']) / 2.0
                df['open'] = df['mid_price']
                df['high'] = df['mid_price']
                df['low'] = df['mid_price']
                df['close'] = df['mid_price']
                df['volume'] = df['bid_size'] + df['ask_size']
                df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
                df['volume_imbalance'] = (df['bid_size'] - df['ask_size']) / df['volume']
                df['ticker'] = contract.ticker
                
                cols_to_keep = [
                    'timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                    'bid', 'ask', 'bid_size', 'ask_size', 'spread_pct', 'volume_imbalance'
                ]
                df = df[cols_to_keep]
                
                df.to_parquet(full_path, index=False )
                
                # 回傳生成的檔案大小 (bytes)，用於進度條預估總容量
                try:
                    return os.path.getsize(full_path)
                except Exception:
                    return 0
                    
        except Exception as e:
            # 發生網絡或極端報錯時，直接 pass 留給下次重試
            pass
        return 0

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
            
        workers_count = min(total_contracts, 100)
        
        print(f"\n[階段 2/2] 🚀 準備高速並發下載 {total_contracts} 個合約的 Quotes 特徵數據")
        print(f"全局並行任務數: {workers_count}")
        print("-" * 50)

        accumulated_bytes = 0
        valid_files_count = 0

        # 全局進度條：可以直觀看到還剩多少時間！
        with tqdm(total=total_contracts, desc="總下載進度", unit="合約", smoothing=0.1) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers_count) as executor:
                # 提交所有合約（攤平跨所有股票）到全局線程池
                futures = [
                    executor.submit(self.download_contract_data_optimized, *task) 
                    for task in all_download_tasks
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        file_bytes = future.result()
                        # 把成功寫盤的真實檔案大小累加上去，並更新有效統計數量
                        if file_bytes is not None and file_bytes > 0:
                            accumulated_bytes += file_bytes
                            valid_files_count += 1
                            
                    except Exception as exc:
                        pass
                    
                    # 計算預先估計數據總量的邏輯 (GigaBytes)
                    if valid_files_count > 0:
                        avg_bytes_per_file = accumulated_bytes / valid_files_count
                        estimated_total_bytes = avg_bytes_per_file * total_contracts
                        est_gb = estimated_total_bytes / (1024 ** 3)
                        pbar.set_postfix_str(f"預估總大小: {est_gb:.2f} GB")
                    
                    # 每完成一個合約，更新全局剩餘時間
                    pbar.update(1)
        
        print("-" * 50)
        print("🎉 所有數據同步完成！")

if __name__ == "__main__":
    API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp"
    DB_PATH = "/home/kingfang007/notebook/stocks.db"
    DATA_ROOT = "/mnt/s990/data/option_quote"
    
    # 手動開關: 是否要從 API 拉取最新合約名單？ 
    # False = 絕對不連 Polygon API 抓合約，純用本地現有的緩存 contracts_list.txt 直接開跑第二階段
    FORCE_FETCH_CONTRACTS = False
    
    downloader = OptionDataDownloader(API_KEY, DB_PATH, DATA_ROOT, force_fetch=FORCE_FETCH_CONTRACTS)
    downloader.run()