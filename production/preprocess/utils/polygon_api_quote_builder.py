import httpx
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm
import logging
from dateutil import parser

# ================= 配置参数 =================
POLYGON_API_KEY = "JXuIcG_dpoRiCE6jP7c73nVWweEVSpUp" 
BASE_URL = "https://api.polygon.io/v3/quotes"

TARGET_SYMBOLS = ['AAPL'] # 先用 AAPL 测 1 天的数据！

START_DATE = datetime(2023, 3, 1) 
END_DATE = datetime(2023, 3, 1)

DTE_MIN = 3     
DTE_MAX = 65    

OUTPUT_DIR = "/home/kingfang007/data/new_option_data_polygon_api"
MAX_WORKERS = 2 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================

def get_contracts_for_underlying(symbol, target_date):
    """
    使用 Polygon Reference API 获取某一天、某个标的的所有期权合约。
    同时应用 DTE 过滤条件，减少后续 Quote API 的请求量。
    """
    date_str = target_date.strftime("%Y-%m-%d")
    contracts = []
    
    # 根据 DTE_MIN 和 DTE_MAX 计算到期日范围
    min_exp_date = (target_date + timedelta(days=DTE_MIN)).strftime("%Y-%m-%d")
    max_exp_date = (target_date + timedelta(days=DTE_MAX)).strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": symbol,
        "as_of": date_str,
        "expiration_date.gte": min_exp_date,
        "expiration_date.lte": max_exp_date,
        "limit": 1000,
        "apiKey": POLYGON_API_KEY
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            while url:
                resp = client.get(url, params=params if "cursor=" not in url else None)
                if resp.status_code == 429:
                    logger.warning("Rate limit hit getting contracts. Sleeping 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                data = resp.json()
                
                for result in data.get('results', []):
                     contracts.append(result['ticker'])
                
                next_url = data.get('next_url')
                if next_url:
                    url = f"{next_url}&apiKey={POLYGON_API_KEY}"
                    params = None # cursor handles pagination
                else:
                    url = None
                    
        return contracts
    except Exception as e:
        logger.error(f"Failed to fetch contracts for {symbol} on {date_str}: {e}")
        return []

def fetch_quotes_for_contract(contract_ticker, target_date):
    """
    注意：Polygon 的 Options Quotes REST API 是获取时间区间内所有 *逐笔 Tick*，
    没有直接的“1分钟聚合 Ask/Bid 快照”。
    """
    date_str = target_date.strftime("%Y-%m-%d")
    
    # 限制在交易时间内 (纽约时间 09:30 - 16:00) 
    # 将纽约时间转换为 UTC 纳秒时间戳由于夏令时比较复杂，这里简化，通过 API 参数限制日期，然后拿到数据用 Pandas 强转时区并过滤
    
    url = f"{BASE_URL}/{contract_ticker}"
    params = {
        "timestamp.gte": date_str, # 可以进一步精确到毫秒
        "timestamp.lte": date_str,
        "limit": 50000, 
        "sort": "timestamp",
        "order": "asc",
        "apiKey": POLYGON_API_KEY
    }
    
    quotes = []
    try:
        with httpx.Client(timeout=30.0) as client:
             while url:
                resp = client.get(url, params=params if "cursor=" not in url else None)
                if resp.status_code == 429:
                     time.sleep(60)
                     continue
                     
                resp.raise_for_status()
                data = resp.json()
                
                for item in data.get('results', []):
                    # item format: {'ask_price': 1.1, 'ask_size': 100, 'bid_price': 1.05, 'bid_size': 100, 'participant_timestamp': 1677682200000000000, ...}
                    quotes.append({
                        'ticker': contract_ticker,
                        'timestamp': item.get('sip_timestamp', item.get('participant_timestamp')),
                        'bid': item.get('bid_price'),
                        'ask': item.get('ask_price'),
                        'bid_size': item.get('bid_size', 0),
                        'ask_size': item.get('ask_size', 0)
                    })
                
                next_url = data.get('next_url')
                if next_url:
                    url = f"{next_url}&apiKey={POLYGON_API_KEY}"
                    params = None
                else:
                    url = None
        return quotes
    except Exception as e:
         return []

def process_daily_symbol(args):
    symbol, target_date = args
    date_str = target_date.strftime("%Y-%m-%d")
    
    # 1. 获取该股票当天所有的有效合约 (附带了 DTE 过滤)
    contracts = get_contracts_for_underlying(symbol, target_date)
    if not contracts:
        return f"⚠️ {symbol} {date_str}: 未找到合约。"
        
    logger.info(f"[{symbol} {date_str}] Found {len(contracts)} valid options contracts.")
    
    all_quotes = []
    # 2. 为每个合约单独调用 Quote REST API
    # 这是一个 N+1 问题 (1个期权请求 = N个底层期权链请求)
    for ticker in tqdm(contracts, desc=f"Fetching {symbol}"):
         contract_quotes = fetch_quotes_for_contract(ticker, target_date)
         all_quotes.extend(contract_quotes)
         time.sleep(0.01) # 防止触发 429 rate limit
         
    if not all_quotes:
        return f"⚠️ {symbol} {date_str}: 没有获取到任何 Quote 数据。"
        
    # 3. 聚合处理 (聚合逻辑同 S3 script)
    df = pd.DataFrame(all_quotes)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True).dt.tz_convert('America/New_York')
    df['minute_ts'] = df['timestamp'].dt.floor('min')
    
    # ... 后续处理逻辑与 S3 一致: RTH 过滤, drop_duplicates(keep='last') 取每分钟最后一笔, 计算 mid_price, 存 parquet
    
    return f"✅ {symbol} {date_str}: 成功处理 {len(all_quotes)} 笔原始 Quotes"

if __name__ == "__main__":
    # 演示代码
    print("开始通过 REST API 提取...");
