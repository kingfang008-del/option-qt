#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 文件名: ibkr.py
# 描述: [实盘数据源头 - V3.0 Turbo Streaming]
#       1. [Booster Mode] 假设拥有 ~400 行行情额度。
#       2. [Full Streaming] 正股、期权、持仓全部使用 reqMktData 实时订阅。
#       3. [Zero IO Wait] 快照循环不再等待网络 IO，直接读取内存 Ticker 缓存。
#       4. [Dynamic Mgmt] 动态管理订阅线，自动释放不再需要的期权订阅。

import asyncio
import logging
import redis
import datetime
import pandas as pd
import numpy as np
import math
import json
import pytz
from typing import List, Dict, Set, Optional, Tuple, Any
from ib_insync import IB, Stock, Option, Contract, Ticker, util

# ================= 配置区 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IBKR_Live_Turbo")

NY_TZ = pytz.timezone('America/New_York')
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
STREAM_KEY_MARKET_SNAPSHOT = 'market_snapshot_stream'

# [发布频率]
# 既然是全流式，我们可以提高快照发布频率给 RL
# 0.5秒一次快照，既保证实时性，又不至于让 Redis 爆炸
PUBLISH_INTERVAL = 0.5 

# [分桶定义]
BUCKET_RULES = {
    'P_ATM_50':   ('P', 1.00, 0),  # 当月 ATM Put
    'P_OTM_25':   ('P', 0.95, 0),  # 当月 OTM Put
    'C_ATM_50':   ('C', 1.00, 0),  # 当月 ATM Call
    'C_OTM_25':   ('C', 1.05, 0),  # 当月 OTM Call
    'P_NEXT_ATM': ('P', 1.00, 1),  # 次月 ATM Put (用于 Term Structure)
    'C_NEXT_ATM': ('C', 1.00, 1),  # 次月 ATM Call (用于 Term Structure)
}

class IBKRConnector:
    def __init__(self, client_id=1):
        self.ib = IB()
        self.client_id = client_id
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        
        self.subscribed_symbols: Set[str] = set()
        
        # ----------------------------------------------------------
        # 核心 Ticker 缓存 (全部为 Live Streaming 对象)
        # ----------------------------------------------------------
        # 正股 Ticker {symbol: Ticker}
        self.stock_tickers: Dict[str, Ticker] = {}
        
        # 分桶 Ticker {symbol: {'P_ATM_50': Ticker, ...}}
        self.bucket_tickers: Dict[str, Dict[str, Ticker]] = {}
        
        # 遗留持仓 Ticker {conId: Ticker}
        self.legacy_tickers: Dict[int, Ticker] = {}
        
        # 辅助：记录当前订阅的合约 ConId，防止重复订阅或用于清理
        # {symbol: {bucket_name: conId}}
        self.active_bucket_map: Dict[str, Dict[str, int]] = {}

    def get_ny_timestamp(self) -> float:
        return datetime.datetime.now(NY_TZ).timestamp()

    async def connect(self, host='127.0.0.1', port=7497):
        logger.info(f"Connecting to IBKR ({host}:{port}) with Turbo Mode...")
        await self.ib.connectAsync(host, port, clientId=self.client_id)
        
        # 设置为 Live Data
        self.ib.reqMarketDataType(1) 
        logger.info("✅ IBKR Connected (Live Streaming Enabled).")

    # ==========================================================================
    # 1. 订阅管理 (Streaming Subscription)
    # ==========================================================================
    
    async def start_stock_stream(self, symbols: List[str]):
        """订阅正股流"""
        self.subscribed_symbols.update(symbols)
        for sym in symbols:
            if sym in self.stock_tickers: continue
            
            contract = Stock(sym, 'SMART', 'USD')
            # 100=OptVol, 101=OptOI, 104=HistVol, 106=ImpliedVol
            # 订阅这些 Generic Ticks 以便获得 IV 数据
            ticker = self.ib.reqMktData(contract, '100,101,104,106', False, False)
            
            self.stock_tickers[sym] = ticker
            self.bucket_tickers[sym] = {}
            self.active_bucket_map[sym] = {}
            
        logger.info(f"📡 Streaming {len(symbols)} Stocks (Base Layer).")
        await asyncio.sleep(1.0) # 等待数据预热

    async def subscribe_legacy_contract(self, contract: Contract):
        """
        订阅遗留持仓 (Persistent Stream)
        对于持仓，我们始终保持连接，直到平仓(由外部逻辑控制取消)
        """
        if not contract.conId:
            await self.ib.qualifyContractsAsync(contract)
            
        if contract.conId in self.legacy_tickers:
            return 

        # 实时流订阅
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.legacy_tickers[contract.conId] = ticker
        logger.info(f"🕯️ Legacy Stream Added: {contract.localSymbol} ({contract.conId})")

    def _update_bucket_subscription(self, symbol: str, bucket_name: str, new_contract: Contract):
        """
        [智能订阅切换]
        如果 Bucket 对应的合约变了(比如 Strike 变了)，取消旧的订阅，订阅新的。
        这对于节省 Lines 额度至关重要。
        """
        current_map = self.active_bucket_map.get(symbol, {})
        old_con_id = current_map.get(bucket_name)
        
        # 如果合约没变，直接返回
        if old_con_id == new_contract.conId:
            return

        # 1. 取消旧订阅 (释放 Line)
        if old_con_id:
            # 找到旧 Ticker
            old_ticker = self.bucket_tickers[symbol].get(bucket_name)
            if old_ticker:
                self.ib.cancelMktData(old_ticker.contract)
                # logger.debug(f"Testing: Unsubscribed old {bucket_name} for {symbol}")

        # 2. 建立新订阅
        # 请求 greeks (genericTickList不用特意填，Option默认带Greeks)
        new_ticker = self.ib.reqMktData(new_contract, '', False, False)
        
        # 3. 更新缓存
        self.bucket_tickers[symbol][bucket_name] = new_ticker
        self.active_bucket_map[symbol][bucket_name] = new_contract.conId
        # logger.info(f"🔄 Stream Switched [{symbol}-{bucket_name}]: {new_contract.localSymbol}")

    # ==========================================================================
    # 2. 维护循环 (Logic Resolver)
    # ==========================================================================
    
    async def maintenance_loop(self, interval=60):
        """
        定期检查 ATM Strike 是否发生变化。
        如果变化，计算新合约并调用 _update_bucket_subscription 进行切换。
        """
        logger.info("🛠 Turbo Maintenance Loop Started...")
        while self.ib.isConnected():
            try:
                for sym in list(self.subscribed_symbols):
                    s_ticker = self.stock_tickers.get(sym)
                    if not s_ticker: continue
                    
                    # 获取现价
                    price = s_ticker.last if s_ticker.last else s_ticker.close
                    if not price or math.isnan(price): 
                        if s_ticker.bid > 0 and s_ticker.ask > 0:
                            price = (s_ticker.bid + s_ticker.ask)/2
                    
                    if not price or price <= 0: continue
                    
                    # 解析当前应有的合约
                    target_contracts = await self._resolve_bucket_contracts(sym, price)
                    
                    # 更新订阅状态
                    for b_name, contract in target_contracts.items():
                        self._update_bucket_subscription(sym, b_name, contract)
                        
            except Exception as e:
                logger.error(f"Maintenance Error: {e}")
            
            await asyncio.sleep(interval)

    async def _resolve_bucket_contracts(self, symbol: str, spot: float) -> Dict[str, Contract]:
        """寻找符合 Moneyness 定义的合约"""
        try:
            # 获取期权链参数 (使用缓存优化，这里简化为直接调)
            chains = await self.ib.reqSecDefOptParamsAsync(symbol, '', 'STK', 0)
            if not chains: return {}
            
            chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])
            
            # 筛选日期
            expirations = sorted([d for d in chain.expirations if d], key=lambda x: x)
            today_str = datetime.datetime.now().strftime('%Y%m%d')
            valid_exps = [e for e in expirations if e > today_str]
            if not valid_exps: return {}

            dt_now = datetime.datetime.now()
            target_front = dt_now + datetime.timedelta(days=30)
            target_next = dt_now + datetime.timedelta(days=60)
            
            def get_nearest(targets, ref_date):
                return min(targets, key=lambda x: abs((datetime.datetime.strptime(x, '%Y%m%d') - ref_date).days))

            exp_front = get_nearest(valid_exps, target_front)
            exp_next = get_nearest(valid_exps, target_next)
            
            # 筛选 Strike
            strikes = sorted(chain.strikes)
            resolved = {}
            
            for b_name, (right, moneyness, m_offset) in BUCKET_RULES.items():
                target_strike = spot * moneyness
                expiry = exp_next if m_offset == 1 else exp_front
                
                best_strike = min(strikes, key=lambda x: abs(x - target_strike))
                
                contract = Option(symbol, expiry, best_strike, right, 'SMART')
                resolved[b_name] = contract
            
            # 批量 Qualify (获取 conId) - 这里的开销是可以接受的，因为 60s 才做一次
            # 且只有在切换合约时才有意义
            await self.ib.qualifyContractsAsync(*resolved.values())
            return resolved

        except Exception as e:
            logger.error(f"Resolve Error ({symbol}): {e}")
            return {}

    # ==========================================================================
    # 3. 发布循环 (Publisher - Zero IO Wait)
    # ==========================================================================

    async def publisher_loop(self):
        """
        高频发布快照。
        不再等待网络请求，直接读取 self.*_tickers 中的内存数据。
        """
        logger.info(f"🚀 Turbo Publisher Started (Interval: {PUBLISH_INTERVAL}s)...")
        
        while self.ib.isConnected():
            loop_start = datetime.datetime.now()
            try:
                frame_time = self.get_ny_timestamp()
                batch_snapshot = {}

                for sym in list(self.subscribed_symbols):
                    # --- A. 正股 ---
                    s_t = self.stock_tickers.get(sym)
                    if not s_t: continue
                    
                    price = self._extract_price(s_t)
                    if price <= 0: continue
                    
                    # --- B. 分桶期权 ---
                    buckets_data = {}
                    front_ivs = []
                    next_ivs = []
                    
                    sym_buckets = self.bucket_tickers.get(sym, {})
                    for b_name, t in sym_buckets.items():
                        opt_p = self._extract_price(t)
                        
                        # 从流数据中读取 Greeks (ib_insync 会自动更新 t.modelGreeks)
                        iv = t.modelGreeks.impliedVol if t.modelGreeks else 0.0
                        delta = t.modelGreeks.delta if t.modelGreeks else 0.0
                        
                        if opt_p > 0:
                            buckets_data[b_name] = {
                                'price': float(opt_p),
                                'iv': float(iv),
                                'delta': float(delta),
                                'conId': t.contract.conId
                            }
                            
                            # 收集 TS 数据
                            if 'ATM' in b_name and 'NEXT' not in b_name and iv > 0.01:
                                front_ivs.append(iv)
                            if 'NEXT_ATM' in b_name and iv > 0.01:
                                next_ivs.append(iv)

                    # 计算 Term Structure
                    ts_val = 0.0
                    if front_ivs and next_ivs:
                        diff = np.mean(next_ivs) - np.mean(front_ivs)
                        if abs(diff) < 0.5: ts_val = float(diff)

                    # --- C. 遗留持仓 ---
                    legacy_data = {}
                    # 遍历所有 legacy tickers (优化: 生产环境可以用 map 索引加速)
                    for conId, t in self.legacy_tickers.items():
                        if t.contract.symbol == sym:
                            lp = self._extract_price(t)
                            if lp > 0:
                                legacy_data[conId] = {'price': float(lp)}

                    # --- D. 组装 ---
                    batch_snapshot[sym] = {
                        'price': float(price),
                        'vol': float(s_t.volume if s_t.volume else 0),
                        'iv': float(s_t.modelGreeks.impliedVol if s_t.modelGreeks else 0),
                        'buckets': buckets_data,
                        'legacy': legacy_data,
                        'term_structure': ts_val
                    }

                # --- E. 推送 ---
                if batch_snapshot:
                    payload = {'ts': frame_time, 'data': batch_snapshot}
                    self.redis_client.xadd(
                        STREAM_KEY_MARKET_SNAPSHOT, 
                        {'json': json.dumps(payload)}, 
                        maxlen=1000
                    )

            except Exception as e:
                logger.error(f"Publisher Error: {e}")
            
            # 极速休眠
            elapsed = (datetime.datetime.now() - loop_start).total_seconds()
            sleep_time = max(0.01, PUBLISH_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

    def _extract_price(self, ticker: Ticker) -> float:
        """从 Streaming Ticker 中提取最优价格"""
        # 实时流优先取 Last，如果无成交取 Mid
        if ticker.last and not math.isnan(ticker.last): return ticker.last
        if ticker.bid > 0 and ticker.ask > 0: return (ticker.bid + ticker.ask) / 2.0
        if ticker.close and not math.isnan(ticker.close): return ticker.close
        # 期权专用兜底
        if ticker.modelGreeks and ticker.modelGreeks.optPrice > 0: return ticker.modelGreeks.optPrice
        return 0.0

    def run(self, symbols: List[str]):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.connect())
            loop.run_until_complete(self.start_stock_stream(symbols))
            
            # 启动并发任务
            loop.create_task(self.maintenance_loop(60))        # 60秒维护一次合约订阅
            loop.create_task(self.publisher_loop())            # 0.5秒发布一次快照
            
            logger.info("🚀 Turbo System Started. Streaming everything...")
            self.ib.run()
        except KeyboardInterrupt:
            self.ib.disconnect()

if __name__ == "__main__":
    # 示例
    connector = IBKRConnector(client_id=101)
    connector.run(["AAPL", "NVDA", "TSLA", "AMD"])