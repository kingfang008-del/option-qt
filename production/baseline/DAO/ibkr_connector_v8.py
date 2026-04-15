#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: ibkr_connector_v8.py
描述: [V8 生产修正版] = [V8 持久化架构] + [V11 存在性优先搜索算法]
核心修复:
    1. 恢复了 _find_contracts 的 V11 算法，解决 Error 200 问题。
    2. 保留了 Lock Persistence (合约锁持久化)，重启秒级恢复。
    3. 修复了启动时可能出现的 active_stocks 访问错误。
"""

import asyncio
import logging
import redis
import datetime
import pandas as pd
import numpy as np
import pickle
import sys
import os

# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import serialization_utils as ser
import os
import time
from typing import List, Dict, Set
import pytz
import psycopg2
from config import PG_DB_URL
from ib_insync import IB, Stock, Option, Order, Contract, TagValue

from config import LOG_DIR

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "IBKR_Connector.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("IBKR_Connector_Prod")

from config import (
    REDIS_CFG as _REDIS_BASE, NY_TZ, IBKR_ACCOUNT_ID as ACCOUNT_ID, IBKR_PORT,
    STREAM_FUSED_MARKET as STREAM_KEY_FUSED, HASH_OPTION_SNAPSHOT as HASH_KEY_SNAPSHOT,
    TRADING_ENABLED, DB_DIR, BUCKET_SPECS, TAG_TO_INDEX,
    ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS
)

NO_OPTION_LOCK_SYMBOLS = set(ALPHA_NORMALIZATION_EXCLUDE_SYMBOLS)

# 🚀 [核心对齐] Buckets 数据索引定义
IDX_PRICE = 0
IDX_DELTA = 1
IDX_GAMMA = 2
IDX_VEGA = 3
IDX_THETA = 4
IDX_STRIKE = 5
IDX_VOLUME = 6
IDX_IV = 7

class IBKRConnectorFinal:
    def __init__(self, host='127.0.0.1', port=IBKR_PORT, client_id=102):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.redis = redis.Redis(host=_REDIS_BASE['host'], port=_REDIS_BASE['port'], db=_REDIS_BASE.get('db', 0), decode_responses=False)
        
        self.active_stocks = {}      # 原始 Symbol -> Stock Contract
        self.last_spot_prices = {}   # Symbol -> Price
        self.locked_contracts = {}   # Symbol -> {Tag: Option Contract}
        self.last_iv_cache = {}      # 缓存 IV
        self.rfr_cache = None        # [NEW] 动态利率缓存
        self.last_tick_time = {}     # [NEW] 记录最后一次收到行情的时间
        
        self.current_lock_date = None
        self.initial_scan_done = False # [New] 用于防止 maintenance_loop 抢跑

        self.last_greeks_cache = {}  # [🔥 核心补丁 1] 新增：缓存全套希腊值
        self._use_tick_driven_publish = os.environ.get('IBKR_TICK_DRIVEN_PUBLISH', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
        self._publish_debounce_sec = max(
            0.0,
            float(os.environ.get('IBKR_TICK_PUBLISH_DEBOUNCE_MS', '80')) / 1000.0
        )
        self._last_bar_cache = {}
        self._last_total_volume = {}
        self._collecting_ts = None
        
        # [NEW] 绑定错误处理
        self.ib.errorEvent += self._on_error
        if self._use_tick_driven_publish:
            self.ib.pendingTickersEvent += self._on_pending_tickers
        logger.info("🧮 IB side Greeks disabled: streaming price/orderbook only; Greeks computed in realtime_feature_engine.")
        if self._use_tick_driven_publish:
            logger.info(f"⚡ Tick-driven publish enabled | debounce={self._publish_debounce_sec * 1000:.0f}ms")
        else:
            logger.info("🕔 Tick-driven publish disabled; using 5s bar trigger.")

    async def connect(self):
        """保持连接活跃"""
        while not self.ib.isConnected():
            try:
                logger.info(f"🔌 Connecting to IBKR ({self.host}:{self.port})...")
                await self.ib.connectAsync(self.host, self.port, self.client_id)
                self.ib.reqMarketDataType(1) # 1=Live
                
                # [NEW] Subscribe to account updates (needed to fetch account values)
                if ACCOUNT_ID:
                    self.ib.reqAccountUpdates(ACCOUNT_ID)
                else:
                    self.ib.reqAccountUpdates() # Default account
                    
                logger.info("✅ IBKR Connected & Subscribed to Account Updates.")
            except Exception as e:
                logger.error(f"Connection failed: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
                
    async def get_account_balance(self):
        """获取账户真实可用净资产 (NetLiquidation USD)"""
        try:
            if not self.ib.isConnected(): 
                return 0.0
            
            # 等待 IB 客户端消化更新数据
            await asyncio.sleep(0.5)
            
            vals = self.ib.accountValues()
            
            # [Fix] 根据用户要求，可交易资金以现金（AvailableFunds）为准，而非总资产（NetLiquidation）
            # 这样可以避免将已有持仓的市值作为额外购买力，导致超额下单。
            for v in vals:
                if v.tag == 'AvailableFunds' and v.currency == 'USD':
                    return float(v.value)
            
            for v in vals:
                if v.tag == 'TotalCashValue' and v.currency == 'USD':
                    return float(v.value)

            for v in vals:
                if v.tag == 'NetLiquidation' and v.currency == 'USD':
                    return float(v.value)
            for v in vals:
                if v.tag == 'NetLiquidationByCurrency' and v.currency == 'BASE':
                    return float(v.value)
            
            # 退而求其次，寻找 AvailableFunds
            for v in vals:
                if v.tag == 'AvailableFunds' and v.currency == 'USD':
                    return float(v.value)
            
            return 0.0
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            return 0.0

    # ================= 交易指令处理逻辑,暂时不使用这种流式发单，仅仅备用 =================
    async def command_execution_loop(self):
        """
        [完整版] 监听 Redis 交易指令流
        核心升级: 收到 PLACE_ORDER 后，不再手动拼单，而是调用 self.place_option_order 
                 以复用云端止损(Stop Loss)和算法单(Adaptive)逻辑。
        """
        logger.info("🎧 Command Execution Loop Started...")
        cmd_stream = "trade_command_stream" 
        
        try:
            self.redis.xgroup_create(cmd_stream, 'connector_group', mkstream=True, id='$')
        except: pass

        while self.ib.isConnected():
            try:
                # 阻塞读取指令
                resp = self.redis.xreadgroup('connector_group', 'worker_1', {cmd_stream: '>'}, count=5, block=1000)
                
                if not resp:
                    await asyncio.sleep(0.01)
                    continue
                    
                for _, msgs in resp:
                    for msg_id, data in msgs:
                        try:
                            # 兼容 pickle/msgpack 格式
                            payload = ser.unpack(data[b'data']) if b'data' in data else ser.unpack(data[b'pickle'])
                            action_type = payload.get('action')
                            
                            # =========================================
                            # 场景 A: 紧急撤单 (最高优先级)
                            # =========================================
                            if action_type == 'CANCEL_ALL':
                                reason = payload.get('reason', 'Unknown')
                                logger.warning(f"🚨 RECEIVED EMERGENCY CANCEL: {reason}")
                                self.ib.reqGlobalCancel()
                                logger.info("✅ Global Cancel Request Sent to IB.")
                            
                            # =========================================
                            # 场景 B: 普通下单 (PLACE_ORDER) -> 转调高级接口
                            # =========================================
                            elif action_type == 'PLACE_ORDER':
                                # 1. 再次检查全局交易开关
                                if not TRADING_ENABLED:
                                    logger.warning(f"🛑 Trading DISABLED. Ignoring order: {payload}")
                                    continue

                                # 2. 解析参数
                                # 注意: contract 对象反序列化后可能丢失 ib 绑定，最好用 localSymbol 重建
                                raw_contract = payload.get('contract')
                                symbol = raw_contract.symbol
                                local_symbol = raw_contract.localSymbol
                                sec_type = raw_contract.secType
                                exchange = raw_contract.exchange or 'SMART'
                                currency = raw_contract.currency or 'USD'
                                
                                # 重建干净的 Contract 对象
                                contract = Contract()
                                contract.symbol = symbol
                                contract.secType = sec_type
                                contract.localSymbol = local_symbol
                                contract.exchange = exchange
                                contract.currency = currency
                                
                                order_action = payload.get('order_action') # 'BUY' / 'SELL'
                                quantity = float(payload.get('quantity', 0))
                                order_type = payload.get('order_type', 'LMT')
                                price = float(payload.get('price', 0.0))
                                
                                # 获取止损比例 (如果 Orchestrator 传了就用，没传就默认)
                                stop_loss_pct = float(payload.get('stop_loss_pct', 0.0))

                                # 3. 调用核心下单逻辑 (含止损挂载 & DRY_RUN 检查)
                                logger.info(f"📨 Stream Trigger: Placing {order_action} {local_symbol}...")
                                
                                # [关键] 复用 self.place_option_order
                                self.place_option_order(
                                    contract=contract,
                                    action=order_action,
                                    qty=quantity,
                                    order_type=order_type,
                                    lmt_price=price,
                                    stop_loss_pct=stop_loss_pct
                                )
                                
                        except Exception as e:
                            logger.error(f"Command Process Error: {e}", exc_info=True)
                        finally:
                            # 无论成功失败，都要 ACK
                            self.r.xack(cmd_stream, 'connector_group', msg_id)
                            
            except Exception as e:
                logger.error(f"Command Loop Error: {e}")
                await asyncio.sleep(1)

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DB_URL)

    def _ensure_lock_table(self):
        """确保 PG 中存在合约锁表"""
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS contract_locks (
                            date TEXT,
                            symbol TEXT,
                            tag TEXT,
                            conId BIGINT,
                            expiry TEXT,
                            strike DOUBLE PRECISION,
                            p_right TEXT,
                            multiplier TEXT,
                            localSymbol TEXT,
                            tradingClass TEXT,
                            PRIMARY KEY (date, symbol, tag)
                        )''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Lock DB Init failed: {e}")

    def _save_locks(self):
        """[弃用] 批量保存已改用增量保存，保留空函数防报错"""
        pass

    def _save_single_lock(self, sym, tag, contract):
        """增量保存单个合约锁到当日 DB"""
        try:
            today_obj = datetime.datetime.now(NY_TZ).date()
            today_str = str(today_obj)
            
            self._ensure_lock_table()
            
            row = (
                today_str, sym, tag,
                int(contract.conId), contract.lastTradeDateOrContractMonth,
                float(contract.strike), contract.right, str(contract.multiplier),
                contract.localSymbol, contract.tradingClass
            )
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute('''
                INSERT INTO contract_locks (date, symbol, tag, conId, expiry, strike, p_right, multiplier, localSymbol, tradingClass)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, symbol, tag) DO UPDATE 
                SET conId=EXCLUDED.conId, expiry=EXCLUDED.expiry, strike=EXCLUDED.strike, p_right=EXCLUDED.p_right, 
                    multiplier=EXCLUDED.multiplier, localSymbol=EXCLUDED.localSymbol, tradingClass=EXCLUDED.tradingClass
            ''', row)
            conn.commit()
            conn.close()
            logger.info(f"💾 [POSTGRES] Saved lock {sym} {tag}")
        except Exception as e:
            logger.error(f"❌ Save single lock failed: {e}")

    def _load_locks(self):
        """从当日 PostgreSQL 加载合约锁，并过滤过期合约"""
        try:
            today_obj = datetime.datetime.now(NY_TZ).date()
            today_str = str(today_obj)
            today_ymd = today_obj.strftime('%Y%m%d') # 用于比对 expiry
            
            # Ensure table exists first
            self._ensure_lock_table()

            conn = self._get_pg_conn()
            c = conn.cursor()
            
            # Check if table has any rows (no need to check existence again after _ensure_lock_table)
            c.execute("SELECT 1 FROM contract_locks LIMIT 1")
            if not c.fetchone():
                conn.close()
                logger.info("⚠️ PostgreSQL 'contract_locks' table is empty. Starting fresh.")
                return False
                
            c.execute("SELECT date, symbol, tag, conId, expiry, strike, p_right, multiplier, localSymbol, tradingClass FROM contract_locks WHERE date=%s", (today_str,))
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                logger.info(f"⚠️ No locks found in PostgreSQL for {today_str}. Starting fresh.")
                return False

            cnt = 0
            for r in rows:
                date_val, sym, tag, conId, expiry, strike, right, multiplier, localSymbol, tradingClass = r
                
                # ==============================================================
                # [🔥 安全防线] 拒绝恢复已过期的废纸合约，以及 DTE 过短的合约
                # ==============================================================
                if str(expiry) < today_ymd:
                    logger.warning(f"⚠️ 拒绝恢复已过期合约: {sym} {tag} (到期日: {expiry})")
                    continue
                
                try:
                    expiry_date = datetime.datetime.strptime(str(expiry), '%Y%m%d').date()
                    dte = (expiry_date - today_obj).days
                    is_front = 'NEXT' not in tag
                    
                    # 解决周一同步上周五合约导致 DTE 太短的问题 (如 4 天)
                    if is_front and dte < 5:
                        logger.warning(f"⚠️ 拒绝恢复周末遗留合约 (DTE={dte} < 5): {sym} {tag}")
                        continue
                except Exception as e:
                    logger.warning(f"⚠️ 解析 expiry 失败: {expiry} ({e})")
                
                contract = Option(
                    symbol=sym, 
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(strike),
                    right=right,
                    exchange='SMART',
                    currency='USD',
                    multiplier=str(multiplier),
                    localSymbol=localSymbol,
                    tradingClass=tradingClass
                )
                contract.conId = int(conId) 
                
                if sym not in self.locked_contracts: self.locked_contracts[sym] = {}
                self.locked_contracts[sym][tag] = contract
                cnt += 1
            
            self.current_lock_date = today_obj
            logger.info(f"♻️ Restored {cnt} valid locks from PostgreSQL.")
            return True

        except Exception as e:
            logger.error(f"❌ Load locks db failed: {e}")
            return False

     

    def _init_ib_connection(self):
        self.ib = IB()
        self.ib.errorEvent += self._on_error
        
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"Connected to IBKR ({self.host}:{self.port}, ID={self.client_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect IBKR: {e}")
            return False

    def _on_error(self, reqId, errorCode, errorString, contract):
        """
        [新增] 自定义错误处理，自动解码 Unicode
        """
        try:
            # 尝试解码 (比如 \u5173 -> 关)
            decoded_msg = errorString.encode('utf-8').decode('unicode_escape')
        except:
            decoded_msg = errorString
            
        # 忽略部分良性错误
        # 2103/2105/2108: Market Data Farm Connection Broken (will auto restore)
        # 2104/2106: Market Data Farm Connection OK
        # 2158: Sec-def Data Farm OK
        # 2107: HMDS data farm inactive
        if errorCode in [2103, 2104, 2105, 2106, 2107, 2108, 2158]: return 
        
        # [NEW] 针对 10197 (Competing Session) 提供明确提示 & 自动切换延迟数据
        if errorCode == 10197:
            logger.warning(f"⛔ IBKR DATA PAUSED (10197): Real account is active elsewhere.")
            logger.warning(f"   Msg: {decoded_msg}")
            
            # [Auto-Fix] 自动切换到延迟数据 (Type 3) 以绕过锁
            logger.warning(f"⚠️ AUTOMATICALLY SWITCHING TO DELAYED DATA (Type 3) TO BYPASS LOCK.")
            self.ib.reqMarketDataType(3) 
            logger.warning(f"✅ Delayed Data Activated. You are now receiving 15-20min old data.")
            return

        logger.error(f"⚠️ IBKR Error {errorCode} (reqId {reqId}): {decoded_msg}")

    # ================= 订阅逻辑 =================
    async def start_stock_stream(self, symbols: List[str]):
        """启动正股行情流，并并发初始单股任务"""
        try:
            # 1. 加载持久化锁 (只需一次)
            self._load_locks()
            
            # 2. 并发启动每个 Symbol 的初始化 (各跑各的，互不等待)
            logger.info(f"🚀 Starting concurrent initialization for {len(symbols)} symbols...")
            tasks = [self._process_single_symbol(sym) for sym in symbols]
            
            # 让它们后台跑，不要用 await gather 阻塞这里，
            # 这样主线程可以立刻完成 start_stock_stream，
            # Dashboard 也能更快看到 partial updates
            for t in tasks:
                 asyncio.create_task(t)

            # 绑定回调
            self.ib.barUpdateEvent += self._on_stock_bar
            logger.info(f"✅ Stock Stream Launcher Finished (Background tasks running).")

        except Exception as e:
            logger.error(f"Start Stock Stream Failed: {e}")
        finally:
            # 这里的 flag 意义变为: "Launcher跑完了"，具体的 symbol 可能还在初始化
            # maintenance_loop 可以配合 check active_stocks 使用
            self.initial_scan_done = True 
            logger.info("🚦 Initial Scan Flag Set to True.")

    async def _process_single_symbol(self, sym: str):
        """[New] 单个标的的全流程初始化: 获取合约 -> 订阅正股 -> 搜期权 -> 订阅期权 -> 保存"""
        try:
            # A. 获取正股合约详情
            # 自动修复 BRK.B
            ibkr_sym = sym.replace('.', ' ') if 'BRK' in sym else sym
            tmp_stock = Stock(ibkr_sym, 'SMART', 'USD')
            
            details = await self.ib.reqContractDetailsAsync(tmp_stock)
            if not details:
                logger.warning(f"⚠️ No contract details found for {sym}")
                return
            
            contract = details[0].contract
            original_key = contract.symbol.replace(' ', '.') if 'BRK' in contract.symbol else contract.symbol
            
            # B. 存入内存 & 订阅正股
            self.active_stocks[original_key] = contract
            
            # 保留 5 秒 Bar 作为 OHLCV / fallback 数据源，但不再作为低延迟发布的唯一触发器。
            self.ib.reqRealTimeBars(contract, 5, 'TRADES', False)
            # 订阅实时行情，用于 tick 驱动的低延迟发布与价格更新。加入 100(Volume), 233(RTVolume) 保证成交量推送
            self.ib.reqMktData(contract, '100,233,236', False, False)
            # logger.info(f"✅ Subscribed Stock: {original_key}")

            # C. 恢复/搜索 期权 (Immediate Check)
            # c1. 如果有缓存锁，恢复订阅
            if original_key in self.locked_contracts:
                for tag, opt_c in self.locked_contracts[original_key].items():
                    self.ib.reqMktData(opt_c, '100,101,106', False, False)
                    # logger.info(f"   ♻️ Restored Option: {opt_c.localSymbol}")
            
            # c2. 如果不是指数，且需要搜索 (锁缺失/无效)
            is_index = original_key in NO_OPTION_LOCK_SYMBOLS
            if not is_index:
                # 检查是否缺锁
                missing_tags = [tag for tag in TAG_TO_INDEX if tag not in self.locked_contracts.get(original_key, {})]
                if missing_tags:
                    # 获取现价用于搜索
                    tickers = await self.ib.reqTickersAsync(contract)
                    spot = 0
                    if tickers:
                         t = tickers[0]
                         spot = t.last if t.last else (t.close if t.close else (t.bid + t.ask)/2)
                    
                    if spot > 0:
                        logger.info(f"🔍 [Instant] Searching missing contracts for {original_key} @ {spot:.2f}...")
                        new_locks = await self._find_contracts(original_key, spot)
                        if new_locks:
                            if original_key not in self.locked_contracts: self.locked_contracts[original_key] = {}
                            
                            for tag, c in new_locks.items():
                                # 存入内存
                                self.locked_contracts[original_key][tag] = c
                                # 订阅 (106: IV/Greeks)
                                self.ib.reqMktData(c, '100,101,106', False, False)
                                # 持久化
                                self._save_single_lock(original_key, tag, c)
                                logger.info(f"   ➕ Subscribed & Saved: {c.localSymbol}")

        except Exception as e:
            logger.error(f"❌ Init failed for {sym}: {e}")

    def _on_stock_bar(self, bars, hasNew):
        """收到正股 5秒 Bar -> [Fallback] 兜底发布"""
        if not hasNew: return
        
        # 1. 记录最新的正股价格与存活状态
        b = bars[-1]
        contract_sym = bars.contract.symbol
        sym = contract_sym.replace(' ', '.') if 'BRK' in contract_sym else contract_sym
        self.last_spot_prices[sym] = b.close
        self.last_tick_time[sym] = time.time()
        
        # 🚀 [核心修复 1] 如果开启了 Tick 驱动，绝对禁止 5秒 Bar 覆写缓存！
        # 否则会破坏 _on_pending_tickers 精心维护的秒级 Volume Delta 累加机制！
        if not self._use_tick_driven_publish:
            # 缓存当前的 Bar 数据到内存，供 Sweep 使用
            if not hasattr(self, '_last_bar_cache'): self._last_bar_cache = {}
            self._last_bar_cache[sym] = {
                'open': b.open_, 'high': b.high, 'low': b.low, 'close': b.close, 'volume': b.volume,
                'ts': b.time.timestamp()
            }

            # 2. [Fallback] 若未开启 tick 驱动，则仍使用 5 秒 Bar 触发整帧发布。
            ts_val = b.time.timestamp()
            if not hasattr(self, '_last_published_batch_ts'): self._last_published_batch_ts = 0
            if ts_val > self._last_published_batch_ts:
                if self._collecting_ts != ts_val:
                    self._collecting_ts = ts_val
                    asyncio.create_task(self._debounced_publish(ts_val))

    @staticmethod
    def _safe_float(val):
        return float(val) if (val is not None and not np.isnan(val)) else 0.0

    @staticmethod
    def _to_symbol(contract_sym: str) -> str:
        return contract_sym.replace(' ', '.') if 'BRK' in contract_sym else contract_sym

    def _resolve_stock_price_from_ticker(self, ticker) -> float:
        last_p = self._safe_float(getattr(ticker, 'last', None))
        if last_p > 0:
            return last_p

        market_p = self._safe_float(ticker.marketPrice()) if hasattr(ticker, 'marketPrice') else 0.0
        if market_p > 0:
            return market_p

        close_p = self._safe_float(getattr(ticker, 'close', None))
        if close_p > 0:
            return close_p

        bid_p = self._safe_float(getattr(ticker, 'bid', None))
        ask_p = self._safe_float(getattr(ticker, 'ask', None))
        if bid_p > 0 and ask_p > 0:
            return (bid_p + ask_p) / 2.0
        return 0.0

    def _on_pending_tickers(self, tickers):
        """使用 reqMktData 驱动 1 秒级本地聚合，避免整条链被 5 秒 Bar 卡住。"""
        if not self._use_tick_driven_publish:
            return

        now_ts = time.time()
        second_ts = float(int(now_ts))
        touched = False

        for ticker in tickers:
            try:
                contract = getattr(ticker, 'contract', None)
                if contract is None or getattr(contract, 'secType', '') != 'STK':
                    continue

                sym = self._to_symbol(contract.symbol)
                if sym not in self.active_stocks:
                    continue

                price = self._resolve_stock_price_from_ticker(ticker)
                if price <= 0:
                    continue

                total_vol = self._safe_float(getattr(ticker, 'volume', None))
                prev_total = self._last_total_volume.get(sym)
                delta_vol = 0.0
                if total_vol > 0:
                    if prev_total is not None and total_vol >= prev_total:
                        delta_vol = total_vol - prev_total
                    self._last_total_volume[sym] = total_vol
                else:
                    # 如果 reqMktData 没有 volume (例如盘前)，我们尝试提取 RTVolume
                    rt_vol = getattr(ticker, 'rtVolume', None)
                    if rt_vol is not None and rt_vol > 0:
                        prev_rt = self._last_total_volume.get(sym)
                        if prev_rt is not None and rt_vol >= prev_rt:
                            delta_vol = rt_vol - prev_rt
                        self._last_total_volume[sym] = rt_vol

                bar = self._last_bar_cache.get(sym)
                current_vol = float(bar.get('volume', 0.0)) if bar else 0.0
                
                if bar is None or int(bar.get('ts', 0)) != int(second_ts):
                    self._last_bar_cache[sym] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': current_vol + max(0.0, delta_vol),
                        'ts': second_ts,
                    }
                else:
                    bar['high'] = max(float(bar['high']), price)
                    bar['low'] = min(float(bar['low']), price)
                    bar['close'] = price
                    bar['volume'] = current_vol + max(0.0, delta_vol)

                self.last_spot_prices[sym] = price
                self.last_tick_time[sym] = now_ts
                touched = True
            except Exception:
                continue

        if not touched:
            return

        if not hasattr(self, '_last_published_batch_ts'):
            self._last_published_batch_ts = 0

        if second_ts > self._last_published_batch_ts and self._collecting_ts != second_ts:
            self._collecting_ts = second_ts
            asyncio.create_task(self._debounced_publish(second_ts))

    async def _debounced_publish(self, ts_val):
        """[🔥 NEW] 聚合窗口延迟触发器"""
        await asyncio.sleep(self._publish_debounce_sec)
        if ts_val > self._last_published_batch_ts:
            self._last_published_batch_ts = ts_val
            self._publish_batch_snapshot(ts_val)

    def _publish_batch_snapshot(self, ts_val):
        """[🔥 Optimization] 全市场快照聚合发布"""
        batch_payloads = []
        
        # 扫描所有活跃标的 (含 SPY/QQQ 等无期权标的)
        for sym in self.active_stocks:
            # 获取该标的最近一次的本地聚合 Bar 缓存
            bar_data = self._last_bar_cache.get(sym)
            if not bar_data: continue
            
            # 如果缓存的数据太陈旧 (超过 10 秒)，说明该 symbol 已经由于某种原因断流了
            if ts_val - bar_data['ts'] > 10.0:
                continue
                
            # 组装 Payload (对齐 feature_compute_service_v8 的接口)
            payload = {
                'symbol': sym,
                'ts': ts_val,
                'stock': {
                    'open': bar_data['open'], 'high': bar_data['high'],
                    'low': bar_data['low'], 'close': bar_data['close'],
                    'volume': bar_data['volume']
                }
            }
            
            # 🚀 [核心修复: 秒级成交量隔离] 
            # 把当前这 1 秒内的累加成交量发送给下游后，必须在本地清零！
            # 否则下游 persistence service 在累加分钟级 VWAP (pv_sum) 时会发生指数级重复计算。
            bar_data['volume'] = 0.0
            
            # 收集期权数据 (如果有锁定的话)
            opt_buckets, opt_contracts = self._collect_option_buckets(sym)
            if opt_buckets:
                payload['option_buckets'] = opt_buckets
                payload['option_contracts'] = opt_contracts
                
            batch_payloads.append(payload)
            
        if batch_payloads:
            try:
                # [优化] 使用统一序列化工具 (Msgpack)
                self.redis.xadd(STREAM_KEY_FUSED, {'batch': ser.pack(batch_payloads)}, maxlen=100000)
                # logger.info(f"🚀 [Soft Batch] Published synchronized frame for {len(batch_payloads)} symbols @ {ts_val}")
            except Exception as e:
                logger.error(f"❌ Failed to publish soft batch: {e}")

    def _collect_option_buckets(self, sym):
        """收集 6 个期权桶的实时快照 & 合约ID。
        约定：IB 侧只推送价格/盘口，不推送 Greeks/IV；Greeks 统一由下游引擎按分钟计算。
        """
        buckets_data = np.zeros((6, 12), dtype=float).tolist()
        contracts_data = [""] * 6 
        locks = self.locked_contracts.get(sym)
        if not locks: return buckets_data, contracts_data
            
        for tag, contract in locks.items():
            idx = TAG_TO_INDEX.get(tag)
            if idx is None: continue
            
            t = self.ib.ticker(contract)
            if not t: continue
            
            # =============================================================
            # [🔥 防弹修复: 严格过滤 NaN，应对重启数据静默期]
            # =============================================================
            def _safe_float(val):
                """安全转换，防止 nan 或 None 污染下游大脑"""
                return float(val) if (val is not None and not np.isnan(val)) else 0.0

            last_p = _safe_float(t.last)
            close_p = _safe_float(t.close)
            bid_p = _safe_float(t.bid)
            ask_p = _safe_float(t.ask)
            bid_size = _safe_float(t.bidSize) # [新增] 提取实盘 Bid 挂单量
            ask_size = _safe_float(t.askSize) # [新增] 提取实盘 Ask 挂单量

            if last_p > 0: price = last_p
            elif close_p > 0: price = close_p
            elif bid_p > 0 and ask_p > 0: price = (bid_p + ask_p) / 2.0
            else: price = 0.0 # 保持 0.0，Orchestrator 识别到 0 会用 BSM 自动兜底！
            
            # Greeks/IV 在 IB 侧固定置零；统一由 realtime_feature_engine 在分钟级计算。
            iv, delta, gamma, vega, theta = 0.0, 0.0, 0.0, 0.0, 0.0

            # 🚀 [核心对齐] 用盘口深度代替真实成交量 (与 step2_thetadata_sniper_v6 训练脚本一致)
            bid_size = _safe_float(t.bidSize)
            ask_size = _safe_float(t.askSize)
            vol = bid_size + ask_size 
            
            # [🔥 修改] 将 Size 追加到列表末尾 (作为索引 10 和 11)
            buckets_data[idx] = [price, delta, gamma, vega, theta, contract.strike, vol, iv, bid_p, ask_p, bid_size, ask_size]
            contracts_data[idx] = contract.localSymbol 

        return buckets_data, contracts_data
    
    async def _find_contracts(self, sym, spot):
        self.ib.reqMarketDataType(1) # Live Data
        found_contracts = {}
        
        # =========================================================
        # [🔥 核心升级 V2] 像素级对齐离线 options_locked_feature.py
        # =========================================================
        ANCHOR_CONFIG = {
            # --- Front (近月) ---
            'FRONT_TARGET_DTE': 9,        # 离线逻辑: 理想锚点
            'FRONT_MIN_DTE': 5,           
            'FRONT_MAX_DTE': 16,          
            
            # --- Next (次月) ---
            'NEXT_MIN_GAP': 20,           # 离 Front 至少 20 天
            'NEXT_MAX_GAP': 50,           # 离 Front 最多 50 天
            
            'ATM_CENTER': 0.50,
            'OTM_CENTER': 0.25 
        }

        # 1. 获取所有过期日列表
        try:
            if sym not in self.active_stocks:
                contract = Stock(sym, 'SMART', 'USD')
                await self.ib.qualifyContractsAsync(contract)
            else:
                contract = self.active_stocks[sym]
            
            chains = await self.ib.reqSecDefOptParamsAsync(contract.symbol, '', 'STK', contract.conId if hasattr(contract, 'conId') else 0)
            
            all_expirations = set()
            for c in chains:
                # [🔥 核心修复 1] 不再限制 c.exchange == 'SMART'。
                # IBKR 经常把完整的期权链挂在其他底层交易所下，这里直接聚合所有日期防止漏掉长线/周权。
                all_expirations.update(c.expirations)
            all_expirations = sorted(list(all_expirations))
            
            if not all_expirations:
                logger.warning(f"❌ No option expirations found for {sym}")
                return {}
        except Exception as e:
            logger.error(f"❌ Failed to get option params for {sym}: {e}")
            return {}

        # 2. 精确选择 Expiration
        today_ny = datetime.datetime.now(NY_TZ).date()
        
        # 构建 (date_str, dte) 列表
        exp_dtes = []
        for exp_str in all_expirations:
            try:
                exp_date = datetime.datetime.strptime(exp_str, '%Y%m%d').date()
                dte = (exp_date - today_ny).days
                if dte >= 2: # 过滤末日轮
                    exp_dtes.append((exp_str, dte))
            except: continue
            
        if not exp_dtes: return {}

        # --- A. 选 Front (Target 9) ---
        front_candidates = [x for x in exp_dtes if ANCHOR_CONFIG['FRONT_MIN_DTE'] <= x[1] <= ANCHOR_CONFIG['FRONT_MAX_DTE']]
        
        selected_front = None
        if front_candidates:
            selected_front = min(front_candidates, key=lambda x: abs(x[1] - ANCHOR_CONFIG['FRONT_TARGET_DTE']))
        else:
            # 兜底: 找最小的可用 DTE
            selected_front = min(exp_dtes, key=lambda x: x[1])
            
        target_exps_front = [selected_front[0]]
        actual_front_dte = selected_front[1]

        # --- B. 选 Next (Target = Front + 28) ---
        next_target_dte = actual_front_dte + 28
        min_next = actual_front_dte + ANCHOR_CONFIG['NEXT_MIN_GAP']
        max_next = actual_front_dte + ANCHOR_CONFIG['NEXT_MAX_GAP']
        
        next_candidates = [x for x in exp_dtes if min_next <= x[1] <= max_next]
        
        selected_next = None
        target_exps_next = [] # 默认空，找不到就不找了
        if next_candidates:
            selected_next = min(next_candidates, key=lambda x: abs(x[1] - next_target_dte))
            target_exps_next = [selected_next[0]]
        else:
            fallbacks = [x for x in exp_dtes if x[1] > actual_front_dte]
            if fallbacks:
                selected_next = min(fallbacks, key=lambda x: abs(x[1] - next_target_dte))
                target_exps_next = [selected_next[0]]
            else:
                # =========================================================
                # [🔥 防御 3：宁缺毋滥] API没数据时，绝不降级复用 Front！
                # =========================================================
                logger.warning(f"❌ API 无法找到 {sym} 大于 {actual_front_dte} 天的期权，放弃 NEXT 合约订阅。")
                target_exps_next = [] # 保持空列表，下游就不会去搜索 NEXT 了

        # [缓存辅助]
        details_cache = {}
        async def get_real_contracts(expiry, right):
            if (expiry, right) in details_cache: return details_cache[(expiry, right)]
            temp_c = Option(contract.symbol, expiry, right=right, exchange='SMART')
            try:
                details = await self.ib.reqContractDetailsAsync(temp_c)
                contracts = [d.contract for d in details]
                details_cache[(expiry, right)] = contracts
                return contracts
            except Exception: return []

        # 3. 扫描各 Bucket
        for tag, spec in sorted(BUCKET_SPECS.items(), key=lambda x: x[1]['bucket_idx']):
            is_front = 'NEXT' not in tag
            # 根据 tag 决定是用 Front 还是 Next 的到期日
            target_exps = target_exps_front if is_front else target_exps_next
            
            right = 'C' if 'CALL' in tag else 'P'
            target_delta = ANCHOR_CONFIG['ATM_CENTER'] if 'ATM' in tag else ANCHOR_CONFIG['OTM_CENTER']
            
            all_candidates = []
            for exp in target_exps:
                cands = await get_real_contracts(exp, right)
                all_candidates.extend(cands)
            if not all_candidates: continue

            # 锁定主力 TradingClass
            tc_counts = {}
            for c in all_candidates:
                tc = c.tradingClass
                tc_counts[tc] = tc_counts.get(tc, 0) + 1
            if not tc_counts: continue
            main_tc = max(tc_counts, key=tc_counts.get)
            valid_candidates = [c for c in all_candidates if c.tradingClass == main_tc]

            ideal_strike = spot if target_delta == 0.50 else (spot * 1.04 if right == 'C' else spot * 0.96)
            valid_candidates.sort(key=lambda c: abs(c.strike - ideal_strike))
            top_candidates = valid_candidates[:12]

            # 4. 批量并发获取 Snapshot
            tasks = [self._get_snapshot_with_oi(c) for c in top_candidates]
            snapshots = await asyncio.gather(*tasks)
            
            data_list = []
            for ticker, oi in snapshots:
                if ticker.modelGreeks and ticker.modelGreeks.delta is not None: 
                    delta = ticker.modelGreeks.delta
                else:
                    approx_d = 0.5 + (spot - ticker.contract.strike) / (spot * 0.1)
                    if right == 'P': approx_d -= 1
                    delta = max(-1.0, min(1.0, approx_d))
                
                delta_abs = abs(delta)
                delta_err = abs(delta_abs - target_delta) 
                vol = ticker.volume if ticker.volume else 0
                
                data_list.append({
                    'contract': ticker.contract,
                    'strike': ticker.contract.strike,
                    'delta_abs': delta_abs,
                    'delta_err': delta_err,
                    'volume': vol,
                    'oi': oi
                })
            
            df = pd.DataFrame(data_list)
            if df.empty: continue
            
            df_good_delta = df[df['delta_err'] < 0.15].copy()
            
            if not df_good_delta.empty:
                df_sorted = df_good_delta.sort_values(by=['volume', 'oi', 'delta_err'], ascending=[False, False, True])
            else:
                df_sorted = df.sort_values(by=['delta_err'], ascending=[True])
            
            best_row = df_sorted.iloc[0]
            best_contract = best_row['contract']
            
            found_contracts[tag] = best_contract
            logger.info(f"   🔒 Locked {tag}: {best_contract.localSymbol} (K:{best_contract.strike}, Delta:{best_row['delta_abs']:.2f}, Vol:{best_row['volume']:.0f}, Exp:{best_contract.lastTradeDateOrContractMonth})")

        return found_contracts

    async def _get_snapshot_with_oi(self, contract):
        """[V8 终极版] 辅助函数: 获取包含 OI、成交量和 Greeks 的单次快照"""
        # 100=Volume, 101=OpenInterest, 106=ImpliedVol/Greeks
        self.ib.reqMktData(contract, '100,101,106', False, False) 
        
        start = time.time()
        ticker = self.ib.ticker(contract)
        
        timeout_limit = 2.0
        # 等待 OI 和 Greeks。如果拿到 OI 后 0.2 秒内没有 Greeks（盘前常态），立即放弃死等，极大提速并保证流动性选取！
        while time.time() - start < timeout_limit:
            has_oi = (ticker.callOpenInterest is not None or ticker.putOpenInterest is not None)
            has_greeks = (ticker.modelGreeks is not None and ticker.modelGreeks.delta is not None)
            
            if has_greeks:
                break
            elif has_oi:
                # 已经确保了 OI (Open Interest) 流动性指标，压缩超时上限，不再苦等盘前不可能出现的 Greeks
                timeout_limit = min(timeout_limit, (time.time() - start) + 0.2)
                
            await asyncio.sleep(0.05)
        
        oi = 0
        if contract.right == 'C': oi = ticker.callOpenInterest
        else: oi = ticker.putOpenInterest
        if oi is None: oi = 0
        
        self.ib.cancelMktData(contract)
        return ticker, oi

    async def _find_contracts_from_pg(self, sym):
        """
        [EOD 静态锁] 直接从 PostgreSQL 查询最新的一条期权快照记录。
        这会自动跨越所有日期分区，速度极快。
        """
        import json
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            today_start = datetime.datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            c.execute("SELECT buckets_json, ts FROM option_snapshots_1m WHERE symbol=%s AND ts >= %s ORDER BY ts DESC LIMIT 1", (sym, today_start))
            row = c.fetchone()
            conn.close()
            
            if row:
                opt_data, snap_ts = row # JSONB is automatically converted to dict in psycopg2
                if isinstance(opt_data, str):
                    opt_data = json.loads(opt_data)
                contract_ids = opt_data.get('contracts', [])
                if len(contract_ids) == 6 and any(contract_ids):
                    logger.info(f"📚 Found historical EOD option lock for {sym} in Postgres")
                    found_contracts = {}
                    
                    for tag, spec in BUCKET_SPECS.items():
                        idx = spec['bucket_idx']
                        con_id = contract_ids[idx]
                        if not con_id: continue
                        
                        clean_id = con_id.replace('O:', '')
                        import re
                        match = re.match(r'^([A-Z]{1,6})(\d{6}[CP]\d{8})$', clean_id)
                        if match:
                            sym_root, rest = match.groups()
                            clean_id = sym_root.ljust(6, ' ') + rest
                            
                        c = Option(symbol=sym, exchange='SMART', currency='USD')
                        c.localSymbol = clean_id
                        found_contracts[tag] = c
                        
                    qualify_tasks = [self.ib.qualifyContractsAsync(c) for c in found_contracts.values()]
                    await asyncio.gather(*qualify_tasks, return_exceptions=True)
                    
                    # =========================================================
                    # [🔥 防御 2：拦截 PG 历史数据中毒] 
                    # =========================================================
                    if 'CALL_ATM' in found_contracts and 'NEXT_CALL_ATM' in found_contracts:
                        if found_contracts['CALL_ATM'].localSymbol == found_contracts['NEXT_CALL_ATM'].localSymbol:
                            logger.warning(f"⚠️ 数据库 EOD 缓存近远月发生重叠，拒绝使用，将转入 API 在线搜索。")
                            return {} # 识别到脏数据，返回空，让它去走实盘搜索

                    return found_contracts
                    
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load static lock from Postgres for {sym}: {e}")
            return {}

    # ================= 维护循环 =================
    async def maintenance_loop(self, interval=60):
        # [Fix] 等待初始扫描完成，避免重复搜索
        while not self.initial_scan_done:
            await asyncio.sleep(1)

        while True:
            # 1. 每日重置
            today = datetime.datetime.now(NY_TZ).date()
            if str(today) != str(self.current_lock_date):
                logger.info(f"📅 New Day Detected ({today}). Clearing locks in memory...")
                self.locked_contracts = {}
                self.current_lock_date = str(today)
                # [注] 因为采用了每日独立 DB 架构，这里无需再执行 DELETE 清理旧库，自然过渡。

            # 2. 扫描缺失的锁
            changes_made = False
            # [Fix] 遍历所有已订阅的正股，不再依赖 _on_stock_bar 的推送
            # 这样即使没有 RealTimeBar，只要有 Snapshot 价格也能触发选合约
            for sym, contract in list(self.active_stocks.items()):
                if sym in NO_OPTION_LOCK_SYMBOLS: continue
                
                # 获取当前价格 (优先用 MarketPrice，含盘前)
                ticker = self.ib.ticker(contract)
                spot = ticker.marketPrice()
                
                # 如果 marketPrice 无效 (nan)，尝试用 last 或 close
                if np.isnan(spot) or spot <= 0:
                    if ticker.last and ticker.last > 0: spot = ticker.last
                    elif ticker.close and ticker.close > 0: spot = ticker.close
                    # 如果仍然无效，跳过
                    else: continue
                
                if sym not in self.locked_contracts: self.locked_contracts[sym] = {}
                
                # =========================================================
                # [🔥 防御 1：撕毁内存毒缓存] 发现近远月同一天，立刻清空强搜！
                # =========================================================
                locks = self.locked_contracts[sym]
                if 'CALL_ATM' in locks and 'NEXT_CALL_ATM' in locks:
                    if locks['CALL_ATM'].lastTradeDateOrContractMonth == locks['NEXT_CALL_ATM'].lastTradeDateOrContractMonth:
                        logger.warning(f"🚨 检测到 {sym} 缓存锁中毒 (近月远月挤在同一天)，撕毁缓存，强制重新搜索！")
                        self.locked_contracts[sym] = {} # 清空，强迫重搜
                
                missing = [tag for tag in TAG_TO_INDEX if tag not in self.locked_contracts[sym]]
                if missing:
                    logger.info(f"🔍 Scanning contracts for {sym} at Spot {spot:.2f}...")
                    try:
                        # 1. 首选方案: 极速离线数据库 EOD 锁定
                        new_locks = await self._find_contracts_from_pg(sym)
                        
                        # 2. 兜底方案: 如果昨日没有数据库遗留，采用在线 API 探索盲锁
                        if not new_locks:
                            logger.info(f"🌐 No EOD cache for {sym}, falling back to live API search...")
                            new_locks = await self._find_contracts(sym, spot)
                            
                        for tag, c in new_locks.items():
                            if tag not in self.locked_contracts[sym]:
                                await self.ib.qualifyContractsAsync(c)
                                self.locked_contracts[sym][tag] = c
                                # [Fix] Error 321 & 10091: 移除 104/106/13，仅保留最基础的 100(Vol)/101(OI) -> [Reversed] 必须加回 106，否则下游全部断流 (IV=0)
                                self.ib.reqMktData(c, '100,101,106', False, False)
                                # [New] 立即持久化，防止中断丢失
                                self._save_single_lock(sym, tag, c)
                                changes_made = True
                    except Exception as e:
                        logger.error(f"Find contracts failed for {sym}: {e}")
            
            # [Fix] 移除批量保存，改为增量保存
            # if changes_made:
            #     self._save_locks()

            # 4. [New] Heartbeat (主动心跳)
            try:
                # 尝试发送请求以探测连接
                await self.ib.reqCurrentTimeAsync()
                logger.info("❤️ IBKR Heartbeat OK")
            except Exception as e:
                logger.error(f"💔 Heartbeat Failed: {e}. Reconnecting...")
                # 断开旧连接，重新连接
                try: self.ib.disconnect()
                except: pass
                await self.connect()
                
                # [Fix] 重连后必须重新订阅数据流！
                if self.active_stocks:
                    logger.info("⚡ Re-subscribing to market data after reconnect...")
                    # 提取当前活跃的 symbol 列表 (original keys)
                    symbols = list(self.active_stocks.keys())
                    await self.start_stock_stream(symbols)

            await asyncio.sleep(interval)

    # --- Hash 推送 (Dashboard用) ---
    async def data_stream_loop(self, interval=0.5):
        logger.info("📡 Data Stream Loop (Hash) Started.")
        last_log = 0
        last_acct_update = 0
        while True:
            # [NEW] 每5秒向 Redis 推送最新的账户资金总额
            if time.time() - last_acct_update > 5.0 and self.ib.isConnected():
                try:
                    vals = self.ib.accountValues()
                    net_liq, avail_funds = 0.0, 0.0
                    for v in vals:
                        if v.currency == 'USD' or v.currency == 'BASE':
                            if v.tag.startswith('NetLiquidation'): net_liq = max(net_liq, float(v.value))
                            elif v.tag == 'AvailableFunds': avail_funds = float(v.value)
                    if net_liq > 0 or avail_funds > 0:
                        info = {'ts': time.time(), 'net_liquidation': net_liq, 'available_funds': avail_funds}
                        self.redis.hset('live_account_info', 'balance', ser.pack(info))
                    last_acct_update = time.time()
                except Exception as e:
                    pass

            if self.locked_contracts:
                if time.time() - last_log > 60:
                     logger.info(f"📡 Data Stream Loop ALIVE. Processing {len(self.locked_contracts)} symbols.")
                     last_log = time.time()
                     
                pipe = self.redis.pipeline()
                ts_now = datetime.datetime.now().timestamp()
                has_data = False
                for sym in self.locked_contracts:
                    # =========================================================
                    # [核心强化 2B] 僵尸数据过滤防线 (Watchdog)
                    # =========================================================
                    last_alive = self.last_tick_time.get(sym, 0)
                    # 如果超过 15 秒没有收到底层行情的更新，判定为断流/盘后，停止发送虚假快照！
                    if time.time() - last_alive > 15.0:
                        continue

                    buckets, contracts = self._collect_option_buckets(sym)
                    msg = {'symbol': sym, 'buckets': buckets, 'contracts': contracts, 'ts': ts_now}
                    pipe.hset(HASH_KEY_SNAPSHOT, sym, ser.pack(msg))
                    has_data = True
                if has_data: 
                    try: pipe.execute()
                    except: pass
            await asyncio.sleep(interval)

    # --- 下单接口 ---
    # --- 下单接口 (增强版: 带云端硬止损) ---
    def place_option_order(self, contract, action, qty, order_type='MKT', lmt_price=0.0, stop_loss_pct=0.07, custom_time=None, reason="", stock_price=0.0):
        """
        下单并自动附加云端止损 (Server-Side Stop Loss) 
        :param stop_loss_pct: 止损百分比 (默认 0.05 = 5%, 与 V3 策略 STOP_LOSS 同步)。设为 0 则不带止损。
        """
 

        # ==========================================================
        # 🚨🚨🚨 核按钮：DRY RUN 模式强制开启 🚨🚨🚨
        # ==========================================================
        # [NEW] 彻底实现“一行代码开关”：只需在 config.py 开启 TRADING_ENABLED，此处不再硬编码阻拦
        if not TRADING_ENABLED:
            logger.warning(f"🛑 [DRY RUN 空跑拦截] config.TRADING_ENABLED=False: {action} {qty}手 {contract.localSymbol} @ {order_type} | 限价: {lmt_price}")
            return None  # 强行返回，绝对不向交易所发任何单！
        
        # [Safety Check] 全局交易开关
        if not TRADING_ENABLED:
            logger.warning(f"🚫 TRADING DISABLED in config. Order BLOCKED: {action} {qty} {contract.localSymbol}")
            return

        # 1. 构建主订单 (Parent)
        parent = Order()
        parent.action = action
        parent.totalQuantity = qty
        parent.orderType = order_type
        
        # 强制设置 SMART, 避免 Ambiguous
        if not contract.exchange: contract.exchange = 'SMART'

        # 算法单设置
        # 算法单设置
        if order_type == 'LMT': 
            parent.lmtPrice = lmt_price
            parent.algoStrategy = 'Adaptive'
            parent.algoParams = [TagValue('adaptivePriority', 'Urgent')]  # <--- 使用 TagValue
        else:
            # 市价单通常不需要 Adaptive，或者用 Normal
            parent.algoStrategy = 'Adaptive'
            parent.algoParams = [TagValue('adaptivePriority', 'Normal')]  # <--- 使用 TagValue
            
        if ACCOUNT_ID: parent.account = ACCOUNT_ID

        # [Fix 1] 彻底移除 Service-Side STP 子订单逻辑
        # 防爆仓级修复：Orchestrator 已经在本地进行止损计算，如果在服务端额外挂载 STP 单，
        # 当期权巨幅波动时，服务端平仓和 Orchestrator 的 MKT SELL 指令会发生双重结算，
        # 导致原本应该归零的仓位被无中生有地打成 "-1" (Naked Short 裸卖空！亏损无上限)。
        # 因此，这里的附加止损单被果断移除，完全信任 Orchestrator 的软止损指令。
        
        logger.info(f"🛒 Order: {action} {qty} {contract.localSymbol} @ {order_type} (Managed by Orchestrator)")
        parent.transmit = True
        trade_parent = self.ib.placeOrder(contract, parent)
        return trade_parent


    def run(self, symbols):
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.connect())
            
            # [Fix] 这里的 start_stock_stream 如果耗时太久(搜合约)，会阻塞后续 task 的创建
            # 改为 create_task 异步执行，不阻塞启动流程
            loop.create_task(self.start_stock_stream(symbols))
            
            loop.create_task(self.maintenance_loop(20)) # [Fix] 缩短心跳间隔至 20s
            loop.create_task(self.data_stream_loop(0.5))
            logger.info("🚀 IBKR V8 Connector Running...")
            self.ib.run()
        except KeyboardInterrupt:
            self.ib.disconnect()

if __name__ == '__main__':
    # [Fix] 从 config.py 加载统一标的
    try:
        from config import TARGET_SYMBOLS
        print(f"🚀 Starting IBKR Connector for {len(TARGET_SYMBOLS)} symbols from config.")
        
        connector = IBKRConnectorFinal(client_id=102) # 默认使用 102 避免冲突，Orchestrator 会用 999
        connector.run(TARGET_SYMBOLS)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error: {e}")
#     SYMBOLS = [
#     # --- Tier 1: 绝对核心 (流动性与波动率的完美平衡) ---
#     'NVDA', 'TSLA', 'AMD', 'AAPL', 'META', 'AMZN', 'MSFT', 'GOOGL',
    
#     # --- Tier 2: 爆发力之王 (高 IV, 高 Gamma, 心跳玩的就是刺激) ---
#     'MSTR',  # 比特币影分身，波动率极大
#     'COIN',  # 币圈，爆发力强
#     'PLTR',  # 散户抱团，趋势性极好
#     'SMCI',  # 妖股，注意风控
#     'NFLX',  # 财报后经常有大波段
    
#     # --- Tier 3: 趋势动量 (SaaS/芯片/软件) ---
#     'AVGO',  # 只要你的本金够买一张它的期权
#     'CRM',   # 软件股代表
#     'ADBE',  # 波动适中
#     'MU',    # 存储芯片，周期性强
#     'UBER',  # 只有在它有明确趋势时才做
#     'APP',   # 近期动量强
# ]
