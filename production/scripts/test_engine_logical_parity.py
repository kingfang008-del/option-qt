#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import asyncio
import numpy as np
import unittest
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime

# --- Paths ---
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("production/baseline"))
sys.path.append(os.path.abspath("production/history_replay"))
sys.path.append(os.path.abspath("production/utils"))

# Define logger for diagnostics
logger = logging.getLogger("ParityTest")
logging.basicConfig(level=logging.INFO)

# --- Mocking minimal heavy dependencies ---
# !! MUST BE DONE BEFORE ANY OTHER IMPORTS !!
def mock_module(name):
    m = MagicMock()
    sys.modules[name] = m
    return m

mock_module("torch")
mock_module("psycopg2")
mock_module("psycopg2.extras")
mock_module("trading_tft_stock_embed")
mock_module("scipy")
mock_module("scipy.stats")
mock_module("scipy.signal")
mock_module("redis")

# Special handling for norm import
mock_stats = sys.modules["scipy.stats"]
mock_stats.norm = MagicMock()

# Special handling for redis.Redis
mock_redis_mod = sys.modules["redis"]
mock_redis_mod.Redis = MagicMock()

from production.utils import serialization_utils as ser
import config
from mock_ibkr_historical import MockIBKRHistorical
from system_orchestrator_v8 import V8Orchestrator
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8

# --- Redis Mock ---
class MockRedisBridge:
    def __init__(self, *args, **kwargs):
        self.message_queue = []
        self.kv_store = {"sync:orch_done": "0"}
        self.hash_store = {}
        
    def xadd(self, stream, doc, **kwargs):
        data = doc.get('data') or doc.get(b'data')
        if data:
            payload = ser.unpack(data)
            self.message_queue.append(payload)
            return "msg_id"

    def hgetall(self, name): 
        return {k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v 
                for k, v in self.hash_store.get(name, {}).items()}

    def hset(self, name, key=None, value=None, mapping=None):
        if name not in self.hash_store: self.hash_store[name] = {}
        if mapping:
            for k, v in mapping.items():
                self.hash_store[name][k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, bytes) else v
        else:
            self.hash_store[name][key.decode() if isinstance(key, bytes) else key] = value.decode() if isinstance(value, bytes) else value

    def delete(self, *names):
        for name in names:
            name_str = name.decode() if isinstance(name, bytes) else name
            if name_str in self.hash_store: del self.hash_store[name_str]
            if name_str in self.kv_store: del self.kv_store[name_str]

    def set(self, key, value):
        self.kv_store[key.decode() if isinstance(key, bytes) else key] = value.decode() if isinstance(value, bytes) else value

    def get(self, key):
        val = self.kv_store.get(key.decode() if isinstance(key, bytes) else key)
        return val.encode() if isinstance(val, str) else val
        
    def xinfo_groups(self, stream): return []
    def xreadgroup(self, group, consumer, streams, count=50, block=100): return [] 
    def pipeline(self): return self
    def execute(self): pass
    def xack(self, stream, group, msg_id): pass
    def xgroup_create(self, stream, group, **kwargs): pass
    def xgroup_destroy(self, stream, group): pass

# Ensure redis.Redis returns our MockRedisBridge instances
mock_redis_mod.Redis.side_effect = lambda *args, **kwargs: MockRedisBridge(*args, **kwargs)

# --- Diagnostics ---
import system_orchestrator_v8
import signal_engine_v8

def patch_symbol_state(mod):
    if not hasattr(mod, 'SymbolState'): return
    original_update = mod.SymbolState.update_indicators
    def patched_update(self, price, raw_alpha, is_new_minute=True):
        res = original_update(self, price, raw_alpha, is_new_minute)
        self.warmup_complete = True # FORCE IT
        return res
    mod.SymbolState.update_indicators = patched_update

patch_symbol_state(system_orchestrator_v8)
patch_symbol_state(signal_engine_v8)

from strategy_core_v1 import StrategyCoreV1
original_decide_entry = StrategyCoreV1.decide_entry

def diagnostic_decide_entry(self, ctx):
    res = original_decide_entry(self, ctx)
    if ctx.get('alpha_z', 0) > 1.0:
        is_ready = ctx.get('is_ready', False)
        is_banned = ctx.get('is_banned', False)
        pos_zero = ctx['position'] == 0
        cooldown = ctx['curr_ts'] >= ctx.get('cooldown_until', 0)
        t = (datetime.fromtimestamp(ctx['curr_ts']) if 'curr_ts' in ctx else datetime.now())
        
        # Use logger instead of print
        logger.info(f"DIAG: {ctx['symbol']} AlphaZ={ctx['alpha_z']:.2f} Pre:{is_ready}/{not is_banned}/{pos_zero}/{cooldown}")
    return res

StrategyCoreV1.decide_entry = diagnostic_decide_entry

# --- Test Case ---
class TestEngineLogicalParity(unittest.TestCase):
    def setUp(self):
        config.RUN_MODE = 'BACKTEST'
        config.IS_SIMULATED = True
        config.IS_BACKTEST = True
        config.INITIAL_ACCOUNT = 50000.0
        
        # Increased to 11 symbols to pass cross-sectional filter
        self.symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD', 'GOOGL', 'AMZN', 'META', 'NFLX', 'INTC', 'PYPL']
        self.ts_base = 1710424800.0 # 2024-03-14 10:00:00 NY
        
    def get_market_sequence(self):
        """Generate 10 minutes of data with momentum to trigger trades"""
        seq = []
        for m in range(10):
            ts = self.ts_base + m * 60
            n = len(self.symbols)
            # Make AAPL have very high alpha
            alphas = np.zeros(n)
            alphas[0] = 5.0 # AAPL
            
            # Increasing price to trigger momentum guards
            price = 150.0 + (m * 0.2)
            
            # [NEW] Pre-populate metrics that the Strategy looks for
            batch = {
                'symbols': self.symbols,
                'ts': ts,
                'stock_price': np.array([price] * n),
                'feed_call_price': np.array([3.0] * n),
                'feed_call_bid': np.array([2.9] * n),
                'feed_call_ask': np.array([3.1] * n),
                'feed_call_id': [f'{s}_C' for s in self.symbols],
                'feed_call_k': np.array([150.0] * n),
                'feed_call_iv': np.array([0.3] * n),
                'feed_put_price': np.array([3.0] * n),
                'feed_put_bid': np.array([2.9] * n),
                'feed_put_ask': np.array([3.1] * n),
                'feed_put_id': [f'{s}_P' for s in self.symbols],
                'feed_put_k': np.array([150.0] * n),
                'feed_put_iv': np.array([0.3] * n),
                'precalc_alpha': alphas,
                'fast_vol': np.array([1.0] * n),
                'is_new_minute': True,
                'volume': np.array([10000.0] * n),
                
                # [NEW] Strategy overrides (in case engine fails to calculate them)
                'alpha_mean': 0.0,
                'alpha_std': 1.0,
                'vol_mean': 0.0,
                'vol_std': 1.0,
                'stock_roc_5min': np.array([0.005] * n), 
                'snap_roc': np.array([0.001] * n),        
                'macd_hist': np.array([0.1] * n),         
                'index_trend': 1,                         
                'spy_roc_5min': np.array([0.002] * n),    
                'qqq_roc_5min': np.array([0.002] * n)     
            }
            seq.append(batch)
        return seq

    async def run_unified_mode(self, seq):
        mock_ibkr = MockIBKRHistorical()
        orch = V8Orchestrator(self.symbols, mode='backtest')
        orch.ibkr = mock_ibkr
        orch.mock_cash = 50000.0
        
        for batch in seq:
            # Force ready state every frame
            for s in self.symbols:
                if s not in orch.states: orch.states[s] = SymbolState(s)
                orch.states[s].warmup_complete = True
            await orch.process_batch(batch)
            
        trades, open_pos = mock_ibkr._match_trades()
        return trades, orch.mock_cash, mock_ibkr.orders, open_pos

    async def run_split_mode(self, seq):
        mock_ibkr = MockIBKRHistorical()
        shared_redis = MockRedisBridge()
        
        se = SignalEngineV8(symbols=self.symbols, mode='backtest')
        se.ibkr = mock_ibkr
        se.r = shared_redis
        
        oms = ExecutionEngineV8(symbols=self.symbols, mode='backtest')
        oms.ibkr = mock_ibkr
        oms.r = shared_redis
        
        for batch in seq:
            # Force ready state every frame
            for s in self.symbols:
                if s not in se.states: se.states[s] = SymbolState(s)
                se.states[s].warmup_complete = True
            
            # 1. SE processes batch
            shared_redis.message_queue.clear()
            await se.process_batch(batch)
            
            # 2. OMS processes signals
            for msg in shared_redis.message_queue:
                await oms.process_trade_signal(msg)
            
            # SE syncs from OMS for NEXT frame
            se._sync_state_from_oms()
            
        trades, open_pos = mock_ibkr._match_trades()
        return trades, oms.mock_cash, mock_ibkr.orders, open_pos

    async def run_shared_mem_mode(self, seq):
        mock_ibkr = MockIBKRHistorical()
        
        se = SignalEngineV8(symbols=self.symbols, mode='backtest')
        se.ibkr = mock_ibkr
        
        # 🚀 [Setup Shared Memory]
        shared_queue = asyncio.Queue()
        se.signal_queue = shared_queue
        se.use_shared_mem = True
        
        oms = ExecutionEngineV8(
            symbols=self.symbols, 
            mode='backtest',
            shared_states=se.states,
            signal_queue=shared_queue
        )
        oms.ibkr = mock_ibkr
        oms.use_shared_mem = True
        
        # Run loop
        oms_task = asyncio.create_task(oms.run())
        await asyncio.sleep(0.1)  # Give OMS time to initialize Redis and reach loop
        
        for batch in seq:
            # Force ready state every frame
            for s in self.symbols:
                se.states[s].warmup_complete = True
            
            # SE processes batch and waits for OMS via signal_queue.join()
            await se.process_batch(batch)
            
            # SE syncs from OMS (instantly clears is_pending in shared mem mode)
            se._sync_state_from_oms()
            
        # Cleanup
        oms_task.cancel()
        trades, open_pos = mock_ibkr._match_trades()
        return trades, oms.mock_cash, mock_ibkr.orders, open_pos

    def test_logic_parity(self):
        seq = self.get_market_sequence()
        
        # 1. Unified Mode (Baseline)
        print("\n🧪 Running Unified Mode...")
        u_trades, u_cash, u_orders, u_open = asyncio.run(self.run_unified_mode(seq))
        
        # 2. Split Redis Mode
        print("🧪 Running Split Redis Mode...")
        r_trades, r_cash, r_orders, r_open = asyncio.run(self.run_split_mode(seq))
        
        # 3. Split Shared Memory Mode (New Optimized Path)
        print("🧪 Running Split Shared Memory Mode...")
        s_trades, s_cash, s_orders, s_open = asyncio.run(self.run_shared_mem_mode(seq))
        
        print(f"\n📊 Unified Mode: {len(u_orders)} orders, Cash: ${u_cash:.2f}")
        print(f"📊 Redis Mode:   {len(r_orders)} orders, Cash: ${r_cash:.2f}")
        print(f"📊 Shared Mem:   {len(s_orders)} orders, Cash: ${s_cash:.2f}")
        
        # --- Parity Checks ---
        # Compare Cash (Precision: 2 decimal places)
        self.assertAlmostEqual(u_cash, r_cash, places=2, msg="Redis Mode cash mismatch")
        self.assertAlmostEqual(u_cash, s_cash, places=2, msg="Shared Memory Mode cash mismatch")
        
        # Compare Order Count
        self.assertEqual(len(u_orders), len(r_orders), "Redis Mode order count mismatch")
        self.assertEqual(len(u_orders), len(s_orders), "Shared Memory Mode order count mismatch")
        
        # Detailed order comparison (against Shared Mem)
        for i in range(len(u_orders)):
            uo, so = u_orders[i], s_orders[i]
            self.assertEqual(uo['symbol'], so['symbol'], f"Order {i} symbol mismatch")
            self.assertEqual(uo['qty'], so['qty'], f"Order {i} qty mismatch")
            self.assertAlmostEqual(uo['price'], so['price'], places=2, msg=f"Order {i} price mismatch")

        print("\n✅ [SUCCESS] Triple-Engine Logic Parity Verified (Unified == Redis == SharedMem)!")

if __name__ == "__main__":
    unittest.main()
