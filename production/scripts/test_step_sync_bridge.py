#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime, time as dt_time, timezone as dt_timezone, tzinfo, timedelta
import asyncio
import numpy as np
import unittest
import logging
from unittest.mock import patch, MagicMock
import pickle

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestStepSync")

sys.path.append(os.path.abspath(".")) # Add project root
try:
    from production.utils import serialization_utils as ser
except ImportError:
    from utils import serialization_utils as ser

sys.path.append(os.path.abspath("production"))
sys.path.append(os.path.abspath("production/baseline"))
sys.path.append(os.path.abspath("production/history_replay"))
sys.path.append(os.path.abspath("production/DB"))

# --- MOCKING ---
def mock_module(name):
    m = MagicMock()
    sys.modules[name] = m
    return m

mock_pd = mock_module("pandas")
mock_pd.to_datetime = lambda x, **kwargs: x if isinstance(x, (datetime, dt_time)) else datetime.fromtimestamp(float(x))
mock_pd.Timestamp = lambda x: x if isinstance(x, datetime) else datetime.fromtimestamp(float(x))

class MockTZ(tzinfo):
    def utcoffset(self, dt): return timedelta(0)
    def dst(self, dt): return timedelta(0)
    def tzname(self, dt): return "UTC"
    def localize(self, dt): return dt.replace(tzinfo=self)
    def normalize(self, dt): return dt

mock_pytz = mock_module("pytz")
mock_pytz.timezone.return_value = MockTZ()
mock_pytz.utc = MockTZ()

mock_module("torch")
mock_module("scipy")
mock_module("scipy.stats")
mock_module("redis")
mock_module("ib_insync")
mock_module("psycopg2")
mock_module("psycopg2.extras")
mock_module("trading_tft_stock_embed")

import liquidity_rules
liquidity_rules.LiquidityRiskManager = MagicMock()
liquidity_rules.LiquidityRiskManager.evaluate_order.return_value = {'final_alloc': 1000.0, 'chunks': 1, 'reason': 'TEST'}

# Set the environment variable to trick the engine into LIVEREPLAY block
os.environ['RUN_MODE'] = 'LIVEREPLAY'

# --- ENGINES ---
import signal_engine_v8
import execution_engine_v8
import orchestrator_execution
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical

class MockRedisBridge:
    def __init__(self):
        self.message_queue = []
        self.kv_store = {"sync:orch_done": "0"}
        self.hash_store = {}
        
    def xadd(self, stream, doc, **kwargs):
        data = doc.get('data') or doc.get(b'data')
        if data:
            payload = ser.unpack(data)
            self.message_queue.append(payload)

    def hget(self, name, key): 
        val = self.hash_store.get(name, {}).get(key)
        return val.encode() if isinstance(val, str) else val

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
        
    def pipeline(self): return self
    def execute(self): pass
    def xack(self, stream, group, msg_id): pass
    def xgroup_create(self, stream, group, **kwargs): pass


class TestStepSyncBridge(unittest.TestCase):
    def setUp(self):
        self.test_symbols = ['SPY', 'AAPL', 'NVDA', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'PLTR', 'AMD', 'NFLX']
        execution_engine_v8.SYNC_EXECUTION = False
        signal_engine_v8.SYNC_EXECUTION = False
        orchestrator_execution.SYNC_EXECUTION = False
        
    def create_mock_engines(self, mode='realtime'):
        with patch('torch.jit.load', return_value=MagicMock()):
            sig_eng = SignalEngineV8(symbols=self.test_symbols, mode=mode)
            sig_eng.trading_enabled = True
            
            # Shared mock redis to inspect messages
            shared_redis = MockRedisBridge()
            sig_eng.r = shared_redis
            for st in sig_eng.states.values(): st.warmup_complete = True
            
            exe_eng = ExecutionEngineV8(symbols=self.test_symbols, mode=mode)
            exe_eng.trading_enabled = True
            exe_eng.r = shared_redis
            for st in exe_eng.states.values(): st.warmup_complete = True
            
            exe_eng.ibkr = MockIBKRHistorical()
            exe_eng.ibkr.place_option_order = MagicMock(return_value=True)
            exe_eng.state_manager.save_state = MagicMock()
            exe_eng.accounting.save_intraday_state_to_db = MagicMock()
            
            return sig_eng, exe_eng, shared_redis

    def get_test_batch(self, ts, alpha_val=5.0):
        n = len(self.test_symbols)
        return {
            'symbols': self.test_symbols,
            'ts': ts,
            'stock_price': np.array([150.0] * n),
            'precalc_alpha': np.array([alpha_val] * n),
            'fast_vol': np.array([1.0] * n), 
            'is_new_minute': True,
            'spy_roc_5min': np.array([0.005] * n),
            'qqq_roc_5min': np.array([0.005] * n),
            'live_options': {}
        }
        
    def mock_opt_data(self):
        return {
            'has_feed': True, 'call_price': 3.0, 'call_bid': 2.9, 'call_ask': 3.1, 'call_iv': 0.3, 'call_id': "CALL_X",
            'call_k': 150.0, 'call_vol': 100, 'call_bid_size': 10, 'call_ask_size': 10,
            'put_price': 3.0, 'put_bid': 2.9, 'put_ask': 3.1, 'put_iv': 0.3, 'put_id': "PUT_X",
            'put_k': 150.0, 'put_vol': 100, 'put_bid_size': 10, 'put_ask_size': 10
        }

    async def run_step_sync_frame(self, sig_eng, exe_eng, shared_redis, batch):
        # 1. Driver sets lock to 0
        shared_redis.set("sync:orch_done", "0")
        
        # 2. Signal Engine computes the batch and emits payloads (including SYNC)
        shared_redis.message_queue.clear()
        await sig_eng.process_batch(batch=batch)
        
        has_sync_marker = any(msg.get('action') == 'SYNC' for msg in shared_redis.message_queue)
        
        # 3. Execution Engine sequentially drains and processes the trade signals
        sync_unlocked = False
        for msg in shared_redis.message_queue:
            await exe_eng.process_trade_signal(msg)
            if shared_redis.get("sync:orch_done") != "0":
                sync_unlocked = True
                
        return has_sync_marker, sync_unlocked

    def test_pipeline_without_trades(self):
        print("\n🧪 [CASE 1] Step-Sync Frame containing NO trades...")
        sig_eng, exe_eng, r = self.create_mock_engines()
        
        # Provide NO entry signals
        sig_eng.strategy.decide_entry = MagicMock(return_value=None)
        
        batch = self.get_test_batch(ts=1696948200.0)
        has_sync, unlocked = asyncio.run(self.run_step_sync_frame(sig_eng, exe_eng, r, batch))
        
        self.assertTrue(has_sync, "Signal Engine failed to emit SYNC_MARKER.")
        self.assertTrue(unlocked, "Execution Engine failed to unlock sync:orch_done.")
        self.assertEqual(r.get("sync:orch_done").decode(), "1696948200.0", "Unlock timestamp mismatch.")
        print("✅ [SUCCESS] Empty frame correctly generated unlock sentinel.")

    def test_pipeline_with_extreme_trades(self):
        print("\n🧪 [CASE 2] Step-Sync Frame containing heavy ICEBERG trades...")
        sig_eng, exe_eng, r = self.create_mock_engines()
        
        def mock_entry(ctx):
            return {'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'score': 10.0, 'reason': 'TEST'}
        
        sig_eng.strategy.decide_entry = MagicMock(side_effect=mock_entry)
        sig_eng._get_opt_data_realtime = MagicMock(return_value=self.mock_opt_data())
        exe_eng.accounting.can_open_position = MagicMock(return_value=True)
        
        batch = self.get_test_batch(ts=1696948260.0)
        has_sync, unlocked = asyncio.run(self.run_step_sync_frame(sig_eng, exe_eng, r, batch))
        
        self.assertTrue(has_sync, "Signal Engine failed to emit SYNC_MARKER.")
        self.assertTrue(unlocked, "Execution Engine failed to unlock sync:orch_done.")
        self.assertEqual(r.get("sync:orch_done").decode(), "1696948260.0", "Unlock timestamp mismatch.")
        print("✅ [SUCCESS] Heavy execution frame sequentially digested and unlocked.")
        
    def test_pipeline_backtest_mode(self):
        print("\n🧪 [CASE 3] Step-Sync Frame in BACKTEST mode...")
        os.environ['RUN_MODE'] = 'BACKTEST'
        sig_eng, exe_eng, r = self.create_mock_engines(mode='backtest')
        
        # Provide NO entry signals
        sig_eng.strategy.decide_entry = MagicMock(return_value=None)
        
        batch = self.get_test_batch(ts=1696948320.0)
        has_sync, unlocked = asyncio.run(self.run_step_sync_frame(sig_eng, exe_eng, r, batch))
        
        self.assertTrue(has_sync, "Signal Engine failed to emit SYNC_MARKER in BACKTEST mode.")
        self.assertTrue(unlocked, "Execution Engine failed to unlock sync:orch_done in BACKTEST mode.")
        self.assertEqual(r.get("sync:orch_done").decode(), "1696948320.0", "Unlock timestamp mismatch in BACKTEST mode.")
        print("✅ [SUCCESS] BACKTEST mode correctly generated and processed unlock sentinel.")


if __name__ == "__main__":
    unittest.main()
