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
logger = logging.getLogger("TestParity")

sys.path.append(os.path.abspath("production/baseline"))
sys.path.append(os.path.abspath("production/history_replay"))
sys.path.append(os.path.abspath("production/DB"))

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
    def __repr__(self): return "MockTZ(UTC)"

mock_pytz = mock_module("pytz")
mock_pytz.timezone.return_value = MockTZ()
mock_pytz.utc = MockTZ()

mock_torch = mock_module("torch")
mock_torch.device.return_value = "cpu"
mock_module("scipy")
mock_module("scipy.stats")
mock_module("redis")
mock_module("ib_insync")
mock_module("psycopg2")
mock_module("psycopg2.extras")
mock_module("trading_tft_stock_embed")

import liquidity_rules
liquidity_rules.LiquidityRiskManager = MagicMock()
liquidity_rules.LiquidityRiskManager.evaluate_order.return_value = {
    'final_alloc': 1000.0,
    'chunks': 1,
    'reason': 'TEST'
}

# --- Import Split Engines ---
import signal_engine_v8
import execution_engine_v8
from signal_engine_v8 import SignalEngineV8
from execution_engine_v8 import ExecutionEngineV8
from mock_ibkr_historical import MockIBKRHistorical
from config import ROLLING_WINDOW

class MockRedisBridge:
    def __init__(self):
        self.message_queue = []
        
    def xadd(self, stream, doc, **kwargs):
        data = doc.get('data') or doc.get(b'data')
        payload = pickle.loads(data)
        self.message_queue.append(payload)

    def hget(self, name, key):
        return None

class TestExecutionParity(unittest.TestCase):
    def setUp(self):
        self.initial_account = 100000.0
        self.test_symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'TSLA', 'PLTR', 'AMD', 'NFLX']
        execution_engine_v8.SYNC_EXECUTION = False
        signal_engine_v8.SYNC_EXECUTION = False
        
    def create_mock_engines(self, mode, sync_exec=False, max_pos=10):
        # PATCH GLOBAL FLAGS
        execution_engine_v8.SYNC_EXECUTION = sync_exec
        signal_engine_v8.SYNC_EXECUTION = sync_exec
        import orchestrator_execution
        orchestrator_execution.SYNC_EXECUTION = sync_exec
        execution_engine_v8.TRADING_ENABLED = True
        signal_engine_v8.TRADING_ENABLED = True
        execution_engine_v8.MAX_POSITIONS = max_pos
        signal_engine_v8.MAX_POSITIONS = max_pos
        
        with patch('torch.jit.load', return_value=MagicMock()):
            # Init Signal Engine
            sig_eng = SignalEngineV8(symbols=self.test_symbols, mode=mode)
            sig_eng.trading_enabled = True
            sig_eng.only_log_alpha = False
            sig_eng.max_positions = max_pos
            sig_eng.r = MockRedisBridge()
            for st in sig_eng.states.values():
                st.warmup_complete = True
            
            # Init OMS
            exe_eng = ExecutionEngineV8(symbols=self.test_symbols, mode=mode)
            exe_eng.trading_enabled = True
            exe_eng.max_positions = max_pos
            for st in exe_eng.states.values():
                st.warmup_complete = True
            exe_eng.ibkr = MockIBKRHistorical()
            exe_eng.ibkr.place_option_order = MagicMock(return_value=True)
            exe_eng.state_manager.save_state = MagicMock()
            exe_eng.accounting.save_intraday_state_to_db = MagicMock()
            
            return sig_eng, exe_eng

    async def run_piped_batch(self, sig_eng, exe_eng, batch):
        sig_eng.r.message_queue.clear()
        await sig_eng.process_batch(batch=batch)
        for msg in sig_eng.r.message_queue:
            # Transfer execution decisions to Execution Engine
            await exe_eng.process_trade_signal(msg)

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

    def mock_opt_data(self, sym, price=3.0, bid_size=10, ask_size=10):
        return {
            'has_feed': True,
            'call_price': price, 'call_bid': price-0.1, 'call_ask': price+0.1, 'call_iv': 0.3, 'call_id': f"{sym}_C",
            'call_k': 150.0, 'call_vol': 100, 'call_bid_size': bid_size, 'call_ask_size': ask_size,
            'put_price': price, 'put_bid': price-0.1, 'put_ask': price+0.1, 'put_iv': 0.3, 'put_id': f"{sym}_P",
            'put_k': 150.0, 'put_vol': 100, 'put_bid_size': bid_size, 'put_ask_size': ask_size
        }

    def test_backtest_vs_sync_realtime(self):
        print("\n🧪 [CASE 1] Basic Parity Check...")
        sig_bt, exe_bt = self.create_mock_engines(mode='backtest', sync_exec=True)
        sig_rt, exe_rt = self.create_mock_engines(mode='realtime', sync_exec=True)

        for sig_eng, exe_eng in [(sig_bt, exe_bt), (sig_rt, exe_rt)]:
            def mock_entry(ctx):
                if ctx['symbol'] == 'AAPL':
                    return {'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'legacy_tag': 'opt_8', 'score': 10.0, 'reason': 'TEST'}
                return None
            sig_eng.strategy.decide_entry = MagicMock(side_effect=mock_entry)
            data = self.mock_opt_data('AAPL')
            sig_eng._get_opt_data_realtime = MagicMock(return_value=data)
            sig_eng._get_opt_data_backtest = MagicMock(return_value=data)
            exe_eng.accounting.can_open_position = MagicMock(return_value=True)

        batch = self.get_test_batch(ts=1709287200.0 + 3600)
        asyncio.run(self.run_piped_batch(sig_bt, exe_bt, batch))
        asyncio.run(self.run_piped_batch(sig_rt, exe_rt, batch))
        
        # Checking execution states
        self.assertEqual(exe_bt.states['AAPL'].position, 1)
        self.assertEqual(exe_rt.states['AAPL'].position, 1)
        print("✅ [SUCCESS] Basic Parity Verified.")

    def test_liquidity_truncation_parity(self):
        print("\n🧪 [CASE 2] Liquidity Truncation Parity...")
        sig_bt, exe_bt = self.create_mock_engines(mode='backtest', sync_exec=True)
        sig_rt, exe_rt = self.create_mock_engines(mode='realtime', sync_exec=True)

        ts = 1709287200.0 + 3660
        for sig_eng, exe_eng in [(sig_bt, exe_bt), (sig_rt, exe_rt)]:
            # State synchronization between Compute and Execution engines
            for st in [sig_eng.states['AAPL'], exe_eng.states['AAPL']]:
                st.position = 5; st.qty = 5; st.entry_price = 2.0; st.entry_ts = ts - 120
            
            sig_eng.strategy.check_exit = MagicMock(side_effect=lambda ctx: {'action': 'SELL', 'reason': 'PROFIT'} if ctx['symbol']=='AAPL' else None)
            data = self.mock_opt_data('AAPL', bid_size=1) 
            sig_eng._get_opt_data_realtime = MagicMock(return_value=data)
            sig_eng._get_opt_data_backtest = MagicMock(return_value=data)

        batch = self.get_test_batch(ts=ts)
        asyncio.run(self.run_piped_batch(sig_bt, exe_bt, batch))
        asyncio.run(self.run_piped_batch(sig_rt, exe_rt, batch))
        
        self.assertEqual(exe_bt.states['AAPL'].position, 0)
        self.assertEqual(exe_rt.states['AAPL'].position, 0)
        print("✅ [SUCCESS] Liquidity Truncation Parity Verified.")

    def test_max_positions_rejection(self):
        print("\n🧪 [CASE 3] Max Positions Rejection Parity...")
        sig_bt, exe_bt = self.create_mock_engines(mode='backtest', sync_exec=True, max_pos=1)
        sig_rt, exe_rt = self.create_mock_engines(mode='realtime', sync_exec=True, max_pos=1)
        
        for sig_eng, exe_eng in [(sig_bt, exe_bt), (sig_rt, exe_rt)]:
            signals = {'AAPL': {'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'score': 100.0, 'reason':'T1'},
                       'NVDA': {'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'score': 90.0, 'reason':'T2'}}
            sig_eng.strategy.decide_entry = MagicMock(side_effect=lambda ctx: signals.get(ctx['symbol']))
            sig_eng._get_opt_data_realtime = MagicMock(return_value=self.mock_opt_data('AAPL'))
            sig_eng._get_opt_data_backtest = MagicMock(return_value=self.mock_opt_data('AAPL'))
            exe_eng.accounting.can_open_position = MagicMock(return_value=True)

        batch = self.get_test_batch(ts=1709287200.0 + 3600)
        asyncio.run(self.run_piped_batch(sig_bt, exe_bt, batch))
        asyncio.run(self.run_piped_batch(sig_rt, exe_rt, batch))
        
        self.assertEqual(exe_bt.states['AAPL'].position, 1)
        self.assertEqual(exe_bt.states['NVDA'].position, 0)
        self.assertEqual(exe_rt.states['AAPL'].position, 1)
        self.assertEqual(exe_rt.states['NVDA'].position, 0)
        print("✅ [SUCCESS] Max Positions Rejection Parity Verified.")

    def test_eod_clear_parity(self):
        print("\n🧪 [CASE 4] EOD Synchronous Clearing Parity...")
        sig_bt, exe_bt = self.create_mock_engines(mode='backtest', sync_exec=True)
        sig_rt, exe_rt = self.create_mock_engines(mode='realtime', sync_exec=True)
        
        ts_eod = 1709308800.0 # 16:00:00 UTC
        for sig_eng, exe_eng in [(sig_bt, exe_bt), (sig_rt, exe_rt)]:
            for st in [sig_eng.states['AAPL'], exe_eng.states['AAPL']]:
                st.position = 1; st.qty = 1; st.entry_price = 2.0; st.entry_ts = ts_eod - 3600
            data = self.mock_opt_data('AAPL')
            sig_eng._get_opt_data_realtime = MagicMock(return_value=data)
            sig_eng._get_opt_data_backtest = MagicMock(return_value=data)

        batch = self.get_test_batch(ts=ts_eod)
        asyncio.run(self.run_piped_batch(sig_bt, exe_bt, batch))
        asyncio.run(self.run_piped_batch(sig_rt, exe_rt, batch))
        
        self.assertEqual(exe_bt.states['AAPL'].position, 0)
        self.assertEqual(exe_rt.states['AAPL'].position, 0)
        print("✅ [SUCCESS] EOD Clear Parity Verified.")

    def test_alpha_scaling_parity(self):
        print("\n🧪 [CASE 5] Alpha Normalization Consistency...")
        sig_bt, exe_bt = self.create_mock_engines(mode='backtest', sync_exec=True)
        sig_rt, exe_rt = self.create_mock_engines(mode='realtime', sync_exec=True)
        
        ts_start = 1709287200.0
        for i in range(ROLLING_WINDOW + 5):
            ts = ts_start + i * 60
            batch = self.get_test_batch(ts=ts, alpha_val=10.0 + i)
            batch['is_new_minute'] = True 
            asyncio.run(self.run_piped_batch(sig_bt, exe_bt, batch))
            asyncio.run(self.run_piped_batch(sig_rt, exe_rt, batch))
            
        # Verify alpha inside the Compute Engine (SignalEngineV8)
        st_bt = sig_bt.states['AAPL']; st_rt = sig_rt.states['AAPL']
        self.assertEqual(len(st_bt.alpha_history), len(st_rt.alpha_history))
        self.assertAlmostEqual(st_bt.last_alpha_z, st_rt.last_alpha_z, places=4)
        print("✅ [SUCCESS] Alpha Scaling Consistency Verified.")

if __name__ == "__main__":
    unittest.main()
