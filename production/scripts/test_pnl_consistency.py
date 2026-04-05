#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import asyncio
import numpy as np
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project paths
sys.path.append(os.path.abspath("production/baseline"))
sys.path.append(os.path.abspath("production/history_replay"))
sys.path.append(os.path.abspath("production/utils"))

# Mock out heavy dependencies
sys.modules['redis'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()
sys.modules['psycopg2.extras'] = MagicMock()
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['trading_tft_stock_embed'] = MagicMock()
mock_scipy = MagicMock()
mock_scipy.__name__ = 'scipy'
sys.modules['scipy'] = mock_scipy
mock_stats = MagicMock()
mock_stats.__name__ = 'scipy.stats'
sys.modules['scipy.stats'] = mock_stats
# Do NOT mock pytz as pandas depends on it

import system_orchestrator_v8
from system_orchestrator_v8 import V8Orchestrator
from mock_ibkr_historical import MockIBKRHistorical
import config

class TestPnLConsistency(unittest.TestCase):
    def setUp(self):
        # Force backtest mode and sync execution
        config.RUN_MODE = 'BACKTEST'
        config.IS_SIMULATED = True
        config.IS_BACKTEST = True
        config.TRADING_ENABLED = False
        config.SYNC_EXECUTION = True
        
        # Use config slippage
        self.slippage_entry = config.SLIPPAGE_ENTRY_PCT
        self.slippage_exit = config.SLIPPAGE_EXIT_PCT
        
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'PLTR']
        self.orch = V8Orchestrator(self.symbols, mode='backtest')
        
        # Ensure MockIBKR has 0 delay as per user edit
        self.orch.ibkr.execution_delay_seconds = 0
        self.orch.ibkr.execution_delay_bars = 0
        self.orch.ibkr.fill_delay = 0

    def test_pnl_calculation_consistency(self):
        print("\n🧪 Testing PnL Consistency: Orchestrator vs MockIBKR...")
        
        sym = 'AAPL'
        for s in self.symbols:
            self.orch.states[s].warmup_complete = True
        
        st = self.orch.states[sym]
        
        # 1. Simulate Entry at TS (approx 10:00 AM NY time on a weekday)
        ts_base = 1710424800.0 # 2024-03-14 10:00:00 AM EDT
        ts_entry = ts_base
        stock_price_entry = 150.0
        opt_price_entry = 3.0
        # For simplicity, mid = bid = ask
        n = len(self.symbols)
        batch_entry = {
            'symbols': self.symbols,
            'ts': ts_entry,
            'stock_price': np.array([stock_price_entry] * n),
            'feed_call_price': np.array([opt_price_entry] * n),
            'feed_call_bid': np.array([opt_price_entry] * n),
            'feed_call_ask': np.array([opt_price_entry] * n),
            'feed_call_id': [f'{s}_C' for s in self.symbols],
            'feed_call_k': np.array([150.0] * n),
            'feed_call_iv': np.array([0.3] * n),
            'precalc_alpha': np.array([5.0] * n),
            'fast_vol': np.array([1.0] * n),
            'is_new_minute': True
        }
        
        # Force Entry signal
        self.orch.strategy.decide_entry = MagicMock(return_value={
            'action': 'BUY', 'dir': 1, 'tag': 'CALL_ATM', 'reason': 'TEST_ENTRY'
        })
        
        asyncio.run(self.orch.process_batch(batch_entry))
        
        # Verify Entry
        self.assertEqual(st.position, 1)
        entry_price_expected = round(opt_price_entry * (1 + self.slippage_entry), 2)
        self.assertAlmostEqual(st.entry_price, entry_price_expected, places=2)
        print(f"✅ Entry recorded at {st.entry_price}")
        
        # 2. Simulate Unrealized PnL later
        ts_mid = ts_base + 300.0 # +5 mins
        opt_price_mid = 3.5
        batch_mid = batch_entry.copy()
        batch_mid['ts'] = ts_mid
        batch_mid['feed_call_price'] = np.array([opt_price_mid] * n)
        batch_mid['feed_call_bid'] = np.array([opt_price_mid] * n)
        batch_mid['feed_call_ask'] = np.array([opt_price_mid] * n)
        
        self.orch.strategy.decide_entry = MagicMock(return_value=None)
        self.orch.strategy.check_exit = MagicMock(return_value=None)
        
        asyncio.run(self.orch.process_batch(batch_mid))
        
        # Check Unrealized PnL
        # Expected: (3.5 - entry_price) * qty * 100
        expected_unrealized = (opt_price_mid - st.entry_price) * st.qty * 100
        
        # Capture logs or check orch.accounting state if possible
        # Here we directly check the logic we've fixed
        unrealized_calc = (st.last_opt_price - st.entry_price) * st.qty * 100
        self.assertAlmostEqual(unrealized_calc, expected_unrealized, places=2)
        print(f"✅ Unrealized PnL consistent: {unrealized_calc}")
        
        # 3. Simulate Exit later
        ts_exit = ts_base + 600.0 # +10 mins
        opt_price_exit = 4.0
        batch_exit = batch_entry.copy()
        batch_exit['ts'] = ts_exit
        batch_exit['feed_call_price'] = np.array([opt_price_exit] * n)
        batch_exit['feed_call_bid'] = np.array([opt_price_exit] * n)
        batch_exit['feed_call_ask'] = np.array([opt_price_exit] * n)
        
        self.orch.strategy.check_exit = MagicMock(return_value={
            'action': 'SELL', 'reason': 'TEST_EXIT'
        })
        
        asyncio.run(self.orch.process_batch(batch_exit))
        
        # Verify Exit
        self.assertEqual(st.position, 0)
        exit_price_expected = round(opt_price_exit * (1 - self.slippage_exit), 2)
        
        # 4. Final Comparison
        orchestrator_realized = self.orch.realized_pnl
        
        # Extract trades from MockIBKR
        trades, _ = self.orch.ibkr._match_trades()
        mock_pnl = sum(t['pnl'] for t in trades)
        
        print(f"📊 Orchestrator Realized PnL: {orchestrator_realized:.2f}")
        print(f"📊 MockIBKR Matched PnL: {mock_pnl:.2f}")
        
        # THEY MUST BE EQUAL NOW!
        self.assertAlmostEqual(orchestrator_realized, mock_pnl, places=2)
        print("✅ [SUCCESS] PnL Parity Verified between Accounting and MockIBKR.")

if __name__ == "__main__":
    unittest.main()
