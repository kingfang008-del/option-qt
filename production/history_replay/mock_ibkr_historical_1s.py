import redis
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time
from config import (
    COMMISSION_PER_CONTRACT,
    EXECUTION_DELAY_BARS,
    EXECUTION_DELAY_SECONDS,
    BACKTEST_1S_DISABLE_DELAY,
    BACKTEST_1S_DISABLE_POSTHOC_REPRICE,
)

logger = logging.getLogger("MockIBKR_Hist")

class MockContract:
    def __init__(self, symbol, tag, localSymbol=None):
        self.symbol = symbol
        self.tag = tag 
        self.localSymbol = localSymbol or f"{symbol}_{tag}_MOCK"
        self.secType = 'OPT'

class MockIBKRHistorical:
    def __init__(self, data_dir=None):
        self.r = redis.Redis(host='localhost', port=6379, db=1)
        self.orders = []
        self.market_history = [] # [New] 存储完整市场行情，用于后验时延分析
        self.initial_capital =  50_000.0 # [对齐 V15] 初始资金
        # =========================================================
        # 🚀 [核心引擎升级] 与 OMS 共用统一执行延迟配置，避免双处手改
        # =========================================================
        self.execution_delay_bars = 0 if BACKTEST_1S_DISABLE_DELAY else EXECUTION_DELAY_BARS
        self.execution_delay_seconds = 0 if BACKTEST_1S_DISABLE_DELAY else EXECUTION_DELAY_SECONDS
        self.disable_posthoc_reprice = BACKTEST_1S_DISABLE_POSTHOC_REPRICE
        self.fill_delay = (
            f"{self.execution_delay_seconds}s"
            if self.execution_delay_seconds > 0
            else f"{self.execution_delay_bars} bar(s)"
        )
        self.ib = self._create_mock_ib()



    @property
    def cash(self):
        """兼容性接口：对接执行引擎与信号引擎的 cash 注入检查"""
        return self.initial_capital

    def _create_mock_ib(self):
        class MockIBInternal:
            def isConnected(self): return True
            def portfolio(self): return []
            def reqMktData(self, *args, **kwargs): pass
            def ticker(self, contract): return None
        return MockIBInternal()

    async def connect(self):
        logger.info(f"🔌 Mock IBKR Connected (Historical Replay Mode).")
        pass

    async def start_stock_stream(self, symbols): pass

    def get_contract(self, symbol, tag):
        return MockContract(symbol, tag)

    def place_option_order(self, contract, action, qty, order_type, lmt_price, reason="", custom_time=None, stock_price=0.0, stop_loss_pct=0.07, chunks=1, liq_reason=""):
        """
        下单：优先使用 custom_time (历史时间戳)
        """
        if custom_time:
            ts = float(custom_time)
        else:
            # 兜底：去 Redis 查当前回放时间
            ts_str = self.r.get("replay:current_ts")
            ts = float(ts_str) if ts_str else time.time()
        
        # [Fix Log] 强制转换为 NY Time 显示，消除歧义
        from pytz import timezone
        ny_tz = timezone('America/New_York')
        utc_tz = timezone('UTC')
        
       # 假设 ts 是 UTC Timestamp (Standard)
        dt_utc = datetime.fromtimestamp(ts, utc_tz)
        dt_ny = dt_utc.astimezone(ny_tz)
        dt_str = dt_ny.strftime('%Y-%m-%d %H:%M:%S') # 这里显示的是 NY 时间
        
        fill = lmt_price
        
        # 记录订单
        local_symbol = getattr(contract, 'localSymbol', '')
        # 如果是标准 IBKR OSI 格式，倒数第 9 位是 C/P。如果不是，根据 tag 判定。
        if len(local_symbol) >= 9 and local_symbol.strip()[-9] in ['C', 'P']:
             opt_dir = 'CALL' if local_symbol.strip()[-9] == 'C' else 'PUT'
        else:
             tag = getattr(contract, 'tag', '').upper()
             opt_dir = 'CALL' if 'CALL' in tag else 'PUT'
             
        self.orders.append({
            'ts': ts,
            'time': dt_str,
            'date': dt_ny.date(),
            'symbol': getattr(contract, 'symbol', ''),
            'action': action,
            'qty': qty,
            'price': fill,
            'stock_price': stock_price,
            'tag': getattr(contract, 'tag', getattr(contract, 'secType', '')),
            'contract': local_symbol,
            'opt_dir': opt_dir,         # [New]
            'reason': reason,
            'liq_chunks': chunks,       # [New]
            'liq_reason': liq_reason    # [New]
        })

    def record_market_data(self, batch, alphas=None):
        """记录每秒/每批次的行情数据，用于回测结束后的延迟分析"""
        try:
            ts = float(batch['ts'])
            symbols = batch['symbols']
            
            # 提取期权价格
            # 在 S5ParquetDriver 中，feed_call_price/feed_put_price 是 numpy array
            call_prices = batch.get('feed_call_price', [])
            put_prices = batch.get('feed_put_price', [])
            
            # 提取 Bid/Ask (用于更精确的模拟)
            call_bids = batch.get('feed_call_bid', [])
            call_asks = batch.get('feed_call_ask', [])
            put_bids = batch.get('feed_put_bid', [])
            put_asks = batch.get('feed_put_ask', [])
            stock_prices = batch.get('stock_price', []) # [New] 提取正股价格
            
            snapshot = {}
            for i, sym in enumerate(symbols):
                snapshot[sym] = {
                    'c': float(call_prices[i]) if i < len(call_prices) else 0.0,
                    'p': float(put_prices[i]) if i < len(put_prices) else 0.0,
                    'cb': float(call_bids[i]) if i < len(call_bids) else 0.0,
                    'ca': float(call_asks[i]) if i < len(call_asks) else 0.0,
                    'pb': float(put_bids[i]) if i < len(put_bids) else 0.0,
                    'pa': float(put_asks[i]) if i < len(put_asks) else 0.0,
                    's': float(stock_prices[i]) if i < len(stock_prices) else 0.0,
                    'a': float(alphas[i]) if alphas is not None and i < len(alphas) else 0.0
                }
            
            self.market_history.append((ts, snapshot))
            
            # 限制内存占用 (如果回测时间过长，只保留最近行情是没有意义的，因为我们需要全局回放)
            # 但这里假设是单次运行，内存足够。如果不足，可以考虑写入 temp sqlite
        except Exception as e:
            logger.error(f"Error recording market data: {e}")

    def _get_price_at_time(self, symbol, ts, opt_dir, action, price_type='mid'):
        """在行情历史中寻找最接近指定时间戳的价格
        price_type: 'mid' (优先 (Bid+Ask)/2), 'best' (Bid/Ask 优先), 'last' (纯 Close)
        """
        if not self.market_history:
            return None
            
        import bisect
        times = [x[0] for x in self.market_history]
        # Use latest snapshot at-or-before ts to avoid look-ahead bias when
        # there are sparse/missing 1s bars.
        idx = bisect.bisect_right(times, ts) - 1
        if idx < 0:
            return None
        if idx >= len(self.market_history):
            idx = len(self.market_history) - 1
            
        best_ts, snapshot = self.market_history[idx]
        
        if abs(best_ts - ts) > 120:
            return None
            
        data = snapshot.get(symbol)
        if not data:
            return None
            
        # 1. 如果显式要求 mid
        if price_type == 'mid':
            if opt_dir == 'CALL':
                if data['cb'] > 0 and data['ca'] > 0:
                    return (data['cb'] + data['ca']) / 2.0
                return data['c']
            else:
                if data['pb'] > 0 and data['pa'] > 0:
                    return (data['pb'] + data['pa']) / 2.0
                return data['p']

        # 2. 如果显式要求 best (原先的逻辑)
        if opt_dir == 'CALL':
            if action == 'SELL': # 卖出平仓，取 Bid
                return data['cb'] if data['cb'] > 0 else data['c']
            else: # 买入
                return data['ca'] if data['ca'] > 0 else data['c']
        else: # PUT
            if action == 'SELL':
                return data['pb'] if data['pb'] > 0 else data['p']
            else:
                return data['pa'] if data['pa'] > 0 else data['p']

    def _get_price_at_bar_offset(self, symbol, ts, opt_dir, action, bar_offset, price_type='mid'):
        """寻找指定时间戳之后第 N 个 Bar 的价格
        price_type: 'mid' (优先 (Bid+Ask)/2), 'best' (Bid/Ask 优先)
        """
        if not self.market_history:
            return None
            
        import bisect
        times = [x[0] for x in self.market_history]
        # Anchor at-or-before ts first, then apply bar offset.
        idx = bisect.bisect_right(times, ts) - 1
        if idx < 0:
            return None
        target_idx = idx + bar_offset
        if target_idx >= len(self.market_history):
            target_idx = len(self.market_history) - 1
            
        best_ts, snapshot = self.market_history[target_idx]
        data = snapshot.get(symbol)
        if not data:
            return None

        if price_type == 'mid':
            if opt_dir == 'CALL':
                if data['cb'] > 0 and data['ca'] > 0: return (data['cb'] + data['ca']) / 2.0
                return data['c']
            else:
                if data['pb'] > 0 and data['pa'] > 0: return (data['pb'] + data['pa']) / 2.0
                return data['p']
            
        if opt_dir == 'CALL':
            if action == 'SELL': return data['cb'] if data['cb'] > 0 else data['c']
            else: return data['ca'] if data['ca'] > 0 else data['c']
        else:
            if action == 'SELL': return data['pb'] if data['pb'] > 0 else data['p']
            else: return data['pa'] if data['pa'] > 0 else data['p']

    def _match_trades(self):
        """支持部分成交 FIFO、手续费、时延重定价的 1s 历史撮合。"""
        open_positions = {}
        trades = []

        sorted_orders = sorted(self.orders, key=lambda x: x['ts'])

        for od in sorted_orders:
            sym = od['symbol']

            if od['action'] == 'BUY':
                if sym not in open_positions:
                    open_positions[sym] = []
                open_positions[sym].append(dict(od))

            elif od['action'] == 'SELL':
                if sym in open_positions and open_positions[sym]:
                    remaining_sell_qty = od['qty']
                    while remaining_sell_qty > 0 and open_positions[sym]:
                        buy_od = open_positions[sym][0]
                        match_qty = min(buy_od['qty'], remaining_sell_qty)

                        entry_price = buy_od['price']
                        exit_price = od['price']
                        opt_dir = buy_od.get('opt_dir', 'CALL')

                        delay_bars = getattr(self, 'execution_delay_bars', 0)
                        delay_secs = getattr(self, 'execution_delay_seconds', 0)

                        if not self.disable_posthoc_reprice:
                            if delay_secs > 0:
                                delayed_entry = self._get_price_at_time(sym, buy_od['ts'] + delay_secs, opt_dir, 'BUY')
                                delayed_exit = self._get_price_at_time(sym, od['ts'] + delay_secs, opt_dir, 'SELL')
                            elif delay_bars > 0:
                                delayed_entry = self._get_price_at_bar_offset(sym, buy_od['ts'], opt_dir, 'BUY', delay_bars, price_type='mid')
                                delayed_exit = self._get_price_at_bar_offset(sym, od['ts'], opt_dir, 'SELL', delay_bars, price_type='mid')
                            else:
                                delayed_entry, delayed_exit = None, None

                            if delayed_entry is not None and delayed_entry > 0.01:
                                entry_price = delayed_entry
                            if delayed_exit is not None and delayed_exit > 0.01:
                                exit_price = delayed_exit

                        entry_commission = match_qty * COMMISSION_PER_CONTRACT
                        exit_commission = match_qty * COMMISSION_PER_CONTRACT
                        cost = (entry_price * match_qty * 100) + entry_commission
                        proceeds = (exit_price * match_qty * 100) - exit_commission
                        pnl = proceeds - cost
                        roi = (exit_price - entry_price) * match_qty * 100 / cost if cost > 0 else 0

                        trades.append({
                            'date': buy_od['date'],
                            'symbol': sym,
                            'entry_time': buy_od['time'],
                            'exit_time': od['time'],
                            'pnl': pnl,
                            'roi': roi,
                            'qty': match_qty,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'reason': od.get('reason', 'N/A'),
                            'duration': (od['ts'] - buy_od['ts']) / 60.0,
                            'entry_ts': buy_od['ts'],
                            'exit_ts': od['ts'],
                            'contract': buy_od.get('contract', ''),
                            'opt_dir': opt_dir,
                            'entry_stock': buy_od.get('stock_price', 0.0),
                            'exit_stock': od.get('stock_price', 0.0),
                            'liq_chunks': buy_od.get('liq_chunks', 1),
                            'liq_reason': buy_od.get('liq_reason', 'N/A')
                        })

                        remaining_sell_qty -= match_qty
                        buy_od['qty'] -= match_qty
                        if buy_od['qty'] <= 0:
                            open_positions[sym].pop(0)

        return trades, open_positions

    def _build_logical_trade_view(self, df_trades: pd.DataFrame) -> pd.DataFrame:
        """把拆单成交行聚合为逻辑交易视图，避免 Iceberg chunk 在报表中看似“重叠开仓”。

        说明:
        - 同一 symbol/方向/原因/合约，并且 entry/exit 落在同一秒的行视为同一逻辑交易；
        - 价格按 qty 加权，PnL 直接求和；
        - roi 按聚合后的成本口径重算，便于和单行口径对齐。
        """
        if df_trades is None or df_trades.empty:
            return pd.DataFrame()

        df = df_trades.copy()
        df['entry_ts_key'] = df['entry_ts'].round(0).astype('int64')
        df['exit_ts_key'] = df['exit_ts'].round(0).astype('int64')

        group_cols = [
            'date', 'symbol', 'opt_dir', 'reason', 'contract',
            'entry_ts_key', 'exit_ts_key',
        ]

        agg = (
            df.groupby(group_cols, as_index=False)
              .apply(lambda g: pd.Series({
                  'entry_ts': float(g['entry_ts'].min()),
                  'exit_ts': float(g['exit_ts'].max()),
                  'entry_time': g.sort_values('entry_ts').iloc[0]['entry_time'],
                  'exit_time': g.sort_values('exit_ts').iloc[-1]['exit_time'],
                  'qty': int(g['qty'].sum()),
                  'entry_price': float(np.average(g['entry_price'], weights=g['qty'])),
                  'exit_price': float(np.average(g['exit_price'], weights=g['qty'])),
                  'entry_stock': float(np.average(g['entry_stock'], weights=g['qty'])),
                  'exit_stock': float(np.average(g['exit_stock'], weights=g['qty'])),
                  'pnl': float(g['pnl'].sum()),
                  'duration': float((g['exit_ts'].max() - g['entry_ts'].min()) / 60.0),
                  'liq_chunks': int(g['liq_chunks'].max()),
                  'liq_reason': str(g.iloc[0].get('liq_reason', 'N/A')),
                  'chunk_rows': int(len(g)),
              }))
              .reset_index(drop=True)
        )

        # 聚合后 ROI 统一按聚合成本重算 (包含双边手续费)
        entry_comm = agg['qty'] * COMMISSION_PER_CONTRACT
        agg_cost = (agg['entry_price'] * agg['qty'] * 100.0) + entry_comm
        agg['roi'] = np.where(agg_cost > 0.0, agg['pnl'] / agg_cost, 0.0)

        # 对齐原始 CSV 常见列顺序
        ordered_cols = [
            'date', 'symbol', 'entry_time', 'exit_time', 'pnl', 'roi', 'qty',
            'entry_price', 'exit_price', 'reason', 'duration',
            'entry_ts', 'exit_ts', 'contract', 'opt_dir',
            'entry_stock', 'exit_stock', 'liq_chunks', 'liq_reason', 'chunk_rows',
        ]
        return agg[ordered_cols].sort_values('exit_ts').reset_index(drop=True)

    def save_trades(self, filename="replay_trades_v8_1s.csv"):
        if not self.orders:
            print("\n⚠️ NO TRADES EXECUTED.")
            return

        trades, open_pos = self._match_trades()
        if not trades:
            print("\n⚠️ NO COMPLETED TRADES (Only open positions?).")
            # 也可以把 open positions 打印出来
            return
            
        # [Verification Log] 打印行情记录条数，用于核对回测完整性
        print(f"\n📊 Market history bars recorded: {len(self.market_history)}")
        
        df_trades = pd.DataFrame(trades)
        # 按平仓时间戳排序，保证 CSV 行序严格递增 (解决乱序问题)
        df_trades = df_trades.sort_values('exit_ts').reset_index(drop=True)
        from pathlib import Path
        log_dir = Path.home() / "quant_project/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        target_path = log_dir / filename
        df_trades.to_csv(target_path, index=False)
        print(f"\n💾 Trades saved to {target_path}")

        # 额外导出逻辑交易视图 (聚合拆单)
        logical_df = self._build_logical_trade_view(df_trades)
        if not logical_df.empty:
            logical_name = f"{target_path.stem}_logical{target_path.suffix}"
            logical_path = target_path.with_name(logical_name)
            logical_df.to_csv(logical_path, index=False)
            print(
                f"💾 Logical trades saved to {logical_path} "
                f"(raw_rows={len(df_trades)} -> logical_rows={len(logical_df)})"
            )
        
        self._print_daily_report(df_trades)
        self._print_summary_report(df_trades, open_pos)

    def _print_daily_report(self, df):
        print("\n" + "="*60)
        print("📅 DAILY PERFORMANCE REPORT")
        print("="*60)
        
        daily_stats = df.groupby('date').agg({
            'pnl': 'sum',
            'symbol': 'count',
            'roi': 'mean'
        }).rename(columns={'symbol': 'trades', 'roi': 'avg_roi'})
        
        daily_stats['equity'] = self.initial_capital + daily_stats['pnl'].cumsum()
        daily_stats['daily_ret'] = daily_stats['pnl'] / self.initial_capital
        
        print(f"{'Date':<12} | {'Trades':<6} | {'PnL ($)':<10} | {'Return':<8} | {'Equity':<10}")
        print("-" * 60)
        
        for date, row in daily_stats.iterrows():
            d_str = date.strftime('%Y-%m-%d')
            print(f"{d_str:<12} | {int(row['trades']):<6} | {row['pnl']:<10.1f} | {row['daily_ret']:<8.2%} | {row['equity']:<10.0f}")
        print("-" * 60)


    def _print_summary_report(self, df, open_pos):
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = df['pnl'].sum()
        final_equity = self.initial_capital + total_pnl
        total_ret = total_pnl / self.initial_capital
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # =========================================================
        # [新增] 计算年化夏普比率 (Annualized Sharpe Ratio)
        # =========================================================
        # 1. 按天聚合 PnL
        daily_pnl = df.groupby('date')['pnl'].sum()
        # 2. 计算每日净值曲线
        daily_equity = self.initial_capital + daily_pnl.cumsum()
        # 3. 计算日收益率
        daily_returns = daily_equity.pct_change()
        if not daily_pnl.empty:
            daily_returns.iloc[0] = daily_pnl.iloc[0] / self.initial_capital
            
        # 4. 年化夏普计算 (假设无风险利率 Rf = 0，年交易日 252)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        else:
            sharpe_ratio = 0.0

        # Max Drawdown Calculation
        df_sorted = df.sort_values('exit_ts')
        equity_curve = [self.initial_capital]
        curr_eq = self.initial_capital
        for pnl in df_sorted['pnl']:
            curr_eq += pnl
            equity_curve.append(curr_eq)
        
        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        drawdown = (eq_series - cummax) / cummax
        max_dd = drawdown.min()

        print("\n" + "="*80)
        print(f"🧠 Matrix Final Report (V8 Replay)")
        print("="*80)
        print(f"初始资金: ${self.initial_capital:,.0f}")
        print(f"最终权益: ${final_equity:,.0f}")
        print(f"总收益率: {total_ret:.2%}")
        print(f"最大回撤: {max_dd:.2%}")
        # print(f"夏普比率: {sharpe_ratio:.2f}")  # <--- 已废弃，改用 Decay Report
        print(f"执行延迟: {self.fill_delay} bar(s) [Original Model]")
        print("-" * 40)
        
        self._print_sharpe_decay_report(df)
        self._print_stock_sharpe_decay_report(df) 
        self._print_pure_directional_ic_report() # [New] 纯净 Alpha 质量诊断
        self._print_spread_stress_test_report(df) 
        self._print_liquidity_impact_report(df)   # [New]
        self._print_pnl_concentration_report(df)
        
        print(f"交易总数: {total_trades}")
        print(f"胜    率: {win_rate:.2%}")
        print(f"盈亏  比: {profit_factor:.2f}")
        print("-" * 80)
        
        if open_pos:
            count = sum(len(v) for v in open_pos.values())
            print(f"\n⚠️ {count} Positions still OPEN at end of replay.")

    def _calculate_sharpe_for_bar_delay(self, df_trades, delay_bars):
        """计算带 Bar 级延迟的信号衰减 (Signal Decay)"""
        delayed_trades = []
        for _, tr in df_trades.iterrows():
            # 【核心修复】进场和出场必须同时向后推迟 delay_bars！
            p_entry = self._get_price_at_bar_offset(tr['symbol'], tr['entry_ts'], tr['opt_dir'], 'BUY', delay_bars, price_type='mid')
            p_exit = self._get_price_at_bar_offset(tr['symbol'], tr['exit_ts'], tr['opt_dir'], 'SELL', delay_bars, price_type='mid')
            
            if p_entry and p_exit:
                new_pnl = (p_exit - p_entry) * tr['qty'] * 100
                delayed_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': new_pnl})
            else:
                # 缺失数据 (比如延迟后碰到了收盘) 视同保本或剥离
                pass
        
        df_delayed = pd.DataFrame(delayed_trades)
        rets = self._get_intraday_returns(df_delayed)
        return self._compute_sharpe(rets, is_intraday=True)

    def _get_daily_returns(self, df):
        if df.empty: return pd.Series()
        daily_pnl = df.groupby('date')['pnl'].sum()
        daily_equity = self.initial_capital + daily_pnl.cumsum()
        daily_returns = daily_equity.pct_change()
        # [Fix] 避免只有一行数据导致的 rets 为空
        if not daily_returns.empty:
            daily_returns.iloc[0] = daily_pnl.iloc[0] / self.initial_capital
        return daily_returns

    def _get_intraday_returns(self, df):
        """Use trade-level returns so single-day 1s replays can still report decay."""
        if df.empty:
            return pd.Series()
        df_sorted = df.sort_values('exit_ts').copy()
        df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
        equity_curve = self.initial_capital + df_sorted['cum_pnl']
        return equity_curve.pct_change().fillna(df_sorted['pnl'].iloc[0] / self.initial_capital)
        
    def _compute_sharpe(self, rets, is_intraday=True):
        if len(rets) > 1 and rets.std() > 1e-9:
            if is_intraday:
                # Same conservative trade-level annualization used by the 1m mock.
                return np.sqrt(252 * 50) * (rets.mean() / rets.std())
            return np.sqrt(252) * (rets.mean() / rets.std())
        return 0.0

    def _print_sharpe_decay_report(self, df_trades):
        print("\n📉 SHARPE DECAY ANALYSIS (Signal Half-Life)")
        print(f"{'Latency':<12} | {'Sharpe':<8} | {'Retained %':<12} | {'Status':<10}")
        print("-" * 55)
        
        base_sharpe = self._calculate_sharpe_for_bar_delay(df_trades, 0)
        bar_delays = [0, 1, 2, 3, 5, 10]
        
        results = []
        for d in bar_delays:
            s = self._calculate_sharpe_for_bar_delay(df_trades, d)
            
            # 只有当基准 Sharpe > 0 时, 计算留存率才有意义
            if base_sharpe > 0:
                retention = (s / base_sharpe * 100)
                retained_str = f"{retention:>10.1f}%"
            else:
                retained_str = "       N/A"
            
            # [New] Robustness Check (Sharpe > 2.0 at 1-2 bar delay)
            status = ""
            if d in [1, 2]:
                status = "✅ READY" if s > 2.0 else "⚠️ WEAK"
            elif d == 0:
                status = "SIGNAL"
                
            print(f"delay {d:<2} bar | {s:<8.2f} | {retained_str} | {status:<10}")
            results.append((d, s))
        
        # 计算 Signal Half-life
        half_life = "N/A"
        if base_sharpe > 0:
            for d, s in results:
                if s <= 0.5 * base_sharpe:
                    half_life = f"{d} bars"
                    break
        
        print("-" * 55)
        print(f"⏱️ Signal Half-life: {half_life}")
        
    def _print_stock_sharpe_decay_report(self, df_trades):
        """[New] 纯正股方向夏普衰减分析 (Pure Alpha Quality Check)"""
        print("\n📈 STOCK-ONLY SHARPE DECAY (Directional Alpha Quality)")
        print(f"{'Latency':<12} | {'Sharpe':<8} | {'Retained %':<12} | {'Status':<10}")
        print("-" * 55)
        
        base_sharpe = self._calculate_stock_sharpe_for_bar_delay(df_trades, 0)
        bar_delays = [0, 1, 2, 3, 5, 10]
        
        results = []
        for d in bar_delays:
            s = self._calculate_stock_sharpe_for_bar_delay(df_trades, d)
            retention = (s / base_sharpe * 100) if base_sharpe > 0 else 0.0
            
            status = ""
            if d in [1, 2]:
                status = "✅ PURE" if s > 1.5 else "⚠️ NOISY"
            elif d == 0:
                status = "ALPHA"
                
            print(f"delay {d:<2} bar | {s:<8.2f} | {retention:>10.1f}% | {status:<10}")
            results.append((d, s))
            
        print("-" * 55)
        print("💡 TIP: High Stock Sharpe means your model predicts stock direction well,")
        print("   even if option point-spread kills the final return.")

    def _calculate_stock_sharpe_for_bar_delay(self, df_trades, delay_bars):
        """计算纯正股波动的夏普 (假设持有 1 股正股的等效多空盈亏)"""
        stock_trades = []
        for _, tr in df_trades.iterrows():
            # 获取延迟后的正股价格
            s_entry = self._get_stock_price_at_bar_offset(tr['symbol'], tr['entry_ts'], delay_bars)
            s_exit = self._get_stock_price_at_bar_offset(tr['symbol'], tr['exit_ts'], delay_bars)
            
            if s_entry and s_exit and s_entry > 0:
                # 定向收益率 (多头为正，空头为负)
                direction = 1 if tr['opt_dir'] == 'CALL' else -1
                pnl_pct = (s_exit - s_entry) / s_entry * direction
                # 将收益率映射为虚拟 PnL 供 _get_daily_returns 计算 (这里乘以 initial_capital 放大)
                virtual_pnl = pnl_pct * self.initial_capital * 0.1 # 假设 10% 杠杆
                stock_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': virtual_pnl})
        
        df_stock = pd.DataFrame(stock_trades)
        rets = self._get_intraday_returns(df_stock)
        return self._compute_sharpe(rets, is_intraday=True)

    def _print_pure_directional_ic_report(self):
        """[Premium] 纯净方向性 IC 诊断 (Pure Directional IC Benchmark)
        脱离期权执行和资金曲线，直接衡量 Alpha 信号对正股波动的预测能力。
        """
        if not self.market_history or len(self.market_history) < 60:
            return

        print("\n🎯 PURE DIRECTIONAL IC BENCHMARK (Universe Alpha Quality)")
        print(f"{'Horizon':<12} | {'Rank IC':<10} | {'Prediction':<10} | {'Status':<10}")
        print("-" * 55)

        # 构建长表数据用于向量化计算
        data_list = []
        for ts, snapshot in self.market_history:
            for sym, d in snapshot.items():
                if d['a'] != 0: # 只统计有信号的样本
                    data_list.append({'ts': ts, 'sym': sym, 'alpha': d['a'], 'price': d['s']})
        
        if not data_list:
            print("⚠️ No valid Alpha signals recorded for IC analysis.")
            return
            
        df = pd.DataFrame(data_list)
        # 1s 回放会把分钟 alpha 广播到每一秒。如果直接 shift(-k)，
        # "1 min fwd" 会被误算成 1 秒 forward。这里先降采样到分钟网格，
        # 让 IC 口径和分钟版保持一致。
        df['minute_ts'] = (df['ts'].astype(np.int64) // 60) * 60
        df = (
            df.sort_values('ts')
              .drop_duplicates(subset=['minute_ts', 'sym'], keep='first')
              .drop(columns=['ts'])
              .rename(columns={'minute_ts': 'ts'})
        )
        # 转换日期和符号索引，方便计算 Forward Return
        df = df.pivot(index='ts', columns='sym', values=['alpha', 'price'])
        
        horizons = [1, 5, 15, 30, 60] # 分钟；上面已经降采样为 1 bar = 1 min
        
        results = []
        for k in horizons:
            # 计算 k 分钟后的正股收益率
            # 注意：market_history 的密度可能不均匀，但通常是 1min 一条
            fwd_price = df['price'].shift(-k)
            fwd_ret = (fwd_price - df['price']) / df['price']
            
            # 计算截面相关性 (Spearman Rank IC)
            # 对每一行（每一个时间点）计算 Alpha 和 Fwd_Ret 的相关性
            ics = df['alpha'].corrwith(fwd_ret, axis=1, method='spearman')
            mean_ic = ics.mean()
            
            status = "💎 PURE" if mean_ic > 0.05 else "⚠️ NOISY"
            pred_type = "BULLISH" if mean_ic > 0 else "BEARISH"
            
            print(f"{k:<2} min fwd  | {mean_ic:>10.4f} | {pred_type:<10} | {status:<10}")
            results.append((k, mean_ic))

        print("-" * 55)
        print("💡 TIP: Pure IC > 0.05 is the 'Holy Grail' of High-Freq Alpha.")
        print("   If IC is high but PnL is negative, focus on SPREAD & GREEKS.")
        print("-" * 55)

    def _get_stock_price_at_bar_offset(self, symbol, ts, bar_offset):
        if not self.market_history: return None
        import bisect
        times = [x[0] for x in self.market_history]
        idx = bisect.bisect_left(times, ts)
        target_idx = idx + bar_offset
        if target_idx >= len(self.market_history): target_idx = len(self.market_history) - 1
        _, snapshot = self.market_history[target_idx]
        data = snapshot.get(symbol)
        return data.get('s', 0.0) if data else None
        
    def _quote_snapshot_at_time(self, symbol, ts, opt_dir):
        """Return option mid/bid/ask from the latest snapshot at or before ts."""
        if not self.market_history:
            return None
        import bisect
        times = [x[0] for x in self.market_history]
        idx = bisect.bisect_right(times, ts) - 1
        if idx < 0:
            return None
        idx = min(idx, len(self.market_history) - 1)
        best_ts, snapshot = self.market_history[idx]
        if abs(best_ts - ts) > 120:
            return None
        data = snapshot.get(symbol)
        if not data:
            return None
        if opt_dir == 'CALL':
            bid, ask, last = float(data.get('cb', 0.0)), float(data.get('ca', 0.0)), float(data.get('c', 0.0))
        else:
            bid, ask, last = float(data.get('pb', 0.0)), float(data.get('pa', 0.0)), float(data.get('p', 0.0))
        if bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
        else:
            mid = last
        return {'bid': bid, 'ask': ask, 'mid': mid, 'last': last}

    def _build_spread_stress_trades(self, df_trades, spread_fraction: float):
        """Reprice trades by moving entry/exit from mid toward ask/bid by fraction."""
        stress_trades = []
        covered = 0
        for _, tr in df_trades.iterrows():
            entry_q = self._quote_snapshot_at_time(tr['symbol'], tr['entry_ts'], tr['opt_dir'])
            exit_q = self._quote_snapshot_at_time(tr['symbol'], tr['exit_ts'], tr['opt_dir'])
            if entry_q and exit_q and entry_q['bid'] > 0 and entry_q['ask'] > 0 and exit_q['bid'] > 0 and exit_q['ask'] > 0:
                covered += 1
                entry_p = entry_q['mid'] + max(0.0, entry_q['ask'] - entry_q['mid']) * spread_fraction
                exit_p = exit_q['mid'] - max(0.0, exit_q['mid'] - exit_q['bid']) * spread_fraction
                entry_commission = tr['qty'] * COMMISSION_PER_CONTRACT
                exit_commission = tr['qty'] * COMMISSION_PER_CONTRACT
                cost = entry_p * tr['qty'] * 100 + entry_commission
                proceeds = exit_p * tr['qty'] * 100 - exit_commission
                pnl = proceeds - cost
                stress_trades.append({
                    'date': tr['date'],
                    'exit_ts': tr['exit_ts'],
                    'pnl': pnl,
                    'base_pnl': tr['pnl'],
                    'reason': tr.get('reason', 'N/A'),
                    'duration': tr.get('duration', 0.0),
                })
            else:
                stress_trades.append({
                    'date': tr['date'],
                    'exit_ts': tr.get('exit_ts', 0.0),
                    'pnl': tr['pnl'],
                    'base_pnl': tr['pnl'],
                    'reason': tr.get('reason', 'N/A'),
                    'duration': tr.get('duration', 0.0),
                })
        return pd.DataFrame(stress_trades), covered

    def _print_spread_stress_attribution(self, df_stress: pd.DataFrame, label: str):
        if df_stress is None or df_stress.empty:
            return
        df = df_stress.copy()
        df['reason_root'] = df['reason'].astype(str).str.split('|').str[0].str.split(':').str[0].str.slice(0, 28)
        df['pnl_decay'] = df['pnl'] - df['base_pnl']
        reason = (
            df.groupby('reason_root')
              .agg(
                  trades=('pnl', 'size'),
                  base_pnl=('base_pnl', 'sum'),
                  stress_pnl=('pnl', 'sum'),
                  decay=('pnl_decay', 'sum'),
                  avg_duration=('duration', 'mean'),
              )
              .sort_values('decay')
              .head(8)
              .reset_index()
        )
        print(f"\n🧪 {label} stress attribution by exit reason")
        print(
            reason.round({
                'base_pnl': 1,
                'stress_pnl': 1,
                'decay': 1,
                'avg_duration': 2,
            }).to_string(index=False)
        )

        bins = [-0.001, 1, 3, 5, 10, 30, 10_000]
        labels = ['<=1m', '1-3m', '3-5m', '5-10m', '10-30m', '>30m']
        df['duration_bucket'] = pd.cut(df['duration'].astype(float), bins=bins, labels=labels)
        bucket = (
            df.groupby('duration_bucket', observed=True)
              .agg(
                  trades=('pnl', 'size'),
                  base_pnl=('base_pnl', 'sum'),
                  stress_pnl=('pnl', 'sum'),
                  decay=('pnl_decay', 'sum'),
              )
              .reset_index()
        )
        print(f"\n🧪 {label} stress attribution by holding time")
        print(bucket.round({'base_pnl': 1, 'stress_pnl': 1, 'decay': 1}).to_string(index=False))

    def _print_spread_stress_test_report(self, df_trades):
        """Stress test incremental spread cost instead of treating full spread as baseline."""
        base_pnl = float(df_trades['pnl'].sum()) if not df_trades.empty else 0.0
        base_ret = base_pnl / self.initial_capital
        base_sharpe = self._compute_sharpe(self._get_intraday_returns(df_trades), is_intraday=True)
        
        print("\n🔥 SPREAD STRESS TEST (Incremental Spread Cost)")
        print(f"{'Condition':<20} | {'PnL($)':>10} | {'Return':>9} | {'Sharpe':>8} | {'Win':>7} | {'Status':<10}")
        print("-" * 84)
        print(f"{'Actual ledger':<20} | {base_pnl:>10.1f} | {base_ret:>8.2%} | {base_sharpe:>8.2f} | {(df_trades['pnl'] > 0).mean():>6.1%} | BASELINE")
        coverage_msg = ""
        attribution_df = None
        attribution_label = ""
        for label, frac in (("+25% spread", 0.25), ("+50% spread", 0.50), ("Full ask→bid", 1.00)):
            df_stress, covered = self._build_spread_stress_trades(df_trades, frac)
            stress_pnl = float(df_stress['pnl'].sum()) if not df_stress.empty else 0.0
            stress_ret = stress_pnl / self.initial_capital
            stress_sharpe = self._compute_sharpe(self._get_intraday_returns(df_stress), is_intraday=True)
            stress_win_rate = (df_stress['pnl'] > 0).mean() if not df_stress.empty else 0.0
            pnl_retained = stress_pnl / base_pnl if abs(base_pnl) > 1e-9 else 0.0
            if frac < 1.0:
                status = "✅ OK" if stress_pnl > 0 and pnl_retained >= 0.5 else "⚠️ SENSITIVE"
            else:
                status = "💎 ROBUST" if stress_pnl > 0 and pnl_retained >= 0.5 else "⚠️ WORST"
            print(f"{label:<20} | {stress_pnl:>10.1f} | {stress_ret:>8.2%} | {stress_sharpe:>8.2f} | {stress_win_rate:>6.1%} | {status}")
            if not coverage_msg:
                coverage = covered / len(df_trades) if len(df_trades) else 0.0
                coverage_msg = f"Coverage: {covered}/{len(df_trades)} ({coverage:.1%})"
            if frac == 0.25:
                attribution_df = df_stress
                attribution_label = label
        print("-" * 84)
        print(coverage_msg or "Coverage: N/A")
        print("💡 Use +25%/+50% as practical stress; Full ask→bid is a worst-case bound.")
        print("-" * 50)
        self._print_spread_stress_attribution(attribution_df, attribution_label)

    def _print_liquidity_impact_report(self, df_trades):
        """[Premium] 流动性影响分析 (Liquidity Impact Analysis)
        分析资金规模对盘口的影响，统计拆单触发频率及平均拆单金额。
        """
        total = len(df_trades)
        if total == 0: return
        
        # 统计拆单情况
        split_df = df_trades[df_trades['liq_chunks'] > 1]
        split_count = len(split_df)
        split_ratio = (split_count / total * 100)
        
        # 计算开仓名义价值 (Notional)
        # 注意：df_trades 里的 pnl 是计算好的，我们需要开仓时的名义本金
        # 我们之前在 _match_trades 里没存 entry_notional，但有 entry_price 和 qty
        entry_notional = df_trades['entry_price'] * df_trades['qty'] * 100
        avg_notional = entry_notional.mean()
        
        avg_split_notional = 0
        if split_count > 0:
            avg_split_notional = (split_df['entry_price'] * split_df['qty'] * 100).mean()
            
        max_chunks = df_trades['liq_chunks'].max()
        
        print("\n💧 LIQUIDITY IMPACT ANALYSIS (Capacity Check)")
        print(f"{'Metric':<25} | {'Value':<15}")
        print("-" * 45)
        print(f"{'Total Trades':<25} | {total:<15}")
        print(f"{'Splitted Trades (Iceberg)':<25} | {split_count:<15} ({split_ratio:.1f}%)")
        print(f"{'Average Entry Capital':<25} | ${avg_notional:,.0f}")
        
        if split_count > 0:
            print(f"{'Avg Capital triggering Split':<25} | ${avg_split_notional:,.0f}")
            print(f"{'Max Chunks Observed':<25} | {max_chunks:<15}")
        else:
            print(f"{'Cap Verification':<25} | ✅ NO IMPACT (Under current exposure)")
            
        print("-" * 45)
        if split_ratio > 30:
            print("⚠️ CAPACITY WARNING: High frequency of splitting. Reduce POSITION_RATIO or switch to CORE symbols.")
        elif split_count > 0:
            print("💡 CAPACITY OK: Occasional splitting detected. Market depth is sufficient.")
        else:
            print("💎 CAPACITY PERFECT: All orders filled in single chunks.")
        print("-" * 45)

    def _print_pnl_concentration_report(self, df):
        """收益集中度测试 (PnL Concentration Test)"""
        if df.empty: return
        
        pos_pnls = df[df['pnl'] > 0]['pnl'].sort_values(ascending=False)
        total_profit = pos_pnls.sum()
        
        if total_profit <= 0:
            print("\n📊 PnL CONCENTRATION TEST: No profitable trades to analyze.")
            return

        print("\n📊 PnL CONCENTRATION TEST (Robustness Check)")
        print(f"{'Metric':<20} | {'Value':<10} | {'Status':<10}")
        print("-" * 50)
        
        # 1. Top N Trades Concentration
        top1_ratio = pos_pnls.iloc[0] / total_profit if len(pos_pnls) >= 1 else 0
        top5_ratio = pos_pnls.iloc[:5].sum() / total_profit if len(pos_pnls) >= 5 else 1.0
        top10_ratio = pos_pnls.iloc[:10].sum() / total_profit if len(pos_pnls) >= 10 else 1.0
        
        def get_status(ratio, threshold, min_count, actual_count):
            if actual_count < min_count: return "N/A"
            return "⚠️ FAIL" if ratio > threshold else "✅ PASS"

        print(f"{'Top 1 Trade %':<20} | {top1_ratio:>9.1%} | {get_status(top1_ratio, 0.30, 1, len(pos_pnls))}")
        print(f"{'Top 5 Trades %':<20} | {top5_ratio:>9.1%} | {get_status(top5_ratio, 0.70, 5, len(pos_pnls))}")
        print(f"{'Top 10 Trades %':<20} | {top10_ratio:>9.1%} | {get_status(top10_ratio, 0.85, 10, len(pos_pnls))}")
        
        # 2. HHI (Herfindahl-Hirschman Index)
        # HHI = 1 表示完全集中在 1 笔交易，接近 0 表示极度分散
        hhi = ((pos_pnls / total_profit) ** 2).sum()
        hhi_status = "✅ PASS" if hhi < 0.25 else "⚠️ WARNING"
        print(f"{'Profit HHI Index':<20} | {hhi:>9.2f} | {hhi_status}")
        
        print("-" * 50)
        if top5_ratio > 0.70 or hhi > 0.25:
            print("❌ DANGER: Your profit is too concentrated in a few outlier trades.")
            print("   The strategy might fail in live trading due to 'Luck' dependency.")
        else:
            print("💎 HEALTHY: Profit is well-distributed across many trades.")
        print("-" * 50)

    def _print_summary_report_back(self, df, open_pos):
        total_trades = len(df)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = df['pnl'].sum()
        final_equity = self.initial_capital + total_pnl
        total_ret = total_pnl / self.initial_capital
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Max Drawdown Calculation
        df_sorted = df.sort_values('exit_ts')
        equity_curve = [self.initial_capital]
        curr_eq = self.initial_capital
        for pnl in df_sorted['pnl']:
            curr_eq += pnl
            equity_curve.append(curr_eq)
        
        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        drawdown = (eq_series - cummax) / cummax
        max_dd = drawdown.min()

        print("\n" + "="*80)
        print(f"🧠 Matrix Final Report (V8 Replay)")
        print("="*80)
        print(f"初始资金: ${self.initial_capital:,.0f}")
        print(f"最终权益: ${final_equity:,.0f}")
        print(f"总收益率: {total_ret:.2%}")
        print(f"最大回撤: {max_dd:.2%}")
        print("-" * 40)
        print(f"交易总数: {total_trades}")
        print(f"胜    率: {win_rate:.2%}")
        print(f"盈亏  比: {profit_factor:.2f}")
        print("-" * 80)
        
        if open_pos:
            count = sum(len(v) for v in open_pos.values())
            print(f"\n⚠️ {count} Positions still OPEN at end of replay.")

 
