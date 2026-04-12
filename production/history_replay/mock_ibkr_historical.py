import redis
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time
from config import COMMISSION_PER_CONTRACT, EXECUTION_DELAY_BARS, EXECUTION_DELAY_SECONDS
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
        self.execution_delay_bars = EXECUTION_DELAY_BARS
        self.execution_delay_seconds = EXECUTION_DELAY_SECONDS
        self.fill_delay = (
            f"{self.execution_delay_seconds}s"
            if self.execution_delay_seconds > 0
            else f"{self.execution_delay_bars} bar(s)"
        )
        self.ib = self._create_mock_ib()



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
        # 🧪 [Debug] 核实引擎是否连接到了本柜台实例
        print(f"🧪 [MockIBKR Debug] Order Received: {getattr(contract, 'symbol', 'Unknown')} | {action} | qty:{qty} | price:{lmt_price}")
        
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
        idx = bisect.bisect_left(times, ts)
        
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
        idx = bisect.bisect_left(times, ts)
        
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
        """[终极对齐版] 绝对忠实的 FIFO 撮合，禁止任何形式的价格篡改"""
        open_positions = {} 
        trades = []
        
        from config import COMMISSION_PER_CONTRACT
        
        sorted_orders = sorted(self.orders, key=lambda x: x['ts'])
        
        for od in sorted_orders:
            sym = od['symbol']
            # print(f"DEBUG_MATCH: Processing {od['action']} for {sym} | qty:{od['qty']} | ts:{od['ts']}")
            
            if od['action'] == 'BUY':
                if sym not in open_positions: open_positions[sym] = []
                open_positions[sym].append(od)
                
            elif od['action'] == 'SELL':
                if sym in open_positions and open_positions[sym]:
                    remaining_sell_qty = od['qty']
                    while remaining_sell_qty > 0 and open_positions[sym]:
                        buy_od = open_positions[sym][0]
                        match_qty = min(buy_od['qty'], remaining_sell_qty)
                        
                        # 默认价格
                        entry_price = buy_od['price']
                        exit_price = od['price']
                        opt_dir = buy_od.get('opt_dir', 'CALL')

                        # =========================================================
                        # 🚀 [时间机器] 时延盘口重定价
                        # =========================================================
                        delay_bars = getattr(self, 'execution_delay_bars', 0)
                        delay_secs = getattr(self, 'execution_delay_seconds', 0)
                        
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

                        # 严格比例扣分手续费
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
                            'contract': buy_od.get('contract', sym),
                            'opt_dir': opt_dir,
                            'entry_stock': buy_od.get('stock_price', 0),
                            'exit_stock': od.get('stock_price', 0),
                            'liq_chunks': od.get('liq_chunks', 1),
                            'liq_reason': od.get('liq_reason', 'N/A')
                        })
                        
                        # 更新库存
                        remaining_sell_qty -= match_qty
                        buy_od['qty'] -= match_qty
                        if buy_od['qty'] <= 0:
                            open_positions[sym].pop(0)
        # [Diagnostic] Final check for orphans
        for sym, pos_list in open_positions.items():
            if pos_list:
                total_qty = sum(p['qty'] for p in pos_list)
                if total_qty > 0.001:
                    print(f"DEBUG: Symbol {sym} has {len(pos_list)} unmatched BUY parts, total qty {total_qty}")

        return trades, open_positions

    def save_trades(self, filename="replay_trades_stable.csv"):
        if not self.orders:
            print("\n⚠️ NO TRADES EXECUTED.")
            return

        trades, open_pos = self._match_trades()
        if not trades:
            print("\n⚠️ NO COMPLETED TRADES (Only open positions?).")
            return
            
        # [Verification Log] 打印行情记录条数，用于核对回测完整性
        print(f"\n📊 Market history bars recorded: {len(self.market_history)}")
        
        df_trades = pd.DataFrame(trades)
        
        # 🚀 [去重逻辑] 彻底根除由于异步任务重叠产生的重复交易条目，防止分析端 pivot 崩溃
        if not df_trades.empty:
            orig_len = len(df_trades)
            # 使用关键字段 (entry_ts, exit_ts, symbol) 进行去重，确保每笔交易唯一
            df_trades = df_trades.drop_duplicates(subset=['entry_ts', 'exit_ts', 'symbol']).reset_index(drop=True)
            if len(df_trades) < orig_len:
                print(f"🛡️ [去重防护] 过滤掉 {orig_len - len(df_trades)} 条重复交易记录。")
                
        # 按平仓时间戳排序，保证 CSV 行序严格递增 (解决乱序问题)
        df_trades = df_trades.sort_values('exit_ts').reset_index(drop=True)
        from pathlib import Path
        log_dir = Path.home() / "quant_project/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        target_path = log_dir / filename
        df_trades.to_csv(target_path, index=False)
        print(f"\n💾 Trades saved to {target_path}")
        
        self._print_daily_report(df_trades)
        self._print_summary_report(df_trades, open_pos)
        
        # 🚀 [巅峰诊断复原] 激活高级统计指标
        self._print_sharpe_decay_report(df_trades)
        self._print_pure_directional_ic_report()
        self._print_spread_stress_test_report(df_trades)
        self._print_pnl_concentration_report(df_trades)

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
        # [新增] 计算日内高频夏普比率 (Intraday Sharpe Ratio)
        # =========================================================
        # 🟢 [🔥 修正] 直接使用日内收益序列计算，支持单日回测
        intraday_rets = self._get_intraday_returns(df)
        sharpe_ratio = self._compute_sharpe(intraday_rets, is_intraday=True)

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
        print(f"年化夏普: {sharpe_ratio:.2f}")
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
        rets = self._get_daily_returns(df_delayed)
        return self._compute_sharpe(rets)

    def _get_daily_returns(self, df):
        if df.empty: return pd.Series()
        # 兼容性处理：如果只有一天的数据，尝试按分钟或笔数计算
        daily_pnl = df.groupby('date')['pnl'].sum()
        daily_equity = self.initial_capital + daily_pnl.cumsum()
        return daily_equity.pct_change().fillna(daily_pnl.iloc[0]/self.initial_capital)
        
    def _get_intraday_returns(self, df):
        """[High-Freq] 获取日内分钟级收益率序列，用于精准计算单日 Sharpe"""
        if df.empty: return pd.Series()
        # 按平仓时间排序
        df_sorted = df.sort_values('exit_ts').copy()
        df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
        
        # 将成交序列映射到日内时间轴 (假设每个 trade 是一个 tick)
        # 对于真正精确的分钟级，需要重采样。这里先使用 trade-level 作为替代
        equity_curve = self.initial_capital + df_sorted['cum_pnl']
        rets = equity_curve.pct_change().fillna(df_sorted['pnl'].iloc[0]/self.initial_capital)
        return rets

    def _compute_sharpe(self, rets, is_intraday=True):
        if len(rets) <= 1: return 0.0
        # 如果是日内交易级数据，使用高频年化系数 (252 * 预计日交易量)
        # 为了保守起见，暂时统一使用 252 结合日内波幅进行粗略年化
        # 或者如果有更多天数据，仍按 252 计算
        factor = np.sqrt(252 * (390 if is_intraday else 1)) 
        mu = rets.mean()
        std = rets.std()
        if std > 1e-9:
            return np.sqrt(252) * (mu / std) if not is_intraday else np.sqrt(252 * 50) * (mu / std) # 粗略估算
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
                stock_trades.append({'date': tr['date'], 'pnl': virtual_pnl})
        
        df_stock = pd.DataFrame(stock_trades)
        rets = self._get_daily_returns(df_stock)
        return self._compute_sharpe(rets)

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
        # 🚀 [Fix] Deduplicate by (ts, sym) to prevent pivot crash
        df = df.drop_duplicates(subset=['ts', 'sym'], keep='last')
        # 转换日期和符号索引，方便计算 Forward Return
        df = df.pivot(index='ts', columns='sym', values=['alpha', 'price'])
        
        horizons = [1, 5, 15, 30, 60] # 分钟 (假设 1 bar = 1 min)
        
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
        
    def _print_spread_stress_test_report(self, df_trades):
        """[Premium] 全点差压力测试 (Spread Stress Test)
        模拟最极端情况：在 Entry 时永远以 Ask 买入，Exit 时永远以 Bid 卖出。
        """
        stress_trades = []
        moderate_trades = [] # 模拟计提 20% 滑点

        for _, tr in df_trades.iterrows():
            # 获取开仓时刻的盘口 (显式使用 best/spread)
            entry_p = self._get_price_at_time(tr['symbol'], tr['entry_ts'], tr['opt_dir'], 'BUY', price_type='best')
            exit_p = self._get_price_at_time(tr['symbol'], tr['exit_ts'], tr['opt_dir'], 'SELL', price_type='best')
            
            # 基础 Mid 价 (对照组)
            entry_mid = self._get_price_at_time(tr['symbol'], tr['entry_ts'], tr['opt_dir'], 'BUY', price_type='mid')
            exit_mid = self._get_price_at_time(tr['symbol'], tr['exit_ts'], tr['opt_dir'], 'SELL', price_type='mid')

            if entry_p and exit_p and entry_mid and exit_mid:
                # 1. Full Stress (100% Spread Impact)
                pnl_full = (exit_p - entry_p) * tr['qty'] * 100
                stress_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': pnl_full})
                
                # 2. Moderate Slippage (计提 20% 的买卖价差作为滑点)
                # 滑点补偿后的买入价 = Mid + 0.2 * (Ask - Mid)
                # 滑点补偿后的卖出价 = Mid - 0.2 * (Mid - Bid)
                slippage_in = abs(entry_p - entry_mid) * 0.2
                slippage_out = abs(exit_p - exit_mid) * 0.2
                p_entry_mod = entry_mid + slippage_in
                p_exit_mod = exit_mid - slippage_out
                pnl_mod = (p_exit_mod - p_entry_mod) * tr['qty'] * 100
                moderate_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': pnl_mod})
            else:
                stress_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': tr['pnl']})
                moderate_trades.append({'date': tr['date'], 'exit_ts': tr['exit_ts'], 'pnl': tr['pnl']})
                
        df_stress = pd.DataFrame(stress_trades)
        df_mod = pd.DataFrame(moderate_trades)
        
        stress_sharpe = self._compute_sharpe(self._get_intraday_returns(df_stress), is_intraday=True)
        mod_sharpe = self._compute_sharpe(self._get_intraday_returns(df_mod), is_intraday=True)
        
        status_full = "💎 ROBUST" if stress_sharpe > 2.0 else "⚠️ FRAGILE"
        status_mod = "💎 HEALTHY" if mod_sharpe > 2.0 else "⚠️ WEAK"
        
        print("\n🔥 SPREAD STRESS TEST (Entry@Ask, Exit@Bid)")
        print(f"{'Condition':<20} | {'Sharpe':<10} | {'Status':<10}")
        print("-" * 50)
        # [🔥 修正] 基准 Sharpe 同样使用日内口径
        std_rets = self._get_intraday_returns(df_trades)
        std_sharpe = self._compute_sharpe(std_rets, is_intraday=True)
        print(f"{'Standard (Mid/Last)':<20} | {std_sharpe:>10.2f} | BASELINE")
        print(f"{'Moderate (20% Slippage)':<20} | {mod_sharpe:>10.2f} | {status_mod}")
        print(f"{'Extreme Full Spread':<20} | {stress_sharpe:>10.2f} | {status_full}")
        print("-" * 55)
        
        if stress_sharpe > 2.0:
            print("👑 ALPHA CONFIRMED: Strategy survives full spread impact.")
        else:
            print("🚨 ALPHA RISK: Strategy may crumble under real-world slippage.")
        print("-" * 50)

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

 
