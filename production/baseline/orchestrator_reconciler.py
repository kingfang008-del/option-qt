import asyncio
import logging
import json
import time
import os
from config import TRADING_ENABLED
from runtime_trading_controls import get_runtime_trading_enabled

logger = logging.getLogger("V8_Orchestrator.Reconciler")

class OrchestratorReconciler:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.reconcile_interval = 10  # 每 10 秒对账一次 (避免把 IBKR 接口打爆)
        self.cash_warn_threshold = float(os.environ.get("CASH_RECON_WARN_USD", "1500"))
        self.cash_pause_threshold = float(os.environ.get("CASH_RECON_PAUSE_USD", "5000"))
        self._last_cash_warn_ts = 0.0
        self._last_cash_pause_ts = 0.0

    def _runtime_trading_enabled(self) -> bool:
        try:
            return bool(
                get_runtime_trading_enabled(
                    default_value=TRADING_ENABLED,
                    r=getattr(self.orch, 'r', None),
                )
            )
        except Exception:
            return bool(TRADING_ENABLED)

    async def run_reconciliation_loop(self):
        """
        [防掉单对账器] 后台守护协程
        """
        logger.info("🛡️ [Reconciler] 防掉单对账器已启动，将严密监控底层券商仓位...")
        
        while True:
            await asyncio.sleep(self.reconcile_interval)
            
            # 仅在实盘且允许交易的模式下进行物理对账
            if self.orch.mode != 'realtime' or not self._runtime_trading_enabled():
                continue
                
            # 确保 IBKR 连接正常
            if not hasattr(self.orch, 'ibkr') or not hasattr(self.orch.ibkr, 'ib') or not self.orch.ibkr.ib.isConnected():
                logger.debug("⚠️ [Reconciler] IBKR 未连接，跳过本轮对账。")
                continue

            try:
                await self._perform_reconciliation()
            except Exception as e:
                logger.error(f"❌ [Reconciler] 对账过程发生异常: {e}", exc_info=True)

    async def _perform_reconciliation(self):
        # 1. 获取券商底层真实仓位 (Ground Truth)
        # ib.positions() 会返回当前账户的所有持仓列表
        # 注意：在异步环境下，ib.positions() 是即时返回缓存的，实际由内部独立连接维护同步
        real_positions = self.orch.ibkr.ib.positions()
        
        # 2. 统计真实持仓 (聚合期权到对应的正股 Symbol)
        broker_state = {}
        for pos in real_positions:
            # pos.contract.symbol 在期权合约中通常代表底层正股 (如 'NVDA')
            sym = pos.contract.symbol
            qty = pos.position # 多头为正，空头为负
            
            if qty != 0:
                # 累加该标的下的所有期权头寸数量 (注意：此处假设每个 symbol 下通常持有一种期权)
                # 如果存在多腿组合，逻辑可能需要更精细，目前按 Symbol 极简对齐
                if sym not in broker_state:
                    broker_state[sym] = 0
                broker_state[sym] += qty

        # 3. 交叉比对本地内存状态
        for sym, st in self.orch.states.items():
            # 本地记录的理论数量 (多头为正，空头为负)
            local_qty = st.qty if st.position >= 0 else -st.qty
            
            # 券商返回的真实数量
            broker_qty = broker_state.get(sym, 0)

            # 🚨 【免死金牌】如果该品种正在执行异步下单/撤单，跳过对账！
            # 因为此时本地账本和券商账本存在合理的时间差。
            if getattr(st, 'is_pending', False):
                continue

            # 4. 捕捉分叉 (State Divergence)
            if local_qty != broker_qty:
                logger.critical(f"🚨 [严重分叉] {sym} 账本不一致! 内存记录: {local_qty}手 vs 券商真实: {broker_qty}手")
                
                # 执行自愈逻辑 (Auto-Healing)
                await self._auto_heal_divergence(sym, st, local_qty, broker_qty)

        await self._perform_cash_reconciliation()

    async def _perform_cash_reconciliation(self):
        """Cash guard: reconcile local mock_cash against broker available funds."""
        if not hasattr(self.orch, 'ibkr') or not hasattr(self.orch.ibkr, 'get_account_balance'):
            return
        try:
            broker_cash = float(await self.orch.ibkr.get_account_balance() or 0.0)
        except Exception:
            return
        if broker_cash <= 0:
            return

        local_cash = float(getattr(self.orch, 'mock_cash', 0.0) or 0.0)
        locked_cash = 0.0
        pending_cnt = 0
        for st in getattr(self.orch, 'states', {}).values():
            try:
                locked_cash += max(0.0, float(getattr(st, 'locked_cash', 0.0) or 0.0))
            except Exception:
                pass
            if bool(getattr(st, 'is_pending', False)):
                pending_cnt += 1
        local_total = local_cash + locked_cash
        diff_cash = broker_cash - local_cash
        diff_total = broker_cash - local_total
        abs_diff = abs(diff_cash)
        now = time.time()

        if abs_diff >= self.cash_warn_threshold and (now - self._last_cash_warn_ts) >= 30:
            self._last_cash_warn_ts = now
            logger.warning(
                f"⚠️ [Cash-Reconcile] broker=${broker_cash:,.2f} local_cash=${local_cash:,.2f} "
                f"locked=${locked_cash:,.2f} local_total=${local_total:,.2f} "
                f"diff_cash={diff_cash:+,.2f} diff_total={diff_total:+,.2f} pending={pending_cnt}"
            )

        # Hard guard: large divergence without pending orders means local cash is likely corrupted.
        if abs_diff >= self.cash_pause_threshold and pending_cnt == 0:
            if (now - self._last_cash_pause_ts) >= 60:
                self._last_cash_pause_ts = now
                self.orch.trading_paused = True
                logger.critical(
                    f"🛑 [Cash-Reconcile-HardGuard] Trading paused: broker=${broker_cash:,.2f}, "
                    f"local_cash=${local_cash:,.2f}, diff={diff_cash:+,.2f} (threshold={self.cash_pause_threshold:.2f})"
                )
                try:
                    self.orch.accounting._emit_trade_log({
                        'ts': now,
                        'symbol': 'SYSTEM',
                        'action': 'CASH_DIVERGENCE',
                        'side': 'SYS_GUARD',
                        'qty': 0,
                        'price': 0.0,
                        'stock_price': 0.0,
                        'strategy_note': json.dumps({
                            'reason': 'CASH_RECONCILE_HARD_GUARD',
                            'broker_cash': broker_cash,
                            'local_cash': local_cash,
                            'locked_cash': locked_cash,
                            'diff_cash': diff_cash,
                            'diff_total': diff_total,
                        }),
                        'mode': 'REALTIME'
                    })
                except Exception:
                    pass

    async def _auto_heal_divergence(self, sym, st, local_qty, broker_qty):
        """
        分叉自愈处理逻辑
        """
        # 第一步：物理锁定，防止策略模块在这个混乱时刻继续开平仓
        st.is_pending = True 
        
        # 场景 A：本地以为没仓位，但券商有（比如人工用手机APP买入了，或者平仓单被莫名撤销）
        if local_qty == 0 and broker_qty != 0:
            logger.warning(f"🧟 [{sym}] 发现幽灵持仓！系统强制接管该头寸...")
            st.qty = abs(broker_qty)
            st.position = 1 if broker_qty > 0 else -1
            # 伪造一个标志性的成本价，迫使策略在下一次逻辑触发时可以识别或优先处理
            st.entry_price = 9999.0 if st.position == 1 else 0.01 
            st.entry_ts = time.time()
            st.max_roi = -0.99 

        # 场景 B：本地以为有仓位，但券商没有（比如期权到期行权了，或者被强平了）
        elif local_qty != 0 and broker_qty == 0:
            logger.warning(f"💨 [{sym}] 持仓已蒸发！本地强制清零...")
            st.qty = 0
            st.position = 0
            
        # 场景 C：数量对不上（比如只部分成交了，但本地扣减逻辑出错了）
        else:
            logger.warning(f"🩹 [{sym}] 数量错位！强制将本地数量对齐为 {broker_qty}...")
            st.qty = abs(broker_qty)
            st.position = 1 if broker_qty > 0 else -1

        # 记录强制修复日志，发送到 Dashboard
        self.orch.accounting._emit_trade_log({
            'ts': time.time(),
            'symbol': sym,
            'action': 'RECONCILE_FIX',
            'side': 'SYS_SYNC',
            'qty': abs(local_qty - broker_qty),
            'price': 0.0,
            'stock_price': 0.0,
            'strategy_note': json.dumps({'reason': f"FIX: Local({local_qty})->Broker({broker_qty})"}),
            'mode': 'REALTIME'
        })
        
        # 解锁并强制持久化
        st.is_pending = False
        self.orch.state_manager.save_state()
