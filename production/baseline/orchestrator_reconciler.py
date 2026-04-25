import asyncio
import logging
import json
import time
import os
import math
from config import TRADING_ENABLED
from runtime_trading_controls import get_runtime_trading_enabled
from orchestrator_state_manager import infer_open_fill_confirmed

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

    @staticmethod
    def _state_has_confirmed_open(st) -> bool:
        return infer_open_fill_confirmed({
            'position': getattr(st, 'position', 0),
            'qty': getattr(st, 'qty', 0),
            'entry_price': getattr(st, 'entry_price', 0.0),
            'entry_ts': getattr(st, 'entry_ts', 0.0),
            'open_fill_confirmed': getattr(st, 'open_fill_confirmed', None),
        })

    def _reconciled_open_fill_confirmed(self, st, broker_qty, broker_entry_price: float = 0.0) -> bool:
        try:
            broker_qty = float(broker_qty or 0.0)
        except Exception:
            broker_qty = 0.0
        if broker_qty == 0:
            return False
        if self._state_has_confirmed_open(st):
            return True

        candidates = [
            float(broker_entry_price or 0.0),
            float(getattr(st, 'last_opt_price', 0.0) or 0.0),
            float(getattr(st, 'entry_price', 0.0) or 0.0),
        ]
        return any(math.isfinite(px) and px > 0.01 for px in candidates)

    async def run_reconciliation_loop(self):
        """
        [防掉单对账器] 后台守护协程
        """
        logger.info("🛡️ [Reconciler] 防掉单对账器已启动，将严密监控底层券商仓位...")
        
        while True:
            await asyncio.sleep(self.reconcile_interval)
            
            # 实盘模式下持续对账。即便 runtime DISARM，也要继续同步真实持仓，
            # 否则 Dashboard 会在停机观察期丢失 broker 真实状态。
            if self.orch.mode != 'realtime':
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
        
        # 2. 统计真实持仓 (仅对齐策略相关的 OPT 仓位；股票仓位不能污染期权策略账本)
        broker_state = {}
        broker_avg_price = {}
        skipped_non_opt = []
        for pos in real_positions:
            contract = getattr(pos, 'contract', None)
            sec_type = str(getattr(contract, 'secType', '') or '').upper()
            if sec_type != 'OPT':
                if contract is not None:
                    skipped_non_opt.append(
                        f"{getattr(contract, 'symbol', '?')}:{sec_type or 'UNKNOWN'}:{getattr(pos, 'position', 0)}"
                    )
                continue

            # pos.contract.symbol 在期权合约中通常代表底层正股 (如 'NVDA')
            sym = getattr(contract, 'symbol', None)
            qty = float(getattr(pos, 'position', 0) or 0.0) # 多头为正，空头为负
            
            if not sym or qty == 0:
                continue

            # 累加该标的下的所有期权头寸数量 (注意：此处假设每个 symbol 下通常持有一种期权)
            # 如果存在多腿组合，逻辑可能需要更精细，目前按 Symbol 极简对齐
            broker_state[sym] = float(broker_state.get(sym, 0.0) or 0.0) + qty

            try:
                avg_cost = float(getattr(pos, 'avgCost', 0.0) or 0.0)
            except Exception:
                avg_cost = 0.0
            try:
                multiplier = float(getattr(contract, 'multiplier', 100) or 100.0)
            except Exception:
                multiplier = 100.0
            multiplier = multiplier if multiplier > 0 else 100.0
            premium = abs(avg_cost) / multiplier if abs(avg_cost) > 0 else 0.0
            if math.isfinite(premium) and premium > 0.01:
                prev = broker_avg_price.get(sym)
                weight = abs(qty)
                if not prev:
                    broker_avg_price[sym] = {'px': premium, 'w': weight}
                else:
                    total_w = float(prev.get('w', 0.0) or 0.0) + weight
                    if total_w > 0:
                        broker_avg_price[sym] = {
                            'px': ((float(prev.get('px', 0.0) or 0.0) * float(prev.get('w', 0.0) or 0.0)) + premium * weight) / total_w,
                            'w': total_w,
                        }

        if skipped_non_opt:
            preview = ", ".join(skipped_non_opt[:8])
            logger.warning(
                f"🧹 [Reconciler] skipped non-OPT broker positions: {preview}"
                + (" ..." if len(skipped_non_opt) > 8 else "")
            )

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
                await self._auto_heal_divergence(
                    sym,
                    st,
                    local_qty,
                    broker_qty,
                    broker_entry_price=float((broker_avg_price.get(sym) or {}).get('px', 0.0) or 0.0),
                )

        await self._perform_cash_reconciliation()

    async def _perform_cash_reconciliation(self):
        """Cash reconciliation disabled: broker cash and OMS paper cash use different bases."""
        return

    def _safe_reconciled_entry_price(self, st, broker_entry_price: float = 0.0) -> float:
        candidates = [
            float(broker_entry_price or 0.0),
            float(getattr(st, 'last_opt_price', 0.0) or 0.0),
            float(getattr(st, 'entry_price', 0.0) or 0.0),
        ]
        for px in candidates:
            if math.isfinite(px) and 0.01 <= px < 1000.0:
                return px
        return 0.01

    async def _auto_heal_divergence(self, sym, st, local_qty, broker_qty, broker_entry_price: float = 0.0):
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
            st.entry_price = self._safe_reconciled_entry_price(st, broker_entry_price)
            st.entry_stock = float(getattr(st, 'last_price', 0.0) or 0.0)
            st.entry_ts = time.time()
            st.max_roi = -0.99 
            st.locked_cash = 0.0
            st.entry_slot_reserved = False
            st.open_fill_confirmed = self._reconciled_open_fill_confirmed(st, broker_qty, broker_entry_price)

        # 场景 B：本地以为有仓位，但券商没有（比如期权到期行权了，或者被强平了）
        elif local_qty != 0 and broker_qty == 0:
            logger.warning(f"💨 [{sym}] 持仓已蒸发！本地强制清零...")
            st.qty = 0
            st.position = 0
            st.entry_price = 0.0
            st.entry_stock = 0.0
            st.locked_cash = 0.0
            st.entry_slot_reserved = False
            st.open_fill_confirmed = False
            
        # 场景 C：数量对不上（比如只部分成交了，但本地扣减逻辑出错了）
        else:
            logger.warning(f"🩹 [{sym}] 数量错位！强制将本地数量对齐为 {broker_qty}...")
            st.qty = abs(broker_qty)
            st.position = 1 if broker_qty > 0 else -1
            st.entry_price = self._safe_reconciled_entry_price(st, broker_entry_price)
            st.entry_stock = float(getattr(st, 'last_price', 0.0) or 0.0)
            st.locked_cash = 0.0
            st.entry_slot_reserved = False
            st.open_fill_confirmed = self._reconciled_open_fill_confirmed(st, broker_qty, broker_entry_price)

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
        try:
            scheduler = getattr(getattr(self.orch, 'accounting', None), '_schedule_live_state_broadcast', None)
            if scheduler is not None:
                scheduler()
        except Exception:
            pass
