import json
import time
import logging
import re
import math
import asyncio
import os
from datetime import datetime, timedelta, time as dt_time
from pytz import timezone
import config
from config import (
    COMMISSION_PER_CONTRACT, MAX_POSITIONS, POSITION_RATIO, MAX_TRADE_CAP, 
    GLOBAL_EXPOSURE_LIMIT, SLIPPAGE_ENTRY_PCT, SLIPPAGE_EXIT_PCT, 
    EXIT_ORDER_TYPE, TRADING_ENABLED,
    DISABLE_ICEBERG, SYNC_EXECUTION, BACKTEST_1S_DISABLE_ICEBERG,
    BACKTEST_1S_STRICT_EXECUTION,
    ENTRY_MAX_REQUOTE_SLIPPAGE_PCT, ENTRY_REQUOTE_STEP_CAP_PCT, AUTO_TRADING_CAPITAL_RATIO,
    LIVE_TRADING_CAPITAL_LIMIT, IS_REALTIME_DRY,
)
from entry_risk_rules import get_entry_min_option_price
from liquidity_rules import LiquidityRiskManager
from runtime_trading_controls import (
    get_runtime_live_trading_capital_limit,
    get_runtime_trading_enabled,
)

logger = logging.getLogger("V8_Orchestrator.Execution")
PURE_ALPHA_REPLAY = os.environ.get('PURE_ALPHA_REPLAY') == '1'

class OrchestratorExecution:
    URGENT_EXIT_REASON_TOKENS = (
        "HARD_STOP",
        "ABSOLUTE_STOP",
        "STOP_LOSS",
        "STOCK_STOP",
        "COND_STOP",
        "EOD",
        "FORCE",
        "FLIP",
    )

    def __init__(self, orchestrator):
        self.orch = orchestrator
        self._refund_once_keys = {}

    def _cash_set(self, value: float, reason: str, sym: str = ""):
        logger.critical(
            f"🚫 [CashInvariant] blocked execution-layer cash set | "
            f"reason={reason} {sym} value={value}"
        )

    def _cash_add(self, delta: float, reason: str, sym: str = ""):
        logger.critical(
            f"🚫 [CashInvariant] blocked execution-layer cash mutation | "
            f"reason={reason} {sym} delta={delta}"
        )

    def _refund_locked_cash_once(self, st, amount: float, reason: str, sym: str, refund_key: str):
        """Release pending allocation without touching cash.

        Remaining Cash is changed only by actual OPEN/CLOSE fills. Pending
        entry allocation uses st.locked_cash to reserve capacity; rejecting or
        cancelling that intent merely releases the reservation.
        """
        now = time.time()
        # prune old keys
        if len(self._refund_once_keys) > 20000:
            cutoff = now - 24 * 3600
            self._refund_once_keys = {k: v for k, v in self._refund_once_keys.items() if v >= cutoff}
        if refund_key in self._refund_once_keys:
            logger.warning(f"🛡️ [RefundGuard] duplicate refund ignored | key={refund_key} | {sym} {reason}")
            return 0.0

        try:
            req = float(amount or 0.0)
        except Exception:
            req = 0.0
        locked = max(0.0, float(getattr(st, 'locked_cash', 0.0) or 0.0))
        refund = min(max(0.0, req), locked)
        if refund <= 0:
            logger.warning(
                f"🛡️ [RefundGuard] no refundable locked cash | key={refund_key} | "
                f"{sym} {reason} | requested={req:.2f} locked={locked:.2f}"
            )
            return 0.0
        if refund + 1e-6 < req:
            logger.warning(
                f"🛡️ [RefundGuard] capped refund by locked cash | key={refund_key} | "
                f"{sym} {reason} | requested={req:.2f} refund={refund:.2f} locked={locked:.2f}"
            )

        st.locked_cash = max(0.0, locked - refund)
        self._refund_once_keys[refund_key] = now
        logger.info(
            f"🔓 [CashReserve] {reason} {sym} | release={refund:.2f} | "
            f"locked_before={locked:.2f} | locked_after={st.locked_cash:.2f} | cash_unchanged={float(getattr(self.orch, 'mock_cash', 0.0) or 0.0):.2f}"
        )
        return refund

    def _count_active_slots(self, include_pending: bool = True, exclude_sym: str = "") -> int:
        cnt = 0
        for s, s_state in self.orch.states.items():
            if exclude_sym and s == exclude_sym:
                continue
            pos = int(getattr(s_state, 'position', 0) or 0)
            reserved = bool(getattr(s_state, 'entry_slot_reserved', False))
            pending = bool(getattr(s_state, 'is_pending', False))
            pending_buy = pending and str(
                getattr(s_state, 'pending_action', None) or getattr(s_state, 'pending_side', '') or ''
            ).upper() == 'BUY'
            if pos != 0 or reserved or (include_pending and pending_buy):
                cnt += 1
        return cnt

    def _effective_position_exposure(self, st, sym: str = "") -> float:
        try:
            qty = abs(float(getattr(st, 'qty', 0.0) or 0.0))
        except Exception:
            qty = 0.0
        try:
            entry_price = float(getattr(st, 'entry_price', 0.0) or 0.0)
        except Exception:
            entry_price = 0.0
        if qty <= 0 or not math.isfinite(entry_price) or entry_price <= 0.0:
            return 0.0

        # Legacy reconciler once injected ghost positions with entry_price=9999,
        # which can explode quota math and permanently freeze new entries.
        if entry_price >= 1000.0 and not bool(getattr(st, 'open_fill_confirmed', False)):
            try:
                fallback = float(getattr(st, 'last_opt_price', 0.0) or 0.0)
            except Exception:
                fallback = 0.0
            if math.isfinite(fallback) and 0.01 <= fallback < 1000.0:
                entry_price = fallback
            else:
                logger.warning(
                    f"🧹 [Exposure-Sanitize] {sym or '?'} ignore abnormal entry_price={getattr(st, 'entry_price', 0.0)} "
                    f"qty={qty} open_fill_confirmed={bool(getattr(st, 'open_fill_confirmed', False))}"
                )
                return 0.0

        return qty * entry_price * 100.0

    @staticmethod
    def _fmt_ny_time(ts: float) -> str:
        if not ts:
            return ""
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return ""

    def _emit_order_event(self, sym, action, side, qty, price=0.0, reason="", extra=None):
        try:
            self.orch.accounting._emit_trade_log({
                'ts': time.time(),
                'symbol': sym,
                'action': action,
                'side': side,
                'qty': float(qty or 0.0),
                'price': float(price or 0.0),
                'stock_price': 0.0,
                'mode': self.orch.mode.upper(),
                'strategy_note': json.dumps({
                    'reason': str(reason or ''),
                    **(extra or {}),
                }, ensure_ascii=True),
            })
        except Exception as e:
            logger.warning(f"⚠️ [OrderEvent] emit failed for {sym} {action}: {e}")

    @staticmethod
    def _make_pending_order_key(sym: str, intent: str, contract_id: str = "", seed_ts: float = 0.0) -> str:
        ts_ms = int(float(seed_ts or time.time()) * 1000.0)
        contract_part = re.sub(r"[^A-Za-z0-9]+", "", str(contract_id or ""))[:24] or "NA"
        return f"{str(sym or '').upper()}:{str(intent or '').upper()}:{contract_part}:{ts_ms}"

    @staticmethod
    def _trade_identifiers(trade) -> dict:
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        return {
            "broker_order_id": getattr(order, "orderId", None),
            "perm_id": getattr(order, "permId", getattr(status, "permId", None)),
            "client_id": getattr(order, "clientId", None),
            "broker_status": str(getattr(status, "status", "") or ""),
        }

    def _upsert_pending_order(self, order_key: str, payload: dict):
        manager = getattr(self.orch, "order_state", None)
        if manager is None:
            return
        try:
            manager.upsert(order_key, payload)
        except Exception as e:
            logger.warning(f"⚠️ pending order sync failed for {order_key}: {e}")

    def _cfg_value(self, key, default):
        cfg = getattr(self.orch, 'cfg', None)
        if cfg is None:
            return default
        return getattr(cfg, key, default)

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

    @classmethod
    def _is_urgent_exit_reason(cls, reason: str, is_force: bool = False) -> bool:
        """Only true risk exits should use emergency execution semantics.

        Avoid generic "STOP" matching: TIME_STOP / ZOMBIE_STOP / SPREAD_STOP
        are not worth a default market order, especially when the trigger is a
        bad spread.
        """
        if is_force:
            return True
        reason_text = str(reason or "").upper()
        return any(token in reason_text for token in cls.URGENT_EXIT_REASON_TOKENS)

    @classmethod
    def _resolve_exit_order_type(cls, configured_order_type: str, reason: str, is_force: bool = False) -> str:
        """Map configured exit order type to the safe effective order type."""
        configured = str(configured_order_type or "LMT").upper()
        if configured != "MKT":
            return "LMT"
        return "MKT" if cls._is_urgent_exit_reason(reason, is_force=is_force) else "LMT"

    def _effective_slippage_pct(self, side: str = 'entry') -> float:
        """[REALTIME_DRY 友好] 返回当前运行模式下实际生效的滑点比例.

        - REALTIME_DRY: 默认 0.0, 便于干净地评估 alpha 链路真实收益.
          若要复刻实盘体验, 设 REALTIME_DRY_APPLY_SLIPPAGE=1 即可恢复配置值.
        - 其他模式 (REALTIME / BACKTEST): 维持原优先级,
          SLIPPAGE_PCT → SLIPPAGE_{ENTRY|EXIT}_PCT → config 默认值.
        """
        try:
            from config import IS_REALTIME_DRY as runtime_is_realtime_dry
        except Exception:
            runtime_is_realtime_dry = False

        if runtime_is_realtime_dry:
            apply_flag = os.environ.get('REALTIME_DRY_APPLY_SLIPPAGE', '').strip().lower()
            if apply_flag not in ('1', 'true', 'yes'):
                return 0.0

        if side == 'entry':
            fallback_key, default_val = 'SLIPPAGE_ENTRY_PCT', SLIPPAGE_ENTRY_PCT
        else:
            fallback_key, default_val = 'SLIPPAGE_EXIT_PCT', SLIPPAGE_EXIT_PCT
        return float(self._cfg_value('SLIPPAGE_PCT', self._cfg_value(fallback_key, default_val)))

    # ------------------------------------------------------------------
    # [🛡️ IBKR Connection Guard]
    # 统一的 IBKR 连接态判断入口, 供 entry / exit 路径在下单前预检.
    # 目的: 一旦 IBKR 掉线, 不要让 ib_insync 抛 ConnectionError 把每一个
    # SELL / BUY 信号都炸成一条 traceback; 也不要走 "not trade → 模拟成交"
    # 的 DRY RUN 兜底, 否则账本会把实际仍持有的仓位清零, 造成账实不符.
    # ------------------------------------------------------------------
    def _ibkr_is_connected(self) -> bool:
        """True 表示 self.ib 当前已连接; 任何异常一律视为未连接."""
        ibkr = getattr(self.orch, "ibkr", None)
        ib = getattr(ibkr, "ib", None) if ibkr is not None else None
        if ib is None:
            return False
        try:
            return bool(ib.isConnected())
        except Exception:
            return False

    def _ibkr_recently_failed(self, window_sec: float = 5.0) -> bool:
        """最近 `window_sec` 秒内 place_option_order 记录过连接失败? 用于区分
        "DRY RUN 预期 None" vs "实盘连接断开 None"."""
        ibkr = getattr(self.orch, "ibkr", None)
        if ibkr is None:
            return True
        last = float(getattr(ibkr, 'last_not_connected_ts', 0.0) or 0.0)
        return last > 0.0 and (time.time() - last) < float(window_sec)

    def _log_ibkr_disconnect_throttled(self, sym: str, reason: str):
        """节流打印 IBKR 断连告警, 避免每秒刷屏 60 条 traceback."""
        now = time.time()
        last = float(getattr(self, '_ibkr_disconnect_warn_ts', 0.0) or 0.0)
        if now - last > 30.0:
            logger.warning(
                f"🔌 [IBKR Down] 跳过下单请求 | sym={sym} reason={reason} "
                f"| 等待 connect()/maintenance_loop 恢复连接后重新处理."
            )
            self._ibkr_disconnect_warn_ts = now

    def _entry_requote_cap_price(self, reference_price: float) -> float:
        ref = float(reference_price or 0.0)
        if ref <= 0.0:
            return float("inf")
        cap_pct = float(self._cfg_value("ENTRY_MAX_REQUOTE_SLIPPAGE_PCT", ENTRY_MAX_REQUOTE_SLIPPAGE_PCT))
        cap_pct = max(0.0, cap_pct)
        return ref * (1.0 + cap_pct)

    def _try_get_live_option_quote(self, contract):
        """
        尝试从实时 IB ticker 获取最新 bid/ask。
        返回: (bid, ask)；失败或无效返回 (0.0, 0.0)。
        """
        try:
            ibkr = getattr(self.orch, "ibkr", None)
            ib = getattr(ibkr, "ib", None)
            if ib is None:
                return 0.0, 0.0
            if not ib.isConnected():
                return 0.0, 0.0
            if contract is None:
                return 0.0, 0.0
            t = ib.ticker(contract)
            if t is None:
                return 0.0, 0.0
            bid = float(getattr(t, "bid", 0.0) or 0.0)
            ask = float(getattr(t, "ask", 0.0) or 0.0)
            if bid > 0.0 and ask > 0.0 and ask >= bid:
                return bid, ask
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    def _next_entry_requote_price(self, sig, prev_limit_price, attempt_no, cap_price, real_contract=None):
        """
        计算下一次建仓追价：
        1) 优先使用实时 bid/ask（若可得）；
        2) 总追价上限始终锚定首笔参考价 cap_price；
        3) 单步追价上限（相对上一笔）限制跳变幅度，避免一次追太猛。
        """
        prev_limit_price = float(prev_limit_price or 0.0)
        if prev_limit_price <= 0:
            return 0.0

        # 优先用实时盘口替换旧的 sig.meta，降低 3s 后的陈旧报价风险
        bid_live, ask_live = self._try_get_live_option_quote(real_contract)
        sig_for_price = sig
        if bid_live > 0 and ask_live > 0:
            sig_for_price = dict(sig)
            meta = dict(sig.get("meta", {}))
            meta["bid"] = bid_live
            meta["ask"] = ask_live
            sig_for_price["meta"] = meta
            base_price = (bid_live + ask_live) / 2.0
            candidate = self._get_entry_limit_price(sig_for_price, base_price, attempt_no=0)
        else:
            candidate = self._get_entry_limit_price(sig_for_price, prev_limit_price, attempt_no=attempt_no)

        if candidate < 0.05:
            return 0.0
        if candidate > cap_price:
            logger.warning(
                f"🛑 [Entry Requote Cap] raw_candidate={candidate:.4f} exceeds cap={cap_price:.4f}; stop chasing."
            )
            return 0.0

        # 单步追价上限（相对上一笔）
        step_cap_pct = float(self._cfg_value("ENTRY_REQUOTE_STEP_CAP_PCT", ENTRY_REQUOTE_STEP_CAP_PCT))
        step_cap_pct = max(0.0, step_cap_pct)
        step_cap_price = prev_limit_price * (1.0 + step_cap_pct)

        return min(candidate, step_cap_price, cap_price)

    @staticmethod
    def _build_timing_fields(alpha_label_ts: float, alpha_available_ts: float, order_submit_ts: float, fill_ts: float) -> dict:
        alpha_label_ts = float(alpha_label_ts or 0.0)
        alpha_available_ts = float(alpha_available_ts or 0.0)
        order_submit_ts = float(order_submit_ts or 0.0)
        fill_ts = float(fill_ts or 0.0)

        alpha_to_submit_ms = ((order_submit_ts - alpha_available_ts) * 1000.0) if alpha_available_ts > 0 and order_submit_ts > 0 else None
        submit_to_fill_ms = ((fill_ts - order_submit_ts) * 1000.0) if order_submit_ts > 0 and fill_ts > 0 else None
        alpha_to_fill_ms = ((fill_ts - alpha_available_ts) * 1000.0) if alpha_available_ts > 0 and fill_ts > 0 else None

        return {
            'alpha_label_ts': alpha_label_ts,
            'alpha_available_ts': alpha_available_ts,
            'order_submit_ts': order_submit_ts,
            'fill_ts': fill_ts,
            'alpha_to_submit_ms': alpha_to_submit_ms,
            'submit_to_fill_ms': submit_to_fill_ms,
            'alpha_to_fill_ms': alpha_to_fill_ms,
        }

    @staticmethod
    def _split_qty_into_chunks(total_qty, chunks):
        if total_qty <= 0 or chunks <= 1:
            return [total_qty] if total_qty > 0 else []
        base_chunk_qty = total_qty // chunks
        remainder_qty = total_qty % chunks
        return [
            base_chunk_qty + (1 if i < remainder_qty else 0)
            for i in range(chunks)
            if (base_chunk_qty + (1 if i < remainder_qty else 0)) > 0
        ]

    @staticmethod
    def _reset_failed_entry_state(st):
        st.position = 0
        st.qty = 0
        st.entry_stock = 0.0
        st.entry_price = 0.0
        st.entry_ts = 0.0
        st.max_roi = 0.0
        st.locked_cash = 0.0
        st.entry_slot_reserved = False
        st.open_fill_confirmed = False

    def _use_gentle_1s_backtest_execution(self) -> bool:
        if self.orch.mode != 'backtest':
            return False
        ibkr_mod = getattr(getattr(self.orch, 'ibkr', None), '__module__', '') or ''
        if 'mock_ibkr_historical_1s' not in ibkr_mod:
            return False
        # strict/gentle 任一开启都强制单块执行，避免拆单在 1s 回测中制造
        # 虚假的“重叠成交/高容量”收益。
        return bool(BACKTEST_1S_DISABLE_ICEBERG or BACKTEST_1S_STRICT_EXECUTION)

    def _get_entry_limit_price(self, sig, base_price, attempt_no=0):
        bid = float(sig.get('meta', {}).get('bid', 0.0) or 0.0)
        ask = float(sig.get('meta', {}).get('ask', 0.0) or 0.0)
        base_price = float(base_price or 0.0)
        limit_buffer_entry = float(self._cfg_value('LIMIT_BUFFER_ENTRY', config.LIMIT_BUFFER_ENTRY))

        if bid > 0.0 and ask > 0.0 and ask >= bid:
            mid = (bid + ask) / 2.0
            spread = max(ask - bid, 0.0)
            # Entry policy: probe inside the spread and never cross the ask.
            # Retry can get more aggressive, but still must remain strictly below
            # the displayed ask to avoid paying the full spread outright.
            # Moderate inside-spread probing:
            # - keep the initial order clearly below ask
            # - let retries lean further into the spread to recover fill rate
            inside_spread_probe = min(0.20 + 0.12 * max(attempt_no, 0), 0.45)
            raw_candidate = mid + spread * inside_spread_probe
            bid_tick = math.floor(bid * 100.0) / 100.0
            ask_tick = math.ceil(ask * 100.0) / 100.0
            ask_minus_tick = max(round(ask_tick - 0.01, 2), round(bid_tick, 2))
            if ask_minus_tick <= 0.0:
                return 0.0
            candidate = math.floor(raw_candidate * 100.0) / 100.0
            candidate = max(round(bid_tick, 2), candidate)
            candidate = min(candidate, ask_minus_tick)
            return round(candidate, 2)

        if base_price <= 0.0:
            return 0.0
        fallback_buf = max(limit_buffer_entry, 1.0)
        requote_step = 0.005 * max(attempt_no, 0)
        return round(base_price * (fallback_buf + requote_step), 2)

    def _get_exit_limit_price(self, base_price, bid=0.0, ask=0.0, is_urgent=False, attempt_no=0):
        bid = float(bid or 0.0)
        ask = float(ask or 0.0)
        base_price = float(base_price or 0.0)
        limit_buffer_exit = float(self._cfg_value('LIMIT_BUFFER_EXIT', config.LIMIT_BUFFER_EXIT))

        if bid > 0.01:
            if is_urgent:
                return round(max(bid - (0.01 * max(attempt_no, 0)), 0.01), 2)
            mid = (bid + ask) / 2.0 if ask > 0.0 else bid
            initial_price = max(mid - 0.01, bid)
            step_down = 0.01 * max(attempt_no, 0)
            return round(max(initial_price - step_down, bid), 2)

        if base_price <= 0.0:
            return 0.01
        requote_discount = 0.005 * max(attempt_no, 0)
        base_discount = max(0.0, 1.0 - limit_buffer_exit)
        total_discount = base_discount + requote_discount
        return round(max(base_price * (1 - total_discount), 0.01), 2)

    def _release_entry_pending(self, st):
        """[🛡️ Ghost Pending Fix]
        入场流程早退 (return) 前统一调用，把 shared_mem 下给的
        `is_pending=True, pending_action='BUY'` 标记清掉。
        不这样做的话，`_execute_entry` 里任何一个 early return (资金不足/价格不合规/
        IV 过低 等) 都会留下幽灵 pending BUY，在 SE 的 `current_active` 统计里
        常驻，很快把 MAX_POSITIONS 的名额堆满，所有新候选都被 Batch-Limit 砍掉。
        """
        try:
            st.is_pending = False
            if hasattr(st, 'pending_action'):
                st.pending_action = ''
            if hasattr(st, 'pending_side'):
                st.pending_side = None
            if hasattr(st, '_pending_frames'):
                st._pending_frames = 0
        except Exception:
            pass

    async def _execute_entry(self, sym, sig, stock_price, curr_ts, batch_idx):
        st = self.orch.states[sym]
        
        # 🛡️ 移除此处重复的 is_pending 检查，该逻辑已统一上移至 ExecutionEngineV8 处理。
        # 这里只管真正的执行。
        
        # 物理加锁
        if not getattr(self.orch, 'use_shared_mem', False):
            st.is_pending = True
            st.locked_cash = 0.0 # 初始化锁定金额，防止 NaN
            
        if st.position != 0:
            # 已有持仓，理论上 SE 不应派发 BUY；出现即视为幽灵，强制清锁后返回。
            logger.warning(f"🚫 [Entry EarlyReturn] {sym} blocked: state already has position={st.position}")
            self._release_entry_pending(st)
            return

        # =================================================================
        # 精准盘点资金池
        # =================================================================
        position_exposure_by_bot = 0.0
        pending_locked_cash_by_bot = 0.0
        active_count = 0
        for s, s_state in self.orch.states.items():
            if s == sym:
                continue
            # [🔥 终极修复] 共享内存下，SE 会批量给所有候选者打上 is_pending 标。
            # 如果 OMS 盘点时还算上这些标，那第一笔开出来之前，所有标都会因“超限”被拒死。
            # 方案：共享内存模式下，只盘点【实体持仓】，挂起逻辑交给 SE 管理。
            is_p = getattr(s_state, 'is_pending', False)
            if getattr(self.orch, 'use_shared_mem', False):
                is_p = False # 屏蔽挂起干扰
                
            reserved = bool(getattr(s_state, 'entry_slot_reserved', False))
            if getattr(s_state, 'position', 0) != 0 or reserved or is_p:
                active_count += 1
                if is_p or reserved:
                    pending_locked_cash_by_bot += getattr(s_state, 'locked_cash', 0.0)
                else:
                    position_exposure_by_bot += self._effective_position_exposure(s_state, sym=str(s))
        locked_cash_by_bot = position_exposure_by_bot + pending_locked_cash_by_bot
        
        max_positions = max(1, int(self._cfg_value('MAX_POSITIONS', MAX_POSITIONS)))
        # Hard gate: regardless of upstream candidate sorting, OMS must never
        # admit an entry when slots are already full (excluding current symbol).
        hard_slots_before = self._count_active_slots(include_pending=True, exclude_sym=sym)
        if hard_slots_before >= max_positions:
            logger.warning(
                f"🚫 [HardCap] {sym} blocked before entry | active_slots={hard_slots_before} >= max={max_positions}"
            )
            self._release_entry_pending(st)
            return
        parity_bypass = (
            os.environ.get('PARITY_MODE') == 'PLAN_A'
            and self.orch.mode == 'backtest'
        )
        if parity_bypass:
            logger.info(
                f"🔍 [PARITY_TRACE] {sym} | active_count={active_count} | "
                f"MAX_POSITIONS={max_positions} (BYPASSED in backtest only)"
            )
        else:
            if active_count >= max_positions:
                logger.warning(
                    f"✋ [风控拒单 - {sym}] 当前已有 {active_count} 个持仓或挂单，"
                    f"达到上限 ({max_positions})，拒绝开仓！"
                )
                self._release_entry_pending(st)
                return

        if math.isnan(self.orch.mock_cash):
            logger.error("🛑 [Fatal] mock_cash is NaN! Emergency fallback to INITIAL_ACCOUNT.")
            self._release_entry_pending(st)
            return

        position_ratio = max(0.0, min(1.0, float(self._cfg_value('POSITION_RATIO', POSITION_RATIO))))
        global_exposure_limit = max(0.0, min(1.0, float(self._cfg_value('GLOBAL_EXPOSURE_LIMIT', GLOBAL_EXPOSURE_LIMIT))))
        max_trade_cap = float(self._cfg_value('MAX_TRADE_CAP', MAX_TRADE_CAP))
        commission_per_contract = float(self._cfg_value('COMMISSION_PER_CONTRACT', COMMISSION_PER_CONTRACT))
        auto_cap_ratio = float(self._cfg_value('AUTO_TRADING_CAPITAL_RATIO', AUTO_TRADING_CAPITAL_RATIO))
        auto_cap_ratio = max(0.0, min(1.0, auto_cap_ratio))
        if auto_cap_ratio <= 0.0:
            logger.warning(f"✋ [风控拒单 - {sym}] AUTO_TRADING_CAPITAL_RATIO=0，自动交易资金池已关闭。")
            self._release_entry_pending(st)
            return
        if position_ratio <= 0.0:
            logger.warning(f"✋ [风控拒单 - {sym}] POSITION_RATIO=0，单标的仓位分配为 0。")
            self._release_entry_pending(st)
            return

        if self.orch.mode == 'realtime' and hasattr(self.orch, '_apply_live_trading_capital_limit'):
            try:
                self.orch._apply_live_trading_capital_limit("entry_runtime_override")
            except Exception:
                pass

        available_cash_for_new_entries = max(0.0, float(self.orch.mock_cash or 0.0) - pending_locked_cash_by_bot)
        bot_total_capital = self.orch.mock_cash + position_exposure_by_bot
        # 自动策略只允许使用自动资金池；手动池由 Dashboard 等人工触发通道使用。
        auto_pool_capital = bot_total_capital * auto_cap_ratio
        runtime_live_cap = float(
            get_runtime_live_trading_capital_limit(
                default_value=LIVE_TRADING_CAPITAL_LIMIT,
                r=getattr(self.orch, 'r', None),
            ) or 0.0
        )
        if self.orch.mode == 'realtime' and not IS_REALTIME_DRY and runtime_live_cap > 0.0:
            auto_pool_capital = min(auto_pool_capital, runtime_live_cap)
        raw_alloc = auto_pool_capital * position_ratio
        max_exposure = auto_pool_capital * global_exposure_limit
        remaining_quota = max(0.0, max_exposure - locked_cash_by_bot)
        final_alloc = min(raw_alloc, remaining_quota, available_cash_for_new_entries, max_trade_cap)
        
        if final_alloc < 200:
            logger.warning(
                f"🚫 [Entry EarlyReturn] {sym} blocked: final_alloc<{200} "
                f"| final_alloc=${final_alloc:,.2f} raw_alloc=${raw_alloc:,.2f} "
                f"| remaining_quota=${remaining_quota:,.2f} available_cash=${available_cash_for_new_entries:,.2f} "
                f"| live_cap=${runtime_live_cap:,.2f} auto_pool=${auto_pool_capital:,.2f}"
            )
            self._release_entry_pending(st)
            return
        
        new_iv = sig.get('meta', {}).get('iv', 0.0)
        if not PURE_ALPHA_REPLAY and new_iv < 0.01 and st.last_valid_iv > 0.01:
            final_alloc *= 0.5
            logger.warning(f"⚠️ Data Sparse for {sym}: Using Stale IV {st.last_valid_iv:.2f} & 50% Size")
        
        price = sig.get('price', 0.0)
        min_option_price = get_entry_min_option_price(getattr(self.orch, 'MIN_OPTION_PRICE', 0.0))
        if price <= 0 or price < min_option_price or math.isnan(price):
            logger.info(f"✋ [风控拒单 - {sym}] 期权价格 {price:.2f} 低于最低限制 {min_option_price}")
            self._release_entry_pending(st)
            return

        if curr_ts < self.orch.global_cooldown_until:
            # [Layer 3 - Cross-Process Observability]
            # 历史 bug: 这里静默 return, SE 感知不到 OMS 已进入熔断, 继续 emit BUY;
            # 外观上就是"熔断消失了"。现在配合 Layer 1/2 的 Redis 广播, 这里再打一条
            # 节流日志 (每 sym 至多 30s 一条), 让运维能直接看到拒单链路。
            try:
                last_map = getattr(self.orch, '_cb_reject_last_log_ts', None)
                if not isinstance(last_map, dict):
                    last_map = {}
                    self.orch._cb_reject_last_log_ts = last_map
                last_ts = float(last_map.get(sym, 0.0))
                if (curr_ts - last_ts) >= 30.0:
                    ny_cool = datetime.fromtimestamp(self.orch.global_cooldown_until, tz=timezone('America/New_York'))
                    logger.warning(
                        f"🛡️ [OMS-Gate] {sym} BUY 被连败熔断拒单, 暂停至 {ny_cool.strftime('%H:%M:%S')} NY"
                    )
                    last_map[sym] = curr_ts
            except Exception:
                pass
            self._release_entry_pending(st)
            return
        
        effective_iv = new_iv if new_iv > 0.01 else st.last_valid_iv
        if not PURE_ALPHA_REPLAY and effective_iv < 0.01:
            logger.warning(
                f"🚫 [Entry EarlyReturn] {sym} blocked: effective_iv too low "
                f"| new_iv={float(new_iv or 0.0):.4f} last_valid_iv={float(getattr(st, 'last_valid_iv', 0.0) or 0.0):.4f}"
            )
            self._release_entry_pending(st)
            return

        # [Cleanup] 原本这里硬编码 `final_alloc = min(final_alloc, 150000.0)`，
        # 既与 config.MAX_TRADE_CAP (默认 100000) 不一致, 又与上方 L359 已经做过的
        # `min(raw_alloc, remaining_quota, mock_cash, max_trade_cap)` 重复且更宽松。
        # 删除该硬编码, 统一走 `max_trade_cap` 配置。
        ask_size = sig.get('meta', {}).get('ask_size', 0.0)
        
        liq_eval = LiquidityRiskManager.evaluate_order(sym, final_alloc, price, mode=self.orch.mode, ask_size=ask_size)
        final_alloc = liq_eval['final_alloc']
        chunks = liq_eval['chunks']
        if self._use_gentle_1s_backtest_execution():
            chunks = 1
        logger.info(f"💧 [流动性拆单评估] {sym} 最终核准额度: ${final_alloc:,.0f} | 拆分笔数: {chunks} | 理由: {liq_eval['reason']}")
        
        if final_alloc <= 0 or chunks < 1:
            logger.warning(
                f"🚫 [Entry EarlyReturn] {sym} blocked after liquidity eval "
                f"| final_alloc=${final_alloc:,.2f} chunks={chunks} reason={liq_eval.get('reason')}"
            )
            self._release_entry_pending(st)
            return

        logger.info(f"🚀 [交易柜台 - 真正发单] {sym} 通过所有风控检查！准备买入 {sig['tag']} @ {price}")
        
        #if self.orch.mode == 'backtest' or SYNC_EXECUTION: 
        # 🚀 [架构对齐 1] S4 和 Pitcher 统一使用纯净原价进行资金分配与数量计算
        fill_price = price

        cost_per_contract = (fill_price * 100) + commission_per_contract
        target_qty = int(final_alloc // cost_per_contract)
        
        if target_qty < 1:
            logger.warning(
                f"🚫 [Entry EarlyReturn] {sym} blocked: target_qty<1 "
                f"| final_alloc=${final_alloc:,.2f} fill_price=${fill_price:,.4f} "
                f"| cost_per_contract=${cost_per_contract:,.2f}"
            )
            self._release_entry_pending(st)
            return
        
        total_est_cost = target_qty * fill_price * 100
        total_est_comm = target_qty * commission_per_contract
        total_new_locked = total_est_cost + total_est_comm

        # 🛡️ [Hard Guard] 下单前最终敞口复核，确保不会突破自动资金池全局上限
        projected_locked = locked_cash_by_bot + total_new_locked
        if projected_locked - max_exposure > 1e-6:
            logger.warning(
                f"✋ [风控拒单 - {sym}] 预测敞口超限: projected=${projected_locked:,.2f} "
                f"> max_exposure=${max_exposure:,.2f} | current_locked=${locked_cash_by_bot:,.2f} "
                f"| new_locked=${total_new_locked:,.2f}"
            )
            self._release_entry_pending(st)
            return
        
        # 下单前只预占额度，不能修改 Remaining Cash。
        # mock_cash 只能在 _process_open_accounting 确认 OPEN 成交后扣减。
        st.locked_cash = total_new_locked
        st.entry_slot_reserved = True
        st.open_fill_confirmed = False
        
        st.position = sig['dir']
        st.qty = target_qty
        st.entry_stock = stock_price
        st.entry_price = fill_price

        # Final invariant check: never allow > MAX_POSITIONS to persist.
        hard_slots_after = self._count_active_slots(include_pending=True, exclude_sym="")
        if hard_slots_after > max_positions:
            logger.critical(
                f"🚨 [HardCap-Breach] {sym} would exceed max positions | after={hard_slots_after} > max={max_positions}. Rolling back this entry."
            )
            self._refund_locked_cash_once(
                st, total_new_locked, "ENTRY_POST_CAP_ROLLBACK", sym,
                refund_key=f"ENTRY_POST_CAP_ROLLBACK:{sym}:{int(curr_ts)}"
            )
            self._reset_failed_entry_state(st)
            self._release_entry_pending(st)
            return

        # =========== 冰山发单分流 [Ghost A: 冰山保护] ===========
        # 🚀 [对齐对冲] 如果处于强制确定性模式，即使是 realtime 也必须禁用冰山，对齐 S4 Atomic Fill
        is_deterministic = config.is_forced_deterministic(self.orch.r)
        entry_contract_id = str(sig.get('meta', {}).get('contract_id', getattr(st, 'contract_id', '') or '') or '')
        if chunks > 1 and self.orch.mode == 'realtime' and not DISABLE_ICEBERG and not SYNC_EXECUTION and not is_deterministic:
            # 关键: 冰山路径也先占用 position slot（上方已赋值并过 HardCap 复核），
            # 避免跨进程同步提前清 pending 时出现 "slot 瞬间丢失" 而超开仓。
            st._async_entry_order_active = True
            st.pending_action = 'BUY'
            st.pending_side = 'BUY'
            order_key = self._make_pending_order_key(sym, "OPEN", entry_contract_id, curr_ts)
            self._upsert_pending_order(order_key, {
                'symbol': sym,
                'contract_id': entry_contract_id,
                'intent': 'OPEN',
                'side': 'BUY',
                'status': 'PENDING_SUBMIT',
                'target_qty': int(target_qty),
                'filled_qty': 0,
                'remaining_qty': int(target_qty),
                'limit_price': float(fill_price or 0.0),
                'retry_count': 0,
                'reserved_cash': float(total_new_locked or 0.0),
                'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                'submit_ts': float(curr_ts or time.time()),
                'last_update_ts': time.time(),
                'reason': str(sig.get('reason', '') or ''),
            })
            asyncio.create_task(self._smart_entry_order(sym, sig, stock_price, curr_ts, target_qty, chunks, fill_price, total_est_cost, total_est_comm, order_key=order_key))
            return
        
        if 'meta' in sig:
            st.strike_price = sig['meta']['strike']
            if sig['meta']['iv'] > 0.01: st.last_valid_iv = sig['meta']['iv']
            st.contract_id = sig['meta']['contract_id']
            try:
                match = re.search(r'(\d{6})[CP]', st.contract_id)
                ny_now = datetime.fromtimestamp(curr_ts, tz=timezone('America/New_York'))
                st.expiry_date = datetime.strptime(match.group(1), '%y%m%d') if match else ny_now + timedelta(days=7)
            except: 
                ny_now = datetime.fromtimestamp(curr_ts, tz=timezone('America/New_York'))
                st.expiry_date = ny_now + timedelta(days=7)

        st.opt_type = 'call' if st.position == 1 else 'put'
        st.entry_ts = curr_ts
        # [🛡️ Defensive] 赋值即强转, 防止 sig.meta 里混入 dict/非数字 (历史曾出现)
        # 污染 st.entry_spy_roc → 进而触发 strategy.check_exit 的 TypeError 把整批
        # SE tick 的 alpha 陪葬.
        _meta = sig.get('meta', {}) or {}
        def _mf(v, d=0.0):
            if isinstance(v, dict): return d
            try: return float(v)
            except (TypeError, ValueError): return d
        def _mi(v, d=0):
            if isinstance(v, dict): return d
            try: return int(v)
            except (TypeError, ValueError): return d
        st.entry_spy_roc = _mf(_meta.get('spy_roc', 0.0))
        st.entry_index_trend = _mi(_meta.get('index_trend', 0))
        st.entry_alpha_z = _mf(_meta.get('alpha_z', 0.0))
        st.entry_iv = _mf(_meta.get('iv', st.last_valid_iv))
        st.max_roi = 0.0

        alpha_label_ts = sig.get('meta', {}).get('alpha_label_ts', 0.0)
        alpha_available_ts = sig.get('meta', {}).get('alpha_available_ts', curr_ts)
        logger.info(
            f"🕒 [Alpha Timing] {sym} | label={self._fmt_ny_time(alpha_label_ts)} "
            f"| available={self._fmt_ny_time(alpha_available_ts)} "
            f"| execution={self._fmt_ny_time(curr_ts)}"
        )
        
        is_simulated_env = (self.orch.mode == 'backtest')
        runtime_trading_enabled = self._runtime_trading_enabled()
        if is_simulated_env or SYNC_EXECUTION:
            order_submit_ts = float(curr_ts)
            fill_ts = float(st.entry_ts)
            timing_fields = self._build_timing_fields(alpha_label_ts, alpha_available_ts, order_submit_ts, fill_ts)
            
            # 🚀 [架构对齐 2] 统一计算包含滑点的最终成交价 (REALTIME_DRY 下默认 0)
            slippage_entry_pct = self._effective_slippage_pct('entry')
            simulated_fill_price = round(fill_price * (1 + slippage_entry_pct), 2)
            
            contract = type('MockContract', (), {'symbol': sym, 'localSymbol': st.contract_id, 'tag': sig['tag'], 'secType': 'OPT'})()
            
            liq_chunks = liq_eval.get('chunks', 1)
            liq_reason = liq_eval.get('reason', 'N/A')
            
            # [🛡️ Live Disconnect Guard]
            # 前一轮 IBKR 断线修复后，place_option_order 在未连接时会返回 None。
            # 如果 TRADING_ENABLED=True 且任一子单返回 None，说明实盘链路断了，
            # 此时绝不可以在本地虚构成交并记账 (否则账本与 IB 端完全分叉)。
            live_disconnect_abort = False
            if hasattr(self.orch.ibkr, 'place_option_order'):
                chunk_qtys = self._split_qty_into_chunks(st.qty, liq_chunks)
                for chunk_idx, chunk_qty in enumerate(chunk_qtys):
                    _ret = self.orch.ibkr.place_option_order(
                        contract,
                        'BUY',
                        chunk_qty,
                        'LMT',
                        simulated_fill_price, # 👈 发送含滑点的价格给 MockIBKR
                        reason=sig['reason'],
                        custom_time=st.entry_ts + (chunk_idx * 1e-6),
                        stock_price=stock_price,
                        chunks=liq_chunks,
                        liq_reason=liq_reason
                    )
                    if runtime_trading_enabled and _ret is None:
                        live_disconnect_abort = True
                        break

            if live_disconnect_abort:
                logger.warning(
                    f"🛑 [Live Disconnect] {sym} 开仓终止：IBKR 未连接，回退 ${total_new_locked:,.2f} 资金并放弃模拟记账。"
                )
                self._refund_locked_cash_once(
                    st, total_new_locked, "ENTRY_ABORT_REFUND", sym,
                    refund_key=f"ENTRY_ABORT_REFUND:{sym}:{int(curr_ts)}"
                )
                # 关键字段复位，避免 st.position=sig['dir'] 残留进状态落盘 → 后续误判为持仓
                self._reset_failed_entry_state(st)
                self._release_entry_pending(st)
                return

            self.orch.accounting._process_open_accounting(
                sym,
                st,
                st.qty,
                simulated_fill_price, # 👈 使用含滑点的价格进行财务记账
                stock_price,
                st.entry_ts,
                sig,
                duration=0.0,
                ratio=1.0,
                timing_fields=timing_fields,
                mode_override='BACKTEST',
                note_suffix=f"|liq_chunks={liq_chunks}|liq_reason={liq_reason}",
            )
            # 🚀 [修复 3] 执行完毕后立刻释放 pending 锁，形成完美闭环！
            st.is_pending = False
        elif self.orch.mode == 'realtime' and not SYNC_EXECUTION:
            trade = None
            runtime_trading_enabled = self._runtime_trading_enabled()
            try:    
                limit_price = self._get_entry_limit_price(sig, fill_price, attempt_no=0)
                if limit_price < 0.05:
                    # 下单前只预占 locked_cash，拒单时释放预占，不修改 cash。
                    self._refund_locked_cash_once(
                        st, total_new_locked, "ENTRY_LIMIT_TOO_LOW_REFUND", sym,
                        refund_key=f"ENTRY_LIMIT_TOO_LOW_REFUND:{sym}:{int(curr_ts)}"
                    )
                    self._reset_failed_entry_state(st)
                    self._release_entry_pending(st)
                    return
                entry_ref_price = float(sig.get('price', fill_price) or fill_price or 0.0)
                cap_price = self._entry_requote_cap_price(entry_ref_price)
                if limit_price > cap_price:
                    logger.warning(
                        f"🛑 [Entry Cap] {sym} 初始限价 {limit_price:.4f} 超过追价上限 {cap_price:.4f}，中止开仓。"
                    )
                    self._refund_locked_cash_once(
                        st, total_new_locked, "ENTRY_CAP_REJECT_REFUND", sym,
                        refund_key=f"ENTRY_CAP_REJECT_REFUND:{sym}:{int(curr_ts)}"
                    )
                    self._reset_failed_entry_state(st)
                    self._release_entry_pending(st)
                    return
                
                # Removing redundant limit_price override as fairness is now standard
                        
                tag = sig.get('tag')
                real_contract = None
                if hasattr(self.orch.ibkr, 'locked_contracts'):
                    real_contract = self.orch.ibkr.locked_contracts.get(sym, {}).get(tag)
                
                if not real_contract:
                    from ib_insync import Contract
                    real_contract = Contract()
                    real_contract.secType = 'OPT'
                    real_contract.symbol = sym
                    real_contract.localSymbol = st.contract_id
                    real_contract.exchange = 'SMART'
                    real_contract.currency = 'USD'
                
                stop_loss = abs(self.orch.strategy.cfg.STOP_LOSS)
                start_time = time.time()
                trade = self.orch.ibkr.place_option_order(real_contract, 'BUY', st.qty, 'LMT', lmt_price=limit_price, stop_loss_pct=stop_loss, custom_time=curr_ts)
                if trade:
                    st.is_pending = True
                    st.pending_action = 'BUY'
                    st.pending_side = 'BUY'
                    order_key = self._make_pending_order_key(sym, "OPEN", getattr(st, 'contract_id', ''), curr_ts)
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'OPEN',
                        'side': 'BUY',
                        'status': 'SUBMITTED',
                        'target_qty': int(st.qty),
                        'filled_qty': int(getattr(getattr(trade, 'orderStatus', None), 'filled', 0) or 0),
                        'remaining_qty': int(st.qty),
                        'limit_price': float(limit_price or 0.0),
                        'retry_count': 0,
                        'reserved_cash': float(total_new_locked or 0.0),
                        'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                        'submit_ts': float(start_time),
                        'last_update_ts': time.time(),
                        'reason': str(sig.get('reason', '') or ''),
                        **self._trade_identifiers(trade),
                    })
                    self._emit_order_event(
                        sym,
                        'ORDER_PENDING',
                        'BUY',
                        st.qty,
                        price=limit_price,
                        reason=sig.get('reason', ''),
                        extra={
                            'tag': sig.get('tag', ''),
                            'status': str(getattr(getattr(trade, 'orderStatus', None), 'status', '') or 'Submitted'),
                            'contract_id': getattr(st, 'contract_id', ''),
                            'pending_action': 'BUY',
                        },
                    )
                    asyncio.create_task(
                        self._monitor_realtime_order(
                            sym,
                            trade,
                            real_contract,
                            total_est_cost,
                            total_est_comm,
                            st.qty,
                            start_time,
                            limit_price,
                            stock_price,
                            sig,
                            st,
                            order_key,
                        )
                    )
                    return # 监控异步处理
                else:
                    if not runtime_trading_enabled:
                        logger.info(f"ℹ️ {sym} 开仓被拦截 (DRY RUN 极速结算)。")
                        # [REALTIME_DRY 无人为滑点]
                        # REALTIME_DRY 目标是尽量贴近“信号评估价(fair/mid)”做链路验证，
                        # 不再使用偏向成交保护的 limit_price(靠近 ask) 作为干跑成交价，
                        # 以避免开仓瞬间因点差出现 -1%~-2% 的人为浮亏观感。
                        try:
                            from config import IS_REALTIME_DRY as runtime_is_realtime_dry
                        except Exception:
                            runtime_is_realtime_dry = False
                        if runtime_is_realtime_dry:
                            simulated_fill_price = float(fill_price or limit_price or 0.0)
                        else:
                            slippage_entry_pct = self._effective_slippage_pct('entry')
                            simulated_fill_price = float(limit_price * (1 + slippage_entry_pct))
                        st.entry_price = simulated_fill_price
                        
                        log_ts = curr_ts if (self.orch.mode == 'backtest' or SYNC_EXECUTION) else time.time()
                        timing_fields = self._build_timing_fields(alpha_label_ts, alpha_available_ts, start_time, log_ts)
                        self.orch.accounting._process_open_accounting(
                            sym,
                            st,
                            st.qty,
                            simulated_fill_price,
                            stock_price,
                            log_ts,
                            sig,
                            duration=0.0,
                            ratio=1.0,
                            timing_fields=timing_fields,
                            mode_override=config.RUN_MODE,
                        )
                    else:
                        logger.error(f"❌ [实盘异常] {sym} 订单被拒！回退资金。")
                        self._refund_locked_cash_once(
                            st, (total_est_cost + total_est_comm), "ENTRY_ORDER_REJECT_REFUND", sym,
                            refund_key=f"ENTRY_ORDER_REJECT_REFUND:{sym}:{int(curr_ts)}"
                        )
                        self._reset_failed_entry_state(st)
            except Exception:
                self._reset_failed_entry_state(st)
                raise
            finally:
                # 只有在非监控模式下才在这里释放 (异步监控由 _monitor_realtime_order 负责解锁)
                if not (self.orch.mode == 'realtime' and trade is not None):
                    self._release_entry_pending(st)
                    
        self.orch.state_manager.save_state()

    async def _execute_exit(self, sym, sig, stock_price, curr_ts, batch_idx):
        st = self.orch.states[sym]
        
        # --- [Ghost Plan B: 退出点差保护] ---
        # 强制至少持仓 60 秒，防止因为开仓后的剧烈波动（或点差洗盘）立即被止损/触发平仓数据噪音
        is_urgent = self._is_urgent_exit_reason(sig.get('reason', ''))
        if not is_urgent and curr_ts - st.entry_ts < 60:
            logger.warning(f"🛡️ [Ghost B] {sym} 处于平仓保护期 (已持仓 {curr_ts - st.entry_ts:.1f}s < 60s)，忽略该信号。")
            return

        logger.info(f"🚪 [Exit 准入] {sym} | 信号: {sig.get('reason')} | 持仓: {st.position} | 数量: {st.qty} | Pending: {st.is_pending}")
        
        # 🛡️ 移除此处重复的 is_pending 检查，该逻辑已统一上移至 ExecutionEngineV8 处理。
        
        bid = sig.get('bid', 0.0)
        ask = sig.get('ask', 0.0)
        reason = sig['reason']
        is_urgent = self._is_urgent_exit_reason(reason)
        
        eval_price = sig.get('price', 0.0)
        base_market_price = sig.get('market_price', eval_price)
        raw_price = self.orch._get_fair_market_price(base_market_price, bid, ask, getattr(st, 'last_opt_price', 0.0))
            
        if math.isnan(raw_price):
            logger.error(f"❌ [Exit Error] {sym} 定价为 NaN，跳过本次平仓指令。")
            st.is_pending = False
            return

        sig['price'] = raw_price
        exit_qty = st.qty  

        logger.info(f"🚀 [Exit 定价完成] {sym} | 最终执行价: {raw_price:.2f} | 理由: {reason} | 紧急: {is_urgent}")

        #if self.orch.mode == 'backtest' or SYNC_EXECUTION:
        try:
            # 🚀 [对齐对冲] 只要处于强制确定性模式，就必须使用 Atomic Fill 对齐 S4
            is_deterministic = config.is_forced_deterministic(self.orch.r)
            is_simulated_env = (self.orch.mode == 'backtest' or is_deterministic)
            runtime_trading_enabled = self._runtime_trading_enabled()
            
            if is_simulated_env or SYNC_EXECUTION:
                # 在同步比对模式下，必须假设无限流动性，否则状态机必分叉！
                actual_fill_qty = exit_qty 
                slippage_exit_pct = self._effective_slippage_pct('exit')
                final_price = round(raw_price * (1 - slippage_exit_pct), 2)
                final_price = max(final_price, 0.01)
                
                # 🚀 [Bug2 修复] 从信号中恢复原始方向，防止共享内存下 st.position 已被 SE 清零
                original_position = sig.get('original_position', st.position)
                
                logger.info(f"📊 [Exit 记账-Backtest] {sym} | 结算价: {final_price} | 成交量: {actual_fill_qty} | 原始方向: {original_position}")
                self.orch.accounting._process_exit_accounting(sym, st, actual_fill_qty, final_price, stock_price, curr_ts, reason, 0.0, 1.0, original_position=original_position)
                
                contract = type('MockContract', (), {'symbol': sym, 'localSymbol': st.contract_id, 'tag': 'EXIT', 'secType': 'OPT'})()
                if hasattr(self.orch.ibkr, 'place_option_order'):
                    self.orch.ibkr.place_option_order(contract, 'SELL', actual_fill_qty, 'MKT', final_price, reason=reason, custom_time=curr_ts, stock_price=stock_price)
                
            elif self.orch.mode == 'realtime' and not SYNC_EXECUTION:
                real_contract = None
                if hasattr(self.orch.ibkr, 'locked_contracts'):
                    locks = self.orch.ibkr.locked_contracts.get(sym, {})
                    for tag, c in locks.items():
                        if c.localSymbol == st.contract_id:
                            real_contract = c; break
                
                if not real_contract:
                    from ib_insync import Contract
                    real_contract = Contract()
                    real_contract.secType = 'OPT'; real_contract.symbol = sym
                    real_contract.localSymbol = st.contract_id; real_contract.exchange = 'SMART'; real_contract.currency = 'USD'
                    
                st.is_pending = True 
                st.pending_action = 'SELL'
                st.pending_side = 'SELL'
                original_position = sig.get('original_position', st.position)
                
                if not runtime_trading_enabled:
                    # SELL 平仓始终吃买盘深度；bid_size 缺失时回退 ask_size。
                    available_size = sig.get('bid_size', sig.get('ask_size', 100))
                    actual_fill_qty = min(exit_qty, int(available_size))
                    if actual_fill_qty <= 0: return 
                    
                    slippage_exit_pct = self._effective_slippage_pct('exit')
                    simulated_exit_price = max(round(raw_price * (1 - slippage_exit_pct), 2), 0.01)
                    self.orch.accounting._process_exit_accounting(sym, st, actual_fill_qty, simulated_exit_price, stock_price, curr_ts, reason, 0.0, 1.0, original_position=original_position)
                else:
                    order_key = self._make_pending_order_key(sym, "CLOSE", getattr(st, 'contract_id', ''), curr_ts)
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'CLOSE',
                        'side': 'SELL',
                        'status': 'PENDING_SUBMIT',
                        'target_qty': int(exit_qty),
                        'filled_qty': 0,
                        'remaining_qty': int(exit_qty),
                        'limit_price': float(raw_price or 0.0),
                        'retry_count': 0,
                        'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                        'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                        'last_update_ts': time.time(),
                        'reason': str(reason or ''),
                    })
                    asyncio.create_task(self._smart_exit_order(sym, real_contract, exit_qty, raw_price, stock_price, curr_ts=curr_ts, is_force=is_urgent, bid=bid, ask=ask, reason=reason, order_key=order_key))
                    return # 监控异步处理
        finally:
            if not (self.orch.mode == 'realtime' and runtime_trading_enabled):
                st.is_pending = False
                st.pending_action = ''
                st.pending_side = None
            self.orch.state_manager.save_state()

    async def _smart_entry_order(self, sym, sig, stock_price, curr_ts, target_total_qty, chunks, fill_price, total_est_cost, total_est_comm, order_key=None):
        st = self.orch.states.get(sym)
        if not st: return
        
        # 🚀 [Ghost A Protections]
        MAX_ICEBERG_DURATION = 3.0     # 子单熔断超时
        MAX_SLIPPAGE_TOLERANCE = 0.02 # 滑点熔断阈值 (2%)
        
        try:
            logger.info(f"🧊 [Iceberg Start] {sym} 开始执行冰山拆单 (Ghost A Protection Enabled)...")
            if target_total_qty < 1: return
            runtime_trading_enabled = self._runtime_trading_enabled()
            commission_per_contract = float(self._cfg_value('COMMISSION_PER_CONTRACT', COMMISSION_PER_CONTRACT))
                
            base_chunk_qty = target_total_qty // chunks
            remainder_qty = target_total_qty % chunks
            
            tag = sig.get('tag')
            real_contract = None
            if hasattr(self.orch.ibkr, 'locked_contracts'):
                real_contract = self.orch.ibkr.locked_contracts.get(sym, {}).get(tag)
            
            total_qty_filled = 0
            total_actual_cost = 0.0
            total_actual_comm = 0.0
            iceberg_start_time = time.time()
            
            for i in range(chunks):
                if time.time() - iceberg_start_time > 15.0: # 总拆单不能过长 (15s)
                    logger.warning(f"🧊 [Iceberg Melt] {sym} 拆单总耗时过长，强制熔断结算。")
                    break

                chunk_num = i + 1
                qty_this_chunk = base_chunk_qty + (1 if i < remainder_qty else 0)
                if qty_this_chunk < 1: continue
                
                limit_price = fill_price  
                chunk_est_cost = qty_this_chunk * limit_price * 100
                chunk_est_comm = qty_this_chunk * commission_per_contract
                
                trade = self.orch.ibkr.place_option_order(real_contract, 'BUY', qty_this_chunk, 'LMT', lmt_price=limit_price, custom_time=curr_ts)
                if trade:
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'OPEN',
                        'side': 'BUY',
                        'status': 'SUBMITTED',
                        'target_qty': int(target_total_qty),
                        'filled_qty': int(total_qty_filled),
                        'remaining_qty': max(int(target_total_qty) - int(total_qty_filled), 0),
                        'limit_price': float(limit_price or 0.0),
                        'retry_count': int(max(i, 0)),
                        'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                        'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                        'last_update_ts': time.time(),
                        **self._trade_identifiers(trade),
                    })
                
                if not trade:
                    if not runtime_trading_enabled:
                        total_qty_filled += qty_this_chunk
                        total_actual_cost += chunk_est_cost
                        total_actual_comm += chunk_est_comm
                    continue
                    
                # [Ghost A] 3秒极速等待，超时不候
                try:
                    await asyncio.wait_for(self._wait_for_fill(trade), timeout=MAX_ICEBERG_DURATION)
                except asyncio.TimeoutError:
                    logger.warning(f"🧊 [Iceberg Timeout] {sym} 子单 {chunk_num} 超时，执行 Melt 熔断。")
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'OPEN',
                        'side': 'BUY',
                        'status': 'CANCEL_REQUESTED',
                        'target_qty': int(target_total_qty),
                        'filled_qty': int(total_qty_filled),
                        'remaining_qty': max(int(target_total_qty) - int(total_qty_filled), 0),
                        'limit_price': float(limit_price or 0.0),
                        'retry_count': int(max(i, 0)),
                        'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                        'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                        'last_update_ts': time.time(),
                        **self._trade_identifiers(trade),
                    })
                    if hasattr(self.orch.ibkr, 'ib'):
                        self.orch.ibkr.ib.cancelOrder(trade.order)
                    await asyncio.sleep(0.5)

                filled_qty = int(trade.orderStatus.filled)
                avg_fill_price = float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else float(limit_price)
                
                # [Ghost A] 价格偏离熔断
                if avg_fill_price > limit_price * (1 + MAX_SLIPPAGE_TOLERANCE):
                    logger.warning(f"🧊 [Iceberg Slippage] {sym} 价格跳涨过快 ({avg_fill_price} > {limit_price}), 直接熔断！")
                    total_qty_filled += filled_qty
                    break

                total_qty_filled += filled_qty
                total_actual_cost += filled_qty * avg_fill_price * 100
                total_actual_comm += filled_qty * commission_per_contract
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'PARTIAL_FILLED' if total_qty_filled < target_total_qty else 'FILLED',
                    'target_qty': int(target_total_qty),
                    'filled_qty': int(total_qty_filled),
                    'remaining_qty': max(int(target_total_qty) - int(total_qty_filled), 0),
                    'limit_price': float(avg_fill_price or limit_price or 0.0),
                    'retry_count': int(max(i, 0)),
                    'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                    'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'last_update_ts': time.time(),
                    **self._trade_identifiers(trade),
                })
                
                if total_qty_filled >= target_total_qty:
                    break

                if chunk_num < chunks:
                    await asyncio.sleep(0.5) # 降低间隔，加速成交

            unspent_cash = (total_est_cost + total_est_comm) - (total_actual_cost + total_actual_comm)
            self._refund_locked_cash_once(
                st, unspent_cash, "ICEBERG_UNSPENT_REFUND", sym,
                refund_key=f"ICEBERG_UNSPENT_REFUND:{sym}:{int(curr_ts)}"
            )
            
            if total_qty_filled > 0:
                st.open_fill_confirmed = False
                st.position = sig['dir']
                st.qty = total_qty_filled
                st.entry_stock = stock_price
                st.entry_price = (total_actual_cost / total_qty_filled) / 100.0
                st.opt_type = 'call' if st.position == 1 else 'put'
                st.entry_ts = curr_ts
                # [🛡️ Defensive] 同上方 _execute_entry 的保护: 赋值即强转, 拒绝 dict.
                _meta2 = sig.get('meta', {}) or {}
                def _mf2(v, d=0.0):
                    if isinstance(v, dict): return d
                    try: return float(v)
                    except (TypeError, ValueError): return d
                def _mi2(v, d=0):
                    if isinstance(v, dict): return d
                    try: return int(v)
                    except (TypeError, ValueError): return d
                st.entry_spy_roc = _mf2(_meta2.get('spy_roc', 0.0))
                st.entry_index_trend = _mi2(_meta2.get('index_trend', 0))
                st.entry_alpha_z = _mf2(_meta2.get('alpha_z', 0.0))
                st.entry_iv = _mf2(_meta2.get('iv', st.last_valid_iv))
                st.max_roi = 0.0
                st.contract_id = _meta2.get('contract_id', '') or ''
                
                fill_ts = time.time()
                self.orch.accounting._process_open_accounting(
                    sym, st, total_qty_filled, st.entry_price, stock_price, fill_ts, sig,
                    duration=fill_ts - iceberg_start_time, ratio=(total_qty_filled / target_total_qty),
                    mode_override='REALTIME', note_suffix=f"|GHOST_A_{chunks}",
                )
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'FILLED' if total_qty_filled >= target_total_qty else 'CANCELLED',
                    'target_qty': int(target_total_qty),
                    'filled_qty': int(total_qty_filled),
                    'remaining_qty': max(int(target_total_qty) - int(total_qty_filled), 0),
                    'limit_price': float(getattr(st, 'entry_price', 0.0) or 0.0),
                    'retry_count': int(max(chunks - 1, 0)),
                    'reserved_cash': 0.0,
                    'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                })
            else:
                self._reset_failed_entry_state(st)
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'CANCELLED',
                    'target_qty': int(target_total_qty),
                    'filled_qty': 0,
                    'remaining_qty': int(target_total_qty),
                    'limit_price': float(fill_price or 0.0),
                    'retry_count': int(max(chunks - 1, 0)),
                    'reserved_cash': 0.0,
                    'slot_reserved': False,
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                })

        except Exception as e:
            logger.error(f"🚨 [Iceberg Error] {sym}: {e}", exc_info=True)
            self._upsert_pending_order(order_key, {
                'symbol': sym,
                'contract_id': getattr(st, 'contract_id', ''),
                'intent': 'OPEN',
                'side': 'BUY',
                'status': 'ERROR',
                'target_qty': int(target_total_qty),
                'filled_qty': 0,
                'remaining_qty': int(target_total_qty),
                'limit_price': float(fill_price or 0.0),
                'last_update_ts': time.time(),
                'terminal_ts': time.time(),
                'is_terminal': True,
                'last_error': str(e),
            })
        finally:
            try:
                st._async_entry_order_active = False
            except Exception:
                pass
            st.locked_cash = 0  
            st.is_pending = False # [Ghost C]
            st.pending_action = ''
            st.pending_side = None
            self.orch.state_manager.save_state()

    async def _wait_for_fill(self, trade):
        """Helper to wait for trade fill status"""
        while trade.orderStatus.status not in ['Filled', 'Cancelled', 'Inactive']:
            await asyncio.sleep(0.5)
            if trade.orderStatus.status == 'Filled':
                return True
        return trade.orderStatus.status == 'Filled'

    async def _monitor_realtime_order(self, sym, trade, real_contract, cost, commission, expected_qty, start_time, limit_price, stock_price, sig, st, order_key=None):
        """实盘超时看门狗"""
        try:
            commission_per_contract = float(self._cfg_value('COMMISSION_PER_CONTRACT', COMMISSION_PER_CONTRACT))
            max_attempts = max(1, int(self._cfg_value('EXIT_ORDER_MAX_RETRIES', getattr(config, 'EXIT_ORDER_MAX_RETRIES', config.ORDER_MAX_RETRIES))))
            wait_per_attempt = max(1, int(self._cfg_value('ORDER_TIMEOUT_SECONDS', config.ORDER_TIMEOUT_SECONDS)))
            total_filled_qty = 0
            total_actual_cost = 0.0
            remaining_qty = int(expected_qty)
            current_trade = trade
            current_limit_price = float(limit_price)
            entry_ref_price = float(sig.get('price', limit_price) or limit_price or 0.0)
            cap_price = self._entry_requote_cap_price(entry_ref_price)

            for attempt_no in range(max_attempts):
                for _ in range(wait_per_attempt):
                    await asyncio.sleep(1)
                    if current_trade.orderStatus.status == 'Filled':
                        break

                filled_qty = int(current_trade.orderStatus.filled)
                avg_fill_price = float(current_trade.orderStatus.avgFillPrice) if current_trade.orderStatus.avgFillPrice else float(current_limit_price)
                broker_status = str(getattr(current_trade.orderStatus, 'status', '') or '')
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'PARTIAL_FILLED' if filled_qty > 0 and broker_status != 'Filled' else (broker_status or 'SUBMITTED'),
                    'target_qty': int(expected_qty),
                    'filled_qty': int(filled_qty),
                    'remaining_qty': max(int(expected_qty) - int(filled_qty), 0),
                    'limit_price': float(current_limit_price or 0.0),
                    'retry_count': int(attempt_no),
                    'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                    'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'last_update_ts': time.time(),
                    **self._trade_identifiers(current_trade),
                })

                if current_trade.orderStatus.status != 'Filled':
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'OPEN',
                        'side': 'BUY',
                        'status': 'CANCEL_REQUESTED',
                        'target_qty': int(expected_qty),
                        'filled_qty': int(filled_qty),
                        'remaining_qty': max(int(expected_qty) - int(filled_qty), 0),
                        'limit_price': float(current_limit_price or 0.0),
                        'retry_count': int(attempt_no),
                        'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                        'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                        'last_update_ts': time.time(),
                        **self._trade_identifiers(current_trade),
                    })
                    if hasattr(self.orch.ibkr, 'ib') and current_trade.orderStatus.status not in ['Cancelled', 'Inactive', 'ApiCancelled']:
                        self.orch.ibkr.ib.cancelOrder(current_trade.order)
                    await asyncio.sleep(2)
                    filled_qty = int(current_trade.orderStatus.filled)
                    avg_fill_price = float(current_trade.orderStatus.avgFillPrice) if current_trade.orderStatus.avgFillPrice else float(current_limit_price)

                total_filled_qty += filled_qty
                total_actual_cost += filled_qty * avg_fill_price * 100
                remaining_qty = max(expected_qty - total_filled_qty, 0)

                if remaining_qty <= 0:
                    break

                if attempt_no >= max_attempts - 1:
                    break

                current_limit_price = self._next_entry_requote_price(
                    sig=sig,
                    prev_limit_price=current_limit_price,
                    attempt_no=attempt_no + 1,
                    cap_price=cap_price,
                    real_contract=real_contract,
                )
                if current_limit_price < 0.05:
                    break
                if current_limit_price > cap_price:
                    logger.warning(
                        f"🛑 [Entry Requote Cap] {sym} 追价 {current_limit_price:.4f} 超过上限 {cap_price:.4f}，停止追价。"
                    )
                    break

                next_trade = self.orch.ibkr.place_option_order(
                    real_contract,
                    'BUY',
                    remaining_qty,
                    'LMT',
                    lmt_price=current_limit_price,
                    custom_time=sig.get('meta', {}).get('alpha_available_ts', time.time()),
                    reason=f"{sig.get('reason', '')}|REQUOTE_{attempt_no + 1}",
                    stock_price=stock_price,
                )
                if not next_trade:
                    break
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'REPLACED',
                    'target_qty': int(expected_qty),
                    'filled_qty': int(total_filled_qty),
                    'remaining_qty': int(remaining_qty),
                    'limit_price': float(current_limit_price or 0.0),
                    'retry_count': int(attempt_no + 1),
                    'reserved_cash': float(getattr(st, 'locked_cash', 0.0) or 0.0),
                    'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'last_update_ts': time.time(),
                    **self._trade_identifiers(next_trade),
                })
                current_trade = next_trade

            actual_commission = total_filled_qty * commission_per_contract
            refund = (cost + commission) - (total_actual_cost + actual_commission)
            self._refund_locked_cash_once(
                st, refund, "REALTIME_REQUOTE_REFUND", sym,
                refund_key=f"REALTIME_REQUOTE_REFUND:{sym}:{int(start_time)}"
            )
            
            if total_filled_qty > 0:
                avg_fill_price = (total_actual_cost / total_filled_qty) / 100.0
                ratio = total_filled_qty / expected_qty if expected_qty > 0 else 0.0
                self._emit_order_event(
                    sym,
                    'ORDER_FILLED',
                    'BUY',
                    total_filled_qty,
                    price=avg_fill_price,
                    reason=sig.get('reason', ''),
                    extra={
                        'expected_qty': int(expected_qty),
                        'filled_qty': int(total_filled_qty),
                        'fill_ratio': float(ratio),
                        'status': 'Filled' if total_filled_qty >= expected_qty else 'PartialFill',
                    },
                )
                remaining_qty = max(int(expected_qty) - int(total_filled_qty), 0)
                if remaining_qty > 0:
                    self._emit_order_event(
                        sym,
                        'ORDER_CANCELLED',
                        'BUY',
                        remaining_qty,
                        price=current_limit_price,
                        reason=f"{sig.get('reason', '')}|UNFILLED_REMAINDER",
                        extra={
                            'expected_qty': int(expected_qty),
                            'filled_qty': int(total_filled_qty),
                            'status': str(getattr(getattr(current_trade, 'orderStatus', None), 'status', '') or 'Cancelled'),
                        },
                    )
                alpha_label_ts = sig.get('meta', {}).get('alpha_label_ts', 0.0)
                alpha_available_ts = sig.get('meta', {}).get('alpha_available_ts', 0.0)
                fill_ts = time.time()
                timing_fields = self._build_timing_fields(alpha_label_ts, alpha_available_ts, start_time, fill_ts)
                self.orch.accounting._process_open_accounting(
                    sym,
                    st,
                    total_filled_qty,
                    avg_fill_price,
                    stock_price,
                    fill_ts,
                    sig,
                    duration=fill_ts - start_time,
                    ratio=ratio,
                    timing_fields=timing_fields,
                    mode_override='REALTIME',
                )
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'FILLED' if total_filled_qty >= expected_qty else 'CANCELLED',
                    'target_qty': int(expected_qty),
                    'filled_qty': int(total_filled_qty),
                    'remaining_qty': max(int(expected_qty) - int(total_filled_qty), 0),
                    'limit_price': float(avg_fill_price or current_limit_price or 0.0),
                    'retry_count': int(max_attempts - 1),
                    'reserved_cash': 0.0,
                    'slot_reserved': bool(getattr(st, 'entry_slot_reserved', False)),
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                    **self._trade_identifiers(current_trade),
                })
            else:
                self._emit_order_event(
                    sym,
                    'ORDER_CANCELLED',
                    'BUY',
                    expected_qty,
                    price=current_limit_price,
                    reason=sig.get('reason', ''),
                    extra={
                        'expected_qty': int(expected_qty),
                        'filled_qty': 0,
                        'status': str(getattr(getattr(current_trade, 'orderStatus', None), 'status', '') or 'Cancelled'),
                    },
                )
                self._reset_failed_entry_state(st)
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'OPEN',
                    'side': 'BUY',
                    'status': 'CANCELLED',
                    'target_qty': int(expected_qty),
                    'filled_qty': 0,
                    'remaining_qty': int(expected_qty),
                    'limit_price': float(current_limit_price or 0.0),
                    'retry_count': int(max_attempts - 1),
                    'reserved_cash': 0.0,
                    'slot_reserved': False,
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                    **self._trade_identifiers(current_trade),
                })

        except Exception as e:
            logger.error(f"🚨 [Monitor Error] {sym}: {e}", exc_info=True)
            self._upsert_pending_order(order_key, {
                'symbol': sym,
                'contract_id': getattr(st, 'contract_id', ''),
                'intent': 'OPEN',
                'side': 'BUY',
                'status': 'ERROR',
                'target_qty': int(expected_qty),
                'filled_qty': int(getattr(getattr(trade, 'orderStatus', None), 'filled', 0) or 0),
                'remaining_qty': int(expected_qty),
                'limit_price': float(limit_price or 0.0),
                'last_update_ts': time.time(),
                'terminal_ts': time.time(),
                'is_terminal': True,
                'last_error': str(e),
                **self._trade_identifiers(trade),
            })
        finally:
            if st:
                st.is_pending = False
                st.pending_action = ''
                st.pending_side = None
            self.orch.state_manager.save_state()

    async def _smart_exit_order(self, sym, real_contract, total_qty, base_price, stock_price, curr_ts=None, is_force=False, bid=0.0, ask=0.0, reason="", order_key=None):
        """实盘防滑点平仓订单执行器"""
        st = self.orch.states.get(sym)
        if not st: return
        start_time = time.time()
        runtime_trading_enabled = self._runtime_trading_enabled()

        # [🛡️ Connection Guard] 实盘路径下, 若 IBKR 断连则直接跳过本次 exit:
        #   - 不调用 place_option_order (避免 ib_insync 抛 ConnectionError 刷屏);
        #   - 不走 "not trade → 模拟成交" 兜底 (那条路径是 DRY RUN 语义, 实盘用会
        #     让账本清掉实际仍持有的仓位, 严重账实不符);
        #   - is_pending 重置为 False, 让下一个 exit 信号到达时能重新尝试.
        if runtime_trading_enabled and not self._ibkr_is_connected():
            self._log_ibkr_disconnect_throttled(sym, reason)
            try:
                st.is_pending = False
            except Exception:
                pass
            return

        try:
            slippage_exit_pct = self._effective_slippage_pct('exit')
            configured_exit_order_type = str(self._cfg_value('EXIT_ORDER_TYPE', EXIT_ORDER_TYPE)).upper()
            is_urgent = self._is_urgent_exit_reason(reason, is_force=is_force)
            effective_exit_order_type = self._resolve_exit_order_type(
                configured_exit_order_type,
                reason,
                is_force=is_force,
            )
            max_attempts = max(1, int(self._cfg_value('ORDER_MAX_RETRIES', config.ORDER_MAX_RETRIES)))
            order_timeout_seconds = max(1, int(self._cfg_value('ORDER_TIMEOUT_SECONDS', config.ORDER_TIMEOUT_SECONDS)))

            if configured_exit_order_type == 'MKT' and effective_exit_order_type == 'LMT':
                logger.info(
                    f"🧊 [Exit Policy] {sym} non-urgent exit uses LMT instead of configured MKT | reason={reason}"
                )

            if effective_exit_order_type == 'MKT':
                trade = self.orch.ibkr.place_option_order(real_contract, 'SELL', total_qty, 'MKT', base_price, custom_time=curr_ts, reason=reason)
                if trade:
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'CLOSE',
                        'side': 'SELL',
                        'status': str(getattr(getattr(trade, 'orderStatus', None), 'status', '') or 'SUBMITTED').upper(),
                        'target_qty': int(total_qty),
                        'filled_qty': int(getattr(getattr(trade, 'orderStatus', None), 'filled', 0) or 0),
                        'remaining_qty': int(total_qty),
                        'limit_price': float(base_price or 0.0),
                        'retry_count': 0,
                        'last_update_ts': time.time(),
                        **self._trade_identifiers(trade),
                    })
                if not trade:
                    # [🛡️ Safe Fallback] place_option_order 返回 None 有两种语义:
                    #   (a) DRY RUN / TRADING_ENABLED=False → 应该用模拟价清算账本;
                    #   (b) 实盘但 IBKR 断连 → 绝不能模拟清算 (会让账本和持仓不一致).
                    # 用 last_not_connected_ts 窗口 + TRADING_ENABLED 联合判断.
                    if runtime_trading_enabled and self._ibkr_recently_failed():
                        self._log_ibkr_disconnect_throttled(sym, reason)
                    else:
                        simulated_exit_price = max(round(base_price * (1 - slippage_exit_pct), 2), 0.01)
                        self.orch.accounting._process_exit_accounting(sym, st, total_qty, simulated_exit_price, stock_price, curr_ts, reason, 0.0, 1.0)
                return
                
            wait_per_attempt = max(1, order_timeout_seconds // 2) if is_urgent else order_timeout_seconds
            remaining_qty = int(total_qty)
            total_filled_qty = 0
            total_exit_value = 0.0

            limit_sell_price = self._get_exit_limit_price(base_price, bid=bid, ask=ask, is_urgent=is_urgent, attempt_no=0)
            current_trade = self.orch.ibkr.place_option_order(real_contract, 'SELL', remaining_qty, 'LMT', lmt_price=limit_sell_price, custom_time=curr_ts, reason=reason)
            if not current_trade:
                # [🛡️ Safe Fallback] 同上 MKT 分支: 区分 DRY RUN (模拟成交) 与
                # 实盘连接断开 (跳过, 等待重连后由下一个 exit 信号重试).
                if runtime_trading_enabled and self._ibkr_recently_failed():
                    self._log_ibkr_disconnect_throttled(sym, reason)
                    return
                simulated_exit_price = max(round(limit_sell_price * (1 - slippage_exit_pct), 2), 0.01)
                self.orch.accounting._process_exit_accounting(sym, st, total_qty, simulated_exit_price, stock_price, curr_ts, reason, 0.0, 1.0)
                return
            self._upsert_pending_order(order_key, {
                'symbol': sym,
                'contract_id': getattr(st, 'contract_id', ''),
                'intent': 'CLOSE',
                'side': 'SELL',
                'status': 'SUBMITTED',
                'target_qty': int(total_qty),
                'filled_qty': 0,
                'remaining_qty': int(total_qty),
                'limit_price': float(limit_sell_price or 0.0),
                'retry_count': 0,
                'last_update_ts': time.time(),
                **self._trade_identifiers(current_trade),
            })

            self._emit_order_event(
                sym,
                'ORDER_PENDING',
                'SELL',
                total_qty,
                price=limit_sell_price,
                reason=reason,
                extra={
                    'status': str(getattr(getattr(current_trade, 'orderStatus', None), 'status', '') or 'Submitted'),
                    'contract_id': getattr(st, 'contract_id', ''),
                    'pending_action': 'SELL',
                },
            )

            for attempt_no in range(max_attempts):
                for _ in range(wait_per_attempt):
                    await asyncio.sleep(1)
                    if current_trade.orderStatus.status == 'Filled':
                        break

                filled_qty = int(current_trade.orderStatus.filled)
                avg_p = float(current_trade.orderStatus.avgFillPrice) if current_trade.orderStatus.avgFillPrice else float(limit_sell_price)
                broker_status = str(getattr(current_trade.orderStatus, 'status', '') or '')
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'CLOSE',
                    'side': 'SELL',
                    'status': 'PARTIAL_FILLED' if filled_qty > 0 and broker_status != 'Filled' else (broker_status or 'SUBMITTED'),
                    'target_qty': int(total_qty),
                    'filled_qty': int(filled_qty),
                    'remaining_qty': max(int(total_qty) - int(filled_qty), 0),
                    'limit_price': float(limit_sell_price or 0.0),
                    'retry_count': int(attempt_no),
                    'last_update_ts': time.time(),
                    **self._trade_identifiers(current_trade),
                })

                if current_trade.orderStatus.status != 'Filled':
                    self._upsert_pending_order(order_key, {
                        'symbol': sym,
                        'contract_id': getattr(st, 'contract_id', ''),
                        'intent': 'CLOSE',
                        'side': 'SELL',
                        'status': 'CANCEL_REQUESTED',
                        'target_qty': int(total_qty),
                        'filled_qty': int(filled_qty),
                        'remaining_qty': max(int(total_qty) - int(filled_qty), 0),
                        'limit_price': float(limit_sell_price or 0.0),
                        'retry_count': int(attempt_no),
                        'last_update_ts': time.time(),
                        **self._trade_identifiers(current_trade),
                    })
                    if hasattr(self.orch.ibkr, 'ib') and current_trade.orderStatus.status not in ['Cancelled', 'Inactive', 'ApiCancelled']:
                        self.orch.ibkr.ib.cancelOrder(current_trade.order)
                    await asyncio.sleep(2)
                    filled_qty = int(current_trade.orderStatus.filled)
                    avg_p = float(current_trade.orderStatus.avgFillPrice) if current_trade.orderStatus.avgFillPrice else float(limit_sell_price)

                total_filled_qty += filled_qty
                total_exit_value += filled_qty * avg_p
                remaining_qty = max(total_qty - total_filled_qty, 0)

                if remaining_qty <= 0:
                    break

                if attempt_no >= max_attempts - 1:
                    break

                limit_sell_price = self._get_exit_limit_price(
                    base_price,
                    bid=bid,
                    ask=ask,
                    is_urgent=is_urgent,
                    attempt_no=attempt_no + 1,
                )
                next_trade = self.orch.ibkr.place_option_order(
                    real_contract,
                    'SELL',
                    remaining_qty,
                    'LMT',
                    lmt_price=limit_sell_price,
                    custom_time=curr_ts,
                    reason=f"{reason}|REQUOTE_{attempt_no + 1}",
                )
                if not next_trade:
                    break
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'CLOSE',
                    'side': 'SELL',
                    'status': 'REPLACED',
                    'target_qty': int(total_qty),
                    'filled_qty': int(total_filled_qty),
                    'remaining_qty': int(remaining_qty),
                    'limit_price': float(limit_sell_price or 0.0),
                    'retry_count': int(attempt_no + 1),
                    'last_update_ts': time.time(),
                    **self._trade_identifiers(next_trade),
                })
                current_trade = next_trade

            if total_filled_qty > 0:
                avg_exit_price = total_exit_value / total_filled_qty
                fill_ratio = total_filled_qty / total_qty if total_qty > 0 else 0.0
                self._emit_order_event(
                    sym,
                    'ORDER_FILLED',
                    'SELL',
                    total_filled_qty,
                    price=avg_exit_price,
                    reason=reason,
                    extra={
                        'expected_qty': int(total_qty),
                        'filled_qty': int(total_filled_qty),
                        'fill_ratio': float(fill_ratio),
                        'status': 'Filled' if total_filled_qty >= total_qty else 'PartialFill',
                    },
                )
                remaining_qty = max(int(total_qty) - int(total_filled_qty), 0)
                if remaining_qty > 0:
                    self._emit_order_event(
                        sym,
                        'ORDER_CANCELLED',
                        'SELL',
                        remaining_qty,
                        price=limit_sell_price,
                        reason=f"{reason}|UNFILLED_REMAINDER",
                        extra={
                            'expected_qty': int(total_qty),
                            'filled_qty': int(total_filled_qty),
                            'status': str(getattr(getattr(current_trade, 'orderStatus', None), 'status', '') or 'Cancelled'),
                        },
                    )
                self.orch.accounting._process_exit_accounting(
                    sym,
                    st,
                    total_filled_qty,
                    avg_exit_price,
                    stock_price,
                    curr_ts,
                    reason,
                    time.time() - start_time,
                    fill_ratio,
                )
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'CLOSE',
                    'side': 'SELL',
                    'status': 'FILLED' if total_filled_qty >= total_qty else 'CANCELLED',
                    'target_qty': int(total_qty),
                    'filled_qty': int(total_filled_qty),
                    'remaining_qty': max(int(total_qty) - int(total_filled_qty), 0),
                    'limit_price': float(avg_exit_price or limit_sell_price or 0.0),
                    'retry_count': int(max_attempts - 1),
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                    **self._trade_identifiers(current_trade),
                })
            else:
                self._emit_order_event(
                    sym,
                    'ORDER_CANCELLED',
                    'SELL',
                    total_qty,
                    price=limit_sell_price,
                    reason=reason,
                    extra={
                        'expected_qty': int(total_qty),
                        'filled_qty': 0,
                        'status': str(getattr(getattr(current_trade, 'orderStatus', None), 'status', '') or 'Cancelled'),
                    },
                )
                self._upsert_pending_order(order_key, {
                    'symbol': sym,
                    'contract_id': getattr(st, 'contract_id', ''),
                    'intent': 'CLOSE',
                    'side': 'SELL',
                    'status': 'CANCELLED',
                    'target_qty': int(total_qty),
                    'filled_qty': 0,
                    'remaining_qty': int(total_qty),
                    'limit_price': float(limit_sell_price or 0.0),
                    'retry_count': int(max_attempts - 1),
                    'last_update_ts': time.time(),
                    'terminal_ts': time.time(),
                    'is_terminal': True,
                    **self._trade_identifiers(current_trade),
                })

        except Exception as e:
            logger.error(f"🚨 [Exit Error] {sym}: {e}", exc_info=True)
            self._upsert_pending_order(order_key, {
                'symbol': sym,
                'contract_id': getattr(st, 'contract_id', ''),
                'intent': 'CLOSE',
                'side': 'SELL',
                'status': 'ERROR',
                'target_qty': int(total_qty),
                'filled_qty': 0,
                'remaining_qty': int(total_qty),
                'limit_price': float(base_price or 0.0),
                'last_update_ts': time.time(),
                'terminal_ts': time.time(),
                'is_terminal': True,
                'last_error': str(e),
            })
        finally:
            st.is_pending = False
            st.pending_action = ''
            st.pending_side = None
            self.orch.state_manager.save_state()

    async def force_close_all(self):
        """用于在回放结束时清理所有未平仓位"""
        has_pos = any(st.position != 0 for st in self.orch.states.values())
        if not has_pos:
            return

        logger.warning("🧹 [End of Replay] Forcing all open positions to close.")
        
        # 构造空 batch 即可触发 _force_clear_all 中的 fallback 逻辑 (使用 st.entry_stock)
        dummy_batch = {'symbols': [], 'stock_price': []}
        
        # 获取逻辑时间戳 (优先使用 last_curr_ts)
        curr_ts = getattr(self.orch, 'last_curr_ts', time.time())
        from pytz import timezone
        ny_now = datetime.fromtimestamp(curr_ts, tz=timezone('America/New_York'))
        
        await self._force_clear_all(dummy_batch, "REPLAY_FINAL_CLEAR", curr_ts, ny_now)

    async def _force_clear_all(self, batch, reason, custom_ts, custom_dt):
        """强制平仓所有持仓"""
        symbols = batch['symbols']
        stock_prices = batch['stock_price']
        
        for sym, st in self.orch.states.items():
            if st.position != 0:
                in_batch = sym in symbols
                idx = symbols.index(sym) if in_batch else -1
                curr_stock = float(stock_prices[idx]) if in_batch else st.entry_stock
                
                is_simulated_env = (self.orch.mode == 'backtest')
                if is_simulated_env or SYNC_EXECUTION:
                    if in_batch:
                        opt_data = self.orch._get_opt_data_backtest(batch, idx, sym, st)
                        raw_price = self.orch._get_fair_market_price(
                            opt_data.get('call_price' if st.position==1 else 'put_price', 0), 
                            opt_data.get('call_bid' if st.position==1 else 'put_bid', 0), 
                            opt_data.get('call_ask' if st.position==1 else 'put_ask', 0),
                            getattr(st, 'last_opt_price', 0.0)
                        )
                    else:
                        # 🚨 [核心修复] 尾盘空 batch 强平时，强制使用 SE 传来的最新市价
                        raw_price = getattr(st, 'last_opt_price', 0.0)
                        
                        # 🛡️ [防投毒防线] 如果期权价竟然大于股价的 50%，显然是数据打架 (把股价错当成了期权价)
                        if raw_price > curr_stock * 0.5:
                            logger.error(f"🚨 [Price Poison] {sym} raw_price({raw_price}) suspiciously high vs stock({curr_stock})! Falling back to entry_price.")
                            raw_price = st.entry_price

                        if raw_price < 0.05:
                            raw_price = st.entry_price if st.entry_price > 0 else 0.01

                    slippage_exit_pct = self._effective_slippage_pct('exit')
                    final_p = max(round(raw_price * (1 - slippage_exit_pct), 2), 0.01)
                    self.orch.accounting._process_exit_accounting(sym, st, st.qty, final_p, curr_stock, custom_ts, f"FORCE_{reason}", 0.0, 1.0)
                    if hasattr(self.orch.ibkr, 'place_option_order'):
                        contract = type('MockContract', (), {'symbol': sym, 'localSymbol': st.contract_id, 'tag': 'EXIT', 'secType': 'OPT'})()
                        self.orch.ibkr.place_option_order(contract, 'SELL', st.qty, 'MKT', final_p, reason=f"FORCE_{reason}", custom_time=custom_ts, stock_price=curr_stock)
                else:
                    runtime_trading_enabled = self._runtime_trading_enabled()
                    st.is_pending = True 
                    if not runtime_trading_enabled:
                        slippage_exit_pct = self._effective_slippage_pct('exit')
                        sim_exit = max(round(raw_price * (1 - slippage_exit_pct), 2), 0.01)
                        self.orch.accounting._process_exit_accounting(sym, st, st.qty, sim_exit, curr_stock, custom_ts, f"FORCE_{reason}", 0.0, 1.0)
                        st.is_pending = False
                    else:
                        asyncio.create_task(self._smart_exit_order(sym, None, st.qty, raw_price, curr_stock, curr_ts=custom_ts, is_force=True, reason=f"FORCE_{reason}"))
        self.orch.state_manager.save_state()

    async def _emergency_cancel_all(self, reason="Unknown"):
        if self.orch.mode == 'backtest': return
        logger.critical(f"🚨 EMERGENCY: Global Cancel! Reason: {reason}")
        self.orch.trading_paused = True 
        if self.orch.mode == 'realtime' and hasattr(self.orch.ibkr, 'ib') and self.orch.ibkr.ib.isConnected():
            try:
                self.orch.ibkr.ib.reqGlobalCancel()
            except Exception as e:
                logger.error(f"❌ Global Cancel Failed: {e}")
