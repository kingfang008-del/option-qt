import json
import time
import logging
import os
import threading
import pickle
import hashlib
import re
from pathlib import Path
from datetime import datetime
from pytz import timezone
import psycopg2
from psycopg2 import sql
import sqlite3
import numpy as np
from config import (
    PG_DB_URL,
    IS_SIMULATED,
    IS_BACKTEST,
    IS_REALTIME_DRY,
    RUN_MODE,
    OMS_STATE_NAMESPACE,
    INITIAL_ACCOUNT,
    ROLLING_WINDOW,
    TARGET_SYMBOLS,
    DB_DIR,
    REDIS_CFG,
    NY_TZ,
)
try:
    from Domain.shadow_router import get_domain_shadow_router
except Exception:  # pragma: no cover
    get_domain_shadow_router = None

logger = logging.getLogger("V8_Orchestrator.StateManager")


def safe_state_partition_name(namespace: str) -> str:
    """Build a short, safe PostgreSQL partition table name for a namespace."""
    raw = re.sub(r"[^a-zA-Z0-9_]+", "_", str(namespace or "default").lower()).strip("_")
    raw = raw or "default"
    digest = hashlib.md5(str(namespace or "default").encode("utf-8")).hexdigest()[:8]
    return f"symbol_state_{raw[:38]}_{digest}"


def tag_state_snapshot_rows(state_data, *, namespace=OMS_STATE_NAMESPACE, run_mode=RUN_MODE):
    """Stamp every state row with mode/namespace metadata before persistence."""
    for data in state_data.values():
        if isinstance(data, dict):
            data.setdefault("mode", run_mode)
            data.setdefault("state_namespace", namespace)
    return state_data


def infer_open_fill_confirmed(state_row) -> bool:
    """Best-effort detection for whether a position has a confirmed OPEN fill."""
    if not isinstance(state_row, dict):
        return False
    explicit = state_row.get("open_fill_confirmed")
    if explicit is not None:
        return bool(explicit)
    try:
        pos = int(state_row.get("position", 0) or 0)
        qty = int(state_row.get("qty", 0) or 0)
        entry_price = float(state_row.get("entry_price", 0.0) or 0.0)
        entry_ts = float(state_row.get("entry_ts", 0.0) or 0.0)
    except Exception:
        return False
    return pos != 0 and qty > 0 and entry_price > 0 and entry_ts > 0


def zero_position_state_row(state_row):
    """Drop trading-state fields but keep warmup/history buffers."""
    cleaned = dict(state_row or {})
    cleaned.update({
        "position": 0,
        "qty": 0,
        "entry_price": 0.0,
        "entry_stock": 0.0,
        "entry_ts": 0.0,
        "entry_spy_roc": 0.0,
        "entry_index_trend": 0,
        "entry_alpha_z": 0.0,
        "entry_iv": 0.0,
        "max_roi": -1.0,
        "cooldown_until": 0.0,
        "entry_slot_reserved": False,
        "open_fill_confirmed": False,
    })
    return cleaned


def sanitize_restored_mock_cash(
    restored_cash,
    *,
    initial_cash=INITIAL_ACCOUNT,
    is_realtime_dry=IS_REALTIME_DRY,
    max_multiplier=None,
):
    """Validate restored mock cash before it can become OMS truth.

    REALTIME_DRY is a paper ledger. If a legacy/corrupted _GLOBAL_STATE_ says
    the paper cash is 10x the configured initial cash, restarting OMS should not
    rebroadcast that amount as the authoritative Redis ledger.
    """
    try:
        cash = float(restored_cash)
        initial = float(initial_cash)
    except Exception:
        return float(initial_cash), "non_numeric"

    if not is_realtime_dry:
        return cash, None

    if max_multiplier is None:
        try:
            max_multiplier = float(os.environ.get("OMS_DRY_CASH_RESTORE_MAX_MULTIPLIER", "3.0"))
        except Exception:
            max_multiplier = 3.0
    max_cash = max(initial, 1.0) * max(float(max_multiplier), 1.0)
    if cash <= 0:
        return initial, "non_positive"
    if cash > max_cash:
        return initial, f"above_dry_restore_cap:{cash:.2f}>{max_cash:.2f}"
    return cash, None

class OrchestratorStateManager:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.state_namespace = OMS_STATE_NAMESPACE

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DB_URL)

    def _init_state_db(self):
        """[修改] 在统一 PG 中创建 namespace 分区状态表"""
        if self.orch.mode == 'backtest': return
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute("""
                SELECT c.relkind
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = 'symbol_state'
            """)
            row = c.fetchone()
            if row and row[0] != 'p':
                backup_name = f"symbol_state_legacy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                c.execute(
                    sql.SQL("ALTER TABLE public.symbol_state RENAME TO {}")
                    .format(sql.Identifier(backup_name))
                )
                logger.warning(
                    f"🧱 symbol_state was not partitioned; renamed to {backup_name} "
                    f"and creating namespace-partitioned symbol_state."
                )

            c.execute("""
                CREATE TABLE IF NOT EXISTS symbol_state (
                    namespace TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    data TEXT,
                    updated_at DOUBLE PRECISION,
                    CONSTRAINT symbol_state_ns_pkey PRIMARY KEY (namespace, symbol)
                ) PARTITION BY LIST (namespace)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_state_ns_updated_at
                ON symbol_state (namespace, updated_at)
            """)
            part_name = safe_state_partition_name(self.state_namespace)
            c.execute(
                sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} PARTITION OF symbol_state
                    FOR VALUES IN (%s)
                """).format(sql.Identifier(part_name)),
                (self.state_namespace,),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ State DB Init Error: {e}")

    def _load_state(self):
        """[Modified] Load state ONLY from PostgreSQL (Intraday Enforcement)"""
        # 👇 [🔥 核心修复：回放模式坚决不读旧库，强制初始化]
        if IS_SIMULATED:
            logger.info("🔄 回测模式 (SIMULATED) 启动，清空历史状态，重置为初始资金 50,000...")
            self.orch.mock_cash = 50000.0  # 你的回放初始资金
            self.orch.locked_cash = 0.0
            self.orch.positions = {}
            self.orch.pending_orders = {}
            return
        # 👆 修复结束

        self._init_state_db() 
        
        # Only load from PostgreSQL
        state_data = self._load_state_from_db()

        if not state_data:
            logger.info("⚠️ No valid intraday state found in PostgreSQL. Starting Fresh.")
            return

        try:
            restored_count = 0
            for sym, data in state_data.items():
                if sym in self.orch.states:
                    self.orch.states[sym].from_dict(data)
                    restored_count += 1
            
            logger.info(f"♻️ Restored state for {restored_count} symbols from PostgreSQL")
            
        except Exception as e:
            logger.error(f"❌ Failed to restore state: {e}")

    def _load_state_from_db(self):
        """[修改] 从统一 PG 恢复状态"""
        if self.orch.mode == 'backtest': return {}
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            
            # 必须先检查表是否存在，防止冷启动时报错
            c.execute("SELECT to_regclass('public.symbol_state')")
            if c.fetchone()[0] is None:
                conn.close()
                return {}
                
            c.execute(
                "SELECT symbol, data, updated_at FROM symbol_state WHERE namespace = %s",
                (self.state_namespace,),
            )
            rows = c.fetchall()
            conn.close()
            
            restored = {}
            ny_tz = timezone('America/New_York')
            now_ny = datetime.now(ny_tz).date()
            
            # [🛡️ 跨天丢弃] 所有状态记录（包括 _GLOBAL_STATE_）只在同一交易日内沿用
            # 之前的 bug：_GLOBAL_STATE_ 不受跨天约束，导致 mock_cash 隔夜漂移
            # 现在统一按 updated_at 的 NY 日期判定
            dropped_count = 0
            dropped_global = False
            for sym, data_str, updated_at in rows:
                try:
                    ts_dt = datetime.fromtimestamp(updated_at, ny_tz).date()
                    if ts_dt != now_ny:
                        dropped_count += 1
                        if sym == '_GLOBAL_STATE_':
                            dropped_global = True
                        continue
                    restored[sym] = json.loads(data_str)
                except: pass

            ghost_rows = []
            mode_mismatch_rows = []
            for sym, data in list(restored.items()):
                if sym == '_GLOBAL_STATE_' or not isinstance(data, dict):
                    continue
                state_mode = str(data.get("mode") or "").upper()
                if state_mode and state_mode != RUN_MODE:
                    restored[sym] = zero_position_state_row(data)
                    mode_mismatch_rows.append((sym, state_mode))
                    continue
                try:
                    pos = int(data.get("position", 0) or 0)
                    qty = int(data.get("qty", 0) or 0)
                    entry_price = float(data.get("entry_price", 0.0) or 0.0)
                    entry_ts = float(data.get("entry_ts", 0.0) or 0.0)
                except Exception:
                    pos, qty, entry_price, entry_ts = 0, 0, 0.0, 0.0
                if pos != 0 and (
                    qty <= 0
                    or entry_price <= 0
                    or entry_ts <= 0
                    or not infer_open_fill_confirmed(data)
                ):
                    restored[sym] = zero_position_state_row(data)
                    ghost_rows.append((sym, pos, qty, entry_price, entry_ts))

            # [新增] 恢复全局资金状态（仅当天有效）
            if '_GLOBAL_STATE_' in restored:
                global_data = restored['_GLOBAL_STATE_']
                if 'mock_cash' in global_data:
                    state_mode = str(global_data.get('mode') or '').upper()
                    if state_mode and state_mode != RUN_MODE:
                        logger.warning(
                            f"🚫 Ignore _GLOBAL_STATE_ cash from mode={state_mode}; "
                            f"current RUN_MODE={RUN_MODE}, keep mock_cash=${self.orch.mock_cash:,.2f}"
                        )
                    else:
                        restored_cash, reject_reason = sanitize_restored_mock_cash(
                            global_data['mock_cash'],
                            initial_cash=getattr(self.orch.cfg, 'INITIAL_ACCOUNT', INITIAL_ACCOUNT),
                            is_realtime_dry=IS_REALTIME_DRY,
                        )
                        if reject_reason:
                            self.orch.mock_cash = restored_cash
                            logger.warning(
                                f"🚫 Reject restored _GLOBAL_STATE_ mock_cash={global_data['mock_cash']} "
                                f"reason={reject_reason}; reset to ${self.orch.mock_cash:,.2f}"
                            )
                        else:
                            self.orch.mock_cash = restored_cash
                            legacy_tag = " legacy-no-mode" if not state_mode else ""
                            logger.info(
                                f"💰 Restored mock_cash from DB (same-day{legacy_tag}): "
                                f"${self.orch.mock_cash:,.2f}"
                            )
            elif dropped_global:
                logger.info("🧹 _GLOBAL_STATE_ from previous day dropped; mock_cash keeps default (fresh start).")

            if dropped_count > 0:
                logger.info(f"🧹 Ignored {dropped_count} state records from previous days.")
            if mode_mismatch_rows:
                preview = ", ".join(f"{sym}({mode})" for sym, mode in mode_mismatch_rows[:8])
                logger.warning(
                    f"🧹 Zeroized {len(mode_mismatch_rows)} restored rows with mode mismatch: {preview}"
                )
            if ghost_rows:
                preview = ", ".join(
                    f"{sym}(pos={pos},qty={qty},ep={entry_price:.2f},ets={entry_ts:.0f})"
                    for sym, pos, qty, entry_price, entry_ts in ghost_rows[:8]
                )
                logger.warning(
                    f"🧹 Zeroized {len(ghost_rows)} ghost restored positions without confirmed OPEN fill: {preview}"
                )
            return restored
        except Exception as e:
            logger.error(f"❌ DB Load Error: {e}")
            return {}

    def _save_state_to_db(self, state_data, snapshot_ts=None):
        """[修改] 异步备份状态到 PG，彻底剥离 I/O 阻塞"""
        def _write_task(data):
            try:
                self._init_state_db() # 确保表存在
                conn = self._get_pg_conn()
                c = conn.cursor()
                ts = float(snapshot_ts if snapshot_ts is not None else time.time())
                namespace = self.state_namespace
                data_list = [(namespace, sym, json.dumps(d), ts) for sym, d in data.items()]
                import psycopg2.extras
                psycopg2.extras.execute_batch(c, """
                    INSERT INTO symbol_state (namespace, symbol, data, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (namespace, symbol) DO UPDATE
                    SET data=EXCLUDED.data, updated_at=EXCLUDED.updated_at
                    WHERE symbol_state.updated_at IS NULL OR symbol_state.updated_at <= EXCLUDED.updated_at
                """, data_list)
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"❌ DB Save Error: {e}")
                
        # 放进后台静默线程，绝不卡主循环一点点时间
        threading.Thread(target=_write_task, args=(state_data,), daemon=True).start()

    def save_state(self):
        """[Atomic Save] 原子写入状态 (PostgreSQL ONLY)"""
        if getattr(self.orch, 'disable_db_save', False): return
        
        if self.orch.mode == 'backtest': 
            return

        try:
            # Gather state for saving
            state_data = {}
            for sym, st in self.orch.states.items():
                if not (st.position != 0 or st.warmup_complete):
                    continue
                row = st.to_dict()
                if int(row.get("position", 0) or 0) != 0 and not infer_open_fill_confirmed(row):
                    logger.warning(
                        f"🧹 Skip persisting unconfirmed OPEN state for {sym}: "
                        f"qty={row.get('qty')} entry_price={row.get('entry_price')} entry_ts={row.get('entry_ts')}"
                    )
                    row = zero_position_state_row(row)
                if int(row.get("position", 0) or 0) != 0 or row.get("warmup_complete"):
                    state_data[sym] = row
            
            # [新增] 注入全局资金状态
            state_data['_GLOBAL_STATE_'] = {
                'mock_cash': self.orch.mock_cash,
                'updated_at': time.time(),
                'mode': RUN_MODE,
                'engine_mode': str(getattr(self.orch, 'mode', '') or '').upper(),
            }
            state_data = tag_state_snapshot_rows(
                state_data,
                namespace=self.state_namespace,
                run_mode=RUN_MODE,
            )
            if get_domain_shadow_router is not None:
                try:
                    get_domain_shadow_router().on_state_snapshot(
                        state_data,
                        namespace=self.state_namespace,
                        run_mode=RUN_MODE,
                    )
                except Exception as e:
                    logger.warning(f"[DomainShadow] state_snapshot hook failed: {e}")

            # Write to PostgreSQL
            self._save_state_to_db(state_data, snapshot_ts=time.time())
            
            # Remove JSON file if exists to avoid confusion
            json_path = Path("positions_ignore_v8.json")
            if json_path.exists():
                try: os.remove(json_path)
                except: pass
            
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def _publish_warmup_status(self):
        """[Monitor] 将当前 Alpha History 长度发布到 Redis"""
        try:
            status = {}
            for sym, st in self.orch.symbol_states.items():
                status[sym] = len(st.alpha_history)
            
            key = f"monitor:warmup:orch"
            if status:
                status['INDEX_TREND'] = self.orch.last_index_trend # [NEW] 为 Dashboard 提供大盘趋势显示
                self.orch.r.hset(key, mapping=status)
                self.orch.r.expire(key, 3600)
        except: pass

    def _recover_warmup_from_pg(self):
        """[新增] 从 PostgreSQL 恢复 Alpha 历史，支持多日跨度 Warmup"""
        try:
            from config import NY_TZ
            today_str = datetime.now(NY_TZ).strftime('%Y%m%d')
            
            # Start of today NY Time, convert to ts
            now_ny = datetime.now(NY_TZ)
            start_dt = NY_TZ.localize(datetime.combine(now_ny.date(), datetime.min.time()))
            start_ts = start_dt.timestamp()

            logger.info(f"🔄 Recovering Warmup Data from PostgreSQL...")
            
            try:
                conn = self._get_pg_conn()
                cursor = conn.cursor()
            except Exception as e:
                logger.error(f"⚠️ Failed to connect to PG for warmup: {e}")
                return

            # 2. 对每个 Symbol 恢复历史
            for sym, st in self.orch.states.items():
                if st.warmup_complete: continue # 已有状态则跳过

                recovered_data = [] # List of (ts, price, alpha)
                
                try:
                    # 获取该股票最多 N 条过去的历史数据 (过滤掉当前时间之后)
                    cursor.execute(
                        "SELECT ts, price, alpha FROM alpha_logs WHERE symbol = %s AND ts < %s ORDER BY ts DESC LIMIT %s", 
                        (sym, start_ts, ROLLING_WINDOW * 2) 
                    )
                    rows = cursor.fetchall()
                    
                    if rows:
                        # rows are DESC (latest first), reverse them to get chronological chunk
                        chunk = rows[::-1] 
                        recovered_data = chunk
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ Error reading PG for {sym}: {e}")
                    conn.rollback()

                # 3. 重建 State
                if recovered_data:
                    # 截取最近的 N 个
                    valid_data = recovered_data[-(ROLLING_WINDOW + 10):]
                    
                    # 必须按时间顺序重放
                    for ts, price, alpha in valid_data:
                        try:
                            st.update_indicators(price, alpha)
                        except Exception as e:
                            logger.error(f"Warmup recovery calculation error for {sym}: {e}")
                    
                    if st.warmup_complete:
                        logger.info(f"  ✅ {sym} Warmup Recovery Done. Buffer size: {len(st.alpha_history)}. Mode: {st.correction_mode}")
                    else:
                        logger.info(f"  🟠 {sym} Warmup Recovery Partial. Buffer size: {len(st.alpha_history)} / {ROLLING_WINDOW}")
            
            # 统一关闭连接
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Failed to full warmup from PG: {e}")

    def _recover_warmup_from_sqlite(self):
        """[NEW] 从本地 SQLite 数据库恢复 Alpha 历史 (专门针对 s2_run_realtime_replay_sqlite)"""
        try:
            from config import DB_DIR, NY_TZ, TARGET_SYMBOLS
            import sqlite3
            
            # 1. 获取当前回放时间
            ts_str = self.orch.r.get("replay:current_ts")
            if not ts_str: return
            curr_ts = float(ts_str)
            dt_ny = datetime.fromtimestamp(curr_ts, tz=NY_TZ)
            
            # 2. 找到最近的几个数据库文件 (按日期倒序)
            all_dbs = sorted([f for f in DB_DIR.glob("market_*.db") if len(f.stem) == 15], reverse=True)
            target_date_str = dt_ny.strftime('%Y%m%d')
            
            # 过滤出当前日期及之前的数据库
            relevant_dbs = [f for f in all_dbs if f.stem.split('_')[1] <= target_date_str][:3]
            if not relevant_dbs:
                logger.info("🧊 No SQLite databases found for warmup.")
                return

            logger.info(f"🔄 Recovering Warmup Data from SQLite ({[f.name for f in relevant_dbs]})...")
            
            for sym in TARGET_SYMBOLS:
                if sym not in self.orch.states: continue
                st = self.orch.states[sym]
                if st.warmup_complete: continue
                
                recovered_count = 0
                for db_path in relevant_dbs:
                    if st.warmup_complete: break
                    
                    try:
                        # 使用只读模式打开，防止锁冲突
                        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
                        cursor = conn.cursor()
                        
                        # 检查是否有 alpha_logs 表
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alpha_logs'")
                        if not cursor.fetchone():
                            conn.close()
                            continue
                            
                        # 获取历史数据
                        cursor.execute(
                            "SELECT ts, price, alpha FROM alpha_logs WHERE symbol = ? AND ts < ? ORDER BY ts DESC LIMIT ?",
                            (sym, curr_ts, ROLLING_WINDOW * 2)
                        )
                        rows = cursor.fetchall()
                        conn.close()
                        
                        if rows:
                            # 逆序变正序
                            chunk = rows[::-1]
                            for ts_val, price, alpha in chunk:
                                st.update_indicators(price, alpha)
                            recovered_count += len(rows)
                            
                    except Exception as e:
                        logger.debug(f"  ⚠️ Error reading SQLite {db_path.name} for {sym}: {e}")
                        continue
                
                if recovered_count > 0:
                    logger.debug(f"  ✅ {sym}: Recovered {recovered_count} points from SQLite.")

        except Exception as e:
            logger.error(f"❌ Failed to recover warmup from SQLite: {e}")
