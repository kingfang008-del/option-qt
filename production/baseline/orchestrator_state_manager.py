import json
import time
import logging
import os
import threading
import pickle
from pathlib import Path
from datetime import datetime
from pytz import timezone
import psycopg2
import sqlite3
import numpy as np
from config import PG_DB_URL, IS_SIMULATED, IS_BACKTEST, ROLLING_WINDOW, TARGET_SYMBOLS, DB_DIR, REDIS_CFG, NY_TZ

logger = logging.getLogger("V8_Orchestrator.StateManager")

class OrchestratorStateManager:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DB_URL)

    def _init_state_db(self):
        """[修改] 在统一 PG 中创建状态表"""
        if self.orch.mode == 'backtest': return
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS symbol_state (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at REAL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ State DB Init Error: {e}")

    def _load_state(self):
        """[Modified] Load state ONLY from PostgreSQL (Intraday Enforcement)"""
        # 👇 [🔥 核心修复：回放模式坚决不读旧库，强制初始化]
        if IS_SIMULATED:
            logger.info("🔄 LIVEREPLAY 或回测模式 (SIMULATED) 启动，清空历史状态，重置为初始资金 50,000...")
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
                
            c.execute("SELECT symbol, data, updated_at FROM symbol_state")
            rows = c.fetchall()
            conn.close()
            
            restored = {}
            ny_tz = timezone('America/New_York')
            now_ny = datetime.now(ny_tz).date()
            
            dropped_count = 0
            for sym, data_str, updated_at in rows:
                try:
                    ts_dt = datetime.fromtimestamp(updated_at, ny_tz).date()
                    if ts_dt != now_ny and sym != '_GLOBAL_STATE_':
                        dropped_count += 1
                        continue
                    restored[sym] = json.loads(data_str)
                except: pass
                
            # [新增] 恢复全局资金状态
            if '_GLOBAL_STATE_' in restored:
                global_data = restored['_GLOBAL_STATE_']
                if 'mock_cash' in global_data:
                    self.orch.mock_cash = float(global_data['mock_cash'])
                    logger.info(f"💰 Restored mock_cash from DB: ${self.orch.mock_cash:,.2f}")

            if dropped_count > 0:
                logger.info(f"🧹 Ignored {dropped_count} state records from previous days.")
            return restored
        except Exception as e:
            logger.error(f"❌ DB Load Error: {e}")
            return {}

    def _save_state_to_db(self, state_data):
        """[修改] 异步备份状态到 PG，彻底剥离 I/O 阻塞"""
        def _write_task(data):
            try:
                self._init_state_db() # 确保表存在
                conn = self._get_pg_conn()
                c = conn.cursor()
                ts = time.time()
                data_list = [(sym, json.dumps(d), ts) for sym, d in data.items()]
                import psycopg2.extras
                psycopg2.extras.execute_batch(c, """
                    INSERT INTO symbol_state (symbol, data, updated_at) VALUES (%s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE 
                    SET data=EXCLUDED.data, updated_at=EXCLUDED.updated_at
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
            state_data = {sym: st.to_dict() for sym, st in self.orch.states.items() if st.position != 0 or st.warmup_complete}
            
            # [新增] 注入全局资金状态
            state_data['_GLOBAL_STATE_'] = {
                'mock_cash': self.orch.mock_cash,
                'updated_at': time.time()
            }

            # Write to PostgreSQL
            self._save_state_to_db(state_data)
            
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
