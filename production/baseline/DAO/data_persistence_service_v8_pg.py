import logging
import psycopg2
import psycopg2.extras
import redis
import json
import time
import pickle
import sys
import os

# [NEW] Add project root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import serialization_utils as ser
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
from config import REDIS_CFG, STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE, DATA_DIR, PG_DB_URL

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPersistenceServicePG:
    def __init__(self, start_date=None):
        self.r = redis.Redis(
            host=REDIS_CFG['host'],
            port=REDIS_CFG['port'],
            db=REDIS_CFG['db'],
            decode_responses=False
        )
        # Create or Reset Consumer Group
        for stream in [STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE]:
            try:
                self.r.xgroup_create(stream, REDIS_CFG['pg_group'], mkstream=True, id='0')
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # 🚀 [CRITICAL] Force reset to '0' to catch up missed history
                    self.r.xgroup_setid(stream, REDIS_CFG['pg_group'], id='0')
                else: 
                    logger.error(f"Group Error ({stream}): {e}")

        self.conn = None
        self.current_date = None # [Fix] Set to None initially to force _check_date_rotation to run on first call
        self.target_start_date = str(start_date) if start_date else None
        
        # Buffer
        self.bar_buffer_1s = {}
        self.bar_buffer = {}
        self.option_buffer = {}
        self.option_buffer_1s = {}
        self.bar_buffer_5m = {}
        self.option_buffer_5m = {}
        
        # [NEW] 5min Accumulator to ensure market_bars_5m is synthesized correctly
        self.acc_5m = {} # {(symbol, 5m_ts): [bars]}
        self.last_synthesized = {} # {symbol: last_5m_ts}
        
        self.trade_buffer = []
        self.alpha_buffer = []
        self.last_flush = time.time()
        # 默认分钟表质量优先：只落 Greeks 已就绪的 option snapshot
        flag_env = os.environ.get("OPTION_1M_WAIT_GREEKS_READY", "1").strip().lower()
        self.option_1m_wait_greeks_ready = flag_env not in {"0", "false", "no", "off"}
        logger.info(f"🧭 Option 1m write gate: wait_greeks_ready={self.option_1m_wait_greeks_ready}")
        
        # [Fix] Enforce Master Tables exist on startup
        try:
            self._init_master_tables()
        except Exception as e:
            logger.error(f"Failed to auto-init Master Tables: {e}")
            
        self._check_date_rotation() # This will create the initial partition if needed

    def _get_pg_conn(self):
        if not self.conn or self.conn.closed != 0:
            self.conn = psycopg2.connect(PG_DB_URL)
        return self.conn

    def _init_master_tables(self):
        """Initialize PostgreSQL Master Partitioned Tables"""
        conn = self._get_pg_conn()
        c = conn.cursor()
        
        # 1. market_bars_1m & 5m
        for tbl in ['market_bars_1s', 'market_bars_1m', 'market_bars_5m']:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    symbol TEXT,
                    ts DOUBLE PRECISION,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    vwap DOUBLE PRECISION,
                    PRIMARY KEY (symbol, ts)
                ) PARTITION BY RANGE (ts);
            """)
        
        # 2. trade_logs
        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                ts DOUBLE PRECISION,
                datetime_ny TEXT,
                symbol TEXT,
                action TEXT,
                qty DOUBLE PRECISION,
                price DOUBLE PRECISION,
                details_json TEXT
            ) PARTITION BY RANGE (ts);
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trd_sym_ts ON trade_logs (symbol, ts)")

        # 3. trade_logs_backtest
        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs_backtest (
                ts DOUBLE PRECISION,
                datetime_ny TEXT,
                symbol TEXT,
                action TEXT,
                qty DOUBLE PRECISION,
                price DOUBLE PRECISION,
                details_json TEXT
            ) PARTITION BY RANGE (ts);
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_trd_bk_sym_ts ON trade_logs_backtest (symbol, ts)")

        # 4. alpha_logs
        c.execute("""
            CREATE TABLE IF NOT EXISTS alpha_logs (
                ts DOUBLE PRECISION,
                datetime_ny TEXT,
                symbol TEXT,
                alpha DOUBLE PRECISION,
                iv DOUBLE PRECISION,
                price DOUBLE PRECISION,
                vol_z DOUBLE PRECISION,
                PRIMARY KEY (symbol, ts)
            ) PARTITION BY RANGE (ts);
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_alpha_ts ON alpha_logs (ts)")

        # 5. option_snapshots_1m & 5m
        for tbl in ['option_snapshots_1s', 'option_snapshots_1m', 'option_snapshots_5m']:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    symbol TEXT,
                    ts DOUBLE PRECISION,
                    buckets_json JSONB,
                    PRIMARY KEY (symbol, ts)
                ) PARTITION BY RANGE (ts);
            """)
        
        # 6. feature_logs
        c.execute("""
            CREATE TABLE IF NOT EXISTS feature_logs (
                symbol TEXT,
                ts DOUBLE PRECISION,
                fast_norm_blob BYTEA,
                slow_norm_blob BYTEA,
                PRIMARY KEY (symbol, ts)
            ) PARTITION BY RANGE (ts);
        """)
        conn.commit()
        c.close()

    def _create_daily_partition(self, ny_date: datetime):
        """动态创建今日的物理分区 (使用 autocommit 避免 DDL 污染主事务)"""
        from config import NY_TZ
        
        start_dt = NY_TZ.localize(datetime.combine(ny_date.date(), datetime.min.time()))
        end_dt = start_dt + timedelta(days=1)
        
        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()
        
        date_str = ny_date.strftime('%Y%m%d')
        
        tables = [
            'market_bars_1s', 'market_bars_1m', 'market_bars_5m',
            'trade_logs', 'trade_logs_backtest',
            'alpha_logs',
            'option_snapshots_1s', 'option_snapshots_1m', 'option_snapshots_5m',
            'feature_logs'
        ]
        
        # DDL 用独立的 autocommit 连接，防止异常污染主连接事务
        ddl_conn = psycopg2.connect(PG_DB_URL)
        ddl_conn.autocommit = True
        c = ddl_conn.cursor()
        
        for table in tables:
            part_name = f"{table}_{date_str}"
            try:
                c.execute(f"""
                    CREATE TABLE IF NOT EXISTS {part_name} PARTITION OF {table} 
                    FOR VALUES FROM ({start_ts}) TO ({end_ts});
                """)
                logger.info(f"📂 Created/verified Postgres partition: {part_name}")
            except Exception as e:
                logger.warning(f"Partition {part_name} creation warning (likely exists): {e}")
                
        c.close()
        ddl_conn.close()

    def _check_date_rotation(self, ts=None):
        from config import NY_TZ
        now_dt = datetime.fromtimestamp(float(ts), NY_TZ) if ts is not None else datetime.now(NY_TZ)
        now_date_str = now_dt.strftime('%Y%m%d')
        
        if now_date_str != self.current_date:
            logger.info(f"📅 Date rotation: {self.current_date} -> {now_date_str}")
            self.current_date = now_date_str
            self.bar_buffer_1s.clear()
            self.bar_buffer.clear()
            self.option_buffer.clear()
            self.option_buffer_1s.clear()
            self.bar_buffer_5m.clear()
            self.option_buffer_5m.clear()
            self.acc_5m.clear()
            self.last_synthesized.clear()
            self.trade_buffer.clear()
            try:
                self._init_master_tables()
                self._create_daily_partition(now_dt)
            except Exception as e:
                logger.error(f"Date rotation init failed: {e}")
                # rollback 防止主连接进入 aborted 状态
                try:
                    if self.conn and self.conn.closed == 0:
                        self.conn.rollback()
                except Exception:
                    self.conn = None

    def process_feature_data(self, payload, write_feature_logs=True, write_option_snapshots=True):
        try:
            ts = int(payload.get('ts', time.time()))
            
            if 'symbols' in payload and isinstance(payload['symbols'], list):
                symbols = payload['symbols']
                fast_batch = payload.get('fast_1m')
                slow_batch = payload.get('slow_1m')
                
                rows_to_insert = []
                for i, sym in enumerate(symbols):
                    fast_blob = None
                    if fast_batch is not None and len(fast_batch) > i:
                        arr = fast_batch[i]
                        if len(arr.shape) > 1: arr = arr[-1]
                        fast_blob = psycopg2.Binary(arr.astype(np.float32).tobytes())
                    
                    slow_blob = None
                    if slow_batch is not None and len(slow_batch) > i:
                        arr = slow_batch[i]
                        if len(arr.shape) > 1: arr = arr[-1]
                        slow_blob = psycopg2.Binary(arr.astype(np.float32).tobytes())
                    
                    if fast_blob or slow_blob:
                        rows_to_insert.append((sym, ts, fast_blob, slow_blob))
                
                if write_feature_logs and rows_to_insert:
                    conn = self._get_pg_conn()
                    c = conn.cursor()
                    psycopg2.extras.execute_batch(c, """
                        INSERT INTO feature_logs (symbol, ts, fast_norm_blob, slow_norm_blob) 
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (symbol, ts) DO UPDATE 
                        SET fast_norm_blob = EXCLUDED.fast_norm_blob, slow_norm_blob = EXCLUDED.slow_norm_blob
                    """, rows_to_insert)
                    conn.commit()
                    c.close()

                # 🚀 [SDS 2.0 Greeks 强制绑定逻辑]
                # 检查是否存在期权快照计算结果，如果存在，则将其同步写入 option_snapshots_1m 表。
                # ！！注意：FCS 下发的键名是 'live_options'，必须保持一致！！
                option_snapshots = payload.get('live_options') or payload.get('option_snapshots')
                if write_option_snapshots and option_snapshots and isinstance(option_snapshots, dict):
                    opt_rows_all = []
                    opt_rows_1m = []
                    sample_sym = None
                    sample_g_sum = 0
                    skipped_not_ready = 0
                    is_new_minute = bool(payload.get('is_new_minute'))
                    nvda_diag = None
                    
                    for opt_sym, opt_data in option_snapshots.items():
                        if not isinstance(opt_data, dict):
                            continue
                        # 只有在 buckets 字段不为空时才处理
                        buckets = opt_data.get('buckets')
                        if buckets is None:
                            continue
                        # 兼容 list / np.ndarray；避免 ndarray 在 truth-value 判定时抛错
                        if isinstance(buckets, np.ndarray):
                            if buckets.size == 0:
                                continue
                            buckets_for_json = buckets.tolist()
                        elif isinstance(buckets, list):
                            if len(buckets) == 0:
                                continue
                            buckets_for_json = buckets
                        else:
                            continue

                        payload_opt = dict(opt_data)
                        payload_opt['buckets'] = buckets_for_json
                        row = (opt_sym, ts, json.dumps(payload_opt))
                        opt_rows_all.append(row)

                        greeks_ready = bool(opt_data.get('greeks_ready', True))
                        if (not self.option_1m_wait_greeks_ready) or greeks_ready:
                            opt_rows_1m.append(row)
                        elif is_new_minute:
                            skipped_not_ready += 1
                            
                        # [诊断日志抽样] 检查希腊值是否真的有效（非 0）
                        if sample_sym is None and len(buckets_for_json) > 0:
                            # 抽样检查第一行的 Greeks 列 (1,2,3,4)
                            first_row = buckets_for_json[0]
                            if isinstance(first_row, (list, tuple)) and len(first_row) > 5:
                                g_sum = sum(abs(float(x)) for x in first_row[1:5])
                                if g_sum > 1e-6:
                                    sample_sym = opt_sym
                                    sample_g_sum = g_sum

                        # NVDA 专项写前校验：确认 Greeks 与 IV 已正确打包
                        if str(opt_sym).upper() == 'NVDA':
                            try:
                                arr = np.asarray(buckets_for_json, dtype=np.float64)
                                if arr.ndim == 2 and arr.shape[1] >= 8:
                                    greeks_sum = float(np.sum(np.abs(arr[:, 1:5])))
                                    call_iv = float(arr[2, 7]) if arr.shape[0] > 2 else 0.0
                                    put_iv = float(arr[0, 7]) if arr.shape[0] > 0 else 0.0
                                    nvda_diag = {
                                        'greeks_sum': greeks_sum,
                                        'call_iv': call_iv,
                                        'put_iv': put_iv,
                                        'greeks_ready': bool(opt_data.get('greeks_ready', True)),
                                    }
                            except Exception:
                                nvda_diag = {'parse_error': True}

                    if opt_rows_all:
                        if nvda_diag is not None:
                            if nvda_diag.get('parse_error'):
                                logger.warning(f"⚠️ [PG-NVDA-Precheck] ts={ts} parse buckets failed before write.")
                            else:
                                if nvda_diag['greeks_sum'] > 1e-6:
                                    logger.info(
                                        f"🧪 [PG-NVDA-Precheck] ts={ts} greeks_sum={nvda_diag['greeks_sum']:.6f} "
                                        f"call_iv={nvda_diag['call_iv']:.4f} put_iv={nvda_diag['put_iv']:.4f} "
                                        f"ready={nvda_diag['greeks_ready']}"
                                    )
                                else:
                                    logger.warning(
                                        f"⚠️ [PG-NVDA-Precheck] ts={ts} ZERO greeks before write "
                                        f"(call_iv={nvda_diag['call_iv']:.4f}, put_iv={nvda_diag['put_iv']:.4f}, "
                                        f"ready={nvda_diag['greeks_ready']})"
                                    )

                        if sample_sym:
                            logger.info(f"📊 [PG-Greeks-Bind] Detected enriched Greeks for {sample_sym} (sum={sample_g_sum:.4f}). Upserting {len(opt_rows_all)} symbols.")
                        elif is_new_minute:
                            logger.warning(f"⚠️ [PG-Greeks-Bind] Upserting {len(opt_rows_all)} symbols but sampled Greeks sum is zero.")
                        if is_new_minute and skipped_not_ready > 0:
                            logger.warning(f"⏳ [PG-Option1M-Gate] Skip {skipped_not_ready} symbols (greeks not ready) at ts={ts}.")
                        
                        conn = self._get_pg_conn()
                        c = conn.cursor()
                        
                        # 1. 写入分钟表 (仅在分钟结算点触发)
                        if is_new_minute and opt_rows_1m:
                            psycopg2.extras.execute_batch(c, """
                                INSERT INTO option_snapshots_1m (symbol, ts, buckets_json) 
                                VALUES (%s, %s, %s)
                                ON CONFLICT (symbol, ts) DO UPDATE 
                                SET buckets_json = EXCLUDED.buckets_json
                            """, opt_rows_1m)
                        
                        # 2. 写入秒级表 (每秒实时更新，包含最新求解结果)
                        psycopg2.extras.execute_batch(c, """
                            INSERT INTO option_snapshots_1s (symbol, ts, buckets_json) 
                            VALUES (%s, %s, %s)
                            ON CONFLICT (symbol, ts) DO UPDATE 
                            SET buckets_json = EXCLUDED.buckets_json
                        """, opt_rows_all)
                        
                        conn.commit()
                        c.close()
                
        except Exception as e:
            logger.error(f"Feature Write Error: {e}")

    def process_market_data(self, payload):
        try:
            ts = float(payload['ts'])
            symbol = payload['symbol']
            self._check_date_rotation(ts)
            second_ts = int(ts)
            
            minute_ts = int(ts) // 60 * 60
            key = (symbol, minute_ts)
            key_1s = (symbol, second_ts)

            # 1min Bar
            stock = payload.get('stock', {})
            price = float(stock.get('close', 0.0))
            vol = float(stock.get('volume', 0.0))
            vwap_val = float(stock.get('vwap', price))

            if price > 0:
                self.bar_buffer_1s[key_1s] = {
                    'symbol': symbol,
                    'ts': second_ts,
                    'open': float(stock.get('open', price)),
                    'high': float(stock.get('high', price)),
                    'low': float(stock.get('low', price)),
                    'close': price,
                    'volume': vol,
                    'vwap': vwap_val,
                }
            
            if price > 0:
                if key not in self.bar_buffer:
                    self.bar_buffer[key] = {
                        'symbol': symbol, 
                        'ts': minute_ts,
                        'open': price, 'high': price, 'low': price, 'close': price,
                        'volume': vol,
                        'pv_sum': vwap_val * vol
                    }
                else:
                    b = self.bar_buffer[key]
                    b['high'] = max(b['high'], price)
                    b['low'] = min(b['low'], price)
                    b['close'] = price
                    b['volume'] += vol
                    b['pv_sum'] += vwap_val * vol
            
            # 5min Bar (🚀 自动合成逻辑：基于 acc_5m 累加器实现稳健合成)
            # 🚀 [强制收敛] 5min Bar: 彻底废弃上游推送，100% 依赖 1m 线底层滚动聚合
            five_min_ts = (minute_ts // 300) * 300 
            acc_key = (symbol, five_min_ts)
            
            if acc_key not in self.acc_5m: 
                self.acc_5m[acc_key] = []
                
            b1m = self.bar_buffer.get(key)
            if b1m:
                # 将当前真实的 1min 线存入累加器（自动去重）
                if not self.acc_5m[acc_key] or self.acc_5m[acc_key][-1]['ts'] < b1m['ts']:
                    self.acc_5m[acc_key].append(b1m.copy())
                else:
                    # 🚀 [关键修复] 如果是同一分钟，则更新最后一条记录，确保成交量累加被反映到 5m 聚合中
                    self.acc_5m[acc_key][-1] = b1m.copy()
                    
                # 仅保留最近 5 分钟的数据防止内存无限增长
                if len(self.acc_5m[acc_key]) > 10: 
                    self.acc_5m[acc_key] = self.acc_5m[acc_key][-10:]
            
            # 实时计算当前 5m 桶的最新状态（利用真实的 O H L C V 聚合）
            bars = self.acc_5m[acc_key]
            if bars:
                total_vol = sum(b['volume'] for b in bars)
                total_pv = sum(b['pv_sum'] for b in bars)
                
                self.bar_buffer_5m[acc_key] = {
                    'symbol': symbol, 
                    'ts': five_min_ts,
                    'open': bars[0]['open'],
                    'high': max(b['high'] for b in bars),
                    'low': min(b['low'] for b in bars),
                    'close': bars[-1]['close'],
                    'volume': total_vol,
                    # 真实滚动 VWAP 保护：如果总交易量为 0，则 fallback 到最新收盘价
                    'vwap': total_pv / total_vol if total_vol > 0 else bars[-1]['close']
                }
                
                # if time.time() % 60 < 2: # 降低日志频率，每分钟只打一次心跳
                #     logger.info(f"🔄 [Rolling-5m] 强制本地聚合 {symbol} @ {five_min_ts} (Accumulated {len(bars)}/5 bars)")
            # 1min Options
            if 'option_buckets' in payload:
                snap_obj = payload['option_buckets']
                contracts = payload.get('option_contracts', [])
                if contracts:
                    snap_obj = {'buckets': payload['option_buckets'], 'contracts': contracts}
                self.option_buffer_1s[key_1s] = json.dumps(snap_obj)
                # 1m 快照改由 process_feature_data 的 live_options 统一落库，
                # 避免 raw 行情快照反复覆盖分钟级 Greeks 结果。

            # 5min Options Fallback (🚀 复用 1min 桶填补 5min 缺口)
            opt_buckets_5m = payload.get('option_buckets_5m')
            option_5m_ts_raw = payload.get('option_5m_ts')
            if opt_buckets_5m and option_5m_ts_raw is None:
                has_current_option_5m = minute_ts % 300 == 0
                option_5m_ts = minute_ts
            elif opt_buckets_5m:
                option_5m_ts = int(float(option_5m_ts_raw))
                has_current_option_5m = option_5m_ts == minute_ts and minute_ts % 300 == 0
            else:
                option_5m_ts = minute_ts
                has_current_option_5m = False

            if has_current_option_5m:
                contracts = payload.get('option_contracts_5m') or payload.get('option_contracts', [])
                snap_obj_5m = {'buckets': opt_buckets_5m, 'contracts': contracts} if contracts else opt_buckets_5m
                self.option_buffer_5m[(symbol, option_5m_ts)] = json.dumps(snap_obj_5m)
            elif minute_ts % 300 == 0 and payload.get('option_buckets'):
                contracts = payload.get('option_contracts', [])
                fallback_5m = payload.get('option_buckets')
                snap_obj_5m = {'buckets': fallback_5m, 'contracts': contracts} if contracts else fallback_5m
                self.option_buffer_5m[key] = json.dumps(snap_obj_5m)
                
        except Exception as e:
            logger.error(f"❌ [Market Process Error] Symbol: {symbol}, TS: {ts}, Error: {e}", exc_info=True)

    def flush(self):
        if not (
            self.bar_buffer_1s or self.bar_buffer or self.bar_buffer_5m
            or self.option_buffer_1s or self.option_buffer or self.option_buffer_5m
            or self.trade_buffer or self.alpha_buffer
        ):
            return
        
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            current_cutoff = time.time() - 300 
            
            # 1. Bars (1m & 5m)
            for buf, tbl in [
                (self.bar_buffer_1s, 'market_bars_1s'),
                (self.bar_buffer, 'market_bars_1m'),
                (self.bar_buffer_5m, 'market_bars_5m'), # 🚀 [核心修复] 补上 5 分钟 K 线的落盘队列
            ]:
                if buf:
                    rows = []
                    keys_to_del = []
                    for key, b in buf.items():
                        v_val = b.get('volume', 0.0)
                        vwap_val = b.get('vwap')
                        if vwap_val is None:
                            pv = b.get('pv_sum', 0.0)
                            vwap_val = pv / v_val if v_val > 0 else b['close']
                        
                        rows.append((b['symbol'], b['ts'], b['open'], b['high'], b['low'], b['close'], v_val, vwap_val))
                        if key[1] < current_cutoff: keys_to_del.append(key)
                    
                    if rows:
                        #logger.info(f"📊 [Flush-Check] Inserting {len(rows)} rows into {tbl} (Sample TS: {rows[0][1]})")
                        psycopg2.extras.execute_batch(c, f"""
                            INSERT INTO {tbl} (symbol, ts, open, high, low, close, volume, vwap) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, ts) DO UPDATE 
                            SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume, vwap=EXCLUDED.vwap
                        """, rows)
                    for k in keys_to_del: del buf[k]

            # 2. Options (1m & 5m)
            for buf, tbl in [
                (self.option_buffer_1s, 'option_snapshots_1s'),
                (self.option_buffer_5m, 'option_snapshots_5m'),
            ]:
                if buf:
                    rows = []
                    keys_to_del = []
                    for key, val in buf.items():
                        rows.append((key[0], key[1], val))
                        if key[1] < current_cutoff: keys_to_del.append(key)
                    
                    psycopg2.extras.execute_batch(c, f"""
                        INSERT INTO {tbl} (symbol, ts, buckets_json) 
                        VALUES (%s, %s, %s)
                        ON CONFLICT (symbol, ts) DO UPDATE 
                        SET buckets_json = EXCLUDED.buckets_json
                    """, rows)
                    for k in keys_to_del: del buf[k]

            # 3. Trades
            if self.trade_buffer:
                realtime_logs = []
                backtest_logs = []
                
                for item in self.trade_buffer:
                    ts = item[0]
                    from config import NY_TZ
                    dt_ny = datetime.fromtimestamp(ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                    row = (ts, dt_ny, item[1], item[2], item[3], item[4], item[5]) 
                    mode = item[6]
                    if mode in ['BACKTEST', 'LIVEREPLAY', 'SHADOW']:
                        backtest_logs.append(row)
                    else:
                        realtime_logs.append(row)
                
                if realtime_logs:
                    psycopg2.extras.execute_batch(c, "INSERT INTO trade_logs VALUES (%s,%s,%s,%s,%s,%s,%s)", realtime_logs)
                
                if backtest_logs:
                    psycopg2.extras.execute_batch(c, "INSERT INTO trade_logs_backtest VALUES (%s,%s,%s,%s,%s,%s,%s)", backtest_logs)
                    
                self.trade_buffer = []

            conn.commit()
            c.close()
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Flush Error (Market/Trade): {e}")
            # 必须 rollback 否则 PostgreSQL 连接会卡到 aborted 状态
            try:
                if self.conn and self.conn.closed == 0:
                    self.conn.rollback()
            except Exception:
                pass
            # 如果分区不存在导致写入失败，尝试创建当日分区再重试
            if 'no partition' in str(e).lower() or 'violates' in str(e).lower():
                try:
                    from config import NY_TZ
                    data_dt = None
                    if self.bar_buffer:
                        any_key = next(iter(self.bar_buffer.keys()))
                        data_dt = datetime.fromtimestamp(any_key[1], NY_TZ)
                    elif self.trade_buffer:
                        data_dt = datetime.fromtimestamp(self.trade_buffer[0][0], NY_TZ)
                    elif self.option_buffer:
                        any_key = next(iter(self.option_buffer.keys()))
                        data_dt = datetime.fromtimestamp(any_key[1], NY_TZ)
                    
                    if data_dt:
                        self._create_daily_partition(data_dt)
                    else:
                        self._create_daily_partition(datetime.now(NY_TZ))
                except Exception as pe:
                    logger.error(f"Partition creation failed: {pe}")

        # 4. Alphas — 独立事务，与 Bars/Trades 完全隔离，任何其他表的报错不影响 alpha 写入
        if self.alpha_buffer:
            # [DEBUG] Alpha flush logging
            #logger.info(f"💾 [Alpha Debug] Flushing {len(self.alpha_buffer)} alpha records to database...")
            alpha_snapshot = list(self.alpha_buffer)  # 先快照，避免写入期间被修改
            try:
                conn = self._get_pg_conn()
                c = conn.cursor()
                alpha_rows = []
                from config import NY_TZ
                for item in alpha_snapshot:
                    ts = item[0]
                    dt_ny = datetime.fromtimestamp(float(ts), NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                    alpha_rows.append((float(ts), dt_ny, item[1], item[2], item[3], item[4], item[5]))
                
                psycopg2.extras.execute_batch(c, """
                    INSERT INTO alpha_logs (ts, datetime_ny, symbol, alpha, iv, price, vol_z) 
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (symbol, ts) DO UPDATE 
                    SET alpha=EXCLUDED.alpha, iv=EXCLUDED.iv, price=EXCLUDED.price, vol_z=EXCLUDED.vol_z
                """, alpha_rows)
                conn.commit()          # ← 先 commit 成功
                c.close()
                self.alpha_buffer = []  # ← 再清空 buffer（绝不提前清！）
                logger.debug(f"✅ Alpha flush OK: {len(alpha_rows)} rows")
            except Exception as e:
                logger.error(f"Alpha Flush Error: {e}")
                try:
                    if self.conn and self.conn.closed == 0:
                        self.conn.rollback()
                except Exception:
                    pass
                if 'no partition' in str(e).lower() or 'violates' in str(e).lower():
                    try:
                        from config import NY_TZ
                        # [Fix] 根据数据本身的时间戳创建分区，而不是根据当前系统时间
                        if alpha_snapshot:
                            first_ts = float(alpha_snapshot[0][0])
                            data_dt = datetime.fromtimestamp(first_ts, NY_TZ)
                            self._create_daily_partition(data_dt)
                    except Exception as pe:
                        logger.error(f"Alpha partition creation failed: {pe}")
    
    def _init_consumer_groups(self):
        """统一创建/恢复消费者组"""
        from config import STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE
        for stream in [STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE]:
            try:
                self.r.xgroup_create(stream, REDIS_CFG['pg_group'], mkstream=True, id='0')
            except redis.exceptions.ResponseError as e:
                pass

    def run(self):
        # [🔥 动态 DB 切换] 启动时刷新 Redis 连接，应对回放模式
        from config import get_redis_db
        target_db = get_redis_db()
        if self.r.connection_pool.connection_kwargs.get('db') != target_db:
            logger.info(f"🔄 Re-connecting Redis to DB {target_db} (Dynamic Mode Detection)")
            self.r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=target_db)
            # [🔥 关键修复] 为新 DB 建立必要的消费者组
            for stream in [STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE]:
                try:
                    self.r.xgroup_create(stream, REDIS_CFG['pg_group'], mkstream=True, id='0')
                except Exception: pass
            
            # [Diagnostic] Check Group Status
            try:
                groups = self.r.xinfo_groups(STREAM_FUSED_MARKET)
                for g in groups:
                    if g[b'name'].decode() == REDIS_CFG['pg_group']:
                        logger.info(f"🚩 [Stream-Status] Group: {REDIS_CFG['pg_group']}, Last-Delivered-ID: {g[b'last-delivered-id'].decode()}, Pending: {g[b'pending']}")
            except Exception: pass
            
        logger.info(f"💾 PG Persistence Service Started (Tracking Date: {self.current_date}, DB: {target_db})")
        streams = {STREAM_FUSED_MARKET: '>',
                   STREAM_TRADE_LOG: '>',
                   STREAM_INFERENCE: '>' }
        
        last_throughput_log = time.time()
        msg_count = 0

        while True:
            try:
                self._check_date_rotation()
                consumer_name = REDIS_CFG.get('consumer', 'pg_worker_1')
                
                try:
                    resp = self.r.xreadgroup(REDIS_CFG['pg_group'], consumer_name, streams, count=100, block=1000)
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                    logger.warning("📡 Redis Connection Lost. Reconnecting...")
                    time.sleep(1)
                    self.r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=target_db)
                    continue

                if resp:
                    for sname, msgs in resp:
                        stream_name = sname.decode('utf-8')
                        msg_count += len(msgs)
                        for msg_id, data in msgs:
                            try:
                                payload = None
                                if b'batch' in data:
                                    # [New] Support for vectorized batch payloads (all streams)
                                    try:
                                        batch = ser.unpack(data[b'batch'])
                                        if isinstance(batch, list):
                                            for p in batch:
                                                if stream_name == STREAM_FUSED_MARKET:
                                                    self.process_market_data(p)
                                                elif stream_name == STREAM_INFERENCE and isinstance(p, dict):
                                                    self.process_feature_data(p)
                                        elif isinstance(batch, dict):
                                            payload = batch
                                    except Exception as e:
                                        logger.error(f"Batch Parse Error: {e}")
                                elif b'pickle' in data:
                                    payload = ser.unpack(data[b'pickle'])
                                elif b'data' in data:
                                    try: payload = ser.unpack(data[b'data'])
                                    except: pass
                                
                                if payload:
                                    if stream_name == STREAM_FUSED_MARKET:
                                        self.process_market_data(payload)
                                    elif stream_name == STREAM_TRADE_LOG:
                                        action = payload.get('action', '')
                                        if action == 'ALPHA':
                                            # [DEBUG] Alpha reception logging
                                            #logger.info(f"📥 [Alpha Debug] Received ALPHA signal for {payload.get('symbol')} from {stream_name}. Value: {payload.get('alpha')}")
                                            self.alpha_buffer.append((
                                                payload['ts'], payload['symbol'], 
                                                float(payload.get('alpha', 0)),
                                                float(payload.get('iv', 0)),
                                                float(payload.get('price', 0)),
                                                float(payload.get('vol_z', 0)),
                                            ))
                                        else:
                                            mode = payload.get('mode', 'REALTIME')
                                            self.trade_buffer.append((
                                                payload['ts'], payload['symbol'], payload['action'],
                                                float(payload.get('qty', 0)), float(payload.get('price', 0)),
                                                json.dumps(payload), mode
                                            ))
                                            
                                    elif stream_name == STREAM_INFERENCE:
                                        self.process_feature_data(payload)
                                        
                                self.r.xack(stream_name, REDIS_CFG['pg_group'], msg_id)

                            except Exception as e:
                                logger.error(f"Message Parse Error ({stream_name}): {e}")
                                self.r.xack(stream_name, REDIS_CFG['pg_group'], msg_id)

                else:
                    # 🚀 [DEBUG] 带频率限制的空读提示
                    if time.time() % 10 < 1:
                        logger.info(f"💤 [Persistence] Waiting for new messages in streams ({list(streams.keys())})...")

                if time.time() - self.last_flush > 1.0:
                    self.flush()

                # 🚀 [Heartbeat] Periodic throughput summary
                if time.time() - last_throughput_log > 60:
                    # if msg_count > 0:
                    #     logger.info(f"📊 [Persistence Heartbeat] Processed {msg_count} messages in last 60s.")
                    msg_count = 0
                    last_throughput_log = time.time()

            except redis.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("⚠️ Consumer Group missing for Persistence. Recreating...")
                    self._init_consumer_groups()
                else:
                    logger.error(f"Redis Error: {e}")
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                # 必须 rollback 否则 PostgreSQL 连接会卡到 aborted 状态
                try:
                    if self.conn and self.conn.closed == 0:
                        self.conn.rollback()
                except Exception:
                    self.conn = None  # 强制重连
                time.sleep(1)

if __name__ == "__main__":
    service = DataPersistenceServicePG()
    service.run()
