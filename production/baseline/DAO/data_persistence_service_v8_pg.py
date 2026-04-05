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
    def __init__(self):
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
        self.current_date = None
        
        # Buffer
        self.bar_buffer = {}
        self.option_buffer = {}
        self.bar_buffer_5m = {}
        self.option_buffer_5m = {}
        
        self.trade_buffer = []
        self.alpha_buffer = []
        self.last_flush = time.time()
        
        self._check_date_rotation() # This will init db and create today's partition

    def _get_pg_conn(self):
        if not self.conn or self.conn.closed != 0:
            self.conn = psycopg2.connect(PG_DB_URL)
        return self.conn

    def _init_master_tables(self):
        """Initialize PostgreSQL Master Partitioned Tables"""
        conn = self._get_pg_conn()
        c = conn.cursor()
        
        # 1. market_bars_1m & 5m
        for tbl in ['market_bars_1m', 'market_bars_5m']:
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    symbol TEXT,
                    ts DOUBLE PRECISION,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
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
        for tbl in ['option_snapshots_1m', 'option_snapshots_5m']:
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
            'market_bars_1m', 'market_bars_5m', 'trade_logs', 'trade_logs_backtest', 
            'alpha_logs', 'option_snapshots_1m', 'option_snapshots_5m', 'feature_logs'
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

    def _check_date_rotation(self):
        from config import NY_TZ
        now_dt = datetime.now(NY_TZ)
        now_date_str = now_dt.strftime('%Y%m%d')
        
        if now_date_str != self.current_date:
            logger.info(f"📅 Date rotation: {self.current_date} -> {now_date_str}")
            self.current_date = now_date_str
            self.bar_buffer.clear()
            self.option_buffer.clear()
            self.bar_buffer_5m.clear()
            self.option_buffer_5m.clear()
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

    def process_feature_data(self, payload):
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
                
                if rows_to_insert:
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
                
        except Exception as e:
            logger.error(f"Feature Write Error: {e}")

    def process_market_data(self, payload):
        try:
            ts = float(payload['ts'])
            symbol = payload['symbol']
            
            minute_ts = int(ts) // 60 * 60
            key = (symbol, minute_ts)

            # 1min Bar
            stock = payload.get('stock', {})
            price = float(stock.get('close', 0.0))
            vol = float(stock.get('volume', 0.0))
            
            if price > 0:
                if key not in self.bar_buffer:
                    self.bar_buffer[key] = {
                        'symbol': symbol, 
                        'ts': minute_ts,
                        'open': price, 'high': price, 'low': price, 'close': price,
                        'volume': vol
                    }
                else:
                    b = self.bar_buffer[key]
                    b['high'] = max(b['high'], price)
                    b['low'] = min(b['low'], price)
                    b['close'] = price
                    b['volume'] += vol
            
            # 5min Bar
            stock_5m = payload.get('stock_5m', {})
            price_5m = float(stock_5m.get('close', 0.0))
            if price_5m > 0:
                self.bar_buffer_5m[key] = {
                    'symbol': symbol, 'ts': minute_ts,
                    'open': float(stock_5m['open']), 'high': float(stock_5m['high']),
                    'low': float(stock_5m['low']), 'close': price_5m,
                    'volume': float(stock_5m['volume'])
                }

            # 1min Options
            if 'option_buckets' in payload:
                snap_obj = payload['option_buckets']
                contracts = payload.get('option_contracts', [])
                if contracts:
                    snap_obj = {'buckets': payload['option_buckets'], 'contracts': contracts}
                self.option_buffer[key] = json.dumps(snap_obj)

            # 5min Options Fallback (🚀 复用 1min 桶填补 5min 缺口)
            opt_buckets_5m = payload.get('option_buckets_5m')
            if not opt_buckets_5m and (minute_ts % 300 == 0):
                opt_buckets_5m = payload.get('option_buckets')

            if opt_buckets_5m:
                contracts = payload.get('option_contracts', [])
                snap_obj_5m = {'buckets': opt_buckets_5m, 'contracts': contracts} if contracts else opt_buckets_5m
                self.option_buffer_5m[key] = json.dumps(snap_obj_5m)
                
        except Exception as e:
            pass

    def flush(self):
        if not (self.bar_buffer or self.trade_buffer or self.option_buffer or self.alpha_buffer): return
        
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            current_cutoff = time.time() - 300 
            
            # 1. Bars (1m & 5m)
            for buf, tbl in [(self.bar_buffer, 'market_bars_1m'), (self.bar_buffer_5m, 'market_bars_5m')]:
                if buf:
                    rows = []
                    keys_to_del = []
                    for key, b in buf.items():
                        rows.append((b['symbol'], b['ts'], b['open'], b['high'], b['low'], b['close'], b['volume']))
                        if key[1] < current_cutoff: keys_to_del.append(key)
                    
                    psycopg2.extras.execute_batch(c, f"""
                        INSERT INTO {tbl} (symbol, ts, open, high, low, close, volume) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, ts) DO UPDATE 
                        SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
                    """, rows)
                    for k in keys_to_del: del buf[k]

            # 2. Options (1m & 5m)
            for buf, tbl in [(self.option_buffer, 'option_snapshots_1m'), (self.option_buffer_5m, 'option_snapshots_5m')]:
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
                                    # [New] Support for V8 Vectorized Batch payloads
                                    try:
                                        batch = ser.unpack(data[b'batch'])
                                        for p in batch:
                                            if stream_name == STREAM_FUSED_MARKET:
                                                self.process_market_data(p)
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
                                                float(payload.get('vol_z', 0))
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
                    if msg_count > 0:
                        logger.info(f"📊 [Persistence Heartbeat] Processed {msg_count} messages in last 60s.")
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
