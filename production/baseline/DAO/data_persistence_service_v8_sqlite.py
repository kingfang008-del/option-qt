import redis
import sqlite3
import pickle
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
# ================= 配置区域 =================


import sys
sys.path.append(str(Path(__file__).parent.parent)) # Add V8 root
from config import REDIS_CFG, DB_DIR, STREAM_FUSED_MARKET, STREAM_TRADE_LOG, STREAM_INFERENCE, LOG_DIR, NY_TZ
from utils import serialization_utils as ser
import os

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [Persistence_1M] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "Persistence.log", mode='a', encoding='utf-8')
    ]
)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Persistence_1M] - %(levelname)s - %(message)s')
logger = logging.getLogger("DataPersistence")

# Local consumer group config (extends base)
REDIS_CFG['group'] = 'persistence_group'
REDIS_CFG['consumer'] = 'sqlite_writer_1m'

# [关键] 不要在这里捕获 DB_DIR，否则后续 Patch 无效
# DATA_DIR = DB_DIR 

class DataPersistenceServiceSQLite:
    def __init__(self, start_date=None, db_dir=None):
        # 仅过滤出 redis.Redis 接受的参数
        redis_kwargs = {k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db', 'password', 'decode_responses']}
        self.r = redis.Redis(**redis_kwargs)
        
        # 🚀 [终极修复] 允许从外部传入逻辑开始日期 (回测/回放专用)
        if start_date:
            self.current_date = str(start_date)
            logger.info(f"📅 Persistence Service Initialized with Logical Date: {self.current_date}")
        else:
            self.current_date = datetime.now(NY_TZ).strftime('%Y%m%d')
            logger.info(f"📅 Persistence Service Initialized with System Date: {self.current_date}")
        
        # [NEW] 允许指定数据库目录
        self.forced_db_dir = db_dir
        
        self.db_path = None
        self.conn = None
        self.cursor = None
        
        # 内存聚合缓冲区
        # bar_buffer[key] = {symbol, ts, open, high, low, close, volume}
        # key = (symbol, minute_ts)
        self.bar_buffer = {}
        
        # [新增] 期权快照缓冲区
        # option_buffer[key] = buckets_json_string
        # key = (symbol, minute_ts)
        self.option_buffer = {}
        
        self.trade_buffer = []
        self.alpha_buffer = [] # [New]
        self.last_flush = time.time()
        self.flush_interval = 1.0 # 秒
        
        self._init_db()

    def _init_db(self):
        """初始化数据库 (分钟线模式 + WAL)"""
        base_dir = self.forced_db_dir if self.forced_db_dir else DB_DIR
        db_path = os.path.join(base_dir, f"market_{self.current_date}.db")
        
        # check_same_thread=False 允许在 Flush 线程操作
        self.conn = sqlite3.connect(str(db_path), timeout=60.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # 开启 WAL，解决读写锁冲突
        self.conn.execute("PRAGMA synchronous=NORMAL;") 
        self.cursor = self.conn.cursor()
        
        # 1. 建表: 1分钟 K线 (market_bars_1m)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_bars_1m (
                symbol TEXT,
                ts INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, ts)
            )
        """)
        
        # 2. 建表: 交易日志
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                ts REAL,
                datetime_ny TEXT,  -- [New] 存储 NY 时间字符串
                symbol TEXT,
                action TEXT,
                qty REAL,
                price REAL,
                details_json TEXT
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trd_sym_ts ON trade_logs (symbol, ts)")

        # [新增] 2b. 建表: 交易日志 (Backtest) — 完全独立
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs_backtest (
                ts REAL,
                datetime_ny TEXT,  -- [New] 存储 NY 时间字符串
                symbol TEXT,
                action TEXT,
                qty REAL,
                price REAL,
                details_json TEXT
            )
        """)
        
        # [新增] 3. 建表: Alpha 历史记录 (用于 Warmup 回溯)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpha_logs (
                ts REAL,
                datetime_ny TEXT,
                symbol TEXT,
                alpha REAL,
                iv REAL,
                price REAL,
                vol_z REAL,
                event_prob REAL,
                PRIMARY KEY (symbol, ts)
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alpha_ts ON alpha_logs (ts)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_trd_bk_sym_ts ON trade_logs_backtest (symbol, ts)")

        # [Migration] 如果表已存在但没有 datetime_ny 或 event_prob 列，补全
        for table in ['trade_logs', 'trade_logs_backtest']:
            try:
                self.cursor.execute(f"SELECT datetime_ny FROM {table} LIMIT 1")
            except:
                self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN datetime_ny TEXT")

        try:
            self.cursor.execute("SELECT event_prob FROM alpha_logs LIMIT 1")
        except:
            self.cursor.execute("ALTER TABLE alpha_logs ADD COLUMN event_prob REAL DEFAULT 0.0")

        # [新增] 3. 建表: 期权快照 (1分钟聚合)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_snapshots_1m (
                symbol TEXT,
                ts INTEGER,
                buckets_json TEXT,
                PRIMARY KEY (symbol, ts)
            )
        """)
        
        # [新增] 4. 建表: 归一化特征日志 (用于 EOD 回测和 Debug)
        # 使用 BLOB 存储 float32 数组，节省空间且读写最快
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_logs (
                symbol TEXT,
                ts INTEGER,
                fast_norm_blob BLOB,  -- Fast Channel Normalized Tensor
                slow_norm_blob BLOB,  -- Slow Channel Normalized Tensor
                PRIMARY KEY (symbol, ts)
            )
        """)
        self.conn.commit()
        logger.info(f"📂 SQLite DB Initialized: {db_path} (Mode: 1-Min Bars + Options)")

    # 再次确认 data_persistence_service_v8_sqlite.py 中的此方法逻辑
    def process_feature_data(self, payload):
        try:
            ts = int(payload.get('ts', time.time()))
            
            # [关键] 检查是否为 Batch 模式
            if 'symbols' in payload and isinstance(payload['symbols'], list):
                symbols = payload['symbols']
                fast_batch = payload.get('fast_1m')
                slow_batch = payload.get('slow_1m')
                
                rows_to_insert = []
                for i, sym in enumerate(symbols):
                    fast_blob = None
                    if fast_batch is not None and len(fast_batch) > i:
                        arr = fast_batch[i]
                        if len(arr.shape) > 1: arr = arr[-1] # 取最新时间步
                        fast_blob = arr.astype(np.float32).tobytes()
                    
                    slow_blob = None
                    if slow_batch is not None and len(slow_batch) > i:
                        arr = slow_batch[i]
                        if len(arr.shape) > 1: arr = arr[-1] # 取最新时间步
                        slow_blob = arr.astype(np.float32).tobytes()
                    
                    if fast_blob or slow_blob:
                        rows_to_insert.append((sym, ts, fast_blob, slow_blob))
                
                if rows_to_insert:
                    self.cursor.executemany(
                        "REPLACE INTO feature_logs (symbol, ts, fast_norm_blob, slow_norm_blob) VALUES (?, ?, ?, ?)",
                        rows_to_insert
                    )
                    # 可以在这里加一个日志确认入库成功
                    # logger.info(f"💾 Persisted features for {len(rows_to_insert)} symbols")

            # 兼容旧版
            elif 'symbol' in payload:
                # ... 旧版逻辑 ...
                pass
                
        except Exception as e:
            logger.error(f"Feature Write Error: {e}")
            
    def _check_date_rotation(self, ts=None):
        from config import NY_TZ
        # 🚀 [终极修复] 优先使用 Payload TS 进行回放时序对齐，否则回拨模式下无法跨天
        if ts is not None:
            now_date = datetime.fromtimestamp(ts, NY_TZ).strftime('%Y%m%d')
        else:
            now_date = datetime.now(NY_TZ).strftime('%Y%m%d')
            
        if now_date != self.current_date:
            logger.info(f"📅 [Sync Rotation] Date: {self.current_date} -> {now_date}")
            # 切换前强制刷盘
            self.flush()
            
            if self.conn: self.conn.close()
            self.current_date = now_date
            self.bar_buffer.clear()
            self.option_buffer.clear()
            self.trade_buffer.clear()
            self.alpha_buffer.clear()
            self._init_db()

    def process_market_data(self, payload):
        """
        核心逻辑: 将接收到的 Tick/Snapshot 聚合成 1分钟 K线 和 期权快照
        """
        try:
            # 🚀 [终极修复] 兼容 Batch 模式 (list of dicts)
            if isinstance(payload, list):
                for item in payload:
                    self.process_market_data(item)
                return

            ts = float(payload['ts'])
            symbol = payload['symbol']
            
            # 对齐到分钟起始时间 (向下取整)
            minute_ts = int(ts) // 60 * 60
            key = (symbol, minute_ts)

            # --- 1. 处理 Stock Data ---
            stock = payload.get('stock', {})
            price = float(stock.get('close', 0.0))
            vol = float(stock.get('volume', 0.0))
            
            if price > 0:
                if key not in self.bar_buffer:
                    # 新的分钟 bar
                    self.bar_buffer[key] = {
                        'symbol': symbol, 
                        'ts': minute_ts,
                        'open': price, 'high': price, 'low': price, 'close': price,
                        'volume': vol
                    }
                else:
                    # 更新现有 bar
                    b = self.bar_buffer[key]
                    b['high'] = max(b['high'], price)
                    b['low'] = min(b['low'], price)
                    b['close'] = price
                    b['volume'] += vol # 假设是 Tick 增量或需要累加
            
            # --- [新增] 2. 处理 Option Data ---
            if 'option_buckets' in payload:
                # 策略: 每分钟保留最新的快照 (Last Snapshot Wins)
                snap_obj = {'buckets': payload['option_buckets']}
                contracts = payload.get('option_contracts', [])
                if contracts:
                    snap_obj['contracts'] = contracts
                self.option_buffer[key] = json.dumps(snap_obj)
                
        except Exception as e:
            logger.error(f"Process Data Error inside process_market_data: {e}")
            pass

    def flush(self):
        """将内存数据刷入磁盘"""
        if not self.conn: return
        if not (self.bar_buffer or self.trade_buffer or self.option_buffer or self.alpha_buffer): return
        
        try:
            # 为了防止内存溢出，我们只保留最近 300 秒的 bar 在内存，旧的从 buffer 移除
            current_cutoff = time.time() - 300 # 5分钟前的数据才清理 Buffer (防止乱序)
            
            # 1. 写入 K线 (使用 REPLACE INTO 实现更新)
            if self.bar_buffer:
                bars_to_write = []
                keys_to_delete = []
                
                for key, bar in self.bar_buffer.items():
                    bars_to_write.append((
                        bar['symbol'], bar['ts'], 
                        bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                    ))
                    if key[1] < current_cutoff:
                        keys_to_delete.append(key)

                self.cursor.executemany(
                    "REPLACE INTO market_bars_1m (symbol, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    bars_to_write
                )
                
                for k in keys_to_delete:
                    del self.bar_buffer[k]

            # [新增] 2. 写入期权快照
            if self.option_buffer:
                opts_to_write = []
                opt_keys_to_delete = []
                
                for key, buckets_json in self.option_buffer.items():
                    opts_to_write.append((key[0], key[1], buckets_json))
                    
                    if key[1] < current_cutoff:
                        opt_keys_to_delete.append(key)
                
                self.cursor.executemany(
                    "REPLACE INTO option_snapshots_1m (symbol, ts, buckets_json) VALUES (?, ?, ?)",
                    opts_to_write
                )
                
                for k in opt_keys_to_delete:
                    del self.option_buffer[k]

            # 3. 写入交易 (分表逻辑)
            if self.trade_buffer:
                realtime_logs = []
                backtest_logs = []
                
                for item in self.trade_buffer:
                    # item = (ts, symbol, action, qty, price, details_json, mode)
                    ts = item[0]
                    # 计算 NY Time String
                    from config import NY_TZ
                    dt_ny = datetime.fromtimestamp(ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 组装 Row: (ts, datetime_ny, symbol, action, qty, price, details_json)
                    row = (ts, dt_ny, item[1], item[2], item[3], item[4], item[5]) 
                    mode = item[6]
                    
                    if mode == 'BACKTEST':
                        backtest_logs.append(row)
                    else:
                        realtime_logs.append(row)
                
                if realtime_logs:
                    self.cursor.executemany(
                        "INSERT INTO trade_logs VALUES (?,?,?,?,?,?,?)",
                        realtime_logs
                    )
                
                if backtest_logs:
                    self.cursor.executemany(
                        "INSERT INTO trade_logs_backtest VALUES (?,?,?,?,?,?,?)",
                        backtest_logs
                    )
                    
                self.trade_buffer = []

            # [新增] 4. 写入 Alpha Logs
            if self.alpha_buffer:
                alpha_rows = []
                for item in self.alpha_buffer:
                    # item = (ts, symbol, alpha, iv, price, vol_z, event_prob)
                    ts = item[0]
                    from config import NY_TZ
                    dt_ny = datetime.fromtimestamp(ts, NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                    alpha_rows.append((ts, dt_ny, item[1], item[2], item[3], item[4], item[5], item[6]))
                
                self.cursor.executemany("REPLACE INTO alpha_logs VALUES (?,?,?,?,?,?,?,?)", alpha_rows)
                self.alpha_buffer = []

            self.conn.commit()
            self.last_flush = time.time()
            if len(self.alpha_buffer) > 0 or len(self.trade_buffer) > 0:
                logger.debug(f"💾 Flush Done: {len(self.alpha_buffer)} Alphas, {len(self.trade_buffer)} Trades")
            
        except Exception as e:
            logger.error(f"Flush Error: {e}")

    # 替换 data_persistence_service_v8_sqlite.py 中的 run 方法

    def run(self):
        logger.info(f"💾 Persistence Service Started (DB: market_{self.current_date}.db)")
        streams = {STREAM_FUSED_MARKET: '>',
                   STREAM_TRADE_LOG: '>',
                   STREAM_INFERENCE: '>' }
        
        while True:
            try:
                resp = self.r.xreadgroup(REDIS_CFG['group'], REDIS_CFG['consumer'], streams, count=100, block=1000)

                if resp:
                    for sname, msgs in resp:
                        stream_name = sname.decode('utf-8')
                        for msg_id, data in msgs:
                            try:
                                payload = None
                                # [关键修复] 使用统一解码器
                                if b'data' in data:
                                    payload = ser.unpack(data[b'data'])
                                elif b'batch' in data:
                                    payload = ser.unpack(data[b'batch'])
                                elif b'pickle' in data: # 兼容旧格式
                                    payload = ser.unpack(data[b'pickle'])
                                
                                if payload:
                                    # 🚀 [核心修复] 执行毫秒级日期探测
                                    msg_ts = None
                                    if isinstance(payload, list) and len(payload) > 0:
                                        msg_ts = payload[0].get('ts')
                                    elif isinstance(payload, dict):
                                        msg_ts = payload.get('ts')
                                    
                                    if msg_ts:
                                        self._check_date_rotation(ts=float(msg_ts))

                                    if stream_name == STREAM_FUSED_MARKET:
                                        self.process_market_data(payload)
                                    elif stream_name == STREAM_TRADE_LOG:
                                        # [关键] 区分 Trade Log 和 Alpha Log
                                        action = payload.get('action', '')
                                        
                                        if action == 'ALPHA':
                                            # Alpha Log 写入 buffer
                                            # payload: {ts, symbol, action='ALPHA', alpha, iv, price, vol_z, event_prob}
                                            self.alpha_buffer.append((
                                                payload['ts'], payload['symbol'], 
                                                float(payload.get('alpha', 0)),
                                                float(payload.get('iv', 0)),
                                                float(payload.get('price', 0)),
                                                float(payload.get('vol_z', 0)),
                                                float(payload.get('event_prob', 0))
                                            ))
                                        else:
                                            # Trade Log 写入 buffer
                                            mode = payload.get('mode', 'REALTIME')
                                            self.trade_buffer.append((
                                                payload['ts'], payload['symbol'], payload['action'],
                                                float(payload.get('qty', 0)), float(payload.get('price', 0)), 
                                                json.dumps(payload),
                                                mode
                                            ))
                                            
                                    elif stream_name == STREAM_INFERENCE:
                                        # 调用刚才修复过的 Batch 处理逻辑
                                        self.process_feature_data(payload)
                                        
                                # ACK
                                self.r.xack(stream_name, REDIS_CFG['group'], msg_id)

                            except Exception as e:
                                logger.error(f"Message Parse Error ({stream_name}): {e}")
                                # Skip bad message
                                self.r.xack(stream_name, REDIS_CFG['group'], msg_id)

                if time.time() - self.last_flush > 1.0:
                    self.flush()

            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    service = DataPersistenceServiceSQLite()
    service.run()