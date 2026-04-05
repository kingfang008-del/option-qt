import streamlit as st
import redis
import json
import pickle
import sys
import os
from pathlib import Path

# [NEW] Add project root to sys.path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import serialization_utils as ser
import pandas as pd
import numpy as np
import time

import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, time as dt_time
from pathlib import Path
from pytz import timezone
import subprocess
import shutil
import shutil
import sys

# [Fix 1] 设置页面为宽屏模式 (必须在其他 st 命令前)
st.set_page_config(page_title="V8 Ultimate Monitor", layout="wide", page_icon="📈")

# [Modified] Load Config from Parent
import sys
sys.path.append(str(Path(__file__).parent.parent)) # Add V8 root to path
from config import (
    REDIS_CFG, DB_DIR, NY_TZ, DASHBOARD_REFRESH_RATE, DATA_DIR,
    HASH_OPTION_SNAPSHOT, STREAM_FUSED_MARKET, STREAM_INFERENCE,
    STREAM_TRADE_LOG, BUCKET_SPECS, IBKR_PORT, PG_DB_URL # [New] 引入 PG_DB_URL
)

# [Fix 1] 补全 Dashboard 所需的 Redis Key 映射
REDIS_CFG['hash_snapshot']    = HASH_OPTION_SNAPSHOT     # 'live_option_snapshot'
REDIS_CFG['raw_stream']       = STREAM_FUSED_MARKET      # 'fused_market_stream'
REDIS_CFG['inference_stream'] = STREAM_INFERENCE          # 'unified_inference_stream'
REDIS_CFG['trade_log']        = STREAM_TRADE_LOG          # 'trade_log_stream'
REDIS_CFG['alpha_key_prefix'] = 'alpha_log'              # Redis List: alpha_log:{symbol}

# 路径配置
BASE_PROJECT_DIR = Path(__file__).parent.parent  # V8 dir
SQLITE_DATA_DIR  = DB_DIR
 
BACKTEST_ROOT    = DATA_DIR / "backtest"
SCRIPT_DIR       = BASE_PROJECT_DIR / "daily_backtest"  # [Fix 3] S0/S5 脚本在此目录

# [Fix 2] 加载特征名 (从 JSON 配置文件)
import json as _json

def _load_feat_names(json_path, resolution=None):
    """从 feature JSON 提取 features[].name 列表，可选过滤分辨率"""
    try:
        if not json_path.exists(): return []
        with open(json_path, 'r') as f:
            cfg = _json.load(f)
        feats = cfg.get('features', [])
        if resolution:
            return [feat['name'] for feat in feats if feat.get('resolution', '1min') == resolution]
        return [feat['name'] for feat in feats]
    except Exception:
        return []

def fetch_latest_mock_cash():
    """从 PostgreSQL symbol_state 表中读取 _GLOBAL_STATE_ 里的 mock_cash"""
    # 👇 [🔥 修复：新增回放模式拦截] 👇
    from config import IS_LIVEREPLAY, TRADING_ENABLED
    # 如果系统处于回放模式，或者禁止了实盘交易，绝对不能读数据库里的历史残留资金！
    if IS_LIVEREPLAY or not TRADING_ENABLED:
        return None
    # 👆 修复结束 👆

    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute("SELECT data FROM symbol_state WHERE symbol = '_GLOBAL_STATE_'")
        row = c.fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            return float(data.get('mock_cash', 0.0))
    except: pass
    return None

_FEAT_JSON_DIR = SCRIPT_DIR  # daily_backtest/
FAST_FEAT_NAMES = _load_feat_names(_FEAT_JSON_DIR / "fast_feature.json")
SLOW_FEAT_NAMES = _load_feat_names(_FEAT_JSON_DIR / "slow_feature.json")

# Fallback: 如果 JSON 不存在，用占位名
if not FAST_FEAT_NAMES:
    FAST_FEAT_NAMES = [f"fast_{i}" for i in range(30)]
if not SLOW_FEAT_NAMES:
    SLOW_FEAT_NAMES = [f"slow_{i}" for i in range(22)]

# 模型评估参数
IC_WINDOW = 30      # 滚动 IC 窗口
FORWARD_PERIOD = 5  # 预测未来 5min 收益

# 样式美化与主题切换
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark'

theme_col1, theme_col2 = st.sidebar.columns([1, 2])
theme_col1.write("🎨 Theme:")
if theme_col2.toggle("Light Mode", value=(st.session_state['theme'] == 'light')):
    st.session_state['theme'] = 'light'
else:
    st.session_state['theme'] = 'dark'

is_light = (st.session_state['theme'] == 'light')

# 定义全局背景与字体颜色
bg_color = "#FFFFFF" if is_light else "#0E1117"
text_color = "#111111" if is_light else "#FAFAFA"
card_bg = "#F0F2F6" if is_light else "#1E1E1E"
PLOTLY_THEME = "plotly_white" if is_light else "plotly_dark"

# 侧边栏和顶部的背景色 (浅色模式使用浅蓝色，深色模式稍微深一点)
sidebar_bg = "#EBF5FB" if is_light else "#111826"
header_bg = "#EBF5FB" if is_light else "transparent"

st.markdown(f"""
    <style>
    /* 强制全量覆盖基础背景色 */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    /* 左侧边栏 */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    /* 顶部 Navbar */
    header[data-testid="stHeader"] {{
        background-color: {header_bg} !important;
    }}
    .metric-card {{
        background-color: {card_bg};
        border-left: 5px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ height: 45px; }}
    </style>
    """, unsafe_allow_html=True)

# ================= 2. 数据访问层 (Data Access) =================

@st.cache_resource
def get_redis_client():
    try:
        pool = redis.ConnectionPool(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=REDIS_CFG['db'], decode_responses=False)
        return redis.Redis(connection_pool=pool)
    except Exception as e:
        st.error(f"Redis Connection Failed: {e}")
        return None
    

def get_pg_health_status():
    """
    检查 Postgres 统一数据库的健康状况：
    1. K线数量 (用于 SMA 计算)
    2. 数据新鲜度 (是否还在更新)
    3. 期权快照完整性 (是否包含 Greeks/IV)
    """
    status = {
        'exists': True,
        'path': 'PostgreSQL quant_trade',
        'size_mb': 0,
        'last_update': datetime.now().strftime('%H:%M:%S'),
        'stocks': {}, # symbol -> {count, last_ts, status}
        'options': {} # symbol -> {has_snap, shape_ok, last_ts}
    }
    
    try:
        conn = get_pg_conn()
        cursor = conn.cursor()
        
        # 获取 DB 大小
        cursor.execute("SELECT pg_size_pretty(pg_database_size('quant_trade')), pg_database_size('quant_trade')")
        row = cursor.fetchone()
        if row:
            status['size_mb'] = round(row[1] / (1024 * 1024), 2)
        
        # --- 1. 检查 Stock Bars (用于 VWAP/SMA) ---
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'market_bars_1m');")
        if cursor.fetchone()[0]:
            # 统计当天数据 (0级性能)
            today_start_ts = int(datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            cursor.execute(f"""
                SELECT symbol, COUNT(*) as cnt, MAX(ts) as last_ts, MIN(ts) as first_ts
                FROM market_bars_1m 
                WHERE ts >= {today_start_ts}
                GROUP BY symbol
            """)
            for row in cursor.fetchall():
                sym, cnt, last_ts, first_ts = row
                lag = time.time() - last_ts
                health = "🟢 OK"
                if lag > 300: health = "🔴 Stale (>5m)"
                if cnt < 60: health = "🟡 Warmup (<60m)"
                
                status['stocks'][sym] = {
                    'count': cnt,
                    'lag_sec': int(lag),
                    'health': health,
                    'last_time': datetime.fromtimestamp(last_ts).strftime('%H:%M'),
                    'first_time': datetime.fromtimestamp(first_ts).strftime('%H:%M')
                }

        # --- 2. 检查 Option Snapshots ---
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'option_snapshots_1m');")
        if cursor.fetchone()[0]:
            today_start_ts = int(datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            cursor.execute(f"SELECT DISTINCT symbol FROM option_snapshots_1m WHERE ts >= {today_start_ts}")
            symbols = [r[0] for r in cursor.fetchall()]
            
            for sym in symbols:
                cursor.execute("""
                    SELECT ts, buckets_json FROM option_snapshots_1m 
                    WHERE symbol = %s ORDER BY ts DESC LIMIT 1
                """, (sym,))
                row = cursor.fetchone()
                if row:
                    ts, json_str = row
                    try:
                        data = dict(json_str) if isinstance(json_str, dict) else json.loads(json_str)
                        if isinstance(data, dict):
                            arr = np.array(data.get('buckets', []))
                        else:
                            arr = np.array(data)
                        shape = arr.shape
                        shape_ok = (len(shape) == 2 and shape[0] == 6 and shape[1] >= 10)
                        
                        status['options'][sym] = {
                            'last_ts': datetime.fromtimestamp(ts).strftime('%H:%M:%S'),
                            'lag_sec': int(time.time() - ts),
                            'shape': str(shape),
                            'quality': "🟢 Good" if shape_ok else f"🔴 Bad Shape {shape}"
                        }
                    except:
                        status['options'][sym] = {'quality': "❌ Corrupt JSON"}
        
        conn.close()
    except Exception as e:
        status['error'] = str(e)
        status['exists'] = False
        
    return status


def get_stream_lag_and_data(r, stream_key, symbol=None):
    """读取 Stream 最新一条并计算物理延迟 (System Latency)"""
    try:
        items = r.xrevrange(stream_key, count=1)
        if not items: return 999.0, None
        
        # items[0] = (message_id, data_dict)
        msg_id_str = items[0][0].decode('utf-8')
        arrival_ts = int(msg_id_str.split('-')[0]) / 1000.0
        lag = time.time() - arrival_ts
        
        payload = items[0][1]
        val = None
        if b'pickle' in payload: val = ser.unpack(payload[b'pickle'])
        elif b'data' in payload: val = ser.unpack(payload[b'data'])
        
        # 过滤 Symbol
        if val and symbol and 'batch_packet' in val:
            if symbol not in val['batch_packet']['symbols']:
                return 999.0, None 
                
        return lag, val
    except Exception:
        return 999.0, None
    
def get_hash_snapshot(r, sym):
    try:
        data = r.hget(REDIS_CFG['hash_snapshot'], sym)
        if data:
            val = ser.unpack(data)
            return time.time() - val.get('ts', 0), val
    except: pass
    return 999.0, None

@st.cache_data(ttl=300)
def fetch_missing_trades(days=1):
    """
    [New] 统计满足条件的 alpha / vol 但未开仓的信号
    条件: abs(alpha) >= 0.8 且 vol_z >= -1.0
    """
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        # 今天 0 点
        today_start = datetime.now(NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        ts_start = int(today_start.timestamp())
        
        query = f"""
            WITH potentials AS (
                SELECT ts, datetime_ny as time, symbol, alpha, iv, price, vol_z
                FROM alpha_logs
                WHERE ts >= {ts_start}
                AND (alpha >= 0.8 OR alpha <= -0.8)
                AND vol_z >= -1.0
            ),
            actual_trades AS (
                SELECT DISTINCT symbol, ts
                FROM trade_logs
                WHERE ts >= {ts_start} AND action = 'OPEN'
            )
            SELECT p.*
            FROM potentials p
            LEFT JOIN actual_trades t ON p.symbol = t.symbol 
                AND ABS(p.ts - t.ts) < 120
            WHERE t.symbol IS NULL
            ORDER BY p.ts DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching missing trades: {e}")
        return pd.DataFrame()

def fetch_model_logs(r, symbol):
    """从 PostgreSQL 拉取历史 Alpha 记录 (支持周末自动回退)"""
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        ts_3_days_ago = int((datetime.now(NY_TZ) - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        # Fetch up to last 3 days to handle weekends
        query = f"""
            SELECT ts, alpha as a, iv, price as p, vol_z as v 
            FROM alpha_logs 
            WHERE symbol='{symbol}' 
            AND ts >= {ts_3_days_ago}
            ORDER BY ts ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if not df.empty:
            df['dt'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            return df
    except Exception as e:
        print(f"Error fetching model logs from PG: {e}")
            
    return pd.DataFrame()

def load_today_sqlite_data(symbol):
    """从 PostgreSQL 加载 1分钟 K线数据 (支持周末自动回退)"""
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        ts_3_days_ago = int((datetime.now(NY_TZ) - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        query = f"""
            SELECT ts, open, high, low, close as price, volume 
            FROM market_bars_1m 
            WHERE symbol = '{symbol}'
            AND ts >= {ts_3_days_ago}
            ORDER BY ts ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            return df
    except Exception as e:
        print(f"Error loading market bars from PG: {e}")
        pass
            
    return pd.DataFrame()

def load_trade_logs_for_chart(r, symbol):
    """加载特定标的的所有交易记录用于绘图"""
    try:
        entries = r.xrevrange(REDIS_CFG['trade_log'], count=1000)
        trades = []
        for _, payload in entries:
            if b'pickle' in payload:
                msg = ser.unpack(payload[b'pickle'])
                # [Fix] 过滤掉 ALPHA 日志，只保留真实交易
                if msg.get('action') == 'ALPHA': continue
                
                if msg.get('symbol') == symbol:
                    try:
                        note = json.loads(msg.get('strategy_note', '{}'))
                        msg['tag'] = note.get('tag', '')
                    except:
                        msg['tag'] = ''
                    trades.append(msg)
        
        if not trades: return pd.DataFrame()
        df = pd.DataFrame(trades)
        df['datetime'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        return df
    except:
        return pd.DataFrame()

def format_option_matrix(buckets, contracts=None):
    """格式化期权矩阵, 可选附带合约号"""
    # [修改前] cols = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Strike', 'Vol(Cum)', 'IV']
    # [修改后] 加入 Bid 和 Ask 匹配 10 个维度
    cols = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Strike', 'Vol(Cum)', 'IV', 'Bid', 'Ask', 'BidSize', 'AskSize']
    rows = ['Front P_ATM', 'Front P_OTM', 'Front C_ATM', 'Front C_OTM', 'Next P_ATM', 'Next C_ATM']
    
    arr = np.array(buckets)
    if arr.ndim < 2: return pd.DataFrame() # 防御性编程
    
    # [修改前] if arr.shape[0] < 6: arr = np.vstack([arr, np.zeros((6-arr.shape[0], 8))])
    # [修改后] 动态匹配宽度，防止崩溃
    width = arr.shape[1]
    if arr.shape[0] < 6: 
        arr = np.vstack([arr, np.zeros((6-arr.shape[0], width))])
        
    df = pd.DataFrame(arr[:, :12], columns=cols[:width], index=rows) # 截取前 12 列防溢出
    if contracts and len(contracts) >= 6:
        df.insert(0, 'Contract', contracts[:6])
    else:
        df.insert(0, 'Contract', ['—'] * 6)
    return df


def load_option_price_history(symbol):
    """从 PostgreSQL option_snapshots_1m 加载期权价格历史 (用于 Tab 1 趋势图)
    
    Returns:
        dict: {bucket_index: [(datetime, price, iv), ...]}  — 每个桶的时序数据
        list: contract_names — 最新一条快照的合约名列表
    """
    ny_tz = timezone('America/New_York')
    bucket_history = {i: [] for i in range(6)}  # 6 个桶
    contract_names = [''] * 6
    
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        ts_3_days_ago = int((datetime.now(NY_TZ) - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        query = f"""
            SELECT ts, buckets_json FROM option_snapshots_1m
            WHERE symbol = '{symbol}'
            AND ts >= {ts_3_days_ago}
            ORDER BY ts ASC
        """
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        for ts_int, json_val in rows:
            try:
                # json_val may be dict or str in pg JSONB
                snap = json_val if isinstance(json_val, dict) else json.loads(json_val) if isinstance(json_val, str) else json_val
                
                # 兼容新旧格式: list (旧) 或 dict {'buckets':..., 'contracts':...} (新)
                if isinstance(snap, dict):
                    buckets_data = snap.get('buckets', [])
                    cids = snap.get('contracts', [])
                    if cids:
                        contract_names = cids
                elif isinstance(snap, list):
                    buckets_data = snap
                else:
                    continue
                
                dt = datetime.fromtimestamp(ts_int, tz=ny_tz)
                for i, row in enumerate(buckets_data[:6]):
                    if len(row) >= 8:
                        price = float(row[0])
                        iv    = float(row[7])
                        if price > 0:
                            bucket_history[i].append((dt, price, iv))
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error loading option price history from PG: {e}")
        pass
    
    return bucket_history, contract_names

# ================= 3. 业务逻辑与计算 (Logic) =================
def calculate_model_health(df):
    """计算 IC 和 Forward Return (修复冷启动报错)"""
    # 如果数据太少，预先建立空列，防止下游 dropna 找不到列名
    if len(df) < FORWARD_PERIOD + 2: 
        df['fwd_price'] = np.nan
        df['fwd_ret'] = np.nan
        df['rolling_ic'] = np.nan
        return df
        
    df['fwd_price'] = df['p'].shift(-FORWARD_PERIOD)
    df['fwd_ret'] = (df['fwd_price'] - df['p']) / df['p']
    df['rolling_ic'] = df['a'].rolling(window=IC_WINDOW).corr(df['fwd_ret'])
    return df
 
class SystemStatus:
    def __init__(self, r, symbol):
        self.symbol = symbol
        self.r = r
        
        # 1. 基础延迟数据 (用于判断颜色/健康度)
        self.src_lag, self.src_data = get_hash_snapshot(r, symbol)
        self.raw_lag, self.raw_data = get_stream_lag_and_data(r, REDIS_CFG['raw_stream'])
        self.eng_lag, self.eng_data = get_stream_lag_and_data(r, REDIS_CFG['inference_stream'], symbol)
        
        # === [核心修复] 补全特征数据提取逻辑 (供 Tab 3 使用) ===
        self.slow_seq = None
        self.norm_max = 0.0
        
        if self.eng_data:
            # 兼容性处理：有些 payload 包了一层 batch_packet
            batch = self.eng_data.get('batch_packet', self.eng_data)
            
            # 检查当前 symbol 是否在 batch 中
            if batch and 'symbols' in batch and self.symbol in batch['symbols']:
                try:
                    idx = batch['symbols'].index(self.symbol)
                    
                    # 获取特征序列
                    if 'slow_1m' in batch:
                        seq_data = batch['slow_1m'][idx]
                        self.slow_seq = np.array(seq_data) if not isinstance(seq_data, np.ndarray) else seq_data
                    # 兼容旧版本 key
                    elif 'x_stock' in batch and 'x_option' in batch:
                        self.slow_seq = np.concatenate([batch['x_stock'][idx], batch['x_option'][idx]], axis=-1)
                    
                    # 计算最大值 (用于监控归一化是否爆炸)
                    if self.slow_seq is not None and self.slow_seq.size > 0:
                        self.norm_max = np.max(np.abs(self.slow_seq[-1]))
                        
                except Exception as e:
                    # 容错处理，避免拖累主流程
                    print(f"Error parsing engine data: {e}")
        # ========================================================

        # 2. 预热进度数据 (用于显示百分比)
        self.warmup = {
            "IBKR": 0, "Stream": 0, "Engine": 0, "Norm": 0, "Orch": 0
        }
        # [新增] 原始计数 (用于显示 X/Y)
        self.warmup_raw = {
            "Engine": "0/100", "Norm": "0/100", "Orch": "0/30"
        }
        
        # 3. 节点健康状态 (OK/WARN/CRIT)
        self.health = {
            "IBKR": "CRIT", "Stream": "CRIT", "Engine": "CRIT", "Norm": "CRIT", "Orch": "CRIT"
        }
        
        self._analyze_status()
    

    def _calculate_completeness(self, table_name, target_window):
        """
        [🔥 核心升级] 
        通过直接探测 PostgreSQL 真实物理时间戳，计算近期断流程度。
        如果在过去的 target_window 根 K 线中存在大于 1 分钟且小于 10 小时的间隔，
        则判定为日内数据断流，从而极其真实地拉低 Dashboard 上的完整度百分比！
        """
        actual_cnt = 0
        expected_span = target_window
        try:
            conn = psycopg2.connect(PG_DB_URL)
            c = conn.cursor()
            
            # [🔥 升级] 在 DB 层面过滤 RTH (09:30 - 16:00) 盘前盘后数据，和 Engine 的清洗逻辑对齐
            # 这样计算完整度时，就不会被半夜的稀疏长 Gap 误伤！
            if table_name == "market_bars_1m":
                query = f"""
                    SELECT ts FROM {table_name} 
                    WHERE symbol=%s 
                    AND EXTRACT(HOUR FROM timezone('America/New_York', to_timestamp(ts))) * 60 + 
                        EXTRACT(MINUTE FROM timezone('America/New_York', to_timestamp(ts))) BETWEEN 570 AND 960
                    ORDER BY ts DESC LIMIT %s
                """
            else:
                query = f"SELECT ts FROM {table_name} WHERE symbol=%s ORDER BY ts DESC LIMIT %s"
                
            c.execute(query, (self.symbol, target_window))
            q_rows = c.fetchall()
            conn.close()
            rows = [r[0] for r in q_rows]
            
            rows = sorted(list(set(rows)), reverse=True)[:target_window]
            actual_cnt = len(rows)
            
            if actual_cnt >= 2:
                missing_bars = 0
                for i in range(actual_cnt - 1):
                    gap = rows[i] - rows[i+1]
                    # 2小时 (7200秒) 以内的 Gap 视为日内严重断流，超过则视为合理跨日/跨周末的停盘时间
                    if 65 < gap < 7200:
                        missing_bars += int((gap / 60) - 1)
                
                actual_cnt = max(0, actual_cnt - missing_bars)
                expected_span = target_window
        except Exception:
            pass
            
        return actual_cnt, expected_span

    def _analyze_status(self):
        # --- A. IBKR & Stream ---
        self.health["IBKR"] = "OK" if self.src_lag < 3 else ("WARN" if self.src_lag < 10 else "CRIT")
        self.health["Stream"] = "OK" if self.raw_lag < 2 else ("WARN" if self.raw_lag < 5 else "CRIT")
        self.warmup["IBKR"] = 100 if self.src_lag < 60 else 0
        self.warmup["Stream"] = 100 if self.raw_lag < 60 else 0

        # --- B. Engine & Norm (通过 PostgreSQL 强硬审查过去 100 根线的真实紧密度) ---
        VISUAL_TARGET_ENGINE = 100
        eng_actual, eng_expected = self._calculate_completeness("market_bars_1m", VISUAL_TARGET_ENGINE)
        
        self.health["Engine"] = "OK" if self.eng_lag < 3 else ("WARN" if self.eng_lag < 10 else "CRIT")
        self.warmup["Engine"] = min(100, int((eng_actual / max(1, eng_expected)) * 100)) 
        self.warmup_raw["Engine"] = f"{eng_actual}/{eng_expected}"
        
        self.health["Norm"] = self.health["Engine"]
        self.warmup["Norm"] = self.warmup["Engine"]
        self.warmup_raw["Norm"] = self.warmup_raw["Engine"]

        # --- C. Orchestrator (通过 PostgreSQL 审查过去 38 根 alpha_logs 连续性) ---
        VISUAL_TARGET_ORCH = 38
        orch_actual, orch_expected = self._calculate_completeness("alpha_logs", VISUAL_TARGET_ORCH)
        
        self.warmup["Orch"] = min(100, int((orch_actual / max(1, orch_expected)) * 100))
        self.warmup_raw["Orch"] = f"{orch_actual}/{orch_expected}"
        
        if self.health["Engine"] == "OK" and self.warmup["Norm"] > 10:
            self.health["Orch"] = "OK"
        else:
            self.health["Orch"] = "WARN"
            
    def get_node_visuals(self, node):
        """返回 (Color, Label_Text, Pct)"""
        status_val = self.health.get(node, "CRIT")
        pct = self.warmup.get(node, 0)
        
        # 1. 颜色只看延迟/健康度
        color = "#00CC96" if status_val == "OK" else ("#FECB52" if status_val == "WARN" else "#EF553B")
        
        # 2. 文字显示详细信息 (使用 raw counts, with CAP)
        sub = ""
        if node == "IBKR": 
            sub = f"Lag: {self.src_lag:.1f}s"
        elif node == "Stream":
            sub = f"Lag: {self.raw_lag:.1f}s"
        elif node == "Engine":
            # [Cap] Engine/Norm 目标 100
            raw_cnt = 0
            try: raw_cnt = int(self.warmup_raw.get('Engine','0').split('/')[0])
            except: pass
            disp_cnt = min(raw_cnt, 100)
            sub = f"Bar: {disp_cnt}/100"
            
        elif node == "Norm":
            # [Cap]
            raw_cnt = 0
            try: raw_cnt = int(self.warmup_raw.get('Norm','0').split('/')[0])
            except: pass
            disp_cnt = min(raw_cnt, 100)
            sub = f"Norm: {disp_cnt}/100"
            
        elif node == "Orch":
            sub = f"Win: {self.warmup_raw.get('Orch','?')}"
            
        # 进度条符号
        bars = "░▒▓█"
        p_idx = 3 if pct >= 100 else int(pct/33)
        prog_icon = bars[p_idx]
        
        # [新增] 预热来源提示 (Visual Cue)
        # 如果 pct < 100，说明正在逐个积累 (Live Warmup)，显示黄色惊叹号
        # 如果 pct >= 100，说明已通过历史数据到位 (Hot Start) 或累积完成，隐藏惊叹号
        warn_mark = " ⚠️" if (node in ["Engine", "Norm", "Orch"] and pct < 100) else ""
        
        label = f"<b>{node}{warn_mark}</b> {prog_icon}<br>{sub}"
        
        return color, label, pct

# ================= 4. UI 组件 (UI Components) =================

# [Compatibility] Dialog Shim for older Streamlit versions
if hasattr(st, "dialog"):
    _dialog = st.dialog
elif hasattr(st, "experimental_dialog"):
    _dialog = st.experimental_dialog
else:
    # Fallback: Use Expander if dialog not supported
    def _dialog(title, **kwargs):
        def decorator(func):
            def wrapper(*args, **kws):
                with st.expander(f"Pop-up: {title}", expanded=True):
                    func(*args, **kws)
            return wrapper
        return decorator

@_dialog("🔍 预热数据透视 (Warmup Inspector)", width="large")
def show_warmup_inspector(symbol, node_type):
    # 1. 获取 Service 内部状态 (Redis)
    r = get_redis_client()
    service_cnt = 0
    target_cnt = 100 if node_type in ["Engine", "Norm"] else 38
    
    try:
        if node_type in ["Engine", "Norm"]:
            val = r.hget("monitor:warmup:norm", symbol)
            if val: service_cnt = int(val)
        elif node_type == "Orch":
            val = r.hget("monitor:warmup:orch", symbol)
            if val: service_cnt = int(val)
    except: pass

    st.caption(f"正在检查 **{symbol}** - 节点: **{node_type}**")
    
    # 2. 获取 DB 数据
    db_rows = 0
    df = pd.DataFrame()
    cols_show = []
    
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        ts_3_days_ago = int((datetime.now(NY_TZ) - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        if node_type in ["Engine", "Norm"]:
            # [🔥 升级] 过滤盘前盘后数据，只显示 RTH (09:30-16:00) 的 DB 记录
            query = f"""
                SELECT * FROM market_bars_1m 
                WHERE symbol='{symbol}' AND ts >= {ts_3_days_ago} 
                AND EXTRACT(HOUR FROM timezone('America/New_York', to_timestamp(ts))) * 60 + 
                    EXTRACT(MINUTE FROM timezone('America/New_York', to_timestamp(ts))) BETWEEN 570 AND 960
                ORDER BY ts DESC LIMIT 100
            """
            df = pd.read_sql(query, conn)
            cols_show = ['time_str', 'open', 'high', 'low', 'close', 'volume']
        elif node_type == "Orch":
            query = f"SELECT * FROM alpha_logs WHERE symbol='{symbol}' AND ts >= {ts_3_days_ago} ORDER BY ts DESC LIMIT 100"
            df = pd.read_sql(query, conn)
            cols_show = ['time_str', 'alpha', 'iv', 'price', 'vol_z']
        conn.close()
        db_rows = len(df)
    except Exception as e:
        # Ignore if table not yet fully populated
        pass

    # 3. 对比展示 (中文)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🧠 服务内存缓存 (State)", f"{service_cnt} 条", help="Service 内部计数。如果大于 0，说明服务已从磁盘恢复了历史状态 (Hot Start)。")
    with c2:
        st.metric("💾 数据库近期记录 (DB)", f"{db_rows} 条", help="近期写入 PostgreSQL 的条数。")
        
    if service_cnt > db_rows + 10:
        st.info(f"ℹ️ **状态说明**: 服务内存中有 **{service_cnt}** 条数据 (从历史状态恢复)，但数据库近期只有 **{db_rows}** 条。这是正常的 **热启动 (Hot Start)** 现象，保证了指标计算不会中断。")

    if not df.empty:
        df['time_str'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.strftime('%H:%M:%S')
        final_cols = [c for c in cols_show if c in df.columns]
        st.dataframe(df[final_cols], use_container_width=True, height=300)
    else:
        st.warning("该股票在数据库中暂无记录。")

def draw_topology(status):
    # 节点坐标
    nodes_layout = {
        "IBKR": (0, 1), 
        "Stream": (1, 1), 
        "Engine": (2, 1), 
        "Norm": (3, 1), 
        "Orch": (4, 1)
    }
    
    edges = [("IBKR", "Stream"), ("Stream", "Engine"), ("Engine", "Norm"), ("Norm", "Orch")]
    
    x_nodes, y_nodes = [], []
    colors, texts = [], []
    sizes = []
    
    for name, (x, y) in nodes_layout.items():
        x_nodes.append(x)
        y_nodes.append(y)
        
        color, label, pct = status.get_node_visuals(name)
        colors.append(color)
        texts.append(label)
        
        # 节点大小：预热完成后稍微变大一点，或者保持一致
        sizes.append(50)

    fig = go.Figure()

    # 1. 连线
    for u, v in edges:
        u_pos, v_pos = nodes_layout[u], nodes_layout[v]
        fig.add_trace(go.Scatter(
            x=[u_pos[0], v_pos[0], None], 
            y=[u_pos[1], v_pos[1], None], 
            mode='lines', 
            line=dict(width=3, color='#555'), 
            hoverinfo='none'
        ))

    # 2. 节点 (外圈光晕表示预热进度)
    # 如果预热未完成 (pct < 100)，加一个灰色/黄色的外圈？
    # 这里简单处理：直接画节点
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=sizes, 
            color=colors, 
            line=dict(width=2, color='white')
        ),
        text=texts,
        textposition="bottom center", # 文字在下方，不遮挡球体
        textfont=dict(color='white', size=11),
        hoverinfo='text'
    ))
    
    # 3. 在节点中心显示百分比数值
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='text',
        text=[f"{status.warmup[n]}%" for n in nodes_layout],
        textposition="middle center",
        textfont=dict(color='black', size=10, family="Arial Black"), # 黑色字体在亮色球体上
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': f"📡 System Latency & Warmup: {status.symbol}",
            'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=16, color='#AAA')
        },
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 4.5]),
        yaxis=dict(visible=False, range=[0.5, 1.5]),
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def render_debug_inspector():
    """读取每日统一 DB 并可视化 debug_slow/fast 原始特征"""
    st.markdown("## 🐞 特征引擎透视 (Feature Engine Inspector)")
    
    ny_tz = timezone('America/New_York')
    today_str = datetime.now(ny_tz).strftime('%Y%m%d')
    
    # 1. 布局
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        source_opt = ["Slow (debug_slow)", "Fast (debug_fast)"]
        table_source = st.selectbox("📡 选择数据通道", source_opt)
        
        if "Slow" in table_source:
            base_table_name = "debug_slow"
        else:
            base_table_name = "debug_fast"
            
        table_name = f"{base_table_name}_{today_str}"
    
    try:
        conn = psycopg2.connect(PG_DB_URL)
        # 检查表存在 PostgreSQL 中
        cursor = conn.cursor()
        cursor.execute("SELECT to_regclass(%s)", (table_name,))
        if not cursor.fetchone()[0]:
            st.warning(f"表 `{table_name}` 尚不存在。请确保特征引擎已启动并产生数据。")
            if st.button("🔄 刷新状态"): st.rerun()
            conn.close()
            return
        
        # 获取所有 Symbol
        symbols_df = pd.read_sql(f"SELECT DISTINCT symbol FROM {table_name}", conn)
        if symbols_df.empty:
            st.warning("数据库中暂无数据。")
            conn.close()
            return
            
        all_symbols = symbols_df['symbol'].tolist()
        
        with col2:
            selected_sym = st.selectbox("🔍 选择股票", all_symbols, index=0)
            
        with col3:
            limit = st.slider("显示最近行数", 10, 500, 50)

        # 2. 查询数据
        if selected_sym:
            query = f"SELECT * FROM {table_name} WHERE symbol = '{selected_sym}' ORDER BY ts DESC LIMIT {limit}"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                st.info("该股票暂无特征数据。")
                return

            # 转换时间
            df['time_str'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.strftime('%H:%M:%S')
            
            # 调整列顺序
            cols = ['time_str', 'created_at'] + [c for c in df.columns if c not in ['time_str', 'created_at', 'ts', 'symbol']]
            df_display = df[cols].copy()

            st.markdown(f"### 📊 {selected_sym} Raw Features ({table_name})")
            
            # 动态检查全0或NaN
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            zero_cols = [c for c in numeric_cols if (df_display[c] == 0).all()]
            nan_cols = [c for c in numeric_cols if df_display[c].isna().all()]
            
            if zero_cols:
                st.error(f"⚠️ **全0特征警告 (可能计算失效):** {', '.join(zero_cols)}")
            if nan_cols:
                st.error(f"⚠️ **全NaN特征警告:** {', '.join(nan_cols)}")
            
            # 数据表 (带热力图)
            st.dataframe(
                df_display.style.background_gradient(axis=0, cmap='RdYlGn', subset=numeric_cols),
                use_container_width=True,
                height=300
            )
            
            # 趋势图
            st.markdown("### 📈 特征趋势追踪")
            default_cols = [c for c in numeric_cols if c not in zero_cols][:5]
            selected_features = st.multiselect("选择特征绘图", numeric_cols.tolist(), default=default_cols)
            
            if selected_features:
                chart_data = df.set_index('time_str')[selected_features]
                st.line_chart(chart_data.iloc[::-1]) # 时间正序

    except Exception as e:
        st.error(f"读取数据库出错: {e}")

# ================= 5. 主执行逻辑 (Script Execution) =================

# 初始化 Redis
r = get_redis_client()
if not r:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("🎮 Controller")

    # [New] 显示当前环境 (Paper / Real)
    env_color = "red" if IBKR_PORT == 4001 else "green"
    env_text = "REAL MONEY" if IBKR_PORT == 4001 else "PAPER / SIM"
    st.markdown(f"**ENV**: <span style='color:{env_color}'>**{env_text}**</span> (Port {IBKR_PORT})", unsafe_allow_html=True)
    
    # [NEW] 实时拉取 & 显示账户总资产
    try:
        raw_acct = r.hget('live_account_info', 'balance')
        from config import get_synced_funds
        
        if raw_acct:
            acct_data = ser.unpack(raw_acct)
            real_net_liq = acct_data.get('net_liquidation', 0.0)
            real_avail = acct_data.get('available_funds', 0.0)
        else:
            real_net_liq = 0.0
            real_avail = 0.0

        net_liq = get_synced_funds(real_net_liq)
        avail = get_synced_funds(real_avail)
        
        if net_liq > 0:
            st.markdown(f"""
            <div style='background-color: {card_bg}; padding: 10px; border-radius: 5px; margin-top: 10px; margin-bottom: 10px;'>
                <div style='font-size: 13px; color: gray;'>Net Liquidation</div>
                <div style='font-size: 20px; font-weight: bold; color: #00CC96;'>${net_liq:,.2f}</div>
                <div style='font-size: 13px; color: gray; margin-top: 8px;'>Available Funds</div>
                <div style='font-size: 16px; font-weight: bold; color: #3498db;'>${avail:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Waiting for Account Data...")
            
    except Exception as e:
        st.warning("Account Data Offline")
        
    st.markdown(f"**Redis DB**: {REDIS_CFG['db']}")
    st.markdown("---")
  

    symbols =  [
     
           # --- Tier 1: 巨无霸 ---
    'NVDA', 'AAPL', 'META', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 'MSTR', 'COIN',
    # --- Tier 2: 核心蓝筹 ---
    'NFLX', 'CRWV', 'AVGO', 'MSFT', 'HOOD', 'MU', 'APP', 'GOOGL', 'GS',  'WMT',
    # --- Tier 3: 高流动性 --- 
    'SMCI', 'ADBE', 'CRM', 'ORCL', 'NKE', 'XOM', 'INTC', 'DELL', 'IWM', 'GLD'
    # --- Indices & Macro ---
        ]
    
    symbol = st.selectbox("Symbol", symbols, index=1)
    auto_refresh = st.checkbox("Auto Refresh (1s)", True)
    if st.button("Manual Refresh"): st.rerun()
    st.divider()
    st.markdown("### Model Health Guide")
    st.info("IC > 0.05: Healthy\nIC < 0: Inverted\nIC ~ 0: Decay")

# --- Top: Topology ---
status = SystemStatus(r, symbol)
st.plotly_chart(draw_topology(status), use_container_width=True)

# [New] Interactive Warmup Buttons
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
if c2.button(f"🔍 Engine: {status.warmup['Engine']}%", help="Click to view Market Bars"):
    show_warmup_inspector(symbol, "Engine")
if c3.button(f"🔍 Norm: {status.warmup['Norm']}%", help="Click to view Market Bars"):
    show_warmup_inspector(symbol, "Norm")
if c4.button(f"🔍 Orch: {status.warmup['Orch']}%", help="Click to view Alpha Logs"):
    show_warmup_inspector(symbol, "Orch")

st.divider()

# --- Main: Tabs ---
tab1, tab10, tab2, tab11, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Market Feed", 
    "🏦 Positions",
    "📜 History & Replay", 
    "Live Features",
    "🧠 Feature Heatmap", 
    "🏥 Model Health (IC)", 
    "📝 Trade Log",
    "🐞 Feature Debug",
    "💾 Warmup Audit" ,
    "🧪 Norm Health",
    "🔄 Live -> Backtest"
])

# === Tab 1: 市场透视 ===
with tab1:
    st.subheader("Option Chain Matrix")
    if status.src_data:
        # [Fix 4] 读取 contracts 字段 (来自 ibkr_connector 新增)
        contracts = status.src_data.get('contracts', [])
        df_opt = format_option_matrix(status.src_data.get('buckets', []), contracts)
        # 数值列格式化, Contract 列不格式化
        numeric_cols = ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Strike', 'Vol(Cum)', 'IV']
        st.dataframe(
            df_opt.style.format({c: "{:.3f}" for c in numeric_cols})
            .background_gradient(subset=['Vol(Cum)'], cmap='Greens')
            .background_gradient(subset=['IV'], cmap='Purples'),
            use_container_width=True
        )
    else:
        st.info("Waiting for Snapshot...")

    # --- [Fix 4] 期权价格走势 + 交易标记 ---
    st.divider()
    
    # 加载历史快照
    bucket_history, hist_contracts = load_option_price_history(symbol)
    bucket_labels = ['Front P_ATM', 'Front P_OTM', 'Front C_ATM', 'Front C_OTM', 'Next P_ATM', 'Next C_ATM']
    
    # [Fix] Dropdown Selection
    # 构造选项列表: "All", "Contract 1", "Contract 2"...
    valid_contracts = [c for c in hist_contracts if c]
    if not valid_contracts: valid_contracts = bucket_labels
    options = ["ALL"] + valid_contracts
    
    c_sel1, c_sel2 = st.columns([1, 3])
    with c_sel1:
        st.subheader(f"📉 Option Trends")
    with c_sel2:
        selected_contract = st.selectbox("Select Filter", options, index=0, label_visibility="collapsed")

    colors = ['#EF553B', '#FF6692', '#00CC96', '#19D3F3', '#AB63FA', '#FFA15A']
    
    has_data = any(len(pts) > 0 for pts in bucket_history.values())
    
    if has_data:
        fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 绘制每个桶的价格曲线
        for i in range(6):
            pts = bucket_history[i]
            if not pts: continue
            
            # Label Decision
            label = hist_contracts[i] if (i < len(hist_contracts) and hist_contracts[i]) else bucket_labels[i]
            
            # [Filter Logic]
            if selected_contract != "ALL" and label != selected_contract:
                continue
                
            dts, prices, ivs = zip(*pts)
            
            fig_opt.add_trace(
                go.Scatter(x=list(dts), y=list(prices), mode='lines',
                           name=f"💲 {label}", line=dict(color=colors[i], width=2)),
                secondary_y=False
            )
            fig_opt.add_trace(
                go.Scatter(x=list(dts), y=list(ivs), mode='lines',
                           name=f"IV {label}", line=dict(color=colors[i], width=1, dash='dot'),
                           opacity=0.4, showlegend=False),
                secondary_y=True
            )
        
        # 叠加交易标记 (从 Redis trade_log)
        # [Fix] 暂时隐藏交易打点，避免显示回测数据
        # df_opt_trades = load_trade_logs_for_chart(r, symbol)
        df_opt_trades = pd.DataFrame()
        # [Fix] 暂时隐藏交易打点
        # if not df_opt_trades.empty:
        #    pass
        
        fig_opt.update_layout(
            height=480,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, font=dict(size=10)),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            template=PLOTLY_THEME
        )
        fig_opt.update_yaxes(title_text="Option Price ($)", secondary_y=False, showgrid=True)
        fig_opt.update_yaxes(title_text="IV", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_opt, use_container_width=True)
        
        # 快捷统计
        total_points = sum(len(pts) for pts in bucket_history.values())
        active_buckets = sum(1 for pts in bucket_history.values() if len(pts) > 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Active Contracts", f"{active_buckets}/6")
        c2.metric("Data Points", total_points)
        c3.metric("Trade Overlays", len(df_opt_trades) if not df_opt_trades.empty else 0)
    else:
        st.info("⏳ No option price history yet — data accumulates after market open.")

# === [Tab 10] 🏦 Positions & Orders ===
with tab10:
    st.header("🏦 Active Positions & IBKR Account")
    st.markdown("Retrieve live positions directly from IBKR Gateway and force close positions if needed.")
    
    # helper for async ib connection
    def fetch_ibkr_positions():
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        import ib_insync
        ib = ib_insync.IB()
        try:
            # Connect with a random client_id to avoid conflicts
            ib.connect('127.0.0.1', IBKR_PORT, clientId=111)
            portfolio = ib.portfolio()
            account_values = ib.accountValues()
            ib.disconnect()
            return portfolio, account_values
        except Exception as e:
            return None, str(e)
            
    def close_ibkr_position(contract, qty, action='SELL'):
        import asyncio
        import time # Added import for time
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        import ib_insync
        ib = ib_insync.IB()
        try:
            ib.connect('127.0.0.1', IBKR_PORT, clientId=112)
            # Qualify the contract
            ib.qualifyContracts(contract)
            order = ib_insync.MarketOrder(action, abs(qty))
            trade = ib.placeOrder(contract, order)
            ib.sleep(2) # Give it 2 seconds to execute
            ib.disconnect()
            return True, "Order Placed"
        except Exception as e:
            return False, str(e)

    if st.button("🔄 Refresh Positions"):
        with st.spinner("Connecting to IBKR..."):
            portfolio, account_vals = fetch_ibkr_positions()
            
            if isinstance(account_vals, str):
                st.error(f"Failed to connect to IBKR: {account_vals}")
            else:
                # 1. Display Account Summary
                st.subheader("Account Summary")
                net_liq = "N/A"
                av_funds = "N/A"
                for val in account_vals:
                    if val.tag == 'NetLiquidation' and val.currency == 'USD':
                        net_liq = f"${float(val.value):,.2f}"
                    if val.tag == 'AvailableFunds' and val.currency == 'USD':
                        av_funds = f"${float(val.value):,.2f}"
                
                c1, c2 = st.columns(2)
                c1.metric("Net Liquidation", net_liq)
                c2.metric("Available Funds", av_funds)
                
                # 2. Display Positions
                st.subheader("Active Positions")
                if not portfolio:
                    st.info("No active positions found in IBKR account.")
                else:
                    pos_data = []
                    for item in portfolio:
                        # Extract basic info from PortfolioItem
                        p_contract = item.contract
                        sym = p_contract.symbol
                        sec_type = p_contract.secType
                        qty = item.position
                        avg_cost = item.averageCost
                        mkt_price = item.marketPrice
                        mkt_val = item.marketValue
                        unrealized_pnl = item.unrealizedPNL
                        
                        desc = f"{sym} {sec_type}"
                        if sec_type == 'OPT':
                            desc = f"{sym} {p_contract.lastTradeDateOrContractMonth} {p_contract.strike} {p_contract.right}"
                            
                        # Use dict.get or ternary for safety in case some fields are None
                        pos_data.append({
                            "Account": item.account,
                            "Contract": desc,
                            "Symbol": sym,
                            "SecType": sec_type,
                            "Qty": qty,
                            "Avg Cost": f"${avg_cost:.2f}" if avg_cost is not None else "N/A",
                            "Mkt Price": f"${mkt_price:.2f}" if mkt_price is not None else "N/A",
                            "Mkt Value": f"${mkt_val:.2f}" if mkt_val is not None else "N/A",
                            "Unrealized PnL": f"${unrealized_pnl:.2f}" if unrealized_pnl is not None else "N/A",
                            "_RawContract": p_contract # Store raw contract for close button
                        })
                    
                    df_pos = pd.DataFrame(pos_data)
                    st.dataframe(df_pos.drop(columns=['_RawContract']), use_container_width=True)
                    
                    # 3. Close Buttons
                    st.subheader("Danger Zone: Close Positions")
                    st.warning("Clicking these buttons will send a MARKET order directly to IBKR. Use with caution.")
                    
                    for idx, row in df_pos.iterrows():
                        col1, col2 = st.columns([3, 1])
                        col1.write(f"**{row['Contract']}** (Qty: {row['Qty']})")
                        
                        action_str = "SELL" if row['Qty'] > 0 else "BUY" # Cover short positions
                        button_key = f"close_pos_{idx}_{row['Symbol']}"
                        
                        if col2.button(f"Close Position", key=button_key, type="secondary"):
                            with st.spinner(f"Sending {action_str} MKT order for {row['Contract']}..."):
                                success, msg = close_ibkr_position(row['_RawContract'], row['Qty'], action=action_str)
                                if success:
                                    st.success(f"Order sent! Refresh to verify.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to send order: {msg}")

# === Tab 2: 历史回溯 ===
with tab2:
    st.subheader(f"🕰️ Intraday Replay: {symbol}")
    
    # [Fix] 自动寻找最近的 DB 文件 (如果当日没数据)
    df_price = load_today_sqlite_data(symbol)
    # Fallback Logic is now handled by load_today_sqlite_data (which fetches 3 days of history from PG)
    if df_price.empty:
        st.warning(f"No historical data found for {symbol} (Checked {DB_DIR})")
    df_alpha = fetch_model_logs(r, symbol)
    df_trades = load_trade_logs_for_chart(r, symbol)
    
    if df_price.empty and df_alpha.empty:
        st.warning(f"No historical data found for {symbol} today.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace A: 股票价格 (左轴)
        if not df_price.empty:
            fig.add_trace(go.Scatter(x=df_price['datetime'], y=df_price['price'], mode='lines', name='Price (DB)', line=dict(color='#1f77b4', width=2)), secondary_y=False)
        elif not df_alpha.empty:
            fig.add_trace(go.Scatter(x=df_alpha['dt'], y=df_alpha['p'], mode='lines', name='Price (Redis)', line=dict(color='#1f77b4', width=2)), secondary_y=False)

        # Trace B: Alpha 信号 (右轴)
        if not df_alpha.empty:
            fig.add_trace(go.Scatter(x=df_alpha['dt'], y=df_alpha['a'], mode='lines', name='Alpha', line=dict(color='#ff7f0e', width=1, dash='dot'), opacity=0.7), secondary_y=True)

        # Trace C: 交易标记
        # [Fix] 恢复交易打点
        # df_trades = pd.DataFrame() # REMOVED: This was overwriting the data loaded at line 946
        
        if not df_trades.empty:
            buys = df_trades[df_trades['action'] == 'OPEN']
            sells = df_trades[df_trades['action'] == 'CLOSE']
            
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['datetime'], y=buys['price'], 
                    mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)
                ), secondary_y=False)
                
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['datetime'], y=sells['price'], 
                    mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)
                ), secondary_y=False)

        fig.update_layout(
            height=500, 
            hovermode="x unified", 
            legend=dict(orientation="h", y=1.02), 
            margin=dict(l=10, r=10, t=10, b=10),
            template=PLOTLY_THEME
        )
        fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True, gridcolor='#333')
        fig.update_yaxes(title_text="Alpha", secondary_y=True, showgrid=False, range=[-3, 3])
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("DuckDB Bars", len(df_price))
        c2.metric("Alpha Ticks", len(df_alpha))
        c3.metric("Trades", len(df_trades))

# === Tab 3: 特征热图 ===
with tab3:
    if status.slow_seq is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Feature Z-Score History (30 steps)")
            fig_heat = go.Figure(data=go.Heatmap(
                z=status.slow_seq.T,
                x=[f"T-{i}" for i in range(29, -1, -1)],
                y=[f"F{i}" for i in range(status.slow_seq.shape[1])],
                colorscale='RdBu_r', zmid=0, zmin=-4, zmax=4
            ))
            fig_heat.update_layout(height=400, margin=dict(t=0,b=0), template=PLOTLY_THEME)
            st.plotly_chart(fig_heat, use_container_width=True)
        with c2:
            st.subheader("Stats")
            st.metric("Max Z", f"{status.norm_max:.2f}", delta="Exploded" if status.norm_max>5 else "OK", delta_color="inverse")
            st.metric("Energy", f"{np.sum(np.abs(status.slow_seq[-1])):.1f}")
    else:
        st.info("Waiting for Feature Engine...")

# === Tab 4: 模型健康度 ===
with tab4:
    st.subheader(f"🩺 Model Health Check: {symbol}")
    df_log = fetch_model_logs(r, symbol)
    if not df_log.empty:
        df_health = calculate_model_health(df_log)
        df_valid = df_health.dropna(subset=['fwd_ret'])
        if not df_valid.empty:
            latest = df_valid.iloc[-1]
            last_ic = latest['rolling_ic']
            avg_ic = df_valid['rolling_ic'].mean()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Alpha", f"{latest['a']:.3f}")
            k2.metric("Forward Return (5m)", f"{latest['fwd_ret']:.2%}")
            ic_state = "normal" if last_ic > 0 else "inverse"
            k3.metric("Rolling IC (30m)", f"{last_ic:.3f}", delta=f"{last_ic - avg_ic:.3f}", delta_color=ic_state)
            status_msg = "✅ Healthy"
            if last_ic < -0.1: status_msg = "🚨 INVERTED"
            elif last_ic < 0.01: status_msg = "⚠️ DECAY"
            k4.metric("Status", status_msg)
            st.divider()
            c_left, c_right = st.columns(2)
            with c_left:
                st.caption("IC Trend")
                fig_ic = go.Figure()
                fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_ic.add_trace(go.Scatter(x=df_valid['dt'], y=df_valid['rolling_ic'], mode='lines', line=dict(color='#FF4B4B', width=2)))
                fig_ic.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), template=PLOTLY_THEME)
                st.plotly_chart(fig_ic, use_container_width=True)
            with c_right:
                st.caption("Alpha vs Return Scatter")
                fig_scat = px.scatter(df_valid, x='a', y='fwd_ret', color='rolling_ic', color_continuous_scale='RdBu')
                fig_scat.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), template=PLOTLY_THEME)
                st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Gathering data for IC calculation...")
    else:
        st.warning(f"No Alpha Logs found for {symbol}.")

# === Tab 5: 交易日志 ===
with tab5:
    st.header("📊 Trade Records & Performance (PostgreSQL & Live)")
    try:
        conn = psycopg2.connect(PG_DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT to_regclass('trade_logs')")
        if not cursor.fetchone()[0]:
            st.info("No 'trade_logs' table in PostgreSQL yet.")
            conn.close()
        else:
            # ✅ 0. 日期过滤选择
            c_date1, c_date2 = st.columns([1, 3])
            with c_date1:
                query_date = st.date_input("Inquiry Date", value=datetime.now(NY_TZ).date())
            
            # 计算查询日期的起止时间戳
            target_start_dt = datetime.combine(query_date, dt_time(0, 0, 0))
            target_start_ts = int(NY_TZ.localize(target_start_dt).timestamp())
            target_end_ts = target_start_ts + 86400
            
            # [🔥 模式切换] 根据全局开关切换查询目标表，确保空跑和实盘流水不混淆
            from config import TRADING_ENABLED
            target_table = 'trade_logs' if TRADING_ENABLED else 'trade_logs_backtest'
            
            sql = f"SELECT * FROM {target_table} WHERE ts >= {target_start_ts} AND ts < {target_end_ts} ORDER BY ts ASC"
            df_all = pd.read_sql(sql, conn)
            conn.close()
            
            if df_all.empty:
                st.info(f"No trades executed on {query_date}.")
            else:
                # ✅ 1. 解析 log_json 展开关键字段
                def parse_val(row, key, default=np.nan):
                    try:
                        d = json.loads(row)
                        return d.get(key, default)
                    except: return default

                for k in ['pnl', 'roi', 'stock_price', 'entry_stock', 'mode', 'account_cash']:
                    df_all[k] = df_all['details_json'].apply(lambda r: parse_val(r, k))

                # 提取 strategy_note 里的子字段 (alpha, reason, tag)
                def extract_note(row, key):
                    try:
                        d = json.loads(row)
                        note_str = d.get('strategy_note', '{}')
                        note = json.loads(note_str)
                        return note.get(key, '')
                    except: return ''
                    
                df_all['alpha'] = df_all['details_json'].apply(lambda r: extract_note(r, 'alpha'))
                df_all['reason'] = df_all['details_json'].apply(lambda r: extract_note(r, 'reason'))
                df_all['tag'] = df_all['details_json'].apply(lambda r: extract_note(r, 'tag'))

                # ✅ 2. 遍历计算已平仓盈亏 & 找出当前未平仓头寸
                open_positions = {}
                closed_pnl = 0.0
                wins, losses = 0, 0
                
                for _, row in df_all.iterrows():
                    sym = row['symbol']
                    qty = float(row.get('qty', 0))
                    price = float(row.get('price', 0))
                    action = row.get('action', '')
                    tag = row.get('tag', '')
                    
                    if action == 'OPEN':
                        if sym not in open_positions:
                            open_positions[sym] = {'qty': 0, 'cost': 0.0, 'tag': tag}
                        old_qty = open_positions[sym]['qty']
                        old_cost = open_positions[sym]['cost']
                        new_qty = old_qty + qty
                        open_positions[sym]['cost'] = (old_qty * old_cost + qty * price) / new_qty if new_qty > 0 else 0
                        open_positions[sym]['qty'] = new_qty
                        if tag: open_positions[sym]['tag'] = tag
                        
                    elif action == 'CLOSE':
                        if sym in open_positions:
                            open_positions[sym]['qty'] -= qty
                            if open_positions[sym]['qty'] <= 0:
                                del open_positions[sym]
                                
                        pnl_val = float(row.get('pnl', 0) if pd.notna(row.get('pnl')) else 0)
                        closed_pnl += pnl_val
                        if pnl_val > 0: wins += 1
                        elif pnl_val < 0: losses += 1

                # ✅ 3+4. Live Price 流式刷新卡片 (独立每 3s 自动更新，不需全页刷新)
                @st.fragment(run_every=3)
                def _live_price_card(open_positions, closed_pnl, wins, losses):
                    from config import TAG_TO_INDEX
                    unrealized_pnl = 0.0
                    total_market_value = 0.0
                    # [新增] 优先从数据库读取最新现金，无果则从流水尝试
                    latest_cash = fetch_latest_mock_cash()
                    paper_details = []

                    if open_positions:
                        try:
                            r_live = redis.Redis(**{k:v for k,v in REDIS_CFG.items() if k in ['host','port','db']}, decode_responses=False)
                        except: r_live = None

                        for sym, pos in open_positions.items():
                            live_price = pos['cost']
                            tag = pos['tag']
                            found_live = False

                            if r_live:
                                try:
                                    # [修复] HASH_KEY_SNAPSHOT 未定义，应为 HASH_OPTION_SNAPSHOT
                                    raw = r_live.hget(HASH_OPTION_SNAPSHOT, sym)
                                    if raw:
                                        snap = ser.unpack(raw)
                                        idx = TAG_TO_INDEX.get(tag, -1)
                                        if idx != -1 and len(snap['buckets']) > idx:
                                            bucket = snap['buckets'][idx]
                                            _last = float(bucket[0])
                                            _bid  = float(bucket[8]) if len(bucket) > 8 else 0.0
                                            _ask  = float(bucket[9]) if len(bucket) > 9 else 0.0
                                            if _bid > 0.01 and _ask > 0.01:
                                                live_price = (_bid + _ask) / 2.0
                                            else:
                                                live_price = _last
                                            found_live = True
                                except: pass

                            if not found_live:
                                try:
                                    pg_conn = psycopg2.connect(PG_DB_URL)
                                    c_snap = pg_conn.cursor()
                                    c_snap.execute("SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1", (sym,))
                                    row_snap = c_snap.fetchone()
                                    if row_snap:
                                        json_val = row_snap[0]
                                        snap = json_val if isinstance(json_val, dict) else json.loads(json_val) if isinstance(json_val, str) else json_val
                                        buckets = snap.get('buckets', snap) if isinstance(snap, dict) else snap
                                        idx = TAG_TO_INDEX.get(tag, -1)
                                        if idx != -1 and len(buckets) > idx:
                                            bucket = buckets[idx]
                                            _last = float(bucket[0])
                                            _bid  = float(bucket[8]) if len(bucket) > 8 else 0.0
                                            _ask  = float(bucket[9]) if len(bucket) > 9 else 0.0
                                            if _bid > 0.01 and _ask > 0.01:
                                                live_price = (_bid + _ask) / 2.0
                                            else:
                                                live_price = _last
                                    pg_conn.close()
                                except: pass

                            if pos['cost'] > 0:
                                price_deviation = abs(live_price - pos['cost']) / pos['cost']
                                if price_deviation > 5.0:
                                    st.warning(f"⚠️ [{sym}] live_price ({live_price:.2f}) 与开仓成本 ({pos['cost']:.2f}) 偏差异常 ({price_deviation:.0%})，痑似 bucket 数据污染，已回退到成本价")
                                    live_price = pos['cost']

                            paper_pnl_sym = (live_price - pos['cost']) * pos['qty'] * 100
                            unrealized_pnl += paper_pnl_sym
                            market_value_sym = live_price * pos['qty'] * 100
                            total_market_value += market_value_sym
                            
                            roi_pct = ((live_price / pos['cost']) - 1) * 100 if pos['cost'] > 0 else 0
                            pos['paper_pnl'] = paper_pnl_sym
                            pos['roi_pct'] = roi_pct
                            paper_details.append({
                                'Symbol': sym, 'Tag': tag, 'Qty': pos['qty'],
                                'Cost': round(pos['cost'], 2), 'Live 🔴': round(live_price, 2),
                                'MarketValue ($)': round(market_value_sym, 2),
                                'Paper PnL ($)': round(paper_pnl_sym, 2), 'ROI (%)': round(roi_pct, 2)
                            })

                    # 大屏指标
                    st.write("##### 📈 Intraday Performance")
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Realized PnL (Closed)", f"${closed_pnl:.2f}")
                    c2.metric("Unrealized PnL (Open)", f"${unrealized_pnl:.2f}")
                    c3.metric("Total Market Value", f"${total_market_value:,.0f}")
                    
                    # 显示剩余现金
                    display_cash = latest_cash if latest_cash is not None else (df_all['account_cash'].iloc[-1] if not df_all.empty else 0.0)
                    c4.metric("Remaining Cash", f"${display_cash:,.2f}")
                    
                    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                    c5.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}W / {losses}L")
                    c6.metric("Open Positions", f"{len(open_positions)} Symbols")

                    if paper_details:
                        st.dataframe(
                            pd.DataFrame(paper_details).style.applymap(
                                lambda x: 'color: #00CC96' if x > 0 else ('color: #EF553B' if x < 0 else ''),
                                subset=['Paper PnL ($)', 'ROI (%)']
                            ),
                            use_container_width=True
                        )

                _live_price_card(open_positions, closed_pnl, wins, losses)
                st.divider()

                # ✅ 5. 渲染最新 100 条流水日志
                st.write("##### 🕒 Recent Trade Logs")
                df_trade = df_all.sort_values(by='ts', ascending=False).head(100).copy()
                df_trade['time'] = pd.to_datetime(df_trade['ts'], unit='s', utc=True).dt.tz_convert(NY_TZ).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # [New] 把上方算好的纸面盈亏同步赋值给依然 OPEN 的历史日志
                for idx, row in df_trade.iterrows():
                    if row.get('action') == 'OPEN' and row.get('symbol') in open_positions:
                        sym = row['symbol']
                        df_trade.at[idx, 'pnl'] = open_positions[sym].get('paper_pnl', 0.0)
                        df_trade.at[idx, 'roi'] = open_positions[sym].get('roi_pct', 0.0) / 100.0 # 格式化要求小数
                        
                display_cols = ['time', 'symbol', 'action', 'qty', 'fill_duration', 'fill_ratio', 'account_cash', 'price', 'pnl', 'roi', 'stock_price', 'entry_stock', 'mode', 'alpha', 'reason']
                final_cols = [c for c in display_cols if c in df_trade.columns]
                
                st.dataframe(
                    df_trade[final_cols].style.applymap(
                        lambda x: 'color: #00CC96' if x == 'OPEN' else ('color: #EF553B' if x == 'CLOSE' else ''), subset=['action']
                    ).applymap(
                        lambda x: 'color: #00CC96' if pd.notna(x) and x > 0 else ('color: #EF553B' if pd.notna(x) and x < 0 else ''),
                        subset=['pnl', 'roi'] if 'pnl' in final_cols and 'roi' in final_cols else []
                    ),
                    use_container_width=True
                )

                # ✅ [New] 渲染 "缺失交易" (有信号但没成交)
                st.divider()
                st.write("##### 🚫 Missing Trades (High Alpha signals with no OPEN)")
                df_miss = fetch_missing_trades()
                if not df_miss.empty:
                    df_miss_disp = df_miss.copy()
                    df_miss_disp['time'] = pd.to_datetime(df_miss_disp['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert(NY_TZ).dt.strftime('%H:%M:%S')
                    st.dataframe(
                        df_miss_disp[['time', 'symbol', 'alpha', 'vol_z', 'price', 'iv']].style.applymap(
                            lambda x: 'color: #00CC96' if x > 0 else ('color: #EF553B' if x < 0 else ''), subset=['alpha']
                        ),
                        use_container_width=True
                    )
                else:
                    st.info("No missing trades detected today.")
    except Exception as e:
        st.error(f"Trade Log Error: {e}", icon="🚨")

# === Tab 6: 调试透视 (View Only) ===
with tab6:
    render_debug_inspector()
    st.info("ℹ️ Debug Data is Read-Only.")

# === [Modified] Tab 7: 数据管理 (View Only) ===
with tab7:
    st.header("💾 Data Inspector (Warmup & History)")
    st.markdown("""
    **功能说明**: 查看 PostgreSQL 历史数据。
    - **Market Bars**: 1分钟 K线数据 (影响特征计算)。
    - **Alpha Logs**: 模型推理日志 (用于归因分析)。
    """)
    
    # 1. 基础健康检查
    db_status = get_pg_health_status()
    if not db_status['exists']:
        st.error(f"Database not found: {db_status['path']}")
    else:
        st.success(f"Connected to DB: {db_status['path']} ({db_status['size_mb']} MB)")
        st.divider()
        
        # 2. 数据类型选择
        data_type = st.radio("Select Data Context", ["Market Bars (1min)", "Alpha Logs"], horizontal=True)
        
        # 3. 数据查看
        all_symbols = list(db_status['stocks'].keys())
        if data_type == "Alpha Logs":
            try:
                conn = psycopg2.connect(PG_DB_URL)
                sym_df = pd.read_sql("SELECT DISTINCT symbol FROM alpha_logs", conn)
                conn.close()
                if not sym_df.empty: all_symbols = sym_df['symbol'].tolist()
            except: pass

        if not all_symbols:
            st.info("No symbols found in current DB.")
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                target_sym = st.selectbox("Select Symbol", all_symbols)
            
            with c2:
                if data_type == "Market Bars (1min)":
                    info = db_status['stocks'].get(target_sym, {})
                    if info:
                        st.caption(f"Count: {info.get('count','N/A')} | Lag: {info.get('lag_sec','N/A')}s | Range: {info.get('first_time','?')} - {info.get('last_time','?')}")
                else:
                    st.caption(f"Viewing Alpha Logs for {target_sym}")
                
            # 4. 数据预览 (Dataframe)
            st.markdown(f"### 📋 {target_sym} {data_type} Preview (Last 100)")
            try:
                conn = psycopg2.connect(PG_DB_URL)
                if data_type == "Market Bars (1min)":
                    query = f"SELECT * FROM market_bars_1m WHERE symbol='{target_sym}' ORDER BY ts DESC LIMIT 100"
                else:
                    query = f"SELECT * FROM alpha_logs WHERE symbol='{target_sym}' ORDER BY ts DESC LIMIT 100"
                
                df = pd.read_sql(query, conn)
                conn.close()
                
                if not df.empty:
                    # 转换时间
                    df['time_str'] = pd.to_datetime(df['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if data_type == "Market Bars (1min)":
                        cols = ['time_str', 'open', 'high', 'low', 'close', 'volume', 'ts']
                    else:
                        cols = ['time_str', 'alpha', 'iv', 'price', 'vol_z', 'ts'] 
                        
                    final_cols = [c for c in cols if c in df.columns]
                    st.dataframe(df[final_cols], use_container_width=True)
                else:
                    st.warning("No data rows found.")
            except Exception as e:
                st.error(f"Read Error: {e}")
            

            
# === [Tab 8] 归一化健康监控 (PostgreSQL Source) ===
with tab8:
    st.header("🧪 Feature Norm Health (History)")
    st.markdown("""
    **功能**: 检查已持久化的归一化特征质量 (PostgreSQL -> `feature_logs`).
    - 用于排查：特征是否全0 (Dead)、是否NaN (Exploded)、分布是否合理。
    """)
    
    # 1. 日期选择
    c_date, c_btn = st.columns([1, 2])
    with c_date:
        target_date = st.date_input("Select Date", datetime.now(), key="tab8_date")
    
    date_str = target_date.strftime('%Y-%m-%d')
    dt_ny = NY_TZ.localize(datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S"))
    start_ts = int(dt_ny.timestamp())
    end_ts = start_ts + 86400
    
    with c_btn:
        st.write("") # Spacer
        run_check = st.button("🔄 Analyze History Features", key="btn_check_hist")

    if run_check:
        try:
            # 连接 DB (只读)
            conn = psycopg2.connect(PG_DB_URL)
            
            # 获取可用 Symbol
            symbols_df = pd.read_sql(f"SELECT DISTINCT symbol FROM feature_logs WHERE ts >= {start_ts} AND ts < {end_ts}", conn)
            symbols = symbols_df['symbol'].tolist()
            
            if not symbols:
                st.warning(f"No `feature_logs` found in PostgreSQL for {date_str}.")
            else:
                selected_sym = st.selectbox("Select Symbol to Inspect", symbols)
                
                # 读取该股票的数据 (限制 1000 条以防内存爆)
                query = f"SELECT ts, slow_norm_blob FROM feature_logs WHERE symbol='{selected_sym}' AND ts >= {start_ts} AND ts < {end_ts} ORDER BY ts DESC LIMIT 1000"
                df_feats = pd.read_sql(query, conn)
                conn.close()
                    
                if not df_feats.empty:
                    # 解析 Blob -> Matrix
                    arrays = []
                    valid_count = 0
                    for blob in df_feats['slow_norm_blob']:
                        if blob:
                            arr = np.frombuffer(blob, dtype=np.float32)
                            arrays.append(arr)
                            valid_count += 1
                    
                    if arrays:
                        matrix = np.array(arrays)
                        st.info(f"Analyzed {matrix.shape[0]} frames for {selected_sym}. Features Dim: {matrix.shape[1]}")
                        
                        # 计算统计量
                        df_stats = pd.DataFrame({
                            "Index": range(matrix.shape[1]),
                            "Name": SLOW_FEAT_NAMES_ALL if len(SLOW_FEAT_NAMES_ALL) == matrix.shape[1] else [f"F_{i}" for i in range(matrix.shape[1])],
                            "Mean": np.mean(matrix, axis=0),
                            "Std": np.std(matrix, axis=0),
                            "Min": np.min(matrix, axis=0),
                            "Max": np.max(matrix, axis=0),
                            "Zero%": np.mean(matrix == 0, axis=0) * 100,
                            "NaN%": np.mean(np.isnan(matrix), axis=0) * 100
                        })
                        
                        # 高亮样式
                        def highlight_health(row):
                            if row['NaN%'] > 0: return ['background-color: #ffcccc'] * len(row)
                            if row['Zero%'] > 99: return ['background-color: #ffffcc'] * len(row)
                            if abs(row['Mean']) > 0.95: return ['background-color: #ffe5cc'] * len(row)
                            return [''] * len(row)

                        st.dataframe(
                            df_stats.style.apply(highlight_health, axis=1)
                            .format({"Mean": "{:.3f}", "Std": "{:.3f}", "Zero%": "{:.1f}%", "NaN%": "{:.1f}%"}),
                            use_container_width=True,
                            height=600
                        )
                    else:
                        st.warning("Found logs but Blobs were empty.")
                else:
                    st.warning("No feature logs found for this symbol.")
        except Exception as e:
            st.error(f"Analysis Error: {e}")


# === [Tab 9] EOD 回测流水线 (Live -> Backtest) ===
with tab9:
    st.header("🔄 End-of-Day Replay Pipeline")
    st.markdown("""
    **流程说明**:
    1. **Export**: 从 PostgreSQL 导出归一化特征为 Parquet。
    2. **Pack (S0)**: 调用 `s0` 脚本生成 LMDB (Fast/Slow)。
    3. **Build (S5)**: 调用 `s5` 生成回测数据集。
    4. **Replay**: 运行 V8 Historical Replay 并对比实盘记录。
    """)
    
    c_cfg, c_act = st.columns([1, 2])
    with c_cfg:
        replay_date = st.date_input("Replay Date", datetime.now(), key="tab9_date")
        date_str = replay_date.strftime('%Y-%m-%d') # YYYY-MM-DD for folder args
        db_date_str = replay_date.strftime('%Y%m%d') # YYYYMMDD for sqlite
        
    with c_act:
        st.write("")
        start_replay = st.button("🚀 Start Full Replay", type="primary")

    # 状态容器
    status_box = st.container()
    
    if start_replay:
        with status_box:
            # --- Step 1: Export Data ---
            # --- Step 2: Run S0 (LMDB Packing) ---
            st.info("2️⃣ Running S0 (Creating LMDB)...")
            try:
                # 运行 Fast Channel S0
                cmd_s0_fast = [
                    sys.executable, 
                    str(SCRIPT_DIR / "s0_create_fast_channel_lmdb_alpha.py"),
                    "--date", date_str
                ]
                subprocess.run(cmd_s0_fast, check=True)
                
                # 运行 Slow Channel S0
                cmd_s0_slow = [
                    sys.executable, 
                    str(SCRIPT_DIR / "s0_create_slow_channel_lmdb_alpha.py"),
                    "--date", date_str
                ]
                subprocess.run(cmd_s0_slow, check=True)
                st.success("S0 LMDB Created.")
            except subprocess.CalledProcessError as e:
                st.error(f"S0 Error: {e}")
                st.stop()

            # --- Step 3: Run S5 (Delta Builder) ---
            st.info("3️⃣ Running S5 (Data Builder)...")
            try:
                cmd_s5 = [
                    sys.executable,
                    str(SCRIPT_DIR / "s5_new_delta_duckdb_builder_final_test.py"),
                    "--date", date_str
                ]
                subprocess.run(cmd_s5, check=True)
                st.success("S5 Data Build Complete.")
            except subprocess.CalledProcessError as e:
                st.error(f"S5 Error: {e}")
                st.stop()
                
            # --- Step 4: Run Backtest (Historical Replay) ---
            st.info("4️⃣ Running Historical Replay (V8)...")
            try:
                # [Fix 3] 使用 V8 根目录定位 replay 脚本
                replay_script = BASE_PROJECT_DIR / "history_replay" / "run_historical_replay.py"
                if not replay_script.exists():
                     st.error(f"Replay script not found: {replay_script}")
                     st.stop()

                cmd_bt = [
                    sys.executable,
                    str(replay_script),
                    "--date", date_str
                ]
                
                # 实时显示日志 (可选，这里简单调用 check=True)
                subprocess.run(cmd_bt, check=True)
                
                st.success("Prior-Day Replay Complete!")
                st.session_state['replay_complete'] = True
            except subprocess.CalledProcessError as e:
                st.error(f"Backtest Error: {e}")
                st.stop()

    # --- 对比视图 ---
    st.divider()
    st.subheader("🆚 Live vs Replay Comparison")
    
    if st.session_state.get('replay_complete', False):
        col_live, col_bt = st.columns(2)
        
        # A. 读取 Live Trades (从 PostgreSQL trade_logs)
        with col_live:
            st.markdown("### 🟢 Live Execution")
            try:
                conn = psycopg2.connect(PG_DB_URL)
                dt_ny = NY_TZ.localize(datetime.strptime(f"{date_str} 00:00:00", "%Y-%m-%d %H:%M:%S"))
                start_ts = int(dt_ny.timestamp())
                end_ts = start_ts + 86400
                # [🔥 模式切换] 比较面板也遵循全局开关进行显示切换
                from config import TRADING_ENABLED
                target_table = 'trade_logs' if TRADING_ENABLED else 'trade_logs_backtest'
                
                sql = f"SELECT * FROM {target_table} WHERE action != 'ALPHA' AND ts >= {start_ts} AND ts < {end_ts} ORDER BY ts"
                df_live = pd.read_sql(sql, conn)
                conn.close()
                
                if not df_live.empty:
                    df_live['time'] = pd.to_datetime(df_live['ts'], unit='s').dt.strftime('%H:%M:%S')
                    def parse_live_val(r, key):
                        try: return json.loads(r).get(key, np.nan)
                        except: return np.nan
                    df_live['account_cash'] = df_live['details_json'].apply(lambda r: parse_live_val(r, 'account_cash'))
                    st.dataframe(df_live[['time', 'symbol', 'action', 'qty', 'account_cash', 'price']])
                    st.metric("Live Trades", len(df_live))
                else:
                    st.warning("No Live trades found.")
            except Exception as e:
                st.error(f"Load Live Error: {e}")

        # B. 读取 Backtest Trades
        with col_bt:
            st.markdown("### 🔵 Backtest Result")
            # 假设回测输出在固定的 logs 目录
            bt_log_path = Path.home() / "quant_project/logs/replay_trades_v8.csv" # 需确认 V15 输出路径
            
            if bt_log_path.exists():
                df_bt = pd.read_csv(bt_log_path)
                # 过滤出回测日期的 (如果文件包含多天)
                # df_bt = df_bt[df_bt['date'] == date_str] 
                st.dataframe(df_bt)
                
                # 计算 PnL
                if 'pnl' in df_bt.columns:
                    total_pnl = df_bt['pnl'].sum()
                    st.metric("Backtest PnL", f"${total_pnl:,.2f}", delta_color="normal")
            else:
                st.warning(f"Backtest log not found at {bt_log_path}")

        # C. 差异分析
        st.markdown("### ⚠️ Consistency Check")
        # 简单的 Set Difference 对比 (按 Symbol + Action)
        if 'df_live' in locals() and 'df_bt' in locals() and not df_live.empty and not df_bt.empty:
            live_set = set(zip(df_live['symbol'], df_live['action']))
            bt_set = set(zip(df_bt['symbol'], df_bt['action'])) # 需确保列名一致
            
            missed_in_live = bt_set - live_set
            extra_in_live = live_set - bt_set
            
            c1, c2 = st.columns(2)
            if missed_in_live:
                c1.error(f"Missed in Live (Signal Lost): {missed_in_live}")
            else:
                c1.success("No Missed Trades")
                
            if extra_in_live:
                c2.warning(f"Extra in Live (False Positive): {extra_in_live}")
            else:
                c2.success("No Phantom Trades")

# 自动刷新逻辑
if auto_refresh:
    time.sleep(1)
    st.rerun()