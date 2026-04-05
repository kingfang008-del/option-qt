#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: config.py
描述: V8 系统统一配置文件
      所有模块共享的硬编码常量集中管理于此。
      [V8.1 重构版] 采用系统级环境变量 RUN_MODE 作为单一事实来源，彻底解决多进程死锁问题。
"""

import os
import pytz
from pathlib import Path

# ================= 时区 =================
NY_TZ = pytz.timezone('America/New_York')

# ================= 路径配置 =================
PROJECT_ROOT = Path.home() / "quant_project" # V8 根目录
DATA_DIR     = PROJECT_ROOT / "data"
LOG_DIR      = PROJECT_ROOT / "logs"
DB_DIR    = DATA_DIR / "history_sqlite_1m"
DB_DIR_1S    = DATA_DIR / "history_sqlite_1s"
CACHE_DIR    = DATA_DIR / "cache"

# 自动创建目录
for p in [DATA_DIR, LOG_DIR, DB_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_SERVICE_STATE_FILE = CACHE_DIR / "feature_service_state.pkl"

# ================= 系统运行模式 (单一事实来源) =================
# 模式枚举: 'REALTIME' (实盘), 'LIVEREPLAY' (流式回放), 'BACKTEST' (极速回测)
# 通过环境变量隔离，确保所有子进程对当前环境的认知绝对一致
RUN_MODE = os.environ.get("RUN_MODE", "REALTIME").upper()

IS_LIVEREPLAY = (RUN_MODE == 'LIVEREPLAY')
IS_BACKTEST   = (RUN_MODE == 'BACKTEST')
IS_SIMULATED  = (RUN_MODE in ['BACKTEST', 'LIVEREPLAY'])

# 全局交易开关 (如果在仿真模式下，强制关闭真实交易，保护实盘资金)
TRADING_ENABLED = False if IS_SIMULATED else True

# ================= 影子系统验证 (Shadow Validation) =================
ONLY_LOG_ALPHA = False  # [新增] 仅记录 Alpha 信号，不执行任何交易 (用于影子系统验证)

# 动态获取 Redis DB
def get_redis_db():
    if os.environ.get('REDIS_DB'):
        return int(os.environ.get('REDIS_DB'))
    # 仿真模式使用 DB 1，实盘使用 DB 0
    return 1 if IS_SIMULATED else 0

# ================= IBKR 连接配置 =================
IBKR_HOST       = '127.0.0.1'
IBKR_PORT       = 4001          # 🚨 4002=Paper (模拟盘), 4001=Real (实盘)
IBKR_CLIENT_ID  = 102
IBKR_ACCOUNT_ID = "DUK363545"

# ================= Redis 配置 =================
# 🚩 [核心流定义]
STREAM_FUSED_MARKET  = 'fused_market_stream'       # IBKR Connector → Feature Service
STREAM_INFERENCE     = 'unified_inference_stream'  # Feature Service → Orchestrator (SE)
STREAM_ORCH_SIGNAL   = 'orch_trade_signals'        # Orchestrator (SE) → Execution Engine (OMS)
STREAM_TRADE_LOG     = 'trade_log_stream'          # All Engines → Dashboard / Persistence

# 👥 [核心消费组定义]
GROUP_FEATURE        = 'feature_group'             # 特征计算组
GROUP_ORCH           = 'v8_orch_group'             # 信号引擎 (SE) 组
GROUP_OMS            = 'v8_oms_group'              # 执行引擎 (OMS) 组
GROUP_PERSISTENCE    = 'persistence_group'         # 持久化存储组

REDIS_CFG = {
    'host': 'localhost',
    'port': 6379,
    'db': get_redis_db(),  # 动态路由 (Backtest=1, Realtime=0)
    'group': GROUP_FEATURE,
    'orch_group': GROUP_ORCH,
    'oms_group': GROUP_OMS,
    'input_stream': STREAM_INFERENCE,
    'raw_stream': STREAM_FUSED_MARKET,
    'signal_stream': STREAM_ORCH_SIGNAL,
    'trade_log_stream': STREAM_TRADE_LOG,
    'option_stream': 'live_option_snapshot'
}
HASH_OPTION_SNAPSHOT = 'live_option_snapshot'      # IBKR Connector → Dashboard

# ================= 数据库配置 (PostgreSQL) =================
PG_DB_URL = "dbname=quant_trade user=postgres password=postgres host=192.168.50.116 port=5432"

# ================= 核心交易标的 =================
TARGET_SYMBOLS =  [
    'VIXY', 'SPY', 'QQQ', 
    # --- Tier 1: 巨无霸 ---
    'NVDA', 'AAPL', 'META', 'PLTR', 'TSLA', 'UNH', 'AMZN', 'AMD', 'MSTR', 'COIN',
    # --- Tier 2: 核心蓝筹 ---
    'NFLX', 'CRWV', 'AVGO', 'MSFT', 'HOOD', 'MU', 'APP', 'GOOGL', 'WMT',  'GS',
    # --- Tier 3: 高流动性 --- 
    'SMCI', 'ADBE', 'CRM', 'ORCL', 'NKE', 'XOM', 'INTC', 'DELL', 'IWM', 'GLD'
]

# ================= 价格模式 =================
USE_BID_ASK_PRICING = True  # True=使用 Bid/Ask 中间价及 BSM 校准, False=仅使用最新成交价
USE_5M_OPTION_DATA  = True  # 关闭 5min 维度期权数据，用于排查爆仓问题

# ================= 资金管理与风控 =================
INITIAL_ACCOUNT         = 50000.0     # 初始资金 ($)
MAX_POSITIONS           = 4           # 最大同时持仓数
POSITION_RATIO          = 1.0 / 4.0   # 单标的最大仓位比例
MAX_TRADE_CAP           = 150000.0    # 单笔交易最大金额 ($)
GLOBAL_EXPOSURE_LIMIT   = 0.90        # 全局风险敞口上限
COMMISSION_PER_CONTRACT = 0.65        # 手续费 ($/手)

# ================= BSM 定价参数 =================
 

# ================= Alpha / 信号 =================
ROLLING_WINDOW = 30               # Alpha 滚动窗口
CORR_THRESHOLD = -0.1             # 反转相关性阈值

# ================= 订单执行 =================
ORDER_TIMEOUT_SECONDS = 30         # 挂单超时
ORDER_MAX_RETRIES     = 3          # 最大追单次数
COOLDOWN_MINUTES      = 60         # 止损后的品种冷却时间 (分钟)
LIMIT_BUFFER_ENTRY    = 1.03       # 买入限价缓冲 (Ask * 1.03)
LIMIT_BUFFER_EXIT     = 0.97       # 卖出限价缓冲 (Bid * 0.97)
SLIPPAGE_ENTRY_PCT    = 0.001      # 建仓滑点 (0.1%)
SLIPPAGE_EXIT_PCT     = 0.001      # 平仓滑点 (0.1%)
EXIT_ORDER_TYPE       = 'MKT'      # 平仓订单类型 (MKT/LMT)
DISABLE_ICEBERG       = False      # [新增] 是否禁用冰山拆单逻辑 (用于对比秒级与分钟级一致性)
SYNC_EXECUTION        = False      # [新增] 同步执行模式 (用于 bit-perfect 确定性回放验证)

# ================= Dashboard 配置 =================
DASHBOARD_REFRESH_RATE = 1.0

# ================= 6 Bucket 期权锚点 =================
BUCKET_SPECS = {
    'PUT_ATM':       {'delta': -0.50, 'bucket_idx': 0},
    'PUT_OTM':       {'delta': -0.25, 'bucket_idx': 1},
    'CALL_ATM':      {'delta':  0.50, 'bucket_idx': 2},
    'CALL_OTM':      {'delta':  0.25, 'bucket_idx': 3},
    'NEXT_PUT_ATM':  {'delta': -0.50, 'bucket_idx': 4},
    'NEXT_CALL_ATM': {'delta':  0.50, 'bucket_idx': 5},
}
TAG_TO_INDEX = {k: v['bucket_idx'] for k, v in BUCKET_SPECS.items()}

def get_synced_funds(real_balance: float = None) -> float:
    """
    统一的资金同步逻辑：
    - 读取 TRADING_ENABLED 的状态
    - 如果 TRADING_ENABLED == True 且能获取到有效的真实资金 (real_balance > 0)，则使用真实资金
    - 否则 (只读模式或获取失败) 统一返回 INITIAL_ACCOUNT (模拟资金)
    """
    if TRADING_ENABLED and real_balance is not None and real_balance > 0:
        return float(real_balance)
    return float(INITIAL_ACCOUNT)