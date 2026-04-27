import streamlit as st
import redis
import json
import pickle
import sys
import os
import sqlite3
from pathlib import Path

# [NEW] Add project root to sys.path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import serialization_utils as ser
import pandas as pd
import numpy as np
import time
import random

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
    STREAM_TRADE_LOG, BUCKET_SPECS, TAG_TO_INDEX, IBKR_PORT, PG_DB_URL,
    AUTO_TRADING_CAPITAL_RATIO, MANUAL_TRADING_CAPITAL_RATIO, MANUAL_ORDER_ALLOC_RATIO,
    COMMISSION_PER_CONTRACT,
    STREAM_ORCH_SIGNAL, GROUP_OMS, GROUP_ORCH, RUN_MODE, OMS_STATE_NAMESPACE,
    LIVE_TRADING_CAPITAL_LIMIT, TRADING_ENABLED,
)
from dashboard_cash_utils import (
    parse_oms_ledger_hash,
    parse_oms_live_cash_payload,
    select_remaining_cash,
)
from runtime_trading_controls import (
    clear_runtime_live_trading_capital_limit,
    get_runtime_live_trading_capital_limit,
    get_runtime_trading_enabled,
    set_runtime_live_trading_capital_limit,
    set_runtime_trading_enabled,
)
from orchestrator_order_state import namespaced_pending_orders_key

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

def _load_feat_resolution_map(json_path):
    """从 feature JSON 提取 name -> resolution，供 Debug Inspector 解释低频广播特征。"""
    try:
        if not json_path.exists():
            return {}
        with open(json_path, 'r') as f:
            cfg = _json.load(f)
        return {
            feat['name']: feat.get('resolution', '1min')
            for feat in cfg.get('features', [])
            if isinstance(feat, dict) and feat.get('name')
        }
    except Exception:
        return {}

def fetch_latest_mock_cash():
    """从 PostgreSQL symbol_state 表中读取 _GLOBAL_STATE_ 里的 mock_cash"""
    from config import TRADING_ENABLED
    # 如果禁止了实盘交易，绝对不能读数据库里的历史残留资金。
    if not TRADING_ENABLED:
        return None

    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute(
            "SELECT data, updated_at FROM symbol_state WHERE namespace = %s AND symbol = '_GLOBAL_STATE_'",
            (OMS_STATE_NAMESPACE,),
        )
        row = c.fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            updated_at = row[1]
            try:
                # 仅接受 NY 当日快照，避免旧 _GLOBAL_STATE_ 污染 Dashboard 显示
                ts_ny = datetime.fromtimestamp(float(updated_at), NY_TZ).date()
                if ts_ny != datetime.now(NY_TZ).date():
                    return None
            except Exception:
                return None
            return float(data.get('mock_cash', 0.0))
    except: pass
    return None


def fetch_live_oms_cash(
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
    return_source: bool = False,
    expected_modes=None,
):
    """优先读取 OMS 广播现金 (oms:live_positions.____SYSTEM_CASH____).

    Remaining Cash 的核心口径应该是 OMS 当前现金；若仅因心跳短暂过期就回退到
    Trade Log/PG，会造成页面在两种口径间抖动，表现为“无交易也忽大忽小”。
    """
    try:
        r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']}, decode_responses=False)
        # 1) 首选：OMS 账本投影（单一口径，避免 trade_log/PG 推导抖动）
        try:
            ledger = r.hgetall("meta:oms_ledger") or {}
            cash, source = parse_oms_ledger_hash(
                ledger,
                max_age_sec=max_age_sec,
                allow_stale=allow_stale,
                expected_modes=expected_modes,
            )
            if cash is not None:
                return (cash, source) if return_source else cash
        except Exception:
            pass

        # 2) 兼容旧口径：oms:live_positions 里的系统现金心跳
        raw = r.hget("oms:live_positions", "____SYSTEM_CASH____")
        cash, source = parse_oms_live_cash_payload(
            raw,
            max_age_sec=max_age_sec,
            allow_stale=allow_stale,
            expected_modes=expected_modes,
        )
        if cash is not None:
            return (cash, source) if return_source else cash
        return (None, None) if return_source else None
    except Exception:
        return (None, None) if return_source else None


def fetch_live_broker_balance():
    """Read latest broker account snapshot from Redis connector cache."""
    try:
        r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']}, decode_responses=False)
        raw_acct = r.hget('live_account_info', 'balance')
        if not raw_acct:
            return None, None
        acct_data = ser.unpack(raw_acct)
        net_liq = float(acct_data.get('net_liquidation', 0.0) or 0.0)
        avail = float(acct_data.get('available_funds', 0.0) or 0.0)
        return net_liq, avail
    except Exception:
        return None, None


def _safe_quantile(series, q):
    vals = pd.to_numeric(series, errors='coerce').dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(q))


def _build_execution_event_time(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype='datetime64[ns, America/New_York]')
    for col in ['order_submit_ts_ny', 'fill_ts_ny']:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors='coerce')
            if not ts.dropna().empty:
                return ts
    if 'datetime_ny' in df.columns:
        ts = pd.to_datetime(df['datetime_ny'], errors='coerce')
        if getattr(ts.dt, 'tz', None) is None:
            ts = ts.dt.tz_localize(NY_TZ, ambiguous='NaT', nonexistent='NaT')
        else:
            ts = ts.dt.tz_convert(NY_TZ)
        if not ts.dropna().empty:
            return ts
    if 'ts' in df.columns:
        return pd.to_datetime(df['ts'], unit='s', errors='coerce', utc=True).dt.tz_convert(NY_TZ)
    return pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, America/New_York]')


def _execution_bucket_suggestion(row) -> str:
    samples = int(row.get('open_count', 0) or 0)
    avg_fill_ratio = float(row.get('avg_fill_ratio', np.nan))
    full_fill_rate = float(row.get('full_fill_rate', np.nan))
    partial_fill_rate = float(row.get('partial_fill_rate', np.nan))
    p50_ms = float(row.get('submit_to_fill_p50_ms', np.nan))
    p90_ms = float(row.get('submit_to_fill_p90_ms', np.nan))

    if samples < 3:
        return "样本不足，先继续收集"
    if partial_fill_rate >= 0.35 and p90_ms >= 8000:
        return "尾部等待偏长且部分成交高，优先减小单笔名义金额"
    if full_fill_rate < 0.45 and p90_ms >= 5000:
        return "全成率偏低，可提高初始 probe 或加快重提"
    if avg_fill_ratio < 0.75 and p50_ms >= 2500:
        return "成交推进偏慢，重点看 submit->fill 尾部拥堵"
    if full_fill_rate >= 0.85 and p50_ms <= 1200:
        return "成交较充足，可考虑略收窄初始探价"
    return "当前执行质量稳定，暂不建议调整"


def _overall_execution_advice(exec_summary: pd.DataFrame) -> str:
    if exec_summary.empty:
        return "今日 OPEN 样本不足，暂时无法生成建议。"

    sample_count = int(exec_summary['open_count'].sum())
    avg_fill_ratio = float(pd.to_numeric(exec_summary['avg_fill_ratio'], errors='coerce').mean())
    avg_full_fill_rate = float(pd.to_numeric(exec_summary['full_fill_rate'], errors='coerce').mean())
    p90_candidates = pd.to_numeric(exec_summary['submit_to_fill_p90_ms'], errors='coerce').dropna()
    avg_p90_ms = float(p90_candidates.mean()) if not p90_candidates.empty else np.nan

    if sample_count < 5:
        return "OPEN 样本还少，先观察更多 bucket 后再调参数。"
    if avg_full_fill_rate < 0.45 and avg_p90_ms >= 6000:
        return "全成率偏低且尾部等待偏长，优先从初始 probe / 重提 aggressiveness 入手。"
    if avg_fill_ratio < 0.75:
        return "平均 fill_ratio 偏低，建议先减小单笔名义金额，再观察 latency 尾部是否回落。"
    if avg_full_fill_rate >= 0.85 and avg_p90_ms <= 2000:
        return "执行质量已较好，当前参数可维持，后续可尝试小幅收窄初始限价。"
    return "执行质量中性，先盯 p90 submit->fill 与 partial fill 的时段集中度。"


def build_execution_quality_summary(df_open: pd.DataFrame, bucket_minutes: int = 15) -> pd.DataFrame:
    if df_open.empty:
        return pd.DataFrame()

    work = df_open.copy()
    work['event_time_ny'] = _build_execution_event_time(work)
    work['fill_ratio_num'] = pd.to_numeric(work.get('fill_ratio'), errors='coerce')
    work['submit_to_fill_ms_num'] = pd.to_numeric(work.get('submit_to_fill_ms'), errors='coerce')
    work['alpha_to_fill_ms_num'] = pd.to_numeric(work.get('alpha_to_fill_ms'), errors='coerce')
    work = work.dropna(subset=['event_time_ny'])
    if work.empty:
        return pd.DataFrame()

    work['time_bucket'] = work['event_time_ny'].dt.floor(f'{int(bucket_minutes)}min')
    summary = (
        work.groupby('time_bucket', dropna=False)
        .apply(
            lambda g: pd.Series({
                'open_count': int(len(g)),
                'avg_fill_ratio': float(pd.to_numeric(g['fill_ratio_num'], errors='coerce').dropna().mean()) if not pd.to_numeric(g['fill_ratio_num'], errors='coerce').dropna().empty else np.nan,
                'full_fill_rate': float((pd.to_numeric(g['fill_ratio_num'], errors='coerce') >= 0.999).mean()),
                'partial_fill_rate': float(((pd.to_numeric(g['fill_ratio_num'], errors='coerce') > 0) & (pd.to_numeric(g['fill_ratio_num'], errors='coerce') < 0.999)).mean()),
                'submit_to_fill_p50_ms': _safe_quantile(g['submit_to_fill_ms_num'], 0.50),
                'submit_to_fill_p90_ms': _safe_quantile(g['submit_to_fill_ms_num'], 0.90),
                'alpha_to_fill_p90_ms': _safe_quantile(g['alpha_to_fill_ms_num'], 0.90),
            })
        )
        .reset_index()
    )
    summary['suggestion'] = summary.apply(_execution_bucket_suggestion, axis=1)
    return summary

_FEAT_JSON_DIR = BASE_PROJECT_DIR.parent / "CONFIG"
if not (_FEAT_JSON_DIR / "slow_feature.json").exists():
    _FEAT_JSON_DIR = SCRIPT_DIR  # legacy fallback: daily_backtest/
FAST_FEAT_NAMES = _load_feat_names(_FEAT_JSON_DIR / "fast_feature.json")
SLOW_FEAT_NAMES = _load_feat_names(_FEAT_JSON_DIR / "slow_feature.json")
FAST_FEAT_RESOLUTIONS = _load_feat_resolution_map(_FEAT_JSON_DIR / "fast_feature.json")
SLOW_FEAT_RESOLUTIONS = _load_feat_resolution_map(_FEAT_JSON_DIR / "slow_feature.json")

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


def _apply_runtime_control_action(pending_action: dict):
    rds = get_redis_client()
    if not rds:
        raise RuntimeError("Redis unavailable")
    action = str((pending_action or {}).get("action") or "")
    if action == "set_cap":
        new_cap = float((pending_action or {}).get("value") or 0.0)
        saved = set_runtime_live_trading_capital_limit(new_cap, r=rds, source="dashboard")
        return f"Live trading cap saved: ${saved:,.2f}"
    if action == "reset_cap":
        clear_runtime_live_trading_capital_limit(r=rds, source="dashboard_reset")
        return "Live trading cap reset to config default"
    if action == "set_trading":
        enabled = bool((pending_action or {}).get("value"))
        saved = set_runtime_trading_enabled(enabled, r=rds, source="dashboard")
        return "Live trading armed" if saved else "Live trading disarmed"
    raise ValueError(f"Unsupported runtime control action: {action}")


def _render_runtime_control_dialog():
    pending = st.session_state.get("runtime_control_pending")
    if not pending:
        notice = st.session_state.pop("runtime_control_notice", "")
        if notice:
            st.success(notice)
        return

    def _cancel():
        st.session_state.pop("runtime_control_pending", None)
        st.rerun()

    def _confirm():
        notice = _apply_runtime_control_action(pending)
        st.session_state["runtime_control_notice"] = notice
        st.session_state.pop("runtime_control_pending", None)
        st.rerun()

    title = "Confirm Live Control Change"
    body = str(pending.get("message") or "Please confirm this live runtime change.")

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dialog():
            st.warning(body)
            st.caption("Type `CONFIRM` to continue.")
            token = st.text_input("Confirmation", key="runtime_control_confirm_token")
            c1, c2 = st.columns(2)
            if c1.button("Confirm", type="primary", use_container_width=True, disabled=(token.strip().upper() != "CONFIRM")):
                _confirm()
            if c2.button("Cancel", use_container_width=True):
                _cancel()

        _dialog()
        return

    with st.sidebar.expander("Confirm Live Control Change", expanded=True):
        st.warning(body)
        token = st.text_input("Type CONFIRM", key="runtime_control_confirm_token_fallback")
        c1, c2 = st.columns(2)
        if c1.button("Confirm", type="primary", use_container_width=True, disabled=(token.strip().upper() != "CONFIRM")):
            _confirm()
        if c2.button("Cancel", use_container_width=True):
            _cancel()

def get_ibkr_connector_status(rds):
    """读取 ibkr_connector_v8 上报的连接状态。"""
    if rds is None:
        return {}
    try:
        raw = rds.hget("live_ibkr_connector", "status")
        if not raw:
            return {}
        data = ser.unpack(raw)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}
    
def _parse_account_metrics(account_values):
    """从 IB accountValues 提取净值/可用资金（USD）。"""
    out = {
        "net_liq": None,
        "available_funds": None,
    }
    if not account_values:
        return out
    for val in account_values:
        if getattr(val, "tag", "") == "NetLiquidation" and getattr(val, "currency", "") == "USD":
            try:
                out["net_liq"] = float(val.value)
            except Exception:
                pass
        if getattr(val, "tag", "") == "AvailableFunds" and getattr(val, "currency", "") == "USD":
            try:
                out["available_funds"] = float(val.value)
            except Exception:
                pass
    return out

def _calc_auto_open_notional_from_redis(rds):
    """
    统计自动引擎当前占用名义本金（来自 oms:live_positions）。
    口径: sum(qty * entry_price * 100)，仅 position!=0 且 qty>0 的状态。
    """
    if rds is None:
        return 0.0
    total = 0.0
    try:
        raw_map = rds.hgetall("oms:live_positions") or {}
        for _, raw_val in raw_map.items():
            try:
                if isinstance(raw_val, (bytes, bytearray)):
                    txt = raw_val.decode("utf-8", errors="ignore")
                else:
                    txt = str(raw_val)
                state = json.loads(txt)
                pos = int(state.get("position", state.get("pos", 0)) or 0)
                qty = float(state.get("qty", 0.0) or 0.0)
                entry_price = float(state.get("entry_price", state.get("price", 0.0)) or 0.0)
                if pos != 0 and qty > 0 and entry_price > 0:
                    total += qty * entry_price * 100.0
            except Exception:
                continue
    except Exception:
        return 0.0
    return float(total)

def _build_manual_capital_snapshot(total_equity, ib_portfolio, rds):
    """
    计算自动/手动资金池快照:
    - auto_budget = total_equity * AUTO_TRADING_CAPITAL_RATIO
    - manual_budget = total_equity - auto_budget
    - auto_used = Redis 自动仓位占用
    - ib_total_open = IB 当前全部持仓名义市值(绝对值)
    - manual_used = max(0, ib_total_open - auto_used)  # 粗略分解
    """
    total_equity = float(total_equity or 0.0)
    auto_budget = total_equity * float(AUTO_TRADING_CAPITAL_RATIO)
    manual_budget = total_equity - auto_budget
    auto_used = _calc_auto_open_notional_from_redis(rds)
    ib_total_open = 0.0
    for item in (ib_portfolio or []):
        try:
            ib_total_open += abs(float(getattr(item, "marketValue", 0.0) or 0.0))
        except Exception:
            continue
    manual_used = max(0.0, ib_total_open - auto_used)
    return {
        "total_equity": total_equity,
        "auto_budget": max(0.0, auto_budget),
        "manual_budget": max(0.0, manual_budget),
        "auto_used": max(0.0, auto_used),
        "manual_used": max(0.0, manual_used),
        "auto_available": max(0.0, auto_budget - auto_used),
        "manual_available": max(0.0, manual_budget - manual_used),
    }


def fetch_live_oms_positions(
    max_age_sec: float = 120.0,
    allow_stale: bool = False,
    expected_modes=None,
    return_meta: bool = False,
):
    """Return current OMS live positions when the live ledger is fresh/mode-matched."""
    def _valid_bucket_tag(raw_tag: str) -> str:
        tag = str(raw_tag or "").strip().upper()
        return tag if tag in TAG_TO_INDEX else ""

    positions = {}
    meta = {
        "fresh": False,
        "skipped_unconfirmed": 0,
    }
    try:
        r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']}, decode_responses=False)
        ledger = r.hgetall("meta:oms_ledger") or {}
        cash_raw = r.hget("oms:live_positions", "____SYSTEM_CASH____")
        ledger_cash, _ = parse_oms_ledger_hash(
            ledger,
            max_age_sec=max_age_sec,
            allow_stale=allow_stale,
            expected_modes=expected_modes,
        )
        live_cash, _ = parse_oms_live_cash_payload(
            cash_raw,
            max_age_sec=max_age_sec,
            allow_stale=allow_stale,
            expected_modes=expected_modes,
        )
        if ledger_cash is None and live_cash is None:
            return (positions, meta) if return_meta else positions
        meta["fresh"] = True

        raw_map = r.hgetall("oms:live_positions") or {}
        for raw_sym, raw_val in raw_map.items():
            try:
                sym = raw_sym.decode("utf-8", errors="ignore") if isinstance(raw_sym, (bytes, bytearray)) else str(raw_sym)
                if sym == "____SYSTEM_CASH____":
                    continue
                txt = raw_val.decode("utf-8", errors="ignore") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
                state = json.loads(txt)
                pos = int(state.get("position", state.get("pos", 0)) or 0)
                qty = float(state.get("qty", 0.0) or 0.0)
                cost = float(state.get("entry_price", state.get("price", 0.0)) or 0.0)
                stock = float(state.get("entry_stock", state.get("stock", 0.0)) or 0.0)
                if pos == 0 or qty <= 0:
                    continue
                if not bool(state.get("open_fill_confirmed", False)):
                    meta["skipped_unconfirmed"] += 1
                    continue
                opt_type = str(state.get("opt_type", "") or "").strip().lower()
                positions[sym] = {
                    'position': pos,
                    'qty': qty,
                    'cost': cost if np.isfinite(cost) and cost > 0 else 0.0,
                    'stock': stock if np.isfinite(stock) and stock > 0 else 0.0,
                    'tag': _valid_bucket_tag(state.get("tag", "")),
                    'opt_type': opt_type,
                    'contract_id': str(state.get("contract_id", "") or ""),
                    'last_opt_price': float(state.get("last_opt_price", 0.0) or 0.0),
                    'entry_ts': float(state.get("entry_ts", 0.0) or 0.0),
                    'source': 'OMS live',
                }
            except Exception:
                continue
    except Exception:
        return (positions, meta) if return_meta else positions
    return (positions, meta) if return_meta else positions


def submit_oms_manual_close(symbol: str, position: int, qty: float, ref_price: float, stock_price: float = 0.0, contract_id: str = ""):
    """Send a manual close request to OMS, reusing ExecutionEngine's SELL path."""
    sym = str(symbol or "").strip().upper()
    pos = int(position or 0)
    qty = float(qty or 0.0)
    ref_price = float(ref_price or 0.0)
    if not sym:
        return False, "missing symbol"
    if pos == 0 or qty <= 0:
        return False, f"{sym} has no valid OMS position/qty"
    try:
        r_cmd = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']}, decode_responses=False)
        now_ts = time.time()
        request_id = f"dashboard-close-{sym}-{int(now_ts * 1000)}"
        payload = {
            "source": "dashboard_manual_close",
            "action": "SELL",
            "ts": now_ts,
            "symbol": sym,
            "stock_price": float(stock_price or 0.0),
            "batch_idx": -1,
            "frame_id": request_id,
            "sig": {
                "action": "SELL",
                "dir": pos,
                "target_side": pos,
                "original_position": pos,
                "price": max(ref_price, 0.01),
                "market_price": max(ref_price, 0.01),
                "bid": 0.0,
                "ask": 0.0,
                "bid_size": qty,
                "ask_size": qty,
                "reason": f"DASHBOARD_FORCE_CLOSE:{request_id}",
                "meta": {
                    "contract_id": str(contract_id or ""),
                    "manual_close": True,
                    "request_id": request_id,
                },
            },
        }
        r_cmd.xadd(STREAM_ORCH_SIGNAL, {"data": ser.pack(payload)}, maxlen=10000)
        return True, request_id
    except Exception as e:
        return False, str(e)


def fetch_live_pending_orders():
    rows = []
    try:
        r = redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ['host', 'port', 'db']}, decode_responses=False)
        raw_map = r.hgetall(namespaced_pending_orders_key(OMS_STATE_NAMESPACE)) or {}
        for raw_key, raw_val in raw_map.items():
            try:
                order_key = raw_key.decode("utf-8", errors="ignore") if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
                txt = raw_val.decode("utf-8", errors="ignore") if isinstance(raw_val, (bytes, bytearray)) else str(raw_val)
                payload = json.loads(txt)
                rows.append({
                    "order_key": order_key,
                    "symbol": str(payload.get("symbol", "") or ""),
                    "contract_id": str(payload.get("contract_id", "") or ""),
                    "intent": str(payload.get("intent", "") or ""),
                    "side": str(payload.get("side", "") or ""),
                    "status": str(payload.get("status", "") or ""),
                    "target_qty": int(payload.get("target_qty", 0) or 0),
                    "filled_qty": int(payload.get("filled_qty", 0) or 0),
                    "remaining_qty": int(payload.get("remaining_qty", 0) or 0),
                    "limit_price": float(payload.get("limit_price", 0.0) or 0.0),
                    "retry_count": int(payload.get("retry_count", 0) or 0),
                    "reserved_cash": float(payload.get("reserved_cash", 0.0) or 0.0),
                    "slot_reserved": bool(payload.get("slot_reserved", False)),
                    "broker_status": str(payload.get("broker_status", "") or ""),
                    "updated_at": float(payload.get("last_update_ts", 0.0) or 0.0),
                    "reason": str(payload.get("reason", "") or ""),
                })
            except Exception:
                continue
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "updated_at" in df.columns:
        df["updated_at_ny"] = pd.to_datetime(df["updated_at"], unit="s", errors="coerce", utc=True).dt.tz_convert(NY_TZ)
    return df.sort_values(["updated_at", "symbol"], ascending=[False, True], na_position="last")

def _fetch_latest_option_snapshot(symbol):
    """读取某 symbol 最新 buckets/contracts 快照（优先 Redis，失败回退 PG）。"""
    # Redis 优先（实时）
    try:
        rds = get_redis_client()
        if rds:
            raw = rds.hget(HASH_OPTION_SNAPSHOT, symbol)
            if raw:
                snap = ser.unpack(raw)
                if isinstance(snap, dict):
                    return snap.get("buckets", []), snap.get("contracts", [])
                if isinstance(snap, list):
                    return snap, []
    except Exception:
        pass

    # PG 回退（近似最新）
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute(
            "SELECT buckets_json FROM option_snapshots_1m WHERE symbol=%s ORDER BY ts DESC LIMIT 1",
            (symbol,)
        )
        row = c.fetchone()
        conn.close()
        if row:
            snap = row[0]
            if isinstance(snap, str):
                snap = json.loads(snap)
            if isinstance(snap, dict):
                return snap.get("buckets", []), snap.get("contracts", [])
            if isinstance(snap, list):
                return snap, []
    except Exception:
        pass
    return [], []

def _get_bucket_quote_and_contract(symbol, tag):
    """
    返回指定 symbol + bucket(tag) 的实时 quote 与合约标识。
    """
    idx = TAG_TO_INDEX.get(tag, -1)
    if idx < 0:
        return None
    buckets, contracts = _fetch_latest_option_snapshot(symbol)
    if not buckets or len(buckets) <= idx:
        return None
    row = buckets[idx]
    if not isinstance(row, (list, tuple)) or len(row) < 10:
        return None
    bid = float(row[8] or 0.0) if len(row) > 8 else 0.0
    ask = float(row[9] or 0.0) if len(row) > 9 else 0.0
    last = float(row[0] or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 and ask >= bid else (last if last > 0 else 0.0)
    contract_txt = contracts[idx] if contracts and len(contracts) > idx else ""
    return {
        "bucket_idx": idx,
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "contract_text": contract_txt,
    }

def _fetch_locked_contract_row(symbol, tag):
    """
    从 contract_locks 读取当日(若无则最近日期)锁定合约，供人工手动触发开仓使用。
    """
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        c.execute(
            """
            SELECT date, symbol, tag, conid, expiry, strike, p_right, multiplier, localsymbol, tradingclass
            FROM contract_locks
            WHERE symbol=%s AND tag=%s
            ORDER BY date DESC
            LIMIT 1
            """,
            (symbol, tag),
        )
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "date": row[0],
            "symbol": row[1],
            "tag": row[2],
            "conId": int(row[3]) if row[3] is not None else None,
            "expiry": row[4],
            "strike": float(row[5]) if row[5] is not None else 0.0,
            "right": row[6],
            "multiplier": str(row[7]) if row[7] is not None else "100",
            "localSymbol": row[8] or "",
            "tradingClass": row[9] or "",
        }
    except Exception:
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
        current_mode = _normalize_trade_mode(RUN_MODE)
        for _, payload in entries:
            if b'pickle' in payload:
                msg = ser.unpack(payload[b'pickle'])
                # [Fix] 过滤掉 ALPHA 日志，只保留真实交易
                if msg.get('action') == 'ALPHA': continue
                msg_mode = _normalize_trade_mode(msg.get('mode'))
                if msg_mode and msg_mode != current_mode:
                    continue
                if not msg_mode and current_mode != 'REALTIME':
                    continue
                
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


@st.cache_data(ttl=30)
def load_option_quote_quality(symbol, limit=240):
    """读取近期 option_snapshots_1m，并统计盘口质量。"""
    summary = {
        'snapshots': 0,
        'valid_quote_ratio': 0.0,
        'median_spread_pct': np.nan,
        'median_depth': np.nan,
        'stale_gap_count': 0,
        'median_mid_last_bps': np.nan,
    }
    bucket_df = pd.DataFrame()
    try:
        conn = psycopg2.connect(PG_DB_URL)
        from datetime import timedelta
        ts_3_days_ago = int((datetime.now(NY_TZ) - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        query = f"""
            SELECT ts, buckets_json FROM option_snapshots_1m
            WHERE symbol = '{symbol}'
            AND ts >= {ts_3_days_ago}
            ORDER BY ts DESC
            LIMIT {int(limit)}
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if df.empty:
            return summary, bucket_df

        df = df.sort_values('ts').reset_index(drop=True)
        summary['snapshots'] = int(len(df))
        if len(df) >= 2:
            gaps = df['ts'].diff().dropna()
            summary['stale_gap_count'] = int((gaps > 120).sum())

        rows = []
        for snap in df['buckets_json']:
            try:
                blob = snap if isinstance(snap, dict) else json.loads(snap) if isinstance(snap, str) else {}
            except Exception:
                blob = {}
            buckets = blob.get('buckets', []) if isinstance(blob, dict) else []
            for idx, bucket in enumerate(buckets[:6]):
                if not isinstance(bucket, (list, tuple)) or len(bucket) < 12:
                    continue
                price = float(bucket[0] or 0.0)
                bid = float(bucket[8] or 0.0)
                ask = float(bucket[9] or 0.0)
                bid_size = float(bucket[10] or 0.0)
                ask_size = float(bucket[11] or 0.0)
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
                valid = int(bid > 0 and ask > 0 and ask >= bid)
                spread_pct = ((ask - bid) / mid) if valid and mid > 0 else np.nan
                mid_last_bps = (abs(mid - price) / mid * 10000.0) if valid and price > 0 and mid > 0 else np.nan
                rows.append({
                    'bucket': idx,
                    'valid': valid,
                    'spread_pct': spread_pct,
                    'depth': bid_size + ask_size,
                    'mid_last_bps': mid_last_bps,
                })

        if not rows:
            return summary, bucket_df

        bucket_df = pd.DataFrame(rows)
        summary['valid_quote_ratio'] = float(bucket_df['valid'].mean()) if not bucket_df.empty else 0.0
        valid_only = bucket_df[bucket_df['valid'] == 1]
        if not valid_only.empty:
            summary['median_spread_pct'] = float(valid_only['spread_pct'].median())
            summary['median_depth'] = float(valid_only['depth'].median())
            summary['median_mid_last_bps'] = float(valid_only['mid_last_bps'].median())

        bucket_df = (
            bucket_df.groupby('bucket', as_index=False)
            .agg(
                valid_quote_ratio=('valid', 'mean'),
                median_spread_pct=('spread_pct', 'median'),
                median_depth=('depth', 'median'),
                median_mid_last_bps=('mid_last_bps', 'median'),
            )
            .sort_values('bucket')
        )
        return summary, bucket_df
    except Exception as e:
        print(f"Error loading option quote quality: {e}")
        return summary, pd.DataFrame()


@st.cache_data(ttl=30)
def _normalize_trade_mode(value) -> str:
    mode = str(value or "").strip().upper()
    return mode if mode in {"REALTIME", "REALTIME_DRY", "BACKTEST", "SHADOW"} else ""


def _build_trade_mode_mask(df: pd.DataFrame, expected_mode: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    current_mode = _normalize_trade_mode(expected_mode)
    if not current_mode:
        return pd.Series(True, index=df.index)
    mode_series = df.get('mode', pd.Series(index=df.index, dtype=object)).map(_normalize_trade_mode)
    mask = mode_series.eq(current_mode)
    if current_mode == "REALTIME" and 'source_table' in df.columns:
        legacy_live_mask = mode_series.eq("") & df['source_table'].astype(str).eq('trade_logs')
        mask = mask | legacy_live_mask
    return mask.fillna(False)


def fetch_alpha_execution_funnel(query_date, target_table, expected_mode=None):
    """统计 alpha -> open/fill/close 的漏斗。"""
    try:
        start_dt = datetime.combine(query_date, dt_time(0, 0, 0))
        start_ts = int(NY_TZ.localize(start_dt).timestamp())
        end_ts = start_ts + 86400

        conn = psycopg2.connect(PG_DB_URL)
        df_alpha = pd.read_sql(
            f"""
            SELECT ts, symbol, alpha, iv, price, vol_z
            FROM alpha_logs
            WHERE ts >= {start_ts} AND ts < {end_ts}
            """,
            conn,
        )
        df_trade = pd.read_sql(
            f"""
            SELECT ts, symbol, action, details_json
            FROM {target_table}
            WHERE ts >= {start_ts} AND ts < {end_ts}
            """,
            conn,
        )
        conn.close()

        if df_trade.empty:
            df_trade = pd.DataFrame(columns=['ts', 'symbol', 'action', 'details_json'])

        if not df_trade.empty:
            def _parse_detail(raw, key, default=np.nan):
                try:
                    obj = json.loads(raw)
                    return obj.get(key, default)
                except Exception:
                    return default

            df_trade['fill_ratio'] = df_trade['details_json'].apply(lambda r: _parse_detail(r, 'fill_ratio'))
            df_trade['mode'] = df_trade['details_json'].apply(lambda r: _parse_detail(r, 'mode', ''))
            df_trade = df_trade[_build_trade_mode_mask(df_trade, expected_mode or RUN_MODE)].copy()

        eligible_alpha = df_alpha[(df_alpha['alpha'].abs() >= 0.8) & (df_alpha['vol_z'] >= -1.0)].copy() if not df_alpha.empty else pd.DataFrame()
        open_df = df_trade[df_trade['action'] == 'OPEN'].copy() if not df_trade.empty else pd.DataFrame()
        close_df = df_trade[df_trade['action'] == 'CLOSE'].copy() if not df_trade.empty else pd.DataFrame()

        if not open_df.empty and not eligible_alpha.empty:
            matched = eligible_alpha.merge(open_df[['symbol', 'ts']], on='symbol', how='left', suffixes=('', '_open'))
            matched['matched'] = (matched['ts_open'] - matched['ts']).abs() <= 120
            executed_from_eligible = int(matched['matched'].fillna(False).sum())
        else:
            executed_from_eligible = 0

        partial_fills = 0
        full_fills = 0
        if not open_df.empty:
            fill_ratio = pd.to_numeric(open_df['fill_ratio'], errors='coerce')
            partial_fills = int(((fill_ratio > 0) & (fill_ratio < 0.999)).sum())
            full_fills = int((fill_ratio >= 0.999).sum())

        return {
            'alpha_total': int(len(df_alpha)),
            'alpha_eligible': int(len(eligible_alpha)),
            'open_total': int(len(open_df)),
            'eligible_to_open': int(executed_from_eligible),
            'partial_fills': int(partial_fills),
            'full_fills': int(full_fills),
            'close_total': int(len(close_df)),
            'symbol_signaled': int(df_alpha['symbol'].nunique()) if not df_alpha.empty else 0,
            'symbol_opened': int(open_df['symbol'].nunique()) if not open_df.empty else 0,
        }
    except Exception as e:
        print(f"Error fetching alpha execution funnel: {e}")
        return {
            'alpha_total': 0,
            'alpha_eligible': 0,
            'open_total': 0,
            'eligible_to_open': 0,
            'partial_fills': 0,
            'full_fills': 0,
            'close_total': 0,
            'symbol_signaled': 0,
            'symbol_opened': 0,
        }


@st.cache_data(ttl=20)
def load_intraday_pg_alpha(query_date, symbol_filter="ALL"):
    start_dt = datetime.combine(query_date, dt_time(0, 0, 0))
    start_ts = int(NY_TZ.localize(start_dt).timestamp())
    end_ts = start_ts + 86400
    conn = psycopg2.connect(PG_DB_URL)
    sql = f"""
        SELECT ts, symbol, alpha, iv, price, vol_z
        FROM alpha_logs
        WHERE ts >= {start_ts} AND ts < {end_ts}
    """
    if symbol_filter and symbol_filter != "ALL":
        sql += f" AND symbol = '{symbol_filter}'"
    sql += " ORDER BY ts ASC, symbol ASC"
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


@st.cache_data(ttl=20)
def load_intraday_sqlite_replay_alpha(query_date, symbol_filter="ALL"):
    date_str = pd.Timestamp(query_date).strftime("%Y%m%d")
    db_path = SQLITE_DATA_DIR / f"market_{date_str}.db"
    if not db_path.exists():
        return pd.DataFrame(), str(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = """
        SELECT ts, symbol, alpha, iv, price, vol_z
        FROM alpha_logs
    """
    if symbol_filter and symbol_filter != "ALL":
        sql += f" WHERE symbol = '{symbol_filter}'"
    sql += " ORDER BY ts ASC, symbol ASC"
    df = pd.read_sql(sql, conn)
    conn.close()
    return df, str(db_path)


@st.cache_data(ttl=20)
def load_intraday_alpha_audit_csv(query_date, symbol_filter="ALL"):
    p = Path.home() / "quant_project" / "logs" / "alpha_audit.csv"
    if not p.exists():
        return pd.DataFrame(), str(p)
    df = pd.read_csv(p)
    if df.empty:
        return df, str(p)
    df = df.rename(columns={'timestamp': 'ts'})
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
    df = df.dropna(subset=['ts'])
    start_dt = datetime.combine(query_date, dt_time(0, 0, 0))
    start_ts = int(NY_TZ.localize(start_dt).timestamp())
    end_ts = start_ts + 86400
    df = df[(df['ts'] >= start_ts) & (df['ts'] < end_ts)].copy()
    if symbol_filter and symbol_filter != "ALL":
        df = df[df['symbol'] == symbol_filter].copy()
    keep = [c for c in ['ts', 'symbol', 'alpha', 'price', 'vol_z'] if c in df.columns]
    if 'iv' not in df.columns:
        df['iv'] = np.nan
        keep.append('iv')
    df = df[keep].sort_values(['ts', 'symbol']).reset_index(drop=True)
    return df, str(p)


def build_intraday_alpha_diff(pg_df, replay_df, tolerance=1e-6):
    metrics = {
        'live_rows': 0,
        'replay_rows': 0,
        'matched_rows': 0,
        'coverage_live': 0.0,
        'coverage_replay': 0.0,
        'exact_alpha_ratio': 0.0,
        'within_tol_alpha_ratio': 0.0,
        'alpha_max_diff': np.nan,
        'iv_max_diff': np.nan,
        'price_max_diff': np.nan,
        'vol_z_max_diff': np.nan,
    }
    if pg_df is None or replay_df is None or pg_df.empty or replay_df.empty:
        return metrics, pd.DataFrame(), pd.DataFrame()

    left = pg_df.copy()
    right = replay_df.copy()
    left['ts'] = pd.to_numeric(left['ts'], errors='coerce')
    right['ts'] = pd.to_numeric(right['ts'], errors='coerce')
    left = left.dropna(subset=['ts', 'symbol'])
    right = right.dropna(subset=['ts', 'symbol'])

    merged = left.merge(right, on=['ts', 'symbol'], how='outer', suffixes=('_live', '_replay'), indicator=True)
    matched = merged[merged['_merge'] == 'both'].copy()

    metrics['live_rows'] = int(len(left))
    metrics['replay_rows'] = int(len(right))
    metrics['matched_rows'] = int(len(matched))
    metrics['coverage_live'] = float(len(matched) / len(left)) if len(left) > 0 else 0.0
    metrics['coverage_replay'] = float(len(matched) / len(right)) if len(right) > 0 else 0.0

    for field in ['alpha', 'iv', 'price', 'vol_z']:
        lcol = f'{field}_live'
        rcol = f'{field}_replay'
        if lcol in matched.columns and rcol in matched.columns:
            matched[f'{field}_diff'] = pd.to_numeric(matched[lcol], errors='coerce') - pd.to_numeric(matched[rcol], errors='coerce')
            abs_col = matched[f'{field}_diff'].abs()
            metrics[f'{field}_max_diff'] = float(abs_col.max()) if not abs_col.dropna().empty else np.nan

    if 'alpha_diff' in matched.columns and not matched['alpha_diff'].dropna().empty:
        abs_alpha = matched['alpha_diff'].abs()
        metrics['exact_alpha_ratio'] = float((abs_alpha <= 1e-12).mean())
        metrics['within_tol_alpha_ratio'] = float((abs_alpha <= tolerance).mean())

    if not matched.empty:
        matched['dt_ny'] = pd.to_datetime(matched['ts'], unit='s', utc=True).dt.tz_convert('America/New_York')
        matched['minute_ny'] = matched['dt_ny'].dt.floor('1min')
        rolling = (
            matched.groupby('minute_ny', as_index=False)
            .agg(
                alpha_mae=('alpha_diff', lambda s: np.nanmean(np.abs(s))),
                iv_mae=('iv_diff', lambda s: np.nanmean(np.abs(s))),
                price_mae=('price_diff', lambda s: np.nanmean(np.abs(s))),
                vol_z_mae=('vol_z_diff', lambda s: np.nanmean(np.abs(s))),
                matched_rows=('symbol', 'count'),
            )
            .sort_values('minute_ny')
        )
    else:
        rolling = pd.DataFrame()

    detail_cols = ['ts', 'symbol', '_merge']
    for field in ['alpha', 'iv', 'price', 'vol_z']:
        for suffix in ['live', 'replay']:
            col = f'{field}_{suffix}'
            if col in merged.columns:
                detail_cols.append(col)
        diff_col = f'{field}_diff'
        if diff_col in matched.columns and diff_col not in detail_cols:
            pass
    detail = matched.copy()
    if 'alpha_diff' in detail.columns:
        detail = detail.sort_values(by='alpha_diff', key=lambda s: s.abs(), ascending=False)
    return metrics, rolling, detail


@st.cache_data(ttl=20)
def load_alpha_diff_context(ts_val, symbol, replay_source="SQLite Replay DB"):
    """读取 diff 明细对应时刻的上下文，帮助定位输入侧偏差。"""
    context = {
        'live_alpha': pd.DataFrame(),
        'live_bar': pd.DataFrame(),
        'live_option': pd.DataFrame(),
        'replay_alpha': pd.DataFrame(),
    }
    try:
        conn = psycopg2.connect(PG_DB_URL)
        context['live_alpha'] = pd.read_sql(
            f"""
            SELECT ts, symbol, alpha, iv, price, vol_z
            FROM alpha_logs
            WHERE ts = {int(ts_val)} AND symbol = '{symbol}'
            """,
            conn,
        )
        context['live_bar'] = pd.read_sql(
            f"""
            SELECT ts, symbol, open, high, low, close, volume
            FROM market_bars_1m
            WHERE ts = {int(ts_val)} AND symbol = '{symbol}'
            """,
            conn,
        )
        context['live_option'] = pd.read_sql(
            f"""
            SELECT ts, symbol, buckets_json
            FROM option_snapshots_1m
            WHERE ts = {int(ts_val)} AND symbol = '{symbol}'
            """,
            conn,
        )
        conn.close()
    except Exception as e:
        print(f"Error loading live alpha diff context: {e}")

    try:
        if replay_source == "SQLite Replay DB":
            date_str = pd.to_datetime(ts_val, unit='s', utc=True).tz_convert('America/New_York').strftime("%Y%m%d")
            db_path = SQLITE_DATA_DIR / f"market_{date_str}.db"
            if db_path.exists():
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                context['replay_alpha'] = pd.read_sql(
                    f"""
                    SELECT ts, symbol, alpha, iv, price, vol_z
                    FROM alpha_logs
                    WHERE ts = {int(ts_val)} AND symbol = '{symbol}'
                    """,
                    conn,
                )
                conn.close()
        else:
            p = Path.home() / "quant_project" / "logs" / "alpha_audit.csv"
            if p.exists():
                df = pd.read_csv(p)
                df = df.rename(columns={'timestamp': 'ts'})
                df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
                context['replay_alpha'] = df[(df['ts'] == float(ts_val)) & (df['symbol'] == symbol)].copy()
    except Exception as e:
        print(f"Error loading replay alpha context: {e}")

    return context

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

def _safe_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    std = float(s.std(ddof=0))
    if std < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mean = float(s.mean())
    return ((s - mean) / std).fillna(0.0)

@st.cache_data(ttl=5)
def load_momentum_leaderboard(max_symbols=40):
    """
    从 PG 聚合实时动能榜：
    - 价格动量: 5m/15m 收益
    - 交易热度: vol_z
    - 信号确认: 最新 alpha
    """
    try:
        conn = psycopg2.connect(PG_DB_URL)
        now_ts = int(time.time())
        start_ts = now_ts - 3 * 24 * 3600

        sql_px = f"""
            WITH recent AS (
                SELECT
                    symbol,
                    ts,
                    close,
                    volume,
                    LAG(close, 5) OVER (PARTITION BY symbol ORDER BY ts)  AS close_5m_ago,
                    LAG(close, 15) OVER (PARTITION BY symbol ORDER BY ts) AS close_15m_ago,
                    AVG(volume) OVER (
                        PARTITION BY symbol
                        ORDER BY ts
                        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                    ) AS vol_avg_20
                FROM market_bars_1m
                WHERE ts >= {start_ts}
            ),
            latest AS (
                SELECT DISTINCT ON (symbol)
                    symbol, ts, close, volume, close_5m_ago, close_15m_ago, vol_avg_20
                FROM recent
                ORDER BY symbol, ts DESC
            )
            SELECT *
            FROM latest
            WHERE close > 0
        """
        df_px = pd.read_sql(sql_px, conn)

        sql_alpha = f"""
            SELECT DISTINCT ON (symbol)
                symbol, ts AS alpha_ts, alpha, iv, vol_z
            FROM alpha_logs
            WHERE ts >= {start_ts}
            ORDER BY symbol, ts DESC
        """
        df_alpha = pd.read_sql(sql_alpha, conn)
        conn.close()

        if df_px.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        df = df_px.merge(df_alpha, on='symbol', how='left')
        df['ret_5m'] = np.where(df['close_5m_ago'] > 0, df['close'] / df['close_5m_ago'] - 1.0, np.nan)
        df['ret_15m'] = np.where(df['close_15m_ago'] > 0, df['close'] / df['close_15m_ago'] - 1.0, np.nan)
        df['vol_impulse'] = np.where(df['vol_avg_20'] > 0, df['volume'] / df['vol_avg_20'] - 1.0, np.nan)

        # 截面标准化，组成“动能冠军分”
        z_ret_5m = _safe_zscore(df['ret_5m'])
        z_ret_15m = _safe_zscore(df['ret_15m'])
        z_alpha = _safe_zscore(df.get('alpha', 0.0))
        z_volz = _safe_zscore(df.get('vol_z', 0.0))
        z_impulse = _safe_zscore(df['vol_impulse'])

        df['momentum_score'] = (
            0.40 * z_ret_5m +
            0.25 * z_ret_15m +
            0.20 * z_alpha +
            0.10 * z_volz +
            0.05 * z_impulse
        )

        # 基础质量过滤（避免脏值进入榜单）
        quality = (
            pd.to_numeric(df['close'], errors='coerce').fillna(0) > 0
        ) & (
            pd.to_numeric(df['volume'], errors='coerce').fillna(0) >= 0
        )
        df = df[quality].copy()
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        # 取最强/最弱
        top_n = min(int(max_symbols), max(5, int(len(df) * 0.4)))
        long_df = df.sort_values('momentum_score', ascending=False).head(top_n).copy()
        short_df = df.sort_values('momentum_score', ascending=True).head(top_n).copy()

        breadth = float((pd.to_numeric(df['ret_5m'], errors='coerce') > 0).mean())
        median_ret_5m = float(pd.to_numeric(df['ret_5m'], errors='coerce').median())
        top3_strength = float(long_df['momentum_score'].head(3).mean()) if not long_df.empty else 0.0

        if breadth >= 0.60 and median_ret_5m > 0:
            regime = "RISK-ON"
        elif breadth <= 0.40 and median_ret_5m < 0:
            regime = "RISK-OFF"
        else:
            regime = "CHOP"

        stats = {
            'regime': regime,
            'breadth_up_ratio': breadth,
            'median_ret_5m': median_ret_5m,
            'top3_strength': top3_strength,
            'sample_size': int(len(df)),
            'latest_ts': int(pd.to_numeric(df['ts'], errors='coerce').max()),
        }
        return long_df, short_df, stats
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), {}

def render_momentum_panel():
    long_df, short_df, stats = load_momentum_leaderboard(max_symbols=5)
    st.markdown("### 🏆 动能冠军 / 市场状态")

    if not stats:
        st.info("动能榜暂不可用，等待 market_bars_1m / alpha_logs 更新。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Regime", stats.get('regime', 'N/A'))
    c2.metric("Up Breadth", f"{stats.get('breadth_up_ratio', 0.0):.1%}")
    c3.metric("Median 5m Return", f"{stats.get('median_ret_5m', 0.0):.2%}")
    c4.metric("Top3 Strength", f"{stats.get('top3_strength', 0.0):.2f}")

    left, right = st.columns(2)
    with left:
        st.caption("Top Long Leaders")
        if long_df.empty:
            st.info("暂无多头冠军")
        else:
            show_cols = ['symbol', 'momentum_score', 'ret_5m', 'ret_15m', 'alpha', 'vol_z']
            disp = long_df[show_cols].copy()
            st.dataframe(
                disp.style.format({
                    'momentum_score': '{:.2f}',
                    'ret_5m': '{:.2%}',
                    'ret_15m': '{:.2%}',
                    'alpha': '{:.3f}',
                    'vol_z': '{:.2f}',
                }),
                use_container_width=True,
                hide_index=True
            )
    with right:
        st.caption("Top Short Leaders")
        if short_df.empty:
            st.info("暂无空头冠军")
        else:
            show_cols = ['symbol', 'momentum_score', 'ret_5m', 'ret_15m', 'alpha', 'vol_z']
            disp = short_df[show_cols].copy()
            st.dataframe(
                disp.style.format({
                    'momentum_score': '{:.2f}',
                    'ret_5m': '{:.2%}',
                    'ret_15m': '{:.2%}',
                    'alpha': '{:.3f}',
                    'vol_z': '{:.2f}',
                }),
                use_container_width=True,
                hide_index=True
            )

def render_debug_inspector():
    """读取每日统一 DB 并可视化 debug_slow/fast 模型特征。"""
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
            feat_resolutions = SLOW_FEAT_RESOLUTIONS
        else:
            base_table_name = "debug_fast"
            feat_resolutions = FAST_FEAT_RESOLUTIONS
            
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

            st.markdown(f"### 📊 {selected_sym} Model Features ({table_name})")
            five_min_cols = [
                c for c in df_display.select_dtypes(include=[np.number]).columns
                if feat_resolutions.get(c) == '5min'
            ]
            if five_min_cols:
                st.caption(
                    "这些列配置为 `resolution=5min`，FCS 会把 5min 值广播到对应的 1min 行；"
                    "因此连续几分钟相同是预期现象: "
                    + ", ".join(five_min_cols[:12])
                    + (" ..." if len(five_min_cols) > 12 else "")
                )
            
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
    
    # [NEW] 显示 IBKR Connector 实时连接状态（来自 ibkr_connector_v8 Redis 心跳）
    conn_status = get_ibkr_connector_status(r)
    if conn_status:
        state = str(conn_status.get("state", "UNKNOWN"))
        connected = bool(conn_status.get("connected", False))
        ts_val = float(conn_status.get("ts", 0.0) or 0.0)
        age_sec = max(0.0, time.time() - ts_val) if ts_val > 0 else 9999.0
        host = conn_status.get("host", "127.0.0.1")
        port = conn_status.get("port", IBKR_PORT)
        client_id = conn_status.get("client_id", "N/A")
        note = str(conn_status.get("note", "") or "")
        last_error = str(conn_status.get("last_error", "") or "")
        active_stocks = int(conn_status.get("active_stocks", 0) or 0)
        locked_symbols = int(conn_status.get("locked_symbols", 0) or 0)

        if connected and age_sec <= 5.0:
            st.success(f"IBKR Connector: {state}")
        elif connected:
            st.warning(f"IBKR Connector: {state} (stale {age_sec:.1f}s)")
        else:
            st.error(f"IBKR Connector: {state}")
        st.caption(
            f"{host}:{port} | clientId={client_id} | age={age_sec:.1f}s | "
            f"stocks={active_stocks} | locks={locked_symbols}"
        )
        if note:
            st.caption(f"note: {note}")
        if last_error:
            st.caption(f"last_error: {last_error[:180]}")
    else:
        st.warning("IBKR Connector status: missing heartbeat")
    
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
    if RUN_MODE.startswith("REALTIME"):
        st.markdown("### Live Risk Controls")
        st.caption(f"run_mode: {RUN_MODE}")
        current_trading_enabled, current_trading_meta = get_runtime_trading_enabled(
            default_value=TRADING_ENABLED,
            r=r,
            return_meta=True,
        )
        trade_status = "ARMED" if current_trading_enabled else "DISARMED"
        trade_color = "#EF553B" if current_trading_enabled else "#00CC96"
        st.markdown(
            f"**Live Trading**: <span style='color:{trade_color}'>{trade_status}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"source: {current_trading_meta.get('source', 'config')} | by: {current_trading_meta.get('updated_by') or 'config'}"
        )
        if RUN_MODE != "REALTIME":
            st.info(
                "Current process is running in REALTIME_DRY. You can preconfigure the cap here, "
                "but real-order arming stays blocked until the service itself is started in REALTIME."
            )
        tc1, tc2 = st.columns(2)
        if tc1.button(
            "Arm Trading",
            use_container_width=True,
            disabled=(RUN_MODE != "REALTIME" or not TRADING_ENABLED or current_trading_enabled),
        ):
            st.session_state["runtime_control_pending"] = {
                "action": "set_trading",
                "value": True,
                "message": "This will allow REALTIME OMS to submit real orders. Please confirm carefully.",
            }
            st.rerun()
        if tc2.button("Disarm Trading", use_container_width=True, disabled=(not current_trading_enabled)):
            st.session_state["runtime_control_pending"] = {
                "action": "set_trading",
                "value": False,
                "message": "This will stop REALTIME OMS from sending new real orders and switch execution back to dry-style accounting.",
            }
            st.rerun()
        current_live_cap, current_live_cap_meta = get_runtime_live_trading_capital_limit(
            default_value=LIVE_TRADING_CAPITAL_LIMIT,
            r=r,
            return_meta=True,
        )
        live_cap_input = st.number_input(
            "Live Trading Cap ($)",
            min_value=0.0,
            value=float(current_live_cap or 0.0),
            step=500.0,
            key="live_trading_cap_input",
        )
        lc1, lc2 = st.columns(2)
        if lc1.button("Save Cap", use_container_width=True):
            st.session_state["runtime_control_pending"] = {
                "action": "set_cap",
                "value": float(live_cap_input),
                "message": f"This will update the REALTIME live trading cap to ${float(live_cap_input):,.2f}.",
            }
            st.rerun()
        if lc2.button("Reset Cap", use_container_width=True):
            st.session_state["runtime_control_pending"] = {
                "action": "reset_cap",
                "message": "This will clear the Redis override and restore the config default live trading cap.",
            }
            st.rerun()
        cap_source = current_live_cap_meta.get("source", "config")
        cap_updater = current_live_cap_meta.get("updated_by") or "config"
        st.caption(
            f"current: ${float(current_live_cap or 0.0):,.2f} | source: {cap_source} | by: {cap_updater}"
        )
        st.caption(f"config default: ${float(LIVE_TRADING_CAPITAL_LIMIT or 0.0):,.2f}")
    st.divider()
    st.markdown("### Model Health Guide")
    st.info("IC > 0.05: Healthy\nIC < 0: Inverted\nIC ~ 0: Decay")

_render_runtime_control_dialog()

# --- Top: Topology ---
status = SystemStatus(r, symbol)
st.plotly_chart(draw_topology(status), use_container_width=True)
render_momentum_panel()

if RUN_MODE.startswith("REALTIME"):
    top_live_trading_enabled, top_live_trading_meta = get_runtime_trading_enabled(
        default_value=TRADING_ENABLED,
        r=r,
        return_meta=True,
    )
    top_live_trading_source = top_live_trading_meta.get("source", "config")
    top_live_trading_by = top_live_trading_meta.get("updated_by") or "config"
    if RUN_MODE != "REALTIME":
        st.markdown(
            f"""
            <div style='background:#1A2942; border:1px solid #4C78A8; color:#EEF5FF; padding:12px 16px; border-radius:10px; margin:8px 0 14px 0; font-weight:600;'>
                REALTIME_DRY MODE
                <span style='font-weight:400; margin-left:10px; opacity:0.9;'>
                    live orders blocked by run mode | source: {top_live_trading_source} | by: {top_live_trading_by}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif top_live_trading_enabled:
        st.markdown(
            f"""
            <div style='background:#5A1E1E; border:1px solid #EF553B; color:#FFF4F2; padding:12px 16px; border-radius:10px; margin:8px 0 14px 0; font-weight:600;'>
                REAL ORDERS ENABLED
                <span style='font-weight:400; margin-left:10px; opacity:0.9;'>
                    source: {top_live_trading_source} | by: {top_live_trading_by}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background:#153B2E; border:1px solid #00CC96; color:#EFFFF8; padding:12px 16px; border-radius:10px; margin:8px 0 14px 0; font-weight:600;'>
                LIVE ORDERS BLOCKED
                <span style='font-weight:400; margin-left:10px; opacity:0.9;'>
                    source: {top_live_trading_source} | by: {top_live_trading_by}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
tab1, tab10, tab2, tab11, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab_sh, tab_gates = st.tabs([
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
    "🔄 Live -> Backtest",
    "🩺 Stream Health",
    "🧬 策略门禁",
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

    st.divider()
    st.subheader("Quote Quality")
    qq_summary, qq_bucket_df = load_option_quote_quality(symbol)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Snapshots", f"{qq_summary['snapshots']}")
    q2.metric("Valid Quote Ratio", f"{qq_summary['valid_quote_ratio']:.1%}")
    q3.metric(
        "Median Spread",
        f"{qq_summary['median_spread_pct']:.2%}" if pd.notna(qq_summary['median_spread_pct']) else "N/A"
    )
    q4.metric(
        "Median Depth",
        f"{qq_summary['median_depth']:.0f}" if pd.notna(qq_summary['median_depth']) else "N/A"
    )
    q5.metric(
        "Mid-Last Drift",
        f"{qq_summary['median_mid_last_bps']:.1f} bps" if pd.notna(qq_summary['median_mid_last_bps']) else "N/A",
        delta=f"{qq_summary['stale_gap_count']} stale gaps"
    )
    if not qq_bucket_df.empty:
        qq_disp = qq_bucket_df.copy()
        qq_disp['valid_quote_ratio'] = qq_disp['valid_quote_ratio'].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        qq_disp['median_spread_pct'] = qq_disp['median_spread_pct'].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        qq_disp['median_depth'] = qq_disp['median_depth'].map(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        qq_disp['median_mid_last_bps'] = qq_disp['median_mid_last_bps'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(qq_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No recent option snapshot quality data yet.")

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
    
    def _connect_ibkr_with_fallback(ib, host='127.0.0.1', port=IBKR_PORT, preferred_client_ids=None):
        """
        尽量避开 clientId 冲突：
        1) 先试固定优先 ID（便于排错）
        2) 再试随机区间（避开运行中引擎）
        """
        if preferred_client_ids is None:
            preferred_client_ids = [111, 112, 113, 114, 115]

        tried = []
        candidates = list(preferred_client_ids)
        candidates.extend(random.sample(range(1200, 1500), k=5))

        last_err = None
        for cid in candidates:
            tried.append(cid)
            try:
                if ib.isConnected():
                    ib.disconnect()
                ib.connect(host, port, clientId=int(cid), timeout=4)
                return True, int(cid), ""
            except Exception as e:
                last_err = str(e)
                continue

        return False, None, f"{last_err} (tried clientId={tried})"

    def fetch_ibkr_positions():
        import ib_insync
        ib = ib_insync.IB()
        try:
            ok, client_id, err = _connect_ibkr_with_fallback(ib)
            if not ok:
                return None, None, f"Connect failed: {err}"

            portfolio = ib.portfolio()
            account_values = ib.accountValues()
            return portfolio, account_values, None
        except Exception as e:
            return None, None, str(e)
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass

    def close_ibkr_positions(position_rows):
        """
        批量平仓：单次连接、逐笔下单，返回每笔结果。
        position_rows: [{'Account','Contract','Symbol','Qty','_RawContract'}, ...]
        """
        import ib_insync
        ib = ib_insync.IB()
        results = []
        try:
            ok, client_id, err = _connect_ibkr_with_fallback(ib, preferred_client_ids=[116, 117, 118, 119, 120])
            if not ok:
                return False, [{"ok": False, "contract": "ALL", "message": f"IBKR connect failed: {err}"}]

            for row in position_rows:
                contract = row.get('_RawContract')
                qty = float(row.get('Qty', 0) or 0)
                if contract is None or abs(qty) < 1e-8:
                    results.append({
                        "ok": False,
                        "contract": row.get('Contract', 'UNKNOWN'),
                        "message": "invalid contract or zero qty"
                    })
                    continue

                action = "SELL" if qty > 0 else "BUY"
                order_qty = abs(qty)
                account = row.get('Account', "")
                contract_desc = row.get('Contract', getattr(contract, "localSymbol", "UNKNOWN"))

                try:
                    ib.qualifyContracts(contract)
                    order = ib_insync.MarketOrder(action, order_qty)
                    if account:
                        order.account = account

                    trade = ib.placeOrder(contract, order)
                    ib.sleep(1.2)

                    status = ""
                    if trade is not None and getattr(trade, 'orderStatus', None) is not None:
                        status = str(getattr(trade.orderStatus, 'status', '') or '')
                    ok_status = status in {"Submitted", "PreSubmitted", "Filled", "PendingSubmit", "PendingCancel"}
                    results.append({
                        "ok": bool(ok_status),
                        "contract": contract_desc,
                        "message": f"{action} {order_qty} status={status or 'Unknown'} account={account or 'N/A'}"
                    })
                except Exception as e:
                    results.append({
                        "ok": False,
                        "contract": contract_desc,
                        "message": f"{action} {order_qty} failed: {e}"
                    })
        except Exception as e:
            results.append({"ok": False, "contract": "ALL", "message": str(e)})
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass

        all_ok = all(item.get("ok", False) for item in results) if results else False
        return all_ok, results

    def place_manual_atm_order(symbol, tag, qty, account=""):
        """
        按 symbol + bucket tag（CALL_ATM / PUT_ATM）下手动单。
        合约来源：contract_locks；价格来源：当前快照 mid/bid/ask（仅用于展示与估算）。
        """
        import ib_insync
        qty = int(qty or 0)
        if qty <= 0:
            return False, "qty must be > 0", None

        lock_row = _fetch_locked_contract_row(symbol, tag)
        if not lock_row or not lock_row.get("conId"):
            return False, f"No locked contract found for {symbol} {tag} in contract_locks", None

        quote = _get_bucket_quote_and_contract(symbol, tag) or {}
        ib = ib_insync.IB()
        try:
            ok, client_id, err = _connect_ibkr_with_fallback(ib, preferred_client_ids=[121, 122, 123, 124, 125])
            if not ok:
                return False, f"IBKR connect failed: {err}", None

            contract = ib_insync.Contract()
            contract.conId = int(lock_row["conId"])
            contract.secType = "OPT"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.localSymbol = lock_row.get("localSymbol", "") or ""
            if lock_row.get("tradingClass"):
                contract.tradingClass = lock_row.get("tradingClass")
            if lock_row.get("multiplier"):
                contract.multiplier = str(lock_row.get("multiplier"))

            ib.qualifyContracts(contract)
            order = ib_insync.MarketOrder("BUY", int(qty))
            if account:
                order.account = account
            trade = ib.placeOrder(contract, order)
            ib.sleep(1.2)

            status = ""
            if trade is not None and getattr(trade, 'orderStatus', None) is not None:
                status = str(getattr(trade.orderStatus, 'status', '') or '')
            ok_status = status in {"Submitted", "PreSubmitted", "Filled", "PendingSubmit", "PendingCancel"}
            msg = (
                f"BUY {qty} {symbol} {tag} "
                f"(conId={lock_row['conId']}, local={lock_row.get('localSymbol', '')}) "
                f"status={status or 'Unknown'} account={account or 'N/A'}"
            )
            tracker_seed = {
                "symbol": symbol,
                "tag": tag,
                "qty": int(qty),
                "account": account or "",
                "contract_key": lock_row.get("localSymbol", "") or str(lock_row["conId"]),
                "entry_ref_price": float(quote.get("mid", 0.0) or 0.0),
                "conId": int(lock_row["conId"]),
            }
            return bool(ok_status), msg, tracker_seed
        except Exception as e:
            return False, str(e), None
        finally:
            try:
                if ib.isConnected():
                    ib.disconnect()
            except Exception:
                pass

    def _manual_trackers_init():
        if "manual_trackers" not in st.session_state:
            st.session_state["manual_trackers"] = []

    def _upsert_manual_tracker(seed, stop_loss_pct, take_profit_pct, trailing_stop_pct):
        _manual_trackers_init()
        trackers = st.session_state["manual_trackers"]
        key = f"{seed.get('contract_key')}|{seed.get('account')}"
        now_ts = time.time()
        for t in trackers:
            if t.get("key") == key and t.get("active", True):
                t["qty"] = int(seed.get("qty", t.get("qty", 0)))
                t["entry_ref_price"] = float(seed.get("entry_ref_price", t.get("entry_ref_price", 0.0)))
                t["stop_loss_pct"] = float(stop_loss_pct)
                t["take_profit_pct"] = float(take_profit_pct)
                t["trailing_stop_pct"] = float(trailing_stop_pct)
                t["updated_ts"] = now_ts
                return
        trackers.append({
            "key": key,
            "active": True,
            "symbol": seed.get("symbol", ""),
            "tag": seed.get("tag", ""),
            "account": seed.get("account", ""),
            "qty": int(seed.get("qty", 0)),
            "contract_key": seed.get("contract_key", ""),
            "entry_ref_price": float(seed.get("entry_ref_price", 0.0)),
            "highest_price": float(seed.get("entry_ref_price", 0.0)),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "trailing_stop_pct": float(trailing_stop_pct),
            "created_ts": now_ts,
            "updated_ts": now_ts,
        })

    def _sync_auto_trackers_with_portfolio(portfolio):
        _manual_trackers_init()
        trackers = st.session_state["manual_trackers"]
        if not trackers:
            return

        # 建立 localSymbol -> row 映射
        pos_map = {}
        for item in (portfolio or []):
            c = getattr(item, "contract", None)
            if c is None:
                continue
            local = getattr(c, "localSymbol", "") or ""
            con_id = str(getattr(c, "conId", "") or "")
            key1 = f"{local}|{getattr(item, 'account', '')}"
            key2 = f"{con_id}|{getattr(item, 'account', '')}"
            pos_map[key1] = item
            pos_map[key2] = item

        to_close_rows = []
        for t in trackers:
            if not t.get("active", True):
                continue
            key = t.get("key", "")
            pos_item = pos_map.get(key)
            if pos_item is None:
                # 仓位已经不存在，自动标记完成
                t["active"] = False
                t["updated_ts"] = time.time()
                continue

            qty = float(getattr(pos_item, "position", 0.0) or 0.0)
            if abs(qty) < 1e-8:
                t["active"] = False
                t["updated_ts"] = time.time()
                continue

            mkt_price = float(getattr(pos_item, "marketPrice", 0.0) or 0.0)
            if mkt_price <= 0:
                continue

            entry = float(t.get("entry_ref_price", 0.0) or 0.0)
            if entry <= 0:
                entry = mkt_price
                t["entry_ref_price"] = entry
                t["highest_price"] = max(float(t.get("highest_price", entry) or entry), mkt_price)
                continue

            highest = max(float(t.get("highest_price", entry) or entry), mkt_price)
            t["highest_price"] = highest
            roi = (mkt_price / entry) - 1.0
            drawdown = (mkt_price / highest) - 1.0 if highest > 0 else 0.0

            should_close = False
            reason = ""
            if roi <= -abs(float(t.get("stop_loss_pct", 0.0) or 0.0)):
                should_close = True
                reason = f"SL {roi:.2%}"
            elif roi >= abs(float(t.get("take_profit_pct", 0.0) or 0.0)):
                should_close = True
                reason = f"TP {roi:.2%}"
            elif drawdown <= -abs(float(t.get("trailing_stop_pct", 0.0) or 0.0)):
                should_close = True
                reason = f"TRAIL {drawdown:.2%}"

            if should_close:
                to_close_rows.append({
                    "Account": getattr(pos_item, "account", ""),
                    "Contract": f"{t.get('symbol', '')} {t.get('tag', '')} | {reason}",
                    "Symbol": t.get("symbol", ""),
                    "SecType": "OPT",
                    "Qty": qty,
                    "_RawContract": getattr(pos_item, "contract", None),
                })
                t["active"] = False
                t["updated_ts"] = time.time()
            else:
                t["updated_ts"] = time.time()

        if to_close_rows:
            close_ibkr_positions(to_close_rows)

    def _refresh_positions_cache():
        portfolio, account_vals, err = fetch_ibkr_positions()
        st.session_state['ibkr_pos_error'] = err
        st.session_state['ibkr_portfolio'] = portfolio or []
        st.session_state['ibkr_account_vals'] = account_vals or []
        st.session_state['ibkr_pos_refreshed_at'] = datetime.now().strftime("%H:%M:%S")

    if 'ibkr_portfolio' not in st.session_state:
        st.session_state['ibkr_portfolio'] = []
    if 'ibkr_account_vals' not in st.session_state:
        st.session_state['ibkr_account_vals'] = []
    if 'ibkr_pos_error' not in st.session_state:
        st.session_state['ibkr_pos_error'] = None
    if 'ibkr_pos_refreshed_at' not in st.session_state:
        st.session_state['ibkr_pos_refreshed_at'] = None

    c_refresh_1, c_refresh_2 = st.columns([1, 3])
    if c_refresh_1.button("🔄 Refresh Positions"):
        with st.spinner("Connecting to IBKR..."):
            _refresh_positions_cache()
    if st.session_state.get('ibkr_pos_refreshed_at'):
        c_refresh_2.caption(f"Last refresh: {st.session_state['ibkr_pos_refreshed_at']}")

    if st.session_state.get('ibkr_pos_error'):
        st.error(f"Failed to connect to IBKR: {st.session_state['ibkr_pos_error']}")
    else:
        account_vals = st.session_state.get('ibkr_account_vals', [])
        portfolio = st.session_state.get('ibkr_portfolio', [])

        st.subheader("Account Summary")
        net_liq = "N/A"
        av_funds = "N/A"
        for val in account_vals:
            if getattr(val, 'tag', '') == 'NetLiquidation' and getattr(val, 'currency', '') == 'USD':
                net_liq = f"${float(val.value):,.2f}"
            if getattr(val, 'tag', '') == 'AvailableFunds' and getattr(val, 'currency', '') == 'USD':
                av_funds = f"${float(val.value):,.2f}"
        c1, c2 = st.columns(2)
        c1.metric("Net Liquidation", net_liq)
        c2.metric("Available Funds", av_funds)

        # ================= 手动交易控制台（热度榜关联 ATM） =================
        acct_metrics = _parse_account_metrics(account_vals)
        total_equity = float(acct_metrics.get("net_liq") or 0.0)
        cap = _build_manual_capital_snapshot(total_equity, portfolio, r)

        st.markdown("### ⚙️ Capital Split (Auto vs Manual)")
        sp1, sp2, sp3, sp4 = st.columns(4)
        sp1.metric("Auto Pool Ratio", f"{AUTO_TRADING_CAPITAL_RATIO:.0%}", f"Budget ${cap['auto_budget']:,.0f}")
        sp2.metric("Manual Pool Ratio", f"{MANUAL_TRADING_CAPITAL_RATIO:.0%}", f"Budget ${cap['manual_budget']:,.0f}")
        sp3.metric("Auto Used / Avail", f"${cap['auto_used']:,.0f}", f"Avail ${cap['auto_available']:,.0f}")
        sp4.metric("Manual Used / Avail", f"${cap['manual_used']:,.0f}", f"Avail ${cap['manual_available']:,.0f}")

        st.caption(
            "自动池=总资金×AUTO_TRADING_CAPITAL_RATIO；手动池为剩余部分。"
            "自动占用来自 `oms:live_positions`，手动占用按 IB 持仓总额扣除自动占用近似估算。"
        )

        st.markdown("### 🚀 Momentum -> ATM Manual Trigger")
        long_df, short_df, stats = load_momentum_leaderboard(max_symbols=10)
        if not stats:
            st.info("动能榜暂不可用，等待 market_bars_1m / alpha_logs 更新。")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Regime", stats.get("regime", "N/A"))
            m2.metric("Breadth", f"{stats.get('breadth_up_ratio', 0.0):.1%}")
            m3.metric("Top3 Strength", f"{stats.get('top3_strength', 0.0):.2f}")

            direction = st.radio(
                "Trigger Side",
                options=["LONG (CALL_ATM)", "SHORT (PUT_ATM)"],
                horizontal=True,
                key="manual_trigger_side"
            )
            use_long = direction.startswith("LONG")
            tag = "CALL_ATM" if use_long else "PUT_ATM"
            candidate_df = long_df if use_long else short_df

            if candidate_df.empty:
                st.info("当前方向无候选标的。")
            else:
                candidate_symbols = candidate_df["symbol"].astype(str).tolist()
                sel_symbol = st.selectbox(
                    "Candidate Symbol",
                    options=candidate_symbols,
                    index=0,
                    key="manual_trigger_symbol"
                )
                row = candidate_df[candidate_df["symbol"] == sel_symbol].head(1)
                if not row.empty:
                    row = row.iloc[0]
                    st.caption(
                        f"{sel_symbol} | score={float(row.get('momentum_score', 0.0)):.2f} "
                        f"| ret5m={float(row.get('ret_5m', 0.0)):.2%} "
                        f"| alpha={float(row.get('alpha', 0.0)):.3f} "
                        f"| vol_z={float(row.get('vol_z', 0.0)):.2f}"
                    )

                quote = _get_bucket_quote_and_contract(sel_symbol, tag)
                if not quote:
                    st.warning(f"未找到 {sel_symbol} {tag} 的实时快照，无法估算下单数量。")
                else:
                    q1, q2, q3 = st.columns(3)
                    q1.metric("ATM Mid", f"${quote.get('mid', 0.0):.3f}")
                    q2.metric("Bid/Ask", f"{quote.get('bid', 0.0):.3f}/{quote.get('ask', 0.0):.3f}")
                    q3.metric("Contract", quote.get("contract_text", "") or "N/A")

                    alloc_ratio = st.slider(
                        "Manual order alloc ratio (of manual available)",
                        min_value=0.05, max_value=1.00,
                        value=float(MANUAL_ORDER_ALLOC_RATIO), step=0.05,
                        key="manual_order_alloc_ratio"
                    )
                    est_notional = cap["manual_available"] * float(alloc_ratio)
                    est_cost_per_contract = (quote.get("mid", 0.0) * 100.0) + float(COMMISSION_PER_CONTRACT)
                    est_qty = int(est_notional // est_cost_per_contract) if est_cost_per_contract > 0 else 0

                    qa, qb = st.columns(2)
                    qa.metric("Est. Notional", f"${est_notional:,.0f}")
                    qb.metric("Est. Qty", f"{est_qty}")

                    track_auto_exit = st.checkbox("Enable auto-track exit", value=True, key="manual_auto_track_exit")
                    t1, t2, t3 = st.columns(3)
                    stop_loss_pct = t1.number_input("Stop Loss %", min_value=0.005, max_value=0.50, value=0.08, step=0.005, key="manual_sl")
                    take_profit_pct = t2.number_input("Take Profit %", min_value=0.005, max_value=1.00, value=0.15, step=0.005, key="manual_tp")
                    trailing_stop_pct = t3.number_input("Trailing %", min_value=0.005, max_value=0.50, value=0.06, step=0.005, key="manual_trail")

                    all_accounts = sorted([a for a in {getattr(p, "account", "") for p in portfolio if getattr(p, "account", "")} if a])
                    account_pick = st.selectbox(
                        "Order Account",
                        options=(all_accounts if all_accounts else [""]),
                        index=0,
                        key="manual_trigger_account",
                        help="若为空则不指定 account 字段，由 IB 默认账户处理。"
                    )

                    can_submit = est_qty >= 1 and cap["manual_available"] > 0
                    if st.button(f"🟢 Manual BUY {tag} ({sel_symbol})", disabled=not can_submit, key="manual_buy_submit"):
                        with st.spinner("Submitting manual ATM order..."):
                            ok, msg, tracker_seed = place_manual_atm_order(
                                symbol=sel_symbol,
                                tag=tag,
                                qty=est_qty,
                                account=account_pick or ""
                            )
                            if ok:
                                st.success(msg)
                                if track_auto_exit and tracker_seed:
                                    _upsert_manual_tracker(
                                        tracker_seed,
                                        stop_loss_pct=float(stop_loss_pct),
                                        take_profit_pct=float(take_profit_pct),
                                        trailing_stop_pct=float(trailing_stop_pct)
                                    )
                                    st.info("Auto-track exit enabled for this order.")
                                _refresh_positions_cache()
                                st.rerun()
                            else:
                                st.error(msg)

        # 自动跟踪平仓（仅在 Dashboard 页面存活期间生效）
        if st.session_state.get("manual_auto_track_exit", False):
            _sync_auto_trackers_with_portfolio(portfolio)
            trackers = st.session_state.get("manual_trackers", [])
            active_trackers = [t for t in trackers if t.get("active", True)]
            if active_trackers:
                st.markdown("#### 🤖 Manual Auto-Exit Trackers")
                st.dataframe(
                    pd.DataFrame(active_trackers)[[
                        "symbol", "tag", "account", "qty", "entry_ref_price",
                        "highest_price", "stop_loss_pct", "take_profit_pct",
                        "trailing_stop_pct", "updated_ts"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

        st.subheader("Pending Orders")
        df_pending = fetch_live_pending_orders()
        if df_pending.empty:
            st.info("No active OMS pending orders.")
        else:
            display_cols = [
                "symbol", "intent", "side", "status",
                "target_qty", "filled_qty", "remaining_qty",
                "limit_price", "retry_count", "reserved_cash",
                "slot_reserved", "broker_status", "updated_at_ny", "reason",
            ]
            exist_cols = [c for c in display_cols if c in df_pending.columns]
            st.dataframe(df_pending[exist_cols], use_container_width=True, hide_index=True)

        st.subheader("Active Positions")
        if not portfolio:
            st.info("No active positions found in IBKR account.")
        else:
            pos_data = []
            for item in portfolio:
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
                    "_RawContract": p_contract
                })

            df_pos = pd.DataFrame(pos_data)
            st.dataframe(df_pos.drop(columns=['_RawContract']), use_container_width=True)

            st.subheader("Danger Zone: Close Positions")
            st.warning("These actions send MARKET orders directly to IBKR. Please confirm account and quantity before submitting.")

            all_accounts = sorted([a for a in df_pos['Account'].dropna().unique().tolist() if a])
            selected_accounts = st.multiselect("Filter by account", options=all_accounts, default=all_accounts)
            df_visible = df_pos[df_pos['Account'].isin(selected_accounts)] if selected_accounts else df_pos.iloc[0:0]

            if df_visible.empty:
                st.info("No positions visible under current account filter.")
            else:
                # 一键全平（可见范围）
                st.markdown("**Batch Close (visible rows)**")
                confirm_text = st.text_input(
                    "Type `CLOSE ALL` to enable batch close",
                    key="ibkr_close_all_confirm"
                )
                do_batch_close = st.button(
                    f"🚨 Close ALL Visible Positions ({len(df_visible)})",
                    disabled=(confirm_text.strip().upper() != "CLOSE ALL"),
                    type="primary"
                )
                if do_batch_close:
                    with st.spinner("Submitting batch market orders..."):
                        rows = [row for _, row in df_visible.iterrows()]
                        ok, results = close_ibkr_positions(rows)
                        for item in results:
                            if item.get("ok"):
                                st.success(f"{item.get('contract')}: {item.get('message')}")
                            else:
                                st.error(f"{item.get('contract')}: {item.get('message')}")
                        if ok:
                            st.success("Batch close request submitted. Refreshing positions...")
                        else:
                            st.warning("Some orders failed. Please review errors above and retry.")
                        _refresh_positions_cache()
                        st.rerun()

                st.markdown("---")
                for idx, row in df_visible.iterrows():
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{row['Contract']}** | Acct: `{row['Account']}` | Qty: {row['Qty']}")
                    button_key = f"close_pos_{idx}_{row['Symbol']}_{row['Account']}"
                    if col2.button("Close Position", key=button_key, type="secondary"):
                        with st.spinner(f"Submitting market order for {row['Contract']}..."):
                            ok, results = close_ibkr_positions([row])
                            item = results[0] if results else {"ok": False, "message": "Unknown error"}
                            if item.get("ok"):
                                st.success(f"{item.get('contract')}: {item.get('message')}")
                            else:
                                st.error(f"{item.get('contract')}: {item.get('message')}")
                            _refresh_positions_cache()
                            st.rerun()

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
            current_mode = _normalize_trade_mode(RUN_MODE)
            target_table = 'trade_logs' if current_mode == 'REALTIME' else 'trade_logs_backtest'

            funnel = fetch_alpha_execution_funnel(query_date, target_table, expected_mode=current_mode)
            st.write("##### Alpha -> Execution Funnel")
            f1, f2, f3, f4, f5, f6 = st.columns(6)
            f1.metric("Alpha Total", f"{funnel['alpha_total']}")
            f2.metric("Eligible Alpha", f"{funnel['alpha_eligible']}")
            f3.metric("Matched OPEN", f"{funnel['eligible_to_open']}")
            f4.metric("OPEN Total", f"{funnel['open_total']}", delta=f"{funnel['symbol_opened']} syms")
            f5.metric("Full Fills", f"{funnel['full_fills']}", delta=f"Partial {funnel['partial_fills']}")
            f6.metric("CLOSE Total", f"{funnel['close_total']}", delta=f"Signaled {funnel['symbol_signaled']} syms")
            
            # [🛡️ 双表兜底]
            # 重启/模式切换边界下，少量 OPEN/CLOSE 可能被路由到另一张表。
            # 这里主表优先 + 副表兜底，避免 Recent Trade Logs 出现“只有平仓没有开仓”。
            secondary_table = 'trade_logs_backtest' if target_table == 'trade_logs' else 'trade_logs'
            sql_primary = f"SELECT * FROM {target_table} WHERE ts >= {target_start_ts} AND ts < {target_end_ts}"
            sql_secondary = f"SELECT * FROM {secondary_table} WHERE ts >= {target_start_ts} AND ts < {target_end_ts}"
            try:
                df_primary = pd.read_sql(sql_primary, conn)
            except Exception:
                df_primary = pd.DataFrame()
            try:
                df_secondary = pd.read_sql(sql_secondary, conn)
            except Exception:
                df_secondary = pd.DataFrame()
            if not df_primary.empty:
                df_primary['source_table'] = target_table
            if not df_secondary.empty:
                df_secondary['source_table'] = secondary_table
            if df_primary.empty and df_secondary.empty:
                df_all = pd.DataFrame()
            elif df_secondary.empty:
                df_all = df_primary
            elif df_primary.empty:
                df_all = df_secondary
            else:
                df_all = pd.concat([df_primary, df_secondary], ignore_index=True)
                # 防止双写情况下重复展示同一条记录
                df_all = df_all.drop_duplicates(
                    subset=['ts', 'symbol', 'action', 'qty', 'price'],
                    keep='last'
                )
            df_all = df_all.sort_values(by='ts', ascending=True) if not df_all.empty else df_all
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

                for k in [
                    'pnl', 'roi', 'stock_price', 'entry_stock', 'mode', 'account_cash',
                    'alpha_label_ts', 'alpha_available_ts', 'order_submit_ts', 'fill_ts',
                    'alpha_to_submit_ms', 'submit_to_fill_ms', 'alpha_to_fill_ms',
                    'fill_duration', 'fill_ratio'
                ]:
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
                mode_mask = _build_trade_mode_mask(df_all, current_mode)
                filtered_out = int((~mode_mask).sum())
                df_all = df_all[mode_mask].copy()
                if filtered_out > 0:
                    st.caption(f"Mode filter dropped {filtered_out} rows not matching `{current_mode}`.")

                if df_all.empty:
                    st.info(f"No trades executed on {query_date} for mode {current_mode}.")
                else:

                    for ts_col in ['alpha_label_ts', 'alpha_available_ts', 'order_submit_ts', 'fill_ts']:
                        if ts_col in df_all.columns:
                            df_all[f'{ts_col}_ny'] = pd.to_datetime(
                                df_all[ts_col], unit='s', errors='coerce', utc=True
                            ).dt.tz_convert('America/New_York')

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

                    log_open_positions = dict(open_positions)
                    if current_mode.startswith("REALTIME"):
                        live_open_positions, live_positions_meta = fetch_live_oms_positions(
                            max_age_sec=120.0,
                            allow_stale=False,
                            expected_modes=[RUN_MODE],
                            return_meta=True,
                        )
                        if live_positions_meta.get("fresh"):
                            stale_log_symbols = sorted(set(log_open_positions) - set(live_open_positions))
                            broker_only_symbols = sorted(set(live_open_positions) - set(log_open_positions))
                            for sym, pos in live_open_positions.items():
                                log_pos = log_open_positions.get(sym, {})
                                if not pos.get('tag') and log_pos.get('tag'):
                                    pos['tag'] = log_pos.get('tag', '')
                                if pos.get('cost', 0.0) <= 0 and float(log_pos.get('cost', 0.0) or 0.0) > 0:
                                    pos['cost'] = float(log_pos.get('cost', 0.0) or 0.0)
                            open_positions = live_open_positions
                            skipped_unconfirmed = int(live_positions_meta.get("skipped_unconfirmed", 0) or 0)
                            if stale_log_symbols or broker_only_symbols or skipped_unconfirmed:
                                parts = []
                                if stale_log_symbols:
                                    parts.append(f"dropped stale log-only: {', '.join(stale_log_symbols[:6])}")
                                if broker_only_symbols:
                                    parts.append(f"adopted OMS live-only: {', '.join(broker_only_symbols[:6])}")
                                if skipped_unconfirmed:
                                    parts.append(f"ignored unconfirmed pending: {skipped_unconfirmed}")
                                st.caption("Current positions use `oms:live_positions` in realtime; " + " | ".join(parts))
                        else:
                            st.caption("OMS live position projection is stale/unavailable; falling back to trade-log reconstruction.")

                    # NOTE: Trade Log 上面那张 "Current Positions" + 平仓按钮已迁移到
                    # Intraday Performance（持仓表内嵌"请求平仓"列），此处不再重复渲染。

                    open_df = df_all[df_all['action'] == 'OPEN'].copy()
                    if not open_df.empty:
                        lat_cols = st.columns(3)
                        avg_a2s = pd.to_numeric(open_df['alpha_to_submit_ms'], errors='coerce').dropna()
                        avg_s2f = pd.to_numeric(open_df['submit_to_fill_ms'], errors='coerce').dropna()
                        avg_a2f = pd.to_numeric(open_df['alpha_to_fill_ms'], errors='coerce').dropna()
                        lat_cols[0].metric("Alpha->Submit", f"{avg_a2s.mean():.0f} ms" if not avg_a2s.empty else "N/A")
                        lat_cols[1].metric("Submit->Fill", f"{avg_s2f.mean():.0f} ms" if not avg_s2f.empty else "N/A")
                        lat_cols[2].metric("Alpha->Fill", f"{avg_a2f.mean():.0f} ms" if not avg_a2f.empty else "N/A")

                        chain_cols = st.columns(4)
                        alpha_label_delay = (
                            (pd.to_numeric(open_df['alpha_available_ts'], errors='coerce') - pd.to_numeric(open_df['alpha_label_ts'], errors='coerce')) * 1000.0
                        ).dropna()
                        if not alpha_label_delay.empty:
                            chain_cols[0].metric("Label->Available", f"{alpha_label_delay.mean():.0f} ms")
                        else:
                            chain_cols[0].metric("Label->Available", "N/A")
                        if not avg_a2s.empty:
                            chain_cols[1].metric("Available->Submit", f"{avg_a2s.mean():.0f} ms")
                        else:
                            chain_cols[1].metric("Available->Submit", "N/A")
                        if not avg_s2f.empty:
                            chain_cols[2].metric("Submit->Fill", f"{avg_s2f.mean():.0f} ms")
                        else:
                            chain_cols[2].metric("Submit->Fill", "N/A")
                        if not avg_a2f.empty:
                            chain_cols[3].metric("Available->Fill", f"{avg_a2f.mean():.0f} ms")
                        else:
                            chain_cols[3].metric("Available->Fill", "N/A")

                        st.caption("`Label->Available` 目前用 `alpha_label_ts -> alpha_available_ts` 近似，代表分钟标签落点到信号真正可下单之间的链路耗时。")

                        latency_plot_df = pd.DataFrame({
                            'Available->Submit': avg_a2s,
                            'Submit->Fill': avg_s2f,
                            'Available->Fill': avg_a2f,
                        })
                        latency_long = latency_plot_df.melt(var_name='stage', value_name='latency_ms').dropna()
                        if not latency_long.empty:
                            fig_lat = px.box(
                                latency_long,
                                x='stage',
                                y='latency_ms',
                                color='stage',
                                points='outliers',
                                title="Decision Chain Latency Distribution"
                            )
                            fig_lat.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), template=PLOTLY_THEME, showlegend=False)
                            st.plotly_chart(fig_lat, use_container_width=True)

                        st.markdown("#### Execution Quality Advisor")
                        bucket_minutes = st.selectbox(
                            "Time Bucket",
                            options=[5, 15, 30],
                            index=1,
                            key="trade_log_exec_bucket_minutes",
                        )
                        exec_summary = build_execution_quality_summary(open_df, bucket_minutes=bucket_minutes)
                        if exec_summary.empty:
                            st.info("Execution quality summary unavailable for current sample.")
                        else:
                            advisor_cols = st.columns(4)
                            advisor_cols[0].metric(
                                "Avg Fill Ratio",
                                f"{pd.to_numeric(exec_summary['avg_fill_ratio'], errors='coerce').mean():.2f}"
                                if not exec_summary.empty else "N/A",
                            )
                            advisor_cols[1].metric(
                                "Full Fill Rate",
                                f"{pd.to_numeric(exec_summary['full_fill_rate'], errors='coerce').mean():.1%}"
                                if not exec_summary.empty else "N/A",
                            )
                            p90_ms = pd.to_numeric(exec_summary['submit_to_fill_p90_ms'], errors='coerce').dropna()
                            advisor_cols[2].metric(
                                "Submit->Fill P90",
                                f"{p90_ms.mean():.0f} ms" if not p90_ms.empty else "N/A",
                            )
                            advisor_cols[3].metric(
                                "Buckets",
                                f"{len(exec_summary)}",
                                delta=f"samples {int(exec_summary['open_count'].sum())}",
                            )
                            st.caption(_overall_execution_advice(exec_summary))

                            trend_cols = st.columns(2)
                            exec_summary_plot = exec_summary.copy()
                            exec_summary_plot['bucket_label'] = exec_summary_plot['time_bucket'].dt.strftime('%H:%M')
                            with trend_cols[0]:
                                fig_fill = px.line(
                                    exec_summary_plot,
                                    x='bucket_label',
                                    y=['avg_fill_ratio', 'full_fill_rate'],
                                    markers=True,
                                    title="Fill Quality by Time Bucket",
                                )
                                fig_fill.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), template=PLOTLY_THEME, legend_title_text="")
                                st.plotly_chart(fig_fill, use_container_width=True)
                            with trend_cols[1]:
                                fig_tail = px.line(
                                    exec_summary_plot,
                                    x='bucket_label',
                                    y=['submit_to_fill_p50_ms', 'submit_to_fill_p90_ms'],
                                    markers=True,
                                    title="Submit->Fill Latency Tail",
                                )
                                fig_tail.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), template=PLOTLY_THEME, legend_title_text="")
                                st.plotly_chart(fig_tail, use_container_width=True)

                            scatter_df = open_df.copy()
                            scatter_df['fill_ratio_num'] = pd.to_numeric(scatter_df['fill_ratio'], errors='coerce')
                            scatter_df['submit_to_fill_ms_num'] = pd.to_numeric(scatter_df['submit_to_fill_ms'], errors='coerce')
                            scatter_df = scatter_df.dropna(subset=['fill_ratio_num', 'submit_to_fill_ms_num'])
                            if not scatter_df.empty:
                                fig_exec_scatter = px.scatter(
                                    scatter_df,
                                    x='submit_to_fill_ms_num',
                                    y='fill_ratio_num',
                                    color='tag' if 'tag' in scatter_df.columns else None,
                                    hover_data=['symbol', 'reason'] if {'symbol', 'reason'}.issubset(scatter_df.columns) else None,
                                    title="Fill Ratio vs Submit->Fill Latency",
                                )
                                fig_exec_scatter.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), template=PLOTLY_THEME)
                                st.plotly_chart(fig_exec_scatter, use_container_width=True)

                            exec_display = exec_summary.copy()
                            exec_display['time_bucket'] = exec_display['time_bucket'].dt.strftime('%H:%M')
                            exec_display['avg_fill_ratio'] = pd.to_numeric(exec_display['avg_fill_ratio'], errors='coerce').map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            exec_display['full_fill_rate'] = pd.to_numeric(exec_display['full_fill_rate'], errors='coerce').map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            exec_display['partial_fill_rate'] = pd.to_numeric(exec_display['partial_fill_rate'], errors='coerce').map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                            exec_display['submit_to_fill_p50_ms'] = pd.to_numeric(exec_display['submit_to_fill_p50_ms'], errors='coerce').map(lambda x: f"{x:.0f} ms" if pd.notna(x) else "N/A")
                            exec_display['submit_to_fill_p90_ms'] = pd.to_numeric(exec_display['submit_to_fill_p90_ms'], errors='coerce').map(lambda x: f"{x:.0f} ms" if pd.notna(x) else "N/A")
                            exec_display['alpha_to_fill_p90_ms'] = pd.to_numeric(exec_display['alpha_to_fill_p90_ms'], errors='coerce').map(lambda x: f"{x:.0f} ms" if pd.notna(x) else "N/A")
                            st.dataframe(
                                exec_display.rename(columns={
                                    'time_bucket': 'Bucket',
                                    'open_count': 'OPEN Count',
                                    'avg_fill_ratio': 'Avg Fill Ratio',
                                    'full_fill_rate': 'Full Fill Rate',
                                    'partial_fill_rate': 'Partial Fill Rate',
                                    'submit_to_fill_p50_ms': 'Submit->Fill P50',
                                    'submit_to_fill_p90_ms': 'Submit->Fill P90',
                                    'alpha_to_fill_p90_ms': 'Alpha->Fill P90',
                                    'suggestion': 'Suggestion',
                                }),
                                use_container_width=True,
                            )

                        st.markdown("#### Alpha Timing Audit")
                        timing_cols = [
                            'datetime_ny', 'symbol', 'action', 'qty', 'price', 'tag', 'reason',
                            'alpha_label_ts_ny', 'alpha_available_ts_ny', 'order_submit_ts_ny', 'fill_ts_ny',
                            'alpha_to_submit_ms', 'submit_to_fill_ms', 'alpha_to_fill_ms',
                            'fill_ratio', 'fill_duration'
                        ]
                        exist_cols = [c for c in timing_cols if c in open_df.columns]
                        st.dataframe(open_df[exist_cols], use_container_width=True)

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

                        def _effective_bucket_tag(pos: dict) -> str:
                            tag_val = str(pos.get('tag', '') or '').strip().upper()
                            if tag_val in TAG_TO_INDEX:
                                return tag_val
                            opt_type = str(pos.get('opt_type', '') or '').strip().lower()
                            direction = int(pos.get('position', 0) or 0)
                            if opt_type == 'call' or direction == 1:
                                return 'CALL_ATM'
                            if opt_type == 'put' or direction == -1:
                                return 'PUT_ATM'
                            return ''

                        for sym, pos in open_positions.items():
                            live_price = float(pos.get('last_opt_price', 0.0) or 0.0)
                            if live_price <= 0:
                                live_price = pos['cost']
                            tag = _effective_bucket_tag(pos)
                            found_live = False

                            # 估值口径：统一使用 mid = (bid + ask) / 2，反映真实可平仓估值。
                            # last 在期权上常常 stale（OTM/远月合约几分钟无成交），不能作为可靠估值依据。
                            # 开仓瞬间出现的"半个 spread"浮亏属于真实入场成本，不应被 last 美化。
                            # 回退顺序：mid → bid → ask → last → cost（兜底）。
                            def _pick_live_price(_last: float, _bid: float, _ask: float) -> float:
                                _last = float(_last or 0.0)
                                _bid = float(_bid or 0.0)
                                _ask = float(_ask or 0.0)
                                if _bid > 0.01 and _ask > 0.01 and _ask >= _bid:
                                    return (_bid + _ask) / 2.0
                                if _bid > 0.01:
                                    return _bid
                                if _ask > 0.01:
                                    return _ask
                                if _last > 0.01:
                                    return _last
                                return 0.0

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
                                            picked = _pick_live_price(_last, _bid, _ask)
                                            if picked > 0.01:
                                                live_price = picked
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
                                            picked = _pick_live_price(_last, _bid, _ask)
                                            if picked > 0.01:
                                                live_price = picked
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
                    # 实时/空跑模式下，OMS Redis 账本是唯一权威现金源；Trade Log 只做审计展示，
                    # 不能作为实时现金兜底，否则旧回放/旧空跑日志会把 Remaining Cash 顶高。
                    live_cash, live_cash_source = fetch_live_oms_cash(
                        max_age_sec=120.0,
                        allow_stale=False,
                        return_source=True,
                        expected_modes=[RUN_MODE],
                    )
                    log_cash = None
                    try:
                        if not df_all.empty and 'account_cash' in df_all.columns:
                            _cash_series = pd.to_numeric(df_all['account_cash'], errors='coerce').dropna()
                            if not _cash_series.empty:
                                log_cash = float(_cash_series.iloc[-1])
                    except Exception:
                        log_cash = None
                    display_cash, cash_source = select_remaining_cash(
                        live_cash=live_cash,
                        live_cash_source=live_cash_source,
                        log_cash=log_cash,
                        latest_cash=latest_cash,
                        run_mode=RUN_MODE,
                    )
                    c4.metric("Remaining Cash", f"${display_cash:,.2f}")
                    c4.caption(f"source: {cash_source}")
                    
                    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                    c5.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}W / {losses}L")
                    c6.metric("Open Positions", f"{len(open_positions)} Symbols")

                    broker_net_liq, broker_avail = fetch_live_broker_balance()
                    live_trading_enabled, live_trading_meta = get_runtime_trading_enabled(
                        default_value=TRADING_ENABLED,
                        r=r,
                        return_meta=True,
                    )
                    live_cap, live_cap_meta = get_runtime_live_trading_capital_limit(
                        default_value=LIVE_TRADING_CAPITAL_LIMIT,
                        r=r,
                        return_meta=True,
                    )
                    live_cap = float(live_cap or 0.0)
                    broker_cols = st.columns(4)
                    broker_cols[0].metric(
                        "Broker Balance",
                        f"${broker_net_liq:,.2f}" if broker_net_liq is not None and broker_net_liq > 0 else "N/A",
                    )
                    broker_cols[0].caption(
                        f"available: ${broker_avail:,.2f}" if broker_avail is not None and broker_avail > 0 else "available: N/A"
                    )
                    broker_cols[1].metric(
                        "OMS Cash Ledger",
                        f"${live_cash:,.2f}" if live_cash is not None else f"${display_cash:,.2f}",
                    )
                    broker_cols[1].caption(f"source: {live_cash_source or cash_source}")
                    broker_cols[2].metric(
                        "Live Trading Cap",
                        f"${live_cap:,.2f}" if live_cap > 0 else "Unlimited",
                    )
                    broker_cols[2].caption(
                        f"source: {live_cap_meta.get('source', 'config')}" if live_cap > 0 else "cap disabled"
                    )
                    broker_cols[3].metric(
                        "Live Trading",
                        "ARMED" if live_trading_enabled else "DISARMED",
                    )
                    broker_cols[3].caption(
                        f"source: {live_trading_meta.get('source', 'config')} | by: {live_trading_meta.get('updated_by') or 'config'}"
                    )

                    if paper_details:
                        # 🛑 Realtime 下：在持仓表最右一列内嵌"请求平仓"按钮
                        # （Streamlit st.dataframe 无法嵌按钮 → 用 st.columns 自定义行渲染替代）
                        intraday_close_enabled = False
                        show_close_button = bool(current_mode.startswith("REALTIME") and open_positions)

                        # 用带边框的容器包裹整张表（含密码确认 + 表头 + 数据行），视觉更分明
                        with st.container(border=True):
                            if show_close_button:
                                confirm_intraday_close = st.text_input(
                                    "Type `CLOSE` to enable manual close buttons (Quick Close 列)",
                                    key="intraday_manual_close_confirm",
                                    type="password",
                                )
                                intraday_close_enabled = confirm_intraday_close.strip().upper() == "CLOSE"
                                st.caption(
                                    "右侧 `请求平仓` 按钮通过 OMS 复用执行引擎的 LMT 成交优先 + 重试 + MKT fallback 路径。"
                                )

                            # 列宽对齐 Symbol / Tag / Qty / Cost / Live / MV / Paper PnL / ROI / Action
                            if show_close_button:
                                col_ratios = [1.0, 1.0, 0.7, 0.9, 0.9, 1.1, 1.1, 0.9, 1.3]
                            else:
                                col_ratios = [1.0, 1.0, 0.7, 0.9, 0.9, 1.1, 1.1, 0.9]

                            header_cols = st.columns(col_ratios)
                            header_cols[0].markdown("**Symbol**")
                            header_cols[1].markdown("**Tag**")
                            header_cols[2].markdown("**Qty**")
                            header_cols[3].markdown("**Cost**")
                            header_cols[4].markdown("**Live 🔴**")
                            header_cols[5].markdown("**MV ($)**")
                            header_cols[6].markdown("**Paper PnL ($)**")
                            header_cols[7].markdown("**ROI (%)**")
                            if show_close_button:
                                header_cols[8].markdown("**Action**")

                            st.divider()

                            for row in paper_details:
                                sym = row.get('Symbol', '')
                                row_cols = st.columns(col_ratios)
                                paper_pnl_v = float(row.get('Paper PnL ($)', 0.0) or 0.0)
                                roi_v = float(row.get('ROI (%)', 0.0) or 0.0)
                                pnl_color = "#00CC96" if paper_pnl_v > 0 else ("#EF553B" if paper_pnl_v < 0 else "#AAAAAA")
                                roi_color = "#00CC96" if roi_v > 0 else ("#EF553B" if roi_v < 0 else "#AAAAAA")

                                row_cols[0].markdown(f"**{sym}**")
                                row_cols[1].write(str(row.get('Tag', '') or ''))
                                row_cols[2].write(f"{float(row.get('Qty', 0.0) or 0.0):.0f}")
                                row_cols[3].write(f"${float(row.get('Cost', 0.0) or 0.0):.2f}")
                                row_cols[4].write(f"${float(row.get('Live 🔴', 0.0) or 0.0):.2f}")
                                row_cols[5].write(f"${float(row.get('MarketValue ($)', 0.0) or 0.0):,.0f}")
                                row_cols[6].markdown(
                                    f"<span style='color:{pnl_color}'>${paper_pnl_v:,.2f}</span>",
                                    unsafe_allow_html=True,
                                )
                                row_cols[7].markdown(
                                    f"<span style='color:{roi_color}'>{roi_v:+.2f}%</span>",
                                    unsafe_allow_html=True,
                                )
                                if show_close_button:
                                    pos = open_positions.get(sym, {}) or {}
                                    pos_dir = int(pos.get("position", 0) or 0)
                                    qty_val = float(pos.get("qty", 0.0) or 0.0)
                                    if row_cols[8].button(
                                        "请求平仓",
                                        key=f"intraday_oms_manual_close_{sym}",
                                        disabled=(not intraday_close_enabled) or pos_dir == 0 or qty_val <= 0,
                                        type="primary" if intraday_close_enabled else "secondary",
                                        use_container_width=True,
                                    ):
                                        ref_price = float(pos.get("last_opt_price", 0.0) or pos.get("cost", 0.0) or 0.01)
                                        ok, msg = submit_oms_manual_close(
                                            sym,
                                            pos_dir,
                                            qty_val,
                                            ref_price=ref_price,
                                            stock_price=float(pos.get("stock", 0.0) or 0.0),
                                            contract_id=str(pos.get("contract_id", "") or ""),
                                        )
                                        if ok:
                                            st.success(f"{sym} manual close request sent to OMS: {msg}")
                                            st.rerun()
                                        else:
                                            st.error(f"{sym} manual close request failed: {msg}")

                _live_price_card(open_positions, closed_pnl, wins, losses)
                st.divider()

                # ✅ 5. 渲染最新 100 条流水日志
                st.write("##### 🕒 Recent Trade Logs")
                df_trade = df_all.sort_values(by='ts', ascending=False).head(100).copy()
                df_trade['time'] = pd.to_datetime(df_trade['ts'], unit='s', utc=True).dt.tz_convert(NY_TZ).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # OPEN 日志是审计事件，pnl/roi 应保持为空；当前浮盈单独展示为
                # paper_pnl/paper_roi，避免误读成“开仓瞬间继承了上一笔涨幅”。
                df_trade['paper_pnl'] = np.nan
                df_trade['paper_roi'] = np.nan
                for idx, row in df_trade.iterrows():
                    if row.get('action') == 'OPEN' and row.get('symbol') in open_positions:
                        sym = row['symbol']
                        df_trade.at[idx, 'paper_pnl'] = open_positions[sym].get('paper_pnl', 0.0)
                        df_trade.at[idx, 'paper_roi'] = open_positions[sym].get('roi_pct', 0.0) / 100.0
                        
                display_cols = ['time', 'symbol', 'action', 'qty', 'fill_duration', 'fill_ratio', 'account_cash', 'price', 'paper_pnl', 'paper_roi', 'pnl', 'roi', 'stock_price', 'entry_stock', 'mode', 'alpha', 'reason']
                final_cols = [c for c in display_cols if c in df_trade.columns]
                
                st.dataframe(
                    df_trade[final_cols].style.applymap(
                        lambda x: 'color: #00CC96' if x == 'OPEN' else ('color: #EF553B' if x == 'CLOSE' else ''), subset=['action']
                    ).applymap(
                        lambda x: 'color: #00CC96' if pd.notna(x) and x > 0 else ('color: #EF553B' if pd.notna(x) and x < 0 else ''),
                        subset=[c for c in ['paper_pnl', 'paper_roi', 'pnl', 'roi'] if c in final_cols]
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

    st.subheader("🧪 Intraday Alpha Diff Console")
    st.caption("实时将 PostgreSQL `alpha_logs` 视为 live 真相账本，并与 replay 侧 alpha 进行滚动 diff。适合一边实盘推理，一边跑离线回放做日内核对。")

    cmp_c1, cmp_c2, cmp_c3, cmp_c4 = st.columns([1.1, 1.2, 1.0, 1.0])
    with cmp_c1:
        compare_date = st.date_input("Compare Date", value=datetime.now(NY_TZ).date(), key="alpha_diff_date")
    with cmp_c2:
        compare_source = st.selectbox(
            "Replay Source",
            ["SQLite Replay DB", "Signal Alpha Audit CSV"],
            index=0,
            key="alpha_diff_source"
        )
    with cmp_c3:
        compare_symbol = st.selectbox("Symbol Filter", ["ALL"] + symbols, index=0, key="alpha_diff_symbol")
    with cmp_c4:
        diff_tol = st.number_input("Alpha Tol", min_value=0.0, value=1e-6, step=1e-6, format="%.6f", key="alpha_diff_tol")

    live_alpha_df = load_intraday_pg_alpha(compare_date, compare_symbol)
    replay_hint = ""
    if compare_source == "SQLite Replay DB":
        replay_alpha_df, replay_hint = load_intraday_sqlite_replay_alpha(compare_date, compare_symbol)
    else:
        replay_alpha_df, replay_hint = load_intraday_alpha_audit_csv(compare_date, compare_symbol)

    diff_metrics, diff_rolling, diff_detail = build_intraday_alpha_diff(live_alpha_df, replay_alpha_df, tolerance=float(diff_tol))

    alert_msgs = []
    if diff_metrics['matched_rows'] == 0 and (diff_metrics['live_rows'] > 0 or diff_metrics['replay_rows'] > 0):
        alert_msgs.append("No overlapping live/replay alpha rows.")
    if diff_metrics['coverage_live'] < 0.95 and diff_metrics['live_rows'] > 0:
        alert_msgs.append(f"Live coverage dropped to {diff_metrics['coverage_live']:.1%}.")
    if pd.notna(diff_metrics['alpha_max_diff']) and diff_metrics['alpha_max_diff'] > max(float(diff_tol) * 10.0, 0.05):
        alert_msgs.append(f"Alpha max diff too large: {diff_metrics['alpha_max_diff']:.6f}.")
    if pd.notna(diff_metrics['price_max_diff']) and diff_metrics['price_max_diff'] > 0.01:
        alert_msgs.append(f"Price max diff too large: {diff_metrics['price_max_diff']:.6f}.")
    if pd.notna(diff_metrics['iv_max_diff']) and diff_metrics['iv_max_diff'] > 0.01:
        alert_msgs.append(f"IV max diff too large: {diff_metrics['iv_max_diff']:.6f}.")

    if alert_msgs:
        st.error(" | ".join(alert_msgs))
    else:
        st.success("Intraday live/replay alpha diff is within current alert thresholds.")

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Live Rows", f"{diff_metrics['live_rows']}")
    d2.metric("Replay Rows", f"{diff_metrics['replay_rows']}")
    d3.metric("Matched", f"{diff_metrics['matched_rows']}", delta=f"Live {diff_metrics['coverage_live']:.1%}")
    d4.metric("Replay Coverage", f"{diff_metrics['coverage_replay']:.1%}")
    d5.metric("Exact Alpha", f"{diff_metrics['exact_alpha_ratio']:.1%}")
    d6.metric("Within Tol", f"{diff_metrics['within_tol_alpha_ratio']:.1%}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alpha Max Diff", f"{diff_metrics['alpha_max_diff']:.6f}" if pd.notna(diff_metrics['alpha_max_diff']) else "N/A")
    m2.metric("IV Max Diff", f"{diff_metrics['iv_max_diff']:.6f}" if pd.notna(diff_metrics['iv_max_diff']) else "N/A")
    m3.metric("Price Max Diff", f"{diff_metrics['price_max_diff']:.6f}" if pd.notna(diff_metrics['price_max_diff']) else "N/A")
    m4.metric("Vol_Z Max Diff", f"{diff_metrics['vol_z_max_diff']:.6f}" if pd.notna(diff_metrics['vol_z_max_diff']) else "N/A")

    if replay_hint:
        st.caption(f"Replay source path: `{replay_hint}`")

    if not diff_rolling.empty:
        fig_diff = make_subplots(specs=[[{"secondary_y": True}]])
        fig_diff.add_trace(
            go.Scatter(
                x=diff_rolling['minute_ny'],
                y=diff_rolling['alpha_mae'],
                mode='lines+markers',
                name='Alpha MAE',
                line=dict(color='#ff7f0e', width=2)
            ),
            secondary_y=False
        )
        fig_diff.add_trace(
            go.Scatter(
                x=diff_rolling['minute_ny'],
                y=diff_rolling['matched_rows'],
                mode='lines',
                name='Matched Rows',
                line=dict(color='#1f77b4', width=1, dash='dot')
            ),
            secondary_y=True
        )
        fig_diff.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            template=PLOTLY_THEME,
            title="Intraday Rolling Alpha Diff"
        )
        fig_diff.update_yaxes(title_text="Alpha MAE", secondary_y=False)
        fig_diff.update_yaxes(title_text="Matched Rows", secondary_y=True)
        st.plotly_chart(fig_diff, use_container_width=True)

        diff_long = diff_rolling.melt(
            id_vars=['minute_ny'],
            value_vars=['alpha_mae', 'iv_mae', 'price_mae', 'vol_z_mae'],
            var_name='field',
            value_name='mae'
        )
        fig_fields = px.line(
            diff_long,
            x='minute_ny',
            y='mae',
            color='field',
            title="Field-Level Rolling MAE"
        )
        fig_fields.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), template=PLOTLY_THEME)
        st.plotly_chart(fig_fields, use_container_width=True)
    else:
        st.info("No overlapping live/replay alpha rows yet for the selected source/date.")

    if not diff_detail.empty:
        st.markdown("#### Top Mismatches")
        preview = diff_detail.copy().head(100)
        preview['time_ny'] = pd.to_datetime(preview['ts'], unit='s', utc=True).dt.tz_convert('America/New_York').dt.strftime('%H:%M:%S')
        show_cols = [
            'time_ny', 'symbol',
            'alpha_live', 'alpha_replay', 'alpha_diff',
            'iv_live', 'iv_replay', 'iv_diff',
            'price_live', 'price_replay', 'price_diff',
            'vol_z_live', 'vol_z_replay', 'vol_z_diff'
        ]
        final_cols = [c for c in show_cols if c in preview.columns]
        st.dataframe(preview[final_cols], use_container_width=True)

        inspect_df = preview.copy().head(20).reset_index(drop=True)
        inspect_df['label'] = inspect_df.apply(
            lambda r: f"{pd.to_datetime(r['ts'], unit='s', utc=True).tz_convert('America/New_York').strftime('%H:%M:%S')} | {r['symbol']} | alpha_diff={r.get('alpha_diff', np.nan):.6f}",
            axis=1
        )
        selected_label = st.selectbox("Inspect mismatch context", inspect_df['label'].tolist(), key="alpha_diff_inspect")
        selected_row = inspect_df[inspect_df['label'] == selected_label].iloc[0]
        ctx = load_alpha_diff_context(int(selected_row['ts']), selected_row['symbol'], compare_source)

        st.markdown("#### Raw Context")
        c_live1, c_live2 = st.columns(2)
        with c_live1:
            st.caption("Live alpha row")
            if not ctx['live_alpha'].empty:
                st.dataframe(ctx['live_alpha'], use_container_width=True)
            else:
                st.info("No live alpha row.")
            st.caption("Live market bar")
            if not ctx['live_bar'].empty:
                st.dataframe(ctx['live_bar'], use_container_width=True)
            else:
                st.info("No live market bar.")
        with c_live2:
            st.caption(f"Replay alpha row ({compare_source})")
            if not ctx['replay_alpha'].empty:
                st.dataframe(ctx['replay_alpha'], use_container_width=True)
            else:
                st.info("No replay alpha row.")
            st.caption("Live option snapshot raw JSON")
            if not ctx['live_option'].empty:
                opt_df = ctx['live_option'].copy()
                if 'buckets_json' in opt_df.columns:
                    opt_df['buckets_json'] = opt_df['buckets_json'].astype(str).str.slice(0, 500)
                st.dataframe(opt_df, use_container_width=True)
            else:
                st.info("No live option snapshot row.")
    else:
        st.info("No mismatch detail to show yet.")

    st.divider()
    
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

# ============================================================
# 🩺 Stream Health —— 消费状态/积压/PG 残留 可视化诊断
# ============================================================
_STREAM_HEALTH_TARGETS = [
    (STREAM_ORCH_SIGNAL,  GROUP_OMS),   # SE → OMS
    (STREAM_FUSED_MARKET, GROUP_OMS),   # IBKR → OMS (exits)
    (STREAM_FUSED_MARKET, GROUP_ORCH),  # IBKR → SE (features)
    (STREAM_INFERENCE,    GROUP_ORCH),  # FCS → SE
]


def _sh_decode(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    return v


@st.cache_data(ttl=3)
def _sh_inspect_redis_groups():
    """拉取每个 (stream, group) 的消费健康度。返回 list[dict]。"""
    rds = get_redis_client()
    rows = []
    if rds is None:
        return rows, "Redis 连接失败"
    for stream, group in _STREAM_HEALTH_TARGETS:
        row = {
            'stream': stream, 'group': group,
            'stream_exists': False, 'group_exists': False,
            'last_delivered_id': None, 'pending': None, 'lag': None,
            'stream_last_id': None, 'stream_len': None,
        }
        try:
            s_info = rds.xinfo_stream(stream)
            row['stream_exists']   = True
            row['stream_last_id']  = _sh_decode(s_info.get(b'last-generated-id') or s_info.get('last-generated-id'))
            row['stream_len']      = s_info.get(b'length') or s_info.get('length')
        except Exception:
            rows.append(row)
            continue
        try:
            for g in rds.xinfo_groups(stream):
                name = _sh_decode(g.get(b'name') or g.get('name'))
                if name != group:
                    continue
                row['group_exists']       = True
                row['last_delivered_id']  = _sh_decode(g.get(b'last-delivered-id') or g.get('last-delivered-id'))
                row['pending']            = g.get(b'pending') or g.get('pending')
                row['lag']                = g.get(b'lag')     or g.get('lag')
                break
        except Exception:
            pass
        rows.append(row)
    return rows, None


@st.cache_data(ttl=5)
def _sh_inspect_pg_state():
    """查看 PG symbol_state 的跨天残留 / GLOBAL_STATE。"""
    summary = {
        'table_exists': False, 'total_rows': 0,
        'today_symbol_rows': 0, 'stale_symbol_rows': [],
        'global_state_row': None, 'connected': False, 'error': None,
        'namespace': OMS_STATE_NAMESPACE,
    }
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()
        summary['connected'] = True
    except Exception as e:
        summary['error'] = str(e)
        return summary
    try:
        c.execute("SELECT to_regclass('public.symbol_state')")
        if c.fetchone()[0] is None:
            conn.close()
            return summary
        summary['table_exists'] = True
        c.execute("SELECT COUNT(*) FROM symbol_state WHERE namespace = %s", (OMS_STATE_NAMESPACE,))
        summary['total_rows'] = c.fetchone()[0]
        now_ny_date = datetime.now(NY_TZ).date()
        c.execute(
            "SELECT symbol, updated_at FROM symbol_state WHERE namespace = %s ORDER BY updated_at DESC",
            (OMS_STATE_NAMESPACE,),
        )
        for sym, updated_at in c.fetchall():
            try:
                ts_dt = datetime.fromtimestamp(float(updated_at), NY_TZ).date()
            except Exception:
                ts_dt = None
            is_today = (ts_dt == now_ny_date)
            if sym == '_GLOBAL_STATE_':
                summary['global_state_row'] = {
                    'updated_at': updated_at, 'ts_ny': str(ts_dt), 'is_today': is_today,
                }
            else:
                if is_today:
                    summary['today_symbol_rows'] += 1
                else:
                    summary['stale_symbol_rows'].append((sym, str(ts_dt)))
        conn.close()
    except Exception as e:
        summary['error'] = str(e)
        try: conn.close()
        except Exception: pass
    return summary


@st.cache_data(ttl=5)
def _sh_inspect_fcs_state():
    """查看 fcs_state_snapshot 各 namespace 的健康度。"""
    summary = {
        'connected': False, 'table_exists': False,
        'namespaces': [], 'error': None,
    }
    try:
        conn = psycopg2.connect(PG_DB_URL); c = conn.cursor()
        summary['connected'] = True
    except Exception as e:
        summary['error'] = str(e)
        return summary
    try:
        c.execute("SELECT to_regclass('public.fcs_state_snapshot')")
        if c.fetchone()[0] is None:
            conn.close()
            return summary
        summary['table_exists'] = True
        c.execute("""
            SELECT namespace,
                   COUNT(*)                       AS rows,
                   COUNT(*) FILTER (WHERE symbol <> '_META_') AS symbol_count,
                   MAX(schema_version)            AS schema_version,
                   MAX(updated_at)                AS latest_saved
            FROM fcs_state_snapshot
            GROUP BY namespace
            ORDER BY MAX(updated_at) DESC
        """)
        ns_rows = c.fetchall()

        now_ny_date = datetime.now(NY_TZ).date()
        for ns, rows, syms, ver, latest in ns_rows:
            is_today = False
            try:
                if latest is not None:
                    is_today = (datetime.fromtimestamp(float(latest), NY_TZ).date() == now_ny_date)
            except Exception: pass
            meta_row = None
            try:
                c.execute(
                    "SELECT payload FROM fcs_state_snapshot WHERE namespace=%s AND symbol='_META_'",
                    (ns,),
                )
                row = c.fetchone()
                if row and row[0] is not None:
                    import zlib as _zlib, pickle as _pk
                    meta_row = _pk.loads(_zlib.decompress(bytes(row[0])))
            except Exception:
                pass
            summary['namespaces'].append({
                'namespace':      ns,
                'rows':           int(rows or 0),
                'symbol_count':   int(syms or 0),
                'schema_version': int(ver) if ver is not None else None,
                'latest_saved':   latest,
                'is_today':       is_today,
                'meta_row':       meta_row,
            })
        conn.close()
    except Exception as e:
        summary['error'] = str(e)
        try: conn.close()
        except Exception: pass
    return summary


def _sh_group_status(row):
    """返回 (level, msg) — level ∈ {'ok','warn','err','gray'}"""
    if not row['stream_exists']:
        return 'gray', 'stream 不存在'
    if not row['group_exists']:
        return 'warn', '组未创建'
    ldi = row.get('last_delivered_id') or ''
    if ldi.startswith('0-0'):
        return 'err', '从未消费 (重启将全量回放!)'
    try:
        lag = int(row.get('lag') or 0)
    except Exception:
        lag = 0
    if lag >= 500:
        return 'err', f'严重积压 lag={lag}'
    if lag >= 50:
        return 'warn', f'积压 lag={lag}'
    return 'ok', 'healthy'


def _sh_run_action(action: str):
    """直接在 Dashboard 里执行清理，对应脚本里的三个开关。"""
    rds = get_redis_client()
    msgs = []
    if action in ('drop', 'all') and rds is not None:
        for stream, group in _STREAM_HEALTH_TARGETS:
            try:
                # 只对存在的组做 SETID；不存在就跳过
                exists = False
                try:
                    for g in rds.xinfo_groups(stream):
                        if _sh_decode(g.get(b'name') or g.get('name')) == group:
                            exists = True; break
                except Exception: pass
                if not exists:
                    msgs.append(f"⏭ {stream}/{group} 组不存在，跳过")
                    continue
                rds.xgroup_setid(stream, group, '$')
                msgs.append(f"✅ XGROUP SETID {stream} {group} $")
            except Exception as e:
                msgs.append(f"❌ {stream}/{group} SETID 失败: {e}")
    if action in ('reset_gs', 'all'):
        try:
            conn = psycopg2.connect(PG_DB_URL); c = conn.cursor()
            c.execute("SELECT to_regclass('public.symbol_state')")
            if c.fetchone()[0] is not None:
                c.execute(
                    "DELETE FROM symbol_state WHERE namespace = %s AND symbol = %s",
                    (OMS_STATE_NAMESPACE, '_GLOBAL_STATE_'),
                )
                conn.commit()
                msgs.append(f"✅ 已删除 _GLOBAL_STATE_ ({c.rowcount} 行)")
            else:
                msgs.append("⏭ symbol_state 表不存在，跳过")
            conn.close()
        except Exception as e:
            msgs.append(f"❌ 删除 _GLOBAL_STATE_ 失败: {e}")
    if action in ('purge', 'all'):
        try:
            conn = psycopg2.connect(PG_DB_URL); c = conn.cursor()
            c.execute("SELECT to_regclass('public.symbol_state')")
            if c.fetchone()[0] is not None:
                now_ny_date = datetime.now(NY_TZ).date()
                start_of_day = NY_TZ.localize(datetime.combine(now_ny_date, dt_time.min)).timestamp()
                c.execute(
                    "DELETE FROM symbol_state WHERE namespace = %s AND symbol <> %s AND updated_at < %s",
                    (OMS_STATE_NAMESPACE, '_GLOBAL_STATE_', start_of_day),
                )
                conn.commit()
                msgs.append(f"✅ 已清除跨天 symbol_state ({c.rowcount} 行)")
            else:
                msgs.append("⏭ symbol_state 表不存在，跳过")
            conn.close()
        except Exception as e:
            msgs.append(f"❌ 清除跨天 symbol_state 失败: {e}")
    # 清缓存让下一次渲染读到真实状态
    try:
        _sh_inspect_redis_groups.clear()
        _sh_inspect_pg_state.clear()
    except Exception:
        pass
    return msgs


with tab_sh:
    st.header("🩺 Stream Health — 数据消费诊断")
    st.caption(
        "本页展示 OMS / SE 当前的 Redis Stream 消费状态 & PG 状态残留，"
        "可用于盘前体检 / 进程重启后快速确认是否需要清理积压。"
        "此面板缓存 3 秒，可能与 CLI 有短暂滞后。"
    )

    # --- 1. 拉数据 ---
    redis_rows, redis_err = _sh_inspect_redis_groups()
    pg_summary = _sh_inspect_pg_state()

    if redis_err:
        st.error(redis_err)

    # --- 2. 汇总指标 ---
    total_lag = 0
    critical_rows = 0
    for row in redis_rows:
        level, _ = _sh_group_status(row)
        try:
            total_lag += int(row.get('lag') or 0)
        except Exception: pass
        if level == 'err':
            critical_rows += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总积压 lag", f"{total_lag}", delta=("严重" if critical_rows else None), delta_color="inverse" if critical_rows else "off")
    c2.metric("危险分组", f"{critical_rows}/{len(redis_rows)}")
    stale_n = len(pg_summary.get('stale_symbol_rows') or [])
    c3.metric("PG 跨天残留", stale_n, delta=("需清理" if stale_n else None), delta_color="inverse" if stale_n else "off")
    gs_row = pg_summary.get('global_state_row')
    if gs_row is None:
        c4.metric("_GLOBAL_STATE_", "无", delta="fresh", delta_color="off")
    elif gs_row.get('is_today'):
        c4.metric("_GLOBAL_STATE_", "当日有效", delta_color="off")
    else:
        c4.metric("_GLOBAL_STATE_", "跨天残留", delta="需清理", delta_color="inverse")

    # --- 3. Redis group 详表 ---
    st.subheader("🔗 Redis Consumer Groups")
    color_map = {'ok': '🟢', 'warn': '🟡', 'err': '🔴', 'gray': '⚪'}
    rows_view = []
    for row in redis_rows:
        level, msg = _sh_group_status(row)
        rows_view.append({
            '状态': f"{color_map.get(level, '⚪')} {msg}",
            'Stream': row['stream'],
            'Group':  row['group'],
            'last-delivered-id': row['last_delivered_id'] or '-',
            'pending': row['pending'] if row['pending'] is not None else '-',
            'lag': row['lag'] if row['lag'] is not None else '-',
            'stream last-id': row['stream_last_id'] or '-',
            'stream length': row['stream_len'] if row['stream_len'] is not None else '-',
        })
    df_redis = pd.DataFrame(rows_view)
    st.dataframe(df_redis, use_container_width=True, hide_index=True)

    # --- 4. PG 详情 ---
    st.subheader("🐘 PostgreSQL `symbol_state`")
    if not pg_summary.get('connected'):
        st.error(f"PG 连接失败: {pg_summary.get('error')}")
    elif not pg_summary.get('table_exists'):
        st.info("symbol_state 表尚未创建 (首次启动即自动创建)")
    else:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("总行数",    pg_summary['total_rows'])
        cc2.metric("当日 symbol 行", pg_summary['today_symbol_rows'])
        cc3.metric("跨天残留 symbol 行", stale_n)
        if gs_row:
            if gs_row['is_today']:
                st.success(f"_GLOBAL_STATE_ 当日有效, ts_ny={gs_row['ts_ny']} updated_at={gs_row['updated_at']}")
            else:
                st.error(f"⚠️ _GLOBAL_STATE_ 是跨天残留, ts_ny={gs_row['ts_ny']} — 建议清理，否则会污染 mock_cash")
        else:
            st.info("_GLOBAL_STATE_ 不存在 (fresh start)")
        if stale_n:
            with st.expander(f"跨天残留 symbol 预览 ({stale_n} 条)"):
                st.dataframe(
                    pd.DataFrame(pg_summary['stale_symbol_rows'], columns=['symbol', 'ts_ny']),
                    use_container_width=True, hide_index=True,
                )

    # --- 4.5 FCS Snapshot Health (L2 PG-backed) ---
    st.subheader("📦 FCS State Snapshot (`fcs_state_snapshot`)")
    fcs_summary = _sh_inspect_fcs_state()
    if not fcs_summary.get('connected'):
        st.error(f"PG 连接失败: {fcs_summary.get('error')}")
    elif not fcs_summary.get('table_exists'):
        st.info("fcs_state_snapshot 表尚未创建 (FCS 首次启动时自动创建)")
    elif not fcs_summary.get('namespaces'):
        st.info("FCS 暂无快照 (冷启动 / Deep Warmup 中)")
    else:
        rows_view = []
        for ns_row in fcs_summary['namespaces']:
            saved = ns_row.get('latest_saved')
            age_h = None
            try:
                if saved is not None:
                    age_h = (time.time() - float(saved)) / 3600.0
            except Exception: pass
            saved_str = "-"
            if saved is not None:
                try:
                    saved_str = datetime.fromtimestamp(float(saved), NY_TZ).strftime('%Y-%m-%d %H:%M:%S')
                except Exception: pass
            rows_view.append({
                'Namespace':    ns_row['namespace'],
                'Rows':         ns_row['rows'],
                'Symbols':      ns_row['symbol_count'],
                'Schema':       ns_row['schema_version'] if ns_row['schema_version'] is not None else '-',
                'Latest save':  saved_str,
                'Age (h)':      f"{age_h:.2f}" if age_h is not None else '-',
                'Same day':     "✅" if ns_row.get('is_today') else "🕰",
            })
        st.dataframe(pd.DataFrame(rows_view), use_container_width=True, hide_index=True)
        with st.expander("原始 _META_ 行 (逐 namespace)"):
            for ns_row in fcs_summary['namespaces']:
                st.write(f"**{ns_row['namespace']}**: {ns_row.get('meta_row') or '(无)'}")

    # --- 5. 清理操作 (二次确认) ---
    st.markdown("---")
    st.subheader("🛠 清理操作 (Danger Zone)")
    st.caption(
        "和命令行脚本 `production/scripts/oms_state_recovery.py` 等价，"
        "建议盘前/重启后执行。每个按钮需勾选下方确认框才会生效。"
    )
    confirm = st.checkbox("我已确认要修改 Redis 和 PostgreSQL", key='sh_confirm')
    ab1, ab2, ab3, ab4 = st.columns(4)
    if ab1.button("🧹 Drop Backlog", disabled=not confirm, use_container_width=True, key='sh_drop'):
        for m in _sh_run_action('drop'):
            st.write(m)
        st.rerun()
    if ab2.button("🗑 Reset _GLOBAL_STATE_", disabled=not confirm, use_container_width=True, key='sh_reset_gs'):
        for m in _sh_run_action('reset_gs'):
            st.write(m)
        st.rerun()
    if ab3.button("🧽 Purge Stale", disabled=not confirm, use_container_width=True, key='sh_purge'):
        for m in _sh_run_action('purge'):
            st.write(m)
        st.rerun()
    if ab4.button("🚀 Run All", disabled=not confirm, use_container_width=True, key='sh_all'):
        for m in _sh_run_action('all'):
            st.write(m)
        st.rerun()

    st.markdown("**等价 CLI**")
    st.code("python production/scripts/oms_state_recovery.py --all --yes", language='bash')


# ======================================================================
# === Tab 🧬 策略门禁 (Strategy Gates) ===
# 数据源 (全部只读, 不会 touch 任何业务状态):
#   meta:global_gates        - SE 每 5s 刷新的全局门禁聚合快照 (session/CB/positions/exposure)
#   meta:circuit_breaker     - OMS 写的熔断状态 (streak / cb_until)
#   meta:symbol_cooldowns    - OMS 写的 per-symbol cooldown_until
#   meta:gate_trace:{sym}    - SE 每次 decide_entry/check_exit 后写的决策链 (节流)
#   meta:gate_counter:{YYYYMMDD} - SE 拦截累计计数, 按 NY 日
#   meta:strategy_config     - SE 启动广播的 strategy_config0 参数快照
# ======================================================================
with tab_gates:
    st.header("🧬 策略门禁 (V0) — 规则 & 实时命中状态")
    st.caption(
        "每条规则对应 `strategy_core_v0._check_*` 里的一个 early-return。"
        "绿色=放行 / 红色=拦截 / 灰色=未评估或已禁用。"
        "所有数据经 SE 节流写入 Redis, Dashboard 只读, 不会影响策略进程。"
    )

    # ---------- A. 全局门禁横条 ----------
    try:
        gg = r.hgetall("meta:global_gates") or {}
        def _g(k, d=''):
            v = gg.get(k.encode() if isinstance(k, str) else k) or gg.get(k)
            if isinstance(v, bytes):
                v = v.decode('utf-8', errors='ignore')
            return v or d
    except Exception as _e:
        gg = {}
        _g = lambda k, d='': d
        st.warning(f"meta:global_gates 读取失败: {_e}")

    gcols = st.columns(5)
    session = _g('session', 'unknown')
    session_emoji = {
        'pre_open': '🌙 未开盘',
        'entry_open': '🟢 可开仓',
        'no_entry': '🟡 禁入',
        'close_forced': '🔴 强平时段',
    }.get(session, f"❓ {session}")
    gcols[0].metric(
        "交易时段",
        session_emoji,
        f"开={_g('session_open_at')} 禁={_g('session_no_entry_at')} 平={_g('session_close_at')}"
    )
    cb_active = _g('cb_active', '0') == '1'
    streak = _g('consecutive_losses', '0')
    cb_th = _g('cb_threshold', '3')
    cb_badge = "🔥 熔断 ON" if cb_active else "✅ 熔断 OFF"
    gcols[1].metric("G1 熔断 (CB)", cb_badge, f"连败 {streak}/{cb_th}")
    gcols[2].metric(
        "G2 持仓占用",
        f"{_g('positions_used', '0')} / {_g('positions_limit', '-')}"
    )
    try:
        exp_used = float(_g('exposure_used', '0') or 0)
        exp_limit = float(_g('exposure_limit', '0') or 0)
        gcols[3].metric(
            "G5 敞口",
            f"{exp_used:.1%}",
            f"limit={exp_limit:.0%}"
        )
    except Exception:
        gcols[3].metric("G5 敞口", "-", "")
    gcols[4].metric("数据时间", _g('now_ny', '-'), f"upd={_g('updated_at', '')[:10]}")

    # per-symbol cooldown 标签条
    try:
        cd_map = r.hgetall("meta:symbol_cooldowns") or {}
        cd_items = []
        _now = time.time()
        for k, v in cd_map.items():
            sym = k.decode('utf-8') if isinstance(k, bytes) else str(k)
            try:
                cd_ts = float(v.decode('utf-8') if isinstance(v, bytes) else v)
            except Exception:
                continue
            if cd_ts > _now:
                cd_items.append(f"`{sym}` ({(cd_ts - _now) / 60:.0f}m)")
        if cd_items:
            st.markdown("⏳ **单标的冷却中**: " + " · ".join(cd_items))
        else:
            st.markdown("⏳ **单标的冷却**: 无")
    except Exception:
        pass

    st.divider()

    # ---------- B. Per-Symbol Gate Trace 矩阵 ----------
    st.subheader("🎯 每标的决策链")
    # 读取所有 gate_trace:{sym}
    try:
        trace_keys = list(r.scan_iter(match="meta:gate_trace:*", count=200))
    except Exception as _e:
        trace_keys = []
        st.error(f"scan gate_trace 失败: {_e}")

    if not trace_keys:
        st.info("暂无决策链快照。等待 SE 推送下一 tick...")
    else:
        # 排序稳定
        parsed_rows = []
        for key in trace_keys:
            sym = (key.decode('utf-8') if isinstance(key, bytes) else key).split(':')[-1]
            try:
                h = r.hgetall(key) or {}
                def _hget(k):
                    v = h.get(k.encode() if isinstance(k, str) else k) or h.get(k)
                    if isinstance(v, bytes):
                        v = v.decode('utf-8', errors='ignore')
                    return v or ''
                kind = _hget('kind') or 'entry'
                result = _hget('result') or '-'
                last_block = _hget('last_block') or ''
                import json as _json
                trace_list = []
                try:
                    trace_list = _json.loads(_hget('trace_json') or '[]')
                except Exception:
                    trace_list = []
                ts = _hget('ts') or ''
                parsed_rows.append({
                    'sym': sym, 'kind': kind, 'result': result,
                    'last_block': last_block, 'trace': trace_list, 'ts': ts,
                })
            except Exception:
                continue
        parsed_rows.sort(key=lambda x: (x['kind'], x['sym']))

        # 两列并排展示, 信息密度更高
        cols_per_row = 2
        for row_idx in range(0, len(parsed_rows), cols_per_row):
            row_slice = parsed_rows[row_idx:row_idx + cols_per_row]
            cols = st.columns(cols_per_row)
            for c_idx, row in enumerate(row_slice):
                with cols[c_idx]:
                    kind_tag = "🎯 ENTRY" if row['kind'] == 'entry' else "🚪 EXIT"
                    result = row['result']
                    if result == 'BUY':
                        color = "green"
                        verdict = "✅ BUY"
                    elif result.startswith('SELL'):
                        color = "orange"
                        verdict = f"🚪 {result}"
                    elif result.startswith('REJECT'):
                        color = "red"
                        verdict = f"❌ {result}"
                    else:
                        color = "gray"
                        verdict = result
                    st.markdown(f"#### `{row['sym']}` · {kind_tag} · :{color}[{verdict}]")
                    # trace 逐条渲染
                    icons = {'pass': '✅', 'block': '❌', 'skip': '⏸️'}
                    lines = []
                    for g in row['trace']:
                        gate = g.get('gate', '?')
                        status = g.get('status', '?')
                        detail = g.get('detail', '')
                        icon = icons.get(status, '•')
                        # 拦截那条加粗, 其它小字
                        if status == 'block':
                            lines.append(f"**{icon} `{gate}` — {detail}**")
                        elif status == 'skip':
                            lines.append(f"<span style='color:#888'>{icon} `{gate}` — {detail}</span>")
                        else:
                            lines.append(f"{icon} `{gate}` — <span style='color:#666'>{detail}</span>")
                    st.markdown("<br/>".join(lines), unsafe_allow_html=True)

    st.divider()

    # ---------- C. Top-N 拒单计数器 ----------
    st.subheader("🏆 今日拦截 TOP 15")
    try:
        import datetime as _dt, pytz as _pytz
        ny_date = _dt.datetime.now(_pytz.timezone('America/New_York')).strftime('%Y%m%d')
        cnt_map = r.hgetall(f"meta:gate_counter:{ny_date}") or {}
        rows = []
        for k, v in cnt_map.items():
            gk = k.decode('utf-8') if isinstance(k, bytes) else str(k)
            try:
                cnt = int(v.decode('utf-8') if isinstance(v, bytes) else v)
            except Exception:
                cnt = 0
            rows.append((gk, cnt))
        rows.sort(key=lambda x: -x[1])
        if not rows:
            st.info(f"{ny_date} 今日暂无拦截记录。")
        else:
            import pandas as _pd
            df = _pd.DataFrame(rows[:15], columns=['Gate', 'Blocked'])
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as _e:
        st.error(f"读 gate_counter 失败: {_e}")

    st.divider()

    # ---------- D. 策略参数快照 ----------
    with st.expander("📋 strategy_config0 参数快照 (SE 启动时广播)", expanded=False):
        try:
            cfg_map = r.hgetall("meta:strategy_config") or {}
            if not cfg_map:
                st.info("meta:strategy_config 为空, 请确认 SE 已启动且成功广播。")
            else:
                rows = []
                meta_cols = {}
                for k, v in cfg_map.items():
                    kk = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                    vv = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                    if kk.startswith('__') and kk.endswith('__'):
                        meta_cols[kk] = vv
                    else:
                        rows.append((kk, vv))
                if meta_cols:
                    info_parts = []
                    for k in ('__core_version__', '__version__', '__loaded_at__'):
                        if k in meta_cols:
                            info_parts.append(f"**{k}** = `{meta_cols[k]}`")
                    st.markdown(" · ".join(info_parts))
                import pandas as _pd
                df = _pd.DataFrame(sorted(rows, key=lambda x: x[0]), columns=['Param', 'Value'])
                st.dataframe(df, use_container_width=True, hide_index=True, height=420)
        except Exception as _e:
            st.error(f"读 strategy_config 失败: {_e}")


# 自动刷新逻辑
if auto_refresh:
    time.sleep(1)
    st.rerun()
