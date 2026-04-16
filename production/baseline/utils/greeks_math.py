import numpy as np
import pandas as pd
import re
try:
    from py_vollib_vectorized import vectorized_implied_volatility, get_all_greeks
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False

# Constants
IDX_PRICE, IDX_DELTA, IDX_GAMMA, IDX_VEGA, IDX_THETA, IDX_STRIKE, IDX_VOLUME, IDX_IV = 0, 1, 2, 3, 4, 5, 6, 7
IDX_BID, IDX_ASK = 8, 9

def find_iv(price, S, K, T, r, flag):
    """Fallback simple IV finder if py_vollib is missing"""
    raise RuntimeError("py_vollib_vectorized is required for reliable IV calculation")

# =========================================================
# 🏦 [动态利率自标定]：模块级缓存与查找逻辑
# =========================================================
_G_RFR_SERIES = None

def get_current_rfr(current_ts: float):
    global _G_RFR_SERIES
    if _G_RFR_SERIES is None:
        from pathlib import Path
        
        # 尝试多个路径：优先项目根目录，其次硬编码服务器目录
        project_root = Path(__file__).parent.parent.parent
        candidates = [
            project_root / "risk_free_rates.parquet",
            Path("/home/kingfang007/risk_free_rates.parquet"),
            Path("risk_free_rates.parquet")
        ]
        
        cache_file = None
        for p in candidates:
            if p.exists():
                cache_file = p
                break
        
        try:
            if cache_file is None:
                raise FileNotFoundError(f"❌ [RFR_FATAL] Risk-Free Rate cache missing. Checked: {[str(c) for c in candidates]}")
            
            df = pd.read_parquet(cache_file)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df.index = df.index.normalize()
            _G_RFR_SERIES = df['DGS3MO']
            print(f">>>> [RFR_INIT] Successfully loaded RFR from {cache_file.name}")
        except Exception as e:
            raise RuntimeError(f"❌ [CRITICAL] Failed to load RFR source: {e}. Greeks calculation cannot proceed!")

    if current_ts is None:
        raise ValueError("❌ [RFR_ERROR] current_ts is required for dynamic RFR lookup.")
        
    try:
        # 对齐到纽约时区日期进行 As-of 查找
        from datetime import datetime
        import pytz
        NY_TZ = pytz.timezone('America/New_York')
        dt = datetime.fromtimestamp(current_ts, NY_TZ).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
        
        val = _G_RFR_SERIES.asof(dt)
        if pd.isna(val):
             raise RuntimeError(f"❌ [RFR_MISSING] No interest rate data found for date {dt.date()} in cache.")
        val = float(val)
        if abs(val) > 1.0:
            val /= 100.0
        return val
    except Exception as e:
        if isinstance(e, RuntimeError): raise e
        raise RuntimeError(f"❌ [RFR_LOOKUP_FAILED] Date: {current_ts}, Error: {e}")


def _parse_option_type(ticker: str) -> str:
    text = str(ticker or "").replace("O:", "")
    m = re.search(r"\d{6}([CP])", text)
    if not m:
        raise ValueError(f"Cannot parse option type from contract: {ticker}")
    return m.group(1).lower()

def calculate_bucket_greeks(buckets: np.ndarray, S: float, T: float, r: float = None, contracts: list = None, current_ts: float = None):
    """
    Unified logic to supplement/recalculate Greeks for a 6x12 option bucket.
    Updates the buckets array in-place.
    Now supports auto-calibration of R (Risk-Free Rate) based on current_ts.
    """
    # 🔍 [LOUD LOG] 确认物理文件路径和输入
    log_id = getattr(calculate_bucket_greeks, '_log_cnt', 0)

    # 🚀 [🔥 Parity Fix] 动态利率自标定逻辑
    if r is None and current_ts is not None:
        r = get_current_rfr(current_ts)
        
    if r is None:
        # 如果依然拿不到利率，记录警告并返回原始 buckets (杜绝运行时异常中断整个特征链路)
        if log_id < 20: print(f">>>> [RFR_WARNING] r is None. Greeks recalculation aborted.")
        return buckets
    r = float(r)
    if abs(r) > 1.0:
        r /= 100.0

    if not HAS_VOLLIB:
        raise RuntimeError("py_vollib_vectorized is not installed; cannot recalculate Greeks safely")
    # 🚀 [优化] 持续采样追踪：不再只打印前 20 条，改为每 1000 个 Bucket 采样打印一次
    if not hasattr(calculate_bucket_greeks, "_sample_cnt"):
        calculate_bucket_greeks._sample_cnt = 0
    
    show_log = (calculate_bucket_greeks._sample_cnt % 1000 == 0)
    calculate_bucket_greeks._sample_cnt += 1

    # if show_log: 
    #     print(f"\n>>>> [MATH_TRACE] File: {__file__} | Sample_ID: {calculate_bucket_greeks._sample_cnt}")
    #     print(f">>>> [MATH_TRACE] Params: S={S:.2f}, T={T:.8f}, R={r:.4f}, Sample={contracts[:1] if contracts else 'N/A'}")

    # 🚀 [FORCE CLEAR] 起手全量清零，杜绝任何 0.5 残留
    buckets[:, IDX_IV] = 0.0
    buckets[:, IDX_DELTA:IDX_STRIKE] = 0.0 

    if S < 0.01 or T < 1e-7:
        #if show_log: print(f">>>> [MATH_TRACE] Skipped due to S/T")
        return buckets

    # 1. 遍历每一行进行重算
    for i in range(len(buckets)):
        if i >= len(contracts): break
        ticker = contracts[i]
        if not ticker: continue
        
        # 🚨 [FIX] 删除了导致 0.5 泄露的临时占位符
        # buckets[i, IDX_IV] = 0.5 <-- DELETED
        
        try:
            opt_type = _parse_option_type(ticker)
        except ValueError:
            continue
        strike = float(buckets[i, IDX_STRIKE])
        
        if strike < 0.01: continue

        # 强约束：Greeks 计算只允许使用 bid/ask，不允许使用成交价（IDX_PRICE）。
        if buckets.shape[1] <= IDX_ASK:
            continue
        bid = float(buckets[i, IDX_BID])
        ask = float(buckets[i, IDX_ASK])
        if not (bid > 0.0 and ask >= bid):
            # 无有效盘口则跳过该合约，保持 Greeks/IV 为 0
            buckets[i, IDX_PRICE] = 0.0
            continue
        price = 0.5 * (bid + ask)
        # 下游仍有对第0列的读取，统一回写为 mid，避免携带成交价语义
        buckets[i, IDX_PRICE] = price
        
        iv = 0.0
        if price > 0.0001:
            try:
                res = vectorized_implied_volatility(
                    np.array([price]), np.array([S]), np.array([strike]),
                    np.array([T]), np.array([r]), opt_type, return_as='numpy', on_error='ignore'
                )
                iv = float(res[0]) if not np.isnan(res[0]) else 0.0
            except Exception as e:
                iv = 0.0

        # 🚀 [MATH_TRACE] 用户请求：打印计算核参
        if show_log:
            from datetime import datetime
            dt_str = datetime.fromtimestamp(current_ts).strftime('%Y-%m-%d %H:%M:%S') if current_ts else "N/A"
            print(f">>>> [MATH_TRACE] {ticker} | {dt_str} | S:{S:.2f} | K:{strike:.2f} | P:{price:.4f} | IV:{iv:.4f}")

        # 🚨 [SECURITY_FIX] 严禁覆写 Index 0 (Price) 或 Index 5 (Strike)
        # 结果必须精准归位于 Index 7 (IV)
        buckets[i, IDX_IV] = iv
        
        # 希腊值计算
        if iv > 0.001:
            try:
                g_df = get_all_greeks(
                    opt_type, np.array([S]), np.array([strike]),
                    np.array([T]), np.array([r]), np.array([iv]), return_as='dict'
                )
                buckets[i, IDX_DELTA] = float(g_df['delta'][0])
                buckets[i, IDX_GAMMA] = float(g_df['gamma'][0])
                buckets[i, IDX_VEGA] = float(g_df['vega'][0])
                buckets[i, IDX_THETA] = float(g_df['theta'][0])
            except Exception as e:
                pass

    if show_log:
        print(f">>>> [MATH_TRACE] Bucket calculation completed for batch.")
    return buckets
