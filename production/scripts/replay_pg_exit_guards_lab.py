#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 PostgreSQL 拉取指定交易日 MU/TSLA（+SPY/QQQ 指数）分钟数据，按真实 ALPHA_FRAME 契约
喂给 ExecutionEngineV8._process_alpha_frame。

默认（无 --real-entry）：**禁 decide_entry**，在首根有期权桶的 bar **虚拟多 CALL**，只扫平仓链，用于护栏 profile 对比。

**--real-entry**：启用真实 **decide_entry** 与 OMS **_execute_entry / _execute_exit**（与线上一致决策路径）；
仅对 `--seed-long` 标的调用开仓逻辑，QQQ/SPY 仍只作指数上下文。若你实盘约 9:45 开仓而默认 TREND 对不上，多半是核心版本不同，请加 **--strategy-core V0** 与线上一致。

调参思路（利润 vs 被震出）：
  - 若某 profile 把首平从 NO_MOMENTUM/CT_TIMEOUT/FLIP 推迟，且 **peak_max_roi_pct 明显高于 roi_at_exit_pct**，
    说明 baseline 里该规则在「可拿的浮盈」前把你洗出去 → 考虑放宽对应字段（见 strategy_config0.py）。
  - 若推迟后 reason 变成阶梯 TREND_TRAIL / STEP_PROT，但 roi 仍差不多，说明主要矛盾不在时间护栏而在趋势/止盈轨。
  - 本脚本默认关掉 EXIT_LIQUIDITY_GUARD，避免 PG 宽点差掩盖策略护栏；实盘调回前请在单独 profile 里对比。

环境：
  - RUN_MODE=BACKTEST 由脚本强制写入
  - PG: config.PG_DB_URL 或环境变量 PG_DB_URL
  - Redis 使用内存桩

示例：
  cd production/baseline
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --seed-long mu,tsla
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --until 11:20 --full-journey
  # 真实开仓（decide_entry + OMS _execute_entry/exit），仅 seed-long 标的参与开仓决策
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --until 11:20 --real-entry --full-journey
  # 强制 V0 核心（需在进程内先于 execution_engine 首次 import 生效，见 --strategy-core）
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --real-entry --strategy-core V0
  # SE 分钟引擎 → OMS（config.TARGET_SYMBOLS 全日 PG），列 BUY/SELL 与 realized_pnl（需 torch）
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --se-dual
  # 无 torch 时：PG 拼 ALPHA_FRAME 直喂 OMS（TARGET_SYMBOLS），同样列成交与 realized_pnl
  PYTHONPATH=. python ../scripts/replay_pg_exit_guards_lab.py --date 20260501 --pg-oms-dual
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
def _bootstrap_path() -> None:
    baseline = Path(__file__).resolve().parents[1] / "baseline"
    prod = baseline.parent
    hr = prod / "history_replay"
    for p in (str(baseline), str(prod), str(hr)):
        if p not in sys.path:
            sys.path.insert(0, p)


# -----------------------------------------------------------------------------
# In-memory Redis (avoid localhost:6379 dependency)
# -----------------------------------------------------------------------------
class _Pipe:
    def delete(self, *a, **k):
        return self

    def hset(self, *a, **k):
        return self

    def expire(self, *a, **k):
        return self

    def hincrby(self, *a, **k):
        return self

    def execute(self):
        return []


class RedisStub:
    connection_pool = type("CP", (), {"connection_kwargs": {"db": 0}})()

    def pipeline(self):
        return _Pipe()

    def delete(self, *names):
        return 0

    def hset(self, name=None, key=None, value=None, mapping=None, items=None):  # noqa: ARG002
        return 0

    def expire(self, name=None, ttl=None):  # noqa: ARG002
        return True

    def set(self, name=None, value=None, nx=None, ex=None, ttl=None):  # noqa: ARG002
        return True

    def get(self, name=None):  # noqa: ARG002
        return None

    def ttl(self, name=None):  # noqa: ARG002
        return -1

    def eval(self, *a, **k):
        return 1

    def xadd(self, name=None, fields=None, maxlen=None):  # noqa: ARG002
        return b"0-1"

    def xinfo_groups(self, stream):  # noqa: ARG002
        return []

    def xgroup_create(self, stream, group, mkstream=False, id="0"):  # noqa: ARG002
        return True

    def xgroup_setid(self, stream, group, id):  # noqa: ARG002
        return True


def _install_redis_stub() -> None:
    import redis as redis_mod

    redis_mod.Redis = lambda **kwargs: RedisStub()  # noqa: E731


def _today_ny_ymd() -> str:
    from pytz import timezone as tz

    return datetime.now(tz("America/New_York")).strftime("%Y%m%d")


def ny_session_bounds_ymd(ymd: str) -> Tuple[float, float]:
    from pytz import timezone as tz

    ny = tz("America/New_York")
    dt = datetime.strptime(ymd, "%Y%m%d")
    start = ny.localize(dt.replace(hour=9, minute=30, second=0))
    end = ny.localize(dt.replace(hour=16, minute=0, second=0))
    return start.timestamp(), end.timestamp()


def ny_session_ts(ymd: str, hour: int, minute: int, second: int = 0) -> float:
    """美东某日 HH:MM(:ss) 的 Unix 秒（用于截断回放窗口）。"""
    from pytz import timezone as tz

    ny = tz("America/New_York")
    dt = datetime.strptime(ymd, "%Y%m%d")
    return ny.localize(dt.replace(hour=hour, minute=minute, second=second)).timestamp()


def parse_until_ny(ymd: str, until_raw: str) -> Optional[float]:
    """
    解析 --until 如 11:20 / 1120，返回该美东时刻的 epoch 秒（含该分钟 bar，通常 ts 对齐到分钟起点）。
    空字符串则返回 None（表示不额外截断，沿用会话 16:00 拉数）。
    """
    s = str(until_raw or "").strip()
    if not s:
        return None
    s = s.replace(":", "")
    if len(s) == 3:
        s = "0" + s  # 920 -> 0920
    if len(s) != 4 or not s.isdigit():
        raise ValueError(f"--until 需为 HH:MM 或 HHMM，收到: {until_raw!r}")
    h, m = int(s[:2]), int(s[2:])
    if h > 23 or m > 59:
        raise ValueError(f"--until 非法时刻: {until_raw!r}")
    return ny_session_ts(ymd, h, m, 0)


def parse_buckets_json(raw: Optional[str]) -> Dict[str, Any]:
    """与 audit_alpha_executable_edge 一致：ATM PUT idx0, ATM CALL idx2; bid=[8], ask=[9]."""
    if not raw or str(raw).strip() in ("{}", "None", ""):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def opt_data_from_buckets(payload: Dict[str, Any]) -> Dict[str, Any]:
    buckets = payload.get("buckets") or []
    contracts = payload.get("contracts") or []

    def pull(side_idx: int):
        if side_idx >= len(buckets):
            return (0.0,) * 9
        bk = buckets[side_idx]
        mid = float(bk[0]) if len(bk) > 0 else 0.0
        bid = float(bk[8]) if len(bk) > 8 else mid
        ask = float(bk[9]) if len(bk) > 9 else mid
        cid = str(contracts[side_idx]) if len(contracts) > side_idx else ""
        iv = float(bk[3]) if len(bk) > 3 else 0.0
        k = float(bk[1]) if len(bk) > 1 else 0.0
        vol = float(bk[7]) if len(bk) > 7 else 1.0
        bs = float(bk[10]) if len(bk) > 10 else 0.0
        a_s = float(bk[11]) if len(bk) > 11 else 0.0
        return mid, bid, ask, cid, iv, k, vol, bs, a_s

    c_mid, c_bid, c_ask, c_id, c_iv, c_k, c_vol, c_bs, c_asz = pull(2)
    p_mid, p_bid, p_ask, p_id, p_iv, p_k, p_vol, p_bs, p_asz = pull(0)
    has_c = bool(c_bid > 0 and c_ask > 0) or bool(c_mid > 0)
    has_p = bool(p_bid > 0 and p_ask > 0) or bool(p_mid > 0)
    return {
        "has_feed": bool(has_c or has_p),
        "call_price": c_mid or max(c_bid, c_ask, 0.0),
        "call_bid": c_bid,
        "call_ask": c_ask,
        "call_id": c_id,
        "call_k": c_k,
        "call_iv": c_iv,
        "call_vol": c_vol if c_vol > 0 else 1.0,
        "call_bid_size": c_bs,
        "call_ask_size": c_asz,
        "put_price": p_mid or max(p_bid, p_ask, 0.0),
        "put_bid": p_bid,
        "put_ask": p_ask,
        "put_id": p_id,
        "put_k": p_k,
        "put_iv": p_iv,
        "put_vol": p_vol if p_vol > 0 else 1.0,
        "put_bid_size": p_bs,
        "put_ask_size": p_asz,
    }


def fetch_merged_pg(
    start_ts: float,
    end_ts: float,
    symbols: List[str],
    pg_url: str,
):
    try:
        import pandas as pd
        import psycopg2
    except ImportError as e:
        raise RuntimeError(f"需要 pandas psycopg2: {e}") from e

    sy = ",".join("'" + s.replace("'", "") + "'" for s in symbols)
    q = f"""
        SELECT m.symbol,
               CAST(m.ts AS DOUBLE PRECISION) AS ts,
               m.close,
               COALESCE(m.volume, 0) AS volume,
               COALESCE(a.alpha, 0.0)::float AS alpha,
               COALESCE(a.vol_z, 0.0)::float AS vol_z,
               COALESCE(o.buckets_json::text, '{{}}') AS buckets_json
        FROM market_bars_1m m
        LEFT JOIN alpha_logs a ON a.symbol = m.symbol AND a.ts = m.ts
        LEFT JOIN option_snapshots_1m o ON o.symbol = m.symbol AND o.ts = m.ts
        WHERE m.ts >= %(st)s AND m.ts <= %(en)s AND m.symbol IN ({sy})
        ORDER BY m.ts ASC, m.symbol ASC
    """
    conn = psycopg2.connect(pg_url)
    try:
        df = pd.read_sql(q, conn, params={"st": float(start_ts), "en": float(end_ts)})
    finally:
        conn.close()
    if df.empty:
        return df
    df.sort_values(["symbol", "ts"], inplace=True)
    df["roc_5m"] = df.groupby("symbol")["close"].pct_change(5).fillna(0.0)
    return df


def build_frames_from_df(df, symbol_order: List[str]) -> List[Dict[str, Any]]:
    import pandas as pd  # noqa: F401

    frames: List[Dict[str, Any]] = []
    if df.empty:
        return frames

    grp = df.groupby("ts")
    fid = 0

    def pick_index_trend(gdf: pd.DataFrame) -> int:
        if "SPY" not in set(gdf["symbol"].values):
            return 0
        r = float(gdf.loc[gdf["symbol"] == "SPY", "roc_5m"].mean() or 0.0)
        if r > 0.00008:
            return 1
        if r < -0.00008:
            return -1
        return 0

    for ts_val, chunk in grp:
        row_by = {row["symbol"]: row for _, row in chunk.iterrows()}
        spy_roc_common = float(row_by["SPY"]["roc_5m"]) if "SPY" in row_by else 0.0
        qqq_roc_common = float(row_by["QQQ"]["roc_5m"]) if "QQQ" in row_by else 0.0
        spy_rocs = [spy_roc_common for _ in symbol_order]
        qqq_rocs = [qqq_roc_common for _ in symbol_order]

        items: List[dict] = []
        fid += 1
        for i, sym in enumerate(symbol_order):
            row = row_by.get(sym)
            if row is None:
                items.append(_placeholder_item(sym, i, ts_val))
                continue

            buckets = parse_buckets_json(row.get("buckets_json"))
            od = opt_data_from_buckets(buckets)
            alpha = float(row.get("alpha", 0.0) or 0.0)

            items.append({
                "symbol": sym,
                "batch_idx": i,
                "stock_price": float(row["close"]),
                "alpha": alpha,
                "cs_alpha_z": alpha,
                "vol_z": float(row.get("vol_z", 0.0) or 0.0),
                "roc_5m": float(row.get("roc_5m", 0.0) or 0.0),
                "macd": 0.02,
                "macd_slope": 0.001,
                "snap_roc": float(row.get("roc_5m", 0.0) or 0.0) * 0.35,
                "event_prob": 0.0,
                "is_ready": True,
                "last_valid_iv": max(float(od.get("call_iv", 0) or 0), float(od.get("put_iv", 0) or 0)),
                "correction_mode": "NORMAL",
                "alpha_label_ts": float(ts_val),
                "alpha_available_ts": float(ts_val),
                "opt_data": od,
            })

        idx_trend = pick_index_trend(chunk)
        frames.append({
            "source": "alpha_engine_v8",
            "action": "ALPHA_FRAME",
            "ts": float(ts_val),
            "frame_id": f"lab-{int(ts_val)}-{fid}",
            "symbols": list(symbol_order),
            "items": items,
            "index_trend": int(idx_trend),
            "spy_roc_5min": spy_rocs,
            "qqq_roc_5min": qqq_rocs,
            "is_zombie_market": False,
            "global_regime_reversal_cnt": 0,
            "global_is_volatile_regime": False,
            "global_regime_band": "calm",
            "global_regime_score": 0.25,
        })

    return frames


def _placeholder_item(sym: str, batch_idx: int, ts_val: float) -> dict:
    return {
        "symbol": sym,
        "batch_idx": batch_idx,
        "stock_price": 100.0,
        "alpha": 0.0,
        "cs_alpha_z": 0.0,
        "vol_z": 0.0,
        "roc_5m": 0.0,
        "macd": 0.0,
        "macd_slope": 0.0,
        "snap_roc": 0.0,
        "event_prob": 0.0,
        "is_ready": False,
        "last_valid_iv": 0.3,
        "correction_mode": "NORMAL",
        "alpha_label_ts": float(ts_val),
        "alpha_available_ts": float(ts_val),
        "opt_data": {"has_feed": False},
    }


def build_se_dual_minute_batch(
    ts_val: float,
    symbol_order: List[str],
    row_by: Dict[str, Any],
    last_cache: Dict[str, Any],
    is_new_minute: bool,
    frame_id: str,
) -> dict:
    """构造 `SignalEngineV8.process_batch` 的分钟 batch：alpha 用 PG `alpha_logs`，期权用 buckets_json。"""
    import numpy as np

    n = len(symbol_order)
    spy_roc = float(row_by["SPY"]["roc_5m"]) if "SPY" in row_by else float(last_cache.get("__spy_roc__", 0.0) or 0.0)
    qqq_roc = float(row_by["QQQ"]["roc_5m"]) if "QQQ" in row_by else float(last_cache.get("__qqq_roc__", 0.0) or 0.0)
    last_cache["__spy_roc__"] = spy_roc
    last_cache["__qqq_roc__"] = qqq_roc
    spy_rocs = np.full(n, spy_roc, dtype=np.float32)
    qqq_rocs = np.full(n, qqq_roc, dtype=np.float32)

    closes = np.zeros(n, dtype=np.float32)
    alphas = np.zeros(n, dtype=np.float32)
    volzs = np.zeros(n, dtype=np.float32)
    tradable = np.zeros(n, dtype=np.float32)
    edge = np.zeros(n, dtype=np.float32)
    put_prices = np.zeros(n, dtype=np.float32)
    call_prices = np.zeros(n, dtype=np.float32)
    put_ks = np.zeros(n, dtype=np.float32)
    call_ks = np.zeros(n, dtype=np.float32)
    put_ivs = np.zeros(n, dtype=np.float32)
    call_ivs = np.zeros(n, dtype=np.float32)
    put_bids = np.zeros(n, dtype=np.float32)
    put_asks = np.zeros(n, dtype=np.float32)
    call_bids = np.zeros(n, dtype=np.float32)
    call_asks = np.zeros(n, dtype=np.float32)
    feed_put_ids = [""] * n
    feed_call_ids = [""] * n
    symbols_with_data: set = set()

    for i, sym in enumerate(symbol_order):
        r = row_by.get(sym)
        if r is None:
            r = last_cache.get(sym)
        if r is not None:
            last_cache[sym] = r
        if r is None:
            feed_put_ids[i] = f"LAB|{sym}|PUT"
            feed_call_ids[i] = f"LAB|{sym}|CALL"
            continue
        od = opt_data_from_buckets(parse_buckets_json(r.get("buckets_json")))
        px = float(r.get("close", 0.0) or 0.0)
        if px > 0:
            closes[i] = np.float32(px)
        alphas[i] = np.float32(float(r.get("alpha", 0.0) or 0.0))
        volzs[i] = np.float32(float(r.get("vol_z", 0.0) or 0.0))
        if od.get("has_feed") and (
            float(od.get("call_price", 0) or 0) > 0.01 or float(od.get("put_price", 0) or 0) > 0.01
        ):
            symbols_with_data.add(sym)
        put_prices[i] = np.float32(float(od.get("put_price", 0.0) or 0.0))
        call_prices[i] = np.float32(float(od.get("call_price", 0.0) or 0.0))
        put_ks[i] = np.float32(float(od.get("put_k", 0.0) or 0.0))
        call_ks[i] = np.float32(float(od.get("call_k", 0.0) or 0.0))
        put_ivs[i] = np.float32(float(od.get("put_iv", 0.0) or 0.0))
        call_ivs[i] = np.float32(float(od.get("call_iv", 0.0) or 0.0))
        put_bids[i] = np.float32(float(od.get("put_bid", 0.0) or 0.0))
        put_asks[i] = np.float32(float(od.get("put_ask", 0.0) or 0.0))
        call_bids[i] = np.float32(float(od.get("call_bid", 0.0) or 0.0))
        call_asks[i] = np.float32(float(od.get("call_ask", 0.0) or 0.0))
        feed_call_ids[i] = str(od.get("call_id") or "").strip() or f"LAB|{sym}|CALL"
        feed_put_ids[i] = str(od.get("put_id") or "").strip() or f"LAB|{sym}|PUT"

    slow_1m = np.zeros((n, 30, 1), dtype=np.float32)
    return {
        "symbols": list(symbol_order),
        "ts": float(ts_val),
        "frame_id": frame_id,
        "stock_price": closes,
        "fast_vol": volzs.astype(np.float32),
        "precalc_alpha": alphas,
        "tradable_prob": tradable,
        "edge_score": edge,
        "is_new_minute": bool(is_new_minute),
        "symbols_with_data": symbols_with_data,
        "feed_put_price": put_prices,
        "feed_call_price": call_prices,
        "feed_put_k": put_ks,
        "feed_call_k": call_ks,
        "feed_put_iv": put_ivs,
        "feed_call_iv": call_ivs,
        "feed_put_bid": put_bids,
        "feed_put_ask": put_asks,
        "feed_call_bid": call_bids,
        "feed_call_ask": call_asks,
        "feed_put_vol": np.ones(n, dtype=np.float32),
        "feed_call_vol": np.ones(n, dtype=np.float32),
        "feed_call_bid_size": np.full(n, 100.0, dtype=np.float32),
        "feed_call_ask_size": np.full(n, 100.0, dtype=np.float32),
        "feed_put_bid_size": np.full(n, 100.0, dtype=np.float32),
        "feed_put_ask_size": np.full(n, 100.0, dtype=np.float32),
        "slow_1m": slow_1m,
        "feed_put_id": feed_put_ids,
        "feed_call_id": feed_call_ids,
        "spy_roc_5min": spy_rocs,
        "qqq_roc_5min": qqq_rocs,
    }


async def run_pg_oms_alpha_frames_dual(df, symbol_order: List[str], ymd: str) -> None:
    """
    不启动 SignalEngine（免 torch）：用 PG 拼与 SE 同结构的 ALPHA_FRAME（见 build_frames_from_df），
    经 OMS `_handle_trade_signal` 逐帧回放，汇总 BUY/SELL 与 realized_pnl。
    分钟级 macd 等仍为实验室占位，与完整 SE 特征链可能有细微差异。
    """
    import logging

    logging.getLogger().setLevel(logging.WARNING)
    _install_redis_stub()
    _ensure_scipy_stub()
    _ensure_ibkr_stub()
    os.environ["RUN_MODE"] = "BACKTEST"
    os.environ["IS_SIMULATED"] = "1"

    from execution_engine_v8 import ExecutionEngineV8  # noqa: E402
    from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION  # noqa: E402

    symbols = list(symbol_order)
    frames = build_frames_from_df(df, symbols)
    ee = ExecutionEngineV8(symbols=symbols, mode="backtest")
    ee.r = RedisStub()

    async def _noop_broadcast(_self):  # noqa: ARG001
        return None

    ee._broadcast_state_to_redis = types.MethodType(_noop_broadcast, ee)
    ee._publish_gate_trace = lambda *a, **k: None
    ee._publish_entry_diag = lambda *a, **k: None

    fills: List[dict] = []
    _orig_h = ee._handle_trade_signal

    async def _wrap_trade(self, payload, allow_delay_queue=True):
        a = str(payload.get("action") or "").upper()
        if a in ("BUY", "SELL"):
            sig = payload.get("sig") or {}
            fills.append(
                {
                    "action": a,
                    "symbol": str(payload.get("symbol") or ""),
                    "ts": float(payload.get("ts") or 0.0),
                    "reason": str(sig.get("reason", "")),
                }
            )
        await _orig_h(payload, allow_delay_queue=allow_delay_queue)

    ee._handle_trade_signal = types.MethodType(_wrap_trade, ee)
    ee._processed_alpha_frame_set.clear()
    ee._processed_alpha_frame_ids.clear()
    ee._alpha_frame_ready = True

    for fr in frames:
        await ee.process_trade_signal(fr)

    from pytz import timezone as tz

    ny = tz("America/New_York")

    def _fmt(ts: float) -> str:
        return datetime.fromtimestamp(float(ts), ny).strftime("%m-%d %H:%M")

    print("\n=== PG→ALPHA_FRAME→OMS（无 SE / 免 torch）===")
    print(
        f"交易日={ymd} | ACTIVE_STRATEGY_CORE_VERSION={ACTIVE_STRATEGY_CORE_VERSION} | "
        f"标的={symbols} | 帧数={len(frames)}"
    )
    if not fills:
        print("  （无 BUY/SELL）")
    else:
        for f in fills:
            print(f"  [{f['action']}] {f['symbol']:<6} {_fmt(float(f['ts']))}  {str(f.get('reason', ''))[:120]}")
    pnl = float(getattr(ee, "realized_pnl", 0.0) or 0.0)
    tc = int(getattr(ee, "trade_count", 0) or 0)
    print(f"\n[lab] OMS realized_pnl ≈ ${pnl:,.2f}  | trade_count={tc}")


async def run_pg_se_oms_dual(df, symbol_order: List[str], ymd: str) -> None:
    """SignalEngine（分钟 alpha）→ 队列 → ExecutionEngine：汇总 BUY/SELL 与 OMS realized_pnl。"""
    import logging

    logging.getLogger().setLevel(logging.WARNING)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "[lab] 错误: 当前 Python 未安装 torch，无法加载 SignalEngineV8。\n"
            "  可选：① pip install torch 后重试 --se-dual；"
            "② 使用 --pg-oms-dual（PG 拼 ALPHA_FRAME 直喂 OMS，免 torch，见脚本说明）。"
        )
        raise SystemExit(3)

    _install_redis_stub()
    _ensure_scipy_stub()
    _ensure_ibkr_stub()
    os.environ["RUN_MODE"] = "BACKTEST"
    os.environ["IS_SIMULATED"] = "1"

    from execution_engine_v8 import ExecutionEngineV8  # noqa: E402
    from signal_engine_v8 import SignalEngineV8  # noqa: E402
    from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION  # noqa: E402

    symbols = list(symbol_order)
    se = SignalEngineV8(symbols=symbols, mode="backtest", config_paths={}, model_paths={})
    se.r = RedisStub()

    ee = ExecutionEngineV8(symbols=symbols, mode="backtest")
    ee.r = RedisStub()
    for sym in symbols:
        if sym in se.states and sym in ee.states:
            ee.states[sym] = se.states[sym]

    q: asyncio.Queue = asyncio.Queue()
    se.use_shared_mem = True
    se.signal_queue = q
    ee.use_shared_mem = True
    ee.signal_queue = q

    fills: List[dict] = []
    _orig_h = ee._handle_trade_signal

    async def _wrap_trade(self, payload, allow_delay_queue=True):
        a = str(payload.get("action") or "").upper()
        if a in ("BUY", "SELL"):
            sig = payload.get("sig") or {}
            fills.append(
                {
                    "action": a,
                    "symbol": str(payload.get("symbol") or ""),
                    "ts": float(payload.get("ts") or 0.0),
                    "reason": str(sig.get("reason", "")),
                }
            )
        await _orig_h(payload, allow_delay_queue=allow_delay_queue)

    ee._handle_trade_signal = types.MethodType(_wrap_trade, ee)

    last_cache: Dict[str, Any] = {}
    last_min = -1
    fid = 0
    df_sorted = df.sort_values(["ts", "symbol"])

    for ts_val, chunk in df_sorted.groupby("ts", sort=False):
        row_by = {str(r["symbol"]).strip(): r for _, r in chunk.iterrows()}
        cur_min = int(float(ts_val) // 60)
        is_new = cur_min != last_min
        last_min = cur_min
        fid += 1
        batch = build_se_dual_minute_batch(
            float(ts_val),
            symbols,
            row_by,
            last_cache,
            is_new,
            f"dual-{int(float(ts_val))}-{fid}",
        )
        await se.process_batch(batch)
        guard = 0
        while not q.empty() and guard < 500:
            payload = q.get_nowait()
            await ee.process_trade_signal(payload)
            q.task_done()
            guard += 1

    from pytz import timezone as tz

    ny = tz("America/New_York")

    def _fmt(ts: float) -> str:
        return datetime.fromtimestamp(float(ts), ny).strftime("%m-%d %H:%M")

    print("\n=== SE→OMS 全日回放（PG + config.TARGET_SYMBOLS + precalc alpha）===")
    print(
        f"交易日={ymd} | ACTIVE_STRATEGY_CORE_VERSION={ACTIVE_STRATEGY_CORE_VERSION} | "
        f"标的数={len(symbols)} | 分钟去重≈{int(df['ts'].nunique())}"
    )
    print(
        "架构说明：SE 分钟边界发 ALPHA_FRAME/SYNC；**BUY/SELL 仅由 OMS 内 StrategyCore** 触发 "
        "（与线上一致：不是旧版 SE 直接下单）。"
    )
    if not fills:
        print("  （无 BUY/SELL 落 OMS——可能全日无信号或被 ENTRY/名额拒单）")
    else:
        for f in fills:
            rs = str(f.get("reason", "") or "")[:120]
            print(f"  [{f['action']}] {f['symbol']:<6} {_fmt(float(f['ts']))}  {rs}")

    pnl = float(getattr(ee, "realized_pnl", 0.0) or 0.0)
    tc = int(getattr(ee, "trade_count", 0) or 0)
    print(f"\n[lab] OMS realized_pnl ≈ ${pnl:,.2f}  | trade_count={tc}")


GUARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    # --- 单拆：看是哪条在抢跑 ---
    "no_no_momentum": {"NO_MOMENTUM_MINS": 99999},
    "no_mid_time": {"MID_TIME_STOP_MINS": 99999},
    "no_long_time_stop": {"TIME_STOP_MINS": 99999},
    "no_zombie": {"EXIT_ZOMBIE_STOP_ENABLED": False, "ZOMBIE_EXIT_MINS": 99999},
    "no_counter_trend_exit": {"EXIT_COUNTER_TREND_ENABLED": False},
    "no_signal_flip": {"EXIT_SIGNAL_FLIP_ENABLED": False},
    "time_family_off": {
        "NO_MOMENTUM_MINS": 99999,
        "MID_TIME_STOP_MINS": 99999,
        "TIME_STOP_MINS": 99999,
        "EXIT_ZOMBIE_STOP_ENABLED": False,
        "ZOMBIE_EXIT_MINS": 99999,
        "EXIT_SMALL_GAIN_ENABLED": False,
    },
    # --- 温和放宽（仍保留护栏，减少震荡出局）---
    "soft_no_momentum": {
        "NO_MOMENTUM_MINS": 10,
        "NO_MOMENTUM_MIN_MAX_ROI": 0.008,
    },
    "longer_counter_trend": {"COUNTER_TREND_MAX_MINS": 25},
    "relax_trend_snap": {
        "TREND_EXIT_SNAP_BREAK": 0.0015,
        "TREND_EXIT_MACD_BREAK": 0.012,
    },
    # --- 组合：偏「多拿一会儿」---
    "hold_bias_combo": {
        "NO_MOMENTUM_MINS": 10,
        "NO_MOMENTUM_MIN_MAX_ROI": 0.006,
        "COUNTER_TREND_MAX_MINS": 22,
        "EXIT_SIGNAL_FLIP_ENABLED": False,
        "TREND_EXIT_SNAP_BREAK": 0.0012,
        "TREND_EXIT_MACD_BREAK": 0.010,
    },
}


def _ensure_scipy_stub() -> None:
    """本机未装 scipy 时避免 execution_engine_v8 顶层 import 失败。"""
    import types

    try:
        from scipy.stats import norm  # noqa: F401, F811
        assert norm is not None
        return
    except ImportError:
        pass

    scipy_mod = types.ModuleType("scipy")
    scipy_stats_mod = types.ModuleType("scipy.stats")
    scipy_stats_mod.norm = object()
    scipy_mod.stats = scipy_stats_mod
    sys.modules.setdefault("scipy", scipy_mod)
    sys.modules.setdefault("scipy.stats", scipy_stats_mod)


def _ensure_ibkr_stub() -> None:
    import types

    try:
        import ibkr_connector_v8 as _ibc  # noqa: F401
        assert hasattr(_ibc, "IBKRConnectorFinal")
        return
    except ImportError:
        pass

    mod = types.ModuleType("ibkr_connector_v8")

    class IBKRConnectorFinal:
        def __init__(self, *args, **kwargs):
            pass

    mod.IBKRConnectorFinal = IBKRConnectorFinal
    sys.modules["ibkr_connector_v8"] = mod


async def run_profile(
    profile_name: str,
    cfg_delta: Dict[str, Any],
    frames: List[dict],
    symbol_order: List[str],
    seed_syms: List[str],
    df_first_rows: Dict[str, Any],
    track_journey: bool = False,
    real_entry: bool = False,
) -> Dict[str, Any]:
    _install_redis_stub()
    _ensure_scipy_stub()
    _ensure_ibkr_stub()
    os.environ["RUN_MODE"] = "BACKTEST"
    os.environ["IS_SIMULATED"] = "1"

    from execution_engine_v8 import ExecutionEngineV8  # noqa: E402

    ee = ExecutionEngineV8(symbols=list(symbol_order), mode="backtest")
    ee.r = RedisStub()

    # PG buckets 常出现宽点差，先关掉平仓点差护栏便于观察时间/阶梯类 reason
    ee.strategy.cfg = replace(ee.strategy.cfg, **cfg_delta)
    ee.strategy.cfg = replace(ee.strategy.cfg, EXIT_LIQUIDITY_GUARD_ENABLED=False)

    # 静默广播/门槛跟踪，减少 Stub 遗漏方法
    async def _noop_broadcast(_self):  # noqa: ARG001
        return None

    ee._broadcast_state_to_redis = types.MethodType(_noop_broadcast, ee)
    ee._publish_gate_trace = lambda *args, **kwargs: None
    ee._publish_entry_diag = lambda *args, **kwargs: None

    exits: Dict[str, List[dict]] = {s: [] for s in seed_syms}
    buys: Dict[str, List[dict]] = {s: [] for s in seed_syms}
    entry_info: Dict[str, Dict[str, Any]] = {}
    journey_peak: Dict[str, Dict[str, float]] = {
        s: {"peak_max_roi": -1.0, "peak_ts": 0.0} for s in seed_syms
    }

    ee._lab_exit_snap: Dict[str, Dict[str, Any]] = {}
    _orig_check_exit = ee.strategy.check_exit

    def _check_exit_wrap(ctx: dict):
        sig = _orig_check_exit(ctx)
        sym = ctx.get("symbol")
        if sym and sig and sig.get("action") == "SELL":
            h = ctx.get("holding") or {}
            ep = float(h.get("entry_price", 0) or 0)
            cp = float(ctx.get("curr_price", 0) or 0)
            if ep > 0.01 and cp > 0.01:
                roi_here = (cp - ep) / ep
                peak = float(h.get("max_roi", 0) or 0)
                ee._lab_exit_snap[str(sym)] = {
                    "roi_at_exit_pct": roi_here * 100.0,
                    "peak_max_roi_pct": peak * 100.0,
                    "held_mins": float(ctx.get("held_mins", 0) or 0),
                    "reason": str(sig.get("reason", "")),
                    "ts": float(ctx.get("curr_ts", 0) or 0),
                }
        return sig

    ee.strategy.check_exit = _check_exit_wrap

    seed_set = {str(s).strip().upper() for s in seed_syms}

    if real_entry:
        _saved_submit = ee._submit_strategy_order

        async def _lab_submit_real(self, action, sym, sig, stock_price, curr_ts, batch_idx, frame_id=None):
            snap: Dict[str, Any] = {}
            if action == "SELL" and sym in seed_set:
                snap = dict(getattr(self, "_lab_exit_snap", {}).get(sym, {}))
            await _saved_submit(action, sym, sig, stock_price, curr_ts, batch_idx, frame_id=frame_id)
            if action == "BUY" and sym in seed_set:
                st = self.states.get(sym)
                if st is not None and int(getattr(st, "position", 0) or 0) != 0:
                    rec = {
                        "action": "BUY",
                        "reason": str((sig or {}).get("reason", "")),
                        "ts": float(curr_ts),
                        "sig": sig,
                        "entry_ts": float(getattr(st, "entry_ts", 0) or 0),
                        "entry_price": float(getattr(st, "entry_price", 0) or 0),
                        "entry_stock": float(getattr(st, "entry_stock", 0) or 0),
                    }
                    buys.setdefault(sym, []).append(rec)
                    if sym not in entry_info:
                        entry_info[sym] = {
                            "seed_bar_ts": float(curr_ts),
                            "entry_ts_engine": float(st.entry_ts),
                            "entry_price": float(st.entry_price),
                            "entry_stock": float(st.entry_stock),
                            "position_side": "LONG_CALL" if int(st.position) == 1 else f"pos={int(st.position)}",
                            "open_reason": str((sig or {}).get("reason", "")),
                        }
            if action == "SELL" and sym in seed_set:
                exits.setdefault(sym, []).append({
                    "action": "SELL",
                    "reason": str((sig or {}).get("reason", "")),
                    "ts": float(curr_ts),
                    "sig": sig,
                    "metrics": dict(snap),
                })

        ee._submit_strategy_order = types.MethodType(_lab_submit_real, ee)

        _orig_de = ee.strategy.decide_entry

        def _de_seed_only(ctx: dict):
            symu = str(ctx.get("symbol") or "").strip().upper()
            if symu not in seed_set:
                return None
            return _orig_de(ctx)

        ee.strategy.decide_entry = _de_seed_only
    else:

        async def _capture_submit(self, action, sym, sig, stock_price, curr_ts, batch_idx, frame_id=None):
            snap = getattr(self, "_lab_exit_snap", {}).get(sym, {})
            exits.setdefault(sym, []).append({
                "action": action,
                "reason": str((sig or {}).get("reason", "")),
                "ts": float(curr_ts),
                "sig": sig,
                "metrics": dict(snap),
            })
            if action == "SELL":
                st = self.states.get(sym)
                if st is not None:
                    st.position = 0
                    st.entry_price = 0.0
                    st.pending_exit_retry_reason = ""

        ee._submit_strategy_order = types.MethodType(_capture_submit, ee)
        ee.strategy.decide_entry = lambda ctx: None  # noqa: E731

    ee._processed_alpha_frame_set.clear()
    ee._processed_alpha_frame_ids.clear()
    ee._alpha_frame_ready = True

    fair = ee._get_fair_market_price

    if not real_entry:
        for sym in seed_syms:
            row = df_first_rows.get(sym)
            if row is None:
                continue
            od = opt_data_from_buckets(parse_buckets_json(row.get("buckets_json")))
            if not od.get("has_feed"):
                continue
            st = ee.states[sym]
            bp = od["call_price"] or od["call_bid"] or od["call_ask"]
            bid, ask = od["call_bid"], od["call_ask"]
            if bid <= 0 or ask <= 0:
                bid = ask = float(bp or 1.0)
            ep = fair(float(bp or 1.0), float(bid), float(ask), 0.0)
            if ep <= 0:
                continue
            st.position = 1
            st.entry_price = float(ep)
            st.last_opt_price = float(ep)
            st.entry_stock = float(row["close"])
            st.entry_ts = float(row["ts"]) - 120.0
            st.max_roi = 0.0
            st.entry_index_trend = int(1)
            st.entry_spy_roc = 0.0002
            st.warmup_complete = True
            st.pending_exit_retry_reason = ""
            st.is_pending = False
            entry_info[sym] = {
                "seed_bar_ts": float(row["ts"]),
                "entry_ts_engine": float(row["ts"]) - 120.0,
                "entry_price": float(st.entry_price),
                "entry_stock": float(row["close"]),
                "position_side": "LONG_CALL_VIRTUAL",
            }

    for fr in frames:
        await ee._process_alpha_frame(fr)
        if track_journey:
            fts = float(fr.get("ts", 0) or 0)
            for sym in seed_syms:
                stx = ee.states.get(sym)
                if not stx:
                    continue
                # 平仓当根末尾 position 可能已为 0，但 st.max_roi 仍保留本笔峰值，必须照样采样
                mr = float(getattr(stx, "max_roi", -1.0) or -1.0)
                if mr > float(journey_peak[sym]["peak_max_roi"]):
                    journey_peak[sym]["peak_max_roi"] = mr
                    journey_peak[sym]["peak_ts"] = fts

    summary = {}
    buy_summary = {}
    for sym in seed_syms:
        evs = exits.get(sym) or []
        sells = [e for e in evs if e["action"] == "SELL"]
        summary[sym] = sells[0] if sells else None
        bvs = buys.get(sym) or []
        buy_summary[sym] = bvs[0] if bvs else None

    out: Dict[str, Any] = {
        "profile": profile_name,
        "exits_by_sym": exits,
        "buys_by_sym": buys,
        "first_exit": summary,
        "first_buy": buy_summary,
        "entry_info": entry_info,
        "real_entry": bool(real_entry),
    }
    if track_journey:
        out["journey_peak"] = journey_peak
    return out


def main():
    # 覆盖 shell 里的 REALTIME，否则会走陈旧报价平仓等实盘分支
    os.environ["RUN_MODE"] = "BACKTEST"

    parser = argparse.ArgumentParser(description="PG 回放 + ExecutionEngine ALPHA_FRAME 护栏对比实验室")
    parser.add_argument("--date", default="", help="交易日 YYYYMMDD，默认美东今日")
    parser.add_argument(
        "--pg-url",
        default=os.environ.get("PG_DB_URL", ""),
        help="覆盖数据库 URL（默认读环境变量 PG_DB_URL 否则 config.py）",
    )
    parser.add_argument(
        "--seed-long",
        default="MU,TSLA",
        help="逗号分隔，在首日首帧用 ATM CALL 现价虚拟做多，用于只看平仓链",
    )
    parser.add_argument(
        "--until",
        default="",
        help="美东日内截断 HH:MM（如 11:20），仅使用此刻及之前的 1m bar；默认 16:00",
    )
    parser.add_argument(
        "--full-journey",
        action="store_true",
        help="baseline 下逐帧记录峰值 ROI，并打印各 seed 标的虚拟开仓→峰值→首平/窗口内未平 叙事",
    )
    parser.add_argument(
        "--real-entry",
        action="store_true",
        help="走真实 decide_entry + OMS _execute_entry/_execute_exit（不再 09:30 虚拟 seed）；"
        "仅对 --seed-long 标的调用策略开仓，QQQ/SPY 仅作指数上下文",
    )
    parser.add_argument(
        "--strategy-core",
        default="",
        help="覆盖 STRATEGY_CORE_VERSION（V0 / TREND / V1），须在首次 import config 前生效；"
        "与 --real-entry 联用以对齐你本地的 V0/TREND 路由",
    )
    parser.add_argument(
        "--se-dual",
        action="store_true",
        help="SignalEngineV8(分钟 precalc alpha) → 队列 → ExecutionEngineV8；"
        "按 config.TARGET_SYMBOLS 从 PG 拉数据；需安装 torch",
    )
    parser.add_argument(
        "--pg-oms-dual",
        action="store_true",
        help="免 torch：用 PG 按 build_frames_from_df 生成 ALPHA_FRAME，直送 OMS；"
        "同样打印全日 BUY/SELL 与 realized_pnl（macd 等部分字段为实验室占位）",
    )
    args = parser.parse_args()
    if str(getattr(args, "strategy_core", "") or "").strip():
        os.environ["STRATEGY_CORE_VERSION"] = str(args.strategy_core).strip().upper()

    _bootstrap_path()

    ymd = args.date.strip() or _today_ny_ymd()
    start_ts, end_ts = ny_session_bounds_ymd(ymd)
    try:
        until_clip = parse_until_ny(ymd, args.until)
    except ValueError as e:
        print(f"[lab] {e}")
        sys.exit(2)
    if until_clip is not None:
        end_ts = min(float(end_ts), float(until_clip))

    if not args.pg_url:
        from config import PG_DB_URL  # noqa: WPS433

        pg_url = PG_DB_URL
    else:
        pg_url = args.pg_url

    primary = {"MU", "TSLA"}
    seed_syms = [x.strip().upper() for x in args.seed_long.split(",") if x.strip()]
    symbol_order = sorted(list(primary.union({"SPY", "QQQ"})))
    fetch_syms: List[str] = list(symbol_order)
    if args.se_dual or args.pg_oms_dual:
        from config import TARGET_SYMBOLS as _TS_FETCH  # noqa: WPS433

        fetch_syms = sorted(set(_TS_FETCH))

    clip_note = f" until_et={args.until.strip()}" if args.until.strip() else ""
    if args.se_dual:
        mode_note = " | mode=SE→OMS dual (TARGET_SYMBOLS)"
    elif args.pg_oms_dual:
        mode_note = " | mode=PG→ALPHA_FRAME→OMS (TARGET_SYMBOLS)"
    else:
        mode_note = f" seed_long={seed_syms}"
    print(f"[lab] PG replay date={ymd} fetch={fetch_syms}{mode_note}{clip_note}")

    df = fetch_merged_pg(start_ts, end_ts, fetch_syms, pg_url)
    if df is None or df.empty:
        print("[lab] 无数据：检查 PG 是否在会话时段有 market_bars_1m / 网络")
        sys.exit(2)

    if args.se_dual or args.pg_oms_dual:
        from config import TARGET_SYMBOLS  # noqa: WPS433

        have = {str(x).strip() for x in df["symbol"].unique()}
        sym_run = [s for s in TARGET_SYMBOLS if s in have]
        miss = [s for s in TARGET_SYMBOLS if s not in have]
        if miss:
            print(f"[lab] warn: PG 缺以下 TARGET 标的: {miss}")
        if args.pg_oms_dual:
            asyncio.run(run_pg_oms_alpha_frames_dual(df, sym_run, ymd))
        else:
            asyncio.run(run_pg_se_oms_dual(df, sym_run, ymd))
        sys.exit(0)

    df_first_rows: Dict[str, Any] = {}
    for s in seed_syms:
        sub = df[df["symbol"] == s].sort_values("ts")
        got = None
        for _, row in sub.iterrows():
            od0 = opt_data_from_buckets(parse_buckets_json(row.get("buckets_json")))
            if od0.get("has_feed"):
                got = row
                break
        if got is not None:
            df_first_rows[s] = got

    frames = build_frames_from_df(df, symbol_order)
    print(f"[lab] built {len(frames)} ALPHA_FRAME payloads")

    def _print_journey_block(base_r: Dict[str, Any], *, virtual: bool) -> None:
        from pytz import timezone as tz

        ny = tz("America/New_York")

        def _fmt_et(ts: Optional[float]) -> str:
            if ts is None or float(ts) <= 0:
                return "---"
            return datetime.fromtimestamp(float(ts), ny).strftime("%m-%d %H:%M")

        if virtual:
            print("\n=== baseline 完整路径（虚拟多 CALL，与首表同一逻辑）===")
            print(
                "说明：实验室禁 decide_entry，仅在「首根有期权桶」的分钟上虚拟开仓；"
                "entry_ts = 该根 ts − 120s（与引擎 seed 一致）。"
            )
        else:
            print("\n=== baseline 完整路径（真实 decide_entry + OMS 成交）===")
            print(
                "说明：未虚拟 seed；开仓仅在策略对 seed 标的返回 BUY 且 OMS _execute_entry 成功后记账。"
                "若与实盘 9:45 仍不一致，请核对 PG 的 alpha/macd 与实盘信号源、以及 ENTRY 流动性护栏。"
            )
        for sym in seed_syms:
            ei = (base_r.get("entry_info") or {}).get(sym)
            fe = (base_r.get("first_exit") or {}).get(sym)
            fb = (base_r.get("first_buy") or {}).get(sym)
            jp = (base_r.get("journey_peak") or {}).get(sym) or {}
            peak_dec = float(jp.get("peak_max_roi", -1.0) or -1.0)
            peak_pct = peak_dec * 100.0 if peak_dec >= 0 else None
            print(f"\n--- {sym} ---")
            if virtual:
                if not ei:
                    print(f"  未成功 seed（该窗内无有效 buckets）")
                    continue
                print(
                    f"  虚拟开仓锚点 bar（首根有桶）: {_fmt_et(ei['seed_bar_ts'])}  "
                    f"标的收盘≈{ei['entry_stock']:.2f}"
                )
                print(
                    f"  引擎 entry_ts（bar−120s）: {_fmt_et(ei['entry_ts_engine'])}  "
                    f"entry_price(opt fair)≈{ei['entry_price']:.4f}"
                )
            else:
                if not ei and not fb:
                    print("  窗口内未触发真实开仓（decide_entry 未通过或 OMS 拒单）")
                    if fe:
                        print(f"  （异常：无仓却有 SELL 记录）首平 reason={fe.get('reason')}")
                    continue
                if fb:
                    print(
                        f"  首笔 BUY（策略 reason）: {_fmt_et(fb.get('ts'))}  "
                        f"{str(fb.get('reason', ''))[:120]}"
                    )
                if ei:
                    print(
                        f"  成交后 entry_ts: {_fmt_et(ei.get('entry_ts_engine'))}  "
                        f"entry_price≈{ei['entry_price']:.4f}  标的≈{ei['entry_stock']:.2f}"
                    )
            if peak_pct is not None and peak_dec >= 0 and (ei or not virtual):
                print(
                    f"  窗口内峰值 ROI: {peak_pct:.2f}% @ {_fmt_et(jp.get('peak_ts'))} "
                    f"(引擎 st.max_roi 逐帧最大)"
                )
            if fe:
                m = fe.get("metrics") or {}
                r_exit = m.get("roi_at_exit_pct")
                p_exit = m.get("peak_max_roi_pct")
                r_s = f"{float(r_exit):.2f}" if isinstance(r_exit, (int, float)) else "?"
                p_s = f"{float(p_exit):.2f}" if isinstance(p_exit, (int, float)) else "?"
                print(
                    f"  首笔平仓: {_fmt_et(fe.get('ts'))}  "
                    f"held≈{m.get('held_mins', '?')}m  "
                    f"平仓时 ROI={r_s}%  "
                    f"记录峰值={p_s}%"
                )
                print(f"  最终由谁触发（首 SELL reason）: {fe.get('reason', '')}")
            else:
                print(
                    f"  窗口内未出现 SELL（持仓保留至最后一帧 {_fmt_et(frames[-1]['ts']) if frames else '---'}）"
                )

    if args.real_entry:
        from strategy_selector import ACTIVE_STRATEGY_CORE_VERSION  # noqa: WPS433

        print(f"[lab] real_entry=1 | ACTIVE_STRATEGY_CORE_VERSION={ACTIVE_STRATEGY_CORE_VERSION}")
        r0 = asyncio.run(
            run_profile(
                "baseline",
                {},
                frames,
                symbol_order,
                seed_syms,
                {},
                track_journey=bool(args.full_journey),
                real_entry=True,
            )
        )
        if args.full_journey:
            _print_journey_block(r0, virtual=False)
        else:
            print("\n=== 真实开仓回放摘要（加 --full-journey 可看峰值 ROI 叙事）===")
            for sym in seed_syms:
                fb = (r0.get("first_buy") or {}).get(sym)
                fe = (r0.get("first_exit") or {}).get(sym)
                if not fb:
                    print(f"{sym}: 无 BUY")
                else:
                    from pytz import timezone as tz

                    ny = tz("America/New_York")

                    def _fmt_et2(ts: Optional[float]) -> str:
                        if ts is None or float(ts) <= 0:
                            return "---"
                        return datetime.fromtimestamp(float(ts), ny).strftime("%m-%d %H:%M")

                    print(
                        f"{sym}: BUY {_fmt_et2(fb.get('ts'))} | "
                        f"{str(fb.get('reason', ''))[:100]}"
                    )
                if fe:
                    from pytz import timezone as tz

                    ny = tz("America/New_York")

                    def _fmt_et3(ts: Optional[float]) -> str:
                        if ts is None or float(ts) <= 0:
                            return "---"
                        return datetime.fromtimestamp(float(ts), ny).strftime("%m-%d %H:%M")

                    print(
                        f"      SELL {_fmt_et3(fe.get('ts'))} | "
                        f"{fe.get('reason', '')} | "
                        f"roi%={(fe.get('metrics') or {}).get('roi_at_exit_pct')}"
                    )
        sys.exit(0)

    results = []
    for pname, delta in GUARD_PROFILES.items():
        tj = bool(args.full_journey) and pname == "baseline"
        r = asyncio.run(
            run_profile(
                pname,
                delta,
                frames,
                symbol_order,
                seed_syms,
                df_first_rows,
                track_journey=tj,
                real_entry=False,
            )
        )
        results.append(r)

    if args.full_journey:
        base_r = next((x for x in results if x["profile"] == "baseline"), None)
        if base_r:
            _print_journey_block(base_r, virtual=True)

    print("\n=== 首次 SELL：reason + 持仓 + ROI（峰值 vs 平仓时）===")
    hdr = (
        f"{'profile':<20} {'sym':<5} {'exit_et':<11} {'held':>5} "
        f"{'roi%':>7} {'peak%':>7}  {'reason'}"
    )
    print(hdr)
    print("-" * len(hdr))
    from pytz import timezone as tz

    ny = tz("America/New_York")

    def fmt_ts(ts: Optional[float]) -> str:
        if ts is None:
            return "---"
        return datetime.fromtimestamp(ts, ny).strftime("%m-%d %H:%M")

    def fmt_num(x: Optional[float], width: int, prec: int = 1) -> str:
        if x is None:
            return "-" * width
        return f"{x:>{width}.{prec}f}"

    baseline_reasons = {}
    for r in results:
        if r["profile"] == "baseline":
            for sym, ev in r["first_exit"].items():
                baseline_reasons[sym] = (ev or {}).get("reason", "")

    for r in results:
        prof = r["profile"]
        for sym in seed_syms:
            fe = r["first_exit"].get(sym)
            if fe is None:
                print(f"{prof:<20} {sym:<5} {'---':<11} {'--':>5} {'--':>7} {'--':>7}  (no exit / no seed)")
                continue
            rs = fe.get("reason", "")
            m = fe.get("metrics") or {}
            held = m.get("held_mins")
            roi_p = m.get("roi_at_exit_pct")
            peak_p = m.get("peak_max_roi_pct")
            give = None
            if isinstance(roi_p, (int, float)) and isinstance(peak_p, (int, float)):
                give = float(peak_p) - float(roi_p)
            tag = ""
            bs = baseline_reasons.get(sym, "")
            if prof != "baseline" and bs and rs != bs:
                tag = "  << reason≠baseline"
            give_s = f" Δpeak-cur={give:+.1f}%" if give is not None else ""
            print(
                f"{prof:<20} {sym:<5} {fmt_ts(fe.get('ts')):<11} "
                f"{fmt_num(held, 5, 1)} {fmt_num(roi_p, 7, 2)} {fmt_num(peak_p, 7, 2)}  "
                f"{rs}{tag}{give_s}"
            )

    print("\n读表：")
    print("  · peak% 为持仓期 max_roi；若明显高于 roi%，说明平仓时仍有不少「曾到手的浮盈」被吐回或规则提前砍。")
    print("  · 对比 soft_* / hold_bias_combo 与 baseline：首平更晚且 roi% 更高 → 该方向可进 strategy_config0 微调。")
    print("  · 行尾 << reason≠baseline 表示与 baseline 首平原因不同（便于锁定敏感规则）。")


if __name__ == "__main__":
    main()
