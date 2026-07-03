#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层规则栈原型 —— 不依赖 TFT 的重设计验证(对拍旧 proxy+V0 栈)。

  v1: L0 状态机 + L1 z-score + L2 持续性 + L3 风险预算
  v3: Open Drive(09:45) + 规则 dual leg + 可选 straddle + Midday + 共用 L3/L5

用法:
  python tools/prototype_rule_stack.py --stack all --plot reports/rule_stack_ab.png
  python tools/prototype_rule_stack.py --stack v3 --dates 2026-07-02
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_BASELINE = Path(__file__).resolve().parents[1]
_REPO = _BASELINE.parents[1]
for p in (str(_REPO), str(_BASELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 旧栈依赖的环境必须在 strategy 包 import 前设置
os.environ.setdefault("QQQ_BTC_LIVE", "1")
os.environ.setdefault("FAST_GATE_ENABLED", "0")
os.environ.setdefault("COOLDOWN_MINUTES", "5")

import baseline_paths  # noqa: E402,F401

from tools.validate_gates_realday import (  # noqa: E402
    MinuteRow,
    _build_ctx,
    _fetch_minute_aggs,
    _occ_call,
    _pick_atm_strike,
    _polygon_key,
    _roc,
    _rth_overlap,
    NY,
    proxy_edge,
)
from qqq_btc.common.exit_rails import ExitRailsConfig, PositionState, check_exit  # noqa: E402
from qqq_btc.qqq import config as qcfg  # noqa: E402

CACHE_DIR = _BASELINE / "reports" / "replay_cache"

# 趋势模式退出轨道:入场都发生在 L0=TREND, 用比 QQQ 默认更宽的阶梯让利润奔跑;
# 止损侧保持不变(风险由 L3/L4 管), 只放宽利润保护档。
TREND_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.12,
    soft_stop_roi=-0.08,
    early_stop_bars=5,
    early_stop_roi=-0.05,
    time_stop_bars=20,
    time_stop_min_roi=0.03,
    max_hold_bars=45,
    trailing_trigger_roi=0.25,
    trailing_keep_ratio=0.55,
    ladder=(
        (0.12, 0.06),
        (0.20, 0.12),
        (0.35, 0.22),
        (0.50, 0.35),
    ),
    flash_trigger_roi=0.12,
    flash_exit_roi=0.03,
    eod_close_bar_index=380,
)


# =========================================================================
# 数据
# =========================================================================

def load_day(symbol: str, date: str, key: str) -> pd.DataFrame:
    """股票 + ATM Call 分钟线, 本地 parquet 缓存避免反复打 Polygon。"""
    cache = CACHE_DIR / f"proto_{symbol}_{date}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    from datetime import datetime

    stock_rows = _fetch_minute_aggs(symbol, date, key)
    if not stock_rows:
        raise RuntimeError(f"无股票分钟线: {symbol} {date}")
    open_px = float(stock_rows[0].get("o") or stock_rows[0].get("c") or 0)
    strike = _pick_atm_strike(open_px)
    yymmdd = date.replace("-", "")[2:]
    opt_ticker = _occ_call(symbol, yymmdd, strike)
    opt_rows = _fetch_minute_aggs(opt_ticker, date, key)
    if not opt_rows:
        raise RuntimeError(f"无期权分钟线: {opt_ticker}")

    keys = _rth_overlap(stock_rows, opt_rows)
    s_by, o_by = {r["t"]: r for r in stock_rows}, {r["t"]: r for r in opt_rows}
    recs = []
    for k in keys:
        sr, orow = s_by[k], o_by[k]
        ts = datetime.fromtimestamp(k / 1000, NY)
        sc = float(sr.get("c") or 0)
        bid = float(orow.get("l") or orow.get("c") or 0)
        ask = float(orow.get("h") or orow.get("c") or 0)
        mid = float(orow.get("c") or (bid + ask) / 2)
        if bid <= 0 or ask <= 0 or mid <= 0:
            bid, ask = mid * 0.98, mid * 1.02
        recs.append({
            "ts_ms": k,
            "time": ts,
            "stock_close": sc,
            "stock_vw": float(sr.get("vw") or sc),
            "stock_vol": float(sr.get("v") or 0),
            "stock_high": float(sr.get("h") or sc),
            "stock_low": float(sr.get("l") or sc),
            "opt_bid": bid,
            "opt_ask": ask,
            "opt_mid": mid,
            "spread_pct": (ask - bid) / mid if mid > 0 else 0.0,
            "opt_ticker": opt_ticker,
        })
    df = pd.DataFrame(recs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# =========================================================================
# 特征(L0/L1/L2 共用)
# =========================================================================

def add_features(df: pd.DataFrame, er_window: int = 15, or_minutes: int = 15) -> pd.DataFrame:
    d = df.copy()
    c = d["stock_close"]
    # session bar(距 09:30 的分钟数)
    d["session_bar"] = [(t.hour - 9) * 60 + t.minute - 30 for t in d["time"]]
    # VWAP(成交量加权累计)
    pv = (d["stock_vw"] * d["stock_vol"]).cumsum()
    vv = d["stock_vol"].cumsum().replace(0, np.nan)
    d["vwap"] = (pv / vv).fillna(c)
    # 开盘区间(前 or_minutes 根 RTH bar)
    or_n = min(or_minutes, len(d))
    d["or_high"] = d["stock_high"].iloc[:or_n].max()
    # L0: 效率比(位移/路程)
    diff = c.diff().abs()
    d["er30"] = (c - c.shift(er_window)).abs() / diff.rolling(er_window).sum().replace(0, np.nan)
    d["above_vwap_frac30"] = (c > d["vwap"]).rolling(er_window).mean()
    # L1: 噪声归一化 z
    ret1 = c.pct_change()
    sigma1 = ret1.rolling(20).std().clip(lower=1e-5)
    disp5 = c.pct_change(5)
    d["z5"] = disp5 / (sigma1 * np.sqrt(5))
    # L2: 持续性
    d["up6"] = (ret1 > 0).rolling(6).sum()
    return d


# =========================================================================
# 新栈配置与模拟
# =========================================================================

@dataclass
class NewStackConfig:
    # L0
    er_enter: float = 0.25
    er_exit: float = 0.15
    vwap_frac_min: float = 0.65
    state_confirm_bars: int = 2
    # L1
    z_min: float = 1.5
    # L2
    up_min_of6: int = 4
    # 执行
    max_spread_pct: float = 0.06
    entry_end_bar: int = 360        # 15:30 后禁新仓
    warmup_bars: int = 20
    # L3 风险预算
    max_trades_per_day: int = 4
    daily_loss_stop_usd: float = -3000.0   # 账户 -6% (5 万)
    loss_streak_halt: int = 2              # 连续 2 笔亏损 → 当日停机
    cooldown_after_loss_min: int = 20
    cooldown_after_win_min: int = 8
    # L4 仓位
    account: float = 50000.0
    risk_frac_per_trade: float = 0.015     # 每笔风险 1.5% 账户


@dataclass
class DayResult:
    date: str
    trades: List[dict] = field(default_factory=list)
    pnl_usd: float = 0.0
    blocks: dict = field(default_factory=dict)
    halted_reason: str = ""

    @property
    def roi(self) -> float:
        return self.pnl_usd / 50000.0


def run_new_stack(df: pd.DataFrame, cfg: NewStackConfig) -> DayResult:
    d = add_features(df)
    fm, rails = qcfg.FILL_MODEL, TREND_RAILS
    res = DayResult(date=str(d["time"].iloc[0].date()))
    blocks: dict = {}

    state = "CHOP"
    confirm = 0
    position: Optional[dict] = None
    cooldown_until_bar = -1
    loss_streak = 0
    halted = False

    def blk(reason: str):
        blocks[reason] = blocks.get(reason, 0) + 1

    for i in range(len(d)):
        row = d.iloc[i]
        sb = int(row["session_bar"])

        # ---- L0 状态机(每 bar 更新, 带迟滞) ----
        er = row["er30"]
        trend_raw = (
            np.isfinite(er)
            and er >= cfg.er_enter
            and row["stock_close"] > row["vwap"]
            and row["above_vwap_frac30"] >= cfg.vwap_frac_min
        )
        if state != "TREND":
            confirm = confirm + 1 if trend_raw else 0
            if confirm >= cfg.state_confirm_bars:
                state = "TREND"
                confirm = 0
        else:
            exit_raw = (np.isfinite(er) and er < cfg.er_exit) or row["stock_close"] < row["vwap"]
            confirm = confirm + 1 if exit_raw else 0
            if confirm >= cfg.state_confirm_bars:
                state = "CHOP"
                confirm = 0

        # ---- 持仓: L5 退出轨道 ----
        if position is not None:
            reason = check_exit(rails, position["state"], float(row["opt_mid"]), i, sb)
            if reason:
                exit_px = fm.exit_fill(float(row["opt_bid"]), float(row["opt_ask"]))
                if not np.isfinite(exit_px) or exit_px <= 0:
                    exit_px = float(row["opt_mid"])
                net = exit_px / position["entry_px"] - 1.0 - fm.commission_return_drag(position["entry_px"])
                pnl = position["contracts"] * 100.0 * position["entry_px"] * net
                res.trades.append({
                    "entry": position["ets"], "exit": row["time"], "net": float(net),
                    "pnl": float(pnl), "reason": reason, "contracts": position["contracts"],
                    "entry_px": position["entry_px"],
                })
                res.pnl_usd += pnl
                if net < 0:
                    loss_streak += 1
                    cooldown_until_bar = i + cfg.cooldown_after_loss_min
                else:
                    loss_streak = 0
                    cooldown_until_bar = i + cfg.cooldown_after_win_min
                position = None
                # L3 熔断检查
                if loss_streak >= cfg.loss_streak_halt:
                    halted = True
                    res.halted_reason = f"loss_streak>={cfg.loss_streak_halt}"
                if res.pnl_usd <= cfg.daily_loss_stop_usd:
                    halted = True
                    res.halted_reason = "daily_loss_stop"
            continue

        # ---- 无持仓: 逐层入场检查 ----
        if halted:
            blk("L3.halted")
            continue
        if len(res.trades) >= cfg.max_trades_per_day:
            blk("L3.max_trades")
            continue
        if i <= cooldown_until_bar:
            blk("L3.cooldown")
            continue
        if i < cfg.warmup_bars:
            blk("warmup")
            continue
        if sb > cfg.entry_end_bar:
            blk("session_end")
            continue
        if state != "TREND":
            blk("L0.chop")
            continue
        if not (np.isfinite(row["z5"]) and row["z5"] >= cfg.z_min):
            blk("L1.z_low")
            continue
        if row["up6"] < cfg.up_min_of6:
            blk("L2.persistence")
            continue
        if not (row["stock_close"] > row["vwap"] and row["stock_close"] > row["or_high"]):
            blk("L2.structure")
            continue
        if row["spread_pct"] > cfg.max_spread_pct:
            blk("exec.spread")
            continue

        # ---- 入场(L4 风险定仓) ----
        entry_px = fm.entry_fill(float(row["opt_bid"]), float(row["opt_ask"]))
        if not np.isfinite(entry_px) or entry_px <= 0:
            blk("exec.bad_fill")
            continue
        risk_budget = cfg.account * cfg.risk_frac_per_trade
        risk_per_contract = abs(rails.hard_stop_roi) * entry_px * 100.0
        contracts = max(1, int(risk_budget / risk_per_contract))
        position = {
            "entry_px": float(entry_px),
            "contracts": contracts,
            "ets": row["time"],
            "state": PositionState(entry_price=float(entry_px), entry_bar=i),
        }

    # EOD 未平仓强平(rails eod 兜底外的保险)
    if position is not None:
        last = d.iloc[-1]
        exit_px = fm.exit_fill(float(last["opt_bid"]), float(last["opt_ask"]))
        net = exit_px / position["entry_px"] - 1.0 - fm.commission_return_drag(position["entry_px"])
        pnl = position["contracts"] * 100.0 * position["entry_px"] * net
        res.trades.append({
            "entry": position["ets"], "exit": last["time"], "net": float(net),
            "pnl": float(pnl), "reason": "FORCE_EOD", "contracts": position["contracts"],
            "entry_px": position["entry_px"],
        })
        res.pnl_usd += pnl

    res.blocks = blocks
    return res


# =========================================================================
# 旧栈(proxy + V0 门控 + 90% 全仓, 无预算) —— 作为对照
# =========================================================================

def run_old_stack(df: pd.DataFrame) -> DayResult:
    from strategy.config0 import StrategyConfig
    from strategy.core_v0 import StrategyCoreV0
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch
    from qqq_btc.common.time_features import session_minute

    cfg = StrategyConfig()
    cfg.FAST_GATE_ENABLED = False
    apply_strategy_entry_patch(StrategyCoreV0)
    core = StrategyCoreV0(cfg)
    fm, rails = qcfg.FILL_MODEL, qcfg.EXIT_RAILS
    res = DayResult(date=str(df["time"].iloc[0].date()))

    prev_close = None
    rows: List[MinuteRow] = []
    for i in range(len(df)):
        r = df.iloc[i]
        sc = float(r["stock_close"])
        snap = _roc(sc, prev_close) if prev_close else 0.0
        c5 = float(df["stock_close"].iloc[i - 5]) if i >= 5 else None
        roc5 = _roc(sc, c5) if c5 else 0.0
        edge, q10 = proxy_edge(snap, roc5)
        rows.append(MinuteRow(
            int(r["ts_ms"]), r["time"], sc, float(r["opt_bid"]), float(r["opt_ask"]),
            float(r["opt_mid"]), float(r["spread_pct"]), float(r["spread_pct"]) * 1.05,
            snap, roc5, edge, q10,
        ))
        prev_close = sc

    slot = cfg.INITIAL_ACCOUNT * cfg.POSITION_RATIO
    cooldown_until = 0.0
    position: Optional[dict] = None
    for i, row in enumerate(rows):
        sb = int(session_minute(pd.Series([pd.Timestamp(row.time)])).iloc[0])
        ctx = _build_ctx(row, fast_gate_on=False)
        ctx["position"] = 1 if position else 0
        ctx["cooldown_until"] = cooldown_until
        if position:
            reason = check_exit(rails, position["state"], row.opt_mid, i, sb)
            if reason:
                exit_px = fm.exit_fill(row.opt_bid, row.opt_ask)
                if not np.isfinite(exit_px) or exit_px <= 0:
                    exit_px = row.opt_mid
                net = exit_px / position["entry_px"] - 1.0 - fm.commission_return_drag(position["entry_px"])
                res.trades.append({
                    "entry": position["ets"], "exit": row.time, "net": float(net),
                    "pnl": float(net * slot), "reason": reason, "contracts": 0,
                    "entry_px": position["entry_px"],
                })
                res.pnl_usd += net * slot
                position = None
                cooldown_until = row.ts_ms / 1000.0 + cfg.COOLDOWN_MINUTES * 60
        elif core.decide_entry(ctx):
            entry_px = fm.entry_fill(row.opt_bid, row.opt_ask)
            if not np.isfinite(entry_px) or entry_px <= 0:
                entry_px = row.opt_mid
            position = {
                "entry_px": float(entry_px), "ets": row.time,
                "state": PositionState(entry_price=float(entry_px), entry_bar=i),
            }
            cooldown_until = row.ts_ms / 1000.0 + cfg.COOLDOWN_MINUTES * 60
    if position:
        last = rows[-1]
        exit_px = fm.exit_fill(last.opt_bid, last.opt_ask)
        net = exit_px / position["entry_px"] - 1.0 - fm.commission_return_drag(position["entry_px"])
        res.trades.append({
            "entry": position["ets"], "exit": last.time, "net": float(net),
            "pnl": float(net * slot), "reason": "FORCE_EOD", "contracts": 0,
            "entry_px": position["entry_px"],
        })
        res.pnl_usd += net * slot
    return res


# =========================================================================
# v3: Open Drive + 规则 dual leg + 可选 straddle
# =========================================================================

def _occ_put(symbol: str, yymmdd: str, strike: float) -> str:
    strike_int = int(round(strike * 1000))
    return f"O:{symbol}{yymmdd}P{strike_int:08d}"


def _strike_step(open_px: float) -> float:
    return 2.5 if open_px < 200 else 5.0


RUNNER_RAILS = ExitRailsConfig(
    hard_stop_roi=-0.18,
    soft_stop_roi=-0.12,
    early_stop_bars=None,
    time_stop_bars=999,
    time_stop_min_roi=-1.0,
    max_hold_bars=999,
    trailing_trigger_roi=0.30,
    trailing_keep_ratio=0.45,
    ladder=((0.20, 0.10), (0.40, 0.25), (0.70, 0.45)),
    flash_trigger_roi=999.0,
    flash_exit_roi=0.0,
    eod_close_bar_index=380,
)


@dataclass
class V3Config:
    account: float = 50000.0
    or_bars: int = 16                    # 09:30-09:45
    open_drive_end_bar: int = 60         # 10:30 前算 Open Drive 窗口
    midday_end_bar: int = 360
    score_threshold: float = 0.35        # dual leg 最低分
    straddle_ambiguity: float = 0.25     # |call-put| 低于此 → 方向不明
    straddle_or_range_pct: float = 0.004 # OR 振幅 > 0.4% 才考虑跨式
    slope_min: float = 0.04              # $/min 线性回归斜率门槛
    vol_ratio_min: float = 0.85          # 开盘量门槛(日内中位数代理, 不宜过高)
    max_trades_per_day: int = 4
    max_straddles_per_day: int = 1
    daily_loss_stop_usd: float = -2500.0
    loss_streak_chop_halt: int = 2
    open_drive_premium_frac: float = 0.05   # 账户 5% 权利金(Open Drive OTM)
    straddle_premium_frac: float = 0.025
    midday_risk_frac: float = 0.015
    cooldown_loss: int = 25
    cooldown_win: int = 10
    max_spread_pct: float = 0.08
    open_drive_max_spread_pct: float = 0.30   # OTM 开盘 Polygon bid/ask 失真, 原型放宽


def load_day_v3(symbol: str, date: str, key: str) -> pd.DataFrame:
    """股票 + ATM Call + OTM+1 Call + ATM Put, parquet 缓存。"""
    cache = CACHE_DIR / f"proto_v3_{symbol}_{date}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import time
    from datetime import datetime

    stock_rows = _fetch_minute_aggs(symbol, date, key)
    if not stock_rows:
        raise RuntimeError(f"无股票分钟线: {symbol} {date}")
    open_px = float(stock_rows[0].get("o") or stock_rows[0].get("c") or 0)
    step = _strike_step(open_px)
    atm = _pick_atm_strike(open_px)
    otm = atm + step
    yymmdd = date.replace("-", "")[2:]
    tickers = {
        "call_atm": _occ_call(symbol, yymmdd, atm),
        "call_otm": _occ_call(symbol, yymmdd, otm),
        "put_atm": _occ_put(symbol, yymmdd, atm),
    }
    opt_data = {}
    for leg, tk in tickers.items():
        time.sleep(0.4)
        opt_data[leg] = _fetch_minute_aggs(tk, date, key)
        if not opt_data[leg]:
            raise RuntimeError(f"无期权数据 {tk}")

    keys = _rth_overlap(stock_rows, opt_data["call_atm"])
    for leg in ("call_otm", "put_atm"):
        keys = sorted(set(keys) & {r["t"] for r in opt_data[leg]})
    s_by = {r["t"]: r for r in stock_rows}

    def _leg_by(rows):
        return {r["t"]: r for r in rows}

    legs = {k: _leg_by(opt_data[k]) for k in opt_data}

    recs = []
    for k in keys:
        sr = s_by[k]
        ts = datetime.fromtimestamp(k / 1000, NY)
        sc = float(sr.get("c") or 0)
        row = {
            "ts_ms": k, "time": ts, "stock_close": sc,
            "stock_vw": float(sr.get("vw") or sc),
            "stock_vol": float(sr.get("v") or 0),
            "stock_high": float(sr.get("h") or sc),
            "stock_low": float(sr.get("l") or sc),
            "atm_strike": atm, "otm_strike": otm,
        }
        for leg in ("call_atm", "call_otm", "put_atm"):
            orow = legs[leg].get(k)
            if orow is None:
                continue
            bid = float(orow.get("l") or orow.get("c") or 0)
            ask = float(orow.get("h") or orow.get("c") or 0)
            mid = float(orow.get("c") or (bid + ask) / 2)
            if bid <= 0 or ask <= 0 or mid <= 0:
                bid, ask = mid * 0.98, mid * 1.02
            row[f"{leg}_bid"] = bid
            row[f"{leg}_ask"] = ask
            row[f"{leg}_mid"] = mid
            row[f"{leg}_spread"] = (ask - bid) / mid if mid > 0 else 0.0
        recs.append(row)
    df = pd.DataFrame(recs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _dual_scores(d: pd.DataFrame, i: int, n: int) -> tuple[float, float, dict]:
    """规则版 call/put 分数(正股 linreg + 资金流 + VWAP 结构)。"""
    if i + 1 < n:
        return 0.0, 0.0, {}
    c = d["stock_close"].values[: i + 1]
    v = d["stock_vol"].values[: i + 1]
    vw = d["stock_vw"].values[: i + 1]
    win = c[-n:]
    x = np.arange(len(win))
    slope = float(np.polyfit(x, win, 1)[0])
    ret = np.diff(win, prepend=win[0])
    mf = float(np.sum(v[-n:] * np.sign(ret)))
    mf_norm = mf / max(float(np.sum(v[-n:])), 1.0)
    pv = float(np.sum(vw[-n:] * v[-n:]))
    vv = float(np.sum(v[-n:]))
    vwap = pv / vv if vv > 0 else float(win[-1])
    above = 1.0 if win[-1] > vwap else 0.0
    below = 1.0 if win[-1] < vwap else 0.0
    slope_norm = slope / max(float(win[0]), 1.0) * 100.0
    call_score = 0.45 * slope_norm + 0.35 * mf_norm * 10 + 0.20 * above
    put_score = 0.45 * (-slope_norm) + 0.35 * (-mf_norm) * 10 + 0.20 * below
    meta = {"slope": slope, "mf": mf, "vwap": vwap, "slope_norm": slope_norm}
    return call_score, put_score, meta


def _contracts_for_premium(fm, cfg: V3Config, entry_px: float, premium_frac: float, stop_roi: float) -> int:
    budget = cfg.account * premium_frac
    if entry_px <= 0:
        return 0
    # 按目标权利金预算定张数, 同时受 hard stop 约束上限
    by_premium = int(budget / (entry_px * 100.0))
    by_risk = int(cfg.account * cfg.midday_risk_frac / (abs(stop_roi) * entry_px * 100.0))
    return max(1, min(by_premium, max(by_risk, 1)))


def run_v3_stack(df: pd.DataFrame, cfg: V3Config) -> DayResult:
    d = add_features(df)
    d["session_bar"] = [(t.hour - 9) * 60 + t.minute - 30 for t in d["time"]]
    fm = qcfg.FILL_MODEL
    res = DayResult(date=str(d["time"].iloc[0].date()))
    blocks: dict = {}
    vol_med = float(d["stock_vol"].median()) if len(d) else 1.0

    position: Optional[dict] = None
    cooldown_until = -1
    loss_streak = 0
    halted = False
    straddles_today = 0
    open_drive_done = False
    day_trend_confirmed = False

    def blk(r: str):
        blocks[r] = blocks.get(r, 0) + 1

    def close_position(i: int, reason: str, rails: ExitRailsConfig):
        nonlocal position, cooldown_until, loss_streak, halted
        row = d.iloc[i]
        leg = position["leg"]
        if leg == "STRADDLE":
            exit_px = (
                fm.exit_fill(row["call_atm_bid"], row["call_atm_ask"])
                + fm.exit_fill(row["put_atm_bid"], row["put_atm_ask"])
            )
            entry_px = position["entry_px"]
        elif leg == "PUT":
            exit_px = fm.exit_fill(row["put_atm_bid"], row["put_atm_ask"])
            entry_px = position["entry_px"]
        elif leg == "CALL_OTM":
            exit_px = fm.exit_fill(row["call_otm_bid"], row["call_otm_ask"])
            entry_px = position["entry_px"]
        else:
            exit_px = fm.exit_fill(row["call_atm_bid"], row["call_atm_ask"])
            entry_px = position["entry_px"]
        if not np.isfinite(exit_px) or exit_px <= 0:
            exit_px = entry_px
        net = exit_px / entry_px - 1.0 - fm.commission_return_drag(entry_px)
        if leg == "STRADDLE":
            net -= fm.commission_return_drag(entry_px)  # 第二腿佣金(合成价)
        pnl = position["contracts"] * 100.0 * entry_px * net
        res.trades.append({
            "entry": position["ets"], "exit": row["time"], "net": float(net),
            "pnl": float(pnl), "reason": reason, "contracts": position["contracts"],
            "leg": leg, "playbook": position["playbook"], "entry_px": entry_px,
        })
        res.pnl_usd += pnl
        if net < 0:
            loss_streak += 1
            cooldown_until = i + cfg.cooldown_loss
        else:
            loss_streak = 0
            cooldown_until = i + cfg.cooldown_win
        position = None
        if not day_trend_confirmed and loss_streak >= cfg.loss_streak_chop_halt:
            halted = True
            res.halted_reason = "chop_loss_streak"
        if res.pnl_usd <= cfg.daily_loss_stop_usd:
            halted = True
            res.halted_reason = "daily_loss_stop"

    for i in range(len(d)):
        row = d.iloc[i]
        sb = int(row["session_bar"])

        # ---- 持仓管理 ----
        if position is not None:
            rails = position["rails"]
            if position["leg"] == "STRADDLE":
                mtm = float(row["call_atm_mid"]) + float(row["put_atm_mid"])
            elif position["leg"] == "PUT":
                mtm = float(row["put_atm_mid"])
            elif position["leg"] == "CALL_OTM":
                mtm = float(row["call_otm_mid"])
            else:
                mtm = float(row["call_atm_mid"])
            # Open Drive: VWAP 失守额外离场
            vwap_exit = (
                position["playbook"] == "open_drive"
                and row["stock_close"] < row["vwap"]
                and sb >= 20
            )
            reason = check_exit(rails, position["state"], mtm, i, sb)
            if vwap_exit and reason is None:
                reason = "VWAP_LOSS"
            if reason:
                close_position(i, reason, rails)
            continue

        if halted:
            blk("L3.halt")
            continue
        if len(res.trades) >= cfg.max_trades_per_day:
            blk("L3.max_trades")
            continue
        if i <= cooldown_until:
            blk("L3.cooldown")
            continue

        cs, ps, meta = _dual_scores(d, i, cfg.or_bars)
        vol_ratio = float(d["stock_vol"].iloc[max(0, i - cfg.or_bars + 1): i + 1].sum()) / max(vol_med * cfg.or_bars, 1)

        # ---- Playbook A: Open Drive @ 09:45 (bar 15) ----
        if sb == cfg.or_bars - 1 and not open_drive_done and sb <= cfg.open_drive_end_bar:
            open_drive_done = True
            n_or = cfg.or_bars
            # OR 不含当前 bar(09:45 判断用 09:30-09:44 区间)
            or_slice = d.iloc[: max(1, i)]
            or_hi = float(or_slice["stock_high"].max())
            or_lo = float(or_slice["stock_low"].min())
            or_range = (or_hi - or_lo) / max(float(d["stock_close"].iloc[0]), 1.0)
            breakout_up = float(row["stock_close"]) > or_hi * 0.999
            breakout_dn = float(row["stock_close"]) < or_lo * 1.001
            ambiguous = abs(cs - ps) < cfg.straddle_ambiguity
            vol_ok = vol_ratio >= cfg.vol_ratio_min

            chosen = None
            if ambiguous and or_range >= cfg.straddle_or_range_pct and vol_ok and straddles_today < cfg.max_straddles_per_day:
                cb = float(row["call_atm_bid"]); ca = float(row["call_atm_ask"])
                pb = float(row["put_atm_bid"]); pa = float(row["put_atm_ask"])
                entry_px = fm.entry_fill(cb, ca) + fm.entry_fill(pb, pa)
                spread_ok = (ca - cb) / max(row["call_atm_mid"], 0.01) <= cfg.max_spread_pct
                spread_ok &= (pa - pb) / max(row["put_atm_mid"], 0.01) <= cfg.max_spread_pct
                if np.isfinite(entry_px) and entry_px > 0 and spread_ok:
                    chosen = ("STRADDLE", entry_px, cfg.straddle_premium_frac, RUNNER_RAILS)
                    straddles_today += 1
            elif cs >= cfg.score_threshold and cs > ps and meta.get("slope", 0) >= cfg.slope_min and breakout_up and vol_ok:
                bid, ask, mid = float(row["call_otm_bid"]), float(row["call_otm_ask"]), float(row["call_otm_mid"])
                sp = (ask - bid) / max(mid, 0.01)
                if sp <= cfg.open_drive_max_spread_pct:
                    entry_px = fm.entry_fill(bid, ask)
                    if (not np.isfinite(entry_px) or entry_px <= 0) and mid > 0:
                        entry_px = mid  # 极端 spread 时用 mid 近似(原型)
                    if np.isfinite(entry_px) and entry_px > 0:
                        chosen = ("CALL_OTM", entry_px, cfg.open_drive_premium_frac, RUNNER_RAILS)
                        day_trend_confirmed = True
            elif ps >= cfg.score_threshold and ps > cs and meta.get("slope", 0) <= -cfg.slope_min and breakout_dn and vol_ok:
                bid, ask, mid = float(row["put_atm_bid"]), float(row["put_atm_ask"]), float(row["put_atm_mid"])
                sp = (ask - bid) / max(mid, 0.01)
                if sp <= cfg.open_drive_max_spread_pct:
                    entry_px = fm.entry_fill(bid, ask)
                    if (not np.isfinite(entry_px) or entry_px <= 0) and mid > 0:
                        entry_px = mid
                    if np.isfinite(entry_px) and entry_px > 0:
                        chosen = ("PUT", entry_px, cfg.open_drive_premium_frac, RUNNER_RAILS)
                        day_trend_confirmed = True
            else:
                blk("open.no_signal")

            if chosen:
                leg, entry_px, prem_frac, rails = chosen
                contracts = _contracts_for_premium(fm, cfg, entry_px, prem_frac, rails.hard_stop_roi)
                position = {
                    "leg": leg, "entry_px": float(entry_px), "contracts": contracts,
                    "ets": row["time"], "playbook": "open_drive", "rails": rails,
                    "state": PositionState(entry_price=float(entry_px), entry_bar=i),
                }
            continue

        # Midday: 仅在 Open Drive 已盈利 or 日趋势确认后才允许(防震荡日反复换边)
        if sb < cfg.open_drive_end_bar or sb > cfg.midday_end_bar:
            blk("midday.window")
            continue
        if not day_trend_confirmed and not any(t.get("pnl", 0) > 0 for t in res.trades):
            blk("midday.no_day_trend")
            continue
        if cs < cfg.score_threshold and ps < cfg.score_threshold:
            blk("midday.score_low")
            continue
        if cs >= ps and cs >= cfg.score_threshold:
            leg, bid, ask, mid = "CALL_ATM", float(row["call_atm_bid"]), float(row["call_atm_ask"]), float(row["call_atm_mid"])
        elif ps > cs and ps >= cfg.score_threshold:
            leg, bid, ask, mid = "PUT", float(row["put_atm_bid"]), float(row["put_atm_ask"]), float(row["put_atm_mid"])
        else:
            blk("midday.no_side")
            continue
        if (ask - bid) / max(mid, 0.01) > cfg.max_spread_pct:
            blk("midday.spread")
            continue
        if not (row["stock_close"] > row["vwap"] if leg != "PUT" else row["stock_close"] < row["vwap"]):
            blk("midday.vwap")
            continue
        entry_px = fm.entry_fill(bid, ask)
        if not np.isfinite(entry_px) or entry_px <= 0:
            blk("midday.fill")
            continue
        contracts = _contracts_for_premium(fm, cfg, entry_px, cfg.midday_risk_frac, TREND_RAILS.hard_stop_roi)
        position = {
            "leg": leg, "entry_px": float(entry_px), "contracts": contracts,
            "ets": row["time"], "playbook": "midday", "rails": TREND_RAILS,
            "state": PositionState(entry_price=float(entry_px), entry_bar=i),
        }

    if position is not None:
        close_position(len(d) - 1, "FORCE_EOD", position["rails"])

    res.blocks = blocks
    return res


# =========================================================================
# main
# =========================================================================

DEFAULT_DATES = [
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
    "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
]


def _df_for_atm(df: pd.DataFrame) -> pd.DataFrame:
    """v3 宽表 → v1/旧栈用的 ATM 列。"""
    if "call_atm_mid" not in df.columns:
        return df
    out = df.copy()
    out["opt_bid"] = out["call_atm_bid"]
    out["opt_ask"] = out["call_atm_ask"]
    out["opt_mid"] = out["call_atm_mid"]
    out["spread_pct"] = out["call_atm_spread"]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="分层规则栈原型对拍(v1 / v3 / 旧)")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--stack", default="all", choices=("all", "v1", "v3", "old"))
    ap.add_argument("--plot", default="", help="输出权益曲线 png 路径")
    args = ap.parse_args(argv)

    key = _polygon_key()
    ncfg = NewStackConfig()
    v3cfg = V3Config()

    rows_out = []
    for date in args.dates:
        try:
            if args.stack in ("v3", "all"):
                df_raw = load_day_v3(args.symbol.upper(), date, key)
            else:
                df_raw = load_day(args.symbol.upper(), date, key)
        except Exception as exc:
            print(f"{date}  SKIP: {exc}")
            continue
        df_atm = _df_for_atm(df_raw)
        old = run_old_stack(df_atm) if args.stack in ("all", "old") else None
        v1 = run_new_stack(df_atm, ncfg) if args.stack in ("all", "v1") else None
        v3 = run_v3_stack(df_raw, v3cfg) if args.stack in ("all", "v3") else None
        rows_out.append((date, old, v1, v3))

    if args.stack == "all":
        print(f"\n=== {args.symbol} 三栈对拍(账户 $50k) ===")
        print(f"{'date':12s} {'旧ROI':>8s} {'v1ROI':>8s} {'v3ROI':>8s}  v3:笔 主拦截")
        tot = {"old": 0.0, "v1": 0.0, "v3": 0.0}
        for date, old, v1, v3 in rows_out:
            tot["old"] += old.pnl_usd if old else 0
            tot["v1"] += v1.pnl_usd if v1 else 0
            tot["v3"] += v3.pnl_usd if v3 else 0
            top = max(v3.blocks.items(), key=lambda x: x[1])[0] if v3 and v3.blocks else "-"
            halt = f"[{v3.halted_reason}]" if v3 and v3.halted_reason else ""
            print(f"{date:12s} {100*old.roi:+7.2f}% {100*v1.roi:+7.2f}% {100*v3.roi:+7.2f}%  "
                  f"{len(v3.trades) if v3 else 0:2d}  {halt}{top}")
        n = max(1, len(rows_out))
        print(f"\n合计: 旧{100*tot['old']/50000:+.1f}% | v1{100*tot['v1']/50000:+.1f}% | v3{100*tot['v3']/50000:+.1f}%")
        print(f"日均: 旧{100*tot['old']/50000/n:+.2f}% | v1{100*tot['v1']/50000/n:+.2f}% | v3{100*tot['v3']/50000/n:+.2f}%")
        worst = min((v3.roi for _, _, _, v3 in rows_out if v3), default=0)
        print(f"最差单日 v3: {100*worst:+.2f}%")
    elif args.stack == "v3":
        print(f"\n=== v3 Open Drive + dual leg ===")
        for date, _, _, v3 in rows_out:
            if v3 is None:
                continue
            print(f"{date} ROI={100*v3.roi:+.2f}% trades={len(v3.trades)} halt={v3.halted_reason or '-'}")
            for t in v3.trades:
                print(f"  [{t.get('playbook','?')}] {t.get('leg','?')} "
                      f"{t['entry'].strftime('%H:%M')}->{t['exit'].strftime('%H:%M')} "
                      f"{100*t['net']:+.1f}% x{t['contracts']} ${t['pnl']:+.0f} {t['reason']}")
    else:
        # legacy v1-only output
        print(f"\n=== {args.symbol} 新旧规则栈对拍 ===")
        for date, old, v1, _ in rows_out:
            if old and v1:
                print(f"{date:12s} old={100*old.roi:+.2f}% v1={100*v1.roi:+.2f}%")

    print("\n--- v3 逐笔 ---")
    for date, _, _, v3 in rows_out:
        if not v3 or not v3.trades:
            continue
        print(f"  {date}:")
        for t in v3.trades:
            print(f"    [{t.get('playbook','?')}] {t.get('leg','?')} "
                  f"{t['entry'].strftime('%H:%M')}->{t['exit'].strftime('%H:%M')} "
                  f"{100*t['net']:+6.1f}% x{t['contracts']} ${t['pnl']:+7.0f} {t['reason']}")

    if args.plot and args.stack == "all":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eq_o, eq_v1, eq_v3, labels = [50000.0], [50000.0], [50000.0], [""]
        for date, old, v1, v3 in rows_out:
            eq_o.append(eq_o[-1] + (old.pnl_usd if old else 0))
            eq_v1.append(eq_v1[-1] + (v1.pnl_usd if v1 else 0))
            eq_v3.append(eq_v3[-1] + (v3.pnl_usd if v3 else 0))
            labels.append(date[5:])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = range(len(eq_o))
        ax.plot(x, eq_o, "o-", color="#d9534f", label="old: proxy+V0")
        ax.plot(x, eq_v1, "o-", color="#5bc0de", label="v1: layered stack")
        ax.plot(x, eq_v3, "o-", color="#5cb85c", label="v3: open drive + dual leg")
        ax.axhline(50000, color="#999", lw=0.8, ls="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel("account equity ($)")
        ax.set_title(f"{args.symbol} 0DTE — rule stack A/B/C ({len(rows_out)} days)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\n图已保存: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
