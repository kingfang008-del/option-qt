#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开平仓生命周期对拍 —— offline LIVE_REPLAY EXIT vs live fill_audit CLOSE。

匹配键: session_bar(±tol) + leg + reason_family(去 QQQ_BTC_ 前缀)。
另附 fill 审计: model_frac 是否声明 0.775; mock 成交常为 mid(0.5) 不作为硬失败。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qqq_btc.common.signal_collect import collect_replay_signals
from qqq_btc.common.time_features import session_minute
from qqq_btc.live.fill_audit_writer import default_audit_path
from qqq_btc.qqq import config as qcfg


def normalize_exit_reason(reason: object) -> str:
    """STEP_PROTECT / QQQ_BTC_STEP_PROTECT|NO_QUOTE → STEP_PROTECT。"""
    if reason is None or (isinstance(reason, float) and not np.isfinite(reason)):
        return ""
    r = str(reason).strip()
    if not r or r.lower() == "nan":
        return ""
    r = r.split("|", 1)[0].strip()
    if r.startswith("QQQ_BTC_"):
        r = r[len("QQQ_BTC_") :]
    return r


def date_scoped_fill_audit_path(date: str) -> Path:
    ymd = str(pd.Timestamp(date).date()).replace("-", "")
    return Path.home() / "quant_project" / "shadow" / f"fill_audit_{ymd}.csv"


def collect_replay_exits(
    df: pd.DataFrame,
    *,
    target_day: str,
    warmup_from_day: Optional[str] = None,
    warmup_through_day: Optional[str] = None,
    max_session_bar: Optional[int] = None,
) -> pd.DataFrame:
    """离线 LIVE_REPLAY(与 live_sim 同占仓) → EXIT / DISASTER_EXIT。"""
    sig = collect_replay_signals(
        df,
        replay_cfg=qcfg.LIVE_REPLAY,
        rails_cfg=qcfg.EXIT_RAILS,
        fill_model=qcfg.FILL_MODEL,
        warmup_from_day=warmup_from_day,
        warmup_through_day=warmup_through_day or target_day,
        target_day=target_day,
        signal_kinds=("EXIT", "DISASTER_EXIT"),
        source="live_sim_exit",
        signal_only=False,
    )
    if max_session_bar is not None and not sig.empty and "session_bar" in sig.columns:
        sig = sig[pd.to_numeric(sig["session_bar"], errors="coerce") <= int(max_session_bar)].copy()
    if not sig.empty:
        sig["reason_family"] = sig.get("reason", pd.Series(dtype=str)).map(normalize_exit_reason)
    return sig.reset_index(drop=True)


def _leg_from_entry_reason(reason: object) -> str:
    """QQQ_BTC_ENTRY|CALL|E:0.15 → CALL。"""
    r = str(reason or "")
    parts = r.split("|")
    for p in parts:
        p = p.strip().upper()
        if p in ("CALL", "PUT", "STRADDLE"):
            return p
    return ""


def load_fill_audit_exits(
    path: Path | str | None,
    date: str,
    *,
    dedupe: bool = True,
) -> pd.DataFrame:
    """
    读 fill_audit CLOSE 行,筛到 target NY 日,补 session_bar / leg / reason_family。

    dedupe=True: 同 (session_bar, reason_family, leg) 只保留最后一条
    (兼容历史污染日志;按日隔离的新鲜文件通常无需去重)。
    """
    path = Path(path).expanduser() if path else default_audit_path()
    empty_cols = [
        "ts",
        "session_bar",
        "leg",
        "reason",
        "reason_family",
        "net_return",
        "fill_spread_frac",
        "model_frac",
        "kind",
        "source",
    ]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=empty_cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=empty_cols)
    if df.empty or "action" not in df.columns:
        return pd.DataFrame(columns=empty_cols)
    closes = df[df["action"].astype(str).str.upper() == "CLOSE"].copy()
    if closes.empty:
        return pd.DataFrame()

    closes["ts_dt"] = pd.to_datetime(closes["ts"], unit="s", utc=True, errors="coerce")
    closes = closes.dropna(subset=["ts_dt"])
    target = pd.Timestamp(date).date()
    closes["date"] = closes["ts_dt"].dt.tz_convert("America/New_York").dt.date
    closes = closes[closes["date"] == target].copy()
    if closes.empty:
        return pd.DataFrame()

    if "session_bar" in closes.columns and closes["session_bar"].notna().any():
        closes["session_bar"] = pd.to_numeric(closes["session_bar"], errors="coerce")
    else:
        closes["session_bar"] = session_minute(closes["ts_dt"]).astype(float)

    if "leg" in closes.columns and closes["leg"].astype(str).str.len().gt(0).any():
        closes["leg"] = closes["leg"].astype(str).str.upper()
    else:
        # 无 leg 列时:用同 qty 最近 OPEN 的 reason 解析;否则空
        opens = df[df["action"].astype(str).str.upper() == "OPEN"].copy()
        closes["leg"] = ""
        if not opens.empty:
            opens["ts_num"] = pd.to_numeric(opens["ts"], errors="coerce")
            for i, row in closes.iterrows():
                leg = ""
                if "reason" in opens.columns:
                    cand = opens[opens["ts_num"] <= float(row["ts"])]
                    if cand.empty:
                        cand = opens
                    # 优先同 qty
                    same = cand[pd.to_numeric(cand.get("qty"), errors="coerce") == float(row.get("qty") or 0)]
                    use = same if not same.empty else cand
                    leg = _leg_from_entry_reason(use.iloc[-1].get("reason", ""))
                closes.at[i, "leg"] = leg

    closes["reason"] = closes.get("exit_reason", pd.Series(dtype=str)).fillna("")
    closes["reason_family"] = closes["reason"].map(normalize_exit_reason)
    closes["net_return"] = pd.to_numeric(closes.get("net_return"), errors="coerce")
    closes["fill_spread_frac"] = pd.to_numeric(closes.get("fill_spread_frac"), errors="coerce")
    closes["model_frac"] = pd.to_numeric(closes.get("model_frac"), errors="coerce")
    closes["kind"] = closes["reason_family"].map(
        lambda r: "DISASTER_EXIT" if r == "DISASTER_STOP" else "EXIT"
    )
    closes["source"] = "fill_audit"
    closes["ts"] = closes["ts_dt"].map(lambda t: t.isoformat())

    out = closes[
        [
            "ts",
            "session_bar",
            "leg",
            "reason",
            "reason_family",
            "net_return",
            "fill_spread_frac",
            "model_frac",
            "kind",
            "source",
            "date",
        ]
    ].copy()
    out["date"] = out["date"].astype(str)

    if dedupe and not out.empty:
        out = out.sort_values(["session_bar", "ts"], kind="mergesort")
        out = out.drop_duplicates(subset=["session_bar", "reason_family", "leg"], keep="last")

    return out.reset_index(drop=True)


def diff_exit_lifecycle(
    offline_exits: pd.DataFrame,
    live_exits: pd.DataFrame,
    *,
    time_tolerance_bars: int = 1,
    pnl_tol: Optional[float] = None,
) -> dict:
    """贪心匹配:同 leg+reason_family,|Δsession_bar|≤tol;(可选)|Δnet_return|≤pnl_tol。"""
    off = offline_exits.copy() if offline_exits is not None else pd.DataFrame()
    live = live_exits.copy() if live_exits is not None else pd.DataFrame()
    for frame in (off, live):
        if frame.empty:
            continue
        if "reason_family" not in frame.columns:
            frame["reason_family"] = frame.get("reason", pd.Series(dtype=str)).map(normalize_exit_reason)
        frame["session_bar"] = pd.to_numeric(frame.get("session_bar"), errors="coerce")
        frame["leg"] = frame.get("leg", pd.Series(dtype=str)).fillna("").astype(str).str.upper()

    used_live: set[int] = set()
    pairs: list[dict] = []
    replay_only: list[dict] = []

    off_sorted = off.sort_values("session_bar", kind="mergesort") if not off.empty else off
    for _, o in off_sorted.iterrows():
        o_sb = float(o["session_bar"]) if pd.notna(o.get("session_bar")) else np.nan
        o_leg = str(o.get("leg", "") or "")
        o_rf = str(o.get("reason_family", "") or "")
        best_j = None
        best_d = None
        for j, lv in live.iterrows():
            if j in used_live:
                continue
            if str(lv.get("leg", "") or "") != o_leg:
                continue
            if str(lv.get("reason_family", "") or "") != o_rf:
                continue
            l_sb = float(lv["session_bar"]) if pd.notna(lv.get("session_bar")) else np.nan
            if not (np.isfinite(o_sb) and np.isfinite(l_sb)):
                continue
            d = abs(o_sb - l_sb)
            if d > float(time_tolerance_bars):
                continue
            if pnl_tol is not None:
                on = o.get("net_return")
                ln = lv.get("net_return")
                if pd.notna(on) and pd.notna(ln) and abs(float(on) - float(ln)) > float(pnl_tol):
                    continue
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            replay_only.append(o.to_dict())
            continue
        used_live.add(best_j)
        lv = live.loc[best_j]
        pairs.append(
            {
                "session_bar_offline": int(o_sb) if np.isfinite(o_sb) else None,
                "session_bar_stream": int(float(lv["session_bar"])) if pd.notna(lv.get("session_bar")) else None,
                "bar_delta": float(best_d) if best_d is not None else None,
                "leg": o_leg,
                "reason_family": o_rf,
                "net_return_offline": float(o["net_return"]) if pd.notna(o.get("net_return")) else None,
                "net_return_stream": float(lv["net_return"]) if pd.notna(lv.get("net_return")) else None,
            }
        )

    live_only = [live.loc[j].to_dict() for j in live.index if j not in used_live]
    n_r, n_l = len(off), len(live)
    n_m = len(pairs)
    both_empty = n_r == 0 and n_l == 0
    summary = {
        "n_replay": n_r,
        "n_live": n_l,
        "n_matched": n_m,
        "match_rate_replay": n_m / n_r if n_r else 1.0,
        "match_rate_live": n_m / n_l if n_l else 1.0,
        "mode": "exit_lifecycle",
        "time_tolerance_bars": int(time_tolerance_bars),
        "pass": both_empty or (n_m == n_r == n_l and n_m > 0) or (both_empty),
    }
    # 有单边信号则必须全匹配
    if n_r or n_l:
        summary["pass"] = n_m == n_r == n_l
    return {
        "matched": pairs,
        "replay_only": replay_only,
        "live_only": live_only,
        "summary": summary,
    }


def first_exit_diff(
    offline_exits: pd.DataFrame,
    live_exits: pd.DataFrame,
    *,
    time_tolerance_bars: int = 1,
) -> dict:
    """首笔平仓对拍(对称 first_entry_diff)。"""
    empty_summary = {
        "n_replay": 0,
        "n_live": 0,
        "n_matched": 0,
        "match_rate_replay": 1.0,
        "match_rate_live": 1.0,
        "mode": "first_exit",
        "session_bar_offline": None,
        "session_bar_stream": None,
        "leg_offline": None,
        "leg_stream": None,
        "reason_offline": None,
        "reason_stream": None,
        "bar_delta": None,
        "pass": True,
    }
    off = offline_exits.copy() if offline_exits is not None else pd.DataFrame()
    live = live_exits.copy() if live_exits is not None else pd.DataFrame()

    def _first(df: pd.DataFrame) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        work = df.copy()
        work["session_bar"] = pd.to_numeric(work.get("session_bar"), errors="coerce")
        work = work.sort_values("session_bar", kind="mergesort")
        return work.iloc[0]

    o = _first(off)
    s = _first(live)
    summary = dict(empty_summary)
    summary["n_replay"] = 0 if o is None else 1
    summary["n_live"] = 0 if s is None else 1
    if o is None and s is None:
        return {"matched": [], "replay_only": [], "live_only": [], "summary": summary}
    if o is not None:
        summary["session_bar_offline"] = int(o["session_bar"]) if pd.notna(o.get("session_bar")) else None
        summary["leg_offline"] = str(o.get("leg", "") or "")
        summary["reason_offline"] = normalize_exit_reason(o.get("reason_family") or o.get("reason"))
    if s is not None:
        summary["session_bar_stream"] = int(s["session_bar"]) if pd.notna(s.get("session_bar")) else None
        summary["leg_stream"] = str(s.get("leg", "") or "")
        summary["reason_stream"] = normalize_exit_reason(s.get("reason_family") or s.get("reason"))
    if o is None:
        summary["pass"] = False
        return {"matched": [], "replay_only": [], "live_only": [s.to_dict()], "summary": summary}
    if s is None:
        summary["pass"] = False
        return {"matched": [], "replay_only": [o.to_dict()], "live_only": [], "summary": summary}

    o_sb = float(o["session_bar"]) if pd.notna(o.get("session_bar")) else np.nan
    s_sb = float(s["session_bar"]) if pd.notna(s.get("session_bar")) else np.nan
    bar_delta = abs(o_sb - s_sb) if np.isfinite(o_sb) and np.isfinite(s_sb) else np.inf
    summary["bar_delta"] = float(bar_delta) if np.isfinite(bar_delta) else None
    leg_ok = summary["leg_offline"] == summary["leg_stream"]
    # leg 缺失时只比 reason+bar
    if not summary["leg_offline"] or not summary["leg_stream"]:
        leg_ok = True
    reason_ok = summary["reason_offline"] == summary["reason_stream"]
    ok = leg_ok and reason_ok and bar_delta <= float(time_tolerance_bars)
    summary["n_matched"] = 1 if ok else 0
    summary["match_rate_replay"] = 1.0 if ok else 0.0
    summary["match_rate_live"] = 1.0 if ok else 0.0
    summary["pass"] = bool(ok)
    if ok:
        return {
            "matched": [summary],
            "replay_only": [],
            "live_only": [],
            "summary": summary,
        }
    return {
        "matched": [],
        "replay_only": [o.to_dict()],
        "live_only": [s.to_dict()],
        "summary": summary,
    }


def audit_fill_model_declared(
    path: Path | str | None,
    date: str,
    *,
    target_frac: float = 0.775,
    tol: float = 0.02,
) -> dict:
    """
    检查当日 fill_audit 的 model_frac 是否声明为 fill_model(0.775)。
    mock mid 成交的 fill_spread_frac≈0.5 单独报告,不作为本检查失败条件。
    """
    path = Path(path).expanduser() if path else default_audit_path()
    if not path.exists() or path.stat().st_size == 0:
        return {"n": 0, "pass": None, "error": f"missing {path}"}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"n": 0, "pass": None, "error": "empty"}
    if df.empty:
        return {"n": 0, "pass": None, "error": "empty"}
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["ts_dt"])
    day = pd.Timestamp(date).date()
    # OPEN 常被写成 wall-clock;用 CLOSE 的 sim ts 定日,OPEN 一并纳入同文件尾部窗口
    closes = df[df["action"].astype(str).str.upper() == "CLOSE"]
    close_days = closes["ts_dt"].dt.tz_convert("America/New_York").dt.date
    day_closes = closes[close_days == day]
    if day_closes.empty:
        # fallback: 整文件 model_frac
        work = df
    else:
        work = day_closes
    model = pd.to_numeric(work.get("model_frac"), errors="coerce").dropna()
    realized = pd.to_numeric(work.get("fill_spread_frac"), errors="coerce").dropna()
    out = {
        "n": int(len(work)),
        "target": target_frac,
        "model_frac_median": float(model.median()) if len(model) else None,
        "fill_spread_frac_median": float(realized.median()) if len(realized) else None,
        "pass": bool(len(model) and abs(float(model.median()) - target_frac) <= tol),
        "note": "mock IBKR 常成交在 mid(fill≈0.5);pass 看 model_frac 声明是否为 0.775",
    }
    return out
