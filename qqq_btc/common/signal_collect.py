#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入场信号收集 —— strict replay vs live 路径(同一 ReplaySession 状态机)。
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from qqq_btc.common.event_replay import prepare_minute_frame, run_event_replay
from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel, PerpFillModel
from qqq_btc.common.replay_types import ReplayConfig
from qqq_btc.qqq import config as qcfg


def _event_session_bar(ts, minute_df: pd.DataFrame) -> Optional[int]:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        key = int(t.floor("min").timestamp())
        if "_minute_key" in minute_df.columns:
            row = minute_df.loc[minute_df["_minute_key"] == key]
            if not row.empty:
                return int(row.iloc[0]["session_bar"])
        row = minute_df.loc[minute_df["timestamp"] == t]
        if not row.empty:
            return int(row.iloc[0]["session_bar"])
        return None
    except Exception:
        return None


def events_to_signal_frame(
    events: Iterable,
    *,
    minute_df: pd.DataFrame,
    kinds: tuple[str, ...] = ("SIGNAL", "ENTER"),
    source: str = "replay",
) -> pd.DataFrame:
    rows = []
    for ev in events:
        kind = str(getattr(ev, "kind", "") or "")
        if kind not in kinds:
            continue
        ts = pd.Timestamp(getattr(ev, "ts", None))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        sb = _event_session_bar(ts, minute_df)
        extra = getattr(ev, "extra", {}) or {}
        nr = getattr(ev, "net_return", None)
        rows.append(
            {
                "source": source,
                "kind": kind,
                "ts": ts.isoformat(),
                "date": str(ts.tz_convert("America/New_York").date()),
                "session_bar": sb,
                "leg": str(getattr(ev, "leg", "") or ""),
                "edge": float(getattr(ev, "edge", 0.0) or 0.0),
                "threshold": float(extra.get("threshold", np.nan))
                if extra.get("threshold") is not None
                else np.nan,
                "reason": str(getattr(ev, "reason", "") or ""),
                "net_return": float(nr) if nr is not None and np.isfinite(float(nr)) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def collect_replay_signals(
    df: pd.DataFrame,
    *,
    replay_cfg: ReplayConfig,
    rails_cfg: Optional[ExitRailsConfig] = None,
    fill_model: Optional[Union[OptionSpreadFillModel, PerpFillModel]] = None,
    warmup_through_day: Optional[str] = None,
    warmup_from_day: Optional[str] = None,
    target_day: Optional[str] = None,
    edge_col: str = "net_edge",
    edge_q10_col: Optional[str] = None,
    call_edge_col: Optional[str] = None,
    put_edge_col: Optional[str] = None,
    put_gate_col: Optional[str] = None,
    signal_kinds: tuple[str, ...] = ("SIGNAL",),
    source: str = "strict_replay",
    signal_only: bool = False,
) -> pd.DataFrame:
    """跑 event replay 并导出指定 kind 的入场信号(默认 strict replay 的 SIGNAL)。"""
    minute_df = prepare_minute_frame(df)
    if warmup_from_day:
        start = pd.Timestamp(warmup_from_day).date()
        minute_df = minute_df[minute_df["_day"] >= start].copy()
    if warmup_through_day:
        cutoff = pd.Timestamp(warmup_through_day).date()
        minute_df = minute_df[minute_df["_day"] <= cutoff].copy()
    fm = fill_model or qcfg.FILL_MODEL
    rails = rails_cfg or qcfg.EXIT_RAILS
    result = run_event_replay(
        minute_df,
        fm,
        replay_cfg,
        rails,
        tick_df=None,
        edge_col=edge_col,
        edge_q10_col=edge_q10_col or qcfg.EDGE_Q10_COL,
        call_edge_col=call_edge_col or qcfg.CALL_EDGE_COL,
        put_edge_col=put_edge_col or qcfg.PUT_EDGE_COL,
        put_gate_col=put_gate_col or qcfg.PUT_GATE_COL,
        signal_only=signal_only,
    )
    sig = events_to_signal_frame(
        result.events,
        minute_df=minute_df,
        kinds=signal_kinds,
        source=source,
    )
    if target_day and not sig.empty:
        sig = sig[sig["date"] == str(pd.Timestamp(target_day).date())].reset_index(drop=True)
    return sig


def collect_decision_signals(
    df: pd.DataFrame,
    *,
    warmup_through_day: Optional[str] = None,
    warmup_from_day: Optional[str] = None,
    target_day: Optional[str] = None,
    replay_cfg: Optional[ReplayConfig] = None,
) -> pd.DataFrame:
    """纯 choose_entry 决策(不持仓/不延迟),strict 与 live 应对齐。"""
    return collect_replay_signals(
        df,
        replay_cfg=replay_cfg or qcfg.REPLAY,
        warmup_through_day=warmup_through_day,
        warmup_from_day=warmup_from_day,
        target_day=target_day,
        signal_kinds=("SIGNAL",),
        source="decision",
        signal_only=True,
    )


def collect_live_sim_signals(
    df: pd.DataFrame,
    *,
    replay_cfg: Optional[ReplayConfig] = None,
    warmup_through_day: Optional[str] = None,
    warmup_from_day: Optional[str] = None,
    target_day: Optional[str] = None,
) -> pd.DataFrame:
    """Live 路径: LIVE_REPLAY(immediate_entry) → ENTER 事件即同 bar 决策。"""
    return collect_replay_signals(
        df,
        replay_cfg=replay_cfg or qcfg.LIVE_REPLAY,
        warmup_through_day=warmup_through_day,
        warmup_from_day=warmup_from_day,
        target_day=target_day,
        signal_kinds=("ENTER",),
        source="live_sim",
    )


def load_dry_run_signals(path: str | pd.PathLike) -> pd.DataFrame:
    """
    读取 dry-run OMS 信号 audit CSV(signal_audit_writer 或兼容格式)。
    期望列: ts/timestamp, leg; PASS/ENTER 行用于对拍。
    """
    raw = pd.read_csv(path)
    if "decision" in raw.columns:
        raw = raw[raw["decision"].astype(str).str.upper() == "PASS"].copy()
    elif "kind" in raw.columns:
        raw = raw[raw["kind"].astype(str).str.upper().isin(("ENTER", "SIGNAL"))].copy()

    ts_col = "timestamp" if "timestamp" in raw.columns else ("ts" if "ts" in raw.columns else None)
    if ts_col is None or raw.empty:
        if ts_col is None:
            raise ValueError("dry-run CSV 需含 ts 或 timestamp 列")
        return pd.DataFrame(
            columns=["ts", "leg", "edge", "session_bar", "threshold", "source", "kind", "date"]
        )

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    out["leg"] = raw["leg"].astype(str) if "leg" in raw.columns else ""
    out["edge"] = pd.to_numeric(raw.get("edge", raw.get("net_edge_raw", 0.0)), errors="coerce")
    out["session_bar"] = pd.to_numeric(raw["session_bar"], errors="coerce") if "session_bar" in raw.columns else np.nan
    out["threshold"] = pd.to_numeric(raw.get("threshold", np.nan), errors="coerce")
    out["source"] = "dry_run"
    out["kind"] = raw.get("kind", "ENTER")
    out["date"] = out["ts"].dt.tz_convert("America/New_York").dt.date.astype(str)
    return out.dropna(subset=["ts"]).reset_index(drop=True)


def diff_signal_frames(
    replay_sig: pd.DataFrame,
    live_sig: pd.DataFrame,
    *,
    time_tolerance_bars: int = 1,
) -> dict:
    """
    按 session_bar + leg 对齐;live immediate vs replay SIGNAL 允许 ±1 bar(延迟成交)。
    """
    if replay_sig.empty and live_sig.empty:
        return {"matched": [], "replay_only": [], "live_only": [], "summary": {"n_matched": 0}}

    r = replay_sig.copy()
    l = live_sig.copy()
    for df in (r, l):
        if "session_bar" not in df.columns:
            df["session_bar"] = np.nan
        df["session_bar"] = pd.to_numeric(df["session_bar"], errors="coerce")

    matched_r = set()
    matched_l = set()
    pairs = []

    for i, row in r.iterrows():
        sb = row.get("session_bar")
        leg = str(row.get("leg", "") or "")
        if not np.isfinite(sb):
            continue
        for j, lrow in l.iterrows():
            if j in matched_l:
                continue
            lsb = lrow.get("session_bar")
            lleg = str(lrow.get("leg", "") or "")
            if leg != lleg:
                continue
            if abs(float(sb) - float(lsb)) <= time_tolerance_bars:
                matched_r.add(i)
                matched_l.add(j)
                pairs.append(
                    {
                        "session_bar_replay": int(sb),
                        "session_bar_live": int(lsb),
                        "leg": leg,
                        "edge_replay": float(row.get("edge", 0.0) or 0.0),
                        "edge_live": float(lrow.get("edge", 0.0) or 0.0),
                        "ts_replay": row.get("ts"),
                        "ts_live": lrow.get("ts"),
                    }
                )
                break

    replay_only = r.loc[[i for i in r.index if i not in matched_r]].to_dict("records")
    live_only = l.loc[[j for j in l.index if j not in matched_l]].to_dict("records")
    n_r, n_l = len(r), len(l)
    summary = {
        "n_replay": n_r,
        "n_live": n_l,
        "n_matched": len(pairs),
        "match_rate_replay": len(pairs) / n_r if n_r else 1.0,
        "match_rate_live": len(pairs) / n_l if n_l else 1.0,
    }
    return {"matched": pairs, "replay_only": replay_only, "live_only": live_only, "summary": summary}


def first_entry_diff(
    offline_enters: pd.DataFrame,
    stream_passes: pd.DataFrame,
    *,
    time_tolerance_bars: int = 0,
) -> dict:
    """
    占仓感知 OMS 对拍:只比当日首笔 ENTER / PASS。

    offline_enters: live_sim ENTER(已含持仓状态机)
    stream_passes: dry-run OMS audit PASS
    """
    empty = {
        "matched": [],
        "replay_only": [],
        "live_only": [],
        "summary": {
            "n_replay": 0,
            "n_live": 0,
            "n_matched": 0,
            "match_rate_replay": 1.0,
            "match_rate_live": 1.0,
            "mode": "first_entry",
            "session_bar_offline": None,
            "session_bar_stream": None,
            "leg_offline": None,
            "leg_stream": None,
            "bar_delta": None,
        },
    }
    off = offline_enters.copy() if offline_enters is not None else pd.DataFrame()
    st = stream_passes.copy() if stream_passes is not None else pd.DataFrame()
    if off.empty and st.empty:
        return empty

    def _first(df: pd.DataFrame) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        work = df.copy()
        if "session_bar" in work.columns:
            work["session_bar"] = pd.to_numeric(work["session_bar"], errors="coerce")
            work = work.sort_values("session_bar", kind="mergesort")
        elif "ts" in work.columns:
            work = work.sort_values("ts", kind="mergesort")
        return work.iloc[0]

    o = _first(off)
    s = _first(st)
    summary = {
        "n_replay": 0 if o is None else 1,
        "n_live": 0 if s is None else 1,
        "n_matched": 0,
        "match_rate_replay": 1.0,
        "match_rate_live": 1.0,
        "mode": "first_entry",
        "session_bar_offline": None if o is None else int(o.get("session_bar")) if pd.notna(o.get("session_bar")) else None,
        "session_bar_stream": None if s is None else int(s.get("session_bar")) if pd.notna(s.get("session_bar")) else None,
        "leg_offline": None if o is None else str(o.get("leg", "") or ""),
        "leg_stream": None if s is None else str(s.get("leg", "") or ""),
        "bar_delta": None,
    }
    if o is None and s is None:
        return empty
    if o is None:
        return {"matched": [], "replay_only": [], "live_only": [s.to_dict()], "summary": summary}
    if s is None:
        return {"matched": [], "replay_only": [o.to_dict()], "live_only": [], "summary": summary}

    o_sb = float(o.get("session_bar")) if pd.notna(o.get("session_bar")) else np.nan
    s_sb = float(s.get("session_bar")) if pd.notna(s.get("session_bar")) else np.nan
    o_leg = str(o.get("leg", "") or "")
    s_leg = str(s.get("leg", "") or "")
    bar_delta = abs(o_sb - s_sb) if np.isfinite(o_sb) and np.isfinite(s_sb) else np.inf
    summary["bar_delta"] = None if not np.isfinite(bar_delta) else float(bar_delta)
    ok = o_leg == s_leg and bar_delta <= float(time_tolerance_bars)
    if ok:
        summary["n_matched"] = 1
        summary["match_rate_replay"] = 1.0
        summary["match_rate_live"] = 1.0
        pair = {
            "session_bar_replay": int(o_sb),
            "session_bar_live": int(s_sb),
            "leg": o_leg,
            "edge_replay": float(o.get("edge", 0.0) or 0.0),
            "edge_live": float(s.get("edge", 0.0) or 0.0),
            "ts_replay": o.get("ts"),
            "ts_live": s.get("ts"),
        }
        return {"matched": [pair], "replay_only": [], "live_only": [], "summary": summary}

    summary["match_rate_replay"] = 0.0
    summary["match_rate_live"] = 0.0
    return {
        "matched": [],
        "replay_only": [o.to_dict()],
        "live_only": [s.to_dict()],
        "summary": summary,
    }
