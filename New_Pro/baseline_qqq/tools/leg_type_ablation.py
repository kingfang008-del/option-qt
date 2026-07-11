#!/usr/bin/env python3
"""
腿型对照:同一入场信号下,CALL / PUT / STRADDLE 谁兑现 edge。

假设(step7):可预测成分是『gamma 便宜/波动定价』,不是方向。
若成立,则:
  - 强制 CALL 与强制 PUT 应接近(或一正一负但合计接近跨式)
  - 强制 STRADDLE 应更稳(方向风险对冲,靠凸性+rails 不对称兑现)

协议:
  - 信号:已训 TFT(top-k + veto)的入场时刻,完全冻结
  - 腿型:强制 CALL / 强制 PUT / 强制 STRADDLE(互斥,不做腿竞争)
  - 退出:生产 EXIT_RAILS 不变
  - 评估:2025-08 ~ 2026-02 七个月 walk-forward
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.qqq import config as qcfg

import rails_value_lgbm_v2 as v2
from rails_value_lgbm import causal_topk_signal, entry_mask
from train_rails_value_tft import (
    RAW_DIR,
    build_sequences,
    load_cached_days,
    make_model,
)


def _round4(x: float) -> float:
    return round(float(x), 4)


def attach_put_quotes(minute: pd.DataFrame, put_ticks: pd.DataFrame) -> pd.DataFrame:
    """把 PUT ATM(bucket0)分钟盘口挂到 CALL 分钟帧上。"""
    m = minute.copy()
    if put_ticks is None or put_ticks.empty:
        for c in ("exec_put_bid", "exec_put_ask", "exec_put_spread_pct"):
            m[c] = np.nan
        return m
    pm = put_ticks.groupby("minute_ts", as_index=False).agg(
        exec_put_bid=("exec_call_bid", "last"),
        exec_put_ask=("exec_call_ask", "last"),
        exec_put_spread_pct=("exec_call_spread_pct", "last"),
    )
    pm = pm.rename(columns={"minute_ts": "timestamp"})
    m = m.merge(pm, on="timestamp", how="left")
    for c in ("exec_put_bid", "exec_put_ask", "exec_put_spread_pct"):
        m[c] = m[c].ffill()
    return m


def build_tick_dual(call_ticks: pd.DataFrame, put_ticks: pd.DataFrame) -> pd.DataFrame:
    """秒级双腿盘口:CALL 时间轴 left-join PUT(按秒对齐,ffill)。"""
    c = call_ticks[
        ["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]
    ].copy()
    if put_ticks is None or put_ticks.empty:
        c["exec_put_bid"] = np.nan
        c["exec_put_ask"] = np.nan
        c["exec_put_spread_pct"] = np.nan
        return c
    p = put_ticks[
        ["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]
    ].rename(
        columns={
            "exec_call_bid": "exec_put_bid",
            "exec_call_ask": "exec_put_ask",
            "exec_call_spread_pct": "exec_put_spread_pct",
        }
    )
    # 按秒对齐
    c["ts_sec"] = c["timestamp"].dt.floor("s")
    p["ts_sec"] = p["timestamp"].dt.floor("s")
    p = p.drop_duplicates("ts_sec", keep="last")
    out = c.merge(p.drop(columns=["timestamp"]), on="ts_sec", how="left")
    for col in ("exec_put_bid", "exec_put_ask", "exec_put_spread_pct"):
        out[col] = out[col].ffill()
    return out.drop(columns=["ts_sec"])


def predict_signals(days, model, norm, features, device, top_pct: float, veto_thr: float):
    """返回 {date: signal_array},信号已过 top-k + veto。"""
    X, _yt, _yv, _rails, day_idx, bar_idx, _ = build_sequences(days, features, norm)
    preds, vetos = [], []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i : i + 4096]).to(device)
            lt, lv = model(xb)
            preds.append(torch.sigmoid(lt).cpu().numpy())
            vetos.append(torch.sigmoid(lv).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([])
    veto = np.concatenate(vetos) if vetos else np.array([])

    out: Dict[str, np.ndarray] = {}
    for di, (date_str, m) in enumerate(days):
        p = np.full(len(m), -np.inf)
        v = np.ones(len(m))
        sel = day_idx == di
        p[bar_idx[sel]] = pred[sel]
        v[bar_idx[sel]] = veto[sel]
        w = entry_mask(m).to_numpy()
        gate = v < veto_thr
        out[date_str] = causal_topk_signal(p, w, top_pct, gate=gate)
    return out


def summarize(day_rois: List[float], trades_rets: List[float], legs: Dict[str, int]) -> dict:
    dr = np.array(day_rois, dtype=np.float64) if day_rois else np.array([0.0])
    hits = [1.0 if r > 0 else 0.0 for r in trades_rets]
    return {
        "days": len(day_rois),
        "active_days": int((dr != 0).sum()),
        "win_days": int((dr > 0).sum()),
        "trades": len(trades_rets),
        "hit_rate": _round4(float(np.mean(hits))) if hits else 0.0,
        "day_roi_mean": _round4(float(dr.mean())),
        "compound": _round4(float(np.prod(1.0 + dr) - 1.0)),
        "worst_day": _round4(float(dr.min())),
        "avg_trade": _round4(float(np.mean(trades_rets))) if trades_rets else 0.0,
        "legs": legs,
    }


def _base_replay_cfg(*, long_only: bool):
    return replace(
        qcfg.REPLAY,
        long_only=long_only,
        put_gate_min=None,
        morning_fade_min_ret=None,
        morning_fade_max_peak_dd=None,
        put_trend_max_ret=None,
        put_late_session_bar=None,
        put_spot_day_ret_min=None,
        block_call_on_rapid_drop=False,
        call_trend_r2_min=None,
        call_chase_vix_rev_min=None,
        call_spike_range30_min=None,
        call_timing_max_bar=None,
        straddle_entry_threshold=qcfg.REPLAY.entry_threshold,
        max_straddles_per_day=qcfg.REPLAY.max_trades_per_day,
    )


def replay_forced_leg(
    days: List[Tuple[str, pd.DataFrame]],
    signals: Dict[str, np.ndarray],
    tick_cache: Dict[str, pd.DataFrame],
    mode: str,
) -> dict:
    """
    mode: call | put | straddle | route_mom15 | route_cpratio
    强制腿型或因果路由:同一信号列。
    """
    day_rois: List[float] = []
    trade_rets: List[float] = []
    legs: Dict[str, int] = {}

    for date_str, minute in days:
        m = minute.copy()
        sig = signals[date_str]
        m["sig_call"] = -1.0
        m["sig_put"] = -1.0
        m["sig_straddle"] = -1.0
        m["sig_edge"] = -1.0

        if mode == "call":
            replay_cfg = _base_replay_cfg(long_only=True)
            m["sig_edge"] = sig
            r = run_event_replay(
                m, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS,
                tick_df=tick_cache.get(date_str), edge_col="sig_edge",
                event_cfg=EventReplayConfig(tick_disaster_stop=True),
            )
        elif mode == "put":
            replay_cfg = _base_replay_cfg(long_only=False)
            m["sig_put"] = sig
            r = run_event_replay(
                m, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS,
                tick_df=tick_cache.get(date_str), edge_col="sig_edge",
                call_edge_col="sig_call", put_edge_col="sig_put",
                event_cfg=EventReplayConfig(tick_disaster_stop=True),
            )
        elif mode == "straddle":
            replay_cfg = _base_replay_cfg(long_only=False)
            m["sig_straddle"] = sig
            r = run_event_replay(
                m, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS,
                tick_df=tick_cache.get(date_str), edge_col="sig_edge",
                call_edge_col="sig_call", put_edge_col="sig_put",
                straddle_edge_col="sig_straddle",
                event_cfg=EventReplayConfig(tick_disaster_stop=True),
            )
        elif mode in ("route_mom15", "route_cpratio"):
            # 因果路由:信号触发时,用入场 bar 的动量/偏度选腿
            replay_cfg = _base_replay_cfg(long_only=False)
            call_sig = np.full(len(m), -1.0)
            put_sig = np.full(len(m), -1.0)
            if mode == "route_mom15":
                # 优先用现货15m动量;否则权利金动量
                if "s_ret_15m" in m.columns:
                    mom = m["s_ret_15m"].to_numpy()
                elif "ret_15m" in m.columns:
                    mom = m["ret_15m"].to_numpy()
                else:
                    mom = np.zeros(len(m))
                for i in range(len(m)):
                    if sig[i] <= 0:
                        continue
                    if np.isfinite(mom[i]) and mom[i] > 0:
                        call_sig[i] = sig[i]
                    else:
                        put_sig[i] = sig[i]
            else:
                # cp_ratio = log(C/P); >0 偏 CALL 贵/看涨 → 跟 CALL; <0 跟 PUT
                cpr = m["cp_ratio"].to_numpy() if "cp_ratio" in m.columns else np.zeros(len(m))
                for i in range(len(m)):
                    if sig[i] <= 0:
                        continue
                    if np.isfinite(cpr[i]) and cpr[i] >= 0:
                        call_sig[i] = sig[i]
                    else:
                        put_sig[i] = sig[i]
            m["sig_call"] = call_sig
            m["sig_put"] = put_sig
            r = run_event_replay(
                m, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS,
                tick_df=tick_cache.get(date_str), edge_col="sig_edge",
                call_edge_col="sig_call", put_edge_col="sig_put",
                event_cfg=EventReplayConfig(tick_disaster_stop=True),
            )
        else:
            raise ValueError(mode)

        if not r.trades:
            day_rois.append(0.0)
            continue
        rets = [t.net_return for t in r.trades]
        day_rois.append(float(np.prod(1.0 + np.array(rets)) - 1.0))
        trade_rets.extend(rets)
        for t in r.trades:
            legs[t.leg] = legs.get(t.leg, 0) + 1

    return summarize(day_rois, trade_rets, legs)


def main() -> int:
    ap = argparse.ArgumentParser(description="leg-type ablation on frozen TFT signals")
    ap.add_argument("--ckpt", default="/mnt/s990/data/cache/rails_value_tft_2025H1_lr2e4.pt")
    ap.add_argument(
        "--test-globs",
        default=";".join(
            [f"QQQ_2025-{m:02d}-*.parquet" for m in range(8, 13)]
            + ["QQQ_2026-01-*.parquet", "QQQ_2026-02-*.parquet"]
        ),
    )
    ap.add_argument("--top-pct", type=float, default=0.02)
    ap.add_argument("--veto-thr", type=float, default=0.5)
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/qqq_1dte_leg_type_ablation.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    features = ck["features"]
    norm = (ck["norm_mu"], ck["norm_sd"])
    model = make_model(len(features), ck["hidden"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded ckpt val_ic={ck.get('val_ic'):.4f}")

    segs = args.test_globs.split(";")
    all_results = []

    for seg in segs:
        days_raw = load_cached_days(seg.split(","))
        if not days_raw:
            print(f"[{seg}] empty, skip")
            continue

        # 挂 PUT 盘口 + 构建 tick
        days: List[Tuple[str, pd.DataFrame]] = []
        tick_cache: Dict[str, pd.DataFrame] = {}
        for date_str, minute in days_raw:
            fp = RAW_DIR / "QQQ" / f"QQQ_{date_str}.parquet"
            buckets = v2.load_day_buckets(fp)
            call_ticks = buckets.get(2)
            put_ticks = buckets.get(0)
            if call_ticks is None or call_ticks.empty:
                continue
            m = attach_put_quotes(minute, put_ticks)
            days.append((date_str, m))
            tick_cache[date_str] = build_tick_dual(call_ticks, put_ticks)

        signals = predict_signals(
            days, model, norm, features, device, args.top_pct, args.veto_thr
        )

        month = {"segment": seg}
        for mode in ("call", "put", "straddle", "route_mom15", "route_cpratio"):
            stats = replay_forced_leg(days, signals, tick_cache, mode)
            month[mode] = stats
            print(
                f"[{seg}] {mode:<14} win={stats['win_days']}/{stats['days']} "
                f"comp={stats['compound']:+.1%} trades={stats['trades']} "
                f"hit={stats['hit_rate']:.0%} avg={stats['avg_trade']:+.1%} "
                f"legs={stats['legs']}"
            )
        all_results.append(month)

    # 7 个月合计
    totals = {}
    for mode in ("call", "put", "straddle", "route_mom15", "route_cpratio"):
        comps = [m[mode]["compound"] for m in all_results]
        trades = sum(m[mode]["trades"] for m in all_results)
        wins = sum(m[mode]["win_days"] for m in all_results)
        days_n = sum(m[mode]["days"] for m in all_results)
        total_comp = float(np.prod([1.0 + c for c in comps]) - 1.0)
        totals[mode] = {
            "compound": _round4(total_comp),
            "pos_months": int(sum(1 for c in comps if c > 0)),
            "n_months": len(comps),
            "trades": trades,
            "win_days": wins,
            "days": days_n,
            "monthly_compounds": [_round4(c) for c in comps],
        }
        print(
            f"TOTAL {mode:<14} months+ {totals[mode]['pos_months']}/{totals[mode]['n_months']} "
            f"comp={totals[mode]['compound']:+.1%} trades={trades}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": {
                    "ckpt": args.ckpt,
                    "top_pct": args.top_pct,
                    "veto_thr": args.veto_thr,
                    "note": "同一 TFT 入场信号,强制腿型互斥对照;关闭方向相关门控",
                },
                "totals": totals,
                "months": all_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
