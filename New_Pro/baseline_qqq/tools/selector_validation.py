#!/usr/bin/env python3
"""
11:00 因果 selector 验证: 趋势纯净(R²) + 点差最小 → 下午只做选中标的。

对比:
  A_always_qqq   永远下午做 QQQ (oracle 入场 + rails, 含泄漏仅作规则栈 proxy)
  B_selector     11:00 在候选池里选 score 最高且过 gate 的标的, 下午只做它
  C_oracle_pick  事后选下午 PnL 最好的标的 (上界, 含泄漏)

注意: 入场仍用 oracle_edge (未来 5bar 净 edge), 本脚本验证的是
「标的/日选择层」是否有 lift, 不是 TFT 实盘收益。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.replay_types import ReplayConfig
from qqq_btc.common.time_features import SESSION_TZ, session_minute
from qqq_btc.common.trend_features import rolling_linear_fit
from qqq_btc.qqq import config as qcfg

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from raw1s_rule_validation import (  # noqa: E402
    _replay_summary,
    build_minute_frame,
    compute_oracle_edge,
    load_raw1s_bucket_day,
)


@dataclass(frozen=True)
class SelectorConfig:
    decision_bar: int = 90  # 11:00 ET
    min_r2: float = 0.35
    max_spread_pct: float = 0.06
    max_spread_by_symbol: Tuple[Tuple[str, float], ...] = (("AAPL", 0.08),)
    min_morning_ticks: int = 200
    score_r2_weight: float = 1.0
    score_spread_weight: float = 3.0
    afternoon_start_bar: int = 90
    afternoon_end_bar: int = 330


def _max_spread_for(symbol: str, cfg: SelectorConfig) -> float:
    for sym, lim in cfg.max_spread_by_symbol:
        if sym == symbol:
            return lim
    return cfg.max_spread_pct


def _round4(x: float) -> float:
    return round(float(x), 4)


def load_stock_minute_day(raw_dir: Path, symbol: str, date_str: str) -> pd.DataFrame:
    """正股 1s → RTH 1min close (用于 trend R²)。"""
    stk_dir = raw_dir / "stocks" / symbol
    fp = stk_dir / f"{symbol}_{date_str}.parquet"
    if not fp.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(SESSION_TZ, ambiguous="infer")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(SESSION_TZ)
    t = df["timestamp"]
    rth = ((t.dt.hour > 9) | ((t.dt.hour == 9) & (t.dt.minute >= 30))) & (t.dt.hour < 16)
    df = df[rth].copy()
    if df.empty:
        return pd.DataFrame()
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    minute = df.groupby("minute_ts", as_index=False).agg(close=("close", "last"))
    minute = minute.rename(columns={"minute_ts": "timestamp"})
    minute["session_bar"] = session_minute(minute["timestamp"]).astype(int)
    return minute.sort_values("timestamp").reset_index(drop=True)


def _morning_option_spread(ticks: pd.DataFrame, decision_bar: int) -> Tuple[float, int]:
    if ticks.empty:
        return float("nan"), 0
    sb = session_minute(ticks["timestamp"]).astype(int)
    m = ticks[sb <= decision_bar]
    if m.empty:
        return float("nan"), 0
    sp = m["exec_call_spread_pct"].dropna()
    if sp.empty:
        return float("nan"), int(len(m))
    return float(sp.median()), int(len(m))


def _morning_trend_r2(
    stock_minute: pd.DataFrame,
    option_minute: pd.DataFrame,
    decision_bar: int,
) -> Tuple[float, str]:
    """截至 decision_bar 的 OLS R² (优先正股; 窗口=min(上午 bar 数, decision_bar))。"""
    min_bars = 30

    def _r2_from_series(log_px: np.ndarray) -> float:
        if len(log_px) < min_bars:
            return float("nan")
        win = min(len(log_px), decision_bar + 1)
        _, r2_arr = rolling_linear_fit(log_px, win)
        val = r2_arr[-1]
        return float(val) if np.isfinite(val) else float("nan")

    if not stock_minute.empty:
        sub = stock_minute[stock_minute["session_bar"] <= decision_bar].copy()
        if len(sub) >= min_bars:
            px = pd.to_numeric(sub["close"], errors="coerce").astype(np.float64)
            r2 = _r2_from_series(np.log(px.replace(0, np.nan)).to_numpy())
            if np.isfinite(r2):
                return r2, "stock"
    if option_minute.empty:
        return float("nan"), "none"
    sub = option_minute[option_minute["session_bar"] <= decision_bar].copy()
    if len(sub) < min_bars:
        return float("nan"), "none"
    px = pd.to_numeric(sub["exec_call_mid"], errors="coerce").astype(np.float64)
    r2 = _r2_from_series(np.log(px.replace(0, np.nan)).to_numpy())
    return r2, "option_mid_proxy"


def score_candidate(r2: float, spread: float, cfg: SelectorConfig) -> float:
    if not (np.isfinite(r2) and np.isfinite(spread)):
        return float("-inf")
    return cfg.score_r2_weight * r2 - cfg.score_spread_weight * spread


def passes_gate(
    symbol: str,
    r2: float,
    spread: float,
    n_ticks: int,
    cfg: SelectorConfig,
) -> bool:
    if n_ticks < cfg.min_morning_ticks:
        return False
    if not np.isfinite(r2) or r2 < cfg.min_r2:
        return False
    lim = _max_spread_for(symbol, cfg)
    if not np.isfinite(spread) or spread > lim:
        return False
    return True


def slice_afternoon(
    ticks: pd.DataFrame,
    minute: pd.DataFrame,
    start_bar: int,
    end_bar: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if minute.empty:
        return ticks.iloc[0:0].copy(), minute.iloc[0:0].copy()
    sb = minute["session_bar"].astype(int)
    m_af = minute[(sb >= start_bar) & (sb <= end_bar)].copy()
    if m_af.empty:
        return ticks.iloc[0:0].copy(), m_af
    t0 = m_af["timestamp"].min()
    t_af = ticks[ticks["timestamp"] >= t0].copy()
    return t_af, m_af


def replay_day_roi(
    ticks: pd.DataFrame,
    minute: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    replay_cfg: ReplayConfig,
) -> dict:
    if minute.empty or len(minute) < 10:
        return {"skipped": True, "day_roi": 0.0, "trades": 0}
    minute_e = compute_oracle_edge(minute, fill_model, hold_bars=5)
    tick_df = ticks[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
    r = run_event_replay(
        minute_e,
        fill_model,
        replay_cfg,
        qcfg.EXIT_RAILS,
        tick_df=tick_df,
        edge_col="oracle_edge",
        event_cfg=EventReplayConfig(tick_disaster_stop=True),
    )
    sm = _replay_summary(r.trades)
    sm["day_roi"] = sm.pop("total_net_return", 0.0)
    sm["skipped"] = False
    return sm


def discover_symbol_days(
    raw_dir: Path,
    symbol: str,
    glob_pattern: str,
) -> List[Path]:
    """合并 options/ 与 options_databento/ 下匹配文件 (后者优先同名覆盖)。"""
    by_date: Dict[str, Path] = {}
    subdirs = ("options", "options_databento")
    for sub in subdirs:
        opt_dir = raw_dir / sub / symbol
        if not opt_dir.is_dir():
            continue
        for fp in sorted(opt_dir.glob(glob_pattern)):
            date_str = fp.stem.split("_", 1)[-1]
            by_date[date_str] = fp
    # 直接 {raw_dir}/{symbol}/ 布局
    direct = raw_dir / symbol
    if direct.is_dir():
        for fp in sorted(direct.glob(glob_pattern)):
            date_str = fp.stem.split("_", 1)[-1]
            by_date.setdefault(date_str, fp)
    return [by_date[d] for d in sorted(by_date)]


def discover_overlap_dates(
    raw_dir: Path,
    candidates: Sequence[str],
    glob_pattern: str,
) -> List[str]:
    sets: List[set] = []
    for sym in candidates:
        files = discover_symbol_days(raw_dir, sym, glob_pattern)
        dates = {fp.stem.split("_", 1)[-1] for fp in files}
        sets.append(dates)
    if not sets:
        return []
    overlap = sorted(set.intersection(*sets))
    return overlap


def load_candidate_day(
    raw_dir: Path,
    symbol: str,
    date_str: str,
    bucket_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fp = None
    for sub in ("options_databento", "options"):
        cand = raw_dir / sub / symbol / f"{symbol}_{date_str}.parquet"
        if cand.is_file():
            fp = cand
            break
    if fp is None:
        for cand in (raw_dir / symbol / f"{symbol}_{date_str}.parquet",):
            if cand.is_file():
                fp = cand
                break
    ticks = load_raw1s_bucket_day(fp, bucket_id) if fp is not None else pd.DataFrame()
    minute = build_minute_frame(ticks)
    stock_min = load_stock_minute_day(raw_dir, symbol, date_str)
    return ticks, minute, stock_min


def summarize_day_rois(rows: List[dict]) -> dict:
    rois = np.array([r["day_roi"] for r in rows if not r.get("skipped")], dtype=np.float64)
    traded = [r for r in rows if r.get("trades", 0) > 0]
    if len(rois) == 0:
        return {"days": 0}
    bad3 = float(np.sort(rois)[:3].sum()) if len(rois) >= 3 else float(rois.sum())
    return {
        "days": len(rows),
        "active_days": len(traded),
        "win_days": int((rois > 0).sum()),
        "skip_days": sum(1 for r in rows if r.get("skipped") or r.get("trades", 0) == 0),
        "trades": int(sum(r.get("trades", 0) for r in rows)),
        "day_roi_mean": _round4(float(rois.mean())),
        "day_roi_median": _round4(float(np.median(rois))),
        "compound_day_roi": _round4(float(np.prod(1.0 + rois) - 1.0)),
        "worst3_day_sum": _round4(bad3),
        "hit_rate": _round4(
            float(np.mean([t for r in traded for t in [r.get("hit_rate", 0)]]))
            if traded
            else 0.0
        ),
    }


def run_experiment(
    *,
    raw_dir: Path,
    candidates: Sequence[str],
    dates: Sequence[str],
    bucket_by_symbol: Dict[str, int],
    sel_cfg: SelectorConfig,
    fill_model: OptionSpreadFillModel,
    replay_afternoon: ReplayConfig,
) -> dict:
    per_day: List[dict] = []
    mode_rows: Dict[str, List[dict]] = {
        "A_always_qqq": [],
        "B_selector": [],
        "B_soft_score": [],
        "C_oracle_pick": [],
    }

    for date_str in dates:
        cand_data: Dict[str, dict] = {}
        for sym in candidates:
            bucket = bucket_by_symbol.get(sym, 2)
            ticks, minute, stock_min = load_candidate_day(raw_dir, sym, date_str, bucket)
            spread, n_ticks = _morning_option_spread(ticks, sel_cfg.decision_bar)
            r2, r2_src = _morning_trend_r2(stock_min, minute, sel_cfg.decision_bar)
            sc = score_candidate(r2, spread, sel_cfg)
            gate_ok = passes_gate(sym, r2, spread, n_ticks, sel_cfg)
            t_af, m_af = slice_afternoon(
                ticks,
                minute,
                sel_cfg.afternoon_start_bar,
                sel_cfg.afternoon_end_bar,
            )
            aft = replay_day_roi(t_af, m_af, fill_model, replay_afternoon)
            cand_data[sym] = {
                "spread_median": _round4(spread) if np.isfinite(spread) else None,
                "morning_ticks": n_ticks,
                "trend_r2": _round4(r2) if np.isfinite(r2) else None,
                "trend_source": r2_src,
                "score": _round4(sc) if np.isfinite(sc) else None,
                "gate_ok": gate_ok,
                "afternoon": aft,
            }

        # A: always QQQ afternoon
        qqq = cand_data.get("QQQ", {}).get("afternoon", {"skipped": True, "day_roi": 0.0, "trades": 0})
        mode_rows["A_always_qqq"].append(
            {"date": date_str, "symbol": "QQQ", "day_roi": qqq.get("day_roi", 0.0), **qqq}
        )

        # B: selector among gate_ok
        passing = [
            (sym, cand_data[sym]["score"])
            for sym in candidates
            if sym in cand_data and cand_data[sym]["gate_ok"]
        ]
        if passing:
            pick = max(passing, key=lambda x: x[1])[0]
            b = cand_data[pick]["afternoon"]
            mode_rows["B_selector"].append(
                {
                    "date": date_str,
                    "symbol": pick,
                    "day_roi": b.get("day_roi", 0.0),
                    **b,
                }
            )
        else:
            mode_rows["B_selector"].append(
                {"date": date_str, "symbol": None, "skipped": True, "day_roi": 0.0, "trades": 0}
            )

        # B_soft: 不过 gate, 直接选 score 最高 (spread 仍须 <= symbol limit)
        soft = []
        for sym in candidates:
            if sym not in cand_data:
                continue
            c = cand_data[sym]
            lim = _max_spread_for(sym, sel_cfg)
            sp = c["spread_median"]
            if sp is not None and np.isfinite(sp) and sp <= lim and c["score"] is not None:
                soft.append((sym, c["score"]))
        if soft:
            pick_soft = max(soft, key=lambda x: x[1])[0]
            bs = cand_data[pick_soft]["afternoon"]
            mode_rows["B_soft_score"].append(
                {
                    "date": date_str,
                    "symbol": pick_soft,
                    "day_roi": bs.get("day_roi", 0.0),
                    **bs,
                }
            )
        else:
            mode_rows["B_soft_score"].append(
                {"date": date_str, "symbol": None, "skipped": True, "day_roi": 0.0, "trades": 0}
            )

        # C: oracle pick best afternoon roi (leakage)
        best_sym = None
        best_roi = float("-inf")
        for sym in candidates:
            if sym not in cand_data:
                continue
            roi = cand_data[sym]["afternoon"].get("day_roi", 0.0)
            if roi > best_roi:
                best_roi = roi
                best_sym = sym
        c = cand_data.get(best_sym, {}).get("afternoon", {}) if best_sym else {}
        mode_rows["C_oracle_pick"].append(
            {
                "date": date_str,
                "symbol": best_sym,
                "day_roi": c.get("day_roi", 0.0),
                **c,
            }
        )

        per_day.append(
            {
                "date": date_str,
                "candidates": cand_data,
                "A_symbol": "QQQ",
                "B_symbol": mode_rows["B_selector"][-1]["symbol"],
                "C_symbol": best_sym,
            }
        )

    summary = {mode: summarize_day_rois(rows) for mode, rows in mode_rows.items()}
    b = summary.get("B_selector", {})
    a = summary.get("A_always_qqq", {})
    lift = {
        "win_days_delta": (b.get("win_days", 0) - a.get("win_days", 0)),
        "day_roi_mean_delta": _round4(b.get("day_roi_mean", 0) - a.get("day_roi_mean", 0)),
        "compound_delta": _round4(b.get("compound_day_roi", 0) - a.get("compound_day_roi", 0)),
        "worst3_delta": _round4(b.get("worst3_day_sum", 0) - a.get("worst3_day_sum", 0)),
    }

    return {
        "meta": {
            "raw_dir": str(raw_dir),
            "candidates": list(candidates),
            "dates": list(dates),
            "bucket_by_symbol": bucket_by_symbol,
            "selector": asdict(sel_cfg),
            "replay_afternoon": {
                "session_entry_start_bar": replay_afternoon.session_entry_start_bar,
                "session_entry_end_bar": replay_afternoon.session_entry_end_bar,
                "entry_threshold": replay_afternoon.entry_threshold,
            },
            "note": "oracle_edge 入场含未来信息; C_oracle_pick 含标的选择泄漏",
        },
        "summary": summary,
        "lift_B_vs_A": lift,
        "per_day": per_day,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="11:00 trend+spread selector validation")
    ap.add_argument("--raw-dir", default="/mnt/s990/data/raw_1s")
    ap.add_argument("--candidates", default="QQQ,AAPL")
    ap.add_argument("--glob", default="*_2026-03-*.parquet")
    ap.add_argument("--bucket-qqq", type=int, default=2)
    ap.add_argument("--bucket-aapl", type=int, default=2)
    ap.add_argument("--min-r2", type=float, default=0.35)
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument(
        "--out",
        default="New_Pro/baseline_qqq/reports/selector_validation_2026m03.json",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    candidates = [s.strip() for s in args.candidates.split(",") if s.strip()]
    bucket_map = {"QQQ": args.bucket_qqq, "AAPL": args.bucket_aapl}
    for sym in candidates:
        bucket_map.setdefault(sym, 2)

    dates = discover_overlap_dates(raw_dir, candidates, args.glob)
    if not dates:
        print("no overlapping dates found")
        return 1

    sel_cfg = SelectorConfig(min_r2=args.min_r2, max_spread_pct=args.max_spread)
    replay_afternoon = ReplayConfig(
        **{
            **qcfg.REPLAY.__dict__,
            "session_entry_start_bar": sel_cfg.afternoon_start_bar,
            "session_entry_end_bar": sel_cfg.afternoon_end_bar,
        }
    )

    result = run_experiment(
        raw_dir=raw_dir,
        candidates=candidates,
        dates=dates,
        bucket_by_symbol=bucket_map,
        sel_cfg=sel_cfg,
        fill_model=qcfg.FILL_MODEL,
        replay_afternoon=replay_afternoon,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"dates={len(dates)} -> {out}\n")
    for mode in ("A_always_qqq", "B_selector", "B_soft_score", "C_oracle_pick"):
        s = result["summary"][mode]
        print(
            f"{mode}: win={s.get('win_days', 0)}/{s.get('days', 0)} "
            f"active={s.get('active_days', 0)} skip={s.get('skip_days', 0)} "
            f"dayμ={s.get('day_roi_mean', 0):+.1%} compound={s.get('compound_day_roi', 0):+.1%} "
            f"worst3Σ={s.get('worst3_day_sum', 0):+.1%}"
        )
    lift = result["lift_B_vs_A"]
    print(
        f"\nB vs A: Δwin={lift['win_days_delta']:+d} "
        f"Δdayμ={lift['day_roi_mean_delta']:+.1%} "
        f"Δcompound={lift['compound_delta']:+.1%} "
        f"Δworst3={lift['worst3_delta']:+.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
