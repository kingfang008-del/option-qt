#!/usr/bin/env python3
"""QQQ raw_1s bucket quote + qqq_btc exit-rails validation (rule-only, no TFT)."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.common.fill_model import OptionSpreadFillModel
from qqq_btc.common.replay_types import ReplayConfig
from qqq_btc.common.time_features import session_minute
from qqq_btc.qqq import config as qcfg


@dataclass(frozen=True)
class QuoteGateConfig:
    min_seconds_after_open: int = 3
    max_spread_pct: float = 0.06
    min_bid_ask_size: float = 2.0
    stable_quote_seconds: int = 0  # 0 = disabled


def _round4(x: float) -> float:
    return round(float(x), 4)


def resolve_raw1s_option_dir(raw_dir: Path, symbol: str) -> Path:
    """
    兼容多种布局:
      {raw_dir}/options/{symbol}/              # 旧 polygon/raw_1s
      {raw_dir}/options_databento/{symbol}/    # 0DTE databento
      {raw_dir}/dte1_options/{symbol}/         # 1DTE polygon
      {raw_dir}/{symbol}/                      # 直接指向 options 根
    """
    candidates = [
        raw_dir / "options" / symbol,
        raw_dir / "options_databento" / symbol,
        raw_dir / "dte1_options" / symbol,
        raw_dir / symbol,
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"options dir not found for {symbol}; tried: "
        + ", ".join(str(p) for p in candidates)
    )


def discover_raw1s_days(
    raw_dir: Path,
    symbol: str,
    *,
    glob_pattern: Optional[str] = None,
    batch_days: Optional[int] = None,
) -> List[Path]:
    opt_dir = resolve_raw1s_option_dir(raw_dir, symbol)
    if glob_pattern:
        files = sorted(opt_dir.glob(glob_pattern))
    else:
        files = sorted(opt_dir.glob(f"{symbol}_*.parquet"))
    if batch_days is not None and batch_days > 0:
        files = files[-batch_days:]
    return files


def load_raw1s_bucket_day(path: Path, bucket_id: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "bucket_id" in df.columns:
        df["bucket_id"] = pd.to_numeric(df["bucket_id"], errors="coerce")
        sub = df[df["bucket_id"] == bucket_id].copy()
    else:
        sub = df.copy()
    if sub.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(sub["timestamp"]):
        sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    if sub["timestamp"].dt.tz is None:
        sub["timestamp"] = sub["timestamp"].dt.tz_localize(
            "America/New_York", ambiguous="infer"
        )
    else:
        sub["timestamp"] = sub["timestamp"].dt.tz_convert("America/New_York")

    sub = sub.sort_values("timestamp")
    bid = pd.to_numeric(sub["bid"], errors="coerce")
    ask = pd.to_numeric(sub["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    bid_size = pd.to_numeric(sub.get("bid_size", 0), errors="coerce").fillna(0)
    ask_size = pd.to_numeric(sub.get("ask_size", 0), errors="coerce").fillna(0)
    spread_pct = np.where(mid > 0, (ask - bid) / mid, np.nan)

    sub = sub.assign(
        exec_call_bid=bid,
        exec_call_ask=ask,
        exec_call_mid=mid,
        exec_call_spread_pct=spread_pct,
        bid_size=bid_size,
        ask_size=ask_size,
        minute_ts=sub["timestamp"].dt.floor("min"),
    )
    t = sub["timestamp"]
    rth = ((t.dt.hour > 9) | ((t.dt.hour == 9) & (t.dt.minute >= 30))) & (t.dt.hour < 16)
    return sub[rth].reset_index(drop=True)


def build_minute_frame(ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame()
    minute = ticks.groupby("minute_ts", as_index=False).last()
    minute = minute.drop(columns=["timestamp"], errors="ignore")
    minute = minute.rename(columns={"minute_ts": "timestamp"})
    minute["session_bar"] = session_minute(minute["timestamp"]).astype(int)
    return minute.sort_values("timestamp").reset_index(drop=True)


def _sec_from_open(ts: pd.Series) -> pd.Series:
    return (
        (ts.dt.hour - 9) * 3600
        + (ts.dt.minute - 30) * 60
        + ts.dt.second
    )


def compute_quote_stats(ticks: pd.DataFrame, minute: pd.DataFrame) -> dict:
    if minute.empty:
        return {"ticks": 0, "minutes": 0}
    sp = minute["exec_call_spread_pct"].dropna()
    open_ticks = ticks[_sec_from_open(ticks["timestamp"]) >= 0].copy()
    open_ticks["sfo"] = _sec_from_open(open_ticks["timestamp"])
    first_ok = None
    for sec in range(0, 120):
        seg = open_ticks[open_ticks["sfo"] == sec]
        if seg.empty:
            continue
        med_sp = float(seg["exec_call_spread_pct"].median())
        if med_sp <= 0.06:
            first_ok = sec
            break
    return {
        "ticks": int(len(ticks)),
        "minutes": int(len(minute)),
        "spread_median": _round4(float(sp.median())),
        "spread_p90": _round4(float(sp.quantile(0.9))),
        "spread_p99": _round4(float(sp.quantile(0.99))),
        "spread_gt_6pct_frac": _round4(float((sp > 0.06).mean())),
        "first_spread_ok_second_after_open": first_ok,
    }


def compute_intra_minute_stats(ticks: pd.DataFrame) -> dict:
    rows: List[dict] = []
    for _, grp in ticks.groupby("minute_ts"):
        if len(grp) < 3:
            continue
        mid = grp["exec_call_mid"].to_numpy()
        if not np.all(np.isfinite(mid) & (mid > 0)):
            continue
        base = mid[0]
        roi = mid / base - 1.0
        rows.append(
            {
                "range": float(roi.max() - roi.min()),
                "max_dn": float(roi.min()),
                "close_vs_open": float(roi[-1]),
            }
        )
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for col in ("range", "max_dn", "close_vs_open"):
        s = df[col]
        out[col] = {
            "median": _round4(float(s.median())),
            "p90": _round4(float(s.quantile(0.9))),
            "p99": _round4(float(s.quantile(0.99))),
        }
    return out


def compute_tick_rails_conflict(ticks: pd.DataFrame, rails: ExitRailsConfig) -> dict:
    n_min = 0
    false_hard = false_soft = false_disaster = ladder_risk = 0
    for _, grp in ticks.groupby("minute_ts"):
        if len(grp) < 5:
            continue
        entry = float(grp.iloc[0]["exec_call_mid"])
        if not (np.isfinite(entry) and entry > 0):
            continue
        rois = grp["exec_call_mid"].to_numpy() / entry - 1.0
        close = float(rois[-1])
        n_min += 1
        if rois.min() <= rails.hard_stop_roi and close > rails.hard_stop_roi:
            false_hard += 1
        if rois.min() <= rails.soft_stop_roi and close > rails.soft_stop_roi:
            false_soft += 1
        if rails.disaster_stop_roi is not None and rois.min() <= rails.disaster_stop_roi:
            false_disaster += 1
        if rois.max() >= 0.08 and close < 0.05:
            ladder_risk += 1
    if n_min == 0:
        return {"minutes": 0}
    return {
        "minutes": n_min,
        "shadow_hard_but_close_ok_frac": _round4(false_hard / n_min),
        "shadow_soft_but_close_ok_frac": _round4(false_soft / n_min),
        "shadow_disaster_frac": _round4(false_disaster / n_min),
        "peak_ge_8pct_close_lt_5pct_frac": _round4(ladder_risk / n_min),
    }


def compute_oracle_edge(
    minute: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    hold_bars: int = 5,
) -> pd.DataFrame:
    out = minute.copy()
    n = len(out)
    bid = out["exec_call_bid"].to_numpy(dtype=np.float64)
    ask = out["exec_call_ask"].to_numpy(dtype=np.float64)
    fwd = np.full(n, np.nan)
    for i in range(n - hold_bars):
        ef = fill_model.entry_fill(bid[i], ask[i])
        xf = fill_model.exit_fill(bid[i + hold_bars], ask[i + hold_bars])
        if ef > 0 and xf > 0:
            fwd[i] = xf / ef - 1.0 - fill_model.commission_return_drag(ef)
    out["oracle_edge"] = fwd
    return out


def _replay_summary(trades: Sequence[Any]) -> dict:
    if not trades:
        return {"trades": 0, "total_net_return": 0.0, "hit_rate": 0.0, "exit_reasons": {}}
    rets = np.array([t.net_return for t in trades])
    reasons = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()
    return {
        "trades": int(len(trades)),
        "total_net_return": _round4(float(np.prod(1.0 + rets) - 1.0)),
        "avg_net_return": _round4(float(rets.mean())),
        "hit_rate": _round4(float((rets > 0).mean())),
        "worst_trade": _round4(float(rets.min())),
        "exit_reasons": {str(k): int(v) for k, v in reasons.items()},
    }


def run_day_rails_replay(
    ticks: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    replay_cfg: ReplayConfig,
    rails_cfg: ExitRailsConfig,
    fill_model: OptionSpreadFillModel,
    edge_col: str = "oracle_edge",
) -> dict:
    if minute.empty or len(minute) < 30:
        return {"skipped": True, "reason": "insufficient_minutes"}
    tick_df = ticks[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
    r_l1 = run_event_replay(
        minute, fill_model, replay_cfg, rails_cfg, tick_df=None, edge_col=edge_col
    )
    r_l2 = run_event_replay(
        minute,
        fill_model,
        replay_cfg,
        rails_cfg,
        tick_df=tick_df,
        edge_col=edge_col,
        event_cfg=EventReplayConfig(tick_disaster_stop=True),
    )
    return {
        "l1_minute": _replay_summary(r_l1.trades),
        "l2_1s_disaster": _replay_summary(r_l2.trades),
    }


def compute_false_kill_stats(
    ticks: pd.DataFrame,
    minute: pd.DataFrame,
    fill_model: OptionSpreadFillModel,
    *,
    entry_threshold: float = 0.015,
    early_stop_roi: float = -0.05,
    early_bars: int = 5,
    time_stop_roi: float = 0.05,
    time_bars: int = 15,
) -> dict:
    minute = compute_oracle_edge(minute, fill_model, hold_bars=5)
    entries = minute[minute["oracle_edge"] >= entry_threshold]
    early_killed = early_saved = time_killed = time_saved = 0
    total = 0
    for _, row in entries.iterrows():
        mk = row["timestamp"].floor("min")
        grp = ticks[ticks["minute_ts"] >= mk].head(1800)
        if len(grp) < 60:
            continue
        ef = fill_model.entry_fill(row["exec_call_bid"], row["exec_call_ask"])
        if not (np.isfinite(ef) and ef > 0):
            continue
        roi = grp["exec_call_mid"].to_numpy() / ef - 1.0
        final = float(roi[min(len(roi) - 1, 1799)])
        roi_early = float(roi[min(early_bars * 60 - 1, len(roi) - 1)])
        roi_time = float(roi[min(time_bars * 60 - 1, len(roi) - 1)])
        total += 1
        if roi_early <= early_stop_roi:
            if final > 0:
                early_killed += 1
            else:
                early_saved += 1
        if roi_time < time_stop_roi:
            if final > time_stop_roi:
                time_killed += 1
            else:
                time_saved += 1
    if total == 0:
        return {"entry_points": 0}
    return {
        "entry_points": total,
        "early_stop_false_kill_frac": _round4(early_killed / total),
        "early_stop_correct_kill_frac": _round4(early_saved / total),
        "time_stop_false_kill_frac": _round4(time_killed / total),
        "time_stop_correct_kill_frac": _round4(time_saved / total),
    }


def quote_gate_grid(
    ticks: pd.DataFrame,
    gate_grid: Sequence[QuoteGateConfig],
) -> List[dict]:
    if ticks.empty:
        return []
    ticks = ticks.copy()
    ticks["sfo"] = _sec_from_open(ticks["timestamp"])
    results = []
    for gate in gate_grid:
        ok = (
            (ticks["sfo"] >= gate.min_seconds_after_open)
            & (ticks["exec_call_spread_pct"] <= gate.max_spread_pct)
            & (ticks["bid_size"] >= gate.min_bid_ask_size)
            & (ticks["ask_size"] >= gate.min_bid_ask_size)
        )
        if gate.stable_quote_seconds > 0:
            stable = np.zeros(len(ticks), dtype=bool)
            for i in range(len(ticks)):
                if not ok.iloc[i]:
                    continue
                t0 = ticks.iloc[i]["timestamp"]
                window = ticks[
                    (ticks["timestamp"] >= t0 - pd.Timedelta(seconds=gate.stable_quote_seconds - 1))
                    & (ticks["timestamp"] <= t0)
                ]
                stable[i] = len(window) >= gate.stable_quote_seconds and ok.loc[window.index].all()
            passed = int(stable.sum())
        else:
            passed = int(ok.sum())
        results.append(
            {
                "gate": asdict(gate),
                "ticks_passed": passed,
                "ticks_passed_frac": _round4(passed / len(ticks)),
            }
        )
    return results


def rails_variants() -> List[Tuple[str, ExitRailsConfig, Optional[ReplayConfig]]]:
    base_replay = qcfg.REPLAY
    cur = qcfg.EXIT_RAILS
    variants: List[Tuple[str, ExitRailsConfig, Optional[ReplayConfig]]] = [
        ("current", cur, None),
        (
            "no_early_stop",
            ExitRailsConfig(**{**cur.__dict__, "early_stop_bars": None}),
            None,
        ),
        (
            "time_stop_20min_3pct",
            ExitRailsConfig(**{**cur.__dict__, "time_stop_bars": 20, "time_stop_min_roi": 0.03}),
            None,
        ),
        (
            "time_stop_off",
            ExitRailsConfig(**{**cur.__dict__, "time_stop_bars": 999}),
            None,
        ),
        (
            "ladder_v0_first_tier",
            ExitRailsConfig(
                **{
                    **cur.__dict__,
                    "ladder": (
                        (0.05, 0.02),
                        (0.08, 0.05),
                        (0.12, 0.08),
                        (0.18, 0.12),
                        (0.25, 0.18),
                    ),
                    "flash_trigger_roi": 0.05,
                    "flash_exit_roi": 0.02,
                }
            ),
            None,
        ),
        (
            "entry_threshold_0.020",
            cur,
            ReplayConfig(
                **{
                    **base_replay.__dict__,
                    "entry_threshold": 0.020,
                    "entry_threshold_schedule": ((0, 0.020), (270, 0.025), (330, 0.030)),
                }
            ),
        ),
    ]
    return variants


def run_sensitivity_batch(
    all_days: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
    fill_model: OptionSpreadFillModel,
    *,
    fill_fracs: Sequence[float] = (0.775,),
) -> dict:
    out: Dict[str, dict] = {}
    for frac in fill_fracs:
        fm = OptionSpreadFillModel(entry_frac=frac, exit_frac=frac)
        frac_key = f"fill_{frac:.3f}"
        out[frac_key] = {}
        for label, rails, replay_override in rails_variants():
            trades = []
            for _date, ticks, minute in all_days:
                if minute.empty or len(minute) < 60:
                    continue
                minute_e = compute_oracle_edge(minute, fm, hold_bars=5)
                replay_cfg = replay_override or qcfg.REPLAY
                tick_df = ticks[["timestamp", "exec_call_bid", "exec_call_ask", "exec_call_spread_pct"]]
                r = run_event_replay(
                    minute_e,
                    fm,
                    replay_cfg,
                    rails,
                    tick_df=tick_df,
                    edge_col="oracle_edge",
                    event_cfg=EventReplayConfig(tick_disaster_stop=True),
                )
                trades.extend(r.trades)
            out[frac_key][label] = _replay_summary(trades)
    return out


def _mean_dict(dicts: List[dict], key: str) -> Optional[float]:
    vals = [d[key] for d in dicts if key in d and d[key] is not None]
    return _round4(float(np.mean(vals))) if vals else None


def aggregate_per_day_reports(per_day: List[dict]) -> dict:
    quote_stats = [d["quote_stats"] for d in per_day if d.get("quote_stats")]
    conflicts = [d["tick_rails_conflict"] for d in per_day if d.get("tick_rails_conflict")]
    replays = [
        d["rails_replay"]["l2_1s_disaster"]
        for d in per_day
        if d.get("rails_replay") and "l2_1s_disaster" in d["rails_replay"]
    ]
    false_kill = [d["false_kill"] for d in per_day if d.get("false_kill")]
    return {
        "days": len(per_day),
        "days_with_ticks": sum(1 for d in per_day if d.get("quote_stats", {}).get("ticks", 0) > 0),
        "quote_stats": {
            "spread_median_mean": _mean_dict(quote_stats, "spread_median"),
            "spread_gt_6pct_frac_mean": _mean_dict(quote_stats, "spread_gt_6pct_frac"),
        },
        "tick_rails_conflict_mean": {
            "shadow_hard_but_close_ok_frac": _mean_dict(conflicts, "shadow_hard_but_close_ok_frac"),
            "shadow_soft_but_close_ok_frac": _mean_dict(conflicts, "shadow_soft_but_close_ok_frac"),
        },
        "oracle_replay_l2_mean": {
            "trades_total": int(sum(r.get("trades", 0) for r in replays)),
            "hit_rate_mean": _mean_dict(replays, "hit_rate"),
        },
        "false_kill_mean": {
            "time_stop_false_kill_frac": _mean_dict(false_kill, "time_stop_false_kill_frac"),
            "early_stop_false_kill_frac": _mean_dict(false_kill, "early_stop_false_kill_frac"),
        },
    }


def validate_raw1s_batch(
    *,
    raw_dir: Path,
    symbol: str,
    bucket_id: int,
    files: Sequence[Path],
    fill_model: OptionSpreadFillModel,
    rails_cfg: ExitRailsConfig,
    replay_cfg: ReplayConfig,
    gate_grid: Sequence[QuoteGateConfig],
    run_sensitivity: bool,
    fill_sensitivity: Sequence[float],
) -> dict:
    per_day: List[dict] = []
    all_days: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []

    for fp in files:
        date_str = fp.stem.split("_", 1)[-1] if "_" in fp.stem else fp.stem
        ticks = load_raw1s_bucket_day(fp, bucket_id)
        minute = build_minute_frame(ticks)
        day_report: dict = {
            "date": date_str,
            "source": str(fp),
            "bucket_id": bucket_id,
        }
        if ticks.empty:
            day_report["skipped"] = True
            day_report["reason"] = "no_bucket_ticks"
            per_day.append(day_report)
            continue

        day_report["quote_stats"] = compute_quote_stats(ticks, minute)
        day_report["intra_minute"] = compute_intra_minute_stats(ticks)
        day_report["tick_rails_conflict"] = compute_tick_rails_conflict(ticks, rails_cfg)
        minute_e = compute_oracle_edge(minute, fill_model, hold_bars=5)
        day_report["false_kill"] = compute_false_kill_stats(ticks, minute_e, fill_model)
        day_report["quote_gate"] = quote_gate_grid(ticks, gate_grid)
        day_report["rails_replay"] = run_day_rails_replay(
            ticks,
            minute_e,
            replay_cfg=replay_cfg,
            rails_cfg=rails_cfg,
            fill_model=fill_model,
        )
        per_day.append(day_report)
        all_days.append((date_str, ticks, minute))

    result: dict = {
        "meta": {
            "mode": "raw_1s_batch",
            "raw_1s_dir": str(raw_dir),
            "symbol": symbol,
            "bucket_id": bucket_id,
            "days_requested": len(files),
            "rails": {
                "hard_stop_roi": rails_cfg.hard_stop_roi,
                "soft_stop_roi": rails_cfg.soft_stop_roi,
                "early_stop_bars": rails_cfg.early_stop_bars,
                "early_stop_roi": rails_cfg.early_stop_roi,
                "time_stop_bars": rails_cfg.time_stop_bars,
                "time_stop_min_roi": rails_cfg.time_stop_min_roi,
                "tick_fast_hard_roi": rails_cfg.tick_fast_hard_roi,
                "tick_fast_hard_smooth_n": rails_cfg.tick_fast_hard_smooth_n,
                "disaster_stop_roi": rails_cfg.disaster_stop_roi,
                "disaster_smooth_n": rails_cfg.disaster_smooth_n,
                "tick_profit_trigger_roi": rails_cfg.tick_profit_trigger_roi,
                "tick_profit_keep_ratio": rails_cfg.tick_profit_keep_ratio,
                "tick_profit_ladder": list(rails_cfg.tick_profit_ladder),
            },
            "replay": {
                "entry_threshold": replay_cfg.entry_threshold,
                "max_spread_pct": replay_cfg.max_spread_pct,
                "max_trades_per_day": replay_cfg.max_trades_per_day,
            },
            "fill_model": {
                "entry_frac": fill_model.entry_frac,
                "exit_frac": fill_model.exit_frac,
            },
        },
        "aggregate": aggregate_per_day_reports(per_day),
        "per_day": per_day,
    }
    if run_sensitivity and all_days:
        result["sensitivity"] = run_sensitivity_batch(
            all_days, fill_model, fill_fracs=fill_sensitivity
        )
    return result


def write_batch_reports(result: dict, out: Path, *, write_per_day: bool = True) -> Tuple[Path, Optional[Path]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_day_dir = None
    if write_per_day:
        per_day_dir = out.parent / f"{out.stem}_days"
        per_day_dir.mkdir(parents=True, exist_ok=True)
        for day in result.get("per_day", []):
            day_path = per_day_dir / f"{day.get('date', 'unknown')}.json"
            day_path.write_text(
                json.dumps(day, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return out, per_day_dir
