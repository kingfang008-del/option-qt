#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FT56 每周分钟基线 vs 生产秒级护栏回归。"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg


def _build_ticks(root: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("QQQ_*.parquet")):
        day = path.stem.replace("QQQ_", "")
        if not start <= day <= end:
            continue
        raw = pd.read_parquet(path, columns=["timestamp", "bucket_id", "bid", "ask"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.floor("s")
        raw = raw[raw["bucket_id"].isin([0, 2])].copy()
        raw["leg"] = raw["bucket_id"].map({0: "put", 2: "call"})
        wide = raw.pivot_table(
            index="timestamp", columns="leg", values=["bid", "ask"], aggfunc="last"
        )
        seconds = pd.date_range(
            wide.index.min().ceil("s"), wide.index.max().floor("s"), freq="1s", tz="UTC"
        )
        wide = wide.reindex(seconds).ffill().dropna()
        wide.columns = [f"exec_{leg}_{field}" for field, leg in wide.columns]
        frames.append(wide.reset_index().rename(columns={"index": "timestamp"}))
    if not frames:
        raise FileNotFoundError(f"no QQQ tick files for {start}..{end} under {root}")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def _summarize(result, position_frac: float) -> tuple[pd.DataFrame, dict]:
    trades = result.trades_frame().copy()
    if trades.empty:
        return trades, {
            "account_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "hit_rate": 0.0,
            "worst_trade": 0.0,
            "exit_reasons": {},
        }
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    equity = (1.0 + position_frac * trades["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return trades, {
        "account_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "trades": int(len(trades)),
        "hit_rate": float((trades["net_return"] > 0).mean()),
        "worst_trade": float(trades["net_return"].min()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--infer",
        default="qqq_btc/results/ft56_julw1_honest_infer_fixed5m/test_infer.parquet",
    )
    parser.add_argument(
        "--tick-root",
        default="/mnt/s990/data/v4_original_jul5/databento_july_w1_openwin/raw_1s/QQQ",
    )
    parser.add_argument(
        "--vix-feature",
        default="~/train_data/july_w1_v4_honest_openwin/quote_features_raw/"
        "QQQ/regular/09:30-16:00/1min/2026-07.parquet",
    )
    parser.add_argument("--edge-q10-floor", type=float, default=-0.20)
    parser.add_argument("--entry-quantile-min-obs", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    minute = pd.read_parquet(Path(args.infer).expanduser()).copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True)
    ny_day = minute["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    minute = minute[(ny_day >= args.start) & (ny_day <= args.end)].reset_index(drop=True)

    vix = pd.read_parquet(
        Path(args.vix_feature).expanduser(), columns=["timestamp", "vix_level"]
    )
    vix["timestamp"] = pd.to_datetime(vix["timestamp"], utc=True)
    vix = (
        vix.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .rename(columns={"vix_level": "put_gate"})
    )
    vix["timestamp"] += pd.Timedelta(minutes=1)
    minute["put_gate"] = pd.merge_asof(
        minute[["timestamp"]], vix, on="timestamp", direction="backward"
    )["put_gate"].to_numpy()
    ticks = _build_ticks(Path(args.tick_root).expanduser(), args.start, args.end)

    replay_cfg = replace(qcfg.LIVE_REPLAY, edge_q10_floor=args.edge_q10_floor)
    if args.entry_quantile_min_obs is not None:
        replay_cfg = replace(
            replay_cfg, entry_quantile_min_obs=args.entry_quantile_min_obs
        )
    common = dict(
        edge_col="net_edge",
        edge_q10_col="net_edge_q10",
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
        put_gate_col="put_gate",
    )
    runs = {
        "minute_baseline": run_strict_replay(
            minute, qcfg.FILL_MODEL, replay_cfg, qcfg.EXIT_RAILS, **common
        ),
        "production_tick_rails": run_event_replay(
            minute,
            qcfg.FILL_MODEL,
            replay_cfg,
            qcfg.EXIT_RAILS,
            tick_df=ticks,
            event_cfg=EventReplayConfig(use_tick_quotes_for_minute_close=False),
            **common,
        ),
    }

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "period": {"start": args.start, "end": args.end},
        "config": {
            "position_frac": replay_cfg.position_frac,
            "edge_q10_floor": replay_cfg.edge_q10_floor,
            "entry_quantile_min_obs": replay_cfg.entry_quantile_min_obs,
            "tick_fast_hard_roi": qcfg.EXIT_RAILS.tick_fast_hard_roi,
            "tick_fast_hard_smooth_n": qcfg.EXIT_RAILS.tick_fast_hard_smooth_n,
            "tick_stop_cooldown_bars": replay_cfg.tick_stop_cooldown_bars,
            "tick_stop_lock_leg_for_day": replay_cfg.tick_stop_lock_leg_for_day,
            "disaster_stop_roi": qcfg.EXIT_RAILS.disaster_stop_roi,
            "tick_profit_trigger_roi": qcfg.EXIT_RAILS.tick_profit_trigger_roi,
            "tick_profit_ladder": qcfg.EXIT_RAILS.tick_profit_ladder,
        },
        "runs": {},
    }
    for name, result in runs.items():
        trades, metrics = _summarize(result, replay_cfg.position_frac)
        summary["runs"][name] = metrics
        trades.to_parquet(out / f"{name}_trades.parquet", index=False)
    summary["tick_vs_minute_account_delta"] = (
        summary["runs"]["production_tick_rails"]["account_return"]
        - summary["runs"]["minute_baseline"]["account_return"]
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
