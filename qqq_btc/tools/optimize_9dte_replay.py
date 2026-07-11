#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
9DTE legacy replay 逐步优化: val 标定 → test 验证。

Step 1: entry_threshold × entry_quantile 网格 (val)
Step 2: exit rails MAE/MFE 标定 (val, 用 step1 最优阈值)
Step 3: 合并参数在 test 月 strict replay

用法:
  python qqq_btc/tools/optimize_9dte_replay.py
  python qqq_btc/tools/optimize_9dte_replay.py --step 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.common.exit_rails import ExitRailsConfig
from qqq_btc.qqq import config_9dte_legacy as cfg9
from qqq_btc.tools.calibrate_rails import suggest_rails, compute_roi_paths, entry_bar_mask, merge_bucket_suggestions
from qqq_btc.tools.eval_9dte_legacy_replay import run_month

logger = logging.getLogger("optimize_9dte")


def month_return(result) -> float:
    f = result.summary().get("position_frac", 0.25)
    eq = 1.0
    for t in result.trades:
        eq *= 1.0 + f * t.net_return
    return (eq - 1.0) * 100.0


def replay_df(df: pd.DataFrame, replay_cfg, rails_cfg) -> tuple[Any, float]:
    r = run_strict_replay(
        df,
        cfg9.FILL_MODEL,
        replay_cfg,
        rails_cfg,
        edge_col="net_edge",
        edge_q10_col=cfg9.EDGE_Q10_COL,
        call_edge_col=cfg9.CALL_EDGE_COL,
        put_edge_col=cfg9.PUT_EDGE_COL,
        put_gate_col=cfg9.PUT_GATE_COL,
    )
    return r, month_return(r)


def step1_threshold_grid(df: pd.DataFrame, rails) -> dict:
    thresholds = [0.0008, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]
    quantiles = [None, 0.80, 0.85, 0.90]
    rows = []
    best = None
    for th in thresholds:
        for q in quantiles:
            rep = replace(
                cfg9.REPLAY,
                entry_threshold=th,
                entry_threshold_schedule=((15, th), (270, th * 1.5), (330, th * 2.0)),
                entry_quantile=q,
            )
            r, ret = replay_df(df, rep, rails)
            s = r.summary()
            row = {
                "entry_threshold": th,
                "entry_quantile": q,
                "month_return_pct": ret,
                "trades": s.get("trades", 0),
                "hit_rate": s.get("hit_rate", 0),
                "profit_factor": s.get("profit_factor", 0),
                "max_drawdown_mtm": s.get("max_drawdown_mtm", 0),
            }
            rows.append(row)
            if best is None or ret > best["month_return_pct"]:
                best = row
            logger.info(
                "step1 th=%.4f q=%s ret=%.2f%% trades=%d pf=%.2f",
                th, q, ret, row["trades"], row["profit_factor"] or 0,
            )
    return {"grid": rows, "best": best}


def step2_calibrate_rails(df: pd.DataFrame, entry_threshold: float, entry_quantile) -> dict:
    mask = entry_bar_mask(
        df,
        edge_col="net_edge",
        entry_threshold=entry_threshold,
        max_spread_pct=cfg9.REPLAY.max_spread_pct,
        edge_q10_col=cfg9.EDGE_Q10_COL,
        edge_q10_floor=cfg9.REPLAY.edge_q10_floor,
        session_entry_end_bar=cfg9.REPLAY.session_entry_end_bar,
    )
    paths = compute_roi_paths(
        df, cfg9.FILL_MODEL,
        max_hold_bars=cfg9.EXIT_RAILS.max_hold_bars,
        horizon_bars=(cfg9.EXIT_RAILS.early_stop_bars or 15, cfg9.EXIT_RAILS.time_stop_bars),
        entry_mask=mask,
    )
    report = suggest_rails(
        paths,
        early_stop_horizon=cfg9.EXIT_RAILS.early_stop_bars or 15,
        time_stop_horizon=cfg9.EXIT_RAILS.time_stop_bars,
    )
    merged = report.get("_merged_conservative") or merge_bucket_suggestions(
        {k: v for k, v in report.items() if not k.startswith("_")}
    )
    return {"report": report, "merged": merged, "path_samples": int(len(paths))}


def apply_rails_suggestion(base: ExitRailsConfig, merged: dict) -> ExitRailsConfig:
    if not merged:
        return base
    kw = {}
    for k in (
        "soft_stop_roi", "hard_stop_roi", "early_stop_bars", "early_stop_roi",
        "time_stop_bars", "time_stop_min_roi", "trailing_trigger_roi",
        "trailing_keep_ratio", "flash_trigger_roi", "flash_exit_roi", "disaster_stop_roi",
    ):
        if k in merged and merged[k] is not None:
            kw[k] = merged[k]
    if merged.get("ladder"):
        kw["ladder"] = tuple(tuple(x) for x in merged["ladder"])
    return replace(base, **kw)


def step3_eval(df: pd.DataFrame, replay_cfg, rails_cfg) -> dict:
    r, ret = replay_df(df, replay_cfg, rails_cfg)
    s = r.summary()
    s["month_return_pct"] = ret
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="9DTE replay stepwise optimize")
    parser.add_argument("--val-month", default="2026-01")
    parser.add_argument("--test-month", default="2026-02")
    parser.add_argument(
        "--checkpoint",
        default=str(_REPO / "checkpoints_qqq_9dte_janval_febtest/best.pth"),
    )
    parser.add_argument("--feature-root", default=str(cfg9.FEATURE_VAL_ROOT))
    parser.add_argument("--option-1m-root", default=str(cfg9.RAW_1M_ROOT))
    parser.add_argument("--config", default=str(cfg9.FEATURE_CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(_REPO / "qqq_btc/results"))
    parser.add_argument("--step", default="all", choices=["all", "1", "2", "3"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path("/tmp/qqq_btc_opt_9dte")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sym = json.loads((_REPO / "qqq_btc/CONFIG/symbol_map.json").read_text())["QQQ"]

    # infer + attach (val & test)
    val_infer_dir = cache / "val"
    test_infer_dir = cache / "test"
    if not (val_infer_dir / f"infer_{args.val_month.replace('-', '')}.parquet").exists():
        run_month(
            args.val_month,
            checkpoint=Path(args.checkpoint),
            feature_root=Path(args.feature_root),
            option_1m_root=Path(args.option_1m_root),
            output_dir=val_infer_dir,
            config=cfg,
            stock_id=int(sym["stock_id"]),
            sector_id=int(sym["sector_id"]),
        )
    if not (test_infer_dir / f"infer_{args.test_month.replace('-', '')}.parquet").exists():
        run_month(
            args.test_month,
            checkpoint=Path(args.checkpoint),
            feature_root=Path(args.feature_root),
            option_1m_root=Path(args.option_1m_root),
            output_dir=test_infer_dir,
            config=cfg,
            stock_id=int(sym["stock_id"]),
            sector_id=int(sym["sector_id"]),
        )

    df_val = pd.read_parquet(val_infer_dir / f"infer_{args.val_month.replace('-', '')}.parquet")
    df_test = pd.read_parquet(test_infer_dir / f"infer_{args.test_month.replace('-', '')}.parquet")

    result: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "val_month": args.val_month,
        "test_month": args.test_month,
        "baseline_val": step3_eval(df_val, cfg9.REPLAY, cfg9.EXIT_RAILS),
        "baseline_test": step3_eval(df_test, cfg9.REPLAY, cfg9.EXIT_RAILS),
    }

    replay_cfg = cfg9.REPLAY
    rails_cfg = cfg9.EXIT_RAILS

    if args.step in ("all", "1"):
        logger.info("=== Step 1: entry threshold grid on val ===")
        s1 = step1_threshold_grid(df_val, cfg9.EXIT_RAILS)
        result["step1"] = s1
        b = s1["best"]
        replay_cfg = replace(
            cfg9.REPLAY,
            entry_threshold=b["entry_threshold"],
            entry_threshold_schedule=(
                (15, b["entry_threshold"]),
                (270, b["entry_threshold"] * 1.5),
                (330, b["entry_threshold"] * 2.0),
            ),
            entry_quantile=b["entry_quantile"],
        )
        result["after_step1_val"] = step3_eval(df_val, replay_cfg, rails_cfg)
        (out_dir / "9dte_opt_step1_grid.json").write_text(
            json.dumps(s1, indent=2, default=str), encoding="utf-8",
        )

    if args.step in ("all", "2"):
        if "step1" not in result:
            state = json.loads((out_dir / "9dte_opt_step1_grid.json").read_text())
            b = state["best"]
            replay_cfg = replace(
                cfg9.REPLAY,
                entry_threshold=b["entry_threshold"],
                entry_threshold_schedule=(
                    (15, b["entry_threshold"]),
                    (270, b["entry_threshold"] * 1.5),
                    (330, b["entry_threshold"] * 2.0),
                ),
                entry_quantile=b["entry_quantile"],
            )
        logger.info("=== Step 2: exit rails calibrate on val ===")
        s2 = step2_calibrate_rails(
            df_val,
            replay_cfg.entry_threshold,
            replay_cfg.entry_quantile,
        )
        rails_cfg = apply_rails_suggestion(cfg9.EXIT_RAILS, s2.get("merged") or {})
        result["step2"] = s2
        result["after_step2_val"] = step3_eval(df_val, replay_cfg, rails_cfg)
        (out_dir / "9dte_opt_step2_rails.json").write_text(
            json.dumps(s2, indent=2, default=str), encoding="utf-8",
        )

    if args.step in ("all", "3"):
        logger.info("=== Step 3: test replay with optimized params ===")
        result["optimized_test"] = step3_eval(df_test, replay_cfg, rails_cfg)
        result["optimized_replay_cfg"] = {
            "entry_threshold": replay_cfg.entry_threshold,
            "entry_quantile": replay_cfg.entry_quantile,
            "entry_threshold_schedule": list(replay_cfg.entry_threshold_schedule),
        }
        result["optimized_rails"] = {
            k: getattr(rails_cfg, k)
            for k in rails_cfg.__dataclass_fields__
            if k not in ("tick_profit_ladder",)
        }
        result["optimized_rails"]["ladder"] = list(rails_cfg.ladder)

    summary_path = out_dir / "9dte_opt_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "baseline_test_return_pct": result["baseline_test"].get("month_return_pct"),
        "optimized_test_return_pct": result.get("optimized_test", {}).get("month_return_pct"),
        "step1_best": result.get("step1", {}).get("best"),
        "summary": str(summary_path),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
