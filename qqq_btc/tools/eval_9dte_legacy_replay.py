#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
9DTE legacy 专用 infer + strict replay —— 不触碰 qqq.config (V4/0DTE)。

用法:
  python qqq_btc/tools/eval_9dte_legacy_replay.py --month 2026-03
  python qqq_btc/tools/eval_9dte_legacy_replay.py --feature-root ~/train_data/quote_features_val_9dte_legacy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.tools.eval_test_set import (
    merge_1m_5m,
    _feat_names_by_res,
    drop_embedded_exec_columns,
    attach_exec_quotes,
    label_metrics,
)
from qqq_btc.tools.run_inference import load_model, run_inference_df
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config_9dte_legacy as cfg9

logger = logging.getLogger("eval_9dte_legacy")


def run_month(
    month: str,
    *,
    checkpoint: Path,
    feature_root: Path,
    option_1m_root: Path,
    output_dir: Path,
    config: dict,
    symbol: str = "QQQ",
    stock_id: int = 0,
    sector_id: int = 0,
    device: str = "auto",
) -> dict:
    import torch

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = feature_root.expanduser() / symbol / "regular" / "09:30-16:00"
    f1 = root / "1min" / f"{month}.parquet"
    f5 = root / "5min" / f"{month}.parquet"
    if not f1.exists():
        raise FileNotFoundError(f"missing feature file: {f1}")

    _, feats_5m = _feat_names_by_res(config)
    model = load_model(checkpoint, config, dev)

    logger.info("infer %s (%d rows)", f1.name, len(pd.read_parquet(f1, columns=["timestamp"])))
    df = merge_1m_5m(f1, f5, feats_5m)
    pred = run_inference_df(
        df, model, config,
        stock_id=stock_id, sector_id=sector_id,
        device=dev, use_carryover=True,
    )
    pred = drop_embedded_exec_columns(pred)
    pred = attach_exec_quotes(
        pred, option_1m_root, symbol,
        call_bucket=cfg9.TRADE_BUCKET_ID,
        put_bucket=cfg9.PUT_BUCKET_ID,
    )

    n_call = int((pd.to_numeric(pred.get("exec_call_bid"), errors="coerce") > 0).sum())
    n_put = int((pd.to_numeric(pred.get("exec_put_bid"), errors="coerce") > 0).sum())
    edge = pred["net_edge"]
    logger.info(
        "exec call=%d put=%d | net_edge max=%.6f pct>=0.001=%.1f%%",
        n_call, n_put, edge.max(), 100 * (edge >= 0.001).mean(),
    )

    metrics = label_metrics(pred)
    result = run_strict_replay(
        pred,
        cfg9.FILL_MODEL,
        cfg9.REPLAY,
        cfg9.EXIT_RAILS,
        edge_col="net_edge",
        edge_q10_col=cfg9.EDGE_Q10_COL,
        call_edge_col=cfg9.CALL_EDGE_COL,
        put_edge_col=cfg9.PUT_EDGE_COL,
        put_gate_col=cfg9.PUT_GATE_COL,
    )
    summary = result.summary()
    f = cfg9.REPLAY.position_frac
    eq = 1.0
    for t in result.trades:
        eq *= 1.0 + f * t.net_return

    summary.update({
        "profile": cfg9.PROFILE,
        "month": month,
        "checkpoint": str(checkpoint),
        "feature_root": str(feature_root),
        "option_1m_root": str(option_1m_root),
        "replay_config": "qqq_btc.qqq.config_9dte_legacy",
        "v4_replay_untouched": True,
        "label_metrics": metrics,
        "n_rows": len(pred),
        "month_return_pct": (eq - 1.0) * 100.0,
        "exec_call_rows": n_call,
        "exec_put_rows": n_put,
        "entry_threshold_9dte": cfg9.REPLAY.entry_threshold,
        "entry_threshold_v4_ref": 0.03,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(output_dir / f"infer_{month.replace('-', '')}.parquet", index=False)
    (output_dir / f"replay_summary_{month.replace('-', '')}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    if result.trades:
        pd.DataFrame([
            {
                "entry_ts": t.entry_ts,
                "net_return": t.net_return,
                "exit_reason": t.exit_reason,
                "leg": t.leg,
                "bars_held": t.bars_held,
            }
            for t in result.trades
        ]).to_parquet(output_dir / f"replay_trades_{month.replace('-', '')}.parquet", index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="9DTE legacy replay (isolated from V4/0DTE config)")
    parser.add_argument("--month", default="2026-03", help="YYYY-MM parquet stem")
    parser.add_argument("--checkpoint", default=str(cfg9.CHECKPOINT_FILL))
    parser.add_argument("--feature-root", default=str(cfg9.FEATURE_VAL_ROOT))
    parser.add_argument("--option-1m-root", default=str(cfg9.RAW_1M_ROOT))
    parser.add_argument("--config", default=str(cfg9.FEATURE_CONFIG_PATH))
    parser.add_argument("--symbol-map", default="qqq_btc/CONFIG/symbol_map.json")
    parser.add_argument("--output-dir", default="/tmp/qqq_btc_eval_9dte_legacy")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    with open(_REPO / args.symbol_map, encoding="utf-8") as f:
        sym = json.load(f)["QQQ"]

    out_sub = Path(args.output_dir) / args.month
    summary = run_month(
        args.month,
        checkpoint=Path(args.checkpoint),
        feature_root=Path(args.feature_root),
        option_1m_root=Path(args.option_1m_root),
        output_dir=out_sub,
        config=config,
        stock_id=int(sym["stock_id"]),
        sector_id=int(sym["sector_id"]),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
